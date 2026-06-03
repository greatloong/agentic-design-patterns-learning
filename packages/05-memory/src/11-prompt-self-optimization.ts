/**
 * 11 - Prompt Self-Optimization：让 Agent 自己优化自己的 Prompt
 *
 * ── 与 09 / 10 的区别 ────────────────────────────────────────────────────
 *
 *   09 Feedback Learning:  用户纠正 → 加一条行为规则（additive）
 *   10 Heuristic Reflection: 任务成败 → 加一条 Heuristic 到 Pool（additive）
 *   11 Prompt Self-Opt:    批量评估 → 直接重写 prompt 本身（mutative）
 *
 *   关键差异：09/10 是"在 prompt 旁边加东西"，11 是"改 prompt 本身"。
 *   优化完成后部署的就是改好的 prompt，推理时零额外开销。
 *
 * ── 方案：SkillOpt 精简版（Microsoft, 2026, arXiv:2605.23904）─────────────
 *
 *   核心思想：把 Skill 文档当作"可训练状态"（相当于神经网络的权重），
 *   用一个 optimizer LLM 把"打分后的 rollout"转化为对文档的 add/delete/replace 编辑。
 *
 *   保留 4 个最能体现思想的机制：
 *     1. Skill Document = 可训练状态（一段 markdown，相当于权重）
 *     2. Rollout + Grade：用当前 skill 跑一批任务 → LLM-as-Judge 打分
 *     3. Natural Language Gradient：分析失败 → 提出结构化编辑（文本梯度）
 *     4. Validation Gate + Rollback：新 skill 必须在 held-out split 上严格提升
 *        才接受，否则回滚；被拒的编辑进 rejected buffer，避免重复犯错
 *
 *   砍掉的（避免 demo 变框架）：epoch-wise slow/meta update、cosine 学习率衰减
 *
 * ── Natural Language Gradient Descent（ProTeGi/APO 的核心类比）────────────
 *
 *   传统 ML:     Loss → 计算数值梯度 → 更新权重
 *   Prompt Opt:  评估失败 → 生成文本"梯度"(批评) → 编辑 prompt
 *
 * ── 本节演示 ──────────────────────────────────────────────────────────────
 *
 *   初始 skill 很简陋（"根据用户问题推荐商品"）。
 *   跑 N 个 epoch，观察 skill.md 如何自动长出"过滤噪声类目""检查预算"
 *   "歧义先确认"等规则——这些规则不是写死的，是从失败中自动进化出来的。
 */

import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// ── 模型配置 ────────────────────────────────────────────────────────────

// 被优化的 Agent（task model），用 pro
const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

// optimizer / grader（反思类，用 flash，便宜快）
const optimizerLlm = new ChatOpenAI({
  model: "deepseek-v4-flash",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
  temperature: 0,
});

// ══════════════════════════════════════════════════════════════════════════
// 训练超参（SkillOpt 风格）
// ══════════════════════════════════════════════════════════════════════════

const EPOCHS = 3;
const TEXTUAL_LEARNING_RATE = 2; // 每个 epoch 最多接受的编辑数量（防止 prompt 暴走）
const PASS_THRESHOLD = 0.7;      // 单个任务 grade >= 此值算通过

// ══════════════════════════════════════════════════════════════════════════
// 数据集：训练集 + 验证集（held-out，防止过拟合）
// ══════════════════════════════════════════════════════════════════════════

interface Task {
  id: string;
  userQuery: string;
  // 评分时给 grader 的"标准答案要点"（不是给 agent 的）
  rubric: string;
}

// 训练集：optimizer 能看到这些任务的失败，据此生成梯度
const TRAIN_SET: Task[] = [
  {
    id: "train-1",
    userQuery: "搜索结果：[手机壳]iPhone15透明壳¥89 / [贴膜]iPhone15钢化膜¥29 / [手机壳]iPhone14硅胶壳¥59 / [保护壳]MacBook壳¥199。用户问：'有 iPhone 15 手机壳推荐吗？'",
    rubric: "应只推荐 iPhone 15 的手机壳（透明壳），过滤掉贴膜（类目错）、iPhone14（型号错）、MacBook壳（完全不相关）。",
  },
  {
    id: "train-2",
    userQuery: "搜索结果：[耳机]Sony WF-C500运动款¥299 / [音箱]JBL GO3¥259 / [耳机]AirPods Pro2¥1599 / [配件]耳机收纳盒¥29 / [耳机]华为FreeBuds SE3¥249。用户问：'300以内适合跑步的蓝牙耳机'",
    rubric: "应只推荐 300 以内的运动蓝牙耳机（Sony WF-C500、华为FreeBuds SE3），过滤音箱（类目错）、AirPods（超预算）、收纳盒（配件）。",
  },
  {
    id: "train-3",
    userQuery: "用户最近订单：ORD-001蓝牙耳机 / ORD-002充电宝 / ORD-003数据线。用户说：'帮我退了那个不好用的'",
    rubric: "用户没指明退哪个，有 3 笔订单存在歧义。应先向用户确认是哪一笔，不能擅自假设。",
  },
  {
    id: "train-4",
    userQuery: "搜索结果：[手表]小米手环8 Pro¥179 / [手表]华为Watch GT4¥1488 / [手表]Keep B4¥229 / [手机]红米Note¥899。用户问：'200块左右的运动手表'",
    rubric: "应推荐 200 左右的运动手表（小米手环8 Pro、Keep B4），过滤超预算的华为GT4、以及完全不相关的手机。",
  },
  {
    id: "train-5",
    userQuery: "搜索结果：[键盘]机械键盘¥299 / [鼠标]游戏鼠标¥199 / [键盘]无线键盘¥159。用户问：'有没有安静一点的键盘，办公用'",
    rubric: "办公'安静'指向静音/薄膜键盘，机械键盘通常偏吵。应说明机械键盘可能不符'安静'诉求，优先推荐无线键盘并提示需确认轴体静音性。",
  },
];

// 验证集：optimizer 看不到，只用来做 validation gate（防过拟合）
const VAL_SET: Task[] = [
  {
    id: "val-1",
    userQuery: "搜索结果：[充电宝]小米10000mAh¥99 / [数据线]Type-C线¥19 / [充电宝]Anker20000mAh¥199 / [插座]插线板¥39。用户问：'出差用的大容量充电宝'",
    rubric: "应推荐大容量充电宝（Anker20000mAh优先），过滤数据线、插座等不相关类目，小容量的次选。",
  },
  {
    id: "val-2",
    userQuery: "用户最近订单：ORD-010连衣裙 / ORD-011鞋子。用户说：'我要退货'",
    rubric: "有 2 笔订单且未指明，应先确认退哪一笔，不能擅自假设。",
  },
  {
    id: "val-3",
    userQuery: "搜索结果：[显示器]27寸4K¥1599 / [显示器]24寸1080P¥699 / [支架]显示器支架¥99。用户问：'预算800的显示器'",
    rubric: "应推荐 800 预算内的显示器（24寸1080P），过滤超预算的4K、以及支架（配件非显示器）。",
  },
];

// ══════════════════════════════════════════════════════════════════════════
// 初始 Skill（很简陋，故意的——看它怎么进化）
// ══════════════════════════════════════════════════════════════════════════

const INITIAL_SKILL = `# 商品推荐 Skill

根据用户的问题推荐合适的商品。`;

// ══════════════════════════════════════════════════════════════════════════
// Rollout：用当前 skill 执行单个任务
// ══════════════════════════════════════════════════════════════════════════

async function rollout(skill: string, task: Task): Promise<string> {
  const systemPrompt = `你是电商客服 Agent。请遵循以下 Skill 文档来回应用户：

${skill}`;

  const resp = await llm.invoke([
    new SystemMessage(systemPrompt),
    new HumanMessage(task.userQuery),
  ]);
  return resp.content as string;
}

// ══════════════════════════════════════════════════════════════════════════
// Grade：LLM-as-Judge 给单个 rollout 打分（0~1）
// ══════════════════════════════════════════════════════════════════════════

interface GradeResult {
  taskId: string;
  score: number;
  feedback: string;
  output: string;
}

async function grade(task: Task, output: string): Promise<GradeResult> {
  const gradePrompt = `你是 QA 评估员。根据评分要点给 Agent 的回应打分。

## 用户问题
${task.userQuery}

## 评分要点（标准）
${task.rubric}

## Agent 的回应
${output}

## 评分标准
- 1.0：完全符合要点，过滤了所有干扰项/正确处理了歧义
- 0.5：部分符合，有遗漏或包含少量干扰项
- 0.0：完全不符合要点

## 输出 JSON
{"score": 0.0~1.0, "feedback": "一句话指出具体问题（哪些该过滤没过滤/哪里该确认没确认）"}

只输出 JSON。`;

  const resp = await optimizerLlm.invoke([new HumanMessage(gradePrompt)]);
  const content = (resp.content as string).trim();

  try {
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    const parsed = JSON.parse(jsonMatch?.[0] || content);
    return {
      taskId: task.id,
      score: parsed.score,
      feedback: parsed.feedback,
      output,
    };
  } catch {
    return { taskId: task.id, score: 0, feedback: "评分解析失败", output };
  }
}

/** 跑完一个数据集，返回平均分 + 每个任务的明细 */
async function evaluateSet(
  skill: string,
  taskSet: Task[]
): Promise<{ avgScore: number; results: GradeResult[] }> {
  const results: GradeResult[] = [];
  for (const task of taskSet) {
    const output = await rollout(skill, task);
    const g = await grade(task, output);
    results.push(g);
  }
  const avgScore =
    results.reduce((sum, r) => sum + r.score, 0) / results.length;
  return { avgScore, results };
}

// ══════════════════════════════════════════════════════════════════════════
// Natural Language Gradient：分析失败 → 提出结构化编辑
// ══════════════════════════════════════════════════════════════════════════

interface SkillEdit {
  op: "add" | "delete" | "replace";
  content: string;   // add: 新增内容；replace: 新内容；delete: 要删的内容
  reason: string;    // 这条编辑对应的"梯度"（为什么改）
}

const GRADIENT_PROMPT = `你是一个 Prompt 优化器（类似 SkillOpt 的 optimizer model）。
你的任务：分析当前 Skill 文档在训练任务上的失败，提出对文档的结构化编辑。

## 当前 Skill 文档
{skill}

## 本轮训练任务的评分结果（含失败反馈 = "文本梯度"）
{gradients}

## 已被拒绝的历史编辑（不要重复这些，它们没用甚至有害）
{rejected}

## 你的任务
分析失败模式，提出 1~{lr} 条对 Skill 文档的编辑。
编辑应该让文档长出"能避免这些失败"的具体规则。

### 要求
- 规则要具体、可执行（不是"要小心"这种废话）
- 要可迁移（适用于类似任务，不绑定具体商品名）
- 优先解决出现频率最高、扣分最多的失败模式
- 不要让文档无限膨胀，能合并的规则就合并

### 输出格式（JSON 数组，最多 {lr} 条）
[
  {"op": "add", "content": "要追加到文档的规则文本", "reason": "对应哪个失败"},
  {"op": "replace", "content": "新内容||旧内容", "reason": "..."}
]

只输出 JSON 数组。`;

async function computeGradientEdits(
  skill: string,
  failedResults: GradeResult[],
  rejectedBuffer: string[]
): Promise<SkillEdit[]> {
  const gradients = failedResults
    .map((r) => `- [${r.taskId}] 得分 ${r.score}：${r.feedback}`)
    .join("\n");

  const rejected =
    rejectedBuffer.length > 0
      ? rejectedBuffer.map((r) => `- ${r}`).join("\n")
      : "（无）";

  const prompt = GRADIENT_PROMPT.replace("{skill}", skill)
    .replace("{gradients}", gradients)
    .replace("{rejected}", rejected)
    .replace(/\{lr\}/g, String(TEXTUAL_LEARNING_RATE));

  const resp = await optimizerLlm.invoke([new HumanMessage(prompt)]);
  const content = (resp.content as string).trim();

  try {
    const jsonMatch = content.match(/\[[\s\S]*\]/);
    const parsed = JSON.parse(jsonMatch?.[0] || content);
    return Array.isArray(parsed) ? parsed.slice(0, TEXTUAL_LEARNING_RATE) : [];
  } catch {
    return [];
  }
}

/** 把编辑应用到 skill 文档，返回新文档 */
function applyEdits(skill: string, edits: SkillEdit[]): string {
  let updated = skill;
  for (const edit of edits) {
    if (edit.op === "add") {
      updated += `\n- ${edit.content}`;
    } else if (edit.op === "replace") {
      const [newText, oldText] = edit.content.split("||");
      if (oldText && updated.includes(oldText.trim())) {
        updated = updated.replace(oldText.trim(), newText.trim());
      } else {
        updated += `\n- ${newText}`;
      }
    } else if (edit.op === "delete") {
      if (updated.includes(edit.content)) {
        updated = updated.replace(edit.content, "");
      }
    }
  }
  return updated.trim();
}

// ══════════════════════════════════════════════════════════════════════════
// 训练循环：Rollout → Grade → Gradient → Validation Gate → Accept/Reject
// ══════════════════════════════════════════════════════════════════════════

interface VersionedSkill {
  version: number;
  skill: string;
  trainScore: number;
  valScore: number;
  acceptedAt: string;
  note: string;
}

async function train() {
  console.log("═".repeat(60));
  console.log("11 - Prompt Self-Optimization (SkillOpt 精简版)");
  console.log("═".repeat(60));
  console.log(`\n超参: epochs=${EPOCHS}, 文本学习率=${TEXTUAL_LEARNING_RATE}, 通过阈值=${PASS_THRESHOLD}`);

  // 当前最佳 skill（相当于 best_skill.md）
  let bestSkill = INITIAL_SKILL;
  const rejectedBuffer: string[] = []; // 被拒编辑缓冲
  const history: VersionedSkill[] = [];

  // ── 基线评估 ──
  console.log("\n" + "─".repeat(60));
  console.log("📏 基线评估（初始 skill）");
  console.log("─".repeat(60));
  console.log(`初始 Skill:\n${INITIAL_SKILL}`);

  const baseTrain = await evaluateSet(bestSkill, TRAIN_SET);
  const baseVal = await evaluateSet(bestSkill, VAL_SET);
  console.log(`\n  Train 平均分: ${baseTrain.avgScore.toFixed(3)}`);
  console.log(`  Val   平均分: ${baseVal.avgScore.toFixed(3)}`);

  let bestValScore = baseVal.avgScore;
  let bestTrainScore = baseTrain.avgScore;
  history.push({
    version: 0,
    skill: bestSkill,
    trainScore: baseTrain.avgScore,
    valScore: baseVal.avgScore,
    acceptedAt: new Date().toISOString(),
    note: "初始基线",
  });

  // ── 训练 epoch ──
  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    console.log("\n" + "═".repeat(60));
    console.log(`🔄 Epoch ${epoch}/${EPOCHS}`);
    console.log("═".repeat(60));

    // 1. Rollout + Grade（在当前 best skill 上）
    const { avgScore: trainScore, results } = await evaluateSet(
      bestSkill,
      TRAIN_SET
    );
    console.log(`\n  [1] Rollout + Grade → Train 平均分: ${trainScore.toFixed(3)}`);

    const failed = results.filter((r) => r.score < PASS_THRESHOLD);
    console.log(`  [2] 发现 ${failed.length} 个未通过任务（梯度来源）：`);
    failed.forEach((r) =>
      console.log(`      - [${r.taskId}] ${r.score} | ${r.feedback}`)
    );

    if (failed.length === 0) {
      console.log("\n  ✅ 全部通过，提前收敛！");
      break;
    }

    // 3. 计算文本梯度 → 提出编辑
    const edits = await computeGradientEdits(bestSkill, failed, rejectedBuffer);
    console.log(`\n  [3] 文本梯度 → 提出 ${edits.length} 条编辑：`);
    edits.forEach((e) =>
      console.log(`      - [${e.op}] ${e.content.slice(0, 50)}... (理由: ${e.reason.slice(0, 30)})`)
    );

    if (edits.length === 0) {
      console.log("      ⚠️ 未提出有效编辑，跳过本轮");
      continue;
    }

    // 4. 应用编辑 → 候选 skill
    const candidateSkill = applyEdits(bestSkill, edits);

    // 5. Validation Gate：候选 skill 在 val split 上必须严格提升
    console.log(`\n  [4] Validation Gate（在 held-out val split 上验证）...`);
    const candTrain = await evaluateSet(candidateSkill, TRAIN_SET);
    const candVal = await evaluateSet(candidateSkill, VAL_SET);
    console.log(`      候选 skill → Train: ${candTrain.avgScore.toFixed(3)} | Val: ${candVal.avgScore.toFixed(3)}`);
    console.log(`      当前 best  → Train: ${bestTrainScore.toFixed(3)} | Val: ${bestValScore.toFixed(3)}`);

    if (candVal.avgScore > bestValScore) {
      // 接受
      bestSkill = candidateSkill;
      bestValScore = candVal.avgScore;
      bestTrainScore = candTrain.avgScore;
      history.push({
        version: epoch,
        skill: bestSkill,
        trainScore: candTrain.avgScore,
        valScore: candVal.avgScore,
        acceptedAt: new Date().toISOString(),
        note: `接受 ${edits.length} 条编辑`,
      });
      console.log(`      ✅ ACCEPT：val 分提升 ${bestValScore.toFixed(3)} → ${candVal.avgScore.toFixed(3)}`);
    } else {
      // 拒绝 + 记入 rejected buffer
      edits.forEach((e) => rejectedBuffer.push(`${e.op}: ${e.content.slice(0, 60)} (val 未提升)`));
      console.log(`      ❌ REJECT：val 分未提升（${candVal.avgScore.toFixed(3)} <= ${bestValScore.toFixed(3)}），回滚 + 记入 rejected buffer`);
    }
  }

  // ── 结果汇总 ──
  console.log("\n\n" + "═".repeat(60));
  console.log("📊 训练完成 — Skill 进化历史");
  console.log("═".repeat(60));
  for (const v of history) {
    console.log(`  v${v.version} | Train ${v.trainScore.toFixed(3)} | Val ${v.valScore.toFixed(3)} | ${v.note}`);
  }

  console.log("\n" + "─".repeat(60));
  console.log("🏆 最终进化出的 Skill 文档（best_skill.md）：");
  console.log("─".repeat(60));
  console.log(bestSkill);

  console.log("\n" + "─".repeat(60));
  console.log(`📈 提升：Val ${baseVal.avgScore.toFixed(3)} → ${bestValScore.toFixed(3)} (+${(bestValScore - baseVal.avgScore).toFixed(3)})`);
  console.log(`🗑️  Rejected buffer：${rejectedBuffer.length} 条被拒编辑`);
  console.log("─".repeat(60));

  console.log("\n" + "═".repeat(60));
  console.log("✅ SkillOpt 循环演示完成：");
  console.log("   1. Skill 文档 = 可训练状态（相当于权重）");
  console.log("   2. Rollout + Grade → 文本梯度（失败反馈）");
  console.log("   3. optimizer 提出 add/delete/replace 编辑（受学习率约束）");
  console.log("   4. Validation Gate：只接受 val 提升的编辑，否则回滚");
  console.log("   5. 部署的就是 best_skill.md，推理时零额外开销");
  console.log("═".repeat(60));
}

train().catch(console.error);
