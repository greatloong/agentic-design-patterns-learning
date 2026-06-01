/**
 * 10 - Heuristic Reflection：ERL 风格——从任务结果中反思提炼经验
 *
 * ── 与 09 的区别 ─────────────────────────────────────────────────────────
 *
 *   09: 用户说"太正式了" → 提炼偏好规则（用户教 Agent 做人）
 *   10: Agent 执行失败 → 自己反思为什么 → 提炼策略规则（Agent 自己学做事）
 *
 * ── 核心流程（ERL 论文简化版）────────────────────────────────────────────
 *
 *   Phase 1 — 离线积累：
 *     执行任务 → 获得结果（成功/失败）
 *     → LLM 反思 trajectory
 *     → 提炼为 Trigger-Action Heuristic
 *     → 向量化存入 Heuristic Pool
 *
 *   Phase 2 — 在线使用：
 *     新任务到来
 *     → 用任务描述作为 query，Embedding 检索 Top-K 相关 Heuristic
 *     → 注入 System Prompt
 *     → Agent 带着经验执行
 *
 * ── 质量控制：三种审查模式 ────────────────────────────────────────────────
 *
 *   模式 1 — 全自动（无审查）：反思 → 直接存入 production pool
 *     适合低风险场景（个人助手、内部工具）
 *
 *   模式 2 — Staged Rollout（分阶段上线）：
 *     反思 → 存入 shadow pool → 跑 N 次对比效果 → 正向才提升到 production
 *     ACE 框架的做法：shadow → staging → prod，检测到回归自动回滚
 *
 *   模式 3 — Human-in-the-Loop（人工审核）：
 *     反思 → 存入 pending pool → 人类审核 approve/reject/edit → 才进入 production
 *     适合金融、医疗、客服等不能出错的场景
 *
 * ── 本节演示 ──────────────────────────────────────────────────────────────
 *
 *   1. Agent 真实执行任务（ReAct loop + Mock Tools）→ Tracer 自动记录轨迹
 *   2. LLM-as-Judge 自动评估 outcome（不靠硬编码）
 *   3. 反思提炼 Heuristic → 质量控制审查（shadow / pending）
 *   4. 新任务带 Heuristic 指导执行 vs 无指导对比
 */

import "dotenv/config";
import { InMemoryStore } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// ── 模型配置 ────────────────────────────────────────────────────────────

const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

const reflectionLlm = new ChatOpenAI({
  model: "deepseek-v4-flash",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
  temperature: 0,
});

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-v4",
  dimensions: 512,
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: {
    baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  },
});

// ── Heuristic Pool（向量化存储）────────────────────────────────────────

const heuristicStore = new InMemoryStore({
  index: {
    dims: 512,
    embeddings,
  },
});

const HEURISTIC_NAMESPACE = ["heuristics", "task-agent"];

// ══════════════════════════════════════════════════════════════════════════
// 数据结构：带质量控制字段的 Heuristic
// ══════════════════════════════════════════════════════════════════════════

type HeuristicStatus = "shadow" | "staging" | "production" | "pending" | "rejected";

interface HeuristicEntry {
  task: string;
  outcome: "success" | "failure";
  analysis: string;
  trigger: string;
  action: string;
  rationale: string;
  heuristicText: string;
  // 质量控制字段
  status: HeuristicStatus;
  confidence: number;       // 0~1，初始 0.5
  usedCount: number;        // 被检索注入的次数
  validatedCount: number;   // 注入后任务成功的次数
  createdAt: string;
  lastUsedAt: string | null;
  reviewNote?: string;      // 人工审核备注
}

// ══════════════════════════════════════════════════════════════════════════
// Phase 1：执行任务 → 反思 → 提炼 Heuristic
// ══════════════════════════════════════════════════════════════════════════

interface TaskExecution {
  task: string;
  trajectory: string;
  outcome: "success" | "failure";
}

const REFLECTION_PROMPT = `你是一个 AI Agent 的经验反思器。你需要分析一次任务执行的轨迹，提炼出可复用的经验规则。

## 任务描述
{task}

## 执行轨迹
{trajectory}

## 执行结果
{outcome}

## 你的任务

分析这次执行，提炼一条可迁移的 Heuristic（经验规则）。

### 要求
- 如果是失败：找到 Breakpoint（哪一步出错了），提炼"如何避免"
- 如果是成功：找到关键决策（哪一步做对了），提炼"为什么有效"
- 规则必须足够抽象，能适用于类似但不完全相同的任务
- 不要太具体（绑定某个具体值），也不要太泛（"要小心"）

### 输出格式（JSON）
{
  "analysis": "简要分析成功/失败原因（1-2句）",
  "heuristic": {
    "trigger": "IF [具体触发条件]",
    "action": "THEN [具体行动建议]",
    "rationale": "因为[原因]"
  }
}

只输出 JSON。`;

/**
 * 对一次任务执行进行反思，提炼 Heuristic 并存入 Pool
 * @param initialStatus 初始状态：决定走哪种审查模式
 *   - "production": 模式 1（全自动，直接生效）
 *   - "shadow": 模式 2（分阶段，先 shadow 验证）
 *   - "pending": 模式 3（人工审核）
 */
async function reflectAndStore(
  execution: TaskExecution,
  initialStatus: HeuristicStatus = "shadow"
): Promise<{ id: string; heuristicText: string }> {
  const prompt = REFLECTION_PROMPT.replace("{task}", execution.task)
    .replace("{trajectory}", execution.trajectory)
    .replace("{outcome}", execution.outcome);

  const resp = await reflectionLlm.invoke([new HumanMessage(prompt)]);
  const content = (resp.content as string).trim();

  let parsed: any;
  try {
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    parsed = JSON.parse(jsonMatch?.[0] || content);
  } catch {
    console.log("  ⚠️ 反思输出解析失败，跳过");
    return { id: "", heuristicText: "" };
  }

  const heuristicText = `${parsed.heuristic.trigger} ${parsed.heuristic.action} (${parsed.heuristic.rationale})`;

  const id = `h-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
  const entry: HeuristicEntry = {
    task: execution.task,
    outcome: execution.outcome,
    analysis: parsed.analysis,
    trigger: parsed.heuristic.trigger,
    action: parsed.heuristic.action,
    rationale: parsed.heuristic.rationale,
    heuristicText,
    status: initialStatus,
    confidence: 0.5,
    usedCount: 0,
    validatedCount: 0,
    createdAt: new Date().toISOString(),
    lastUsedAt: null,
  };

  await heuristicStore.put(HEURISTIC_NAMESPACE, id, entry as any);
  return { id, heuristicText };
}

// ══════════════════════════════════════════════════════════════════════════
// 模式 2：Staged Rollout — Shadow → Validate → Promote
// ══════════════════════════════════════════════════════════════════════════

/**
 * 模拟 shadow 验证：检查规则在历史任务上是否有正向效果
 * 生产中这会跑真实的 A/B test 或 eval suite
 */
async function validateInShadow(heuristicId: string): Promise<{
  passed: boolean;
  score: number;
  reason: string;
}> {
  const results = await heuristicStore.search(HEURISTIC_NAMESPACE, {
    query: heuristicId,
    limit: 100,
  });
  const entry = results.find((r) => r.key === heuristicId);
  if (!entry) return { passed: false, score: 0, reason: "未找到规则" };

  const value = entry.value as any as HeuristicEntry;

  // 模拟验证逻辑：用 LLM 评估规则质量
  const evalPrompt = `评估以下经验规则的质量（0-10分）：

规则：${value.heuristicText}

评判标准：
1. 具体性（不是"要小心"这种废话）
2. 可迁移性（能适用于类似场景，不绑定具体值）
3. 可操作性（有明确的行动步骤）
4. 无害性（不会导致错误行为）

输出 JSON：{"score": 0-10, "reason": "一句话评价"}`;

  const resp = await reflectionLlm.invoke([new HumanMessage(evalPrompt)]);
  const content = (resp.content as string).trim();

  try {
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    const parsed = JSON.parse(jsonMatch?.[0] || content);
    const passed = parsed.score >= 7; // 7 分以上才通过
    return { passed, score: parsed.score, reason: parsed.reason };
  } catch {
    return { passed: false, score: 0, reason: "评估解析失败" };
  }
}

/**
 * 将通过验证的 Heuristic 从 shadow 提升到 production
 */
async function promoteToProduction(heuristicId: string): Promise<boolean> {
  const results = await heuristicStore.search(HEURISTIC_NAMESPACE, {
    query: "all",
    limit: 100,
  });
  const entry = results.find((r) => r.key === heuristicId);
  if (!entry) return false;

  const value = entry.value as any as HeuristicEntry;
  value.status = "production";
  value.confidence = 0.7; // 通过验证后置信度提升

  await heuristicStore.delete(HEURISTIC_NAMESPACE, heuristicId);
  await heuristicStore.put(HEURISTIC_NAMESPACE, heuristicId, value as any);
  return true;
}

// ══════════════════════════════════════════════════════════════════════════
// 模式 3：Human-in-the-Loop — Pending → Approve/Reject/Edit
// ══════════════════════════════════════════════════════════════════════════

/**
 * 获取所有待审核的 Heuristic（供人类审核者查看）
 */
async function getPendingHeuristics(): Promise<
  Array<{ id: string; entry: HeuristicEntry }>
> {
  const results = await heuristicStore.search(HEURISTIC_NAMESPACE, {
    query: "pending review",
    limit: 100,
  });

  return results
    .filter((r) => (r.value as any).status === "pending")
    .map((r) => ({ id: r.key, entry: r.value as any as HeuristicEntry }));
}

/**
 * 人工审核：Approve（通过）→ 进入 production
 */
async function approveHeuristic(
  heuristicId: string,
  reviewNote?: string
): Promise<void> {
  const results = await heuristicStore.search(HEURISTIC_NAMESPACE, {
    query: "all",
    limit: 100,
  });
  const entry = results.find((r) => r.key === heuristicId);
  if (!entry) return;

  const value = entry.value as any as HeuristicEntry;
  value.status = "production";
  value.confidence = 0.8; // 人工审核通过，高置信度
  value.reviewNote = reviewNote || "人工审核通过";

  await heuristicStore.delete(HEURISTIC_NAMESPACE, heuristicId);
  await heuristicStore.put(HEURISTIC_NAMESPACE, heuristicId, value as any);
}

/**
 * 人工审核：Reject（拒绝）→ 标记为 rejected，不参与检索
 */
async function rejectHeuristic(
  heuristicId: string,
  reviewNote: string
): Promise<void> {
  const results = await heuristicStore.search(HEURISTIC_NAMESPACE, {
    query: "all",
    limit: 100,
  });
  const entry = results.find((r) => r.key === heuristicId);
  if (!entry) return;

  const value = entry.value as any as HeuristicEntry;
  value.status = "rejected";
  value.confidence = 0;
  value.reviewNote = reviewNote;

  await heuristicStore.delete(HEURISTIC_NAMESPACE, heuristicId);
  await heuristicStore.put(HEURISTIC_NAMESPACE, heuristicId, value as any);
}

// ══════════════════════════════════════════════════════════════════════════
// Phase 2：检索（只取 production 状态的）
// ══════════════════════════════════════════════════════════════════════════

/**
 * 检索与当前任务相关的 Top-K Heuristic（只取 production 且 confidence > 阈值）
 */
async function retrieveHeuristics(
  taskDescription: string,
  topK: number = 5
): Promise<string[]> {
  const results = await heuristicStore.search(HEURISTIC_NAMESPACE, {
    query: taskDescription,
    limit: topK * 3, // 多取一些，后面过滤
  });

  return results
    .filter((r) => {
      const v = r.value as any;
      return v.status === "production" && v.confidence > 0.3;
    })
    .slice(0, topK)
    .map((r) => (r.value as any).heuristicText);
}

/**
 * 带 Heuristic 指导执行任务
 */
async function executeWithHeuristics(
  task: string,
  useHeuristics: boolean
): Promise<string> {
  let systemPrompt = `你是一个任务执行 Agent。用户会给你一个任务，你需要思考执行步骤并给出方案。
请先列出你的执行步骤（Step 1, 2, 3...），然后给出最终结果。`;

  if (useHeuristics) {
    const heuristics = await retrieveHeuristics(task);
    if (heuristics.length > 0) {
      systemPrompt += `\n\n## 来自过去经验的指导（执行时请参考）：\n${heuristics.map((h, i) => `${i + 1}. ${h}`).join("\n")}`;
    }
  }

  const resp = await llm.invoke([
    new SystemMessage(systemPrompt),
    new HumanMessage(task),
  ]);

  return resp.content as string;
}

// ══════════════════════════════════════════════════════════════════════════
// Mock Tool Environment：模拟电商 Agent 的工具集
// ══════════════════════════════════════════════════════════════════════════

interface ToolCall {
  tool: string;
  args: Record<string, any>;
}

interface ToolResult {
  tool: string;
  args: Record<string, any>;
  result: string;
  isError: boolean;
}

/**
 * 模拟电商 API 环境
 * 故意设计一些"坑"：噪声数据、参数格式要求、歧义等
 */
const MOCK_TOOLS: Record<string, (args: Record<string, any>) => { result: string; isError: boolean }> = {
  "UserAPI.resolve": ({ identifier }) => {
    if (/^U\d+$/.test(identifier)) {
      return { result: `用户信息: {id: "${identifier}", name: "张三", email: "zhang3@test.com", phone: "138xxxx1234"}`, isError: false };
    }
    if (identifier === "张三") {
      return { result: `Error 400: identifier 必须是用户ID格式(U+数字)，不接受姓名。请先调用 UserAPI.searchByName`, isError: true };
    }
    return { result: `Error 404: 用户 ${identifier} 不存在`, isError: true };
  },

  "UserAPI.searchByName": ({ name }) => {
    const users: Record<string, string> = { "张三": "U12345", "李四": "U67890" };
    return users[name]
      ? { result: `找到用户: {name: "${name}", userId: "${users[name]}"}`, isError: false }
      : { result: `未找到名为"${name}"的用户`, isError: true };
  },

  "UserAPI.getPreferences": ({ userId }) => {
    return { result: `用户偏好: {priceRange: "200-500", brands: ["华为","小米"], categories: ["数码","运动"], notificationMethod: "短信"}`, isError: false };
  },

  "SearchAPI.search": ({ query, filters }) => {
    if (query.includes("蓝牙耳机")) {
      return {
        result: `搜索结果 (共18条):
  1. [耳机] Sony WF-C500 运动蓝牙耳机 ¥299 (评分4.5, 标签:运动,防水)
  2. [耳机] 漫步者 TWS1 蓝牙耳机 ¥129 (评分4.2, 标签:日常,性价比)
  3. [音箱] JBL GO3 蓝牙音箱 ¥259 (评分4.7, 标签:户外) ← 类目:音箱,非耳机
  4. [耳机] AirPods Pro 2 ¥1599 (评分4.9, 标签:降噪) ← 超出预算
  5. [配件] 耳机收纳盒 ¥29 (标签:配件) ← 类目:配件,非耳机
  6. [耳机] 华为 FreeBuds SE3 ¥249 (评分4.3, 标签:运动,防汗)
  7. [耳机] Beats Fit Pro ¥899 (评分4.6, 标签:运动) ← 超出预算
  8. [线材] 3.5mm耳机转接线 ¥15 ← 完全不相关`,
        isError: false,
      };
    }
    if (query.includes("手机壳") || query.includes("iPhone")) {
      return {
        result: `搜索结果 (共20条):
  1. [手机壳] iPhone 15 MagSafe 透明壳 ¥89
  2. [贴膜] iPhone 15 钢化膜 ¥29 ← 类目:贴膜,非手机壳
  3. [手机壳] iPhone 14 硅胶壳 ¥59 ← 型号不匹配
  4. [手机壳] iPhone 15 防摔壳 ¥129
  5. [保护壳] MacBook Air 保护壳 ¥199 ← 完全不相关
  6. [手机壳] iPhone 15 Pro Max 皮质壳 ¥199`,
        isError: false,
      };
    }
    return { result: `搜索"${query}"返回 0 条结果`, isError: false };
  },

  "OrderAPI.getRecent": ({ userId, limit }) => {
    return {
      result: `最近 ${limit || 5} 笔订单:
  - ORD-001: 蓝牙耳机 Sony WF-C500 ¥299 (3天前, 已签收)
  - ORD-002: 充电宝 小米20000mAh ¥149 (5天前, 已签收)
  - ORD-003: Type-C数据线 ¥19 (7天前, 已签收)`,
      isError: false,
    };
  },

  "OrderAPI.getStatus": ({ orderId }) => {
    if (!orderId || !orderId.startsWith("ORD-")) {
      return { result: `Error 400: orderId 格式错误，需要 ORD-xxx 格式`, isError: true };
    }
    return { result: `订单 ${orderId} 状态: 已签收 (签收时间: 2026-05-28)`, isError: false };
  },

  "RefundAPI.create": ({ orderId, reason }) => {
    return { result: `退货单已创建: RF-${Date.now().toString().slice(-4)}, 订单: ${orderId}, 原因: ${reason}`, isError: false };
  },

  "RecommendAPI.rank": ({ items, userProfile, context }) => {
    return { result: `重排序完成，按相关性排序结果已返回（基于用户偏好 + 场景匹配）`, isError: false };
  },

  "MessageAPI.send": ({ userId, content, channel }) => {
    if (!channel) {
      return { result: `Error 400: 缺少 channel 参数（sms/push/email）`, isError: true };
    }
    return { result: `消息已通过${channel}发送给用户 ${userId}`, isError: false };
  },
};

const TOOL_LIST_DESC = Object.keys(MOCK_TOOLS)
  .map((name) => `- ${name}`)
  .join("\n");

// ══════════════════════════════════════════════════════════════════════════
// Agent Executor：ReAct-style 循环，自动记录 Trajectory
// ══════════════════════════════════════════════════════════════════════════

const AGENT_SYSTEM_PROMPT = `你是一个电商客服 Agent，拥有以下工具：

${TOOL_LIST_DESC}

## 工具调用规则
每一步只能调用一个工具。输出格式：
Action: ToolName({"param": "value"})

如果任务已完成，输出：
Final: [最终结果摘要]

## 重要
- 每次只输出一个 Action 或一个 Final
- 不要输出解释，只输出 Action/Final 行`;

const MAX_STEPS = 8;

/**
 * 执行单个任务，返回完整轨迹（动态生成，非硬编码）
 */
async function runAgentTask(task: string): Promise<TaskExecution> {
  const steps: string[] = [];
  const messages: any[] = [
    new SystemMessage(AGENT_SYSTEM_PROMPT),
    new HumanMessage(`任务: ${task}`),
  ];

  let finalResult = "";

  for (let step = 1; step <= MAX_STEPS; step++) {
    const resp = await llm.invoke(messages);
    const output = (resp.content as string).trim();

    // 检查是否结束
    const finalMatch = output.match(/Final:\s*(.+)/s);
    if (finalMatch) {
      finalResult = finalMatch[1].trim();
      steps.push(`Step ${step}: [完成] ${finalResult}`);
      break;
    }

    // 解析 Action
    const actionMatch = output.match(/Action:\s*(\w+(?:\.\w+)?)\((.+)\)/s);
    if (!actionMatch) {
      steps.push(`Step ${step}: [Agent 输出异常] ${output.slice(0, 100)}`);
      messages.push(resp, new HumanMessage("请严格按格式输出 Action 或 Final。"));
      continue;
    }

    const toolName = actionMatch[1];
    let toolArgs: Record<string, any> = {};
    try {
      toolArgs = JSON.parse(actionMatch[2]);
    } catch {
      toolArgs = { raw: actionMatch[2] };
    }

    // 调用 Mock Tool
    const toolFn = MOCK_TOOLS[toolName];
    let toolResult: ToolResult;

    if (!toolFn) {
      toolResult = { tool: toolName, args: toolArgs, result: `Error: 工具 ${toolName} 不存在`, isError: true };
    } else {
      const { result, isError } = toolFn(toolArgs);
      toolResult = { tool: toolName, args: toolArgs, result, isError };
    }

    // 记录轨迹
    const argStr = JSON.stringify(toolArgs);
    const errorTag = toolResult.isError ? " [ERROR]" : "";
    steps.push(`Step ${step}: 调用 ${toolName}(${argStr})${errorTag}\n  → ${toolResult.result}`);

    // 追加到对话
    messages.push(resp, new HumanMessage(`Observation: ${toolResult.result}`));
  }

  if (!finalResult && steps.length >= MAX_STEPS) {
    steps.push(`[超过最大步数 ${MAX_STEPS}，强制终止]`);
  }

  const trajectory = steps.join("\n");

  // LLM-as-Judge 评估 outcome
  const outcome = await evaluateOutcome(task, trajectory);

  return { task, trajectory, outcome };
}

/**
 * LLM-as-Judge：评估任务执行质量
 */
async function evaluateOutcome(
  task: string,
  trajectory: string
): Promise<"success" | "failure"> {
  const evalPrompt = `你是一个 QA 评估员。判断以下任务执行是否成功。

## 任务
${task}

## 执行轨迹
${trajectory}

## 评判标准
- 任务目标是否达成？
- 过程中是否有明显错误（调错API、推荐不相关商品、未确认就操作）？
- 如果有错误但最终修正了，仍算 failure（因为过程低效）
- 如果顺利完成且结果质量高，算 success

## 输出
只输出一个词: success 或 failure`;

  const resp = await reflectionLlm.invoke([new HumanMessage(evalPrompt)]);
  const content = (resp.content as string).trim().toLowerCase();
  return content.includes("success") ? "success" : "failure";
}

// ══════════════════════════════════════════════════════════════════════════
// Demo
// ══════════════════════════════════════════════════════════════════════════

async function main() {
  console.log("═".repeat(60));
  console.log("10 - Heuristic Reflection: ERL + 质量控制");
  console.log("═".repeat(60));

  // ── Phase 1: Agent 真实执行任务 → 自动记录轨迹 → 反思 ──
  console.log("\n📚 Phase 1: Agent 执行任务 → 自动记录轨迹 → 反思");
  console.log("─".repeat(40));

  const tasks = [
    "用户（U12345）问'有没有 iPhone 15 手机壳推荐'，搜索商品并推荐 Top-3 给用户",
    "用户说'帮我退了那个不好用的'（用户ID: U12345），处理退货请求",
    "用户（U67890）问'300以内适合跑步的蓝牙耳机'，推荐商品",
    "查询用户张三的最近订单状态",
  ];

  const executions: TaskExecution[] = [];
  for (const task of tasks) {
    console.log(`\n  🏃 执行任务: "${task}"`);
    console.log("  " + "·".repeat(36));
    const execution = await runAgentTask(task);
    executions.push(execution);
    console.log(`  📝 轨迹:\n${execution.trajectory.split("\n").map(l => "    " + l).join("\n")}`);
    console.log(`  🏷️  Outcome: ${execution.outcome}`);
  }

  // ── 反思并存入 Pool ──
  console.log("\n\n🧠 反思：从轨迹中提炼 Heuristic");
  console.log("─".repeat(40));

  const heuristicIds: string[] = [];
  for (let i = 0; i < executions.length; i++) {
    const exec = executions[i];
    const mode: HeuristicStatus = i < 2 ? "shadow" : "pending";
    console.log(
      `\n  反思: "${exec.task.slice(0, 30)}..." [${exec.outcome}] → ${mode}`
    );
    const { id, heuristicText } = await reflectAndStore(exec, mode);
    if (id) {
      heuristicIds.push(id);
      console.log(`  → ${heuristicText}`);
    }
  }

  // ── 模式 2 演示：Shadow → Validate → Promote ──
  console.log("\n\n🔬 模式 2: Staged Rollout（Shadow → Validate → Promote）");
  console.log("─".repeat(40));

  for (let i = 0; i < Math.min(2, heuristicIds.length); i++) {
    const id = heuristicIds[i];
    if (!id) continue;

    console.log(`\n  验证 Heuristic [${id}]...`);
    const validation = await validateInShadow(id);
    console.log(
      `  → 评分: ${validation.score}/10 | 通过: ${validation.passed} | 原因: ${validation.reason}`
    );

    if (validation.passed) {
      await promoteToProduction(id);
      console.log(`  ✅ 已提升到 production`);
    } else {
      console.log(`  ⏸️  未通过，保留在 shadow`);
    }
  }

  // ── 模式 3 演示：Pending → 人工 Approve/Reject ──
  console.log("\n\n👤 模式 3: Human-in-the-Loop（Pending → Approve/Reject）");
  console.log("─".repeat(40));

  const pendingList = await getPendingHeuristics();
  console.log(`\n  待审核队列: ${pendingList.length} 条`);
  for (const item of pendingList) {
    console.log(`  [${item.id}] ${item.entry.heuristicText}`);
  }

  if (pendingList.length >= 1) {
    const first = pendingList[0];
    console.log(`\n  审核员: APPROVE [${first.id}]`);
    await approveHeuristic(first.id, "规则具体可执行，通过");
    console.log(`  ✅ 进入 production（confidence: 0.8）`);
  }

  if (pendingList.length >= 2) {
    const second = pendingList[1];
    console.log(`\n  审核员: REJECT [${second.id}]`);
    await rejectHeuristic(second.id, "过于泛化或场景有限");
    console.log(`  ❌ 已拒绝`);
  }

  // ── Phase 2: 新任务——带 Heuristic 指导 vs 无指导对比 ──
  console.log("\n\n📋 Phase 2: 新任务对比（有/无 Heuristic 指导）");
  console.log("─".repeat(40));

  const newTask =
    "用户（U12345）问'帮我推荐几款200块左右的运动手表'，搜索并推荐";

  const retrieved = await retrieveHeuristics(newTask);
  console.log(`\n  🔍 检索到 ${retrieved.length} 条 production Heuristic：`);
  retrieved.forEach((h, i) => console.log(`     ${i + 1}. ${h}`));

  console.log("\n  ▶ 带 Heuristic 指导的执行方案：");
  console.log("  " + "─".repeat(36));
  const result = await executeWithHeuristics(newTask, true);
  console.log("  " + result.split("\n").join("\n  "));

  // ── 最终 Pool 状态 ──
  console.log("\n\n📊 最终 Heuristic Pool 状态：");
  console.log("─".repeat(40));
  const all = await heuristicStore.search(HEURISTIC_NAMESPACE, {
    query: "all rules",
    limit: 100,
  });
  for (const item of all) {
    const v = item.value as any as HeuristicEntry;
    const statusIcon =
      v.status === "production"
        ? "✅"
        : v.status === "rejected"
          ? "❌"
          : v.status === "shadow"
            ? "👁️"
            : "⏳";
    console.log(
      `  ${statusIcon} [${v.status}] confidence=${v.confidence} | ${v.trigger} ${v.action}`
    );
  }

  console.log("\n" + "═".repeat(60));
  console.log("✅ 完整 ERL 循环演示完成：");
  console.log("   1. Agent 真实执行 → Tracer 记录轨迹");
  console.log("   2. LLM-as-Judge 评估 outcome");
  console.log("   3. 反思提炼 Heuristic → 质量控制审查");
  console.log("   4. 新任务带经验指导执行");
  console.log("═".repeat(60));
}

main().catch(console.error);
