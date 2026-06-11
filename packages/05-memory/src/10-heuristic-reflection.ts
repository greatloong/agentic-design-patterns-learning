/**
 * 10 - Heuristic Reflection：投资分析 Agent 从任务结果中反思提炼经验（ERL 风格）
 *
 * ── 与 09 的区别 ─────────────────────────────────────────────────────────
 *
 *   09: 用户说"太正式了" → 提炼偏好规则（用户教 Agent 做人）
 *   10: Agent 执行投资分析 → 自己反思流程缺陷与依据可靠性 → 提炼策略规则
 *
 * ── 架构（对应 ERL 论文的 Hot/Cold Path 分离）──────────────────────────────
 *
 *   Hot Path（在线，延迟敏感）:
 *     LangGraph ReAct loop: agent node ⇄ tools node (web_search)
 *     ├─ Tracer: trajectory 内嵌于 graph state，节点执行时零成本追加记录
 *     └─ agent node: 分层 system prompt
 *          Layer 1 Core（不可变核心指令）
 *          Layer 2 Learned Heuristics（每次执行时从本地文件动态加载）
 *
 *   Cold Path（离线，延迟不敏感）:
 *     任务结束后 → 反思 LLM 分析 trajectory，两个维度：
 *       A. 执行流程改进：报错、重复调用、低效查询、死循环
 *       B. 结论依据真实性：论断是否有搜索结果出处、是否把推测当事实、
 *          是否用过时数据 → 提炼提升分析准确性的规则
 *     → Heuristic 直接写入本地文件（无审核环节）
 *     → 最多保留 20 条，FIFO 淘汰最老的
 *
 * ── 运行 ─────────────────────────────────────────────────────────────────
 *
 *   pnpm run 10            正常运行（heuristics 跨运行累积，体现持续学习）
 *   pnpm run 10 -- --fresh 清空 heuristic 文件后运行
 */

import "dotenv/config";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { ChatOpenAI } from "@langchain/openai";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  Annotation,
  MessagesAnnotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";

// ── 模型配置 ────────────────────────────────────────────────────────────

const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
  streaming: true, // messages 流模式依赖模型 token streaming
});

const reflectionLlm = new ChatOpenAI({
  model: "deepseek-v4-flash",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
  temperature: 0,
});

// ══════════════════════════════════════════════════════════════════════════
// Heuristic 持久化：本地文件，最多 20 条，FIFO 淘汰最老的
// ══════════════════════════════════════════════════════════════════════════

const DATA_DIR = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  "../heuristics",
);
const HEURISTIC_FILE = path.join(DATA_DIR, "heuristics-10.json");
const MAX_HEURISTICS = 20;

interface HeuristicEntry {
  id: string;
  dimension: "flow" | "evidence"; // A 流程改进 | B 依据真实性
  trigger: string;
  action: string;
  rationale: string;
  sourceTask: string;
  createdAt: string;
}

function loadHeuristics(): HeuristicEntry[] {
  try {
    return JSON.parse(fs.readFileSync(HEURISTIC_FILE, "utf-8"));
  } catch {
    return [];
  }
}

function addHeuristics(newEntries: HeuristicEntry[]): {
  total: number;
  evicted: number;
} {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const all = [...loadHeuristics(), ...newEntries];
  // FIFO：数组按插入顺序即时间顺序，超出上限时淘汰最早的
  const evicted = Math.max(0, all.length - MAX_HEURISTICS);
  const kept = all.slice(-MAX_HEURISTICS);
  fs.writeFileSync(HEURISTIC_FILE, JSON.stringify(kept, null, 2), "utf-8");
  return { total: kept.length, evicted };
}

// ══════════════════════════════════════════════════════════════════════════
// 工具：web_search（Brave Search API）
// ══════════════════════════════════════════════════════════════════════════


const webSearch = tool(
  async ({ query }: { query: string }) => {
    const url = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=5`;
    const resp = await fetch(url, {
      headers: {
        Accept: "application/json",
        "X-Subscription-Token": process.env.BRAVE_API_KEY!,
      },
    });
    if (!resp.ok) {
      throw new Error(`Brave API ${resp.status}: ${await resp.text()}`);
    }

    const data: any = await resp.json();
    const results = data.web?.results ?? [];
    if (results.length === 0) return `搜索"${query}"未找到结果`;

    return results
      .slice(0, 5)
      .map((r: any, i: number) => {
        const desc = (r.description ?? "").replace(/<[^>]+>/g, "");
        const age = r.page_age ? ` (${r.page_age.slice(0, 10)})` : "";
        return `${i + 1}. ${r.title}${age}\n   来源: ${r.url}\n   ${desc}`;
      })
      .join("\n");
  },
  {
    name: "web_search",
    description:
      "搜索互联网获取最新的财经/市场/公司信息。每次输入一个具体的查询关键词。",
    schema: z.object({
      query: z
        .string()
        .describe("搜索关键词，如 'NVDA stock price earnings 2026'"),
    }),
  },
);

const TOOLS_BY_NAME: Record<string, typeof webSearch> = {
  web_search: webSearch,
};

const llmWithTools = llm.bindTools([webSearch]);

// ══════════════════════════════════════════════════════════════════════════
// Graph State：messages + trajectory（Hot Path Tracer 的载体）
// ══════════════════════════════════════════════════════════════════════════

const GraphState = Annotation.Root({
  ...MessagesAnnotation.spec,
  // Tracer：每个节点把自己做的事 append 进来，零 LLM 成本
  trajectory: Annotation<string[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
  loopCount: Annotation<number>({
    reducer: (_, b) => b,
    default: () => 0,
  }),
});

type State = typeof GraphState.State;

const MAX_LOOPS = 6;

// ══════════════════════════════════════════════════════════════════════════
// 分层 System Prompt：Core（不可变）+ Learned Heuristics（动态加载）
// ══════════════════════════════════════════════════════════════════════════

const CORE_PROMPT = `你是一个投资分析 Agent。用户会给你一个投资分析任务。

## 核心要求（不可变）
- 用 web_search 工具收集最新信息（股价、财报、新闻、分析师观点）
- 每次搜索用具体、聚焦的关键词；信息足够后停止搜索
- 最终输出：明确的 买入/持有/卖出 倾向 + 核心依据（标注信息来源）
- 这不是投资建议，仅为研究分析`;

function buildLayeredSystemPrompt(): {
  prompt: string;
  loadedCount: number;
} {
  // Layer 2：每次调用时从本地文件动态加载（cold path 的学习成果即时生效）
  const heuristics = loadHeuristics();

  let prompt = CORE_PROMPT;
  if (heuristics.length > 0) {
    const lines = heuristics
      .map((h, i) => `${i + 1}. [${h.dimension}] ${h.trigger} ${h.action}`)
      .join("\n");
    prompt += `\n\n## 来自过往任务反思的经验规则（Learned Layer，动态加载）\n${lines}`;
  }
  return { prompt, loadedCount: heuristics.length };
}

// ══════════════════════════════════════════════════════════════════════════
// Graph Nodes（Hot Path）
// ══════════════════════════════════════════════════════════════════════════

async function agentNode(state: State): Promise<Partial<State>> {
  const { prompt, loadedCount } = buildLayeredSystemPrompt();
  const loop = state.loopCount + 1;

  const resp = (await llmWithTools.invoke([
    new SystemMessage(prompt),
    ...state.messages,
  ])) as AIMessage;

  const traces: string[] = [];
  if (loop === 1) {
    traces.push(
      `[Tracer] agent 启动，Learned Layer 注入 ${loadedCount} 条 heuristic`,
    );
  }

  if (resp.tool_calls && resp.tool_calls.length > 0) {
    for (const tc of resp.tool_calls) {
      traces.push(`[决策] 调用 ${tc.name}(${JSON.stringify(tc.args)})`);
    }
  } else {
    const text =
      typeof resp.content === "string"
        ? resp.content
        : JSON.stringify(resp.content);
    traces.push(`[最终结论]\n${text}`);
  }

  return { messages: [resp], trajectory: traces, loopCount: loop };
}

async function toolsNode(state: State): Promise<Partial<State>> {
  const last = state.messages.at(-1) as AIMessage;
  const outputs: ToolMessage[] = [];
  const traces: string[] = [];

  for (const tc of last.tool_calls ?? []) {
    const fn = TOOLS_BY_NAME[tc.name];
    try {
      if (!fn) throw new Error(`工具 ${tc.name} 不存在`);
      const result = (await fn.invoke(tc.args as any)) as string;
      traces.push(
        `[工具] ${tc.name}(${JSON.stringify(tc.args)})\n  → ${result.slice(0, 600)}`,
      );
      outputs.push(new ToolMessage({ content: result, tool_call_id: tc.id! }));
    } catch (e: any) {
      traces.push(`[工具][ERROR] ${tc.name} → ${e.message}`);
      outputs.push(
        new ToolMessage({
          content: `Error: ${e.message}`,
          tool_call_id: tc.id!,
        }),
      );
    }
  }

  return { messages: outputs, trajectory: traces };
}

function shouldContinue(state: State): "tools" | typeof END {
  const last = state.messages.at(-1) as AIMessage;
  if (state.loopCount >= MAX_LOOPS) {
    return END; // 防失控：超过最大循环数强制结束
  }
  return last.tool_calls && last.tool_calls.length > 0 ? "tools" : END;
}

const graph = new StateGraph(GraphState)
  .addNode("agent", agentNode)
  .addNode("tools", toolsNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, ["tools", END])
  .addEdge("tools", "agent")
  .compile();

/**
 * Hot Path：执行任务（streaming 输出），返回轨迹 + 最终结论
 *
 * 流式输出策略（只给提示，不刷屏）：
 *   - thinking token：仅在本 turn 首次出现时打印一行「💭 思考中…」提示，
 *     不展示具体思考内容（太长、会刷屏）
 *   - tool_call 参数：仅在本 turn 首次出现时打印一行「🔧 调用工具…」提示，
 *     完整的调用参数随后由 [决策] 行干净展示
 *   - 正文 token：正常流式输出
 *   - [决策] / [工具] 等结构化行：用 console.log 正常打印
 */
async function runTask(task: string): Promise<{
  trajectory: string;
  finalAnswer: string;
}> {
  const stream = await graph.stream(
    { messages: [new HumanMessage(task)] },
    { recursionLimit: 30, streamMode: ["messages", "updates"] },
  );

  const trajectoryLines: string[] = [];
  let finalAnswer = "";

  // 当前 agent turn 的状态（updates 事件到达时重置）
  let reasoningHinted = false; // 本 turn 是否已打印「思考中」提示
  let toolHinted = false; // 本 turn 是否已打印「调用工具」提示
  let textStreamed = false;

  for await (const item of stream) {
    const [mode, payload] = item as [string, any];

    if (mode === "messages") {
      const [chunk, meta] = payload as [any, Record<string, any>];
      // 只处理 agent 节点的 AI 输出；ToolMessage 等不在此流式打印
      if (meta?.langgraph_node !== "agent") continue;
      const type = chunk.getType?.() ?? chunk._getType?.();
      if (type !== "ai") continue;

      // 1) thinking token：本 turn 首次出现时打印一次提示，不展示具体内容
      const reasoning = chunk.additional_kwargs?.reasoning_content;
      if (reasoning) {
        if (!reasoningHinted) {
          console.log("  💭 思考中…");
          reasoningHinted = true;
        }
        continue;
      }

      // 2) tool_call_chunks：本 turn 首次出现时打印一次提示（完整参数见后续 [决策] 行）
      if (chunk.tool_call_chunks?.length > 0 && !chunk.content) {
        if (!toolHinted) {
          console.log("  🔧 调用工具…");
          toolHinted = true;
        }
        continue;
      }

      // 3) 正文 token：正常流式输出
      const text =
        typeof chunk.content === "string"
          ? chunk.content
          : Array.isArray(chunk.content)
            ? chunk.content
                .map((b: any) => (typeof b === "string" ? b : (b?.text ?? "")))
                .join("")
            : "";
      if (text) {
        if (!textStreamed) process.stdout.write("\n");
        textStreamed = true;
        process.stdout.write(text);
      }
    } else if (mode === "updates") {
      const update = payload as Record<string, any>;

      if (update.agent) {
        const traces: string[] = update.agent.trajectory ?? [];
        trajectoryLines.push(...traces);

        for (const t of traces) {
          if (t.startsWith("[最终结论]")) {
            finalAnswer = t.replace(/^\[最终结论\]\n?/, "");
            if (textStreamed) process.stdout.write("\n"); // 流式正文收尾
          } else {
            // [Tracer] / [决策] 行：完整 tool call 的干净展示
            console.log(`  ${t}`);
          }
        }
        // 重置 turn 状态
        reasoningHinted = false;
        toolHinted = false;
        textStreamed = false;
      }

      if (update.tools) {
        const traces: string[] = update.tools.trajectory ?? [];
        trajectoryLines.push(...traces);
        for (const t of traces) {
          // live 显示截断到 200 字符；完整 600 字符版本保留在 trajectory 供反思
          const firstLine = t.split("\n")[0];
          const body = t.slice(firstLine.length).replace(/\s+/g, " ").trim();
          console.log(`  ${firstLine}`);
          if (body)
            console.log(
              `    ${body.slice(0, 200)}${body.length > 200 ? "…" : ""}`,
            );
        }
      }
    }
  }

  const trajectory = trajectoryLines
    .map((t, i) => `Step ${i + 1}: ${t}`)
    .join("\n");

  return { trajectory, finalAnswer: finalAnswer || "（无最终结论）" };
}

// ══════════════════════════════════════════════════════════════════════════
// Cold Path：反思提炼 Heuristic（两个维度，无审核，直接入池）
// ══════════════════════════════════════════════════════════════════════════

const REFLECTION_PROMPT = `你是一个投资分析 Agent 的经验反思器。分析以下任务执行轨迹，提炼可复用的经验规则。

## 任务
{task}

## 执行轨迹（含每次搜索的关键词、返回结果、最终结论）
{trajectory}

## 你的任务：从两个维度反思

### 维度 A — 执行流程改进（dimension: "flow"）
- 工具调用是否有报错、重复搜索、关键词过宽/过窄导致结果质量差？
- 是否搜索次数过多/过少？是否有明显遗漏的信息源（如财报、分析师评级）？

### 维度 B — 结论依据真实性（dimension: "evidence"）
逐条检查最终结论中的关键论断（股价、财务数字、估值、评级、事件）：
- 该论断能否在搜索结果中找到明确出处？还是 Agent 凭参数化记忆编造的？
- 是否把过时信息当作最新信息？是否把推测/观点当作事实陈述？
- 提炼出能提升分析准确性的规则

## 要求
- 每个维度最多提炼 2 条，总共最多 3 条；没有值得提炼的维度就跳过
- 规则必须可迁移（适用于其他股票/资产的分析，不绑定本次的具体公司）
- 规则必须具体可执行，不要"要小心""要严谨"这类废话`;

// Heuristic 的结构化输出 schema（withStructuredOutput 底层走 function calling，
// 顶层必须是 object，所以把数组包一层）
const reflectionSchema = z.object({
  heuristics: z
    .array(
      z.object({
        dimension: z
          .enum(["flow", "evidence"])
          .describe("flow=执行流程改进, evidence=结论依据真实性"),
        trigger: z.string().describe("IF [触发条件]"),
        action: z.string().describe("THEN [行动建议]"),
        rationale: z.string().describe("因为[原因]"),
      }),
    )
    .max(3)
    .describe("提炼出的经验规则，最多 3 条"),
});

const structuredReflectionLlm =
  reflectionLlm.withStructuredOutput(reflectionSchema);

async function reflectAndStore(
  task: string,
  trajectory: string,
): Promise<HeuristicEntry[]> {
  const prompt = REFLECTION_PROMPT.replace("{task}", task).replace(
    "{trajectory}",
    trajectory,
  );

  let parsed: z.infer<typeof reflectionSchema>;
  try {
    parsed = await structuredReflectionLlm.invoke([new HumanMessage(prompt)]);
  } catch (e: any) {
    console.log(`  ⚠️ 反思结构化输出失败，跳过：${e.message}`);
    return [];
  }

  const entries: HeuristicEntry[] = parsed.heuristics.slice(0, 3).map((p) => ({
    id: `h-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    dimension: p.dimension,
    trigger: p.trigger,
    action: p.action,
    rationale: p.rationale,
    sourceTask: task.slice(0, 50),
    createdAt: new Date().toISOString(),
  }));

  if (entries.length > 0) {
    const { total, evicted } = addHeuristics(entries);
    console.log(
      `  💾 写入 ${entries.length} 条 → 文件共 ${total} 条${evicted > 0 ? `（FIFO 淘汰了最老的 ${evicted} 条）` : ""}`,
    );
  }
  return entries;
}

// ══════════════════════════════════════════════════════════════════════════
// Demo
// ══════════════════════════════════════════════════════════════════════════

async function main() {
  if (process.argv.includes("--fresh")) {
    fs.rmSync(HEURISTIC_FILE, { force: true });
    console.log("🗑️  已清空 heuristic 文件\n");
  }

  console.log("═".repeat(60));
  console.log("10 - Heuristic Reflection: 投资分析 Agent（ERL 风格）");
  console.log("═".repeat(60));
  console.log(`\nHeuristic 文件: ${HEURISTIC_FILE}`);
  console.log(`当前池中: ${loadHeuristics().length}/${MAX_HEURISTICS} 条`);

  // ── Phase 1: Hot Path 执行任务（Tracer 自动记录轨迹）──
  const task1 =
    "分析英伟达(NVDA)当前的投资价值：结合最新股价、最近一季财报和近期新闻，给出买入/持有/卖出倾向和核心依据";

  console.log("\n" + "─".repeat(60));
  console.log(`📈 Phase 1 [Hot Path] 执行任务:\n   ${task1}`);
  console.log("─".repeat(60));

  const run1 = await runTask(task1);
  console.log(
    `\n📝 Tracer 共记录 ${run1.trajectory.split("\nStep ").length} 步（完整轨迹交给 Cold Path 反思）`,
  );

  // ── Cold Path: 反思提炼（两个维度）──
  console.log("\n" + "─".repeat(60));
  console.log(
    "🧠 [Cold Path] 反思提炼 Heuristic（维度 A 流程 / 维度 B 依据真实性）",
  );
  console.log("─".repeat(60));

  const learned = await reflectAndStore(task1, run1.trajectory);
  for (const h of learned) {
    console.log(`\n  [${h.dimension}] ${h.trigger}`);
    console.log(`         ${h.action}`);
    console.log(`         (${h.rationale})`);
  }

  // ── Phase 2: 新任务，agent node 动态加载 Learned Layer ──
  const task2 =
    "分析特斯拉(TSLA)当前的投资价值：结合最新股价、交付数据和近期新闻，给出买入/持有/卖出倾向和核心依据";

  console.log("\n" + "─".repeat(60));
  console.log(
    `📈 Phase 2 [Hot Path] 新任务（system prompt 动态加载 ${loadHeuristics().length} 条 heuristic）:\n   ${task2}`,
  );
  console.log("─".repeat(60));

  const run2 = await runTask(task2);
  console.log(
    `\n📝 Tracer 共记录 ${run2.trajectory.split("\nStep ").length} 步（完整轨迹交给 Cold Path 反思）`,
  );

  // Phase 2 同样会反思 → 持续学习（池子持续增长，到 20 条触发 FIFO 淘汰）
  console.log("\n" + "─".repeat(60));
  console.log("🧠 [Cold Path] 对 Phase 2 同样反思（持续学习）");
  console.log("─".repeat(60));
  await reflectAndStore(task2, run2.trajectory);

  // ── 最终池状态 ──
  const pool = loadHeuristics();
  console.log("\n" + "═".repeat(60));
  console.log(
    `📊 最终 Heuristic 池（${pool.length}/${MAX_HEURISTICS}，文件持久化，跨运行累积）`,
  );
  console.log("═".repeat(60));
  for (const h of pool) {
    console.log(`  [${h.dimension}] ${h.trigger} ${h.action}`);
  }

  console.log("\n" + "═".repeat(60));
  console.log("✅ 演示完成：");
  console.log("   1. Hot Path: LangGraph ReAct loop（agent ⇄ web_search）");
  console.log("   2. Tracer 内嵌 graph state，零成本记录轨迹");
  console.log("   3. Cold Path 反思：A 流程改进 + B 依据真实性");
  console.log("   4. Heuristic 直接写本地文件（无审核），上限 20 条 FIFO 淘汰");
  console.log(
    "   5. agent node 分层 system prompt：Core + Learned（动态加载）",
  );
  console.log("═".repeat(60));
}

main().catch(console.error);
