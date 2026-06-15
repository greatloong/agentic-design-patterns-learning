/**
 * 03 - Hierarchical（分层多 Agent）
 *
 * ── 是什么 ──────────────────────────────────────────────────────────────────
 * Supervisor V2 的递归套娃。
 * 顶层 Supervisor 不直接管 worker，而是管"中层 Team"，每个 Team 内部
 * 又是一个 Supervisor 管自己的 worker。
 *
 *                    Top Supervisor
 *                   ↙              ↘
 *           Research Team       Writing Team
 *           (子 Supervisor)     (子 Supervisor)
 *            ↙        ↘            ↙        ↘
 *         search    summarize   drafter   editor
 *
 * ── 为什么要分层（核心动机）──────────────────────────────────────────────
 * 单层 Supervisor 在以下情况会崩：
 *   1) Agent 数量多（10+）→ supervisor prompt 巨长，路由准确率暴跌
 *   2) 职责需要分组 → 同组 Agent 高内聚，跨组应该低耦合
 *   3) 上下文污染 → 所有 Agent 中间产物堆给顶层 supervisor 看
 *
 * 分层后：
 *   - 每层 supervisor 只在 2-4 个选项中决策，准确率高
 *   - Team 内部细节不冒泡到顶层
 *   - 真实业务（产品/工程/法务）天然分组，prompt 写起来自然
 *
 * ── 关键实现要点 ──────────────────────────────────────────────────────────
 * 1. 每个 Team 是一个独立 compile() 的子图
 *    对父图来说，Team 就是一个普通 node
 *
 * 2. Team 子图与父图的接口契约：
 *      父 → Team：传入 task: string（高层任务描述）
 *      Team → 父：返回 summary: string（最终产出，单条 AIMessage）
 *    Team 内部的 messages / 中间决策一概不冒泡
 *    （Isolated + Summary 模式，从 V2 继承下来叠加使用）
 *
 * 3. 顶层 Supervisor 的 routing schema 只列 Team 名，不列 worker 名
 *    顶层只需关心"该让哪个 Team 干"，worker 调度交给 Team 内部 supervisor
 *
 * 4. 状态隔离三层：
 *      Top State：messages + next + finalReport
 *      ResearchTeam State：messages + next + research_notes
 *      WritingTeam State：messages + next + draft
 *    各自独立 Annotation.Root，互不可见
 *
 * ── 与 Supervisor V2 的对比 ──────────────────────────────────────────────
 *   V2：1 个 supervisor → 2 个 worker
 *   本例：1 个顶层 supervisor → 2 个 Team supervisor → 4 个 worker
 *   多了一层调度，多了一些 LLM 调用，但换来：
 *     - 容易扩展到 N 个 Team × M 个 worker
 *     - 每层 prompt 短、决策准
 *     - Team 可独立替换/单测
 *
 * ── 何时不要用 Hierarchical ──────────────────────────────────────────────
 * - worker ≤ 5 个，单层 Supervisor 完全够用，分层是过度设计
 * - 强行 2 个 Team 各塞 2 个 worker → 纯粹增加一次顶层 LLM 调用
 *
 * ── 本例场景 ──────────────────────────────────────────────────────────────
 * 用户给一个研究主题，系统产出一篇短报告：
 *   Research Team：负责搜资料、做笔记
 *     - searcher：调 Brave Search 找资料
 *     - summarizer：把搜索结果整理成结构化笔记
 *   Writing Team：基于笔记写报告
 *     - drafter：起草报告
 *     - editor：润色定稿
 *
 * 为减少真实 API 调用、专注架构：所有 worker 都用 LLM 直接生成内容，
 * searcher 用 mock 数据（生产里替换为真实 search tool 即可）
 */

import "dotenv/config";
import {
  StateGraph,
  START,
  END,
  Annotation,
  MemorySaver,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { z } from "zod";

// ── LLM ───────────────────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

// ════════════════════════════════════════════════════════════════════════
// LAYER 1: Research Team（子图）
// ════════════════════════════════════════════════════════════════════════
// 子图 State：内部独立维护
const ResearchState = Annotation.Root({
  task: Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  // 内部进度
  searchResults: Annotation<string>({
    reducer: (_p, n) => n,
    default: () => "",
  }),
  notes: Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  // Team supervisor 决策字段
  next: Annotation<"searcher" | "summarizer" | "FINISH">({
    reducer: (_p, n) => n,
    default: () => "searcher",
  }),
});

// Worker 1: searcher（mock，生产里换成真实 search tool）
async function searcherNode(state: typeof ResearchState.State) {
  console.log(`  [Research/searcher] 搜索: ${state.task}`);
  // mock 一些资料
  const mockResults = `搜索结果（关于"${state.task}"）：
[1] LangGraph 是 LangChain 推出的状态机式 Agent 编排框架
[2] 多 Agent 系统主要有 Supervisor / Swarm / Hierarchical 三种范式
[3] Hierarchical 适合大型企业系统，能映射真实组织架构
[4] 工业界普遍采用 Supervisor 作为生产首选`;
  return { searchResults: mockResults };
}

// Worker 2: summarizer（用 LLM 把搜索结果整理成笔记）
async function summarizerNode(state: typeof ResearchState.State) {
  console.log(`  [Research/summarizer] 整理笔记`);
  const resp = (await llm.invoke([
    new SystemMessage(
      "你是研究助手。把下面的原始搜索结果整理成 3-5 条结构化笔记，每条一行，简洁有力。"
    ),
    new HumanMessage(`任务：${state.task}\n\n原始资料：\n${state.searchResults}`),
  ])) as AIMessage;
  return { notes: resp.content as string };
}

// Team Supervisor：决定下一步是 searcher / summarizer / FINISH
const researchRouteSchema = z.object({
  next: z.enum(["searcher", "summarizer", "FINISH"]),
  reason: z.string(),
});

async function researchSupervisorNode(state: typeof ResearchState.State) {
  console.log(
    `  [Research/supervisor] 决策（已有 search=${!!state.searchResults}, notes=${!!state.notes}）`
  );
  const decision = await llm.withStructuredOutput(researchRouteSchema, {
    method: "functionCalling",
  }).invoke([
    new SystemMessage(
      `你是 Research Team 的调度官。规则：
- 如果还没有搜索结果 → next=searcher
- 已有搜索结果但还没整理笔记 → next=summarizer
- 笔记已生成 → next=FINISH
严格按以上顺序，不要跳步。`
    ),
    new HumanMessage(
      `任务: ${state.task}\n搜索结果是否就绪: ${!!state.searchResults}\n笔记是否就绪: ${!!state.notes}`
    ),
  ]);
  console.log(`  [Research/supervisor] → ${decision.next}（${decision.reason}）`);
  return { next: decision.next };
}

function routeResearch(state: typeof ResearchState.State) {
  if (state.next === "FINISH") return END;
  return state.next;
}

const researchTeamGraph = new StateGraph(ResearchState)
  .addNode("supervisor", researchSupervisorNode)
  .addNode("searcher", searcherNode)
  .addNode("summarizer", summarizerNode)
  .addEdge(START, "supervisor")
  .addConditionalEdges("supervisor", routeResearch, {
    searcher: "searcher",
    summarizer: "summarizer",
    [END]: END,
  })
  .addEdge("searcher", "supervisor")
  .addEdge("summarizer", "supervisor")
  .compile();

// ════════════════════════════════════════════════════════════════════════
// LAYER 1: Writing Team（子图）
// ════════════════════════════════════════════════════════════════════════
const WritingState = Annotation.Root({
  task: Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  // 输入（从顶层传入）
  notes: Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  // 内部进度
  draft: Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  finalDoc: Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  next: Annotation<"drafter" | "editor" | "FINISH">({
    reducer: (_p, n) => n,
    default: () => "drafter",
  }),
});

async function drafterNode(state: typeof WritingState.State) {
  console.log(`  [Writing/drafter] 起草`);
  const resp = (await llm.invoke([
    new SystemMessage(
      "你是撰稿人。基于研究笔记写一篇 200 字以内的短报告，只写正文，不要标题。"
    ),
    new HumanMessage(`主题：${state.task}\n\n研究笔记：\n${state.notes}`),
  ])) as AIMessage;
  return { draft: resp.content as string };
}

async function editorNode(state: typeof WritingState.State) {
  console.log(`  [Writing/editor] 润色`);
  const resp = (await llm.invoke([
    new SystemMessage(
      "你是编辑。润色下面的草稿：保持核心信息，提升表达流畅度，加一个标题。直接输出最终稿，不要解释你做了什么修改。"
    ),
    new HumanMessage(`草稿：\n${state.draft}`),
  ])) as AIMessage;
  return { finalDoc: resp.content as string };
}

const writingRouteSchema = z.object({
  next: z.enum(["drafter", "editor", "FINISH"]),
  reason: z.string(),
});

async function writingSupervisorNode(state: typeof WritingState.State) {
  console.log(
    `  [Writing/supervisor] 决策（draft=${!!state.draft}, final=${!!state.finalDoc}）`
  );
  const decision = await llm.withStructuredOutput(writingRouteSchema, {
    method: "functionCalling",
  }).invoke([
    new SystemMessage(
      `你是 Writing Team 调度官。规则：
- 还没有草稿 → drafter
- 有草稿但没定稿 → editor
- 已有定稿 → FINISH`
    ),
    new HumanMessage(
      `草稿就绪: ${!!state.draft}\n定稿就绪: ${!!state.finalDoc}`
    ),
  ]);
  console.log(`  [Writing/supervisor] → ${decision.next}（${decision.reason}）`);
  return { next: decision.next };
}

function routeWriting(state: typeof WritingState.State) {
  if (state.next === "FINISH") return END;
  return state.next;
}

const writingTeamGraph = new StateGraph(WritingState)
  .addNode("supervisor", writingSupervisorNode)
  .addNode("drafter", drafterNode)
  .addNode("editor", editorNode)
  .addEdge(START, "supervisor")
  .addConditionalEdges("supervisor", routeWriting, {
    drafter: "drafter",
    editor: "editor",
    [END]: END,
  })
  .addEdge("drafter", "supervisor")
  .addEdge("editor", "supervisor")
  .compile();

// ════════════════════════════════════════════════════════════════════════
// LAYER 0: Top Supervisor（父图）
// ════════════════════════════════════════════════════════════════════════
// 关键：父图只关心 Team 级调度，不关心 worker
const TopState = Annotation.Root({
  topic: Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  // Team 产出（父图只看接口契约，不看内部）
  notes: Annotation<string>({ reducer: (_p, n) => n, default: () => "" }),
  finalReport: Annotation<string>({
    reducer: (_p, n) => n,
    default: () => "",
  }),
  next: Annotation<"research_team" | "writing_team" | "FINISH">({
    reducer: (_p, n) => n,
    default: () => "research_team",
  }),
});

// Team 节点：调用子图，提取产出回写到父 State
// ★ 这就是 Hierarchical 的核心粘合层 ★
async function researchTeamNode(state: typeof TopState.State) {
  console.log(`\n[TOP] 调度 → Research Team`);
  // 调用子图，传入接口契约要求的 task 字段
  const result = await researchTeamGraph.invoke({ task: state.topic });
  // 仅把"产出"回写到父 State，丢弃子图的 messages / next / 中间状态
  return { notes: result.notes };
}

async function writingTeamNode(state: typeof TopState.State) {
  console.log(`\n[TOP] 调度 → Writing Team`);
  const result = await writingTeamGraph.invoke({
    task: state.topic,
    notes: state.notes,
  });
  return { finalReport: result.finalDoc };
}

const topRouteSchema = z.object({
  next: z.enum(["research_team", "writing_team", "FINISH"]),
  reason: z.string(),
});

async function topSupervisorNode(state: typeof TopState.State) {
  console.log(
    `\n[TOP/supervisor] 决策（notes=${!!state.notes}, report=${!!state.finalReport}）`
  );
  const decision = await llm.withStructuredOutput(topRouteSchema, {
    method: "functionCalling",
  }).invoke([
    new SystemMessage(
      `你是顶层调度官，管理 Research Team 和 Writing Team。规则：
- 还没有研究笔记 → research_team
- 有研究笔记但还没有最终报告 → writing_team
- 最终报告已就绪 → FINISH`
    ),
    new HumanMessage(
      `主题: ${state.topic}\n研究笔记就绪: ${!!state.notes}\n最终报告就绪: ${!!state.finalReport}`
    ),
  ]);
  console.log(`[TOP/supervisor] → ${decision.next}（${decision.reason}）`);
  return { next: decision.next };
}

function routeTop(state: typeof TopState.State) {
  if (state.next === "FINISH") return END;
  return state.next;
}

const topGraph = new StateGraph(TopState)
  .addNode("supervisor", topSupervisorNode)
  .addNode("research_team", researchTeamNode)
  .addNode("writing_team", writingTeamNode)
  .addEdge(START, "supervisor")
  .addConditionalEdges("supervisor", routeTop, {
    research_team: "research_team",
    writing_team: "writing_team",
    [END]: END,
  })
  .addEdge("research_team", "supervisor")
  .addEdge("writing_team", "supervisor")
  .compile({ checkpointer: new MemorySaver() });

// ── Demo ──────────────────────────────────────────────────────────────────
async function main() {
  const topic = "多 Agent 系统的三种主流架构对比";
  console.log("═".repeat(70));
  console.log(`【主题】${topic}`);
  console.log("═".repeat(70));

  const final = await topGraph.invoke(
    { topic },
    { configurable: { thread_id: "hier-1" }, recursionLimit: 25 }
  );

  console.log("\n" + "═".repeat(70));
  console.log("【最终报告】");
  console.log("═".repeat(70));
  console.log(final.finalReport);
  console.log("\n" + "─".repeat(70));
  console.log("【State 隔离效果】父图 State 只保留高层契约字段：");
  console.log(`  - topic:       "${final.topic}"`);
  console.log(`  - notes:       ${final.notes.length} 字符（Research Team 产出）`);
  console.log(`  - finalReport: ${final.finalReport.length} 字符（Writing Team 产出）`);
  console.log(
    `  注意：子图内部的 searchResults / draft / 各层 next 字段，父图完全看不到`
  );
}

main().catch(console.error);
