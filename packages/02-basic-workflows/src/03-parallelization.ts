/**
 * 03 - Parallelization（并行化）
 *
 * ── 是什么 ──────────────────────────────────────────────────────────────────
 * 把互不依赖的子任务同时执行，用 Send API 动态派发，结果通过追加 reducer 汇总。
 * 总耗时 = 最慢子任务的时间（而非所有任务之和）。
 *
 * 图结构：
 *   START → dispatcher → (Send × N) → analyzeNode → aggregator → END
 *
 * ── Send API vs Promise.all ──────────────────────────────────────────────────
 * Promise.all：自己管理并发，LangGraph 完全不感知，Checkpointer/Streaming/重试全部失效
 * Send API：LangGraph 运行时管理，每个分支都有 checkpoint，支持断点恢复、streaming、错误处理
 *
 * ── 关键：追加 reducer ────────────────────────────────────────────────────────
 * 并行的多个节点会同时向同一个字段写入结果。
 * 必须用追加 reducer（prev.concat(next)），否则后写入的会覆盖前面的结果。
 *
 * ── 与其他模式的对比 ─────────────────────────────────────────────────────────
 * vs Prompt Chaining：串行（有依赖）vs 并行（无依赖），可组合使用
 * vs Routing：Routing 是 N 选 1；Parallelization 是全部同时跑
 *
 * ── 场景：多维度分析一篇文章 ─────────────────────────────────────────────────
 * 三个分析维度互不依赖，并行执行：
 *   1. 摘要：提炼核心内容
 *   2. 风险：识别潜在问题
 *   3. 亮点：提取创新之处
 */

import "dotenv/config";
import { StateGraph, Annotation, START, END, Send } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// ── State ──────────────────────────────────────────────────────────────────
const GraphState = Annotation.Root({
  article: Annotation<string>({ reducer: (_, n) => n }),

  // 并行任务的派发列表：dispatcher 节点写入，Send API 读取
  // 每个元素是一个分析维度的配置
  tasks: Annotation<Array<{ dimension: string; prompt: string }>>({
    reducer: (_, n) => n,
    default: () => [],
  }),

  // 并行节点的结果：必须用追加 reducer
  // 三个 analyzeNode 并发执行，各自追加一条结果到这个数组
  analyses: Annotation<Array<{ dimension: string; result: string }>>({
    reducer: (prev, next) => prev.concat(next),
    default: () => [],
  }),

  // 汇总节点的最终报告
  report: Annotation<string>({ reducer: (_, n) => n, default: () => "" }),
});

type State = typeof GraphState.State;

// analyzeNode 的局部 State：每个 Send 携带的数据结构
// Send 可以给节点传入任意数据，不必和主 State 结构完全一致
type AnalyzeInput = {
  article: string;
  dimension: string;
  prompt: string;
};

// ── LLM ───────────────────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

// ── 节点 ───────────────────────────────────────────────────────────────────

// 1. dispatcher 节点：把要派发的任务写入 State
function dispatcherNode(state: State): Partial<State> {
  const tasks = [
    { dimension: "摘要", prompt: "用3句话提炼文章的核心内容。" },
    { dimension: "风险", prompt: "识别文章中提到的潜在风险或问题，没有则说明。" },
    { dimension: "亮点", prompt: "提取文章中最有价值的创新点或关键洞察。" },
  ];
  console.log(`\n[dispatcher] 准备派发 ${tasks.length} 个并行任务`);
  return { tasks };
}

// 条件边函数：读取 State 中的任务列表，返回 Send[]
// Send API 的约定：条件边函数返回 Send[]，LangGraph 并发执行它们
// 与 dispatcherNode 分开，避免节点被执行两次
function createSends(state: State): Send[] {
  return state.tasks.map(
    ({ dimension, prompt }) =>
      new Send("analyzeNode", {
        article: state.article,
        dimension,
        prompt,
      } satisfies AnalyzeInput)
  );
}

// 2. analyzeNode：被并发调用 N 次，每次处理一个维度
// 注意：参数类型是 AnalyzeInput，而不是完整的 State
// Send 传入什么，节点就收到什么
async function analyzeNode(input: AnalyzeInput): Promise<Partial<State>> {
  console.log(`[analyzeNode] 开始分析维度: ${input.dimension}`);

  const response = await llm.invoke([
    new SystemMessage(`你是一个文章分析师。任务：${input.prompt} 回复控制在100字以内。`),
    new HumanMessage(`文章内容：\n${input.article}`),
  ]);

  const result = response.content as string;
  console.log(`[analyzeNode] 完成: ${input.dimension}`);

  // 返回追加到 analyses 数组的一条记录
  return {
    analyses: [{ dimension: input.dimension, result }],
  };
}

// 3. aggregator 节点：所有并行任务完成后才执行，汇总所有分析结果
async function aggregatorNode(state: State): Promise<Partial<State>> {
  console.log(`\n[aggregator] 汇总 ${state.analyses.length} 个分析结果`);

  const analysesText = state.analyses
    .map((a) => `【${a.dimension}】\n${a.result}`)
    .join("\n\n");

  const response = await llm.invoke([
    new SystemMessage("你是一个报告撰写员。将以下多维度分析整合成一份简洁的综合报告。"),
    new HumanMessage(`各维度分析结果：\n\n${analysesText}`),
  ]);

  return { report: response.content as string };
}

// ── 构建图 ─────────────────────────────────────────────────────────────────
const graph = new StateGraph(GraphState)
  .addNode("dispatcher", dispatcherNode)
  .addNode("analyzeNode", analyzeNode)
  .addNode("aggregator", aggregatorNode)
  .addEdge(START, "dispatcher")
  // dispatcher 节点执行完后，由 createSends 函数决定派发哪些并行任务
  .addConditionalEdges("dispatcher", createSends)
  // 所有 analyzeNode 完成后，汇聚到 aggregator（fan-in）
  .addEdge("analyzeNode", "aggregator")
  .addEdge("aggregator", END)
  .compile();

// ── 运行 ───────────────────────────────────────────────────────────────────
const article = `
LangGraph 是 LangChain 团队推出的用于构建有状态 Agent 的框架。
它以图（Graph）为核心抽象，节点代表计算单元，边代表数据流向。
相比传统的 Chain，LangGraph 支持循环、条件分支和状态持久化，
使得复杂的 Agent 行为（如 ReAct 循环、Human-in-the-loop）得以优雅实现。
目前已被多家企业用于生产环境的 AI 工作流编排。
`;

console.log("═".repeat(50));
console.log("Parallelization：并行多维度分析文章");
console.log("═".repeat(50));

const startTime = Date.now();
const result = await graph.invoke({ article });
const elapsed = Date.now() - startTime;

console.log("\n" + "═".repeat(50));
console.log(`执行完成，耗时 ${elapsed}ms`);
console.log("═".repeat(50));

console.log("\n── 各维度分析结果 ──");
for (const analysis of result.analyses) {
  console.log(`\n【${analysis.dimension}】`);
  console.log(analysis.result);
}

console.log("\n── 综合报告 ──");
console.log(result.report);
