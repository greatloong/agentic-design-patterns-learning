/**
 * 02 - 条件路由（Conditional Edge）
 *
 * 目标：根据 State 的值动态决定下一个节点
 * 这是 Agent 能"做决策"的基础机制
 *
 * 场景：判断一个数字是正数、负数还是零，走不同的处理路径
 */

import { StateGraph, Annotation, END, START } from "@langchain/langgraph";

const GraphState = Annotation.Root({
  input: Annotation<number>({
    reducer: (_, next) => next,
  }),
  steps: Annotation<string[]>({
    reducer: (prev, next) => [...prev, ...next],
    default: () => [],
  }),
  result: Annotation<string>({
    reducer: (_, next) => next,
    default: () => "",
  }),
});

type State = typeof GraphState.State;

// ── Nodes ──────────────────────────────────────────────────────────────────

function classifyNode(state: State): Partial<State> {
  console.log(`[classify] 输入: ${state.input}`);
  return { steps: [`classify: received ${state.input}`] };
}

function positiveNode(state: State): Partial<State> {
  const msg = `${state.input} 是正数`;
  console.log(`[positive] ${msg}`);
  return { result: msg, steps: ["routed to: positive"] };
}

function negativeNode(state: State): Partial<State> {
  const msg = `${state.input} 是负数`;
  console.log(`[negative] ${msg}`);
  return { result: msg, steps: ["routed to: negative"] };
}

function zeroNode(state: State): Partial<State> {
  const msg = `输入是零`;
  console.log(`[zero] ${msg}`);
  return { result: msg, steps: ["routed to: zero"] };
}

// ── 条件路由函数 ────────────────────────────────────────────────────────────
// 返回值是下一个节点的名称（字符串）
// LangGraph 用这个返回值去查找对应的边

function routeBySign(state: State): "positive" | "negative" | "zero" {
  if (state.input > 0) return "positive";
  if (state.input < 0) return "negative";
  return "zero";
}

// ── 构建图 ─────────────────────────────────────────────────────────────────

const graph = new StateGraph(GraphState)
  .addNode("classify", classifyNode)
  .addNode("positive", positiveNode)
  .addNode("negative", negativeNode)
  .addNode("zero", zeroNode)
  .addEdge(START, "classify")
  // addConditionalEdges：从 classify 出发，由 routeBySign 决定去哪
  .addConditionalEdges("classify", routeBySign, {
    positive: "positive",
    negative: "negative",
    zero: "zero",
  })
  // 三条路径都汇聚到 END
  .addEdge("positive", END)
  .addEdge("negative", END)
  .addEdge("zero", END)
  .compile();

// ── 运行三种情况 ───────────────────────────────────────────────────────────

for (const input of [7, -3, 0]) {
  console.log(`\n${"─".repeat(40)}`);
  const state = await graph.invoke({ input });
  console.log(`结果: ${state.result}`);
  console.log(`步骤: ${state.steps.join(" → ")}`);
}
