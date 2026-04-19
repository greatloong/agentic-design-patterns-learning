/**
 * 01 - LangGraph 核心原语
 *
 * 目标：理解 StateGraph 的三个核心概念
 *   - State：图中流转的数据结构
 *   - Node：处理 State 的函数
 *   - Edge：控制执行流向
 *
 * 这个例子不调用 LLM，用纯逻辑演示图的执行机制
 */

import { StateGraph, Annotation, END, START } from "@langchain/langgraph";

// ── 1. 定义 State ──────────────────────────────────────────────────────────
// Annotation 是 LangGraph 定义 state schema 的方式
// 每个字段需要指定 reducer：决定如何合并新旧值
const GraphState = Annotation.Root({
  // 输入的数字
  input: Annotation<number>({
    reducer: (_, next) => next, // 直接替换
  }),
  // 处理步骤的日志
  steps: Annotation<string[]>({
    reducer: (prev, next) => [...prev, ...next], // 追加
    default: () => [],
  }),
  // 最终结果
  result: Annotation<number>({
    reducer: (_, next) => next,
    default: () => 0,
  }),
});

// 类型别名，方便后续使用
type State = typeof GraphState.State;

// ── 2. 定义 Node ───────────────────────────────────────────────────────────
// Node 是普通函数：接收当前 State，返回 State 的部分更新（Partial<State>）

function doubleNode(state: State): Partial<State> {
  const doubled = state.input * 2;
  console.log(`[doubleNode] ${state.input} × 2 = ${doubled}`);
  return {
    result: doubled,
    steps: [`doubled: ${state.input} → ${doubled}`],
  };
}

function addTenNode(state: State): Partial<State> {
  const added = state.result + 10;
  console.log(`[addTenNode] ${state.result} + 10 = ${added}`);
  return {
    result: added,
    steps: [`addTen: ${state.result} → ${added}`],
  };
}

function logNode(state: State): Partial<State> {
  console.log(`[logNode] 最终结果: ${state.result}`);
  console.log(`[logNode] 执行步骤: ${state.steps.join(" → ")}`);
  return {};
}

// ── 3. 构建图 ──────────────────────────────────────────────────────────────

const graph = new StateGraph(GraphState)
  // 注册节点
  .addNode("double", doubleNode)
  .addNode("addTen", addTenNode)
  .addNode("log", logNode)
  // 定义边：START → double → addTen → log → END
  .addEdge(START, "double")
  .addEdge("double", "addTen")
  .addEdge("addTen", "log")
  .addEdge("log", END)
  .compile();

// ── 4. 运行图 ──────────────────────────────────────────────────────────────

const result = await graph.invoke({ input: 5 });

console.log("\n── 最终 State ──");
console.log(result);
