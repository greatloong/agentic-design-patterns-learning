/**
 * 01 - Prompt Chaining（提示链）
 *
 * 模式：复杂任务拆成串行步骤，每步输出是下步输入
 *
 * 图结构：
 *   START → outline → draft → polish → END
 *
 * 场景：根据主题生成技术博客
 *   1. outline：生成文章大纲
 *   2. draft：根据大纲写草稿
 *   3. polish：润色草稿输出最终版本
 */

import "dotenv/config";
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// ── State ──────────────────────────────────────────────────────────────────
const GraphState = Annotation.Root({
  topic: Annotation<string>({ reducer: (_, n) => n }),
  outline: Annotation<string>({ reducer: (_, n) => n, default: () => "" }),
  draft: Annotation<string>({ reducer: (_, n) => n, default: () => "" }),
  final: Annotation<string>({ reducer: (_, n) => n, default: () => "" }),
});

type State = typeof GraphState.State;

// ── LLM ───────────────────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
});

// ── 节点 ───────────────────────────────────────────────────────────────────

async function outlineNode(state: State): Promise<Partial<State>> {
  console.log(`\n[outline] 为主题生成大纲: "${state.topic}"`);
  const response = await llm.invoke([
    new SystemMessage("你是一个技术写作专家。只输出大纲，不超过4个要点，每点一行。"),
    new HumanMessage(`为以下主题生成博客大纲：${state.topic}`),
  ]);
  const outline = response.content as string;
  console.log(`[outline] 完成:\n${outline}`);
  return { outline };
}

async function draftNode(state: State): Promise<Partial<State>> {
  console.log(`\n[draft] 根据大纲写草稿...`);
  const response = await llm.invoke([
    new SystemMessage("你是一个技术博客作者。根据大纲写一篇简短草稿，每个要点2-3句话。"),
    new HumanMessage(`主题：${state.topic}\n\n大纲：\n${state.outline}`),
  ]);
  const draft = response.content as string;
  console.log(`[draft] 完成（${draft.length} 字）`);
  return { draft };
}

async function polishNode(state: State): Promise<Partial<State>> {
  console.log(`\n[polish] 润色草稿...`);
  const response = await llm.invoke([
    new SystemMessage("你是一个文字编辑。润色以下草稿，使其更流畅专业，保持简洁。"),
    new HumanMessage(`草稿：\n${state.draft}`),
  ]);
  const final = response.content as string;
  console.log(`[polish] 完成（${final.length} 字）`);
  return { final };
}

// ── 构建图 ─────────────────────────────────────────────────────────────────
const graph = new StateGraph(GraphState)
  .addNode("outlineNode", outlineNode)
  .addNode("draftNode", draftNode)
  .addNode("polishNode", polishNode)
  .addEdge(START, "outlineNode")
  .addEdge("outlineNode", "draftNode")
  .addEdge("draftNode", "polishNode")
  .addEdge("polishNode", END)
  .compile();

// ── 运行 ───────────────────────────────────────────────────────────────────
console.log("═".repeat(50));
console.log("Prompt Chaining：串行步骤生成技术博客");
console.log("═".repeat(50));

const result = await graph.invoke({ topic: "LangGraph 中的状态管理" });

console.log("\n" + "═".repeat(50));
console.log("最终文章：");
console.log("═".repeat(50));
console.log(result.final);

console.log("\n── 中间产物 ──");
console.log("大纲：\n" + result.outline);
