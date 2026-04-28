/**
 * 01 - Prompt Chaining（提示链）
 *
 * ── 是什么 ──────────────────────────────────────────────────────────────────
 * 把一个复杂任务拆成多个串行 LLM 调用，每步的输出是下步的输入。
 * 每个节点有自己专属的 system prompt，只做一件事。
 *
 * 图结构：
 *   START → outline → draft → polish → END
 *
 * ── 解决什么问题 ─────────────────────────────────────────────────────────────
 * 让 LLM 一次性完成多件事时质量下降（"大锅炖"问题）。
 * 拆开后：每步更专注、可独立调试、失败只需重跑出错的那步。
 *
 * ── 适用场景 ─────────────────────────────────────────────────────────────────
 * 任务的步骤和顺序可以提前确定时使用。
 * 如果步骤需要 LLM 在运行时动态决定，应改用 ReAct Agent。
 *
 * ── State 设计原则 ───────────────────────────────────────────────────────────
 * 每步结果存独立字段（topic / outline / draft / final），而不是共享 messages 列表。
 * 好处：职责清晰，每个节点只读取和写入自己负责的字段。
 * 代价：后续节点看不到前序节点的完整推理过程，只能看到输出结果（省 token，但信息有损）。
 *
 * ── 与其他模式的对比 ─────────────────────────────────────────────────────────
 * vs ReAct Agent：
 *   - Prompt Chaining：开发者掌控流程，步骤固定，每次只传当前步骤需要的信息
 *   - ReAct Agent：LLM 掌控流程，步骤动态，每次传入完整 messages 历史
 *
 * vs Human-in-the-loop：
 *   - 结构相同（线性固定步骤）
 *   - HITL 在步骤间插入 interrupt() 等待人工输入，Prompt Chaining 全自动流水线
 *   - 可以理解为：HITL = Prompt Chaining + 人工确认节点
 *
 * ── 场景：根据主题生成技术博客 ───────────────────────────────────────────────
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
