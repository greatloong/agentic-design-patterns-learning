/**
 * 04 - Checkpointer（状态持久化）
 *
 * 核心概念：
 *   - Checkpointer：在每个节点执行后自动保存 State 快照
 *   - thread_id：会话标识，同一 thread_id 的调用共享 State
 *   - config：调用时传入 { configurable: { thread_id } } 指定会话
 *
 * 演示：
 *   1. 同一 thread 的多轮对话（有记忆）
 *   2. 不同 thread 的对话互相隔离
 *   3. 查看 checkpoint 快照内容
 */

import "dotenv/config";
import { StateGraph, START, END, MessagesAnnotation, MemorySaver } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";

// ── LLM ───────────────────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
});

// ── 节点 ───────────────────────────────────────────────────────────────────
async function agentNode(state: typeof MessagesAnnotation.State) {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
}

// ── 构建图（关键：compile 时传入 checkpointer）─────────────────────────────
const checkpointer = new MemorySaver();

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", agentNode)
  .addEdge(START, "agent")
  .addEdge("agent", END)
  .compile({ checkpointer }); // ← 这里传入 checkpointer

// ── 工具函数 ───────────────────────────────────────────────────────────────
async function chat(threadId: string, userMessage: string) {
  console.log(`\n[用户 @${threadId}] ${userMessage}`);

  const result = await graph.invoke(
    { messages: [new HumanMessage(userMessage)] },
    // config 指定 thread_id，LangGraph 用它查找对应的 checkpoint
    { configurable: { thread_id: threadId } }
  );

  const reply = result.messages[result.messages.length - 1].content;
  console.log(`[AI   @${threadId}] ${reply}`);
  return reply;
}

// ── 演示 1：同一 thread 的多轮对话 ────────────────────────────────────────
console.log("═".repeat(50));
console.log("演示 1：多轮对话（同一 thread_id）");
console.log("═".repeat(50));

await chat("thread-alice", "你好，我叫 Alice，我是一名前端工程师");
await chat("thread-alice", "我最近在学习 AI Agent 开发");
await chat("thread-alice", "你还记得我叫什么名字，是做什么的吗？");

// ── 演示 2：不同 thread 互相隔离 ──────────────────────────────────────────
console.log("\n" + "═".repeat(50));
console.log("演示 2：不同 thread 互相隔离");
console.log("═".repeat(50));

await chat("thread-bob", "你好，我叫 Bob");
// Bob 的 thread 里没有 Alice 的信息
await chat("thread-bob", "你知道 Alice 是谁吗？");

// ── 演示 3：查看 checkpoint 内容 ──────────────────────────────────────────
console.log("\n" + "═".repeat(50));
console.log("演示 3：查看 checkpoint 快照");
console.log("═".repeat(50));

const checkpoint = await checkpointer.get({
  configurable: { thread_id: "thread-alice" },
});

// v1 里 channel_values 是 Record<string, unknown>，messages 需显式断言为消息数组
const snapshotMessages = (checkpoint?.channel_values?.messages ?? []) as Array<{
  _getType: () => string;
}>;
console.log(`thread-alice 的消息数: ${snapshotMessages.length}`);
console.log("消息角色序列:", snapshotMessages.map((m) => m._getType()).join(" → "));
