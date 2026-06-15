/**
 * 06 - Streaming（流式输出）
 *
 * 核心概念：
 *   - graph.stream()：替代 invoke()，返回异步迭代器
 *   - streamMode "values"：每个节点后输出完整 State
 *   - streamMode "updates"：每个节点后只输出变更部分
 *   - streamMode "messages"：LLM 逐 token 流出
 *
 * 图结构：复用 ReAct Agent（agent ↔ tools 循环）
 */

import "dotenv/config";
import { StateGraph, START, END, MessagesAnnotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { z } from "zod";

// ── 工具 ───────────────────────────────────────────────────────────────────
const weatherTool = tool(
  ({ city }: { city: string }) => {
    const mock: Record<string, string> = {
      北京: "晴，25°C",
      上海: "多云，22°C",
    };
    return mock[city] ?? "未知城市";
  },
  {
    name: "get_weather",
    description: "查询城市天气",
    schema: z.object({ city: z.string() }),
  }
);

const tools = [weatherTool];

// ── LLM + 图 ───────────────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
}).bindTools(tools);

async function agentNode(state: typeof MessagesAnnotation.State) {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
}

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", agentNode)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "agent")
  .addConditionalEdges("agent", (state) => {
    const last = state.messages.at(-1) as AIMessage;
    return last.tool_calls?.length ? "tools" : END;
  })
  .addEdge("tools", "agent")
  .compile();

const input = { messages: [new HumanMessage("北京今天天气怎么样？")] };

// ── 演示 1：streamMode "updates" ───────────────────────────────────────────
console.log("═".repeat(50));
console.log('演示 1：streamMode "updates"（只看每个节点的变更）');
console.log("═".repeat(50));

for await (const chunk of await graph.stream(input, { streamMode: "updates" })) {
  // chunk 结构：{ 节点名: 该节点返回的 Partial<State> }
  const [nodeName, update] = Object.entries(chunk)[0];
  const msgs = (update as any).messages ?? [];
  const lastMsg = msgs.at(-1);
  console.log(`\n[${nodeName}] 新增消息: ${lastMsg?.content || "(tool call)"}`);
}

// ── 演示 2：streamMode "values" ────────────────────────────────────────────
console.log("\n" + "═".repeat(50));
console.log('演示 2：streamMode "values"（每步后的完整 State）');
console.log("═".repeat(50));

for await (const state of await graph.stream(input, { streamMode: "values" })) {
  // state 是当前完整的 State
  console.log(`\n当前消息数: ${state.messages.length}`);
  console.log(`最新消息: [${state.messages.at(-1)?._getType()}] ${state.messages.at(-1)?.content || "(tool call)"}`);
}

// ── 演示 3：streamMode "messages"（LLM 逐 token 流出）─────────────────────
console.log("\n" + "═".repeat(50));
console.log('演示 3：streamMode "messages"（逐 token 流出）');
console.log("═".repeat(50));
console.log("LLM 输出：");

for await (const [msg, metadata] of await graph.stream(input, { streamMode: "messages" })) {
  // msg 是 token 片段，metadata 包含来源节点
  if (metadata.langgraph_node === "agent" && msg.content) {
    process.stdout.write(msg.content as string);
  }
}
console.log(); // 换行
