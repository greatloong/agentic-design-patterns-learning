/**
 * 03 - ReAct Agent
 *
 * 模式：Reasoning + Acting 循环
 *
 * 图结构：
 *   START → agent → (条件边) → tools → agent → ... → END
 *
 * 完整执行流程：
 *
 * [Step 1] bindTools(tools)
 *   把工具的 name / description / schema 序列化成 OpenAI tools 格式。
 *   之后每次调用该 llm，HTTP 请求体都会自动携带这份工具描述，
 *   LLM 才知道"我有哪些工具可以调用"。
 *
 * [Step 2] agentNode 第一次执行
 *   llm.invoke(state.messages) 发出 HTTP 请求，携带：
 *     - messages: [用户问题]
 *     - tools:    [get_weather schema, calculator schema]
 *   LLM 决定调用工具，响应体不含自然语言，而是：
 *     { role: "assistant", content: "", tool_calls: [
 *         { id: "call_abc", function: { name: "get_weather", arguments: '{"city":"北京"}' } },
 *         { id: "call_def", function: { name: "calculator",  arguments: '{"expression":"123*456"}' } }
 *     ]}
 *   LangChain 将其包装成 AIMessage，tool_calls 已解析为 JS 对象数组，追加进 State。
 *
 * [Step 3] shouldContinue 条件边
 *   检查最后一条 AIMessage 是否有 tool_calls：
 *     有 → 路由到 tools 节点
 *     无 → 路由到 END
 *
 * [Step 4] ToolNode 执行工具
 *   遍历 tool_calls，找到对应函数并调用：
 *     get_weather({ city: "北京" })  → "晴，25°C，东南风 3 级"
 *     calculator({ expression: "123 * 456" }) → "56088"
 *   把每个结果包成 ToolMessage（携带对应的 tool_call_id），追加进 State。
 *
 * [Step 5] agentNode 第二次执行
 *   此时 state.messages 包含：用户消息 + AIMessage(tool_calls) + 2 个 ToolMessage。
 *   LLM 看到完整上下文，理解工具已执行完毕，生成最终自然语言回答，无 tool_calls。
 *
 * [Step 6] shouldContinue 再次检查
 *   最后一条 AIMessage 无 tool_calls → 路由到 END，图结束。
 *
 * 核心驱动力：state.messages 是不断增长的历史列表，LLM 每次都能"看到"完整上下文。
 */

import "dotenv/config";
import { StateGraph, Annotation, START, END, MessagesAnnotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage } from "@langchain/core/messages";
import { z } from "zod";

// ── 1. 定义工具 ────────────────────────────────────────────────────────────
// tool() 是 LangChain core 提供的工具定义函数
// 注意：这里我们用的是底层原语，不是 LangChain 的 Agent 封装

const calculatorTool = tool(
  ({ expression }: { expression: string }) => {
    // 简单计算器，只处理基本四则运算
    const result = Function(`"use strict"; return (${expression})`)();
    console.log(`[calculator] ${expression} = ${result}`);
    return String(result);
  },
  {
    name: "calculator",
    description: "计算数学表达式，例如 '2 + 3 * 4'",
    schema: z.object({
      expression: z.string().describe("要计算的数学表达式"),
    }),
  }
);

const weatherTool = tool(
  ({ city }: { city: string }) => {
    // Mock 数据，演示用
    const mockData: Record<string, string> = {
      北京: "晴，25°C，东南风 3 级",
      上海: "多云，22°C，东风 2 级",
      深圳: "小雨，28°C，南风 4 级",
    };
    const result = mockData[city] ?? "未知城市，无法查询";
    console.log(`[weather] ${city}: ${result}`);
    return result;
  },
  {
    name: "get_weather",
    description: "查询城市的实时天气",
    schema: z.object({
      city: z.string().describe("城市名称，例如：北京、上海"),
    }),
  }
);

const tools = [calculatorTool, weatherTool];

// ── 2. 初始化 LLM ──────────────────────────────────────────────────────────
// DeepSeek 兼容 OpenAI 协议，配置 baseURL 即可
const llm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: {
    baseURL: "https://api.deepseek.com/v1",
  },
}).bindTools(tools); // 把工具绑定给 LLM，LLM 才知道有哪些工具可用

// ── 3. 定义 State ──────────────────────────────────────────────────────────
// MessagesAnnotation 是 LangGraph 内置的 messages state
// 它的 reducer 是追加：每次节点返回新消息，都会追加到历史里
// 等价于：
//   messages: Annotation<BaseMessage[]>({
//     reducer: (prev, next) => prev.concat(next),
//     default: () => [],
//   })
const GraphState = MessagesAnnotation;
type State = typeof GraphState.State;

// ── 4. 定义节点 ────────────────────────────────────────────────────────────

// agent 节点：调用 LLM，返回 LLM 的响应（可能包含 tool call）
async function agentNode(state: State): Promise<Partial<State>> {
  console.log(`\n[agent] 调用 LLM，当前消息数: ${state.messages.length}`);
  const response = await llm.invoke(state.messages);
  console.log(`[agent] LLM 响应: ${response.content || "(tool call)"}`);
  // 返回新消息，MessagesAnnotation 的 reducer 会自动追加
  return { messages: [response] };
}

// ToolNode 是 LangGraph 提供的内置节点
// 它会自动解析最后一条 AIMessage 里的 tool_calls，执行对应工具，返回 ToolMessage
const toolsNode = new ToolNode(tools);

// ── 5. 条件路由 ────────────────────────────────────────────────────────────
// 检查 LLM 的响应里是否包含 tool call
// 有 → 去执行工具；没有 → 结束

function shouldContinue(state: State): "tools" | typeof END {
  const lastMessage = state.messages[state.messages.length - 1] as AIMessage;
  const hasToolCall = lastMessage.tool_calls && lastMessage.tool_calls.length > 0;
  console.log(`[router] 有 tool call: ${hasToolCall}`);
  return hasToolCall ? "tools" : END;
}

// ── 6. 构建图 ──────────────────────────────────────────────────────────────

const graph = new StateGraph(GraphState)
  .addNode("agent", agentNode)
  .addNode("tools", toolsNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue)
  // 工具执行完，无条件回到 agent 继续思考
  .addEdge("tools", "agent")
  .compile();

// ── 7. 运行 ────────────────────────────────────────────────────────────────

import { HumanMessage } from "@langchain/core/messages";

console.log("═".repeat(50));
console.log("问题：北京今天天气怎么样？另外 123 * 456 等于多少？");
console.log("═".repeat(50));

const result = await graph.invoke({
  messages: [
    new HumanMessage("北京今天天气怎么样？另外 123 * 456 等于多少？"),
  ],
});

console.log("\n── 最终回答 ──");
console.log(result.messages[result.messages.length - 1].content);

console.log("\n── 完整消息历史 ──");
for (const msg of result.messages) {
  const role = msg._getType();
  const content = msg.content || "(tool call)";
  console.log(`[${role}] ${content}`);
}
