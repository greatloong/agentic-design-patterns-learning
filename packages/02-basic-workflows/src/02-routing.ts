/**
 * 02 - Routing（路由）
 *
 * 模式：LLM 作为分类器，根据输入意图分发到不同处理路径
 *
 * 图结构：
 *   START → classifier → (条件边) → technical → END
 *                                 → billing   → END
 *                                 → chitchat  → END
 *
 * 场景：客服系统，根据用户问题类型路由到不同处理节点
 */

import "dotenv/config";
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";

// ── State ──────────────────────────────────────────────────────────────────
const GraphState = Annotation.Root({
  userMessage: Annotation<string>({ reducer: (_, n) => n }),
  category: Annotation<string>({ reducer: (_, n) => n, default: () => "" }),
  response: Annotation<string>({ reducer: (_, n) => n, default: () => "" }),
});

type State = typeof GraphState.State;

// ── LLM ───────────────────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
});

// DeepSeek 不支持 response_format，改用 tool calling 实现结构化输出
const classifyTool = {
  name: "classify",
  description: "对用户消息进行分类",
  schema: z.object({
    category: z.enum(["technical", "billing", "chitchat"]).describe("消息类别"),
    reason: z.string().describe("分类原因"),
  }),
};
const classifierLlm = llm.bindTools([
  { name: classifyTool.name, description: classifyTool.description, schema: classifyTool.schema },
]);

// ── 节点 ───────────────────────────────────────────────────────────────────

// 分类节点：LLM 判断用户意图
async function classifierNode(state: State): Promise<Partial<State>> {
  console.log(`\n[classifier] 分析: "${state.userMessage}"`);

  const response = await classifierLlm.invoke([
    new SystemMessage(
      `将用户消息分类为以下类别之一：
      - technical：技术问题（产品使用、bug、功能咨询）
      - billing：账单问题（付款、退款、订阅）
      - chitchat：闲聊（问候、随意对话）
      必须调用 classify 工具返回结果。`
    ),
    new HumanMessage(state.userMessage),
  ]);

  // 从 tool_calls 里取出分类结果
  const toolCall = response.tool_calls?.[0];
  const category = toolCall?.args?.category ?? "chitchat";
  const reason = toolCall?.args?.reason ?? "";
  console.log(`[classifier] 类别: ${category}（原因: ${reason}）`);
  return { category };
}

// 三条处理路径，各自有专门的 system prompt
async function technicalNode(state: State): Promise<Partial<State>> {
  console.log(`[technical] 处理技术问题...`);
  const res = await llm.invoke([
    new SystemMessage("你是技术支持专员，专注解决产品技术问题，回答简洁专业。"),
    new HumanMessage(state.userMessage),
  ]);
  return { response: res.content as string };
}

async function billingNode(state: State): Promise<Partial<State>> {
  console.log(`[billing] 处理账单问题...`);
  const res = await llm.invoke([
    new SystemMessage("你是账单支持专员，处理付款和订阅问题，态度友好耐心。"),
    new HumanMessage(state.userMessage),
  ]);
  return { response: res.content as string };
}

async function chitchatNode(state: State): Promise<Partial<State>> {
  console.log(`[chitchat] 处理闲聊...`);
  const res = await llm.invoke([
    new SystemMessage("你是一个友好的助手，轻松愉快地和用户聊天。"),
    new HumanMessage(state.userMessage),
  ]);
  return { response: res.content as string };
}

// ── 路由函数 ───────────────────────────────────────────────────────────────
function routeByCategory(state: State): "technical" | "billing" | "chitchat" {
  return state.category as "technical" | "billing" | "chitchat";
}

// ── 构建图 ─────────────────────────────────────────────────────────────────
const graph = new StateGraph(GraphState)
  .addNode("classifier", classifierNode)
  .addNode("technical", technicalNode)
  .addNode("billing", billingNode)
  .addNode("chitchat", chitchatNode)
  .addEdge(START, "classifier")
  .addConditionalEdges("classifier", routeByCategory)
  .addEdge("technical", END)
  .addEdge("billing", END)
  .addEdge("chitchat", END)
  .compile();

// ── 运行三种场景 ───────────────────────────────────────────────────────────
const testCases = [
  "我的 API 调用一直返回 429 错误，怎么解决？",
  "我上个月被多扣了一次费用，能退款吗？",
  "你好，今天天气真不错！",
];

console.log("═".repeat(50));
console.log("Routing：LLM 分类器驱动的客服路由");
console.log("═".repeat(50));

for (const msg of testCases) {
  console.log(`\n${"─".repeat(40)}`);
  const result = await graph.invoke({ userMessage: msg });
  console.log(`\n回复: ${result.response}`);
}
