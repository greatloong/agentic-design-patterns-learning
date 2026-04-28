/**
 * 02 - Routing（路由）
 *
 * ── 是什么 ──────────────────────────────────────────────────────────────────
 * 用 LLM 作为分类器，根据输入的语义将任务分发到不同的处理路径。
 * 每条路径有自己专属的处理节点和 prompt，只关注自己的场景。
 *
 * 图结构：
 *   START → classifier → (条件边) → techSupport  → END
 *                                 → refundHandler → END
 *                                 → generalChat   → END
 *
 * ── 解决什么问题 ─────────────────────────────────────────────────────────────
 * 用一个通用节点处理所有输入时，prompt 臃肿、效果差。
 * Routing 让每条路径只做一件事，专人做专事。
 *
 * ── withStructuredOutput：强制 LLM 返回结构化 JSON ──────────────────────────
 * LangChain 把 Zod schema 转成 OpenAI function calling 格式，强制 LLM 只能
 * 返回符合 schema 的 JSON，不会有多余文字，不会解析失败。
 * 这是生产中分类器节点的标准写法。
 *
 * ── 与其他模式的对比 ─────────────────────────────────────────────────────────
 * vs 条件路由（阶段零 0.2）：
 *   - 阶段零：硬编码逻辑做路由（判断数字、布尔值）
 *   - Routing Pattern：LLM 做分类器，理解自然语言语义
 *
 * vs Prompt Chaining：
 *   - Prompt Chaining：所有输入走同一条线性路径
 *   - Routing：不同输入走不同分叉路径
 *   - 两者可组合：先路由，再在每条路径上做 chaining
 */

import "dotenv/config";
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";

// ── State ──────────────────────────────────────────────────────────────────
const GraphState = Annotation.Root({
  input: Annotation<string>({ reducer: (_, n) => n }),
  // 分类结果：classifier 节点写入，条件边读取
  intent: Annotation<"technical" | "refund" | "general">({
    reducer: (_, n) => n,
    default: () => "general",
  }),
  response: Annotation<string>({ reducer: (_, n) => n, default: () => "" }),
});

type State = typeof GraphState.State;

// ── LLM ───────────────────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
});

// ── 分类器：用 withStructuredOutput 强制返回枚举值 ─────────────────────────
// Zod schema 定义期望的输出结构
const intentSchema = z.object({
  intent: z.enum(["technical", "refund", "general"]).describe(
    "technical: 技术问题/故障/使用帮助；refund: 退款/订单/费用；general: 其他咨询"
  ),
});

// withStructuredOutput 返回的 classifier 每次调用都会返回 { intent: "..." }
// 不会有多余文字，不会解析失败
//
// 注意：DeepSeek 不支持 response_format: json_schema，需要指定 method: "functionCalling"
// functionCalling 方式：把 schema 转成 tool definition，LLM 通过 tool call 返回结构化结果
// 效果完全相同，只是底层传输方式不同
const classifier = llm.withStructuredOutput(intentSchema, { method: "functionCalling" });

// ── 节点 ───────────────────────────────────────────────────────────────────

// 1. classifier 节点：LLM 判断用户意图，写入 intent 字段
async function classifierNode(state: State): Promise<Partial<State>> {
  console.log(`\n[classifier] 分析意图: "${state.input}"`);

  const result = await classifier.invoke([
    new SystemMessage(
      "你是一个客服意图分类器。根据用户消息判断意图类型，只能返回指定的枚举值。"
    ),
    new HumanMessage(state.input),
  ]);

  console.log(`[classifier] 意图识别结果: ${result.intent}`);
  return { intent: result.intent };
}

// 2. 三个处理节点，各有专属 prompt
async function techSupportNode(state: State): Promise<Partial<State>> {
  console.log(`[techSupport] 处理技术问题...`);
  const response = await llm.invoke([
    new SystemMessage("你是一个技术支持专家。用简洁专业的语言解答技术问题，如需要可提供排查步骤。"),
    new HumanMessage(state.input),
  ]);
  return { response: response.content as string };
}

async function refundHandlerNode(state: State): Promise<Partial<State>> {
  console.log(`[refundHandler] 处理退款请求...`);
  const response = await llm.invoke([
    new SystemMessage("你是一个退款处理专员。同理心回应用户，说明退款流程，预计3-5个工作日到账。"),
    new HumanMessage(state.input),
  ]);
  return { response: response.content as string };
}

async function generalChatNode(state: State): Promise<Partial<State>> {
  console.log(`[generalChat] 处理一般咨询...`);
  const response = await llm.invoke([
    new SystemMessage("你是一个友善的客服助手。热情回答用户的一般性问题。"),
    new HumanMessage(state.input),
  ]);
  return { response: response.content as string };
}

// ── 条件边：读取 state.intent，决定路由目标 ────────────────────────────────
function routeByIntent(state: State): "techSupport" | "refundHandler" | "generalChat" {
  // state.intent 由 classifierNode 写入，这里直接读取
  const routes = {
    technical: "techSupport" as const,
    refund: "refundHandler" as const,
    general: "generalChat" as const,
  };
  return routes[state.intent];
}

// ── 构建图 ─────────────────────────────────────────────────────────────────
const graph = new StateGraph(GraphState)
  .addNode("classifier", classifierNode)
  .addNode("techSupport", techSupportNode)
  .addNode("refundHandler", refundHandlerNode)
  .addNode("generalChat", generalChatNode)
  .addEdge(START, "classifier")
  // classifier 之后走条件边，根据 intent 分发
  .addConditionalEdges("classifier", routeByIntent)
  // 三条路径都直接到 END
  .addEdge("techSupport", END)
  .addEdge("refundHandler", END)
  .addEdge("generalChat", END)
  .compile();

// ── 演示：三种不同意图的输入 ───────────────────────────────────────────────
const testCases = [
  "我的 App 一直崩溃，打开就闪退，怎么办？",
  "我上周买的课程想申请退款，订单号是 A12345",
  "你们平台有哪些课程分类？",
];

for (const input of testCases) {
  console.log("\n" + "═".repeat(50));
  console.log(`输入: ${input}`);
  console.log("═".repeat(50));

  const result = await graph.invoke({ input });

  console.log(`意图: ${result.intent}`);
  console.log(`回复:\n${result.response}`);
}
