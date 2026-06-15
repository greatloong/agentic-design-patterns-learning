/**
 * 02 - Tool Use（工具使用）
 *
 * ── 工具设计三原则 ────────────────────────────────────────────────────────────
 * 1. 职责单一：一个工具只做一件事，不要设计"万能工具"
 * 2. 描述精准：description 是 LLM 决定调用哪个工具的唯一依据，描述模糊 = 误调用
 * 3. 参数最小化：只暴露 LLM 需要传入的参数，内部逻辑不要透出
 *
 * ── 错误处理两种策略 ─────────────────────────────────────────────────────────
 * 策略A - 工具内部捕获，返回错误字符串（推荐）：
 *   LLM 读到错误信息后自己决策（重试、换工具、告知用户）
 *   适合：生产环境，Agent 需要自我恢复
 *   优势：跨 LLM provider 稳定（有些 provider 对 ToolMessage 的 content 格式敏感）
 *
 * 策略B - 直接抛出异常（依赖 ToolNode 捕获）：
 *   ToolNode 默认会捕获异常并包成 ToolMessage，但不同 provider 表现不一致
 *   Anthropic 要求 tool_result content 非空、格式严格，抛异常时有可能产生不合规的消息
 *   因此：统一用策略A更安全，除非你明确希望让程序崩溃（开发/测试阶段）
 *
 * ── 工具链（Tool Chaining）────────────────────────────────────────────────────
 * LLM 会自动规划多步工具调用：先查用户 → 再查订单 → 最后申请退款
 * 后一步的输入依赖前一步的输出，无需开发者显式编排
 *
 * ── 如何强制 LLM 走完必要步骤（三种做法）────────────────────────────────────
 * 问题：LLM 可能认为某一步"不必要"而跳过（比如用户已说明身份就不调 get_user）
 *
 * 方法1 - prompt 约束：system prompt 里明确强制步骤（最轻量但不可靠）
 * 方法2 - schema 约束：用必填参数形成依赖（见 apply_refund 的 contactEmail）
 *         LLM 必须先调 get_user 才能填出 contactEmail，自然形成工具链
 * 方法3 - 图结构约束：拆成多个节点按顺序执行（最强，见 Ch.6 Planning）
 *
 * ── 场景：客服退款 Agent ─────────────────────────────────────────────────────
 * 用户请求退款 → Agent 依次调用：查用户 → 查订单 → 申请退款
 * 退款工具有概率失败，演示 LLM 如何处理工具错误并自我恢复
 */

import "dotenv/config";
import { StateGraph, START, END, MessagesAnnotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// ── Mock 数据库 ────────────────────────────────────────────────────────────
const users: Record<string, { name: string; email: string }> = {
  U001: { name: "张三", email: "zhangsan@example.com" },
  U002: { name: "李四", email: "lisi@example.com" },
};

const orders: Record<string, { userId: string; amount: number; status: string; product: string }> = {
  "ORD-001": { userId: "U001", amount: 299, status: "delivered", product: "TypeScript 进阶课程" },
  "ORD-002": { userId: "U002", amount: 599, status: "processing", product: "LangGraph 实战课" },
};

// ── 工具定义 ───────────────────────────────────────────────────────────────

// 工具1：查询用户信息
// description 精准描述：何时用、参数含义、返回什么
const getUserTool = tool(
  ({ userId }: { userId: string }) => {
    const user = users[userId];
    if (!user) {
      // 策略A：返回错误字符串，让 LLM 决策
      return `错误：用户 ${userId} 不存在`;
    }
    console.log(`[tool:getUser] 查询用户 ${userId}`);
    return JSON.stringify({ userId, ...user });
  },
  {
    name: "get_user",
    description:
      "根据用户ID查询用户基本信息。成功时返回 JSON（含 userId、name、email），" +
      "失败时返回错误信息字符串。当需要确认用户身份时调用。",
    schema: z.object({
      userId: z.string().describe("用户ID，格式为 U 开头加3位数字，例如 U001"),
    }),
  }
);

// 工具2：查询订单信息
const getOrderTool = tool(
  ({ orderId }: { orderId: string }) => {
    const order = orders[orderId];
    if (!order) {
      return `错误：订单 ${orderId} 不存在`;
    }
    console.log(`[tool:getOrder] 查询订单 ${orderId}`);
    return JSON.stringify({ orderId, ...order });
  },
  {
    name: "get_order",
    description:
      "根据订单ID查询订单详情。成功时返回 JSON（含 orderId、userId、product、amount、status），" +
      "失败时返回错误信息字符串。处理退款前必须先查询订单确认信息。",
    schema: z.object({
      orderId: z.string().describe("订单ID，格式为 ORD- 开头，例如 ORD-001"),
    }),
  }
);

// 工具3：申请退款
// 关键设计：通过 schema 强制要求 contactEmail 参数
// LLM 要填这个字段，必须先调 get_user 拿到邮箱 → 形成强制的工具链
let refundAttempts = 0;
const refundTool = tool(
  ({ orderId, reason, contactEmail }: { orderId: string; reason: string; contactEmail: string }) => {
    refundAttempts++;
    const order = orders[orderId];

    if (!order) {
      return `退款失败：订单 ${orderId} 不存在`;
    }

    // 模拟第一次调用失败（演示 LLM 如何处理错误并重试）
    if (refundAttempts === 1) {
      console.log(`[tool:refund] 第 ${refundAttempts} 次尝试 → 模拟网络超时`);
      return "退款失败：退款服务暂时不可用（网络超时），建议稍后重试。";
    }

    console.log(`[tool:refund] 第 ${refundAttempts} 次尝试 → 退款成功，通知邮箱 ${contactEmail}`);
    return JSON.stringify({
      success: true,
      orderId,
      amount: order.amount,
      reason,
      contactEmail,
      refundId: `REF-${Date.now()}`,
      message: `退款 ¥${order.amount} 已受理，3-5 个工作日到账，结果将发送至 ${contactEmail}`,
    });
  },
  {
    name: "apply_refund",
    description:
      "为指定订单申请退款。成功时返回 JSON（含 success、refundId、amount、contactEmail、message），" +
      "失败时返回以'退款失败'开头的错误字符串，此时可稍后重试一次。" +
      "仅在用户明确要求退款时调用。",
    schema: z.object({
      orderId: z.string().describe("要退款的订单ID"),
      reason: z.string().describe("退款原因，用用户的原话描述"),
      // 通过必填参数强制形成工具链：LLM 必须先调 get_user 才能拿到邮箱
      contactEmail: z
        .string()
        .email()
        .describe("用户邮箱（必须通过 get_user 工具查询获得，不可由用户输入或自行生成）"),
    }),
  }
);

const tools = [getUserTool, getOrderTool, refundTool];

// ── LLM ───────────────────────────────────────────────────────────────────
// 注意：这里用 DeepSeek 而不是 Claude。
// 原因：通过 pumpkinai.vip 中转调用 Claude 时，多轮 tool_use 场景下
// 中转服务会把空 text 块传给 Anthropic，触发 "text: Field required" 错误。
// DeepSeek 用原生 OpenAI 协议，没有这个问题。
const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
}).bindTools(tools);

// ── 图结构（标准 ReAct 循环）─────────────────────────────────────────────
async function agentNode(state: typeof MessagesAnnotation.State) {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
}

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", agentNode)
  // ToolNode 默认捕获工具抛出的异常，包成 ToolMessage 传回 LLM
  // 这就是策略B能工作的原因：不需要在工具里 try/catch
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "agent")
  .addConditionalEdges("agent", (state) => {
    const last = state.messages.at(-1) as AIMessage;
    return last.tool_calls?.length ? "tools" : END;
  })
  .addEdge("tools", "agent")
  .compile();

// ── 运行 ───────────────────────────────────────────────────────────────────
console.log("═".repeat(50));
console.log("Tool Use：工具链 + 错误处理");
console.log("═".repeat(50));

const result = await graph.invoke({
  messages: [
    new SystemMessage(
      "你是一个客服助手，帮助用户处理订单和退款问题。" +
      "处理退款时，先查询订单确认信息，再申请退款。遇到工具错误时自动重试一次。"
    ),
    new HumanMessage(
      "你好，我是用户 U001，我的订单 ORD-001 想申请退款，原因是课程内容与描述不符。"
    ),
  ],
});

console.log("\n" + "═".repeat(50));
console.log("最终回复：");
console.log("═".repeat(50));
console.log(result.messages.at(-1)?.content);

console.log("\n── 完整工具调用链 ──");
for (const msg of result.messages) {
  const type = msg._getType();
  if (type === "ai") {
    const ai = msg as AIMessage;
    if (ai.tool_calls?.length) {
      for (const tc of ai.tool_calls) {
        console.log(`[AI → tool] ${tc.name}(${JSON.stringify(tc.args)})`);
      }
    }
  } else if (type === "tool") {
    const content = typeof msg.content === "string"
      ? msg.content.slice(0, 80) + (msg.content.length > 80 ? "..." : "")
      : "(complex)";
    console.log(`[tool → AI] ${content}`);
  }
}
