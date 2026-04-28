/**
 * 02 - Swarm Mesh（去中心化协作，Mesh 全互联拓扑）
 *
 * ── 是什么 ──────────────────────────────────────────────────────────────────
 * 没有主管，每个 Agent 平等。Agent 自己判断"我能不能搞定？"——
 *   - 能：直接回用户，结束
 *   - 不能：调用 transfer_to_X 工具把控制权交给另一个 Agent
 *
 * 拓扑（Mesh = 全互联）：
 *
 *        frontdesk
 *        ↕      ↕
 *   technical ↔ billing
 *
 * frontdesk 是入口（接首条用户消息）；三个 Agent 之间可以两两 handoff。
 *
 * ── Handoff 是怎么做的 ─────────────────────────────────────────────────────
 * 1. 把"转给某 Agent"包装成一个**handoff tool**：
 *      transferToTechnical = tool(..., { name: "transfer_to_technical", ... })
 *    Agent 在 LLM 调用里通过 tool_call 调用它，相当于宣告"我要转手"。
 *
 * 2. handoff tool 的 handler 不真的"执行"什么逻辑，
 *    它返回一个**特殊 ToolMessage**（content 写明转手原因），
 *    控制权转移由父图条件边读取最后一条 AIMessage 的 tool_calls 来决定。
 *
 * 3. 父图节点路由：
 *      frontdesk 节点 → 条件边读最后一条 AIMessage：
 *        - 若 tool_calls 里有 transfer_to_X → 路由到对应 Agent 节点
 *        - 否则 → END（直接回用户）
 *
 * ── 兜底回退是什么 ─────────────────────────────────────────────────────────
 * "兜底"指的是 Agent 发现自己接错了任务时，能转回到合适的 Agent。
 * 在 Mesh 里有两种走法：
 *   - 直接转给真正合适的同行（technical → billing）
 *   - 退回 frontdesk 让它重新分诊（technical → frontdesk）
 *
 * 本例两种都允许。代码里通过给 technical / billing 同时绑定
 * "transfer_to_frontdesk" 和 "transfer_to_<peer>" 两个工具实现。
 *
 * ── 防循环 ────────────────────────────────────────────────────────────────
 * 1. handoffHistory: State 里维护交接路径，检测 [X→Y→X] 立即 END
 * 2. recursion_limit: LangGraph 默认 25，硬性兜底
 * 3. handoff schema 强制带 reason 字段，让 LLM 多想一步再转
 *
 * ── State 设计（生产偏好版）────────────────────────────────────────────────
 *   messages:        共享的对话流，但子 Agent 内部 tool 细节不写回（隔离）
 *   activeAgent:     当前接管的 Agent 名（路由依据）
 *   handoffHistory:  交接路径，用于循环检测和可观测
 *
 * ── 与 Supervisor 的关键区别 ─────────────────────────────────────────────
 *   Supervisor: 决策(supervisor LLM) + 执行(子 Agent LLM) 是两次 LLM 调用
 *   Swarm:      决策融在 Agent 自己的 LLM 调用里（同一次调用既能 handoff
 *               也能直接回用户），简单问题省一次 LLM
 */

import "dotenv/config";
import {
  StateGraph,
  START,
  END,
  Annotation,
  MemorySaver,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  BaseMessage,
} from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// ── LLM ───────────────────────────────────────────────────────────────────
// 全部用 DeepSeek（直连，OpenAI 协议，多轮 tool 兼容性好）
const llm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
});

// ── State ─────────────────────────────────────────────────────────────────
type AgentName = "frontdesk" | "technical" | "billing";

const SwarmState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
  activeAgent: Annotation<AgentName>({
    reducer: (_p, n) => n,
    default: () => "frontdesk", // 入口默认 frontdesk
  }),
  handoffHistory: Annotation<AgentName[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
});

// ── Handoff Tools ─────────────────────────────────────────────────────────
// 工厂函数：为目标 Agent 生成对应的 handoff 工具
// handler 不执行实际逻辑，只返回一条说明性 ToolMessage
// 真正的"控制权转移"发生在父图条件边读取这条 tool_call 之后
function makeHandoffTool(targetAgent: AgentName, description: string) {
  return tool(
    async ({ reason }: { reason: string }) => {
      // 返回值会被包成 ToolMessage 追加到 messages
      // 内容只用于人类调试和供下一个 Agent 看到上下文，不做控制流决策
      return `[handoff] 已转交给 ${targetAgent}。原因：${reason}`;
    },
    {
      name: `transfer_to_${targetAgent}`,
      description,
      schema: z.object({
        reason: z
          .string()
          .describe(
            "为什么要把这个任务转给该 Agent？必须明确具体（防止冲动转手）"
          ),
      }),
    }
  );
}

// 三个 Agent 各自能转给的对方（Mesh 拓扑）
const transferToFrontdesk = makeHandoffTool(
  "frontdesk",
  "把任务转回前台分诊。当用户问题超出你的领域，或者意图不清时调用。"
);
const transferToTechnical = makeHandoffTool(
  "technical",
  "把任务转给技术支持。当用户问题涉及代码、bug、API 配置、技术错误时调用。"
);
const transferToBilling = makeHandoffTool(
  "billing",
  "把任务转给售后/账务。当用户问题涉及订单、退款、物流、发票时调用。"
);

// ── 三个 Agent 的定义 ─────────────────────────────────────────────────────
// 每个 Agent 是一个简单的 LLM 节点（带 handoff 工具，不做内部 ReAct 多轮）
// 简化是为了让 Swarm 范式更突出；生产里每个 Agent 内部可以是子图

const FRONTDESK_SYSTEM = `你是【前台分诊】客服。职责：
- 接待用户并理解他们的问题
- 简单问题（如营业时间、官网网址、产品介绍）你直接回答，不要 handoff
- 涉及技术问题（代码、bug、配置）→ 调用 transfer_to_technical
- 涉及订单/退款/物流 → 调用 transfer_to_billing
- 调用 handoff 时必须说明清晰原因
- 不要重复 handoff 给已经 handoff 过的 Agent`;

const TECHNICAL_SYSTEM = `你是【技术支持】专员。职责：
- 处理代码、bug、API 配置类问题
- 遇到订单/物流/退款类问题 → 调用 transfer_to_billing 转给同行
- 用户意图模糊不清 → 调用 transfer_to_frontdesk 退回分诊
- 能处理就直接答完，不要冗余 handoff`;

const BILLING_SYSTEM = `你是【售后】专员。职责：
- 处理订单、退款、物流、发票类问题
- 遇到代码/技术错误 → 调用 transfer_to_technical 转给同行
- 用户意图模糊 → 调用 transfer_to_frontdesk
- 能处理就直接答完`;

const frontdeskLlm = llm.bindTools([transferToTechnical, transferToBilling]);
const technicalLlm = llm.bindTools([transferToBilling, transferToFrontdesk]);
const billingLlm = llm.bindTools([transferToTechnical, transferToFrontdesk]);

// ── Agent 节点工厂 ───────────────────────────────────────────────────────
// 每个节点：调一次 LLM；如果 LLM 调了 handoff tool，则附带写入 handoffHistory
function makeAgentNode(
  name: AgentName,
  bindedLlm: typeof frontdeskLlm,
  systemPrompt: string
) {
  return async (
    state: typeof SwarmState.State
  ): Promise<Partial<typeof SwarmState.State>> => {
    console.log(
      `\n[${name}] 接管，messages=${state.messages.length}，路径=[${state.handoffHistory.join(
        " → "
      )}]`
    );

    const response = (await bindedLlm.invoke([
      new SystemMessage(systemPrompt),
      ...state.messages,
    ])) as AIMessage;

    // 给 AIMessage 标注是哪个 Agent 说的（方便观察）
    response.name = name;

    // 解析 LLM 是否调用了 handoff tool
    const handoffCall = response.tool_calls?.find((tc) =>
      tc.name?.startsWith("transfer_to_")
    );

    if (handoffCall) {
      const target = handoffCall.name!.replace(
        "transfer_to_",
        ""
      ) as AgentName;
      console.log(
        `[${name}] → 决定 handoff 给 ${target}（${handoffCall.args.reason}）`
      );

      // 关键：必须为 LLM 发出的每个 tool_call 配一条 ToolMessage，
      // 否则下一个 Agent 拿到 messages 时会因 tool_call 没收到回复而报错
      const { ToolMessage } = await import("@langchain/core/messages");
      const toolReply = new ToolMessage({
        tool_call_id: handoffCall.id!,
        content: `[handoff] 已转交给 ${target}。原因：${handoffCall.args.reason}`,
      });

      return {
        messages: [response, toolReply],
        activeAgent: target,
        handoffHistory: [target],
      };
    }

    // 没 handoff，说明 Agent 自己回答完了
    console.log(`[${name}] 直接回复用户`);
    return {
      messages: [response],
      // activeAgent 不变；handoffHistory 不追加
    };
  };
}

const frontdeskNode = makeAgentNode(
  "frontdesk",
  frontdeskLlm,
  FRONTDESK_SYSTEM
);
const technicalNode = makeAgentNode(
  "technical",
  technicalLlm,
  TECHNICAL_SYSTEM
);
const billingNode = makeAgentNode("billing", billingLlm, BILLING_SYSTEM);

// ── 路由条件边 ─────────────────────────────────────────────────────────────
// 决策依据：节点末尾是否追加了 ToolMessage（=发生了 handoff）
//   - 是：跳到 state.activeAgent 指向的下一个 Agent
//   - 否：Agent 自己答完了 → END
// 同时做防循环兜底
function routeFromAgent(
  state: typeof SwarmState.State
): AgentName | typeof END {
  const h = state.handoffHistory;

  // 防循环：最近形成 X→Y→X，强制 END
  if (h.length >= 3 && h[h.length - 1] === h[h.length - 3]) {
    console.warn(`[router] ⚠️ 检测到循环 ${h.slice(-3).join(" → ")}，强制 END`);
    return END;
  }

  const last = state.messages.at(-1);
  // 节点刚刚 handoff 时，最后一条是 ToolMessage（见 makeAgentNode 返回值）
  if (last?.getType?.() === "tool") {
    return state.activeAgent;
  }
  return END;
}

// ── 组装 Mesh 父图 ─────────────────────────────────────────────────────────
// Mesh 拓扑：每个 Agent 节点都可以走条件边到其他任一 Agent 节点或 END
const graph = new StateGraph(SwarmState)
  .addNode("frontdesk", frontdeskNode)
  .addNode("technical", technicalNode)
  .addNode("billing", billingNode)
  // 入口固定从 frontdesk 开始
  .addEdge(START, "frontdesk")
  // 每个 Agent 节点完成后，通过统一的路由函数决定下一站
  .addConditionalEdges("frontdesk", routeFromAgent, {
    technical: "technical",
    billing: "billing",
    frontdesk: "frontdesk",
    [END]: END,
  })
  .addConditionalEdges("technical", routeFromAgent, {
    technical: "technical",
    billing: "billing",
    frontdesk: "frontdesk",
    [END]: END,
  })
  .addConditionalEdges("billing", routeFromAgent, {
    technical: "technical",
    billing: "billing",
    frontdesk: "frontdesk",
    [END]: END,
  })
  .compile({ checkpointer: new MemorySaver() });

// ── Demo ──────────────────────────────────────────────────────────────────
async function main() {
  const cases = [
    {
      title: "案例 1：简单咨询（frontdesk 直接答完，0 次 handoff）",
      input: "你们今天营业吗？营业时间是几点？",
    },
    {
      title: "案例 2：技术问题（frontdesk → technical）",
      input: "我调用你们的 API 时返回 401 错误，能帮我看看是什么原因吗？",
    },
    {
      title:
        "案例 3：意图错位 + 同行直转（先看起来像技术，实际是订单 → technical → billing）",
      input:
        "我下的订单 #ORD-001 在你们系统里报错，物流一直没更新，能帮我查吗？",
    },
  ];

  for (const [i, c] of cases.entries()) {
    console.log("\n" + "═".repeat(70));
    console.log(`【Case ${i + 1}】${c.title}`);
    console.log(`用户: ${c.input}`);
    console.log("═".repeat(70));

    const finalState = await graph.invoke(
      { messages: [new HumanMessage(c.input)] },
      {
        configurable: { thread_id: `swarm-mesh-case-${i + 1}` },
        recursionLimit: 10, // 兜底防失控
      }
    );

    const finalMsg = finalState.messages.at(-1) as AIMessage;
    console.log(`\n[最终回复 by ${finalMsg.name}]`);
    console.log(finalMsg.content);
    console.log(
      `\n[trace] 交接路径: ${
        finalState.handoffHistory.length
          ? "frontdesk → " + finalState.handoffHistory.join(" → ")
          : "frontdesk（未 handoff）"
      }`
    );
    console.log(`[trace] LLM 调用次数 ≈ ${finalState.handoffHistory.length + 1}`);
  }
}

main().catch(console.error);
