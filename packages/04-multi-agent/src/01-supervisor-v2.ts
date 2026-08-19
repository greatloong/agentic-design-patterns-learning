/**
 * 01 - Supervisor V2（生产级实现）
 *
 * ── 相比 V1 的两个关键升级 ───────────────────────────────────────────────
 *
 * 1. 路由信号走独立 State 字段（不污染 messages）
 *    V1：supervisor 把决策塞进 messages 末尾的 AIMessage（[ROUTE:xxx]）
 *        → 条件边解析字符串路由
 *        → 缺点：messages 被污染、Anthropic API 拒绝末尾是 AI 的对话
 *    V2：父图 State 增加 next 字段，supervisor 节点直接 return { next: "..." }
 *        → 条件边读 state.next
 *        → messages 永远干净
 *
 * 2. Isolated + Summary（子 Agent 上下文隔离）
 *    V1：子图工具调用产生的 tool_calls / ToolMessage 全部追加进父图 messages
 *        → 父图 messages 飞速膨胀
 *        → supervisor 看到一堆中间细节，token 浪费 + 协议踩坑
 *    V2：子 Agent 内部用自己的子图 messages 做完整 ReAct，
 *        只把"最终总结"作为一条 AIMessage 回写到父图
 *        → 父图永远只有：用户问题 / researcher 总结 / coder 总结
 *        → token 省，trace 清晰，协议不踩坑
 *
 * ── 父图 State 设计 ─────────────────────────────────────────────────────
 *   messages: 用户与子 Agent 的高层对话（不含子 Agent 内部 tool 细节）
 *   next:     supervisor 决策的下一步（"researcher" | "coder" | "FINISH"）
 *   reason:   决策理由（仅日志用）
 *
 * ── 子图独立性 ─────────────────────────────────────────────────────────
 *   researcherGraph / coderGraph 各自管理自己的 messages，不和父图共享
 *   父图 → 子图：把高层 messages 作为初始 messages 注入
 *   子图 → 父图：把 finalMsg.content 包成新 AIMessage 回写，仅一条
 *
 * ── 与 V1 共用 ─────────────────────────────────────────────────────────
 *   - LLM 选型：supervisor/coder = Claude，researcher = DeepSeek
 *   - searchTool 来自共享模块
 *   - 三条决策规则不变
 */

import "dotenv/config";
import {
  StateGraph,
  START,
  END,
  Annotation,
  MessagesAnnotation,
  MemorySaver,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  BaseMessage,
} from "@langchain/core/messages";
import { z } from "zod";
import { searchTool } from "./shared/search-tool.js";

// ── LLM ───────────────────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-v4-flash",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

const toolUseLlm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

// ── 父图 State ─────────────────────────────────────────────────────────────
type RouteTarget = "researcher" | "coder" | "FINISH";

const SupervisorState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
  next: Annotation<RouteTarget>({
    reducer: (_prev, next) => next, // 替换语义
    default: () => "FINISH",
  }),
  reason: Annotation<string>({
    reducer: (_prev, next) => next,
    default: () => "",
  }),
});

// ── 子 Agent 1：Researcher（独立子图，使用 MessagesAnnotation）────────────
const RESEARCHER_SYSTEM = new SystemMessage(
  "你是一个研究员。任务：搜集技术资料并整理。可用 search 工具查找信息（建议英文专业术语）。" +
    "拿到资料后，用 2-4 句话总结要点交回主管。不要写代码。"
);

const researcherLlm = toolUseLlm.bindTools([searchTool]);

async function researcherLlmNode(
  state: typeof MessagesAnnotation.State
): Promise<Partial<typeof MessagesAnnotation.State>> {
  const response = await researcherLlm.invoke([
    RESEARCHER_SYSTEM,
    ...state.messages,
  ]);
  return { messages: [response] };
}

const researcherGraph = new StateGraph(MessagesAnnotation)
  .addNode("agent", researcherLlmNode)
  .addNode("tools", new ToolNode([searchTool]))
  .addEdge(START, "agent")
  .addConditionalEdges("agent", (state) => {
    const last = state.messages.at(-1) as AIMessage;
    return last.tool_calls?.length ? "tools" : END;
  })
  .addEdge("tools", "agent")
  .compile();

// ── 子 Agent 2：Coder（独立子图）─────────────────────────────────────────
const CODER_SYSTEM = new SystemMessage(
  "你是一个 TypeScript 编码专家。基于已有的对话上下文（特别是研究员提供的资料），" +
    "写出简洁、有类型注解的代码示例。只输出代码块和最简洁的说明，不要重复研究内容。"
);

async function coderLlmNode(
  state: typeof MessagesAnnotation.State
): Promise<Partial<typeof MessagesAnnotation.State>> {
  const response = await llm.invoke([CODER_SYSTEM, ...state.messages]);
  return { messages: [response] };
}

const coderGraph = new StateGraph(MessagesAnnotation)
  .addNode("agent", coderLlmNode)
  .addEdge(START, "agent")
  .addEdge("agent", END)
  .compile();

// ── 子 Agent 节点（父图视角）─────────────────────────────────────────────
// 关键：只把子图最终输出的 AIMessage 摘要回写父图，丢弃中间 tool 细节
async function researcherAgentNode(
  state: typeof SupervisorState.State
): Promise<Partial<typeof SupervisorState.State>> {
  console.log(`\n[researcher] 接管，父图消息数: ${state.messages.length}`);
  const result = await researcherGraph.invoke({ messages: state.messages });
  const finalMsg = result.messages.at(-1) as AIMessage;
  const summary = new AIMessage({
    content: finalMsg.content,
    name: "researcher",
  });
  console.log(
    `[researcher] 摘要: ${(summary.content as string).slice(0, 60)}...`
  );
  return { messages: [summary] };
}

async function coderAgentNode(
  state: typeof SupervisorState.State
): Promise<Partial<typeof SupervisorState.State>> {
  console.log(`\n[coder] 接管，父图消息数: ${state.messages.length}`);
  const result = await coderGraph.invoke({ messages: state.messages });
  const finalMsg = result.messages.at(-1) as AIMessage;
  const summary = new AIMessage({ content: finalMsg.content, name: "coder" });
  console.log(`[coder] 摘要长度: ${(summary.content as string).length} 字`);
  return { messages: [summary] };
}

// ── Supervisor ─────────────────────────────────────────────────────────────
const routeSchema = z.object({
  next: z
    .enum(["researcher", "coder", "FINISH"])
    .describe(
      "下一步派给哪个 Agent；如果用户问题已得到充分解答，返回 FINISH"
    ),
  reason: z.string().describe("决策的简短理由（一句话）"),
});

const supervisorRouter = llm.withStructuredOutput(routeSchema, {
  method: "functionCalling",
});

async function supervisorNode(
  state: typeof SupervisorState.State
): Promise<Partial<typeof SupervisorState.State>> {
  console.log(`\n[supervisor] 决策中，messages: ${state.messages.length}`);

  const decision = await supervisorRouter.invoke([
    new SystemMessage(
      "你是一个团队主管，协调以下子 Agent：\n" +
        "- researcher：研究员，擅长用 search 工具查找技术资料并总结\n" +
        "- coder：编码专家，擅长基于已有资料写 TypeScript 代码\n\n" +
        "决策规则：\n" +
        "1. 如果用户问题需要查资料，且还没查过 → next=researcher\n" +
        "2. 如果用户问题需要写代码，且必要资料已就绪 → next=coder\n" +
        "3. 如果用户问题已得到完整回答（既有资料也有代码，或仅需其一已满足）→ next=FINISH\n" +
        "不要重复派发同一个 Agent，除非用户明确提出新问题。"
    ),
    ...state.messages,
    new HumanMessage("根据以上对话，请决定下一步派给哪个 Agent，或 FINISH。"),
  ]);

  console.log(`[supervisor] 决策: ${decision.next}（${decision.reason}）`);

  // ✨ 关键：只更新 next/reason 字段，不动 messages
  return { next: decision.next, reason: decision.reason };
}

// ── 路由条件边（读 state.next，不解析 message 字符串）────────────────────
function routeFromSupervisor(
  state: typeof SupervisorState.State
): "researcher" | "coder" | typeof END {
  if (state.next === "FINISH") return END;
  return state.next;
}

// ── 组装父图 ───────────────────────────────────────────────────────────────
const graph = new StateGraph(SupervisorState)
  .addNode("supervisor", supervisorNode)
  .addNode("researcher", researcherAgentNode)
  .addNode("coder", coderAgentNode)
  .addEdge(START, "supervisor")
  .addConditionalEdges("supervisor", routeFromSupervisor, {
    researcher: "researcher",
    coder: "coder",
    [END]: END,
  })
  // 子 Agent 完成后回到 supervisor，让它再决策（形成循环）
  .addEdge("researcher", "supervisor")
  .addEdge("coder", "supervisor")
  .compile({ checkpointer: new MemorySaver() });

// ── 多轮对话演示 ───────────────────────────────────────────────────────────
async function main() {
  const thread = { configurable: { thread_id: "supervisor-v2-demo" } };

  const turns = [
    "帮我搜一下 LangGraph 的 Send API 是什么？",
    "基于刚才的资料，给我写一个最简单的 TypeScript Send API 使用示例",
  ];

  for (const [i, userInput] of turns.entries()) {
    console.log("\n" + "═".repeat(60));
    console.log(`【第 ${i + 1} 轮】用户: ${userInput}`);
    console.log("═".repeat(60));

    const finalState = await graph.invoke(
      { messages: [new HumanMessage(userInput)] },
      thread
    );

    const finalMsg = finalState.messages.at(-1) as AIMessage;
    console.log(`\n[最终回复]\n${finalMsg.content}\n`);
    console.log(
      `[父图 messages 总数: ${finalState.messages.length}（V1 通常是这个数 × 2~3）]`
    );
  }
}

main().catch(console.error);
