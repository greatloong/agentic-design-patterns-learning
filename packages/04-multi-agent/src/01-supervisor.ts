/**
 * 01 - Supervisor（主管模式）
 *
 * ── 是什么 ──────────────────────────────────────────────────────────────────
 * 一个 Supervisor Agent 协调多个专家子 Agent。Supervisor 看到用户请求和当前
 * 进展后，决定下一步派给哪个子 Agent，或者判定任务完成。子 Agent 完成后，
 * 控制权回到 Supervisor，由它继续决策。
 *
 * 图结构（关键：有回边形成循环）：
 *   START → supervisor → (条件边) → researcher → supervisor
 *                                  → coder      → supervisor
 *                                  → END
 *
 * ── 与 Routing 的本质区别 ────────────────────────────────────────────────────
 * Routing：classifier 一次决策 → handler → END（DAG，一次性）
 * Supervisor：每轮重新决策 + 子 Agent 完成后回到 supervisor（循环图）
 * 一轮用户输入可能在 supervisor 和子 Agent 间往返多次。
 *
 * ── Context 共享策略：本例使用"全共享" ──────────────────────────────────────
 * 所有 Agent 读写同一份 messages 列表，子 Agent 的产出直接追加进去，
 * supervisor 和后续 Agent 都能看到。
 * 优点：实现最简单，能直接看到协作过程
 * 缺点：长期对话下 token 膨胀严重（生产中常切换到"隔离+摘要"策略）
 *
 * ── Supervisor 的本质 ───────────────────────────────────────────────────────
 * Supervisor 是一个用 withStructuredOutput 强制输出"下一步去哪"的 LLM 节点。
 * 它的 system prompt 必须清楚说明：
 *   1. 有哪些子 Agent，各自擅长什么
 *   2. 什么时候该选哪个 Agent
 *   3. 什么时候判定任务完成（FINISH）
 *
 * ── 子 Agent = 子图（图的可组合性）──────────────────────────────────────────
 * 每个子 Agent 都是一个独立编译的 StateGraph（researcherGraph / coderGraph），
 * 父图通过普通节点函数调用 subGraph.invoke(state) 把它接进来。
 * 优点：
 *   - 子图能独立测试、可视化、debug
 *   - 子图内部的 streaming 事件能传播到外层
 *   - 真正体现 LangGraph "图的可组合性" 设计
 * 不推荐做法：在节点函数里手写 for 循环模拟 ReAct（享受不到框架能力）
 *
 * ── 多轮对话特性 ─────────────────────────────────────────────────────────────
 * 每轮用户输入都会重新经过 supervisor，所以第二轮可能派给和第一轮完全不同的 Agent。
 * 子 Agent 能看到之前所有产出（因为共享 messages），天然支持上下文延续。
 */

import "dotenv/config";
import {
  StateGraph,
  START,
  END,
  MessagesAnnotation,
  MemorySaver,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { z } from "zod";
import { searchTool } from "./shared/search-tool.js";

// ── LLM ───────────────────────────────────────────────────────────────────
// 生产实践：不同节点用不同模型，按任务特性匹配。
//
// supervisor / coder：Claude（中转）
//   - 单次调用，结构化决策可靠
//   - withStructuredOutput 正常可用
//
// researcher：DeepSeek（直连）
//   - 多轮 ReAct（assistant tool_calls + tool_result + assistant ...）
//   - pumpkinai 把 OpenAI 协议转 Anthropic 时对这种场景有 bug：
//     "messages.X.content.X.text: Field required"（参考 02-tool-use.ts）
//   - DeepSeek 原生 OpenAI 协议，无中转层，稳定
const llm = new ChatOpenAI({
  model: "claude-sonnet-4-6",
  apiKey: process.env.CUSTOM_API_KEY,
  configuration: { baseURL: "https://new.pumpkinai.vip/v1" },
});

const toolUseLlm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

// ── 子 Agent 1：Researcher ─────────────────────────────────────────────────
// 内部是一个 ReAct 循环：能调用通用 search 工具（Brave Search 封装，见 ./shared/search-tool）

// Researcher 是一个真正的 ReAct 子图（生产推荐做法）
// 关键：子图本身是一个完整的 StateGraph，编译后可作为节点嵌入到父图中
// 用 toolUseLlm（DeepSeek）而非 Claude，避开 pumpkinai 中转层的多轮 tool 转换 bug
const researcherLlm = toolUseLlm.bindTools([searchTool]);

const RESEARCHER_SYSTEM = new SystemMessage(
  "你是一个研究员。任务：搜集技术资料并整理。可用 search 工具查找信息。" +
  "拿到资料后，用 1-2 句话总结要点交回主管。不要写代码。"
);

// Anthropic API（pumpkinai 中转）要求 message content 中的 text block 不能为空。
// LangChain 把"纯 tool_calls 的 AIMessage"转给中转层时，content 可能是 ""、[]、
// 或包含 {type:"text", text:""} 的数组，都会触发 "text: Field required"。
// Workaround：把所有 AIMessage 的 content 强制规范化为非空字符串。
function isEmptyContent(content: unknown): boolean {
  if (content == null) return true;
  if (typeof content === "string") return content.trim() === "";
  if (Array.isArray(content)) {
    if (content.length === 0) return true;
    // 数组里全部是空 text block
    return content.every(
      (b) =>
        b == null ||
        (typeof b === "object" &&
          (b as { type?: string; text?: string }).type === "text" &&
          ((b as { text?: string }).text ?? "").trim() === "")
    );
  }
  return false;
}

function fixEmptyContent(
  messages: typeof MessagesAnnotation.State["messages"]
) {
  return messages.map((m) => {
    if (m.getType?.() === "ai" && isEmptyContent(m.content)) {
      const ai = m as AIMessage;
      return new AIMessage({
        content: " ",
        tool_calls: ai.tool_calls,
        additional_kwargs: ai.additional_kwargs,
        name: ai.name,
        id: ai.id,
      });
    }
    return m;
  });
}

async function researcherLlmNode(
  state: typeof MessagesAnnotation.State
): Promise<Partial<typeof MessagesAnnotation.State>> {
  const cleaned = fixEmptyContent(state.messages);
  const response = await researcherLlm.invoke([RESEARCHER_SYSTEM, ...cleaned]);
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
  .compile();  // 子图编译时不带 checkpointer，由父图统一管理

// 剥掉父图末尾的 supervisor marker AIMessage（[ROUTE:xxx]），
// 它只用于路由，不应作为子 Agent 的对话上下文。
// 否则 Claude 会拒绝（"conversation must end with a user message"）。
function stripSupervisorMarker(
  messages: typeof MessagesAnnotation.State["messages"]
) {
  const last = messages.at(-1);
  if (last && last.getType?.() === "ai" && (last as AIMessage).name === "supervisor") {
    return messages.slice(0, -1);
  }
  return messages;
}

// Isolated + Summary 策略：子 Agent 内部的 tool_calls / tool_result 不暴露给父图，
// 只把最终总结作为一条干净的 AIMessage 回写。
// 好处：
//   1. 父图（supervisor）messages 永远是 user/assistant 简单交替，不会触发协议兼容问题
//   2. 节省 token，避免上下文爆炸
//   3. 路由 trace 清晰，每个子 Agent 只贡献一条结论
async function researcherAgentNode(
  state: typeof MessagesAnnotation.State
): Promise<Partial<typeof MessagesAnnotation.State>> {
  console.log(`\n[researcher] 接管，消息数: ${state.messages.length}`);
  const cleanMessages = stripSupervisorMarker(state.messages);
  const result = await researcherGraph.invoke({ messages: cleanMessages });
  // 只取子图最后一条 AIMessage（最终总结），不把中间 tool_calls/tool_result 带回父图
  const finalMsg = result.messages.at(-1) as AIMessage;
  const summary = new AIMessage({
    content: finalMsg.content,
    name: "researcher",
  });
  console.log(`[researcher] 输出: ${(summary.content as string).slice(0, 60)}...`);
  return { messages: [summary] };
}

// ── 子 Agent 2：Coder（也是一个子图，演示统一的图嵌套范式）──────────────
const CODER_SYSTEM = new SystemMessage(
  "你是一个 TypeScript 编码专家。基于已有的对话上下文（特别是研究员提供的资料），" +
  "写出简洁、有类型注解的代码示例。只输出代码块和最简洁的说明，不要重复研究内容。"
);

async function coderLlmNode(
  state: typeof MessagesAnnotation.State
): Promise<Partial<typeof MessagesAnnotation.State>> {
  const response = await llm.invoke([
    CODER_SYSTEM,
    ...fixEmptyContent(state.messages),
  ]);
  return { messages: [response] };
}

// Coder 子图：单节点（无工具），但保持和 Researcher 相同的图嵌套范式
const coderGraph = new StateGraph(MessagesAnnotation)
  .addNode("agent", coderLlmNode)
  .addEdge(START, "agent")
  .addEdge("agent", END)
  .compile();

async function coderAgentNode(
  state: typeof MessagesAnnotation.State
): Promise<Partial<typeof MessagesAnnotation.State>> {
  console.log(`\n[coder] 接管，消息数: ${state.messages.length}`);
  const cleanMessages = stripSupervisorMarker(state.messages);
  const result = await coderGraph.invoke({ messages: cleanMessages });
  const finalMsg = result.messages.at(-1) as AIMessage;
  const summary = new AIMessage({ content: finalMsg.content, name: "coder" });
  console.log(`[coder] 输出代码片段（${(summary.content as string).length} 字）`);
  return { messages: [summary] };
}

// ── Supervisor ─────────────────────────────────────────────────────────────
// 用 withStructuredOutput 强制 supervisor 输出结构化决策（next + reason）
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
  state: typeof MessagesAnnotation.State
): Promise<Partial<typeof MessagesAnnotation.State>> {
  console.log(`\n[supervisor] 决策中，已有消息数: ${state.messages.length}`);

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
    ...fixEmptyContent(stripSupervisorMarker(state.messages)),
    // Claude 要求对话末尾是 user message，子 Agent 回完是 AIMessage，
    // 这里追加一条决策 prompt 作为收尾
    new HumanMessage("根据以上对话，请决定下一步派给哪个 Agent，或 FINISH。"),
  ]);

  console.log(`[supervisor] 决策: ${decision.next}（理由：${decision.reason}）`);

  // 决策结果塞进 marker 消息，由条件边读取
  return {
    messages: [
      new AIMessage({
        content: `[ROUTE:${decision.next}] ${decision.reason}`,
        name: "supervisor",
      }),
    ],
  };
}

// 条件边：读取 supervisor 刚写入的 marker，决定路由
function routeFromSupervisor(
  state: typeof MessagesAnnotation.State
): "researcher" | "coder" | typeof END {
  const last = state.messages.at(-1) as AIMessage;
  const content = last.content as string;

  if (content.includes("[ROUTE:researcher]")) return "researcher";
  if (content.includes("[ROUTE:coder]")) return "coder";
  return END;
}

// ── 构建图 ─────────────────────────────────────────────────────────────────
const checkpointer = new MemorySaver(); // 用于多轮对话的持久化

const graph = new StateGraph(MessagesAnnotation)
  .addNode("supervisor", supervisorNode)
  .addNode("researcher", researcherAgentNode)
  .addNode("coder", coderAgentNode)
  .addEdge(START, "supervisor")
  // supervisor 之后由条件边决定
  .addConditionalEdges("supervisor", routeFromSupervisor)
  // 关键：子 Agent 完成后必须回到 supervisor（形成循环）
  .addEdge("researcher", "supervisor")
  .addEdge("coder", "supervisor")
  .compile({ checkpointer });

// ── 演示：多轮对话 ─────────────────────────────────────────────────────────
const config = { configurable: { thread_id: "demo-1" } };

async function chat(userInput: string, round: number) {
  console.log("\n" + "═".repeat(60));
  console.log(`【第 ${round} 轮】用户: ${userInput}`);
  console.log("═".repeat(60));

  const result = await graph.invoke(
    { messages: [new HumanMessage(userInput)] },
    config
  );

  // 最终回复 = 倒数第二条消息（最后一条是 supervisor 的 [ROUTE:FINISH] marker）
  const finalReply = [...result.messages]
    .reverse()
    .find(
      (m) =>
        m._getType() === "ai" &&
        !((m.content as string)?.startsWith("[ROUTE:"))
    );

  console.log("\n" + "─".repeat(60));
  console.log("最终回复：");
  console.log("─".repeat(60));
  console.log(finalReply?.content ?? "(无内容)");
}

// 第 1 轮：需要查资料
await chat("帮我搜一下 LangGraph 的 Send API 是什么？", 1);

// 第 2 轮：基于上一轮结果写代码（supervisor 应该派给 coder，而不是 researcher）
await chat("基于上面的信息，写一个使用 Send API 的最小 TypeScript 示例。", 2);
