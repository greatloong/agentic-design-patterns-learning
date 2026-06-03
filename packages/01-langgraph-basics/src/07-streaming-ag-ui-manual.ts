/**
 * 07（手写版 / 参考实现）- Streaming + AG-UI 协议（流式输出 + 思考内容）
 *
 * ⚠️ 这是「手写 LangGraph→AG-UI 转换器」的参考版本，进程内直接跑图、纯 SSE。
 *    正式版 07-streaming-ag-ui.ts 已改为官方方案（@ag-ui/langgraph 的
 *    LangGraphAgent + langgraph dev Platform server）。两者对照着看：
 *      - 本文件：自己实现 token 流 → AG-UI 事件的状态机，看得清协议细节；
 *      - 官方版：转换交给 LangGraphAgent，自己只起 SSE 服务，最贴官方生态。
 *    本文件用 deepseek-v4-pro（DashScope OpenAI 兼容接口），reasoning 在
 *    additional_kwargs.reasoning_content；官方版改用 DeepSeek 的 Anthropic
 *    兼容端点 + thinking content block，才能被官方解析器识别为 REASONING_*。
 *
 * ── 这节在 06-streaming 之上回答的问题 ────────────────────────────────────
 *
 *   06 解决的是「Agent 内部怎么流式」：graph.stream() 的三种 streamMode。
 *   07 解决的是「Agent 怎么把流式结果吐给前端」：把 LangGraph 的 token 流
 *   翻译成一套**标准化、前端可直接消费**的事件协议 —— AG-UI。
 *
 * ── 什么是 AG-UI ──────────────────────────────────────────────────────────
 *
 *   AG-UI（Agent-User Interaction Protocol，by CopilotKit）是一套
 *   「Agent ↔ 前端」的交互协议。后端通过 SSE 持续吐出一串带类型的事件，
 *   前端按事件类型增量渲染。核心事件分三类：
 *
 *     - 运行生命周期：RUN_STARTED / RUN_FINISHED / RUN_ERROR
 *                     STEP_STARTED / STEP_FINISHED（对应图里一个节点）
 *     - 正式回答：    TEXT_MESSAGE_START → ...CONTENT → ...END
 *     - 思考过程：    REASONING_START → REASONING_MESSAGE_START
 *                     → ...CONTENT → ...END → REASONING_END
 *     - 工具调用：    TOOL_CALL_START → ...ARGS → ...END → TOOL_CALL_RESULT
 *
 *   ⚠️ 注意：老的 THINKING_* 事件已废弃（1.0.0 移除），新代码一律用 REASONING_*。
 *
 * ── reasoning 内容从哪来 ──────────────────────────────────────────────────
 *
 *   DeepSeek 思考模式：在 content（最终答案）之前，先输出一段 reasoning_content
 *   （思维链）。两者同级。思考模式是「请求参数」而非特定模型：
 *     - 开关：thinking.type = enabled（默认 enabled）
 *     - 强度：reasoning_effort = high / max
 *   流式时，reasoning_content 逐 token 出现在 chunk.additional_kwargs 里。
 *
 *   本节用 deepseek-v4-pro（经 DashScope 兼容接口），思考模式天然开启。
 *
 * ── 翻译思路（本节核心）──────────────────────────────────────────────────
 *
 *   graph.stream(input, { streamMode: ["updates", "messages"] }) 同时拿到：
 *     - "messages"：agent 节点 LLM 的逐 token 流（含 reasoning_content / 文本 /
 *                    tool_call 参数）→ 翻译成 REASONING_* / TEXT_MESSAGE_* / TOOL_CALL_*
 *     - "updates"： 每个节点跑完后的状态变更 → agent 跑完则收尾上面的消息；
 *                    tools 跑完则把 ToolMessage 翻译成 TOOL_CALL_RESULT
 *   一个小状态机负责「按需开/关」各类消息事件，保证事件成对出现。
 *
 * ── ⚠️ 一个生产级的坑：tool 轮次必须回传 reasoning_content ──────────────────
 *
 *   DeepSeek 规定：未调用工具的轮次，中间 assistant 的 reasoning_content 可丢弃；
 *   但**调用了工具的轮次**，后续所有请求必须完整回传该 reasoning_content，
 *   否则 API 返回 400。ReAct 循环天然是「思考→调工具→再思考」的多轮，
 *   所以 agentNode 把上一条带 tool_calls 的 AIMessage 的 reasoning_content
 *   保留在 additional_kwargs 里随历史一起回传（见 agentNode 注释）。
 *
 * ── 运行 ─────────────────────────────────────────────────────────────────
 *
 *   pnpm run 07
 *   curl -N -X POST http://localhost:8787/agui \
 *     -H 'Content-Type: application/json' \
 *     -d '{"messages":[{"role":"user","content":"搜索一下 LangGraph 最新版本有什么新特性"}]}'
 *   （-N 关闭 curl 缓冲，才能实时看到 SSE 一条条到达）
 */

import "dotenv/config";
import http from "node:http";
import { randomUUID } from "node:crypto";
import { StateGraph, START, END, MessagesAnnotation } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { EventEncoder } from "@ag-ui/encoder";
import { EventType, type BaseEvent } from "@ag-ui/core";
import { searchTool } from "./shared/search-tool.js";

// ══════════════════════════════════════════════════════════════════════════
// 1. ReAct 图（agent ↔ tools），模型用 deepseek-v4-pro（DashScope）+ 思考模式
// ══════════════════════════════════════════════════════════════════════════

const tools = [searchTool];

const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
  // 思考模式参数：思考默认开启，这里显式调高思考强度。
  // 思考模式下 temperature/top_p 等采样参数不生效（传了也会被忽略）。
  modelKwargs: { reasoning_effort: "high" },
}).bindTools(tools);

async function agentNode(state: typeof MessagesAnnotation.State) {
  // 直接把完整 messages 历史回传给 LLM。
  // 历史里若有「带 tool_calls 的 AIMessage」，其 additional_kwargs.reasoning_content
  // 会随之回传 —— 这是 DeepSeek 思考模式 + 工具调用的硬性要求（否则 400）。
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

// ══════════════════════════════════════════════════════════════════════════
// 2. 翻译器：LangGraph token 流 → AG-UI 事件流
// ══════════════════════════════════════════════════════════════════════════

/**
 * 把一次 Agent 运行翻译成 AG-UI 事件序列（async generator）。
 * 调用方只管把 yield 出来的 event 编码进 SSE 即可，无需关心 LangGraph 细节。
 */
async function* runAgentAsAGUI(
  question: string,
  threadId: string,
  runId: string
): AsyncGenerator<BaseEvent> {
  // ── 状态机：保证每类消息的 start/content/end 成对出现 ──
  let stepOpen = false; // 当前是否在一个 agent 步骤内
  let reasoningOpen = false; // 思考消息是否已 start
  let textOpen = false; // 正式回答消息是否已 start
  let reasoningMsgId = "";
  let textMsgId = "";
  const openToolCalls = new Map<number, { id: string; name: string }>(); // 按 index 跟踪流式中的 tool_call

  // 收尾当前 agent 步骤里所有还开着的消息（思考/回答/工具）
  function* closeAgentArtifacts(): Generator<BaseEvent> {
    if (reasoningOpen) {
      yield { type: EventType.REASONING_MESSAGE_END, messageId: reasoningMsgId } as BaseEvent;
      yield { type: EventType.REASONING_END, messageId: reasoningMsgId } as BaseEvent;
      reasoningOpen = false;
    }
    if (textOpen) {
      yield { type: EventType.TEXT_MESSAGE_END, messageId: textMsgId } as BaseEvent;
      textOpen = false;
    }
    for (const tc of openToolCalls.values()) {
      yield { type: EventType.TOOL_CALL_END, toolCallId: tc.id } as BaseEvent;
    }
    openToolCalls.clear();
  }

  yield { type: EventType.RUN_STARTED, threadId, runId } as BaseEvent;

  try {
    const stream = await graph.stream(
      { messages: [new HumanMessage(question)] },
      { streamMode: ["updates", "messages"] }
    );

    for await (const [mode, data] of stream as AsyncIterable<[string, any]>) {
      // ── A. token 级流：只处理 agent 节点的 LLM 输出 ──
      if (mode === "messages") {
        const [chunk, metadata] = data as [any, any];
        if (metadata?.langgraph_node !== "agent") continue;

        if (!stepOpen) {
          stepOpen = true;
          reasoningMsgId = randomUUID();
          textMsgId = randomUUID();
          yield { type: EventType.STEP_STARTED, stepName: "agent" } as BaseEvent;
        }

        // A1. 思考内容（reasoning_content，先于正式答案出现）
        const reasoning = chunk?.additional_kwargs?.reasoning_content;
        if (typeof reasoning === "string" && reasoning.length > 0) {
          if (!reasoningOpen) {
            yield { type: EventType.REASONING_START, messageId: reasoningMsgId } as BaseEvent;
            yield {
              type: EventType.REASONING_MESSAGE_START,
              messageId: reasoningMsgId,
              role: "reasoning",
            } as BaseEvent;
            reasoningOpen = true;
          }
          yield {
            type: EventType.REASONING_MESSAGE_CONTENT,
            messageId: reasoningMsgId,
            delta: reasoning,
          } as BaseEvent;
        }

        // A2. 正式回答文本：一旦开始出文本，就先关掉思考
        const text = typeof chunk?.content === "string" ? chunk.content : "";
        if (text.length > 0) {
          if (reasoningOpen) {
            yield { type: EventType.REASONING_MESSAGE_END, messageId: reasoningMsgId } as BaseEvent;
            yield { type: EventType.REASONING_END, messageId: reasoningMsgId } as BaseEvent;
            reasoningOpen = false;
          }
          if (!textOpen) {
            yield {
              type: EventType.TEXT_MESSAGE_START,
              messageId: textMsgId,
              role: "assistant",
            } as BaseEvent;
            textOpen = true;
          }
          yield { type: EventType.TEXT_MESSAGE_CONTENT, messageId: textMsgId, delta: text } as BaseEvent;
        }

        // A3. 工具调用参数（增量），按 index 聚合
        for (const tcc of chunk?.tool_call_chunks ?? []) {
          if (reasoningOpen) {
            yield { type: EventType.REASONING_MESSAGE_END, messageId: reasoningMsgId } as BaseEvent;
            yield { type: EventType.REASONING_END, messageId: reasoningMsgId } as BaseEvent;
            reasoningOpen = false;
          }
          const idx: number = tcc.index ?? 0;
          if (!openToolCalls.has(idx)) {
            const tcId: string = tcc.id ?? `${runId}-tool-${idx}`;
            openToolCalls.set(idx, { id: tcId, name: tcc.name ?? "" });
            yield {
              type: EventType.TOOL_CALL_START,
              toolCallId: tcId,
              toolCallName: tcc.name ?? "",
              parentMessageId: textMsgId,
            } as BaseEvent;
          }
          if (tcc.args) {
            yield {
              type: EventType.TOOL_CALL_ARGS,
              toolCallId: openToolCalls.get(idx)!.id,
              delta: tcc.args,
            } as BaseEvent;
          }
        }
        continue;
      }

      // ── B. 节点级更新：收尾 agent 步骤 / 产出工具结果 ──
      if (mode === "updates") {
        const update = data as Record<string, any>;
        for (const [node, partial] of Object.entries(update)) {
          if (node === "agent") {
            yield* closeAgentArtifacts();
            if (stepOpen) {
              yield { type: EventType.STEP_FINISHED, stepName: "agent" } as BaseEvent;
              stepOpen = false;
            }
          } else if (node === "tools") {
            yield { type: EventType.STEP_STARTED, stepName: "tools" } as BaseEvent;
            const msgs: ToolMessage[] = partial?.messages ?? [];
            for (const m of msgs) {
              yield {
                type: EventType.TOOL_CALL_RESULT,
                messageId: m.id ?? randomUUID(),
                toolCallId: (m as any).tool_call_id,
                content: typeof m.content === "string" ? m.content : JSON.stringify(m.content),
                role: "tool",
              } as BaseEvent;
            }
            yield { type: EventType.STEP_FINISHED, stepName: "tools" } as BaseEvent;
          }
        }
      }
    }

    // 流结束兜底（正常情况下 updates(agent) 已收尾，这里防御性再关一次）
    yield* closeAgentArtifacts();
    if (stepOpen) {
      yield { type: EventType.STEP_FINISHED, stepName: "agent" } as BaseEvent;
      stepOpen = false;
    }

    yield { type: EventType.RUN_FINISHED, threadId, runId } as BaseEvent;
  } catch (err) {
    yield {
      type: EventType.RUN_ERROR,
      message: (err as Error).message ?? String(err),
    } as BaseEvent;
  }
}

// ══════════════════════════════════════════════════════════════════════════
// 3. HTTP SSE 服务端（POST /agui）
// ══════════════════════════════════════════════════════════════════════════

const PORT = Number(process.env.PORT ?? 8787);
const encoder = new EventEncoder();

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c as Buffer));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}

/** 从 AG-UI 风格 body（messages 数组）或 { question } 中取出用户问题 */
function extractQuestion(body: any): string {
  if (Array.isArray(body?.messages)) {
    const lastUser = [...body.messages].reverse().find((m: any) => m.role === "user");
    if (lastUser?.content) return String(lastUser.content);
  }
  if (typeof body?.question === "string") return body.question;
  return "搜索一下 LangGraph 最新版本有什么新特性";
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);

  if (req.method === "POST" && url.pathname === "/agui") {
    let body: any = {};
    try {
      const raw = await readBody(req);
      body = raw ? JSON.parse(raw) : {};
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "invalid JSON body" }));
      return;
    }

    const question = extractQuestion(body);
    const threadId = body?.threadId ?? randomUUID();
    const runId = body?.runId ?? randomUUID();

    res.writeHead(200, {
      "Content-Type": encoder.getContentType(), // text/event-stream
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    console.log(`\n[run ${runId.slice(0, 8)}] 问题: ${question}`);
    try {
      for await (const event of runAgentAsAGUI(question, threadId, runId)) {
        console.log(`  → ${event.type}`);
        res.write(encoder.encode(event)); // EventEncoder 直接产出 "data: {...}\n\n"
      }
    } catch (err) {
      // 兜底：流已开始无法改 header，只能发一条 RUN_ERROR 再结束
      res.write(encoder.encode({ type: EventType.RUN_ERROR, message: String(err) } as BaseEvent));
    } finally {
      res.end();
    }
    return;
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Not Found. Use POST /agui" }));
});

server.listen(PORT, () => {
  console.log("═".repeat(64));
  console.log(`AG-UI SSE 服务已启动: http://localhost:${PORT}/agui (POST)`);
  console.log("试试（-N 实时看 SSE 流）:");
  console.log(
    `  curl -N -X POST http://localhost:${PORT}/agui \\\n` +
      `    -H 'Content-Type: application/json' \\\n` +
      `    -d '{"messages":[{"role":"user","content":"搜索一下 LangGraph 最新版本有什么新特性"}]}'`
  );
  console.log("═".repeat(64));
});
