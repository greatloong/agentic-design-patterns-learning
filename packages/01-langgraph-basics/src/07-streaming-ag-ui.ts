/**
 * 07 - Streaming + AG-UI 协议（官方方案：@ag-ui/langgraph + langgraph dev）
 *
 * ── 这节解决的问题 ────────────────────────────────────────────────────────
 *
 *   06 解决「Agent 内部怎么流式」（graph.stream 的 streamMode）。
 *   07 解决「Agent 怎么把流式结果按标准协议吐给前端」—— AG-UI。
 *   本文件用**官方库**做这件事，对照 07-streaming-ag-ui-manual.ts（手写转换器）。
 *
 * ── 官方方案是「双进程」 ──────────────────────────────────────────────────
 *
 *   官方 LangGraphAgent 只是个**客户端**，必须连一个 LangGraph Platform server，
 *   而 published 版没有进程内跑图的能力（那是未发布的 LangGraphLocalAgent）。
 *   所以官方路径绕不开两个进程：
 *
 *     进程 ①  langgraph dev（@langchain/langgraph-cli）
 *               读 langgraph.json → 加载 src/agui/graph.ts 的 graph
 *               → 起 LangGraph Server（默认 :2024），暴露 runs/threads/assistants
 *
 *     进程 ②  本文件（SSE 服务，:8787）
 *               new LangGraphAgent({ deploymentUrl: ":2024", graphId: "agent" })
 *               每个请求构造 RunAgentInput → agent.run(input) 得到事件 Observable
 *               → EventEncoder 编码成 AG-UI SSE 吐给前端
 *
 *   reasoning 怎么来：图用 DeepSeek 的 Anthropic 兼容端点 + thinking，思维链以
 *   Anthropic thinking block 流出，被官方 resolveReasoningContent 识别为
 *   REASONING_* 事件（细节见 src/agui/graph.ts 注释）。
 *
 * ── 运行（两个终端）─────────────────────────────────────────────────────
 *
 *   终端 A:  pnpm run 07:server      # 起 langgraph dev（:2024），等它就绪
 *   终端 B:  pnpm run 07             # 起 SSE 服务（:8787）
 *
 *   curl -N -X POST http://localhost:8787/agui \
 *     -H 'Content-Type: application/json' \
 *     -d '{"messages":[{"role":"user","content":"搜索一下 LangGraph 最新版本有什么新特性"}]}'
 */

import "dotenv/config";
import http from "node:http";
import { randomUUID } from "node:crypto";
import { LangGraphAgent } from "@ag-ui/langgraph";
import { EventEncoder } from "@ag-ui/encoder";
import { EventType, type RunAgentInput, type BaseEvent } from "@ag-ui/core";

// ── 连到 langgraph dev 起的 Platform server ──
const DEPLOYMENT_URL = process.env.LANGGRAPH_URL ?? "http://localhost:2024";
const GRAPH_ID = process.env.LANGGRAPH_GRAPH_ID ?? "agent"; // 对应 langgraph.json 里的 key
const PORT = Number(process.env.PORT ?? 8787);

const encoder = new EventEncoder();

// 官方 LangGraphAgent 默认会额外吐两类「噪音」事件：
//   - RAW：转发 LangGraph streamEvents 的每条底层原始事件（reasoning 模型一个 token 一条，量极大）
//   - STATE_SNAPSHOT / STATE_DELTA / MESSAGES_SNAPSHOT：给 CopilotKit 共享状态同步用
// 我们是纯 SSE、无共享状态，前端只需要 RUN_*/STEP_*/REASONING_*/TEXT_MESSAGE_*/TOOL_CALL_*。
// 默认过滤掉噪音；想看全量（调试/学习）时设 AGUI_VERBOSE=1。
const VERBOSE = process.env.AGUI_VERBOSE === "1";
const DROPPED_EVENTS = new Set<string>([
  EventType.RAW,
  EventType.STATE_SNAPSHOT,
  EventType.STATE_DELTA,
  EventType.MESSAGES_SNAPSHOT,
]);

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

/**
 * 把一次运行的 AG-UI 事件流写进 SSE 响应。
 * LangGraphAgent.run() 返回 RxJS Observable<BaseEvent>，我们订阅它，
 * 每来一条事件就用 EventEncoder 编码成 `data: {...}\n\n` 写出去。
 */
function streamRun(input: RunAgentInput, res: http.ServerResponse): Promise<void> {
  // 每个请求一个 agent 实例：它内部持有该次运行的状态（消息/思考进度），
  // 独立实例避免并发请求互相串台。构造本身很轻（只是建一个 HTTP 客户端）。
  const agent = new LangGraphAgent({ deploymentUrl: DEPLOYMENT_URL, graphId: GRAPH_ID });

  return new Promise<void>((resolve) => {
    const subscription = agent.run(input).subscribe({
      next: (event: BaseEvent) => {
        if (!VERBOSE && DROPPED_EVENTS.has(event.type)) return;
        console.log(`  → ${event.type}`);
        // 官方会给几乎每条事件附带 rawEvent（原始 LangGraph 事件的完整副本），
        // 前端用不到却显著撑大 payload；默认剥掉，VERBOSE 时保留。
        const payload = VERBOSE ? event : ({ ...event, rawEvent: undefined } as BaseEvent);
        res.write(encoder.encode(payload));
      },
      error: (err: unknown) => {
        console.error("[run error]", err);
        res.write(
          encoder.encode({
            type: EventType.RUN_ERROR,
            message: err instanceof Error ? err.message : String(err),
          } as BaseEvent)
        );
        res.end();
        resolve();
      },
      complete: () => {
        res.end();
        resolve();
      },
    });

    // 客户端断开就取消订阅，避免无谓地继续拉取上游
    res.on("close", () => subscription.unsubscribe());
  });
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
    const runId = randomUUID();

    // RunAgentInput 是 AG-UI 的标准入参。我们只填最小集合：
    // 一条 user 消息 + 空的 tools/context（工具在图里 bindTools，不从这里传）。
    const input: RunAgentInput = {
      threadId,
      runId,
      state: {},
      messages: [{ id: randomUUID(), role: "user", content: question }],
      tools: [],
      context: [],
      forwardedProps: {},
    };

    res.writeHead(200, {
      "Content-Type": encoder.getContentType(), // text/event-stream
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });

    console.log(`\n[run ${runId.slice(0, 8)}] 问题: ${question}`);
    await streamRun(input, res);
    return;
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Not Found. Use POST /agui" }));
});

server.listen(PORT, () => {
  console.log("═".repeat(64));
  console.log(`AG-UI SSE 服务已启动: http://localhost:${PORT}/agui (POST)`);
  console.log(`上游 LangGraph Server: ${DEPLOYMENT_URL}  (graphId=${GRAPH_ID})`);
  console.log("⚠️  需先在另一个终端跑: pnpm run 07:server  (langgraph dev :2024)");
  console.log("试试（-N 实时看 SSE 流）:");
  console.log(
    `  curl -N -X POST http://localhost:${PORT}/agui \\\n` +
      `    -H 'Content-Type: application/json' \\\n` +
      `    -d '{"messages":[{"role":"user","content":"搜索一下 LangGraph 最新版本有什么新特性"}]}'`
  );
  console.log("═".repeat(64));
});
