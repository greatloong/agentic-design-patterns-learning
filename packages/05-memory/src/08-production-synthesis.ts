/**
 * 08 - Production Synthesis：生产级端到端记忆系统
 *
 * ── 目标 ──────────────────────────────────────────────────────────────────
 *
 * 将 01-07 的全部知识点按照生产架构组装成一个完整的 Agent 记忆系统：
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                Hot Path（同步，低延迟）                           │
 * │  1. PostgresSaver 加载 thread 历史                              │
 * │  2. Layered Compression 控制 token                             │
 * │  3. Mem0.search() 召回 Semantic Memory                         │
 * │  4. 组装 Working Memory → LLM 生成回复                         │
 * │  5. PostgresSaver 自动写入新 checkpoint                        │
 * └───────────────────────────┬─────────────────────────────────────┘
 * │                           │ 异步入队
 * │                           ▼
 * │ ┌─────────────────────────────────────────────────────────────┐
 * │ │            Cold Path（异步，后台）                            │
 * │ │  6. Mem0.add() → 内部 Extract + AUDN + Embed + Store        │
 * │ └─────────────────────────────────────────────────────────────┘
 *
 * ── 对比：教学版 vs 生产版 ────────────────────────────────────────────────
 *
 * │ 教学版 (01-07)          │ 生产版 (08)                          │ 为什么替换          │
 * │ MemorySaver             │ PostgresSaver                       │ 重启不丢数据        │
 * │ InMemoryStore           │ Mem0 (内置 vector store)             │ 封装 AUDN+Hybrid    │
 * │ 手写 AUDN+BM25+RRF     │ Mem0.add() 内部处理                  │ 生产不重复造轮子    │
 * │ 同步写入                │ ColdPathQueue 异步                   │ 不阻塞用户          │
 * │ 无压缩                  │ Layered Compression                 │ 控制 token 开销     │
 * │ 单用户                  │ userId namespace 隔离               │ 多租户              │
 *
 * ── 前置条件 ──────────────────────────────────────────────────────────────
 *
 * 1. Docker: docker compose up -d  (packages/05-memory/docker-compose.yml)
 *    → Postgres + pgvector on localhost:5433
 * 2. .env: DEEPSEEK_API_KEY, DASHSCOPE_API_KEY
 */

import "dotenv/config";
import { PostgresSaver } from "@langchain/langgraph-checkpoint-postgres";
import {
  StateGraph,
  START,
  END,
  MessagesAnnotation,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import {
  HumanMessage,
  SystemMessage,
  BaseMessage,
} from "@langchain/core/messages";
import { Memory } from "mem0ai/oss";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ═══════════════════════════════════════════════════════════════════════════
// 1. 基础配置
// ═══════════════════════════════════════════════════════════════════════════

const PG_CONN = "postgresql://agent:agent123@localhost:5433/agent_memory";

// Hot Path 主对话用强模型（v4-pro 默认开启 thinking mode，回答质量高）
const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

// ═══════════════════════════════════════════════════════════════════════════
// 2. Mem0 初始化 — Semantic Memory 的生产 drop-in 方案
//    内部封装了：Embedding + AUDN + Hybrid Retrieval + Vector Store
// ═══════════════════════════════════════════════════════════════════════════

const mem0 = new Memory({
  version: "v1.1",
  // Cold Path 辅助任务用轻量模型（flash 无 thinking，响应快、成本低）
  // 生产典型做法：主对话用强模型，辅助任务（fact extraction、AUDN 判断）用快模型
  llm: {
    provider: "openai",
    config: {
      model: "deepseek-v4-flash",
      apiKey: process.env.DASHSCOPE_API_KEY!,
      baseURL: process.env.DASHSCOPE_BASE_URL,
    } as any,
  },
  // Embedding 用通义千问 text-embedding-v4（1024 维，OpenAI-compatible API）
  embedder: {
    provider: "openai",
    config: {
      model: "text-embedding-v4",
      apiKey: process.env.DASHSCOPE_API_KEY!,
      baseURL: process.env.DASHSCOPE_BASE_URL,
    },
  },
  // 向量存储：本 Demo 用内存（生产中换 Qdrant / pgvector / Pinecone）
  vectorStore: {
    provider: "memory",
    config: {
      collectionName: "agent_memories",
      dimension: 1024, // 须与 embedding 模型输出维度一致
      dbPath: path.resolve(__dirname, "../.mem0-store.db"), // SQLite 存储路径（默认 ~/.mem0）
      // 指定哪些字段参与 embedding 索引（类似 InMemoryStore 的 index.fields）：
      // embeddingFields: ["content", "summary"],
      // Mem0 默认对整条 memory 的文本内容做 embedding；
      // 若你的 memory 是结构化对象（含 content / tags / summary 等字段），
      // 可通过此配置只对指定字段生成向量，避免噪声字段污染语义检索。
    },
  },
  // 禁用 Mem0 内置的 SQLite 历史记录（我们用 PostgresSaver 管理 Short-term）
  disableHistory: true,
});

// ═══════════════════════════════════════════════════════════════════════════
// 3. Cold Path Queue — 异步写入队列（保证顺序，不阻塞 Hot Path）
//
//    为什么需要队列而不是直接 Promise.all 并发？
//    用户连发："我住北京" → "不对，住上海"
//    无序并发可能后发先至 → Store 最终写入"北京"（旧值覆盖新值）
//    队列保证：先来先处理，写入顺序 = 消息顺序（FIFO）
//
//    生产升级路径：
//    - 单实例原型  → 内存 Promise chain（本文件）
//    - 多实例部署  → Redis Streams / BullMQ（跨进程共享）
//    - 大规模      → Kafka / SQS（持久化 + 削峰填谷）
// ═══════════════════════════════════════════════════════════════════════════

class ColdPathQueue {
  private chain: Promise<void> = Promise.resolve();
  private pending = 0;

  /** 将异步任务追加到串行队列尾部 */
  enqueue(task: () => Promise<void>) {
    this.pending++;
    this.chain = this.chain
      .then(task)
      .catch((err) => console.error("[ColdPath] 写入失败:", err.message))
      .finally(() => this.pending--);
  }

  /** 等待队列中所有任务执行完毕 */
  async drain() {
    await this.chain;
  }

  get size() {
    return this.pending;
  }
}

const coldQueue = new ColdPathQueue();

// ═══════════════════════════════════════════════════════════════════════════
// 4. Memory Compression — 分层压缩（近期原文 + 远期摘要）
//
//    问题：200 轮对话 × 500 tokens/轮 = 100k tokens 全塞给 LLM
//    - 成本爆炸（按 input tokens 计费）
//    - 性能下降（lost in the middle 问题）
//    - 物理限制（context window 有上限）
//
//    方案：Layered Compression（生产标配）
//    - 近期 RECENT_WINDOW 条消息保留原文（细节不丢）
//    - 更早的消息用 LLM 压缩为一段摘要（关键事实保留）
//    - 类比人脑：昨天的事记细节，上个月的只记大概
// ═══════════════════════════════════════════════════════════════════════════

const RECENT_WINDOW = 10; // 保留最近 10 条原文（约 5 轮对话）
const MAX_SUMMARY_TOKENS = 300;

async function layeredCompress(
  messages: BaseMessage[],
): Promise<BaseMessage[]> {
  if (messages.length <= RECENT_WINDOW) return messages;

  const older = messages.slice(0, -RECENT_WINDOW);
  const recent = messages.slice(-RECENT_WINDOW);

  const olderText = older
    .map(
      (m) =>
        `${m.getType()}: ${typeof m.content === "string" ? m.content : JSON.stringify(m.content)}`,
    )
    .join("\n");

  const summaryResp = await llm.invoke([
    new SystemMessage(
      `请将以下对话历史压缩为一段简洁的中文摘要（不超过${MAX_SUMMARY_TOKENS}字），保留关键事实和用户偏好：`,
    ),
    new HumanMessage(olderText),
  ]);

  const summaryMsg = new SystemMessage(
    `[历史摘要] ${typeof summaryResp.content === "string" ? summaryResp.content : JSON.stringify(summaryResp.content)}`,
  );

  return [summaryMsg, ...recent];
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Agent Graph — 使用 PostgresSaver 做 Short-term Memory
//
//    PostgresSaver vs MemorySaver：
//    - MemorySaver：进程内存，重启丢失，只适合开发调试
//    - PostgresSaver：持久化到 Postgres，支持多实例共享、thread 恢复
//
//    关键 API：
//    - PostgresSaver.fromConnString(connString) → 从连接字符串创建
//    - await checkpointer.setup() → 首次使用时创建表结构（幂等）
//    - graph.compile({ checkpointer }) → 每个节点执行后自动保存 State 快照
// ═══════════════════════════════════════════════════════════════════════════

async function buildGraph() {
  const checkpointer = PostgresSaver.fromConnString(PG_CONN);
  await checkpointer.setup();

  const graph = new StateGraph(MessagesAnnotation)
    .addNode("agent", async (state) => {
      const userId = "demo-user"; // 生产中从 config 传入

      // ── Hot Path Step 1: 压缩历史 ──
      const compressed = await layeredCompress(state.messages);

      // ── Hot Path Step 2: 从 Mem0 检索相关 Semantic Memory ──
      // Mem0.search() 内部执行：query → embedding → 向量相似度检索
      // 返回与当前用户消息语义最相关的长期记忆（偏好、事实等）
      const lastUserMsg = [...state.messages]
        .reverse()
        .find((m) => m.getType() === "human");
      const query =
        typeof lastUserMsg?.content === "string"
          ? lastUserMsg.content
          : "用户偏好";

      // Mem0 v3.x API 注意：
      // - add() 支持顶层 userId
      // - search() / getAll() 必须用 filters: { user_id: "..." }
      let semanticContext = "";
      try {
        const memories = await mem0.search(query, {
          filters: { user_id: userId },
        });
        if (memories.results && memories.results.length > 0) {
          semanticContext = memories.results
            .map((m: any) => `- ${m.memory}`)
            .join("\n");
          console.log(
            `[Mem0 Search] 命中 ${memories.results.length} 条语义记忆:\n${semanticContext}`,
          );
        } else {
          console.log("[Mem0 Search] 未命中任何语义记忆（store 可能为空）");
        }
      } catch (e: any) {
        console.error("[Mem0 Search] 检索失败:", e.message);
      }

      // ── Hot Path Step 3: 组装 Working Memory → LLM ──
      // Working Memory = System Prompt + Semantic Memory 检索结果 + 压缩后的对话历史
      // 这就是最终送给 LLM 的完整 context
      const systemContent = [
        "你是一个贴心的私人助手。根据用户的长期偏好和当前对话，给出个性化回答。",
        semanticContext ? `\n[用户长期记忆]\n${semanticContext}` : "",
      ].join("");

      const resp = await llm.invoke([
        new SystemMessage(systemContent),
        ...compressed,
      ]);

      // ── Cold Path: 异步将对话写入 Mem0 做 AUDN ──
      // Mem0.add() 内部流程：
      //   1. LLM 从对话中提取事实（fact extraction）
      //   2. AUDN 仲裁：对每条事实决定 Add/Update/Delete/Noop
      //   3. 向量化（embedding）并写入 vector store
      // 这些操作耗时但"下次对话"才用得到，所以异步不阻塞用户
      const userText = query;
      const assistantText =
        typeof resp.content === "string" ? resp.content : "";
      coldQueue.enqueue(async () => {
        try {
          await mem0.add(
            [
              { role: "user", content: userText },
              { role: "assistant", content: assistantText },
            ],
            { userId },
          );
        } catch (e: any) {
          console.error("[Mem0 Add] AUDN 写入失败:", e.message);
        }
      });

      return { messages: [resp] };
    })
    .addEdge(START, "agent")
    .addEdge("agent", END);

  return graph.compile({ checkpointer });
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Demo 运行 — 使用 streamMode "messages" 实现逐 token 流式输出
//
//    streamMode 对比：
//    - "values"：每个节点执行完后输出完整 State（适合调试）
//    - "updates"：每个节点执行完后只输出变更部分（适合监控）
//    - "messages"：LLM 生成过程中逐 token 流出（适合 C 端产品体验）
//
//    生产价值：用户无需等待 LLM 完整生成，首 token 延迟 ~200ms 即可开始阅读
// ═══════════════════════════════════════════════════════════════════════════

/**
 * 流式对话：使用 graph.stream() + streamMode "messages"
 * LLM 每生成一个 token 就立刻输出到终端，用户体验接近实时
 */
async function chat(
  app: any,
  threadId: string,
  message: string,
  label: string = "User",
): Promise<string> {
  console.log(`${label}: ${message}`);
  let fullText = "";
  const stream = await app.stream(
    { messages: [new HumanMessage(message)] },
    { configurable: { thread_id: threadId }, streamMode: "messages" },
  );
  process.stdout.write("AI:   ");
  for await (const [msg, metadata] of stream) {
    if (metadata.langgraph_node === "agent" && msg.content) {
      const chunk = msg.content as string;
      process.stdout.write(chunk);
      fullText += chunk;
    }
  }
  console.log();
  return fullText;
}

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  08 - Production Synthesis: 生产级 Agent 记忆系统");
  console.log("═══════════════════════════════════════════════════════════\n");

  // ── 初始化 ──
  console.log("[Init] 连接 PostgresSaver (localhost:5433)...");
  const app = await buildGraph();
  console.log("[Init] ✓ Graph 构建完成，PostgresSaver 就绪\n");

  const threadId = `thread-prod-${Date.now()}`;
  console.log(`[Config] thread_id = ${threadId}`);
  console.log(`[Config] userId = demo-user\n`);

  // ── Demo 1: 基础对话 + Semantic Memory 写入 ──
  console.log("─── Demo 1: 基础对话，自我介绍 → Mem0 提取偏好 ───────────\n");

  await chat(
    app,
    threadId,
    "你好！我叫小王，是一名前端工程师，最喜欢用React。",
  );

  await chat(app, threadId, "我平时喜欢喝美式咖啡，不加糖。");

  // 等待 Cold Path 完成写入
  console.log("[ColdPath] 等待异步写入完成...");
  await coldQueue.drain();
  console.log("[ColdPath] ✓ 写入完成\n");

  // ── Demo 2: 验证 Semantic Memory 被召回 ──
  console.log("─── Demo 2: 新对话验证 Mem0 记忆召回 ─────────────────────\n");

  await chat(app, threadId, "帮我推荐一个适合我的技术栈？");
  console.log(
    "  → 期望：AI 回答中提到 React / 前端，因为 Mem0 记住了用户偏好\n",
  );

  // ── Demo 3: 多租户隔离验证 ──
  console.log("─── Demo 3: 多租户隔离 ─ 不同 thread 独立 ───────────────\n");

  const threadId2 = `thread-prod-${Date.now()}-b`;
  await chat(app, threadId2, "今天天气怎么样？", "User (thread B)");
  console.log("  → thread B 没有 thread A 的对话历史（PostgresSaver 隔离）\n");

  // ── Demo 4: 查看 Mem0 中存储的记忆 ──
  console.log("─── Demo 4: 查看 Mem0 存储的 Semantic Memory ─────────────\n");

  try {
    const allMemories = await mem0.getAll({
      filters: { user_id: "demo-user" },
    });
    if (allMemories.results && allMemories.results.length > 0) {
      console.log(`[Mem0] 共 ${allMemories.results.length} 条记忆：`);
      allMemories.results.forEach((m: any, i: number) => {
        console.log(`  ${i + 1}. ${m.memory}`);
      });
    } else {
      console.log("[Mem0] 暂无记忆（可能 AUDN 判定为 Noop）");
    }
  } catch (e: any) {
    console.error("[Mem0 GetAll] 读取失败:", e.message);
  }

  // ── 清理 ──
  await coldQueue.drain();
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  完成！生产级记忆系统演示结束");
  console.log("  - PostgresSaver: Short-term Memory 持久化 ✓");
  console.log("  - Mem0: Semantic Memory (AUDN + Hybrid) ✓");
  console.log("  - ColdPathQueue: 异步写入不阻塞 ✓");
  console.log("  - Layered Compression: token 控制 ✓");
  console.log("  - Namespace (userId): 多租户隔离 ✓");
  console.log("═══════════════════════════════════════════════════════════");

  process.exit(0);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
