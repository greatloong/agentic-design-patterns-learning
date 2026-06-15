/**
 * 03 - Store + Embedding 语义检索
 *
 * ── 02 的天花板：Store 的两种查法都需要"预先知道查什么" ──────────────────
 *
 * Store 没有 Embedding 时只有两种查法：
 *   1. get(namespace, key)             → 必须知道 key 叫什么
 *   2. search(prefix, { filter })      → 必须知道 value 里某个字段的精确值
 *
 * 但真实场景是：用户发了一句"帮我推荐一个前端框架"。
 * Store 里存着 7 条记忆，你不知道该 get 哪个 key，也没有能 filter 的字段值。
 * → 有数据但找不到。
 *
 * ── Embedding 解决的核心问题 ─────────────────────────────────────────────
 *
 * 用户用自然语言问了一句话，怎么从一堆记忆里找到"语义相关"的那几条？
 *
 * Embedding = 把文本变成一组数字（向量），语义越近的文本，向量越接近。
 * 有了向量，就能用余弦相似度排序，找到最相关的 top-k。
 *
 * 没有 Embedding → Store 是只能精确查的字典
 * 有了 Embedding → Store 变成能理解自然语言的检索引擎
 *
 * 两者不是二选一，是同一个 Store 的两种查法：
 *   - get / filter  → 你知道要查什么时用（精确）
 *   - search(query) → 你不知道要查什么，让语义匹配来找（模糊）
 *
 * ── Store + Embedding 的工作流 ──────────────────────────────────────────
 *
 * 写入时（put）：
 *   value 里的指定字段 → 调用 Embedding 模型转向量 → 向量和原数据一起存储
 *
 * 检索时（search with query）：
 *   query → 转向量 → 和存储的所有向量算余弦相似度 → 按分数排序返回 top-k
 *
 * ── 本节演示 ──────────────────────────────────────────────────────────────
 * 1. Embedding 直观体验：文本 → 向量 → 余弦相似度
 * 2. 配置 InMemoryStore + IndexConfig → 语义检索
 * 3. 语义检索 vs filter：各自擅长什么，如何组合
 * 4. 在 Graph 节点里用语义检索注入相关记忆到 prompt
 */

import "dotenv/config";
import {
  StateGraph,
  START,
  END,
  MessagesAnnotation,
  MemorySaver,
  InMemoryStore,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import type { LangGraphRunnableConfig } from "@langchain/langgraph";

// ── Embedding 模型（百炼 text-embedding-v4，OpenAI 兼容格式）────────────
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-v4",
  dimensions: 512,
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: {
    baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  },
});

const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

// ═══════════════════════════════════════════════════════════════════════════
// 演示 1：Embedding 模型直观体验
// ═══════════════════════════════════════════════════════════════════════════
console.log("═".repeat(60));
console.log("演示 1：Embedding 是什么 —— 文本 → 向量");
console.log("═".repeat(60));

const vec1 = await embeddings.embedQuery("TypeScript 编程");
const vec2 = await embeddings.embedQuery("JavaScript 开发");
const vec3 = await embeddings.embedQuery("用户住在上海浦东");

console.log(`\n"TypeScript 编程" → 向量维度: ${vec1.length}，前5个数: [${vec1.slice(0, 5).map(n => n.toFixed(4)).join(", ")}...]`);
console.log(`"JavaScript 开发" → 向量维度: ${vec2.length}，前5个数: [${vec2.slice(0, 5).map(n => n.toFixed(4)).join(", ")}...]`);
console.log(`"用户住在上海浦东" → 向量维度: ${vec3.length}，前5个数: [${vec3.slice(0, 5).map(n => n.toFixed(4)).join(", ")}...]`);

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

console.log(`\n余弦相似度：`);
console.log(`  "TS编程" vs "JS开发"    = ${cosineSimilarity(vec1, vec2).toFixed(4)}  ← 语义相近，值高`);
console.log(`  "TS编程" vs "住在上海"   = ${cosineSimilarity(vec1, vec3).toFixed(4)}  ← 语义无关，值低`);
console.log(`  "JS开发" vs "住在上海"   = ${cosineSimilarity(vec2, vec3).toFixed(4)}  ← 语义无关，值低`);

// ═══════════════════════════════════════════════════════════════════════════
// 演示 2：配置带 Embedding 的 Store
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 2：InMemoryStore + IndexConfig → 语义检索");
console.log("═".repeat(60));

/**
 * IndexConfig 的作用：
 * - dims: 向量维度，必须和 Embedding 模型输出维度一致
 * - embeddings: LangChain Embeddings 实例
 * - fields: 指定 value 里哪些字段要被 embed（默认 ["$"] = 整个 value）
 *
 * 配置后，每次 put() 会自动对指定字段调用 embeddings.embedDocuments()，
 * search({ query }) 会自动调用 embeddings.embedQuery() 再算余弦相似度。
 */
const store = new InMemoryStore({
  index: {
    dims: 512,
    embeddings,
    fields: ["content"],
  },
});

// 存入多条不同主题的记忆
const memories = [
  { key: "fact-1", content: "用户偏好 TypeScript 编程语言", category: "tech" },
  { key: "fact-2", content: "用户使用 pnpm 作为包管理器", category: "tech" },
  { key: "fact-3", content: "用户住在上海浦东新区", category: "location" },
  { key: "fact-4", content: "用户喜欢深色主题的编辑器", category: "preference" },
  { key: "fact-5", content: "用户正在学习 LangGraph Agent 开发", category: "activity" },
  { key: "fact-6", content: "用户的宠物是一只名叫 Momo 的猫", category: "personal" },
  { key: "fact-7", content: "用户对 React 和 Vue 都有使用经验", category: "tech" },
];

for (const mem of memories) {
  await store.put(["users", "alice", "facts"], mem.key, {
    content: mem.content,
    category: mem.category,
  });
}
console.log(`\n✅ 写入 ${memories.length} 条记忆（每条自动生成 512 维向量）`);

// ── 语义检索：用自然语言查 ──
const queries = [
  "用户用什么语言写代码？",
  "用户家在哪里？",
  "有没有养宠物？",
  "前端框架经验",
];

for (const q of queries) {
  const results = await store.search(["users", "alice", "facts"], {
    query: q,
    limit: 3,
  });
  console.log(`\n🔍 Query: "${q}"`);
  for (const item of results) {
    const score = item.score?.toFixed(4) ?? "N/A";
    console.log(`   [${score}] ${item.value.content}`);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// 演示 3：语义检索 vs filter 精确匹配 —— 互补关系
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 3：语义检索 vs filter —— 各自擅长什么");
console.log("═".repeat(60));

// 语义检索：擅长模糊/自然语言查询
const semanticResults = await store.search(["users", "alice", "facts"], {
  query: "编程相关的偏好",
  limit: 3,
});
console.log(`\n[语义检索] "编程相关的偏好" → 找到语义相关的：`);
for (const item of semanticResults) {
  console.log(`  [${item.score?.toFixed(4)}] ${item.value.content} (${item.value.category})`);
}

// filter 精确匹配：擅长已知字段值的精确过滤
const filterResults = await store.search(["users", "alice", "facts"], {
  filter: { category: "tech" },
  limit: 10,
});
console.log(`\n[filter] category="tech" → 精确命中：`);
for (const item of filterResults) {
  console.log(`  ${item.value.content}`);
}

// 组合使用：语义 + filter
const combinedResults = await store.search(["users", "alice", "facts"], {
  query: "用什么工具",
  filter: { category: "tech" },
  limit: 3,
});
console.log(`\n[语义 + filter] query="用什么工具" AND category="tech" → 双重筛选：`);
for (const item of combinedResults) {
  console.log(`  [${item.score?.toFixed(4)}] ${item.value.content}`);
}

// ═══════════════════════════════════════════════════════════════════════════
// 演示 4：在 Graph 节点中用语义检索注入相关记忆
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 4：Graph 节点中语义检索 → 注入相关记忆到 prompt");
console.log("═".repeat(60));

/**
 * 完整流程：
 * 1. 用户发消息
 * 2. 节点用用户消息作为 query，从 Store 语义检索相关记忆
 * 3. 把检索到的记忆注入 system prompt
 * 4. LLM 基于记忆回答
 *
 * 这就是 Long-term Memory 在生产中的核心模式：
 * 不是把所有记忆都塞进 prompt，而是只检索和当前问题相关的。
 */

const graphStore = new InMemoryStore({
  index: { dims: 512, embeddings, fields: ["content"] },
});
const checkpointer = new MemorySaver();

// 预存用户记忆
for (const mem of memories) {
  await graphStore.put(["users", "alice", "facts"], mem.key, {
    content: mem.content,
    category: mem.category,
  });
}

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", async (state, config: LangGraphRunnableConfig) => {
    const userId = config.configurable?.user_id as string;
    const st = config.store!;

    // 用当前用户消息做语义检索
    const userMsg = state.messages.at(-1)!.content as string;
    const relevant = await st.search(["users", userId, "facts"], {
      query: userMsg,
      limit: 3,
    });

    let memoryContext = "";
    if (relevant.length > 0) {
      memoryContext = "\n\n## 用户相关记忆（来自 Long-term Store 语义检索）\n";
      for (const item of relevant) {
        memoryContext += `- ${item.value.content}（相关度: ${item.score?.toFixed(2)}）\n`;
      }
    }

    const resp = await llm.invoke([
      new SystemMessage(
        "你是一个简洁的助手，回答控制在一两句话内。" +
        "请参考用户的相关记忆来个性化回答。" +
        memoryContext
      ),
      ...state.messages,
    ]);

    return { messages: [resp] };
  })
  .addEdge(START, "agent")
  .addEdge("agent", END)
  .compile({ checkpointer, store: graphStore });

async function chatSemantic(userId: string, threadId: string, msg: string) {
  console.log(`\n[用户] ${msg}`);
  const result = await graph.invoke(
    { messages: [new HumanMessage(msg)] },
    { configurable: { thread_id: threadId, user_id: userId } },
  );
  const reply = result.messages.at(-1)!.content;
  console.log(`[AI]   ${reply}`);
}

// 测试：不同问题会检索到不同的记忆
await chatSemantic("alice", "t1", "帮我推荐一个前端框架");
await chatSemantic("alice", "t2", "附近有什么好吃的？");
await chatSemantic("alice", "t3", "我的猫应该打什么疫苗？");

// ═══════════════════════════════════════════════════════════════════════════
// 总结
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "─".repeat(60));
console.log("总结：Store + Embedding 语义检索");
console.log("─".repeat(60));
console.log(`
  Embedding = 把文本变成向量（一组数字），语义越近向量越接近

  Store 配置 IndexConfig 后的变化：
    put()   → 自动对指定 fields 生成向量并存储
    search({ query }) → 自动对 query 生成向量 → 余弦相似度排序

  IndexConfig 三个参数：
    dims:       向量维度（必须和 Embedding 模型一致）
    embeddings: LangChain Embeddings 实例
    fields:     value 里哪些字段参与 embedding（默认 ["$"] = 全部）

  语义检索 vs filter：
    语义检索  → 模糊/自然语言查询（"用户用什么语言？"）
    filter    → 已知字段的精确过滤（category = "tech"）
    组合使用  → 双重筛选，更精确

  当前局限：
    InMemoryStore 只支持向量检索，不支持 BM25
    → 精确实体名（如 ORD-123）会失败
    → 下一节（04-hybrid-retrieval）手写 BM25 + 向量 + RRF 融合来解决
`);
