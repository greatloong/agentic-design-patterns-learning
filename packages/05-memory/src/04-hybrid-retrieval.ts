/**
 * 04 - Hybrid Retrieval：BM25 + Vector + RRF
 *
 * ── 03 的天花板：纯向量的两个已知失败模式 ──────────────────────────────
 *
 * 失败 1：精确实体名
 *   用户问 "订单 ORD-123 什么情况？"
 *   向量检索把 ORD-123 / ORD-456 / ORD-789 都编码成"订单号"语义簇 → 区分不了
 *
 * 失败 2：技术术语
 *   用户问 "PostgreSQL 配置怎么改"
 *   向量检索可能把"数据库设置教程"排在真正写了 "PostgreSQL" 的条目前面
 *
 * ── 解法：Hybrid = BM25 + Vector + RRF ───────────────────────────────────
 *
 * BM25（关键词匹配）  →  擅长精确实体名 / 技术术语
 * Vector（语义检索） →  擅长自然语言 / 意图理解
 * RRF（融合算法）    →  把两路排名合并成一个排名
 *
 * 业界一致测出：Hybrid 比纯向量 recall 高 15-30%。
 * Mem0 / Zep / Elasticsearch 等生产系统全部内置这个组合。
 *
 * ── BM25 原理（不用背，理解直觉就行）────────────────────────────────────
 *
 * BM25 = 关键词倒排索引 + TF-IDF 的改进版
 *
 * 核心思路：
 * 1. 把每条文档分词（"用户 住 上海"）
 * 2. 建倒排索引：每个词 → 出现在哪些文档
 * 3. 查询时，query 分词 → 每个词在文档里的词频 × 稀有度 → 加权求和
 *
 * 为什么能精确匹配 ORD-123？
 * 因为 "ORD-123" 作为一个 token，只出现在包含它的文档里。
 * 向量把它 encode 成语义，BM25 直接字符串比对。
 *
 * ── RRF（Reciprocal Rank Fusion）原理 ──────────────────────────────────
 *
 * 最朴素的融合算法，公式：
 *   RRF(d) = Σ  1 / (k + rank_i(d))
 *
 * 其中 k=60（经验常数），rank_i(d) 是文档 d 在第 i 路排名里的位置。
 * 直觉：在每条路里都排名靠前的文档，最终得分最高。
 *
 * ── 本节演示 ──────────────────────────────────────────────────────────────
 * 1. 纯向量的失败案例（复现问题）
 * 2. 手写 BM25 实现（理解原理）
 * 3. 手写 RRF 融合（理解原理）
 * 4. Hybrid vs 纯向量 对比（直观看差异）
 * 5. 封装成生产可用的 HybridMemoryStore
 */

import "dotenv/config";
import { InMemoryStore } from "@langchain/langgraph";
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-v4",
  dimensions: 512,
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: {
    baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  },
});

// 测试数据：电商客服场景，混合了订单号（精确实体）+ 用户偏好（语义）
const memories = [
  { key: "m1", content: "用户在 2026-04-10 下单 ORD-123，购买了 AirPods Pro，物流延迟两天" },
  { key: "m2", content: "用户对订单 ORD-456 的红色连衣裙表示满意，给了五星好评" },
  { key: "m3", content: "用户投诉 ORD-789 的蓝牙耳机有质量问题，申请退款" },
  { key: "m4", content: "用户偏好运动风格服装，特别喜欢 Nike 和 Adidas" },
  { key: "m5", content: "用户经常在大促期间下单，对折扣价格很敏感" },
  { key: "m6", content: "用户的收货地址在上海浦东新区，使用顺丰快递" },
  { key: "m7", content: "用户反映 App 在 iOS 16 上有时会崩溃，影响下单" },
];

// ═══════════════════════════════════════════════════════════════════════════
// 演示 1：纯向量的失败案例（复现问题）
// ═══════════════════════════════════════════════════════════════════════════
console.log("═".repeat(60));
console.log("演示 1：纯向量检索的失败案例");
console.log("═".repeat(60));

const vectorOnlyStore = new InMemoryStore({
  index: { dims: 512, embeddings, fields: ["content"] },
});
for (const m of memories) {
  await vectorOnlyStore.put(["memories"], m.key, { content: m.content });
}

// 失败案例 1：精确订单号查询
const q1 = "ORD-123 这个订单现在什么情况？";
const vectorResults1 = await vectorOnlyStore.search(["memories"], {
  query: q1,
  limit: 4,
});
console.log(`\n🔍 Query: "${q1}"`);
console.log("纯向量结果：");
for (const item of vectorResults1) {
  const hit = item.value.content.includes("ORD-123") ? "✅ 命中" : "❌ 误召回";
  console.log(`  [${item.score?.toFixed(3)}] ${hit} ${item.value.content.slice(0, 50)}...`);
}

// 失败案例 2：技术术语
const q2 = "iOS 16 的 bug 有没有解决";
const vectorResults2 = await vectorOnlyStore.search(["memories"], {
  query: q2,
  limit: 4,
});
console.log(`\n🔍 Query: "${q2}"`);
console.log("纯向量结果：");
for (const item of vectorResults2) {
  const hit = item.value.content.includes("iOS 16") ? "✅ 命中" : "❌ 误召回";
  console.log(`  [${item.score?.toFixed(3)}] ${hit} ${item.value.content.slice(0, 50)}...`);
}

// ═══════════════════════════════════════════════════════════════════════════
// BM25 实现
// ═══════════════════════════════════════════════════════════════════════════

/**
 * 分词器：中文 + 英文混合处理
 * 生产里用 jieba（中文）+ 自定义分词器，这里简化处理
 * 核心逻辑不变：把文本切成 token 数组
 */
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    // 中英文 token 分割（按非字母数字分割，保留中文字符）
    .split(/[\s，。！？、：；""''（）【】\-_]+/)
    .filter((t) => t.length > 0);
}

/**
 * BM25 实现
 *
 * 参数：
 *   k1 = 1.5  调节词频饱和度（词出现 10 次 vs 1 次，贡献差距有限）
 *   b  = 0.75 调节文档长度归一化（长文档不因词多就占便宜）
 */
class BM25 {
  private k1 = 1.5;
  private b = 0.75;
  private docs: { key: string; content: string; tokens: string[] }[] = [];
  private df = new Map<string, number>(); // document frequency：token 出现在几篇文档里
  private avgDocLen = 0;

  add(key: string, content: string) {
    const tokens = tokenize(content);
    this.docs.push({ key, content, tokens });

    // 更新倒排索引（df）
    const seen = new Set<string>();
    for (const t of tokens) {
      if (!seen.has(t)) {
        this.df.set(t, (this.df.get(t) ?? 0) + 1);
        seen.add(t);
      }
    }

    // 更新平均文档长度
    this.avgDocLen =
      this.docs.reduce((sum, d) => sum + d.tokens.length, 0) / this.docs.length;
  }

  search(query: string, topK: number): { key: string; content: string; score: number }[] {
    const queryTokens = tokenize(query);
    const N = this.docs.length;
    const scores: { key: string; content: string; score: number }[] = [];

    for (const doc of this.docs) {
      let score = 0;

      // 计算文档内 token 频率
      const tf = new Map<string, number>();
      for (const t of doc.tokens) tf.set(t, (tf.get(t) ?? 0) + 1);

      for (const qt of queryTokens) {
        const tfVal = tf.get(qt) ?? 0;
        if (tfVal === 0) continue; // 这个词在文档里不存在，跳过

        const dfVal = this.df.get(qt) ?? 0;
        // IDF（逆文档频率）：词越稀有，分越高
        const idf = Math.log((N - dfVal + 0.5) / (dfVal + 0.5) + 1);
        // TF 部分：词频贡献有上限（k1 调节饱和）+ 文档长度归一化（b 调节）
        const tfNorm =
          (tfVal * (this.k1 + 1)) /
          (tfVal + this.k1 * (1 - this.b + this.b * (doc.tokens.length / this.avgDocLen)));

        score += idf * tfNorm;
      }

      scores.push({ key: doc.key, content: doc.content, score });
    }

    return scores.sort((a, b) => b.score - a.score).slice(0, topK);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// RRF 实现
// ═══════════════════════════════════════════════════════════════════════════

/**
 * RRF（Reciprocal Rank Fusion）
 *
 * 公式：score(d) = Σ 1 / (k + rank_i(d))
 * k = 60 是经验值，防止排名第 1 的文档分数过于压制其他文档
 *
 * 直觉：在多路检索里，每路都排名靠前的文档，最终得分最高。
 */
function rrfFuse(
  rankedLists: { key: string; content: string }[][],
  k = 60
): { key: string; content: string; score: number }[] {
  const scores = new Map<string, { content: string; score: number }>();

  for (const list of rankedLists) {
    list.forEach((item, idx) => {
      const rank = idx + 1; // 1-indexed
      const contribution = 1 / (k + rank);
      const existing = scores.get(item.key);
      if (existing) {
        existing.score += contribution;
      } else {
        scores.set(item.key, { content: item.content, score: contribution });
      }
    });
  }

  return Array.from(scores.entries())
    .map(([key, v]) => ({ key, content: v.content, score: v.score }))
    .sort((a, b) => b.score - a.score);
}

// ═══════════════════════════════════════════════════════════════════════════
// 演示 2：BM25 单独检索（看精确匹配效果）
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 2：BM25 单独检索（关键词精确匹配）");
console.log("═".repeat(60));

const bm25 = new BM25();
for (const m of memories) bm25.add(m.key, m.content);

const bm25Results1 = bm25.search(q1, 4);
console.log(`\n🔍 Query: "${q1}"`);
console.log("BM25 结果：");
for (const item of bm25Results1) {
  const hit = item.content.includes("ORD-123") ? "✅ 命中" : "  ";
  console.log(`  [${item.score.toFixed(3)}] ${hit} ${item.content.slice(0, 50)}...`);
}

const bm25Results2 = bm25.search(q2, 4);
console.log(`\n🔍 Query: "${q2}"`);
console.log("BM25 结果：");
for (const item of bm25Results2) {
  const hit = item.content.includes("iOS 16") ? "✅ 命中" : "  ";
  console.log(`  [${item.score.toFixed(3)}] ${hit} ${item.content.slice(0, 50)}...`);
}

// ═══════════════════════════════════════════════════════════════════════════
// 演示 3：Hybrid 检索（BM25 + Vector + RRF）
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 3：Hybrid 检索（BM25 + Vector + RRF 融合）");
console.log("═".repeat(60));

async function hybridSearch(
  query: string,
  topK: number
): Promise<{ key: string; content: string; score: number }[]> {
  // 路1：向量检索（语义）
  const vectorRaw = await vectorOnlyStore.search(["memories"], {
    query,
    limit: topK * 2, // 取更多候选，融合后再截断
  });
  const vectorList = vectorRaw.map((item) => ({
    key: item.key,
    content: item.value.content as string,
  }));

  // 路2：BM25 检索（关键词）
  const bm25Raw = bm25.search(query, topK * 2);
  const bm25List = bm25Raw.map((item) => ({
    key: item.key,
    content: item.content,
  }));

  // RRF 融合两路结果
  const fused = rrfFuse([vectorList, bm25List]);
  return fused.slice(0, topK);
}

// 对比两个 query 的三种检索结果
for (const q of [q1, q2]) {
  const hybrid = await hybridSearch(q, 4);
  const vectorOnly = await vectorOnlyStore.search(["memories"], { query: q, limit: 4 });
  const bm25Only = bm25.search(q, 4);

  const targetKeyword = q.includes("ORD-123") ? "ORD-123" : "iOS 16";

  const hybridRank = hybrid.findIndex((i) => i.content.includes(targetKeyword)) + 1;
  const vectorRank = vectorOnly.findIndex((i) => (i.value.content as string).includes(targetKeyword)) + 1;
  const bm25Rank = bm25Only.findIndex((i) => i.content.includes(targetKeyword)) + 1;

  console.log(`\n🔍 Query: "${q}"`);
  console.log(`  目标 "${targetKeyword}" 的排名：`);
  console.log(`  纯向量:  第 ${vectorRank || ">4"} 名`);
  console.log(`  纯BM25:  第 ${bm25Rank || ">4"} 名`);
  console.log(`  Hybrid:  第 ${hybridRank || ">4"} 名  ← RRF 融合后`);
}

// ═══════════════════════════════════════════════════════════════════════════
// 演示 4：语义查询场景（Hybrid 不差于纯向量）
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 4：语义查询场景（Hybrid 不退化）");
console.log("═".repeat(60));

const semanticQuery = "用户喜欢买什么类型的衣服？";
const hybridSemantic = await hybridSearch(semanticQuery, 3);
const vectorSemantic = await vectorOnlyStore.search(["memories"], {
  query: semanticQuery,
  limit: 3,
});

console.log(`\n🔍 Query: "${semanticQuery}"`);
console.log("\n纯向量 top3：");
for (const item of vectorSemantic) {
  console.log(`  [${item.score?.toFixed(3)}] ${item.value.content.slice(0, 55)}...`);
}
console.log("\nHybrid top3：");
for (const item of hybridSemantic) {
  console.log(`  [${item.score.toFixed(4)}] ${item.content.slice(0, 55)}...`);
}

// ═══════════════════════════════════════════════════════════════════════════
// 演示 5：封装成 HybridMemoryStore（生产可用的工具类）
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 5：封装 HybridMemoryStore（生产工具类）");
console.log("═".repeat(60));

/**
 * 生产级 HybridMemoryStore
 *
 * 封装了：
 * - InMemoryStore（向量检索）
 * - BM25（关键词检索）
 * - RRF（融合）
 *
 * 对外暴露和 Store 一致的 put / search 接口，
 * 让上层代码无感知地享受 Hybrid 能力。
 *
 * 生产替换：把 InMemoryStore 换成 PostgresStore 即可，
 * BM25 换成 Elasticsearch / pgvector + tsvector 全文检索。
 */
class HybridMemoryStore {
  private vectorStore: InMemoryStore;
  private bm25Index: BM25;

  constructor(emb: OpenAIEmbeddings) {
    this.vectorStore = new InMemoryStore({
      index: { dims: 512, embeddings: emb, fields: ["content"] },
    });
    this.bm25Index = new BM25();
  }

  async put(namespace: string[], key: string, value: Record<string, any>) {
    // 同时写入两个索引
    await this.vectorStore.put(namespace, key, value);
    this.bm25Index.add(key, value.content ?? JSON.stringify(value));
  }

  async search(
    namespace: string[],
    opts: { query: string; limit?: number; filter?: Record<string, any> }
  ) {
    const topK = opts.limit ?? 5;

    // 路1：向量
    const vectorRaw = await this.vectorStore.search(namespace, {
      query: opts.query,
      limit: topK * 2,
      filter: opts.filter,
    });
    const vectorList = vectorRaw.map((i) => ({ key: i.key, content: i.value.content as string }));

    // 路2：BM25
    const bm25Raw = this.bm25Index.search(opts.query, topK * 2);
    const bm25List = bm25Raw.map((i) => ({ key: i.key, content: i.content }));

    // RRF 融合
    return rrfFuse([vectorList, bm25List]).slice(0, topK);
  }
}

const hybridStore = new HybridMemoryStore(embeddings);
for (const m of memories) {
  await hybridStore.put(["memories"], m.key, { content: m.content });
}

// 验证：精确实体 + 语义查询都正确
const tests = [
  { query: "ORD-123 订单进展", target: "ORD-123" },
  { query: "退款申请", target: "ORD-789" },
  { query: "用户喜欢什么运动品牌", target: "Nike" },
];

for (const t of tests) {
  const results = await hybridStore.search(["memories"], { query: t.query, limit: 3 });
  const topHit = results[0]?.content ?? "";
  const isCorrect = topHit.includes(t.target) ? "✅" : "❌";
  console.log(`\n  ${isCorrect} Query: "${t.query}"`);
  console.log(`     期望命中: "${t.target}"`);
  console.log(`     实际 top1: ${topHit.slice(0, 60)}...`);
}

// ═══════════════════════════════════════════════════════════════════════════
// 总结
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "─".repeat(60));
console.log("总结：Hybrid Retrieval");
console.log("─".repeat(60));
console.log(`
  为什么 Hybrid 比纯向量好 15-30%？
    纯向量：把 ORD-123/456/789 编码为"订单号"语义簇，区分不了
    BM25：  字符串比对，ORD-123 只会命中包含 "ORD-123" 的文档
    RRF：   两路都排名靠前的文档，最终胜出

  三个组件的分工：
    BM25   → 精确实体名 / 技术术语 / 产品型号
    Vector → 自然语言意图 / 语义相近的模糊查询
    RRF    → 1/(k+rank) 求和，防止单路垄断结果

  生产实现：
    BM25   → Elasticsearch / Postgres tsvector / Tantivy
    Vector → pgvector / Pinecone / Qdrant / Milvus
    RRF    → 应用层手写（简单） / Elasticsearch 内置

  下一节（05-audn-curation-gate）：
    解决"写入时矛盾累积"问题 → LLM 仲裁四叉决策
`);
