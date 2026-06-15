/**
 * 06 - Hot/Cold Pipeline：读写分离
 *
 * ── 05 的遗留问题：AUDN Gate 阻塞了用户响应 ─────────────────────────────
 *
 * 05-Demo6 里的完整循环是串行的：
 *   检索记忆 → 生成回复 → AUDN 仲裁 → 写入 Store → 返回
 *                                    ↑ 用户在等这一段
 *
 * AUDN 写入的结果"下次对话"才用得到，当前对话不依赖它。
 * 串行执行让用户白白多等 2-5 秒（AUDN LLM 调用时间）。
 *
 * ── Hot/Cold 分离的核心思想 ──────────────────────────────────────────────
 *
 * Hot Path（同步，用户等待）：
 *   检索记忆 → 注入 prompt → LLM 生成回复 → 立即返回
 *
 * Cold Path（异步，用户不等）：
 *   写入任务推入队列 → Worker 按顺序逐条处理（AUDN + Store 写入）
 *
 * ── 为什么需要队列串行化，而不是简单的 Promise 并发？──────────────────────
 *
 * 用户连发两条：
 *   T=0s: "我住北京"   → AUDN 写 city=北京
 *   T=1s: "不对，住上海" → AUDN 写 city=上海
 *
 * 无序并发可能后发先至 → 最终 Store 里是旧的"北京"。
 * 队列保证：先来先处理，写入顺序 = 消息顺序。
 *
 * ── 生产架构对比 ────────────────────────────────────────────────────────
 *
 * │ 规模         │ 队列方案                         │ 特点            │
 * │ 单实例/原型  │ 内存队列（Promise chain）          │ 本文件演示       │
 * │ 多实例       │ Redis Streams / Bull             │ 跨进程共享       │
 * │ 大规模       │ Kafka / SQS                      │ 持久化、削峰填谷 │
 * │ 独立服务     │ Mem0 Cloud / Zep Cloud（HTTP API）│ Agent 完全解耦   │
 *
 * ── 本文件演示 ──────────────────────────────────────────────────────────
 *
 * 1. ColdPathQueue 类：最小串行队列实现
 * 2. Agent 循环：Hot Path 立即返回，Cold Path 异步写入
 * 3. 并发写入验证：快速连发消息，验证队列保证顺序
 * 4. 对比：不用队列的并发 race condition 演示
 */

import "dotenv/config";
import { InMemoryStore } from "@langchain/langgraph";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// ── 基础配置 ────────────────────────────────────────────────────────────

const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
  temperature: 0,
});

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-v4",
  dimensions: 512,
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1" },
});

const store = new InMemoryStore({
  index: { dims: 512, embeddings },
});

// ── AUDN Schema（复用 05 的多条决策版本）─────────────────────────────────

const MultiAUDNSchema = z.object({
  decisions: z.array(z.object({
    action: z.enum(["add", "update", "delete", "noop"]),
    key: z.string(),
    value: z.string(),
    existingKey: z.string(),
    reason: z.string(),
  })),
});

const audnLLM = llm.withStructuredOutput(MultiAUDNSchema, { method: "functionCalling" });

// ══════════════════════════════════════════════════════════════════════════
// Cold Path Queue — 最小串行队列
// ══════════════════════════════════════════════════════════════════════════

type ColdTask = {
  userId: string;
  userText: string;
  enqueuedAt: number;
};

/**
 * 核心数据结构：串行 Promise 链
 *
 * 每个新任务 .then() 到链尾，保证：
 * 1. 严格 FIFO 顺序
 * 2. 前一个完成后才开始下一个
 * 3. 单个任务失败不阻塞后续（catch 兜底）
 *
 * 生产里换成 Redis + BullMQ 或 Kafka Consumer，接口不变。
 */
class ColdPathQueue {
  private chain: Promise<void> = Promise.resolve();
  private pending = 0;
  private processed = 0;

  enqueue(task: ColdTask): void {
    this.pending++;
    this.chain = this.chain
      .then(() => this.processTask(task))
      .catch((err) => {
        console.error(`  [Cold] ❌ 任务失败（${task.userText.slice(0, 20)}...）: ${err.message}`);
      })
      .finally(() => {
        this.pending--;
        this.processed++;
      });
  }

  private async processTask(task: ColdTask): Promise<void> {
    const start = Date.now();
    const namespace = ["users", task.userId, "semantic"];

    // Step 1: 语义检索已有记忆
    const existing = await store.search(namespace, { query: task.userText, limit: 5 });
    const existingText = existing.length > 0
      ? existing.map(item => `key="${item.key}": ${item.value.fact}`).join("\n")
      : "（当前无已存记忆）";

    // Step 2: LLM AUDN 仲裁
    const result = await audnLLM.invoke([
      new SystemMessage(`
你是用户记忆管理助手。从用户消息中提取所有值得长期保存的个人事实，对每条做 AUDN 决策。

已有用户记忆：
${existingText}

规则：
- add：全新稳定事实，当前没有
- update：与已有记忆矛盾，需覆盖（同一语义槽优先 update）
- delete：明确撤销
- noop：临时信息/重复

key 命名：小写英文 + 下划线
      `.trim()),
      new HumanMessage(`用户说："${task.userText}"`),
    ]);

    // Step 3: 执行 Store 操作
    for (const d of result.decisions) {
      if (d.action === "add") {
        await store.put(namespace, d.key, { fact: d.value });
      } else if (d.action === "update") {
        const key = d.existingKey || d.key;
        await store.put(namespace, key, { fact: d.value });
      } else if (d.action === "delete") {
        await store.delete(namespace, d.existingKey);
      }
    }

    const elapsed = Date.now() - start;
    const actions = result.decisions
      .filter(d => d.action !== "noop")
      .map(d => `${d.action}(${d.key || d.existingKey})`)
      .join(", ");
    console.log(`  [Cold] ✅ 处理完成（${elapsed}ms）${actions || "noop"} — "${task.userText.slice(0, 25)}"`);
  }

  /** 等待队列清空（测试/演示用，生产里队列常驻运行） */
  async drain(): Promise<void> {
    await this.chain;
  }

  get stats() {
    return { pending: this.pending, processed: this.processed };
  }
}

// ══════════════════════════════════════════════════════════════════════════
// Hot Path — Agent 响应（同步，用户等待的部分）
// ══════════════════════════════════════════════════════════════════════════

async function hotPath(userId: string, userText: string): Promise<string> {
  const namespace = ["users", userId, "semantic"];

  // 检索相关记忆注入 prompt
  const memories = await store.search(namespace, { query: userText, limit: 3 });
  const memoryContext = memories.length > 0
    ? `关于用户的已知信息：\n${memories.map(m => `- ${m.value.fact}`).join("\n")}`
    : "暂无用户记忆。";

  const response = await llm.invoke([
    new SystemMessage(`你是一个友好的助手。用一两句话简短回复。${memoryContext}`),
    new HumanMessage(userText),
  ]);

  return response.content as string;
}

// ══════════════════════════════════════════════════════════════════════════
// Demo 1：标准 Hot/Cold 分离循环
// ══════════════════════════════════════════════════════════════════════════

async function demo1StandardLoop() {
  console.log("\n====== Demo 1：Hot/Cold 分离 — 标准循环 ======\n");

  const queue = new ColdPathQueue();
  const userId = "charlie";

  const turns = [
    "你好，我叫 Charlie，在字节跳动做后端开发。",
    "我最近在研究 Go 微服务架构。",
    "今天北京雾霾好严重。",
    "对了，我下个月要转岗去 AI 部门了。",
  ];

  for (const userText of turns) {
    console.log(`[用户] ${userText}`);

    // ─── Hot Path（同步）───
    const hotStart = Date.now();
    const reply = await hotPath(userId, userText);
    const hotTime = Date.now() - hotStart;
    console.log(`[助手] ${reply.slice(0, 80)}... (${hotTime}ms)`);

    // ─── Cold Path（异步，不 await）───
    queue.enqueue({ userId, userText, enqueuedAt: Date.now() });
    console.log(`  [Cold] 📥 已入队，队列深度=${queue.stats.pending}\n`);
  }

  console.log("[主线程] 所有回复已返回，等待 Cold Path 队列清空...\n");
  await queue.drain();

  // 验证最终 Store 状态
  const all = await store.search(["users", userId, "semantic"], { limit: 20 });
  console.log("\n[Store 最终快照]");
  all.forEach(item => console.log(`  key="${item.key}": ${item.value.fact}`));
}

// ══════════════════════════════════════════════════════════════════════════
// Demo 2：并发写入顺序验证（快速连发矛盾消息）
// ══════════════════════════════════════════════════════════════════════════

async function demo2OrderGuarantee() {
  console.log("\n\n====== Demo 2：队列串行化 — 保证写入顺序 ======\n");

  const queue = new ColdPathQueue();
  const userId = "dave";

  // 模拟用户极快连发 3 条矛盾消息（同一个 key 被反复修改）
  const rapidMessages = [
    "我住在北京。",
    "不对，我住在上海。",
    "再更正一下，我其实住在深圳。",
  ];

  console.log("[模拟] 用户在 1 秒内连发 3 条矛盾消息（不等回复）\n");

  // 全部瞬间入队（模拟并发）
  for (const msg of rapidMessages) {
    console.log(`  📥 入队: "${msg}"`);
    queue.enqueue({ userId, userText: msg, enqueuedAt: Date.now() });
  }

  console.log(`\n[队列] 共 ${queue.stats.pending} 个任务待处理，开始串行消费...\n`);
  await queue.drain();

  // 验证最终结果应该是"深圳"（最后一条）
  const all = await store.search(["users", userId, "semantic"], { limit: 20 });
  console.log("\n[Store 最终快照]（应为「深圳」——最后一条消息的值）");
  all.forEach(item => console.log(`  key="${item.key}": ${item.value.fact}`));

  const cityItem = all.find(item => item.key === "city");
  if (cityItem && (cityItem.value as { fact: string }).fact.includes("深圳")) {
    console.log("\n✅ 验证通过：队列保证了写入顺序，最终状态正确。");
  } else {
    console.log("\n⚠️ 注意：最终状态可能受 LLM 仲裁影响，检查 key 命名。");
  }
}

// ══════════════════════════════════════════════════════════════════════════
// Demo 3：对比 — 无队列并发写入（展示 race condition）
// ══════════════════════════════════════════════════════════════════════════

async function demo3RaceCondition() {
  console.log("\n\n====== Demo 3：无队列并发 — Race Condition 演示 ======\n");

  // 用一个独立 store 演示
  const unsafeStore = new InMemoryStore({ index: { dims: 512, embeddings } });
  const namespace = ["users", "eve", "semantic"];

  // 模拟并发写入：两个 Promise 同时跑，不等彼此
  console.log("[模拟] 同时发起两个写入（不保证顺序）:");
  console.log('  写入 A: city = "北京"');
  console.log('  写入 B: city = "深圳"（应该是最终值）\n');

  // 给 A 人为加 100ms 延迟模拟网络抖动
  const writeA = async () => {
    await new Promise(r => setTimeout(r, 100)); // 模拟慢写入
    await unsafeStore.put(namespace, "city", { fact: "北京" });
    console.log("  写入 A 完成: city=北京");
  };

  const writeB = async () => {
    await unsafeStore.put(namespace, "city", { fact: "深圳" });
    console.log("  写入 B 完成: city=深圳");
  };

  // 并发执行（B 先完成，A 后完成 → 覆盖了正确值）
  await Promise.all([writeA(), writeB()]);

  const result = await unsafeStore.get(namespace, "city");
  const finalCity = (result?.value as { fact: string })?.fact;
  console.log(`\n[结果] Store 中 city = "${finalCity}"`);

  if (finalCity === "北京") {
    console.log("❌ Race Condition 发生！后发的 A 覆盖了正确的 B。");
    console.log("   → 这就是为什么需要队列串行化。");
  } else {
    console.log("✅ 这次碰巧顺序对了，但无保证（多跑几次就会出问题）。");
  }
}

// ── 主入口 ────────────────────────────────────────────────────────────────

async function main() {
  await demo1StandardLoop();
  await demo2OrderGuarantee();
  await demo3RaceCondition();
}

main().catch(console.error);
