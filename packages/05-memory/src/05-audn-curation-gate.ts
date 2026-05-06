/**
 * 05 - AUDN Curation Gate：写入仲裁
 *
 * ── 04 的延伸问题：Hybrid Retrieval 解决了"读"，但"写"怎么办？──────────
 *
 * 你已经能精准检索记忆了。但新信息是否该写进去，是另一个问题：
 *
 * 问题 1：重复写入
 *   用户第 1 次："我喜欢篮球"
 *   用户第 5 次："我也喜欢篮球嗯"
 *   → 如果每次都 store.put，同一条偏好存了两份，检索返回重复噪音。
 *
 * 问题 2：矛盾写入
 *   用户第 1 次："我住北京"
 *   用户第 3 次："我已经搬到上海了"
 *   → 如果只追加，Store 里同时存在"住北京"和"住上海"，LLM 看到两条会混乱。
 *
 * 问题 3：无效写入
 *   用户："今天天气不错"
 *   → 这是临时性信息，不属于稳定的 Semantic Memory，写进去污染检索结果。
 *
 * ── AUDN 四叉决策 ─────────────────────────────────────────────────────────
 *
 * 在每次写入前，让 LLM 做一个 structured output 决策：
 *
 * │ 决策     │ 含义                     │ 操作                        │
 * │ Add      │ 全新事实，以前没存过      │ store.put(...) 新增          │
 * │ Update   │ 更新了旧事实（如搬家）    │ store.put(...) 覆盖同 key    │
 * │ Delete   │ 明确撤销了某条事实        │ store.delete(...)            │
 * │ Noop     │ 临时/重复/无意义信息      │ 什么都不做                   │
 *
 * 结果：Semantic Memory 永远保持「最新、无矛盾、无噪音」的状态。
 *
 * ── 本文件演示的内容 ───────────────────────────────────────────────────────
 *
 * ── V1 局限：单条决策 ────────────────────────────────────────────────────
 *
 * 原始版本 Schema 只返回一个决策对象，遇到"我是 Bob，我是数据工程师"这种
 * 一句话包含多个事实的情况，LLM 只能选一个存，另一个丢掉。
 *
 * ── V2 改进：多条决策（本文件）────────────────────────────────────────────
 *
 * Schema 改为 decisions 数组，LLM 一次调用可返回多条决策。
 * 这是 Mem0 "memory extraction" 的核心思路：
 *   对话 → 提取所有事实列表 → 每条走 AUDN 仲裁 → 批量写入/更新/删除
 *
 * Demo 1：单条 Add — 验证基础功能
 * Demo 2：多条 Add — 一句话提取多个事实（V2 的核心改进）
 * Demo 3：Noop — 临时信息不写入
 * Demo 4：Update — 事实发生变化
 * Demo 5：Delete — 明确撤销某条事实
 * Demo 6：完整 Agent 循环 — 第一条消息同时提取 name + job_title
 */

import "dotenv/config";
import { InMemoryStore } from "@langchain/langgraph";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// ── LLM 配置（DeepSeek，用于 AUDN 决策 + Agent 对话）─────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  temperature: 0,
});

// ── Embedding 模型（阿里云百炼，用于语义检索）────────────────────────────
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-v4",
  dimensions: 512,
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1" },
});

// ── Store（配置 embedding，支持语义检索）────────────────────────────────
const store = new InMemoryStore({
  index: {
    dims: 512,
    embeddings,
  },
});

// ── AUDN 决策 Schema（V2：数组，支持一次多条）────────────────────────────
/**
 * V1 是单个对象 { action, key, value, existingKey, reason }
 * V2 改为 { decisions: [...] }，让 LLM 从一句话里提取所有事实并逐条仲裁。
 *
 * 空数组 = 全部 noop（该消息没有任何值得存的事实）
 */
const SingleDecisionSchema = z.object({
  action: z.enum(["add", "update", "delete", "noop"]),
  key: z.string().describe("新条目的 key（add 时必填，noop 时填空字符串）"),
  value: z.string().describe("事实内容（add/update 时必填，其他时填空字符串）"),
  existingKey: z.string().describe("要更新/删除的已有 key（update/delete 时必填，其他时填空字符串）"),
  reason: z.string().describe("决策理由"),
});

const MultiAUDNSchema = z.object({
  decisions: z
    .array(SingleDecisionSchema)
    .describe("从用户消息中提取的所有事实的 AUDN 决策列表，没有事实时返回空数组"),
});

type SingleDecision = z.infer<typeof SingleDecisionSchema>;

const audnLLM = llm.withStructuredOutput(MultiAUDNSchema, { method: "functionCalling" });

// ── 核心函数：执行 AUDN Gate（V2 多条）──────────────────────────────────
/**
 * 给定用户说的一句话 + 当前 Store 里已有的记忆，
 * 让 LLM 提取所有事实并逐条决策，批量写入 Store。
 *
 * @param userId    用户 ID（namespace 隔离）
 * @param userText  用户这轮说的话
 * @returns         所有决策列表（供调试查看）
 */
async function audnGate(userId: string, userText: string): Promise<SingleDecision[]> {
  const namespace = ["users", userId, "semantic"];

  // 1. 检索已有记忆（让 LLM 知道"已经存了什么"，判断重复/矛盾/新增）
  const existing = await store.search(namespace, { query: userText, limit: 5 });
  const existingText = existing.length > 0
    ? existing.map(item => `key="${item.key}": ${item.value.fact}`).join("\n")
    : "（当前无已存记忆）";

  // 2. 让 LLM 从消息里提取所有事实，逐条做 AUDN 决策
  const result = await audnLLM.invoke([
    new SystemMessage(`
你是用户记忆管理助手。从用户的消息中提取所有值得长期保存的个人事实，对每条事实做 AUDN 决策。

已有的用户记忆：
${existingText}

判断规则（对每条提取出的事实）：
- add：全新的、稳定的个人事实（偏好、身份、地址、职业、习惯等），当前记忆里没有
- update：与某条已有记忆矛盾（如搬家、换工作、改变偏好），需要更新
- delete：用户明确表示某个之前的事实不再成立
- noop：临时信息（天气、心情、今日发生的事）、打招呼、重复已有信息

一句话可以包含多个事实，请全部提取。没有任何事实时返回空数组。
key 命名规则：小写英文 + 下划线，如 city、job_title、favorite_language
    `.trim()),
    new HumanMessage(`用户说："${userText}"`),
  ]);

  const decisions = result.decisions;

  // 3. 批量执行 Store 操作
  for (const d of decisions) {
    if (d.action === "add") {
      await store.put(namespace, d.key, { fact: d.value });
    } else if (d.action === "update") {
      // 防御：先确认 existingKey 存在，不存在时降级为 add
      const existing = await store.get(namespace, d.existingKey);
      const targetKey = existing ? d.existingKey : d.key;
      await store.put(namespace, targetKey, { fact: d.value });
    } else if (d.action === "delete") {
      await store.delete(namespace, d.existingKey);
    }
    // noop：跳过
  }

  return decisions;
}

// ── 辅助：打印决策列表 ───────────────────────────────────────────────────
function printDecisions(decisions: SingleDecision[]) {
  if (decisions.length === 0) {
    console.log("  [AUDN] 无事实提取 → 全部 noop");
    return;
  }
  decisions.forEach((d, i) => {
    const target = d.action === "noop" ? "" :
      d.action === "delete" ? ` existingKey="${d.existingKey}"` :
      d.action === "update" ? ` existingKey="${d.existingKey}" → "${d.value}"` :
      ` key="${d.key}" = "${d.value}"`;
    console.log(`  [AUDN ${i + 1}] ${d.action}${target}  — ${d.reason}`);
  });
}

// ── 辅助：打印当前 Store 内容 ────────────────────────────────────────────
async function printStore(userId: string, label: string) {
  const namespace = ["users", userId, "semantic"];
  const all = await store.search(namespace, { limit: 20 });
  console.log(`\n[Store 快照] ${label}`);
  if (all.length === 0) {
    console.log("  （空）");
  } else {
    all.forEach(item => {
      console.log(`  key="${item.key}": ${item.value.fact}`);
    });
  }
}

// ── Demo 1：单条 Add ──────────────────────────────────────────────────────
async function demo1Add() {
  console.log("\n====== Demo 1：单条 Add ======");
  console.log('用户说："我最喜欢的编程语言是 TypeScript。"');

  const ds = await audnGate("alice", "我最喜欢的编程语言是 TypeScript。");
  printDecisions(ds);
  await printStore("alice", "Add 后");
}

// ── Demo 2：多条 Add（V2 核心改进）───────────────────────────────────────
async function demo2MultiAdd() {
  console.log("\n====== Demo 2：多条 Add — 一句话提取多个事实（V2 核心改进）======");
  console.log('用户说："我住在上海，从事前端开发工作，平时喜欢打羽毛球。"');

  // V1 只会存一条，V2 应该同时存 city + job_title + favorite_sport
  const ds = await audnGate("alice", "我住在上海，从事前端开发工作，平时喜欢打羽毛球。");
  printDecisions(ds);
  await printStore("alice", "多条 Add 后（应有 city + job_title + favorite_sport）");
}

// ── Demo 3：Noop 场景 ─────────────────────────────────────────────────────
async function demo3Noop() {
  console.log("\n====== Demo 3：Noop — 临时信息不写入 ======");
  console.log('用户说："今天天气真好，心情不错。"');

  const ds = await audnGate("alice", "今天天气真好，心情不错。");
  printDecisions(ds);
  await printStore("alice", "Noop 后（Store 应与上方相同）");
}

// ── Demo 4：Update 场景 ───────────────────────────────────────────────────
async function demo4Update() {
  console.log("\n====== Demo 4：Update — 事实发生变化（搬家）======");
  console.log('用户说："对了，我最近搬到北京了。"');

  const ds = await audnGate("alice", "对了，我最近搬到北京了。");
  printDecisions(ds);
  await printStore("alice", "Update 后（city 应变为北京）");
}

// ── Demo 5：Delete 场景 ───────────────────────────────────────────────────
async function demo5Delete() {
  console.log("\n====== Demo 5：Delete — 明确撤销某条事实 ======");
  await store.put(["users", "alice", "semantic"], "favorite_sport", { fact: "用户喜欢打羽毛球" });
  console.log('（手动写入：用户喜欢打羽毛球）');
  await printStore("alice", "Delete 前");

  console.log('用户说："我已经不打羽毛球了，改练游泳了。"');
  // 这句话同时触发：delete(favorite_sport) + add(favorite_sport=游泳)
  const ds = await audnGate("alice", "我已经不打羽毛球了，改练游泳了。");
  printDecisions(ds);
  await printStore("alice", "Delete+Add 后（羽毛球消失，游泳出现）");
}

// ── Demo 6：完整 Agent 循环（验证第一条消息同时提取 name + job_title）─────
async function demo6FullLoop() {
  console.log("\n====== Demo 6：完整 Agent 循环（验证多事实提取）======");

  const userId = "bob";
  const turns = [
    "你好，我是 Bob，我是一名数据工程师。",   // V1 只存了 name，V2 应同时存 name + job_title
    "我特别喜欢用 Python 做数据处理。",
    "最近在学 Rust，感觉语法好难。",
    "今天午饭吃了个麻辣烫，有点辣。",
    "对了，我换工作了，现在做 AI 工程师。",
  ];

  for (const userText of turns) {
    console.log(`\n--- 用户: ${userText}`);

    const namespace = ["users", userId, "semantic"];
    const memories = await store.search(namespace, { query: userText, limit: 3 });
    const memoryContext = memories.length > 0
      ? `关于用户的已知信息：\n${memories.map(m => `- ${m.value.fact}`).join("\n")}`
      : "暂无用户记忆。";

    const response = await llm.invoke([
      new SystemMessage(`你是一个友好的助手。${memoryContext}`),
      new HumanMessage(userText),
    ]);
    console.log(`助手: ${(response.content as string).slice(0, 80)}...`);

    const decisions = await audnGate(userId, userText);
    printDecisions(decisions);
  }

  await printStore(userId, "5 轮对话后（第一条消息应同时存 name + job_title）");
}

// ── 主入口 ────────────────────────────────────────────────────────────────
async function main() {
  await demo1Add();
  await demo2MultiAdd();
  await demo3Noop();
  await demo4Update();
  await demo5Delete();
  await demo6FullLoop();
}

main().catch(console.error);
