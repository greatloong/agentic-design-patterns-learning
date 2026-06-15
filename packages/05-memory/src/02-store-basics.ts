/**
 * 02 - Store API 基础（Long-term Memory 入门）
 *
 * ── 前情回顾 ─────────────────────────────────────────────────────────────
 * 01 里学了 Checkpointer = Short-term Memory：
 *   - 作用域：单个 thread
 *   - 天花板：Alice 的 thread 永远不知道 Bob 说了什么
 *
 * ── 问题 ──────────────────────────────────────────────────────────────────
 * 用户今天告诉 Cursor "我用 pnpm"，明天新对话又给了 npm install。
 * 因为新 thread 没有旧 thread 的 Checkpointer 数据。
 *
 * 要解决"跨 thread 共享知识"，就需要一个独立于 thread 的存储层 → Store API。
 *
 * ── Store vs Checkpointer ─────────────────────────────────────────────────
 *
 *   Checkpointer                        Store
 *   ─────────────────                   ─────────────────
 *   隔离维度：thread_id                  隔离维度：namespace（自定义层级）
 *   数据粒度：整个 State 快照             数据粒度：单条 key-value Item
 *   生命周期：thread 存在期间              生命周期：永久（直到主动删除）
 *   典型用途：对话历史                    典型用途：用户偏好、知识事实
 *   类比：    浏览器标签页的会话            类比：    用户档案数据库
 *
 * ── 本节演示 ──────────────────────────────────────────────────────────────
 * 1. Store 基础操作：put / get / delete
 * 2. Namespace 层级设计：多租户隔离
 * 3. search + filter：按条件检索
 * 4. listNamespaces：探索数据组织结构
 * 5. 在 Graph 节点中读写 Store（通过 config.store）
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
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import type { LangGraphRunnableConfig } from "@langchain/langgraph";

const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

// ═══════════════════════════════════════════════════════════════════════════
// 演示 1：Store 基础 CRUD —— put / get / delete
// ═══════════════════════════════════════════════════════════════════════════
console.log("═".repeat(60));
console.log("演示 1：Store 基础 CRUD");
console.log("═".repeat(60));

const store = new InMemoryStore();

// ── put：写入一条 Item ──
// namespace = ["users", "alice"]  → 类似文件夹路径 users/alice/
// key = "preferences"             → 文件名
// value = { ... }                 → 文件内容（必须是 JSON-serializable 对象）
await store.put(["users", "alice"], "preferences", {
  language: "TypeScript",
  packageManager: "pnpm",
  editor: "Cursor",
});

await store.put(["users", "alice"], "profile", {
  name: "Alice",
  role: "全栈工程师",
  learning: "LangGraph Agent 开发",
});

await store.put(["users", "bob"], "preferences", {
  language: "Rust",
  packageManager: "cargo",
  editor: "Neovim",
});

console.log("\n✅ 写入了 3 条 Item");

// ── get：精确读取 ──
const alicePrefs = await store.get(["users", "alice"], "preferences");
console.log("\n[get] Alice 的偏好：");
console.log(`  key: ${alicePrefs?.key}`);
console.log(`  namespace: ${alicePrefs?.namespace.join("/")}`);
console.log(`  value:`, alicePrefs?.value);
console.log(`  createdAt: ${alicePrefs?.createdAt}`);

// ── get 不存在的 key → 返回 null ──
const ghost = await store.get(["users", "alice"], "nonexistent");
console.log(`\n[get] 不存在的 key: ${ghost}`); // null

// ── delete：删除 ──
await store.put(["users", "alice"], "temp", { data: "临时数据" });
console.log(`\n[delete] 写入 temp → 读到:`, (await store.get(["users", "alice"], "temp"))?.value);
await store.delete(["users", "alice"], "temp");
console.log(`[delete] 删除 temp → 读到:`, await store.get(["users", "alice"], "temp")); // null

// ═══════════════════════════════════════════════════════════════════════════
// 演示 2：Namespace 设计 —— 多租户隔离的关键
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 2：Namespace 层级设计");
console.log("═".repeat(60));

/**
 * Namespace 设计思路（类比文件系统）：
 *
 *   store/
 *   ├── users/
 *   │   ├── alice/
 *   │   │   ├── preferences    ← ["users", "alice"] + "preferences"
 *   │   │   └── profile        ← ["users", "alice"] + "profile"
 *   │   └── bob/
 *   │       └── preferences    ← ["users", "bob"] + "preferences"
 *   └── teams/
 *       └── frontend/
 *           └── conventions    ← ["teams", "frontend"] + "conventions"
 *
 * 生产最佳实践：
 * - 第一层：数据域（users / teams / projects）
 * - 第二层：租户 ID（user_id / team_id）
 * - 第三层（可选）：子类型（preferences / facts / history）
 * - key：具体条目的唯一标识
 */

await store.put(["teams", "frontend"], "conventions", {
  framework: "React + Next.js",
  stateManagement: "Zustand",
  styling: "Tailwind CSS",
});

await store.put(["teams", "backend"], "conventions", {
  framework: "Fastify",
  orm: "Drizzle",
  database: "PostgreSQL",
});

// ── listNamespaces：探索已有的 namespace 结构 ──
const allNs = await store.listNamespaces({});
console.log("\n[listNamespaces] 所有 namespace：");
for (const ns of allNs) {
  console.log(`  ${ns.join("/")}`);
}

// 按前缀过滤
const userNs = await store.listNamespaces({ prefix: ["users"] });
console.log("\n[listNamespaces] users/ 下的 namespace：");
for (const ns of userNs) {
  console.log(`  ${ns.join("/")}`);
}

// ═══════════════════════════════════════════════════════════════════════════
// 演示 3：search —— 按 namespace 前缀 + filter 检索
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 3：search（前缀扫描 + filter 过滤）");
console.log("═".repeat(60));

// 先多写几条数据
await store.put(["users", "alice", "facts"], "fact-1", {
  content: "用户住在上海",
  category: "location",
  confidence: 0.95,
});
await store.put(["users", "alice", "facts"], "fact-2", {
  content: "用户偏好深色主题",
  category: "preference",
  confidence: 0.9,
});
await store.put(["users", "alice", "facts"], "fact-3", {
  content: "用户正在学习 LangGraph",
  category: "activity",
  confidence: 0.99,
});

// ── 基础搜索：扫描 namespace 前缀下所有 Item ──
const allAliceFacts = await store.search(["users", "alice", "facts"]);
console.log(`\n[search] Alice 的所有 facts（${allAliceFacts.length} 条）：`);
for (const item of allAliceFacts) {
  console.log(`  ${item.key}: ${item.value.content} (${item.value.category})`);
}

// ── filter 过滤：按 value 字段精确匹配 ──
const locationFacts = await store.search(["users", "alice", "facts"], {
  filter: { category: "location" },
});
console.log(`\n[search + filter] category=location 的 facts（${locationFacts.length} 条）：`);
for (const item of locationFacts) {
  console.log(`  ${item.key}: ${item.value.content}`);
}

// ── 分页（limit = 每页几条，offset = 跳过几条）──
const page1 = await store.search(["users", "alice", "facts"], { limit: 2, offset: 0 });
const page2 = await store.search(["users", "alice", "facts"], { limit: 2, offset: 2 });
console.log(`\n[分页] 共 3 条数据，limit=2`);
console.log(`  第1页（offset=0，跳过0条）: ${page1.map(i => i.key).join(", ")}`);
console.log(`  第2页（offset=2，跳过2条）: ${page2.map(i => i.key).join(", ")}`);

// ── 跨用户搜索（更宽的 namespace 前缀）──
const allUserPrefs = await store.search(["users"], {
  filter: { language: "TypeScript" },
});
console.log(`\n[跨用户搜索] language=TypeScript 的用户：`);
for (const item of allUserPrefs) {
  console.log(`  ${item.namespace.join("/")}/${item.key} → ${item.value.language}`);
}

// ═══════════════════════════════════════════════════════════════════════════
// 演示 4：在 Graph 节点中使用 Store —— 实现跨 thread 记忆
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 4：Graph 节点中读写 Store（跨 thread 记忆）");
console.log("═".repeat(60));

/**
 * 核心机制：
 * graph.compile({ checkpointer, store }) 传入 store 后，
 * 每个节点函数的第二个参数 config 里会带上 config.store，
 * 节点内部就可以直接 config.store.get() / put() / search()。
 *
 * 典型模式：
 * 1. 节点开头：从 store 读取用户的长期记忆
 * 2. 拼进 system prompt
 * 3. LLM 生成回答
 * 4.（可选）从回答中抽取新的事实写回 store
 */

const memoryStore = new InMemoryStore();
const checkpointer = new MemorySaver();

// 预存一些用户偏好
await memoryStore.put(["users", "alice"], "preferences", {
  language: "TypeScript",
  packageManager: "pnpm",
  editor: "Cursor",
});
await memoryStore.put(["users", "bob"], "preferences", {
  language: "Rust",
  packageManager: "cargo",
  editor: "Neovim",
});

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", async (state, config: LangGraphRunnableConfig) => {
    const userId = config.configurable?.user_id as string;
    const userStore = config.store!;

    // 从 Store 读取用户偏好（跨 thread 持久化的）
    const prefs = await userStore.get(["users", userId], "preferences");

    let systemContent = "你是一个简洁的编程助手，回答控制在一两句话内。";
    if (prefs) {
      systemContent += `\n\n用户偏好（来自长期记忆）：
- 编程语言：${prefs.value.language}
- 包管理器：${prefs.value.packageManager}
- 编辑器：${prefs.value.editor}
请根据这些偏好给出建议。`;
    }

    const resp = await llm.invoke([
      new SystemMessage(systemContent),
      ...state.messages,
    ]);
    return { messages: [resp] };
  })
  .addEdge(START, "agent")
  .addEdge("agent", END)
  .compile({ checkpointer, store: memoryStore });

async function chatWithMemory(userId: string, threadId: string, msg: string) {
  console.log(`\n[用户:${userId} @thread:${threadId}] ${msg}`);
  const result = await graph.invoke(
    { messages: [new HumanMessage(msg)] },
    {
      configurable: {
        thread_id: threadId,
        user_id: userId,
      },
    }
  );
  const reply = result.messages.at(-1)!.content;
  console.log(`[AI → ${userId}] ${reply}`);
  return result;
}

// Alice 在 thread-1 里问
await chatWithMemory("alice", "alice-thread-1", "帮我初始化一个新项目");

// Alice 在新 thread 里问（Store 跨 thread 生效，Checkpointer 不跨 thread）
await chatWithMemory("alice", "alice-thread-2", "帮我装个 HTTP 框架");

// Bob 问同一个问题 → 会得到 Rust/cargo 的回答
await chatWithMemory("bob", "bob-thread-1", "帮我初始化一个新项目");

// ═══════════════════════════════════════════════════════════════════════════
// 演示 5：运行时更新 Store —— 节点写回长期记忆
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 5：节点写回 Store（运行时更新长期记忆）");
console.log("═".repeat(60));

/**
 * 这里用简单的正则演示"从对话中抽取事实并写入 Store"。
 * 生产环境应该用 LLM 抽取 + AUDN 仲裁（后续 05-audn 实现）。
 */

const extractStore = new InMemoryStore();
const extractCheckpointer = new MemorySaver();

const extractGraph = new StateGraph(MessagesAnnotation)
  .addNode("agent", async (state, config: LangGraphRunnableConfig) => {
    const userId = config.configurable?.user_id as string;
    const st = config.store!;

    // 读取已有偏好
    const existing = await st.get(["users", userId], "preferences");
    const prefs = existing?.value ?? {};

    let systemContent = "你是一个简洁的助手，回答控制在一两句话内。";
    if (Object.keys(prefs).length > 0) {
      systemContent += `\n\n已知用户偏好：${JSON.stringify(prefs)}`;
    }

    const resp = await llm.invoke([
      new SystemMessage(systemContent),
      ...state.messages,
    ]);

    // 简单抽取：检测"我用 X""我喜欢 X"等模式
    const lastUserMsg = state.messages.at(-1)?.content as string;
    const langMatch = lastUserMsg?.match(/我(?:用|喜欢|偏好)\s*(\S+)/);
    if (langMatch) {
      const newPrefs = { ...prefs, noted: langMatch[1] };
      await st.put(["users", userId], "preferences", newPrefs);
      console.log(`  [Store 写入] 检测到偏好: "${langMatch[1]}" → 已更新`);
    }

    return { messages: [resp] };
  })
  .addEdge(START, "agent")
  .addEdge("agent", END)
  .compile({ checkpointer: extractCheckpointer, store: extractStore });

// 第一轮：用户提到偏好 → 写入 Store
console.log("\n--- 第一轮对话（提到偏好）---");
await extractGraph.invoke(
  { messages: [new HumanMessage("我用 Vue3 写前端")] },
  { configurable: { thread_id: "t1", user_id: "charlie" } }
);

// 验证写入
const charliePrefs = await extractStore.get(["users", "charlie"], "preferences");
console.log(`\n[验证] Charlie 的 Store 数据:`, charliePrefs?.value);

// 第二轮：新 thread，Store 里的偏好还在
console.log("\n--- 第二轮对话（新 thread，Store 记忆跨 thread）---");
const result = await extractGraph.invoke(
  { messages: [new HumanMessage("推荐一个 UI 组件库")] },
  { configurable: { thread_id: "t2", user_id: "charlie" } }
);
console.log(`[AI] ${result.messages.at(-1)!.content}`);

// ═══════════════════════════════════════════════════════════════════════════
// 总结
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "─".repeat(60));
console.log("总结：Store API 要点");
console.log("─".repeat(60));
console.log(`
  Store = 跨 thread 的持久化 KV 存储 = Long-term Memory 的基础设施

  核心 API：
    put(namespace, key, value)       → 写入 / 更新
    get(namespace, key)              → 精确读取
    delete(namespace, key)           → 删除
    search(namespacePrefix, opts)    → 前缀扫描 + filter + 分页
    listNamespaces(opts)             → 探索 namespace 结构

  Namespace 设计（生产最佳实践）：
    ["users", userId]                → 用户级隔离
    ["users", userId, "facts"]       → 用户的语义记忆
    ["teams", teamId]                → 团队级共享
    第一层定数据域，第二层定租户 ID

  在 Graph 中使用：
    compile({ checkpointer, store }) → 注入
    节点第二个参数 config.store       → 读写

  Checkpointer vs Store 协作：
    Checkpointer → thread 内的对话历史（Short-term）
    Store        → 跨 thread 的用户知识（Long-term）
    两者组合      → 完整的 Agent 记忆系统

  下一节（03-store-semantic）：给 Store 加上 embedding → 语义检索能力。
`);
