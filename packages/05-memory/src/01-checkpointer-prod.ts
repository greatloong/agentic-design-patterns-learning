/**
 * 01 - Checkpointer 进阶（Short-term Memory 的生产用法）
 *
 * ── 回顾 ──────────────────────────────────────────────────────────────────
 * 阶段零我们学了 Checkpointer 的基础：
 *   - compile({ checkpointer }) → 每个节点执行后自动保存 State 快照
 *   - thread_id 隔离不同会话
 *   - 同一 thread_id 的多轮对话自动恢复历史
 *
 * ── 本节新增 ──────────────────────────────────────────────────────────────
 * 这节聚焦三个生产场景：
 *
 * 1. Thread 切换：同一 Agent 实例服务多个用户/会话，验证 thread 隔离
 *    → 生产里一个 Agent 进程要同时服务成百上千个 thread
 *
 * 2. 状态快照读取（getState）：
 *    读出某个 thread 当前的完整 State
 *    → 生产用途：调试、可观测（"这个用户目前聊到哪了？"）
 *
 * 3. 状态热修复（updateState）：
 *    不经过 LLM，直接往 State 里注入/修改消息
 *    → 生产用途：人工介入纠错、注入上下文、回滚到某个时间点
 *
 * ── 关键认知 ──────────────────────────────────────────────────────────────
 * Checkpointer = Short-term Memory
 *   作用域：单个 thread
 *   生命周期：thread 存在期间
 *   生产选型：MemorySaver（开发）→ PostgresSaver（生产）
 *   不能做的事：跨 thread 共享知识（那是 Store/Long-term 的事）
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
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
} from "@langchain/core/messages";

// ── LLM ───────────────────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

// ── 构建一个最简单的对话 Agent ────────────────────────────────────────────
const checkpointer = new MemorySaver();

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", async (state) => {
    const resp = await llm.invoke([
      new SystemMessage("你是一个简洁的助手，回答控制在一两句话内。"),
      ...state.messages,
    ]);
    return { messages: [resp] };
  })
  .addEdge(START, "agent")
  .addEdge("agent", END)
  .compile({ checkpointer });

// 工具函数
async function chat(threadId: string, msg: string) {
  console.log(`\n[用户 @${threadId}] ${msg}`);
  const result = await graph.invoke(
    { messages: [new HumanMessage(msg)] },
    { configurable: { thread_id: threadId } }
  );
  const reply = result.messages.at(-1)!.content;
  console.log(`[AI   @${threadId}] ${reply}`);
  return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// 演示 1：Thread 切换 —— 验证隔离性
// ═══════════════════════════════════════════════════════════════════════════
console.log("═".repeat(60));
console.log("演示 1：Thread 切换（同一进程，多个 thread 互不干扰）");
console.log("═".repeat(60));

await chat("user-alice", "我叫 Alice，我喜欢用 TypeScript");
await chat("user-bob", "我叫 Bob，我喜欢用 Rust");

// Alice 的 thread 不知道 Bob 的信息
await chat("user-alice", "你还记得我喜欢什么语言吗？");
// Bob 的 thread 不知道 Alice 的信息
await chat("user-bob", "你还记得我喜欢什么语言吗？");

// ═══════════════════════════════════════════════════════════════════════════
// 演示 2：getState —— 读取 thread 当前状态快照
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 2：getState（读取 thread 的完整 State 快照）");
console.log("═".repeat(60));

const aliceState = await graph.getState({
  configurable: { thread_id: "user-alice" },
});

console.log(`\n[getState] Alice 的 thread 状态：`);
console.log(`  消息数量: ${aliceState.values.messages.length}`);
console.log(`  消息流：`);
for (const msg of aliceState.values.messages) {
  const role = msg.getType();
  const content =
    typeof msg.content === "string"
      ? msg.content.slice(0, 60)
      : JSON.stringify(msg.content).slice(0, 60);
  console.log(`    [${role}] ${content}${content.length >= 60 ? "..." : ""}`);
}

// 生产用途：
// - 客服系统后台面板展示"当前用户聊到哪了"
// - 出 bug 时快速 dump 用户的 State 做调试
// - 可观测系统定期采集 State 统计信息（消息数、token 用量等）

// ═══════════════════════════════════════════════════════════════════════════
// 演示 3：updateState —— 不经过 LLM，直接修改 State
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 3：updateState（人工注入消息，不经过 LLM）");
console.log("═".repeat(60));

// 场景：客服主管发现 Agent 给了错误信息，需要手动纠正
// 不想重跑整个对话，直接往 State 里注入一条纠正消息

const config = { configurable: { thread_id: "user-alice" } };

// 注入一条"人工纠正"消息
await graph.updateState(config, {
  messages: [
    new AIMessage(
      "[人工纠正] 补充说明：Alice 还提到她正在学习 LangGraph Agent 开发。"
    ),
  ],
});

console.log(`\n[updateState] 已注入人工纠正消息`);

// 验证：继续对话，Agent 应该能看到注入的消息
await chat("user-alice", "你知道我最近在学什么吗？");

// 再次 getState 验证消息数量增长
const aliceStateAfter = await graph.getState(config);
console.log(
  `\n[验证] 注入前消息数: ${aliceState.values.messages.length}，注入+对话后: ${aliceStateAfter.values.messages.length}`
);

// ═══════════════════════════════════════════════════════════════════════════
// 演示 4：状态版本（getStateHistory）—— 查看 checkpoint 历史
// ═══════════════════════════════════════════════════════════════════════════
console.log("\n" + "═".repeat(60));
console.log("演示 4：getStateHistory（checkpoint 版本历史）");
console.log("═".repeat(60));

// 每次 invoke / updateState 都会产生一个新的 checkpoint
// getStateHistory 可以遍历所有历史版本
let versionCount = 0;
for await (const snapshot of graph.getStateHistory(config)) {
  versionCount++;
  const msgCount = snapshot.values.messages?.length ?? 0;
  const source = snapshot.metadata?.source ?? "unknown";
  const step = snapshot.metadata?.step ?? -1;
  console.log(
    `  版本 ${versionCount}: ${msgCount} 条消息，source=${source}，step=${step}，checkpoint_id=${snapshot.config?.configurable?.checkpoint_id?.slice(0, 8)}...`
  );
  // 只打前 6 个版本，避免输出太长
  if (versionCount >= 6) {
    console.log(`  ... (更多历史版本省略)`);
    break;
  }
}

// 生产用途：
// - "回滚"到某个 checkpoint：用 updateState + 指定 checkpoint_id
// - 审计：追踪 State 的每一次变化
// - 调试：对比两个版本之间 State 的 diff

console.log("\n" + "─".repeat(60));
console.log("总结：Checkpointer 的三个生产级操作");
console.log("─".repeat(60));
console.log(`
  getState(config)        → 读当前快照（调试、可观测）
  updateState(config, {}) → 热修改 State（人工介入、纠错）
  getStateHistory(config) → 遍历版本历史（回滚、审计）

  这三个 API + thread_id 隔离，就是 Short-term Memory 的全部。
  它的天花板：只在同一个 thread 内有效。
  下一节（02-store-basics）解锁跨 thread 的 Long-term Memory。
`);
