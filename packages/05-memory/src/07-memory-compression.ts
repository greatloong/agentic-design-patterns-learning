/**
 * 07 - Memory Compression：对话历史压缩
 *
 * ── 问题：Short-term Memory 无限增长 ──────────────────────────────────────
 *
 * 用户聊了 200 轮，每轮 ~500 tokens → 100k tokens 全塞给 LLM。
 * 问题三连：
 *   1. 成本：每次调用按 input tokens 计费，第 200 轮付前 199 轮的全部内容
 *   2. 性能：LLM 对超长 context 早期信息的召回率下降（lost in the middle）
 *   3. 物理限制：再大的窗口也有上限
 *
 * ── 关键认知：压缩的是 Short-term，不是 Long-term ─────────────────────────
 *
 * Long-term（Semantic Memory）不做压缩，做检索——100 万条记忆存着，
 * 每次只取最相关的 3-5 条注入 prompt，token 消耗恒定。
 *
 * Short-term（对话历史）才是需要压缩的对象。
 *
 * ── 三种压缩策略 ────────────────────────────────────────────────────────
 *
 * 策略 1：滑动窗口 — 只保留最近 N 轮，超出丢弃
 *   优点：零延迟，零成本
 *   缺点：早期关键信息永久丢失
 *
 * 策略 2：LLM 摘要 — 旧消息压缩成一段摘要
 *   优点：关键信息不丢
 *   缺点：每次压缩需要一次 LLM 调用（成本 + 延迟）
 *
 * 策略 3：分层混合（生产标配）— 近期原文 + 远期摘要
 *   类似人脑：昨天的事记细节，上个月只记大概
 *
 * ── 生产里最常见的组合 ──────────────────────────────────────────────────
 *
 * Working Memory 构成：
 *   [System Prompt + Procedural]
 *   + [Long-term 检索结果 2-5 条]    ← 独立通道，不受压缩影响
 *   + [远期摘要]                      ← 策略 2/3
 *   + [近期原文 10-20 轮]             ← 策略 1
 *
 * Demo 1：滑动窗口
 * Demo 2：LLM 摘要压缩
 * Demo 3：分层混合（近期原文 + 远期摘要）
 * Demo 4：对比三种策略的 token 消耗
 */

import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  BaseMessage,
} from "@langchain/core/messages";

const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
  temperature: 0,
});

// ── 模拟一段 20 轮对话历史 ────────────────────────────────────────────────

function generateFakeHistory(): BaseMessage[] {
  const turns: [string, string][] = [
    ["你好，我叫 Alice，在上海做前端开发。", "你好 Alice！前端开发很棒，有什么我能帮你的吗？"],
    ["我们公司用 React + TypeScript 技术栈。", "React + TS 是目前前端的主流组合，生态很成熟。"],
    ["最近在研究 Server Components。", "RSC 确实是 React 新方向，能显著减少客户端 JS 体积。"],
    ["我们项目的首屏加载要 4 秒，太慢了。", "4s 确实偏高，建议从代码分割和图片优化入手。"],
    ["已经做了 lazy loading，还是慢。", "那可能是 API 瀑布流问题，试试并行请求或 BFF 层。"],
    ["我们后端是 Java 微服务，沟通成本高。", "这很常见。可以前端自建 BFF 用 Node.js 做聚合。"],
    ["老板想让我用 Next.js 重构。", "Next.js 能解决 SSR/SSG + API Routes 一体化的问题。"],
    ["但我们有 50 万行代码，全面重构风险大。", "可以渐进式迁移——新页面用 Next.js，旧的逐步切。"],
    ["团队里有人想用 Vue，有人想用 React。", "建议统一技术栈，分裂维护成本更高。基于现状选 React 更合理。"],
    ["好的，我决定继续用 React。", "明智的选择！保持技术栈一致性对团队效率很重要。"],
    ["对了，我下个月要去北京出差。", "北京现在天气不错，祝出差顺利！"],
    ["出差是去参加 React Conf。", "React Conf 是很好的学习机会，推荐重点关注 Server Actions。"],
    ["你觉得 AI 会取代前端开发吗？", "不会取代，但会改变工作方式。AI 辅助编码会成为标配。"],
    ["我在学 LangChain 和 LangGraph。", "很好的选择！Agent 开发是当下最有价值的技能之一。"],
    ["LangGraph 的 State 管理有点复杂。", "State + Reducer 模式需要适应，但给了你精确的状态控制。"],
    ["我想做一个客服 Agent。", "客服 Agent 是经典场景，需要 Memory + Tool Use + Routing。"],
    ["客户数据怎么存？", "用 PostgresStore 做 Long-term，Checkpointer 管对话状态。"],
    ["向量检索和 BM25 选哪个？", "生产用 Hybrid（两个都要 + RRF 融合），比纯向量好 15-30%。"],
    ["Mem0 怎么样？", "Mem0 适合快速上线，把 AUDN + 向量 + 图封装成一行代码了。"],
    ["好的，今天先聊到这里，谢谢！", "不客气，祝你项目顺利！有问题随时来聊。"],
  ];

  const messages: BaseMessage[] = [];
  for (const [h, a] of turns) {
    messages.push(new HumanMessage(h));
    messages.push(new AIMessage(a));
  }
  return messages;
}

// ── 工具函数：估算 token 数（粗略，1 中文字 ≈ 2 tokens）────────────────

function estimateTokens(messages: BaseMessage[]): number {
  return messages.reduce((sum, msg) => {
    const text = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
    const chineseChars = (text.match(/[\u4e00-\u9fff]/g) || []).length;
    const otherChars = text.length - chineseChars;
    return sum + chineseChars * 2 + Math.ceil(otherChars / 4);
  }, 0);
}

// ══════════════════════════════════════════════════════════════════════════
// 策略 1：滑动窗口
// ══════════════════════════════════════════════════════════════════════════

function slidingWindow(messages: BaseMessage[], windowSize: number): BaseMessage[] {
  // windowSize = 保留最近多少条消息（Human + AI 各算一条）
  if (messages.length <= windowSize) return messages;
  return messages.slice(-windowSize);
}

// ══════════════════════════════════════════════════════════════════════════
// 策略 2：LLM 摘要压缩
// ══════════════════════════════════════════════════════════════════════════

async function summarizeMessages(messages: BaseMessage[]): Promise<string> {
  const text = messages
    .map(m => `${m._getType() === "human" ? "用户" : "助手"}: ${m.content}`)
    .join("\n");

  const response = await llm.invoke([
    new SystemMessage(
      "请将以下对话历史压缩成一段简洁的摘要（3-5 句话），保留关键事实（用户身份、需求、决策、结论），丢弃寒暄和重复内容。"
    ),
    new HumanMessage(text),
  ]);

  return response.content as string;
}

// ══════════════════════════════════════════════════════════════════════════
// 策略 3：分层混合
// ══════════════════════════════════════════════════════════════════════════

interface LayeredResult {
  summary: string;
  recentMessages: BaseMessage[];
}

async function layeredCompression(
  messages: BaseMessage[],
  recentWindow: number
): Promise<LayeredResult> {
  // 近期保留原文
  const recent = messages.slice(-recentWindow);
  // 远期压缩为摘要
  const older = messages.slice(0, -recentWindow);

  let summary = "";
  if (older.length > 0) {
    summary = await summarizeMessages(older);
  }

  return { summary, recentMessages: recent };
}

// ══════════════════════════════════════════════════════════════════════════
// Demo 1：滑动窗口
// ══════════════════════════════════════════════════════════════════════════

async function demo1SlidingWindow() {
  console.log("====== Demo 1：滑动窗口（window=10 条）======\n");

  const history = generateFakeHistory(); // 40 条消息
  console.log(`原始对话：${history.length} 条消息，~${estimateTokens(history)} tokens`);

  const windowed = slidingWindow(history, 10);
  console.log(`窗口保留：${windowed.length} 条消息，~${estimateTokens(windowed)} tokens`);
  console.log(`丢弃了：${history.length - windowed.length} 条消息\n`);

  console.log("保留的内容（最近 5 轮）：");
  windowed.forEach(m => {
    const role = m._getType() === "human" ? "用户" : "助手";
    console.log(`  ${role}: ${(m.content as string).slice(0, 40)}...`);
  });

  console.log("\n⚠️ 问题：Alice 住上海、做前端、用 React —— 这些全丢了。");
}

// ══════════════════════════════════════════════════════════════════════════
// Demo 2：LLM 摘要
// ══════════════════════════════════════════════════════════════════════════

async function demo2Summarization() {
  console.log("\n\n====== Demo 2：LLM 摘要压缩 ======\n");

  const history = generateFakeHistory();
  console.log(`原始：${history.length} 条消息，~${estimateTokens(history)} tokens`);

  const summary = await summarizeMessages(history);
  const summaryTokens = estimateTokens([new AIMessage(summary)]);

  console.log(`摘要：~${summaryTokens} tokens\n`);
  console.log("摘要内容：");
  console.log(`  ${summary}\n`);
  console.log(`压缩比：${estimateTokens(history)} → ${summaryTokens} tokens（${Math.round((1 - summaryTokens / estimateTokens(history)) * 100)}% 压缩率）`);
  console.log("\n✅ 关键信息保留了（身份、技术栈、决策）。");
  console.log("⚠️ 代价：需要一次 LLM 调用来生成摘要。");
}

// ══════════════════════════════════════════════════════════════════════════
// Demo 3：分层混合（生产标配）
// ══════════════════════════════════════════════════════════════════════════

async function demo3Layered() {
  console.log("\n\n====== Demo 3：分层混合（远期摘要 + 近期原文）======\n");

  const history = generateFakeHistory();
  const recentWindow = 10; // 近期保留 10 条（5 轮）

  console.log(`原始：${history.length} 条消息`);
  console.log(`策略：远期 ${history.length - recentWindow} 条 → 摘要，近期 ${recentWindow} 条 → 原文\n`);

  const { summary, recentMessages } = await layeredCompression(history, recentWindow);

  const summaryMsg = new SystemMessage(`[对话历史摘要] ${summary}`);
  const finalMessages = [summaryMsg, ...recentMessages];
  const finalTokens = estimateTokens(finalMessages);

  console.log("── 最终发给 LLM 的 messages 结构 ──\n");
  console.log(`  [0] SystemMessage（摘要）: ${summary.slice(0, 60)}...`);
  console.log(`  [1-${recentMessages.length}] 近期原文（${recentMessages.length} 条）`);
  recentMessages.slice(0, 4).forEach((m, i) => {
    const role = m._getType() === "human" ? "用户" : "助手";
    console.log(`    [${i + 1}] ${role}: ${(m.content as string).slice(0, 35)}...`);
  });
  console.log(`    ...共 ${recentMessages.length} 条\n`);

  console.log(`Token 对比：`);
  console.log(`  原始全量：~${estimateTokens(history)} tokens`);
  console.log(`  分层混合：~${finalTokens} tokens（${Math.round((1 - finalTokens / estimateTokens(history)) * 100)}% 压缩率）`);
  console.log("\n✅ 近期细节完整 + 远期关键信息保留。");
}

// ══════════════════════════════════════════════════════════════════════════
// Demo 4：三种策略对比
// ══════════════════════════════════════════════════════════════════════════

async function demo4Comparison() {
  console.log("\n\n====== Demo 4：三种策略 Token 消耗对比 ======\n");

  const history = generateFakeHistory();
  const originalTokens = estimateTokens(history);

  // 策略 1
  const windowed = slidingWindow(history, 10);
  const windowTokens = estimateTokens(windowed);

  // 策略 2
  const summary = await summarizeMessages(history);
  const summaryTokens = estimateTokens([new AIMessage(summary)]);

  // 策略 3
  const { summary: layeredSummary, recentMessages } = await layeredCompression(history, 10);
  const layeredTokens = estimateTokens([new SystemMessage(layeredSummary), ...recentMessages]);

  console.log("┌─────────────────────────────────────────────────────┐");
  console.log("│ 策略              │ Tokens │ 压缩率 │ 信息保留       │");
  console.log("├─────────────────────────────────────────────────────┤");
  console.log(`│ 原始全量           │ ${String(originalTokens).padStart(5)}  │   -    │ 100% 但超贵    │`);
  console.log(`│ 滑动窗口(10条)     │ ${String(windowTokens).padStart(5)}  │ ${String(Math.round((1 - windowTokens / originalTokens) * 100)).padStart(3)}%   │ 丢失早期信息   │`);
  console.log(`│ 纯摘要             │ ${String(summaryTokens).padStart(5)}  │ ${String(Math.round((1 - summaryTokens / originalTokens) * 100)).padStart(3)}%   │ 关键信息保留   │`);
  console.log(`│ 分层混合(推荐)     │ ${String(layeredTokens).padStart(5)}  │ ${String(Math.round((1 - layeredTokens / originalTokens) * 100)).padStart(3)}%   │ 最佳平衡       │`);
  console.log("└─────────────────────────────────────────────────────┘");
  console.log("\n生产推荐：分层混合 — 近期原文保细节 + 远期摘要保全局。");
}

// ── 主入口 ────────────────────────────────────────────────────────────────

async function main() {
  await demo1SlidingWindow();
  await demo2Summarization();
  await demo3Layered();
  await demo4Comparison();
}

main().catch(console.error);
