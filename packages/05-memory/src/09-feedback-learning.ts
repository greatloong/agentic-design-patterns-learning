/**
 * 09 - Feedback Learning：从用户反馈中学习行为规则
 *
 * ── 解决的问题 ─────────────────────────────────────────────────────────────
 *
 * 4.1 的 Mem0 让 Agent 记住"用户是谁"（事实），但 Agent 的"做事方式"始终不变。
 * 用户每次都要纠正"别太正式""简短点"——Agent 记住了你喜欢 React，却学不会怎么跟你说话。
 *
 * ── 核心思路 ─────────────────────────────────────────────────────────────
 *
 * 把"行为规则"也当作记忆来管理：
 *   - 存储位置：LangGraph Store（与 03 中存语义记忆一样的 API）
 *   - 写入时机：用户纠正/反馈后，Cold Path 异步提炼规则
 *   - 读取时机：每次生成回复前，Hot Path 从 Store 读取规则注入 prompt
 *   - 更新逻辑：AUDN — 新规则 ADD，同主题冲突 UPDATE，用户否定 DELETE
 *
 * ── 与 4.1 的对比 ───────────────────────────────────────────────────────
 *
 *   4.1 Mem0:
 *     用户说"我喜欢 React" → 提取事实 → 存入 Semantic Memory
 *     读取时注入"用户偏好 React"到 prompt
 *
 *   4.2 Feedback Learning:
 *     用户说"太正式了" → 提炼规则"用非正式语气" → 存入 Procedural Memory
 *     读取时注入"行为规则：用非正式语气"到 prompt
 *
 *   技术上用的是同一套 Store API，区别在于：
 *     - namespace 不同（semantic vs procedural）
 *     - customInstructions/提炼 prompt 不同（提取事实 vs 提炼规则）
 *     - 注入位置不同（事实在中间，规则靠近 user message 尾部）
 *
 * ── 架构 ────────────────────────────────────────────────────────────────
 *
 *   Hot Path:
 *     用户请求 → 从 Store 读规则 → 拼入 system prompt → LLM 生成 → 返回
 *
 *   Cold Path（异步，用户不等）:
 *     用户反馈 → Memory Manager LLM 反思 → AUDN 决策 → 写入 Store
 *     → 下次 Hot Path 读到新规则
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
import {
  HumanMessage,
  SystemMessage,
  AIMessage,
} from "@langchain/core/messages";
import type { LangGraphRunnableConfig } from "@langchain/langgraph";

// ── 模型配置 ────────────────────────────────────────────────────────────

const llm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
});

const reflectionLlm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
  temperature: 0,
});

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-v4",
  dimensions: 512,
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: {
    baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
  },
});

// ── Store 配置 ───────────────────────────────────────────────────────────
// 使用 InMemoryStore + Embedding 索引，生产中可替换为 PostgresStore

const store = new InMemoryStore({
  index: {
    dims: 512,
    embeddings,
  },
});

// ── Namespace 设计 ───────────────────────────────────────────────────────
// procedural/{userId} 存行为规则
// semantic/{userId} 存用户事实（4.1 已学，此处不涉及）
const RULES_NAMESPACE = (userId: string) => ["procedural", userId];

// ══════════════════════════════════════════════════════════════════════════
// Cold Path：Memory Manager — 从用户反馈中提炼行为规则
// ══════════════════════════════════════════════════════════════════════════

const RULE_REFLECTION_PROMPT = `你是行为规则管理器。你的任务是从用户反馈中提炼 Agent 的行为规则。

## 当前已有规则
{existing_rules}

## 用户反馈
{feedback}

## 上下文（Agent 之前的输出，供你分析差距）
{agent_output}

## 你的任务

分析用户反馈，判断是否需要更新行为规则。

### 判断标准
- 如果用户表达的是**永久偏好**（"以后都...", "别再...", 连续多次纠正同一点），→ 提炼为规则
- 如果用户只是**一次性指令**（"这次写正式点", "帮我加个表情"），→ NOOP，不提炼
- 关键区分：用户是在说"这次"还是"以后"？不确定时倾向 NOOP

### 输出格式（JSON）
{
  "action": "ADD" | "UPDATE" | "DELETE" | "NOOP",
  "rule_id": "要更新/删除的规则ID（UPDATE/DELETE时必填）",
  "rule": "IF [触发条件] THEN [行为规则]（ADD/UPDATE时必填）",
  "reasoning": "为什么这么决策（一句话）"
}

只输出 JSON，不要输出其他内容。`;

interface RuleEntry {
  id: string;
  rule: string;
  createdAt: string;
}

/**
 * Cold Path：分析用户反馈，提炼行为规则并写入 Store
 * 生产中这个函数应在后台异步执行（queue / background job）
 */
async function learnFromFeedback(
  userId: string,
  feedback: string,
  agentOutput: string
): Promise<{ action: string; rule?: string; reasoning: string }> {
  // 1. 读取当前已有规则
  const existing = await store.search(RULES_NAMESPACE(userId), {
    query: feedback,
    limit: 20,
  });

  const existingRules: RuleEntry[] = existing.map((item) => ({
    id: item.key,
    rule: (item.value as any).rule,
    createdAt: (item.value as any).createdAt,
  }));

  const existingStr =
    existingRules.length > 0
      ? existingRules.map((r) => `[${r.id}] ${r.rule}`).join("\n")
      : "（暂无规则）";

  // 2. 调用 Memory Manager LLM 做 AUDN 决策
  const prompt = RULE_REFLECTION_PROMPT.replace(
    "{existing_rules}",
    existingStr
  )
    .replace("{feedback}", feedback)
    .replace("{agent_output}", agentOutput);

  const resp = await reflectionLlm.invoke([new HumanMessage(prompt)]);
  const content = (resp.content as string).trim();

  let decision: any;
  try {
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    decision = JSON.parse(jsonMatch?.[0] || content);
  } catch {
    return { action: "NOOP", reasoning: "解析失败，跳过" };
  }

  // 3. 根据决策执行写入
  const now = new Date().toISOString();

  if (decision.action === "ADD" && decision.rule) {
    const ruleId = `rule-${Date.now()}`;
    await store.put(RULES_NAMESPACE(userId), ruleId, {
      rule: decision.rule,
      createdAt: now,
    });
    console.log(`  [Cold Path] ADD: ${decision.rule}`);
  } else if (decision.action === "UPDATE" && decision.rule_id) {
    await store.delete(RULES_NAMESPACE(userId), decision.rule_id);
    const ruleId = `rule-${Date.now()}`;
    await store.put(RULES_NAMESPACE(userId), ruleId, {
      rule: decision.rule,
      createdAt: now,
    });
    console.log(
      `  [Cold Path] UPDATE [${decision.rule_id}] → ${decision.rule}`
    );
  } else if (decision.action === "DELETE" && decision.rule_id) {
    await store.delete(RULES_NAMESPACE(userId), decision.rule_id);
    console.log(`  [Cold Path] DELETE [${decision.rule_id}]`);
  } else {
    console.log(`  [Cold Path] NOOP: ${decision.reasoning}`);
  }

  return {
    action: decision.action,
    rule: decision.rule,
    reasoning: decision.reasoning,
  };
}

// ══════════════════════════════════════════════════════════════════════════
// Hot Path：带规则注入的 Agent Graph
// ══════════════════════════════════════════════════════════════════════════

const CORE_PROMPT = `你是一个邮件起草助手。用户会告诉你要写什么邮件，你来起草。`;

async function buildGraph() {
  const checkpointer = new MemorySaver();

  const graph = new StateGraph(MessagesAnnotation)
    .addNode("agent", async (state, config: LangGraphRunnableConfig) => {
      const userId =
        (config.configurable?.user_id as string) || "default-user";

      // ── Hot Path Step 1: 从 Store 读取行为规则 ──
      const rules = await store.search(RULES_NAMESPACE(userId), {
        query: state.messages[state.messages.length - 1].content as string,
        limit: 10,
      });

      const rulesStr =
        rules.length > 0
          ? rules.map((r) => `- ${(r.value as any).rule}`).join("\n")
          : "";

      // ── Hot Path Step 2: 拼装 System Prompt（Core + Learned）──
      let systemPrompt = CORE_PROMPT;
      if (rulesStr) {
        systemPrompt += `\n\n## 你从历史交互中学到的行为规则（遵守这些，除非与上述核心规则冲突）：\n${rulesStr}`;
      }

      // ── Hot Path Step 3: LLM 生成 ──
      const resp = await llm.invoke([
        new SystemMessage(systemPrompt),
        ...state.messages,
      ]);

      return { messages: [resp] };
    })
    .addEdge(START, "agent")
    .addEdge("agent", END);

  return graph.compile({ checkpointer });
}

// ══════════════════════════════════════════════════════════════════════════
// Demo：完整学习循环
// ══════════════════════════════════════════════════════════════════════════

async function main() {
  const app = await buildGraph();
  const userId = "demo-user";
  const threadId = "thread-feedback-demo";
  const config = {
    configurable: { thread_id: threadId, user_id: userId },
  };

  console.log("═".repeat(60));
  console.log("09 - Feedback Learning: 从用户反馈中学习行为规则");
  console.log("═".repeat(60));

  // ── Round 1: Agent 用默认方式写邮件 ──
  console.log("\n▶ Round 1: 用户请求写邮件（无规则，默认风格）");
  console.log("-".repeat(40));

  const result1 = await app.invoke(
    { messages: [new HumanMessage("帮我给张老师写个邮件，问一下论文进度")] },
    config
  );
  const agentReply1 = result1.messages[result1.messages.length - 1]
    .content as string;
  console.log("Agent 回复:\n", agentReply1);

  // ── Round 2: 用户给出反馈（触发 Cold Path 学习）──
  console.log("\n▶ Round 2: 用户反馈（触发学习）");
  console.log("-".repeat(40));
  const feedback = "太正式了，以后帮我写邮件都用轻松的语气，像朋友聊天一样";
  console.log("用户说:", feedback);

  // Cold Path: 异步学习（生产中放队列/后台，这里同步演示）
  console.log("\n  [Cold Path 开始] 分析反馈，提炼规则...");
  const learnResult = await learnFromFeedback(userId, feedback, agentReply1);
  console.log("  [Cold Path 结束]", JSON.stringify(learnResult, null, 2));

  // 显示当前 Store 中的规则
  const allRules = await store.search(RULES_NAMESPACE(userId), {
    query: "写邮件的规则",
    limit: 20,
  });
  console.log(
    "\n  📦 当前 Store 中的行为规则:",
    allRules.map((r) => (r.value as any).rule)
  );

  // ── Round 3: 用户再次请求写邮件（Agent 已学到规则）──
  console.log("\n▶ Round 3: 用户再次请求写邮件（已学到规则，观察风格变化）");
  console.log("-".repeat(40));

  const result3 = await app.invoke(
    { messages: [new HumanMessage("帮我给李明写个邮件约他周末吃火锅")] },
    { configurable: { thread_id: "thread-feedback-demo-2", user_id: userId } }
  );
  const agentReply3 = result3.messages[result3.messages.length - 1]
    .content as string;
  console.log("Agent 回复:\n", agentReply3);

  // ── Round 4: 用户给出另一个反馈，测试 AUDN 的 UPDATE ──
  console.log("\n▶ Round 4: 用户追加偏好");
  console.log("-".repeat(40));
  const feedback2 = "还有，邮件尽量简短，三句话以内搞定";
  console.log("用户说:", feedback2);

  console.log("\n  [Cold Path 开始] 分析反馈...");
  const learnResult2 = await learnFromFeedback(userId, feedback2, agentReply3);
  console.log("  [Cold Path 结束]", JSON.stringify(learnResult2, null, 2));

  const allRules2 = await store.search(RULES_NAMESPACE(userId), {
    query: "邮件规则",
    limit: 20,
  });
  console.log(
    "\n  📦 当前 Store 中的行为规则:",
    allRules2.map((r) => (r.value as any).rule)
  );

  // ── Round 5: 最终效果 ──
  console.log("\n▶ Round 5: 最终效果（两条规则同时生效）");
  console.log("-".repeat(40));

  const result5 = await app.invoke(
    { messages: [new HumanMessage("帮我给王伟写个邮件问他明天开会几点")] },
    { configurable: { thread_id: "thread-feedback-demo-3", user_id: userId } }
  );
  const agentReply5 = result5.messages[result5.messages.length - 1]
    .content as string;
  console.log("Agent 回复:\n", agentReply5);

  console.log("\n" + "═".repeat(60));
  console.log("✅ 学习循环完成！Agent 已从反馈中学会：轻松语气 + 简短");
  console.log("═".repeat(60));
}

main().catch(console.error);
