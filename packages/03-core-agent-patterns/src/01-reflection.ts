/**
 * 01 - Reflection（反思）
 *
 * ── 是什么 ──────────────────────────────────────────────────────────────────
 * Agent 生成输出后，用同一个（或另一个）LLM 充当审查员来批评输出，
 * 根据反馈重新生成，循环迭代直到质量达标或达到最大迭代次数。
 *
 * 图结构：
 *   START → generator → critic → (条件边) → generator（继续迭代）
 *                                          → END（通过或超限）
 *
 * ── 解决什么问题 ─────────────────────────────────────────────────────────────
 * 单次 LLM 调用质量有天花板。Reflection 让 Agent 像人一样"写完检查一遍"，
 * 通过迭代改进突破单次调用的质量上限。
 *
 * ── 终止条件（两个，取其先）────────────────────────────────────────────────
 * 1. 审查通过：Critic 认为输出质量达标
 * 2. 最大迭代次数：防止无限循环消耗 token（生产中必须有）
 *
 * ── Generator vs Critic 的 system prompt ────────────────────────────────────
 * 可以用同一个 LLM 实例，关键是 system prompt 完全不同：
 * - Generator：专注生成，根据需求和反馈产出内容
 * - Critic：专注审查，严格找问题，输出结构化的评审意见
 * 同一个 LLM 换了身份定位，就能客观批评自己刚生成的内容。
 *
 * ── 与之前模式的核心区别 ─────────────────────────────────────────────────────
 * 这是第一个图中有"回边"的模式（Critic → Generator）。
 * Prompt Chaining / Routing / Parallelization 都是有向无环图（DAG）。
 * Reflection 引入了循环，这是 LangGraph 相比普通函数调用的核心能力之一。
 *
 * ── 场景：迭代改进代码质量 ───────────────────────────────────────────────────
 */

import "dotenv/config";
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";

// ── State ──────────────────────────────────────────────────────────────────
const GraphState = Annotation.Root({
  // 用户的原始需求
  requirement: Annotation<string>({ reducer: (_, n) => n }),

  // 当前生成的代码（每轮被替换）
  code: Annotation<string>({ reducer: (_, n) => n, default: () => "" }),

  // Critic 的审查意见（每轮被替换）
  feedback: Annotation<string>({ reducer: (_, n) => n, default: () => "" }),

  // 审查是否通过
  approved: Annotation<boolean>({ reducer: (_, n) => n, default: () => false }),

  // 迭代计数器：每轮 generator 执行时 +1
  iterationCount: Annotation<number>({
    reducer: (prev, next) => prev + next, // 注意：节点返回增量 1，reducer 累加
    default: () => 0,
  }),
});

type State = typeof GraphState.State;

const MAX_ITERATIONS = 3;

// ── LLM ───────────────────────────────────────────────────────────────────
// Generator 和 Critic 用同一个 LLM 实例，通过不同 system prompt 区分角色
const llm = new ChatOpenAI({
  model: "deepseek-v4-pro",
  apiKey: process.env.DASHSCOPE_API_KEY,
  configuration: { baseURL: process.env.DASHSCOPE_BASE_URL },
});

// Critic 的结构化输出 schema
const reviewSchema = z.object({
  approved: z.boolean().describe("代码是否通过审查"),
  feedback: z
    .string()
    .describe("若不通过，列出具体问题和改进建议；若通过，填写'LGTM'"),
});

const critic = llm.withStructuredOutput(reviewSchema, {
  method: "functionCalling",
});

// ── 节点 ───────────────────────────────────────────────────────────────────

// 1. generator 节点：根据需求（和上一轮反馈）生成代码
async function generatorNode(state: State): Promise<Partial<State>> {
  const isFirstRound = state.iterationCount === 0;
  console.log(`\n[generator] 第 ${state.iterationCount + 1} 轮生成...`);

  const userPrompt = isFirstRound
    ? `需求：${state.requirement}`
    : `需求：${state.requirement}\n\n上一版代码：\n${state.code}\n\n审查意见：\n${state.feedback}\n\n请根据审查意见改进代码。`;

  const response = await llm.invoke([
    new SystemMessage(
      "你是一个 TypeScript 专家。根据需求编写简洁、正确、有类型注解的代码。只输出代码，不要解释。",
    ),
    new HumanMessage(userPrompt),
  ]);

  console.log(`[generator] 生成完成:\n ${response.content}`);
  // iterationCount 返回增量 1，reducer 会累加（0+1=1, 1+1=2 ...）
  return { code: response.content as string, iterationCount: 1 };
}

// 2. critic 节点：审查当前代码，给出结构化反馈
async function criticNode(state: State): Promise<Partial<State>> {
  console.log(`\n[critic] 审查第 ${state.iterationCount} 轮代码...`);

  const result = await critic.invoke([
    new SystemMessage(
      "你是一个严格的 TypeScript 代码审查员。检查：类型安全、错误处理、边界条件、代码简洁性。" +
        "标准：只有代码完全正确且无明显改进空间时才 approved: true。",
    ),
    new HumanMessage(`需求：${state.requirement}\n\n代码：\n${state.code}`),
  ]);

  console.log(
    `[critic] 审查结果: ${result.approved ? "✅ 通过" : "❌ 不通过"}`,
  );
  if (!result.approved) {
    console.log(`[critic] 反馈: ${result.feedback}`);
  }

  return { approved: result.approved, feedback: result.feedback };
}

// ── 条件边：决定继续迭代还是结束 ──────────────────────────────────────────
function shouldContinue(state: State): "generator" | typeof END {
  if (state.approved) {
    console.log(`\n[router] ✅ 审查通过，结束迭代`);
    return END;
  }
  if (state.iterationCount >= MAX_ITERATIONS) {
    console.log(
      `\n[router] ⚠️  已达最大迭代次数 (${MAX_ITERATIONS})，强制结束`,
    );
    return END;
  }
  console.log(`\n[router] 继续迭代（第 ${state.iterationCount + 1} 轮）`);
  return "generator";
}

// ── 构建图 ─────────────────────────────────────────────────────────────────
const graph = new StateGraph(GraphState)
  .addNode("generator", generatorNode)
  .addNode("critic", criticNode)
  .addEdge(START, "generator")
  .addEdge("generator", "critic")
  // critic 之后：通过或超限 → END，否则 → generator（形成循环）
  .addConditionalEdges("critic", shouldContinue)
  .compile();

// ── 运行 ───────────────────────────────────────────────────────────────────
console.log("═".repeat(50));
console.log("Reflection：迭代改进代码质量");
console.log("═".repeat(50));

const result = await graph.invoke({
  requirement:
    "实现一个 TypeScript 函数 safeDiv(a, b)，安全地执行除法，处理除零情况",
});

console.log("\n" + "═".repeat(50));
console.log(
  `完成，共迭代 ${result.iterationCount} 轮，最终状态: ${result.approved ? "通过" : "达到上限"}`,
);
console.log("═".repeat(50));
console.log("\n最终代码：");
console.log(result.code);
