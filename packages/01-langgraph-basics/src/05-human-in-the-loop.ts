/**
 * 05 - Human-in-the-loop（人工介入）
 *
 * 核心概念：
 *   - interrupt()：在节点内暂停图的执行，等待人工输入
 *   - Command.RESUME：携带人工输入恢复执行
 *   - 依赖 Checkpointer：暂停状态需要持久化
 *
 * 场景：Agent 准备执行"危险操作"前，先展示计划，等待人工确认
 *
 * 图结构：
 *   START → plan → confirm（interrupt 在这里）→ execute → END
 */

import "dotenv/config";
import {
  StateGraph,
  Annotation,
  START,
  END,
  MemorySaver,
  interrupt,
  Command,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// ── State ──────────────────────────────────────────────────────────────────
const GraphState = Annotation.Root({
  task: Annotation<string>({ reducer: (_, n) => n }),
  plan: Annotation<string>({ reducer: (_, n) => n, default: () => "" }),
  humanDecision: Annotation<"approved" | "rejected" | "">({
    reducer: (_, n) => n,
    default: () => "",
  }),
  result: Annotation<string>({ reducer: (_, n) => n, default: () => "" }),
});

type State = typeof GraphState.State;

// ── LLM ───────────────────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
});

// ── 节点 ───────────────────────────────────────────────────────────────────

// 1. plan 节点：LLM 制定执行计划
async function planNode(state: State): Promise<Partial<State>> {
  console.log(`\n[plan] 正在为任务制定计划: "${state.task}"`);

  const response = await llm.invoke([
    new SystemMessage("你是一个任务规划助手。用简洁的步骤列出执行计划，不超过3步。"),
    new HumanMessage(`任务：${state.task}`),
  ]);

  const plan = response.content as string;
  console.log(`[plan] 计划制定完成:\n${plan}`);
  return { plan };
}

// 2. confirm 节点：展示计划，等待人工确认
function confirmNode(state: State): Partial<State> {
  console.log("\n" + "─".repeat(40));
  console.log("[confirm] ⚠️  需要人工确认");
  console.log("─".repeat(40));
  console.log(`计划内容：\n${state.plan}`);
  console.log("─".repeat(40));

  // interrupt() 暂停图的执行
  // 传入的值会作为 interrupt 的返回值暴露给调用方
  // 调用方通过 Command.RESUME 传入人工决策
  const decision = interrupt({
    message: "请确认是否执行以上计划？",
    plan: state.plan,
  });

  console.log(`[confirm] 收到人工决策: ${decision}`);
  return { humanDecision: decision as "approved" | "rejected" };
}

// 3. execute 节点：执行计划
async function executeNode(state: State): Promise<Partial<State>> {
  if (state.humanDecision === "rejected") {
    console.log("[execute] 计划已被拒绝，终止执行");
    return { result: "用户拒绝了执行计划，任务已取消。" };
  }

  console.log("[execute] 计划已批准，开始执行...");

  const response = await llm.invoke([
    new SystemMessage("你是一个任务执行助手。模拟执行以下计划并给出执行结果。"),
    new HumanMessage(`执行计划：\n${state.plan}`),
  ]);

  const result = response.content as string;
  console.log(`[execute] 执行完成`);
  return { result };
}

// ── 构建图 ─────────────────────────────────────────────────────────────────
const checkpointer = new MemorySaver();

const graph = new StateGraph(GraphState)
  .addNode("planner", planNode)
  .addNode("confirm", confirmNode)
  .addNode("execute", executeNode)
  .addEdge(START, "planner")
  .addEdge("planner", "confirm")
  .addEdge("confirm", "execute")
  .addEdge("execute", END)
  .compile({ checkpointer });

// ── 演示 ───────────────────────────────────────────────────────────────────
const threadId = "hitl-demo-1";
const config = { configurable: { thread_id: threadId } };

console.log("═".repeat(50));
console.log("场景：Agent 执行危险操作前需要人工确认");
console.log("═".repeat(50));

// 第一次 invoke：图会运行到 interrupt() 处暂停
const firstResult = await graph.invoke(
  { task: "清理数据库中所有超过30天的日志记录，并发送清理报告邮件给管理员" },
  config
);

// 到这里图已经暂停，firstResult 是暂停时的 State
console.log("\n[主程序] 图已暂停，等待人工决策...");

// 模拟人工审查后做出决策
// 实际场景中这里可以是 CLI 输入、Web 表单、Slack 消息等
const humanInput = "approved"; // 改成 "rejected" 可以看拒绝流程
console.log(`[主程序] 人工决策: ${humanInput}`);

// 用 Command.RESUME 恢复执行，传入人工决策
const finalResult = await graph.invoke(
  new Command({ resume: humanInput }),
  config
);

console.log("\n" + "═".repeat(50));
console.log("最终结果：");
console.log(finalResult.result);
