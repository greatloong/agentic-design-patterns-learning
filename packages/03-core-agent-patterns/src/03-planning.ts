/**
 * 03 - Planning（规划，带 DAG 依赖）
 *
 * ── 是什么 ──────────────────────────────────────────────────────────────────
 * Agent 先用 LLM 制定一个带依赖关系的任务 DAG（不是简单的步骤列表），
 * 然后按依赖关系调度执行：依赖已完成的任务可并行启动，依赖未满足的任务等待。
 *
 * 图结构（Plan-and-Execute with DAG）：
 *   START → planner → scheduler → (条件边) → ready 任务并行执行 → scheduler
 *                                          → replanner 判断是否调整
 *                                          → END
 *
 * ── 核心数据结构 ────────────────────────────────────────────────────────────
 * 每个任务：{ id, description, dependsOn: [其他任务id], status, result }
 * status 枚举：pending（待执行）/ running（执行中）/ done（完成）/ skipped（跳过）
 *
 * ── 调度逻辑 ─────────────────────────────────────────────────────────────────
 * 1. 找出所有 "pending 且依赖全部 done" 的任务 → 这些是 ready 任务
 * 2. 用 Send API 并行派发所有 ready 任务
 * 3. 所有任务执行完回到 scheduler，再找下一批 ready 任务
 * 4. 没有 ready 任务时 → Replanner 决定结束还是调整计划
 *
 * ── 与 Parallelization 的区别 ────────────────────────────────────────────────
 * Parallelization：所有子任务完全独立，一次性派发
 * Planning + DAG：任务间有依赖关系，按拓扑序分批派发，每批内部并行
 *
 * ── 与其他模式的本质区别 ─────────────────────────────────────────────────────
 * vs Prompt Chaining：步骤由 LLM 动态生成（含依赖关系），而不是开发者写死
 * vs ReAct：把整个 DAG 作为一等公民存在 State 里，可被审查、修改
 *
 * ── 示例场景：调研产品功能方案 ───────────────────────────────────────────────
 *   A: 收集用户反馈
 *   B: 分析竞品
 *   C: 综合 A+B 提炼核心痛点   ← 依赖 A, B
 *   D: 基于 C 输出改进方案       ← 依赖 C
 *   A 和 B 可并行，C 等 A+B 完成后才能跑，D 等 C 完成后才能跑。
 */

import "dotenv/config";
import { StateGraph, Annotation, START, END, Send } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";

// ── 任务结构 ───────────────────────────────────────────────────────────────
type TaskStatus = "pending" | "running" | "done" | "skipped";

type Task = {
  id: string;
  description: string;
  dependsOn: string[];
  status: TaskStatus;
  result?: string;
};

// ── State ──────────────────────────────────────────────────────────────────
// tasks 是一个 Map 形态：按 id 合并更新，任何并行节点的更新都能正确合并
const GraphState = Annotation.Root({
  objective: Annotation<string>({ reducer: (_, n) => n }),

  // 任务 DAG，按 id 合并（关键：并行执行时多个 executor 同时更新不同 id 不会冲突）
  tasks: Annotation<Record<string, Task>>({
    reducer: (prev, next) => ({ ...prev, ...next }),
    default: () => ({}),
  }),

  finalResponse: Annotation<string>({
    reducer: (_, n) => n,
    default: () => "",
  }),
});

type State = typeof GraphState.State;

const MAX_ITERATIONS = 8;
let iteration = 0;

// ── LLM ───────────────────────────────────────────────────────────────────
const llm = new ChatOpenAI({
  model: "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: { baseURL: "https://api.deepseek.com/v1" },
});

// ── Schemas ────────────────────────────────────────────────────────────────
const planSchema = z.object({
  tasks: z
    .array(
      z.object({
        id: z.string().describe("简短的任务标识符，如 A、B、C"),
        description: z.string().describe("具体可执行的任务描述"),
        dependsOn: z
          .array(z.string())
          .describe("此任务依赖的其他任务 id，无依赖则为空数组"),
      })
    )
    .describe("任务 DAG，每个任务标注依赖关系"),
});

const replanSchema = z.object({
  action: z.enum(["continue", "finish"]),
  finalResponse: z.string().optional(),
});

const planner = llm.withStructuredOutput(planSchema, { method: "functionCalling" });
const replanner = llm.withStructuredOutput(replanSchema, { method: "functionCalling" });

// executor 接收的参数（Send 传入，不是完整 State）
type ExecuteInput = {
  taskId: string;
  description: string;
  dependencyResults: Array<{ id: string; description: string; result: string }>;
  objective: string;
};

// ── 节点 ───────────────────────────────────────────────────────────────────

// 1. Planner：生成带依赖的任务 DAG
async function plannerNode(state: State): Promise<Partial<State>> {
  console.log(`\n[planner] 为目标制定 DAG 计划: "${state.objective}"`);

  const result = await planner.invoke([
    new SystemMessage(
      "你是一个任务规划专家。把用户目标拆解成一个任务 DAG：" +
      "1) 独立可并行的任务不要加依赖；" +
      "2) 需要前序结果的任务要声明 dependsOn；" +
      "3) 任务总数控制在 3-5 个；" +
      "4) 每个任务要具体、可独立完成。"
    ),
    new HumanMessage(`目标：${state.objective}`),
  ]);

  const tasks: Record<string, Task> = {};
  for (const t of result.tasks) {
    tasks[t.id] = {
      id: t.id,
      description: t.description,
      dependsOn: t.dependsOn,
      status: "pending",
    };
  }

  console.log(`[planner] 生成 ${result.tasks.length} 个任务:`);
  for (const t of result.tasks) {
    const deps = t.dependsOn.length > 0 ? `依赖 [${t.dependsOn.join(",")}]` : "无依赖";
    console.log(`  ${t.id}: ${t.description} (${deps})`);
  }

  return { tasks };
}

// 2. Scheduler：找出所有 ready 任务（pending 且依赖全部 done），没有具体副作用
// 实际的并行派发在条件边 dispatchReadyTasks 里完成
function schedulerNode(state: State): Partial<State> {
  iteration++;
  const ready = findReadyTasks(state.tasks);
  const remaining = Object.values(state.tasks).filter(
    (t) => t.status !== "done" && t.status !== "skipped"
  );

  console.log(`\n[scheduler] 轮次 ${iteration}：${remaining.length} 个任务待处理，其中 ${ready.length} 个可立即执行`);
  if (ready.length > 0) {
    console.log(`  并行派发: ${ready.map((t) => t.id).join(", ")}`);
  }

  return {};
}

// 找出所有可立即执行的任务（pending 且依赖全部 done）
function findReadyTasks(tasks: Record<string, Task>): Task[] {
  return Object.values(tasks).filter(
    (t) =>
      t.status === "pending" &&
      t.dependsOn.every((depId) => tasks[depId]?.status === "done")
  );
}

// 3. Executor：执行单个任务
async function executorNode(input: ExecuteInput): Promise<Partial<State>> {
  console.log(`  [executor:${input.taskId}] 开始执行: ${input.description}`);

  const context =
    input.dependencyResults.length > 0
      ? `依赖任务的结果：\n${input.dependencyResults
          .map((d) => `【${d.id}: ${d.description}】\n${d.result}`)
          .join("\n\n")}\n\n`
      : "";

  const response = await llm.invoke([
    new SystemMessage(
      "你是一个任务执行者。根据依赖任务的结果完成当前任务，输出 120 字以内的简洁结果。"
    ),
    new HumanMessage(
      `${context}最终目标：${input.objective}\n\n当前任务：${input.description}\n\n请执行该任务。`
    ),
  ]);

  const result = response.content as string;
  console.log(`  [executor:${input.taskId}] ✓ 完成`);

  // 只更新这一个任务的状态（reducer 按 id 合并，不会影响其他任务）
  return {
    tasks: {
      [input.taskId]: {
        id: input.taskId,
        description: input.description,
        dependsOn: [], // 占位，合并时 prev 里已有正确值，但 TS 需要完整对象
        status: "done",
        result,
      } as Task,
    },
  };
}

// 4. Replanner：所有任务完成后生成最终输出
async function replannerNode(state: State): Promise<Partial<State>> {
  const allDone = Object.values(state.tasks).every(
    (t) => t.status === "done" || t.status === "skipped"
  );

  console.log(
    `\n[replanner] 审视进度：${Object.values(state.tasks).filter(t => t.status === "done").length}/${Object.keys(state.tasks).length} 完成`
  );

  if (iteration >= MAX_ITERATIONS) {
    console.log(`[replanner] 达到最大调度轮数，强制结束`);
  }

  // 汇总所有已完成任务的结果，生成最终回复
  const context = Object.values(state.tasks)
    .filter((t) => t.status === "done")
    .map((t) => `【${t.id}: ${t.description}】\n${t.result}`)
    .join("\n\n");

  const result = await replanner.invoke([
    new SystemMessage(
      "你是一个任务总结专家。基于所有已完成任务的结果，给出面向用户的最终回复。" +
      "如果所有任务都完成且质量够，action=finish 并给出 finalResponse（综合报告）。" +
      "如果还有 pending 任务未执行（异常情况），action=continue。"
    ),
    new HumanMessage(
      `目标：${state.objective}\n\n已完成任务：\n${context}\n\n` +
      `所有任务是否完成：${allDone ? "是" : "否"}`
    ),
  ]);

  if (result.action === "finish") {
    return { finalResponse: result.finalResponse ?? "任务已完成" };
  }
  return {};
}

// ── 条件边：根据是否还有 ready 任务，决定并行派发还是去 replanner ──────
function dispatchOrReplan(state: State): Send[] | "replanner" {
  const ready = findReadyTasks(state.tasks);
  if (ready.length === 0) return "replanner";
  if (iteration >= MAX_ITERATIONS) return "replanner";

  // 并行派发所有 ready 任务到 executor
  return ready.map((task) => {
    const dependencyResults = task.dependsOn.map((depId) => ({
      id: depId,
      description: state.tasks[depId].description,
      result: state.tasks[depId].result ?? "",
    }));
    return new Send("executor", {
      taskId: task.id,
      description: task.description,
      dependencyResults,
      objective: state.objective,
    } satisfies ExecuteInput);
  });
}

// replanner 之后的条件边
function afterReplan(state: State): "scheduler" | typeof END {
  if (state.finalResponse) return END;
  if (iteration >= MAX_ITERATIONS) return END;
  return "scheduler";
}

// ── 构建图 ─────────────────────────────────────────────────────────────────
const graph = new StateGraph(GraphState)
  .addNode("planner", plannerNode)
  .addNode("scheduler", schedulerNode)
  .addNode("executor", executorNode)
  .addNode("replanner", replannerNode)
  .addEdge(START, "planner")
  .addEdge("planner", "scheduler")
  // scheduler 之后：要么并行派发 ready 任务，要么去 replanner 汇总
  .addConditionalEdges("scheduler", dispatchOrReplan)
  // 所有并行 executor 完成后回到 scheduler 看下一批
  .addEdge("executor", "scheduler")
  .addConditionalEdges("replanner", afterReplan)
  .compile();

// ── 运行 ───────────────────────────────────────────────────────────────────
console.log("═".repeat(50));
console.log("Planning + DAG：带依赖关系的并行调度");
console.log("═".repeat(50));

const result = await graph.invoke({
  objective:
    "给'大学生 AI 编程助手'做一次产品调研，输出一份包含核心痛点分析和具体改进建议的简报。" +
    "注意：'收集用户反馈'和'分析竞品'这两件事可以并行，'提炼痛点'依赖这两者，'输出建议'依赖痛点分析。",
});

console.log("\n" + "═".repeat(50));
console.log("任务执行轨迹：");
console.log("═".repeat(50));
for (const t of Object.values(result.tasks)) {
  const deps = t.dependsOn.length > 0 ? `依赖 [${t.dependsOn.join(",")}]` : "无依赖";
  console.log(`\n【${t.id}】${t.description} (${deps}) → ${t.status}`);
  if (t.result) console.log(`  ${t.result}`);
}

console.log("\n" + "═".repeat(50));
console.log("最终简报：");
console.log("═".repeat(50));
console.log(result.finalResponse);
