# Agent 系统性学习计划

> 学习路径：LangGraph 底层原语 → Agentic Design Patterns (21章) → Claude Code 源码分析
> 实践语言：TypeScript + LangGraph
> LLM：DeepSeek API（兼容 OpenAI 协议）
> 参考资料：
> - ADP 书籍：`/Users/wangyuelong/Desktop/Learning/agentic-design-patterns`
> - LangGraph 文档：https://docs.langchain.com/oss/javascript/langgraph/overview

---

## 进度总览

| 阶段 | 内容 | 状态 |
|------|------|------|
| 阶段零 | LangGraph 基础机制 | ✅ 完成 |
| 阶段一 | 基础工作流模式 (Ch.1-3) | ✅ 完成 |
| 阶段二 | 核心 Agent 能力 (Ch.4-6) | ✅ 完成 |
| 阶段三 | 多 Agent 协作 (Ch.7) | ✅ 完成 |
| 阶段四 | 记忆与学习 (Ch.8-9) | ⏳ 进行中 |
| 阶段五 | 工程化能力 (Ch.10-13) | ⏳ 待开始 |
| 阶段六 | 高级主题 (Ch.14-21) | ⏳ 待开始 |
| 阶段七 | 生产级实践 (Claude Code) | ⏳ 待开始 |

---

## 阶段零：LangGraph 基础机制

> package: `packages/01-langgraph-basics`
> 目标：吃透 LangGraph 核心原语，为后续所有实践打地基

| # | 主题 | 文件 | 状态 |
|---|------|------|------|
| 0.1 | State / Node / Edge | `src/index.ts` | ✅ 完成 |
| 0.2 | 条件路由 | `src/02-conditional-edge.ts` | ✅ 完成 |
| 0.3 | ReAct Agent（基础） | `src/03-react-agent.ts` | ✅ 完成 |
| 0.4 | Checkpointer（状态持久化） | `src/04-checkpointer.ts` | ✅ 完成 |
| 0.5 | Human-in-the-loop | `src/05-human-in-the-loop.ts` | ✅ 完成 |
| 0.6 | Streaming | `src/06-streaming.ts` | ✅ 完成 |

### 关键概念笔记

- **State**：图中流转的数据对象，每个字段有 reducer 控制合并方式
- **Node**：`(state) => Partial<state>` 的纯函数，只返回变更的字段
- **Edge**：固定边（A→B）或条件边（根据 State 动态决定下一节点）
- **Reducer**：`(prev, next) => merged`，控制字段如何合并（替换 vs 追加）
- **MessagesAnnotation**：内置的 messages state，reducer 为追加，适合对话场景
- **ToolNode**：内置节点，自动解析 AIMessage 的 tool_calls 并执行工具
- **shouldContinue**：检查最后一条 AIMessage 是否有 tool_calls，驱动 ReAct 循环

---

## 阶段一：基础工作流模式

> 对应 ADP Ch.1-3
> package: `packages/02-basic-workflows`

| # | ADP章节 | Pattern | 核心思想 | 状态 |
|---|---------|---------|---------|------|
| 1.1 | Ch.1 | Prompt Chaining | 复杂任务拆成串行步骤，每步输出是下步输入 | ✅ |
| 1.2 | Ch.2 | Routing | 根据输入内容分发到不同处理路径，LLM 做分类器 | ✅ |
| 1.3 | Ch.3 | Parallelization | 多子任务并行执行，汇总结果，用 Send API | ✅ |

---

## 阶段二：核心 Agent 能力

> 对应 ADP Ch.4-6
> package: `packages/03-core-agent-patterns`

| # | ADP章节 | Pattern | 核心思想 | 状态 |
|---|---------|---------|---------|------|
| 2.1 | Ch.4 | Reflection | Agent 审查自己的输出，迭代改进，循环图 | ✅ |
| 2.2 | Ch.5 | Tool Use | 工具设计原则，复杂工具链，错误处理 | ✅ |
| 2.3 | Ch.6 | Planning | 先制定计划再执行，Plan-Execute 图结构 | ✅ |

---

## 阶段三：多 Agent 协作

> 对应 ADP Ch.7
> package: `packages/04-multi-agent`

| # | ADP章节 | Pattern | 核心思想 | 状态 |
|---|---------|---------|---------|------|
| 3.1 | Ch.7 | Supervisor | 一个 Agent 协调多个子 Agent（V1 全共享 / V2 隔离+总结） | ✅ |
| 3.2 | Ch.7 | Swarm | Agent 间平等协作，handoff tool 移交控制权（Mesh） | ✅ |
| 3.3 | Ch.7 | Hierarchical | Supervisor 递归套娃，Team 子图 + 接口契约 | ✅ |

---

## 阶段四：记忆与学习

> 对应 ADP Ch.8-9
> package: `packages/05-memory`

| # | ADP章节 | Pattern | 核心思想 | 状态 |
|---|---------|---------|---------|------|
| 4.1 | Ch.8 | Memory Management | 短期/长期记忆，跨会话记忆，Checkpointer + 外部存储 | ⏳ |
| 4.2 | Ch.9 | Learning & Adaptation | Agent 从经验中学习，动态调整行为，few-shot 更新 | ⏳ |

---

## 阶段五：工程化能力

> 对应 ADP Ch.10-13
> package: `packages/06-engineering`

| # | ADP章节 | Pattern | 核心思想 | 状态 |
|---|---------|---------|---------|------|
| 5.1 | Ch.10 | MCP | 标准化工具协议，Agent 接入外部系统 | ⏳ |
| 5.2 | Ch.11 | Goal Setting & Monitoring | 目标追踪，执行监控 | ⏳ |
| 5.3 | Ch.12 | Exception Handling | 错误恢复，重试策略，Self-Correction | ⏳ |
| 5.4 | Ch.13 | Human-in-the-Loop | 人工介入，审批流程，interrupt 机制 | ⏳ |

---

## 阶段六：高级主题

> 对应 ADP Ch.14-21
> package: `packages/07-advanced`

| # | ADP章节 | Pattern | 核心思想 | 状态 |
|---|---------|---------|---------|------|
| 6.1 | Ch.14 | RAG | 知识检索增强，向量数据库集成 | ⏳ |
| 6.2 | Ch.15 | A2A | Agent 间通信协议 | ⏳ |
| 6.3 | Ch.16 | Resource-Aware Optimization | 成本与性能优化，token 管理 | ⏳ |
| 6.4 | Ch.17 | Reasoning Techniques | CoT、ToT、ReAct 深入对比 | ⏳ |
| 6.5 | Ch.18 | Guardrails / Safety | 安全护栏，输入输出过滤 | ⏳ |
| 6.6 | Ch.19 | Evaluation & Monitoring | Agent 评估体系，可观测性 | ⏳ |
| 6.7 | Ch.20 | Prioritization | 任务优先级，资源调度 | ⏳ |
| 6.8 | Ch.21 | Exploration & Discovery | 自主探索，未知环境中的决策 | ⏳ |

---

## 阶段七：生产级实践

> Claude Code 源码分析
> package: `packages/08-claude-code-analysis`

| # | 主题 | 状态 |
|---|------|------|
| 7.1 | 整体 Agent 架构分析 | ⏳ |
| 7.2 | Tool 设计方式与错误处理 | ⏳ |
| 7.3 | Context 管理与压缩策略 | ⏳ |
| 7.4 | 可观测性：日志、追踪、调试 | ⏳ |

---

## Packages 目录结构

```
packages/
├── 01-langgraph-basics/        ← 阶段零：LangGraph 基础
├── 02-basic-workflows/         ← 阶段一：Ch.1-3
├── 03-core-agent-patterns/     ← 阶段二：Ch.4-6
├── 04-multi-agent/             ← 阶段三：Ch.7
├── 05-memory/                  ← 阶段四：Ch.8-9
├── 06-engineering/             ← 阶段五：Ch.10-13
├── 07-advanced/                ← 阶段六：Ch.14-21
└── 08-claude-code-analysis/    ← 阶段七：源码分析
```

---

## 每个 Pattern 的学习节奏

1. **预习**：读 ADP 对应章节，理解概念
2. **讲解**：老师讲解（是什么、解决什么问题、适用场景），提出思考问题
3. **⏸ 暂停消化**：学生独立思考，回答问题，确认理解后再继续
4. **实现**：用 LangGraph 原语从底层实现（不用高层封装）
5. **⏸ 暂停验证**：学生跑通代码，观察执行过程，遇到问题自行 debug 或提问
6. **回顾**：老师带领对比分析（与上一个 Pattern 的区别和联系），总结关键点
7. **知识沉淀**：将核心概念、设计原则、模式对比以注释形式写入代码文件顶部
8. **⏸ 暂停确认**：学生用自己的话总结本节内容，通过后方可进入下一章节

> **原则**：每个 ⏸ 暂停点，都需要学生明确说"我理解了，可以继续"，老师才推进下一步。
> 不催进度，以真正掌握为唯一标准。

---

*最后更新：2026-04-28*
