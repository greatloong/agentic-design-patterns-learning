# Agent 系统性学习

> 从 LangGraph 底层原语到 Agentic Design Patterns，再到生产级 Agent 系统的系统性学习实践。

## 概览

本项目是一个基于 **TypeScript + LangGraph** 的 AI Agent 系统性学习仓库，按照 [Agentic Design Patterns](https://github.com/xindoo/agentic-design-patterns) 一书的章节结构，从基础到高级逐步实现各类 Agent 设计模式。

每个模式都使用 LangGraph **底层原语**从零实现（不依赖高层封装），代码风格对齐生产实践。

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | TypeScript (ES2022) |
| Agent 框架 | LangGraph (@langchain/langgraph) |
| LLM | DeepSeek API（兼容 OpenAI 协议） |
| 包管理 | pnpm workspace (monorepo) |
| 运行时 | Node.js + tsx |
| 数据库 | PostgreSQL（记忆持久化） |
| 其他 | Zod (schema)、mem0ai (记忆层) |

## 项目结构

```
packages/
├── 01-langgraph-basics/        # 阶段零：LangGraph 基础机制
├── 02-basic-workflows/         # 阶段一：基础工作流模式 (ADP Ch.1-3)
├── 03-core-agent-patterns/     # 阶段二：核心 Agent 能力 (ADP Ch.4-6)
├── 04-multi-agent/             # 阶段三：多 Agent 协作 (ADP Ch.7)
├── 05-memory/                  # 阶段四：记忆与学习 (ADP Ch.8-9)
├── 06-engineering/             # 阶段五：工程化能力 (ADP Ch.10-13) [计划中]
├── 07-advanced/                # 阶段六：高级主题 (ADP Ch.14-21) [计划中]
└── 08-claude-code-analysis/    # 阶段七：Claude Code 源码分析 [计划中]
```

## 学习进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| 阶段零 | LangGraph 基础机制 | ✅ 完成 |
| 阶段一 | 基础工作流模式 (Ch.1-3) | ✅ 完成 |
| 阶段二 | 核心 Agent 能力 (Ch.4-6) | ✅ 完成 |
| 阶段三 | 多 Agent 协作 (Ch.7) | ✅ 完成 |
| 阶段四 | 记忆与学习 (Ch.8-9) | ⏳ 进行中 |
| 阶段五 | 工程化能力 (Ch.10-13) | 待开始 |
| 阶段六 | 高级主题 (Ch.14-21) | 待开始 |
| 阶段七 | 生产级实践 (Claude Code) | 待开始 |

## 各阶段详细内容

### 阶段零：LangGraph 基础机制

掌握 LangGraph 核心原语：State、Node、Edge、Reducer、Checkpointer。

| # | 主题 | 文件 |
|---|------|------|
| 0.1 | State / Node / Edge | `src/index.ts` |
| 0.2 | 条件路由 | `src/02-conditional-edge.ts` |
| 0.3 | ReAct Agent | `src/03-react-agent.ts` |
| 0.4 | Checkpointer（状态持久化） | `src/04-checkpointer.ts` |
| 0.5 | Human-in-the-loop | `src/05-human-in-the-loop.ts` |
| 0.6 | Streaming | `src/06-streaming.ts` |
| 0.7 | Streaming + AG-UI 协议（官方 @ag-ui/langgraph，双进程） | `src/07-streaming-ag-ui.ts` |
| 0.7* | AG-UI 手写转换器（参考对照版，进程内纯 SSE） | `src/07-streaming-ag-ui-manual.ts` |

### 阶段一：基础工作流模式

| # | Pattern | 核心思想 |
|---|---------|----------|
| 1.1 | Prompt Chaining | 复杂任务拆成串行步骤，每步输出是下步输入 |
| 1.2 | Routing | 根据输入内容分发到不同处理路径 |
| 1.3 | Parallelization | 多子任务并行执行，汇总结果 |

### 阶段二：核心 Agent 能力

| # | Pattern | 核心思想 |
|---|---------|----------|
| 2.1 | Reflection | Agent 审查自己的输出，迭代改进 |
| 2.2 | Tool Use | 工具设计原则，复杂工具链，错误处理 |
| 2.3 | Planning | 先制定计划再执行，Plan-Execute 图结构 |

### 阶段三：多 Agent 协作

| # | Pattern | 核心思想 |
|---|---------|----------|
| 3.1 | Supervisor | 一个 Agent 协调多个子 Agent |
| 3.2 | Swarm / Mesh | Agent 间平等协作，handoff 移交控制权 |
| 3.3 | Hierarchical | Supervisor 递归套娃，Team 子图 + 接口契约 |

### 阶段四：记忆与学习

| # | Pattern | 核心思想 |
|---|---------|----------|
| 4.1 | Memory Management | 短期/长期记忆，Checkpointer + 外部存储，语义检索，Hot/Cold 分层 |
| 4.2 | Learning & Adaptation | Agent 从经验中学习，反馈驱动，Prompt 自优化 |

## 快速开始

### 前置要求

- Node.js >= 18
- pnpm >= 8
- DeepSeek API Key（或其他兼容 OpenAI 协议的 LLM）
- Docker（仅记忆模块 `05-memory` 需要，用于运行 PostgreSQL + pgvector）

### 安装

```bash
pnpm install
```

### 配置环境变量

在对应 package 目录下创建 `.env` 文件：

```bash
# 主要 LLM（DeepSeek，兼容 OpenAI 协议）
DEEPSEEK_API_KEY=your-deepseek-api-key

# 部分示例使用 DashScope（阿里云）
DASHSCOPE_API_KEY=your-dashscope-api-key

# 多 Agent 示例中的搜索工具
BRAVE_API_KEY=your-brave-search-api-key
```

### 运行示例

每个 package 下的脚本都可以通过 pnpm 直接运行：

```bash
# 运行 LangGraph 基础示例
cd packages/01-langgraph-basics
pnpm dev          # State/Node/Edge 基础
pnpm run 03       # ReAct Agent

# 运行多 Agent 协作
cd packages/04-multi-agent
pnpm run 01       # Supervisor 模式
pnpm run 02m      # Swarm Mesh 模式

# 运行记忆系统（需要 PostgreSQL）
cd packages/05-memory
docker compose up -d   # 启动 PostgreSQL
pnpm run 01            # Checkpointer 生产化
pnpm run 06            # Hot/Cold Pipeline
```

## 参考资料

- [Agentic Design Patterns](https://github.com/xindoo/agentic-design-patterns) - 主要学习教材
- [LangGraph 文档](https://docs.langchain.com/oss/javascript/langgraph/overview) - 框架官方文档
- [DeepSeek API](https://platform.deepseek.com/) - LLM 服务

## License

ISC
