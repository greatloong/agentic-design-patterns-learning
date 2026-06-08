# 01 · Pydantic AI 基础（Python 学习线）

> 阶段零的 **Python 平行篇**：用 [Pydantic AI](https://ai.pydantic.dev/) 重走一遍 LangGraph basics 的
> 6 个基础主题，建立"同一套 Agent 概念，在不同框架/语言里怎么落地"的对照认知。
> 顺带回炉 Python 工程化与语言特性（见 [`PYTHON_NOTES.md`](./PYTHON_NOTES.md)）。

## 这个包是什么

主仓库是 TypeScript + LangGraph 的学习线。这是**第一个 Python 包**，独立成项目（方式 A）：
用自己的 `pyproject.toml` + `.venv` 管理依赖，与 pnpm workspace 互不干扰（pnpm 只认 `package.json`）。

技术选型：

| 项 | 选择 | 说明 |
|----|------|------|
| 框架 | `pydantic-ai`（完整版） | Agent 框架，Pydantic 团队出品，类型安全 + 结构化输出是招牌 |
| 包管理 | `uv` | 最快的 Python 包管理器，≈ pnpm |
| LLM | DashScope 的 `deepseek-v4-pro` | 阿里云百炼，兼容 OpenAI 协议（思考模型） |
| 环境变量 | `python-dotenv` | 读 `.env` |

## 主题（对照 LangGraph basics 0.1–0.6）

| #   | 主题 | 文件 | LangGraph 对照 |
|-----|------|------|----------------|
| 0.1 | Agent 与结构化输出 | `src/01_agent_basics.py` | State / Node / Edge（核心抽象） |
| 0.2 | 条件路由 | `src/02_routing.py` | 条件边 |
| 0.3 | ReAct Agent（工具 + 依赖注入） | `src/03_react_tools.py` | ToolNode + shouldContinue 循环 |
| 0.4 | 消息历史与持久化 | `src/04_message_history.py` | Checkpointer |
| 0.5 | Human-in-the-Loop（工具审批） | `src/05_human_in_the_loop.py` | interrupt |
| 0.6 | Streaming（流式 + 过程事件） | `src/06_streaming.py` | graph.stream（messages/updates） |

> 核心认知差异：**LangGraph 以"图"为中心**（你显式声明 State/Node/Edge）；
> **Pydantic AI 以"Agent"为中心**（模型+指令+工具+输出类型的容器，内部也跑一张
> pydantic-graph 状态机，但默认对你隐藏）。

## 快速开始

### 前置

- Python（uv 会按 `.python-version` 自动准备 3.12）
- [uv](https://docs.astral.sh/uv/)（`brew install uv`）
- DashScope API Key

### 安装

```bash
cd packages/01-python-pydanticai-basics
uv sync
```

### 配置环境变量

复制 `.env.example` 为 `.env` 并填入 Key：

```bash
cp .env.example .env
# 编辑 .env，填 DASHSCOPE_API_KEY
```

### 运行示例

```bash
uv run python src/01_agent_basics.py
uv run python src/02_routing.py
uv run python src/03_react_tools.py
uv run python src/04_message_history.py
uv run python src/05_human_in_the_loop.py
uv run python src/06_streaming.py
```

### 代码检查与格式化（Ruff）

本包用 [Ruff](https://docs.astral.sh/ruff/)（astral 出品，和 `uv` 同一家）统一做 lint + 格式化 +
import 排序，规则配置见 `pyproject.toml` 的 `[tool.ruff]`。

```bash
uv run ruff check .          # 检查
uv run ruff check --fix .    # 检查并自动修复（含 import 排序）
uv run ruff format .         # 格式化（≈ Black）
uv run mypy src              # 类型检查
```

> 编辑器里建议装 Ruff 扩展（`charliermarsh.ruff`）+ Mypy 扩展（`ms-python.mypy-type-checker`），
> 保存即检查。

## 踩坑记录（实战要点）

- **思考模型 + 结构化输出的冲突**：`deepseek-v4-pro` 是思考（thinking）模型，DashScope 在思考模式下
  **不允许 `tool_choice=required`**。而 pydantic-ai 默认用「输出工具 + tool_choice=required」做结构化输出，
  会直接报 400。解决：结构化输出改用 `PromptedOutput` 模式（schema 写进提示词、返回 JSON 文本再校验），
  见 `01_agent_basics.py` / `02_routing.py`。普通非思考模型可省略，直接 `output_type=CityInfo`。
- **流式那个坑**：`output_type=str` 时 `run_stream` 把模型最先吐的文字当最终结果，之后的工具调用默认不执行。
  要"边调工具边观察过程"，用 `run_stream_events()` 或 `agent.iter()`，见 `06_streaming.py` 的 demo(2)。
- **`usage` 是属性不是方法**：新版 pydantic-ai 里 `result.usage`（不要写成 `result.usage()`）。
- **直连 DeepSeek 官方端点不稳**：本包统一走 DashScope 兼容端点（与主仓库其他包一致）。

## 目录结构

```
packages/01-python-pydanticai-basics/
├── pyproject.toml          # 依赖与项目元信息（≈ package.json）
├── uv.lock                 # 锁定精确版本（≈ pnpm-lock.yaml，要提交）
├── .python-version         # 指定 Python 3.12
├── .env.example            # 环境变量模板
├── PYTHON_NOTES.md         # Python 工程化 & 语言特性回炉笔记
├── README.md
└── src/
    ├── shared/model.py     # 统一模型工厂（指向 DashScope）
    ├── 01_agent_basics.py
    ├── 02_routing.py
    ├── 03_react_tools.py
    ├── 04_message_history.py
    ├── 05_human_in_the_loop.py
    └── 06_streaming.py
```

## 参考

- [Pydantic AI 文档](https://ai.pydantic.dev/)
- [uv 文档](https://docs.astral.sh/uv/)
- 对照：本仓库 `packages/01-langgraph-basics`（TS + LangGraph 版的同主题）
