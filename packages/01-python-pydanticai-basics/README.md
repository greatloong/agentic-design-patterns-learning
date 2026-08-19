# 01 · Pydantic AI 基础（Python 学习线）

> 阶段零的 **Python 平行篇**：用 [Pydantic AI](https://ai.pydantic.dev/) +
> [`pydantic-graph`](https://ai.pydantic.dev/graph/) 重走一遍 LangGraph basics 的
> 6 个基础主题，建立"同一套 Agent 概念，在不同框架/语言里怎么落地"的对照认知。
> 顺带回炉 Python 工程化与语言特性（见 [`PYTHON_NOTES.md`](./PYTHON_NOTES.md)）。

> **写法说明（重要）**：本系列是**纯 `pydantic-graph` 实现，完全不用 `Agent`**。
> 流程编排靠图（State / Node / Edge），单步 LLM 调用**直接打到底层 `Model.request()`**。
> `Agent` 本质只是 `Model` 上的一层封装（帮你管消息、工具、结构化输出、重试）——这里把它
> 拿掉、亲手做这些事，好和 `01-langgraph-basics`（TS + LangGraph）在同一抽象层一一对照，
> 也彻底展示"graph 并不依赖 Agent"。底层调用的薄助手都收敛在 `shared/model.py`
> （`ask` / `chat` / `stream_text` / `parse_json`，都是对 `model.request` 的极薄封装，**不是 Agent**）。

## 这个包是什么

主仓库是 TypeScript + LangGraph 的学习线。这是**第一个 Python 包**，独立成项目（方式 A）：
用自己的 `pyproject.toml` + `.venv` 管理依赖，与 pnpm workspace 互不干扰（pnpm 只认 `package.json`）。

技术选型：

| 项 | 选择 | 说明 |
|----|------|------|
| 框架 | `pydantic-graph` + `pydantic-ai` 的 `Model` | 图编排流程，节点直接调底层 `model.request()`；**不用 `Agent`** |
| 包管理 | `uv` | 最快的 Python 包管理器，≈ pnpm |
| LLM | DashScope 的 `deepseek-v4-pro` | 阿里云百炼，兼容 OpenAI 协议（思考模型） |
| 环境变量 | `python-dotenv` | 读 `.env` |

## 主题（对照 LangGraph basics 0.1–0.6）

每个示例都**显式建一张 `pydantic-graph` 图**，并用 `graph.mermaid_code()` 打印结构。

| #   | 主题 | 文件 | graph 写法要点 | LangGraph 对照 |
|-----|------|------|----------------|----------------|
| 0.1 | State/Node/Edge 地基 | `src/01_agent_basics.py` | 单节点线性图 + `End[T]`（结构化输出） | State / Node / Edge |
| 0.2 | 条件路由 | `src/02_routing.py` | 节点 `run()` 联合返回类型 = 条件边 | 条件边 |
| 0.3 | 显式 ReAct 循环 | `src/03_react_tools.py` | `Think ⇄ Act` 循环 + `deps` 注入；动作用手写 JSON | ToolNode + shouldContinue 循环 |
| 0.4 | 状态持久化 | `src/04_message_history.py` | 自己维护 `list[ModelMessage]` + `FileStatePersistence` 存/读快照 | Checkpointer |
| 0.5 | Human-in-the-Loop | `src/05_human_in_the_loop.py` | `iter()`/`next()` 在节点处暂停再恢复 | interrupt |
| 0.6 | Streaming（两层流） | `src/06_streaming.py` | `graph.iter()` 节点级 + `model.request_stream` token 级 | graph.stream（updates/messages） |

> 核心认知差异：**LangGraph 与 pydantic-graph 是同一层抽象**（都让你显式声明
> State/Node/Edge）。日常 pydantic-ai 用户多半只接触 `Agent`（它在内部偷偷跑一张图、
> 把 State/Node/Edge 全封装隐藏）；本系列特意把 `Agent` 拿掉、直接用 `pydantic-graph` + 底层
> `Model`，好和 TS + LangGraph 版一一对照。
>
> ⚠️ **API 选择**：`pydantic-graph` 1.105 有新旧两套图 API。新的 `GraphBuilder`
> 尚未接入状态持久化（撑不起 0.4/0.5 的 checkpointer/interrupt），所以本系列统一用稳定的
> `Graph(nodes=...)`（`pydantic_graph.graph.Graph`），并用 `await graph.run(...)` 而非
> `run_sync()`（后者在 Python 3.12 会触发一个无害的 asyncio 事件循环 DeprecationWarning）。

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

- **怎么直接调底层 model**：`Agent` 是 `Model` 的封装；拿掉它就用
  `await model.request(messages, None, ModelRequestParameters())` 拿 `ModelResponse`，
  再从 `resp.parts` 里挑出 `TextPart` 拼成文本（思考模型还会带 `ThinkingPart`，要过滤）。
  流式则用 `model.request_stream(...)`。这些都封装在 `shared/model.py` 的薄助手里。
- **结构化输出靠手写**：没有 `Agent` 的 `output_type` 了，就在 system 提示词里要求"只输出 JSON"，
  自己抓 `{...}` 块再用 Pydantic `model_validate_json` 校验（见 `shared.model.parse_json`）。
  **额外好处**：全程不向模型声明工具、不用 `tool_choice`，自然绕开了思考模型
  （`deepseek-v4-pro`）在 DashScope 下不允许 `tool_choice=required` 的 400 报错。
- **流式别丢开头**：文本分 `PartStartEvent`（首块）+ 后续 `PartDeltaEvent` 两种事件到来，
  两种都要接，否则会丢掉第一块（见 `shared.model.stream_text`）。
- **两套图 API 的取舍**：1.105 新增的 `GraphBuilder` 是官方主推方向，但**尚未接入状态持久化**
  （`graph.run`/`iter` 都不收 `persistence`，也没有 `iter_from_persistence`），撑不起 0.4/0.5。
  故本系列统一用稳定的 `Graph(nodes=...)`（`from pydantic_graph.graph import Graph`）。
- **`run_sync()` 的 DeprecationWarning**：`Graph.run_sync()` 内部调用 `asyncio.get_event_loop()`，
  在 Python 3.12「无运行中的事件循环」时会告警。本系列一律走 `asyncio.run(main())` + `await graph.run(...)`，干净无告警。
- **节点的「边」靠返回注解推断**：`run()` 的返回类型注解（如 `Act | End[str]`）就是出边定义，
  框架据此推断图结构、画 mermaid、并在运行时校验。务必写准返回注解。
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
    ├── shared/model.py     # 模型工厂 + 底层 model 调用薄助手（ask/chat/stream_text/parse_json）
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
