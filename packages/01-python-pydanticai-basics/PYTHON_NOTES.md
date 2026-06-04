# Python 工程化 & 语言特性速记（配合本包食用）

> 给"很久没写 Python"的人准备的回炉笔记。只讲**本包代码里实际用到**的东西，
> 每条都能在 `src/` 里找到对应代码，边看示例边对照。

---

## 一、包管理与工程化（uv）

### 1.1 为什么用 uv

`uv` 是目前最快的 Python 包管理器（Rust 写的），一个工具同时干了过去好几个工具的活：
`pip`（装包）+ `virtualenv`（建虚拟环境）+ `pip-tools`（锁版本）+ `pyenv`（管 Python 版本）。

对照 JS 生态：**uv ≈ pnpm**，`pyproject.toml` ≈ `package.json`，`uv.lock` ≈ `pnpm-lock.yaml`。

### 1.2 关键文件

| 文件 | 作用 | JS 类比 |
|------|------|---------|
| `pyproject.toml` | 声明项目元信息和依赖（写"意图"，如 `>=1.105.0`） | `package.json` |
| `uv.lock` | 锁定**精确**版本，保证可复现（自动生成，**要提交**） | `pnpm-lock.yaml` |
| `.python-version` | 指定本项目用哪个 Python 版本 | `.nvmrc` |
| `.venv/` | 虚拟环境（隔离的依赖安装目录，**不提交**） | `node_modules/` |

### 1.3 常用命令

```bash
uv sync                      # 按 pyproject + lock 安装/同步依赖（首次会自动建 .venv）
uv add pydantic-ai           # 加一个依赖（自动更新 pyproject + lock）
uv add --dev mypy            # 加一个开发期依赖（进 dependency-groups.dev）
uv remove <pkg>              # 删依赖
uv run python src/01_xxx.py  # 在 .venv 里跑脚本（不用手动 activate）
uv lock --upgrade            # 把锁文件里的依赖升到允许范围内的最新
```

> 注意 `uv run` 会**自动**用项目的 `.venv`，所以不需要 `source .venv/bin/activate`。

### 1.4 依赖版本约束怎么写

本包 `pyproject.toml` 里：

```toml
dependencies = [
    "pydantic-ai>=1.105.0",   # 下限 = 写代码时的最新版
    "python-dotenv>=1.2.2",
]
```

- `pyproject.toml` 写**下限/范围**（表达"我至少需要这个版本的特性"）。
- `uv.lock` 钉死**精确版本**（表达"这次实际装的是哪个版本"）。
- 这套分工 = pnpm 里 `package.json` 的 `^1.2.3` + `pnpm-lock.yaml` 的精确版本。

### 1.5 虚拟环境（venv）是什么

Python 没有 JS 那种自动的"本地 `node_modules`"。默认 `pip install` 装到**全局**，
不同项目会互相打架。**venv** 就是给每个项目开一个独立的依赖沙盒（一个文件夹）。
`uv` 帮你自动建在 `.venv/`，所以你基本无感——但要理解：依赖是装在这个文件夹里、与项目绑定的。

---

## 二、语言特性（本包用到的）

### 2.1 类型注解（Type Hints）

Python 的类型注解**默认不强制**（运行时不报错），主要给 IDE 补全 + 静态检查器（mypy）+
**Pydantic 在运行时读取**用。本包到处都是：

```python
def convert_currency(ctx: RunContext[Deps], amount: float, currency: str) -> str: ...
#                         ^泛型参数          ^参数类型              ^返回类型
```

- `list[str]`、`dict[str, float]`：内置容器泛型（3.9+ 可直接小写，不用 `typing.List`）。
- `X | None`：联合类型（3.10+ 写法，等价旧的 `Optional[X]`）。本包 `output_type=[str, DeferredToolRequests]` 也是"多选其一"。
- `Literal["a", "b"]`：字面量类型，把取值限定成有限集合 —— 见 `02_routing.py` 的 `Category`，
  这是"让 LLM 输出收敛成程序可判断的离散值"的关键。

> 关键点：**pydantic-ai / pydantic 会在运行时真的用这些注解**做 schema 和校验。
> 比如工具参数类型、`output_type` 的模型字段——注解不只是注释，是功能的一部分。

### 2.2 `from __future__ import annotations`

每个文件第一行都有它。作用：把所有类型注解**延迟成字符串**求值，于是你可以：
- 用还没定义的类型、避免循环导入；
- 在老一点的 Python 上用新写法（如 `list[str]`、`X | None`）。
现代 Python 项目几乎都加这一行，当成习惯即可。

### 2.3 装饰器（Decorator）

`@something` 放在函数/类上方，本质是"用 `something(原函数)` 把它包一层再替换回去"。本包里：

```python
@agent.tool                      # 把函数注册成 agent 的工具
def convert_currency(...): ...

@agent.tool_plain(requires_approval=True)   # 带参数的装饰器（先调用再装饰）
def delete_file(path): ...
```

对照：和 TS 的装饰器、Java 注解神似，但 Python 装饰器就是普通函数，没有魔法。

### 2.4 `@dataclass`

```python
from dataclasses import dataclass

@dataclass
class Deps:
    rates_to_cny: dict[str, float]
```

自动帮你生成 `__init__`、`__repr__` 等样板代码。`Deps(rates_to_cny={...})` 即可构造。
**dataclass vs Pydantic BaseModel** 的区别很重要：
- `@dataclass`：标准库，**不做数据校验**，适合"我自己代码内部传递的依赖"（见 `deps_type=Deps`）。
- `BaseModel`（pydantic）：**会校验 + 类型转换**，适合"和外部/LLM 交互、不可信的数据"（见 `output_type=CityInfo`）。

### 2.5 Pydantic `BaseModel`

pydantic-ai 的"半条命"。定义一个带类型的数据结构，它负责校验和（反）序列化：

```python
class CityInfo(BaseModel):
    name: str = Field(description="城市名")          # description 会进 JSON schema，喂给 LLM
    population_million: float
    famous_for: list[str]
```

- 模型给的数据若不符合类型，pydantic 抛校验错误；pydantic-ai 会把错误**回喂给模型让它重试**。
- `Field(description=...)` 的描述会变成给 LLM 的 schema 说明——所以描述要写清楚。

### 2.6 `async` / `await` 与事件循环

LLM 调用是 I/O 密集（等网络），所以 pydantic-ai 的核心是异步的。

```python
async def main():                       # 协程函数
    async with agent.run_stream(...) as r:   # 异步上下文管理器
        async for piece in r.stream_text():  # 异步迭代器
            ...

asyncio.run(main())                     # 入口：启动事件循环跑这个协程
```

- `async def` 定义协程；`await` 等一个异步操作完成（期间事件循环可去干别的）。
- `run_sync()` 是 pydantic-ai 给的**同步封装**（内部就是 `asyncio.run(run())`），
  本包 01–05 多用它图省事；06 因为要流式，必须用 `async`。
- `async with` / `async for`：异步版的 `with` / `for`，用于异步资源和异步流。

### 2.7 `if __name__ == "__main__":`

```python
if __name__ == "__main__":
    main()
```

只有当文件被**直接运行**（`python src/01_xxx.py`）时才执行 `main()`；
被别的文件 `import` 时不执行。约等于"模块的入口守卫"。

### 2.8 模块与导入（本包的 `shared`）

```python
from shared.model import get_model
```

- `src/shared/` 里有个空的 `__init__.py`，它把 `shared` 标记成一个**包**（可被 import）。
- 直接 `python src/01_xxx.py` 运行时，Python 会把脚本所在目录 `src/` 加进搜索路径，
  所以能找到同级的 `shared` 包。这就是为什么示例都放在 `src/` 下。

---

## 三、和 TypeScript 仓库的速查对照

| 概念 | 本包（Python + uv） | TS 各包（pnpm） |
|------|---------------------|-----------------|
| 依赖声明 | `pyproject.toml` | `package.json` |
| 锁文件 | `uv.lock` | `pnpm-lock.yaml` |
| 依赖目录 | `.venv/` | `node_modules/` |
| 装依赖 | `uv sync` | `pnpm install` |
| 跑脚本 | `uv run python src/01_xxx.py` | `pnpm dev` / `pnpm run 03` |
| 运行时类型 | type hints（pydantic 运行时读） | TS 类型（编译期擦除） |
| 数据校验 | pydantic `BaseModel` | `zod` |
| 环境变量 | `python-dotenv` 读 `.env` | `dotenv` 读 `.env` |
| 静态检查 | `mypy` | `tsc` |
```
