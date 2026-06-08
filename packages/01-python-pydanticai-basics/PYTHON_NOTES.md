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

## 三、面向对象（class 完整版）

> 给从 JS/TS 回来的人：尽量配 JS 对照。本包用到的 `Deps`(@dataclass)、`CityInfo`(BaseModel)
> 都是 class，这一节把 class 一次讲透。

### 3.1 定义与实例化

```python
class Dog:
    def __init__(self, name: str, age: int):   # 构造器(初始化,不是"创建")
        self.name = name
        self.age = age

    def bark(self) -> str:                      # 实例方法,第一个参数永远是 self
        return f"{self.name} 汪!"

d = Dog("旺财", 3)   # 实例化,无 new
```

- `self` = 实例自身（≈ JS 的 `this`），但 Python 要求**显式**写成方法第一个参数。
- 实例化用 `Dog(...)`，**没有 `new`**。

### 3.2 类属性 vs 实例属性

```python
class Dog:
    species = "Canis"          # 类属性:所有实例共享
    def __init__(self, name):
        self.name = name       # 实例属性:每个实例独立
```

查找顺序：**实例 → 类 → 父类**。

⚠️ 经典陷阱：可变类型当类属性会被所有实例共享：

```python
class Box:
    items = []           # 所有 Box 共用同一个 list!
b1, b2 = Box(), Box()
b1.items.append(1)
b2.items                 # [1] —— 被污染。正确做法是放进 __init__ 成为实例属性
```

> 这也解释了上一节的困惑：普通类里 `name: str = X` 本是**类属性**；但 `CityInfo` 继承
> `BaseModel`，其元类把这些注解改造成了**带校验的实例字段**——所以才"像成员一样"每实例独立。

### 3.3 三种方法：实例 / 类 / 静态

```python
class Pizza:
    count = 0
    def __init__(self, size: int):
        self.size = size
        Pizza.count += 1

    def area(self) -> float:                 # 实例方法:操作具体实例(self)
        return 3.14 * (self.size / 2) ** 2

    @classmethod
    def margherita(cls):                     # 类方法:操作类(cls),常当"命名构造器/工厂"
        return cls(size=30)

    @staticmethod
    def inch_to_cm(inch: float) -> float:    # 静态方法:不碰 self/cls,只是挂在类下的函数
        return inch * 2.54
```

- 实例方法 → `self`；`@classmethod` → `cls`（类本身）；`@staticmethod` → 都不接收。

### 3.4 `@property`：把方法伪装成属性

```python
class Circle:
    def __init__(self, r: float):
        self._r = r                 # 约定:下划线开头 = 内部用

    @property
    def radius(self) -> float:      # 读:c.radius(不加括号)
        return self._r

    @radius.setter
    def radius(self, value: float): # 写:c.radius = 5,可在此校验
        if value < 0:
            raise ValueError("半径不能为负")
        self._r = value

    @property
    def area(self) -> float:        # 只读"计算属性"
        return 3.14 * self._r ** 2
```

> 新版 `result.usage`（属性不是方法）底层就是 `@property`。

### 3.5 封装约定（Python 没有真 private）

靠**命名约定**，不靠关键字：

| 写法 | 含义 |
|------|------|
| `name` | 公开 |
| `_name` | "内部用,别动"（约定,无强制） |
| `__name` | 名称改写 → `_类名__name`，避免子类冲突，**不是真 private** |

没有 `public/private/protected` 关键字。

### 3.6 继承

```python
class Animal:
    def __init__(self, name: str):
        self.name = name
    def speak(self) -> str:
        return "..."

class Cat(Animal):                  # 括号里写父类 = 继承
    def __init__(self, name: str, indoor: bool):
        super().__init__(name)      # 调父类构造器
        self.indoor = indoor
    def speak(self) -> str:         # 重写(override),无需任何装饰器
        return "喵"
```

- `super()` 访问父类实现；重写无需 `@Override`。
- 支持**多继承** `class C(A, B)`，方法解析顺序由 **MRO**（C3 线性化）决定，`C.__mro__` 可查。
- `CityInfo(BaseModel)` 本质就是继承。

### 3.7 常用 dunder（魔术方法）

解释器在特定语法/时机自动调用的 `__xxx__`：

```python
class Money:
    def __init__(self, amount: int):
        self.amount = amount
    def __repr__(self) -> str:               # print()/调试显示
        return f"Money({self.amount})"
    def __eq__(self, other) -> bool:         # ==
        return isinstance(other, Money) and self.amount == other.amount
    def __add__(self, other) -> "Money":     # +
        return Money(self.amount + other.amount)
```

常见还有：`__str__`、`__hash__`、`__len__`、`__iter__`、`__call__`（实例当函数调）、
`__enter__/__exit__`（支持 `with`）。`@dataclass` 和 `BaseModel` 会**自动生成** `__init__/__repr__/__eq__`。

### 3.8 数据类三档进化（本包都用到）

```python
# 档1 纯手写
class P1:
    def __init__(self, name: str, age: int = 0):
        self.name, self.age = name, age

# 档2 @dataclass:自动 __init__/__repr__/__eq__,不校验 —— 见 Deps
from dataclasses import dataclass
@dataclass
class P2:
    name: str
    age: int = 0

# 档3 Pydantic BaseModel:自动生成 + 运行时校验 + JSON schema —— 见 CityInfo
from pydantic import BaseModel
class P3(BaseModel):
    name: str
    age: int = 0
```

选择标准：内部可信数据用 `@dataclass`（轻量）；外部/LLM 不可信数据用 `BaseModel`（校验+schema）；
需要复杂行为而非单纯装数据时用纯 class。

### 3.9 抽象类与 Protocol（接口的两种思路）

```python
# 方式A 抽象基类:显式继承 + 强制实现
from abc import ABC, abstractmethod
class Storage(ABC):
    @abstractmethod
    def save(self, data: str) -> None: ...   # 子类不实现则无法实例化

# 方式B Protocol:鸭子类型/结构化类型(≈ TS 的 interface)
from typing import Protocol
class Saveable(Protocol):
    def save(self, data: str) -> None: ...   # 谁有 save 谁就算 Saveable,无需继承
```

`Protocol` 最接近 TS 的 `interface`——**结构匹配**而非显式继承。

### 3.10 进阶小点

- `__slots__ = ("x", "y")`：固定字段、禁用 `__dict__`，省内存、防乱加属性。
- 类本身也是**对象**：可赋值、当参数传——`deps_type=Deps` 就是"把类当值传"。
- `@dataclass(frozen=True)`：造不可变对象。

---

## 四、和 TypeScript 仓库的速查对照

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
