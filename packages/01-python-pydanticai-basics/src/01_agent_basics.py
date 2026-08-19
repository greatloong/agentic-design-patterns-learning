"""0.1 纯 Graph 地基 —— State / Node / Edge（不依赖 Agent）

> 本系列是**纯 pydantic-graph 实现**：流程编排靠图，单步 LLM 调用直接打到底层
> `Model.request()`，**完全不用 `Agent`**。`Agent` 只是 `Model` 上的封装（帮你管消息、
> 工具、结构化输出、重试）；这里我们把它拿掉，亲手做这些事，彻底展示"graph 不需要 Agent"。
> 底层调用的薄助手都在 `shared/model.py`（`ask` / `chat` / `parse_json` …）。

graph 三件套（对照 LangGraph）：
- **State**：`@dataclass`，随图流动、被各节点读写（≈ Annotation State）。
- **Node**：`BaseNode` 子类，逻辑写在 `async def run()`（≈ Node 函数）。
- **Edge**：由 `run()` 的返回类型注解推断——返回 `End[T]` 即走到终点（≈ `addEdge(node, END)`）。

⚠️ API：pydantic-graph 1.105 的新 `GraphBuilder` 还没接持久化（撑不起 0.4/0.5），
故统一用稳定的 `Graph(nodes=...)`，并用 `await graph.run(...)`（`run_sync()` 在 Py3.12
会触发无害的 asyncio 事件循环 DeprecationWarning）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.graph import Graph

from shared.model import ask, get_model, parse_json

model = get_model()


# ---------- (1) 最朴素：纯文本输出的单节点图 ----------
@dataclass
class TextState:
    """图的 State：问题进来、答案出去。各节点都能读写它。"""

    question: str
    answer: str = ""


@dataclass
class Reply(BaseNode[TextState, None, str]):
    """唯一的节点：直接调底层 model 拿文本，写进 state 并结束整张图。

    返回 `End[str]`：这条出边直接指向终点（≈ `addEdge("reply", END)`）。
    泛型第 3 个参数 `str` 就是图的最终输出类型。
    """

    async def run(self, ctx: GraphRunContext[TextState]) -> End[str]:
        text = await ask(
            model,
            ctx.state.question,
            system="你是一个简洁的助手，回答控制在一句话内。",
        )
        ctx.state.answer = text
        return End(text)


text_graph = Graph(nodes=(Reply,), state_type=TextState)


# ---------- (2) 结构化输出：自己写提示词要 JSON，再解析校验 ----------
class CityInfo(BaseModel):
    """一个城市的结构化信息。"""

    name: str = Field(description="城市名")
    country: str = Field(description="所属国家")
    population_million: float = Field(description="人口（百万为单位）")
    famous_for: list[str] = Field(description="3 个最知名的标签")


# 没有 Agent 的 output_type 帮忙，"结构化"得自己来：提示词里要 JSON、自己 parse_json 校验。
# 好处是不碰 tool_choice，思考模型也稳。
_CITY_SYSTEM = (
    "你是地理百科助手。只输出一个 JSON 对象，不要 markdown、不要解释，形如："
    '{"name": "城市名", "country": "国家", "population_million": 人口数字, '
    '"famous_for": ["标签1", "标签2", "标签3"]}'
)


@dataclass
class CityState:
    query: str
    info: CityInfo | None = None


@dataclass
class Describe(BaseNode[CityState, None, CityInfo]):
    """图的最终输出可以是任意类型——这里是校验后的 CityInfo（`End[CityInfo]`）。"""

    async def run(self, ctx: GraphRunContext[CityState]) -> End[CityInfo]:
        text = await ask(model, ctx.state.query, system=_CITY_SYSTEM)
        info = parse_json(text, CityInfo)  # 解析 + Pydantic 校验
        ctx.state.info = info
        return End(info)


city_graph = Graph(nodes=(Describe,), state_type=CityState)


async def main() -> None:
    print("=== text_graph 结构（mermaid）===")
    print(text_graph.mermaid_code(start_node=Reply))

    print("\n=== (1) 纯文本图 ===")
    r1 = await text_graph.run(Reply(), state=TextState(question="用一句话介绍一下杭州。"))
    print(r1.output)

    print("\n=== (2) 结构化输出图（手写 JSON 解析）===")
    r2 = await city_graph.run(Describe(), state=CityState(query="介绍一下杭州"))
    info = r2.output  # 类型是 CityInfo
    print(repr(info))
    print(f"人口：{info.population_million} 百万")
    print(f"标签：{', '.join(info.famous_for)}")


if __name__ == "__main__":
    asyncio.run(main())
