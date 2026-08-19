"""0.6 Streaming（纯 Graph，无 Agent）—— 两个层次的流：节点级 + token 级

对照 LangGraph 的 `graph.stream`：
- "updates"：**节点级**——每个节点跑完吐一次状态变更。
- "messages"：**token 级**——最终答案逐字增量。

纯 graph 怎么做这两层：
- **节点级**：`async with graph.iter(...) as run: async for node in run` —— 图每推进一个节点
  就 yield 一次（≈ updates）。
- **token 级**：在节点内部直接用底层 `model.request_stream(...)` 拿增量（封装在
  `shared.model.stream_text` 里），把最终答案逐字吐出（≈ messages）。无需 Agent。

本例把两层叠在一次运行里看：图从 Lookup 走到 Summarize（节点级流），
而 Summarize 节点内部又把总结文字逐字流式打印（token 级流）。

执行顺序小知识：`async for node in run` 会**先 yield 节点、再运行它**——所以循环体里打印的
「节点标签」会正好出现在该节点 token 流的前面，输出自然不串行。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.graph import Graph

from shared.model import get_model, stream_text, user_msg

model = get_model()

_SUMMARY_SYSTEM = "你是天气助手，根据给到的预报数据用一句话总结。"


@dataclass
class StreamState:
    city: str
    day: str
    forecast: str = ""
    summary: str = ""


@dataclass
class Lookup(BaseNode[StreamState]):
    """查预报（mock）。对照 ReAct 里的工具节点，这里直接产出数据写回 State。"""

    async def run(self, ctx: GraphRunContext[StreamState]) -> Summarize:
        ctx.state.forecast = f"{ctx.state.city} 在 {ctx.state.day} 晴，24°C，东南风 3 级"
        return Summarize()


@dataclass
class Summarize(BaseNode[StreamState, None, str]):
    """token 级流：节点内部直接调底层 model 的流式接口，逐字打印总结。"""

    async def run(self, ctx: GraphRunContext[StreamState]) -> End[str]:
        prompt = f"用一句话总结这条天气预报：{ctx.state.forecast}"
        messages = [user_msg(prompt, system=_SUMMARY_SYSTEM)]
        text = ""
        print("    （token 流）", end="", flush=True)
        async for piece in stream_text(model, messages):
            print(piece, end="", flush=True)
            text += piece
        print()
        ctx.state.summary = text
        return End(text)


stream_graph = Graph(nodes=(Lookup, Summarize), state_type=StreamState)


async def main() -> None:
    print("=== stream_graph 结构（mermaid）===")
    print(stream_graph.mermaid_code(start_node=Lookup))

    print("\n=== 一次运行里看两层流（节点级 updates + token 级 messages）===")
    state = StreamState(city="杭州", day="2026-07-01")
    async with stream_graph.iter(Lookup(), state=state) as run:
        async for node in run:
            if isinstance(node, End):
                print("  [节点级] 图结束 (END)")
            else:
                print(f"  [节点级] 进入 {type(node).__name__} 节点")

    assert run.result is not None
    print("\n=== 最终答案 ===")
    print(run.result.output)


if __name__ == "__main__":
    asyncio.run(main())
