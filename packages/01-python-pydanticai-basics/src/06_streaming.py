"""0.6 Streaming —— 流式输出与过程事件

对照 LangGraph 的 streaming（graph.stream 的 updates / messages 两种 streamMode）：
- LangGraph 给你 "messages"（token 级增量）和 "updates"（节点级状态变更）两种流。
- Pydantic AI 对应有两个层次的 API：
    (1) `run_stream(...)` + `stream_text()`：聚焦**最终答案**的文本增量（最常用、最简单）。
    (2) `run_stream_events(...)`：把**整个运行过程**的事件都吐出来——模型在调哪个工具、
        工具返回了什么、文本/思维链 delta……（相当于 "updates" + 全过程可观测）。

⚠️ 一个重要的坑（官方文档明确提示）：当 output_type 是 str 时，`run_stream` 把"第一个
匹配输出类型的内容"当作最终结果。如果模型在调用工具**之前**先吐了一句文字，那句话就被
当成最终答案，后面的工具调用**默认不会执行**。所以"边调工具边观察过程"应该用
`run_stream_events()` 或 `agent.iter()`（它们会把整张 agent 图跑到底）。demo(2) 正是用前者。
"""

from __future__ import annotations

import asyncio
from datetime import date

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPartDelta,
)

from shared.model import get_model

agent = Agent(
    get_model(),
    instructions="你是天气助手。需要天气时调用 get_forecast 工具，然后用一句话总结。",
)


@agent.tool_plain
def get_forecast(city: str, day: date) -> str:
    """查询某城市某天的天气预报。"""
    return f"{city} 在 {day} 晴，24°C"


async def demo_text_stream() -> None:
    """(1) 最常用：流式拿最终答案的文本增量（delta=True 给增量片段）。"""
    print("=== (1) 纯文本流式（最终答案逐字出）===")
    async with agent.run_stream("用一句话介绍杭州的西湖。") as response:
        async for piece in response.stream_text(delta=True):
            print(piece, end="", flush=True)
    print()


async def demo_event_stream() -> None:
    """(2) 全过程事件流：观察 ReAct 循环里的每一步。

    run_stream_events 会把整张图跑到底，逐个 yield 事件；最后一个事件是
    AgentRunResultEvent，携带最终结果。
    """
    print("\n=== (2) 全过程事件流（看工具调用 + 最终答案）===")
    async with agent.run_stream_events("杭州 2026-07-01 天气怎么样？") as stream:
        async for event in stream:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                print("  [文本开始]", repr(event.part.content))
            elif isinstance(event, PartDeltaEvent):
                if isinstance(event.delta, TextPartDelta):
                    print(f"  [文本增量] {event.delta.content_delta!r}")
                elif isinstance(event.delta, ToolCallPartDelta):
                    print(f"  [工具入参增量] {event.delta.args_delta}")
            elif isinstance(event, FunctionToolCallEvent):
                print(f"  [调用工具] {event.part.tool_name} args={event.part.args}")
            elif isinstance(event, FunctionToolResultEvent):
                print(f"  [工具返回] {event.part.content}")
            elif isinstance(event, FinalResultEvent):
                print("  [开始产出最终结果]")
            elif isinstance(event, AgentRunResultEvent):
                print(f"\n  最终答案：{event.result.output}")


async def main() -> None:
    await demo_text_stream()
    await demo_event_stream()


if __name__ == "__main__":
    asyncio.run(main())
