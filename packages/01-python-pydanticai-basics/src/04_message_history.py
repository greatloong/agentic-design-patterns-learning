"""0.4 图的状态持久化（纯 Graph，无 Agent）—— 对照 LangGraph 的 Checkpointer

pydantic-graph 用**状态持久化**在每个节点前后给图状态拍快照，可存内存或文件
（`FileStatePersistence` -> JSON）。这就是 graph 版的 Checkpointer。

无 Agent 怎么做多轮记忆：自己维护 `list[ModelMessage]`。每轮把用户消息 append 进历史、
调 `chat(model, history)` 拿 `ModelResponse`、再把响应 append 回历史。模型每次都能看到
完整上下文——这正是 `Agent` 的 `message_history=` 在底层替你做的事。

本节两个层次：
- (1) 同一份 State 跨轮累积（内存里维护 thread）。
- (2) 落盘持久化再恢复：`FileStatePersistence` 写 JSON，模拟"进程重启"后从磁盘读回 State。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.messages import ModelMessage
from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.graph import Graph
from pydantic_graph.persistence.file import FileStatePersistence

from shared.model import chat, get_model, text_of, user_msg

model = get_model()

_SYSTEM = "你是一个会记住上下文的助手，回答简洁。"


@dataclass
class ChatState:
    """图 State 同时承载「短期记忆」：累积的对话消息历史。"""

    user_input: str = ""
    history: list[ModelMessage] = field(default_factory=list)
    reply: str = ""


@dataclass
class Chat(BaseNode[ChatState, None, str]):
    """单节点：把用户消息并进历史、直接调 model、再把响应并回历史。"""

    async def run(self, ctx: GraphRunContext[ChatState]) -> End[str]:
        # 只有第一轮带 system 指令；之后它已存在历史里
        system = _SYSTEM if not ctx.state.history else None
        ctx.state.history.append(user_msg(ctx.state.user_input, system=system))
        resp = await chat(model, ctx.state.history)
        ctx.state.history.append(resp)
        ctx.state.reply = text_of(resp)
        return End(ctx.state.reply)


chat_graph = Graph(nodes=(Chat,), state_type=ChatState)

_STORE = Path(__file__).resolve().parent / "graph_conversation.json"


async def demo_in_memory() -> None:
    """(1) 同一个 state 对象跨轮复用——图 State 即短期记忆。"""
    print("=== (1) 内存多轮对话（复用同一份图 State）===")
    state = ChatState()

    state.user_input = "我叫王悦龙，正在学 Pydantic AI。"
    r1 = await chat_graph.run(Chat(), state=state)
    print("Q1:", state.user_input)
    print("A1:", r1.output)

    state.user_input = "我叫什么名字？在学什么？"
    r2 = await chat_graph.run(Chat(), state=state)
    print("\nQ2:", state.user_input)
    print("A2:", r2.output)


async def demo_persist() -> None:
    """(2) 落盘持久化 + 恢复：FileStatePersistence 充当 Checkpointer。"""
    print("\n=== (2) 持久化到磁盘再恢复 ===")
    if _STORE.exists():
        _STORE.unlink()

    persistence = FileStatePersistence(_STORE)
    persistence.set_graph_types(chat_graph)
    state = ChatState(user_input="记住一个数字：42。")
    await chat_graph.run(Chat(), state=state, persistence=persistence)
    print(f"已把图状态快照写入 {_STORE.name}")

    # 模拟「重启进程」：用新的持久化对象从磁盘读回最后的 State
    reloaded = FileStatePersistence(_STORE)
    reloaded.set_graph_types(chat_graph)
    snapshots = await reloaded.load_all()
    recovered_history = snapshots[-1].state.history
    print(f"从磁盘恢复了 {len(recovered_history)} 条历史消息")

    r = await chat_graph.run(
        Chat(),
        state=ChatState(user_input="我刚才让你记住的数字是多少？", history=recovered_history),
    )
    print("Q:", "我刚才让你记住的数字是多少？")
    print("A:", r.output)


async def main() -> None:
    await demo_in_memory()
    await demo_persist()


if __name__ == "__main__":
    asyncio.run(main())
