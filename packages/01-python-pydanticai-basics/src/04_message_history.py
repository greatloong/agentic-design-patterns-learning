"""0.4 消息历史与持久化 —— 对照 LangGraph 的 Checkpointer

LangGraph 用 Checkpointer 把「图的 State」按 thread_id 存下来，下次自动恢复，实现多轮记忆。
Pydantic AI 的设计更轻量、也更显式：

- 一次 `agent.run()` 是一个 **run**；一段对话（conversation）可以由多个 run 串起来。
- 串联的方式：把上一轮的消息 `result.new_messages()` 通过 `message_history=` 传给下一轮。
  这就是「短期记忆」——模型能记住"他"指代谁。
- 想跨进程/重启也记得（持久化）：用 `ModelMessagesTypeAdapter` 把消息列表序列化成 JSON
  存盘/存库，下次读回来继续传给 `message_history`。这相当于手动版的 Checkpointer。

对比记忆：LangGraph 的 Checkpointer 是"框架托管、自动存取"；
Pydantic AI 是"你拿到消息、自己决定存哪、怎么存"——更少魔法，更好控制。
"""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

from shared.model import get_model

agent = Agent(
    get_model(),
    instructions="你是一个会记住上下文的助手，回答简洁。",
)

_STORE = Path(__file__).resolve().parent / "conversation.json"


def demo_in_memory() -> None:
    """(1) 内存中的多轮对话：靠 message_history 串联。"""
    print("=== (1) 内存多轮对话 ===")
    r1 = agent.run_sync("我叫王悦龙，正在学 Pydantic AI。")
    print("Q1: 我叫王悦龙，正在学 Pydantic AI。")
    print("A1:", r1.output)

    # 把第一轮的消息传进来，模型才知道"我"是谁、在学什么
    r2 = agent.run_sync("我叫什么名字？在学什么？", message_history=r1.new_messages())
    print("\nQ2: 我叫什么名字？在学什么？")
    print("A2:", r2.output)


def demo_persist() -> None:
    """(2) 落盘持久化：序列化 -> 写文件 -> 读回 -> 继续对话。"""
    print("\n=== (2) 持久化到磁盘再恢复 ===")

    # 第一段会话并存盘
    r1 = agent.run_sync("记住一个数字：42。")
    json_bytes = ModelMessagesTypeAdapter.dump_json(r1.all_messages())
    _STORE.write_bytes(json_bytes)
    print(f"已把对话写入 {_STORE.name}（{len(json_bytes)} 字节）")

    # 模拟"重启进程"：从磁盘读回历史
    loaded = ModelMessagesTypeAdapter.validate_json(_STORE.read_bytes())
    r2 = agent.run_sync("我刚才让你记住的数字是多少？", message_history=loaded)
    print("Q: 我刚才让你记住的数字是多少？")
    print("A:", r2.output)


def main() -> None:
    demo_in_memory()
    demo_persist()


if __name__ == "__main__":
    main()
