"""0.2 条件路由 —— 根据输入分发到不同处理路径

对照 LangGraph 的「条件边」：
- LangGraph 里你写一个 router 函数，返回下一个 Node 的名字，由条件边决定走向。
- Pydantic AI 没有显式的图/边给你连，但「路由」本质是一样的：
  先用一个**分类 Agent** 把输入归类（用结构化输出锁定枚举值，避免模型自由发挥），
  再用普通 Python 的 if/分发，把请求交给对应的**专家 Agent**。

要点：分类这一步用 `output_type` 限定为有限取值（这里用 Literal），
就把「LLM 自然语言」收敛成了「程序可判断的离散信号」——这是把 LLM 接进控制流的关键技巧。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent, PromptedOutput

from shared.model import get_model

model = get_model()

Category = Literal["技术支持", "账单", "闲聊"]


class Routing(BaseModel):
    category: Category = Field(description="用户问题所属类别")
    reason: str = Field(description="判断理由，一句话")


# 分类器：只负责把输入归到三类之一
# 用 PromptedOutput 而非默认的工具输出模式，原因见 01_agent_basics.py 的说明
# （deepseek-v4-pro 思考模式不允许 tool_choice=required）。
router_agent = Agent(
    model,
    output_type=PromptedOutput(Routing),
    instructions=(
        "你是客服分诊员。把用户消息归类为「技术支持」「账单」或「闲聊」之一，"
        "并给出简短理由。"
    ),
)

# 三个专家 Agent，各自有不同人设
tech_agent = Agent(
    model,
    instructions="你是技术支持工程师，给出可操作的排查步骤，简洁专业。",
)
billing_agent = Agent(
    model,
    instructions="你是账单专员，耐心解释费用与退款政策。",
)
chat_agent = Agent(
    model,
    instructions="你是亲切的闲聊伙伴，轻松回应即可。",
)

_HANDLERS: dict[Category, Agent] = {
    "技术支持": tech_agent,
    "账单": billing_agent,
    "闲聊": chat_agent,
}


def handle(user_msg: str) -> str:
    decision = router_agent.run_sync(user_msg).output
    print(f"  [路由] -> {decision.category}（{decision.reason}）")
    expert = _HANDLERS[decision.category]
    return expert.run_sync(user_msg).output


def main() -> None:
    samples = [
        "我的 App 一打开就闪退，怎么办？",
        "上个月为什么被扣了两次会员费？",
        "今天天气不错，跟你聊两句～",
    ]
    for msg in samples:
        print(f"\n用户：{msg}")
        answer = handle(msg)
        print(f"回复：{answer}")


if __name__ == "__main__":
    main()
