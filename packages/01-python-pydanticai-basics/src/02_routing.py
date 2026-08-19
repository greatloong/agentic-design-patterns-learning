"""0.2 条件边路由（纯 Graph，无 Agent）—— 把输入分发到不同处理路径

对照 LangGraph 的「条件边」（`addConditionalEdges`）：
- 分流靠**节点 `run()` 的联合返回类型**：`Classify.run()` 返回 `Tech | Billing | Chat`，
  框架就把它理解成「该节点有三条出边」，运行时按你实际 return 的实例走对应分支。

无 Agent 怎么做分类：直接调 model 要一个 JSON（含类别枚举 + 理由），自己 `parse_json` 校验，
再用普通 if 分发。把「LLM 自然语言」收敛成「有限枚举」是接进控制流的关键技巧，
只是这次「分发」体现成了图里实打实的几条边。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.graph import Graph

from shared.model import ask, get_model, parse_json

model = get_model()

Category = Literal["技术支持", "账单", "闲聊"]


class Routing(BaseModel):
    category: Category = Field(description="用户问题所属类别")
    reason: str = Field(description="判断理由，一句话")


_ROUTER_SYSTEM = (
    "你是客服分诊员。把用户消息归类为「技术支持」「账单」「闲聊」之一。"
    "只输出 JSON，不要 markdown、不要解释，形如："
    '{"category": "技术支持", "reason": "一句话理由"}'
)

# 三个专家的人设（system 指令）
_TECH_SYSTEM = "你是技术支持工程师，给出可操作的排查步骤，简洁专业。"
_BILLING_SYSTEM = "你是账单专员，耐心解释费用与退款政策。"
_CHAT_SYSTEM = "你是亲切的闲聊伙伴，轻松回应即可。"


@dataclass
class RouteState:
    user_msg: str
    category: str = ""
    reason: str = ""
    answer: str = ""


@dataclass
class Classify(BaseNode[RouteState]):
    """分诊节点：联合返回类型 = 三条出边（条件边）。"""

    async def run(self, ctx: GraphRunContext[RouteState]) -> Tech | Billing | Chat:
        decision = parse_json(await ask(model, ctx.state.user_msg, system=_ROUTER_SYSTEM), Routing)
        ctx.state.category = decision.category
        ctx.state.reason = decision.reason
        print(f"  [路由] -> {decision.category}（{decision.reason}）")
        if decision.category == "技术支持":
            return Tech()
        if decision.category == "账单":
            return Billing()
        return Chat()


@dataclass
class Tech(BaseNode[RouteState, None, str]):
    async def run(self, ctx: GraphRunContext[RouteState]) -> End[str]:
        answer = await ask(model, ctx.state.user_msg, system=_TECH_SYSTEM)
        ctx.state.answer = answer
        return End(answer)


@dataclass
class Billing(BaseNode[RouteState, None, str]):
    async def run(self, ctx: GraphRunContext[RouteState]) -> End[str]:
        answer = await ask(model, ctx.state.user_msg, system=_BILLING_SYSTEM)
        ctx.state.answer = answer
        return End(answer)


@dataclass
class Chat(BaseNode[RouteState, None, str]):
    async def run(self, ctx: GraphRunContext[RouteState]) -> End[str]:
        answer = await ask(model, ctx.state.user_msg, system=_CHAT_SYSTEM)
        ctx.state.answer = answer
        return End(answer)


route_graph = Graph(nodes=(Classify, Tech, Billing, Chat), state_type=RouteState)


async def main() -> None:
    print("=== route_graph 结构（mermaid，看 Classify 的三条分支）===")
    print(route_graph.mermaid_code(start_node=Classify))

    samples = [
        "我的 App 一打开就闪退，怎么办？",
        "上个月为什么被扣了两次会员费？",
        "今天天气不错，跟你聊两句～",
    ]
    for msg in samples:
        print(f"\n用户：{msg}")
        result = await route_graph.run(Classify(), state=RouteState(user_msg=msg))
        print(f"回复：{result.output}")


if __name__ == "__main__":
    asyncio.run(main())
