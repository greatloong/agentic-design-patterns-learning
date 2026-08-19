"""0.3 显式 ReAct 循环（纯 Graph，无 Agent）—— 把「想—做—观察」拆成节点

对照 LangGraph 的 ReAct（agent 节点 + ToolNode + shouldContinue 条件边的循环）：
    Think（想：选一个动作）──> Act（做：执行工具，记录观察）──┐
        ▲                                                      │
        └──────────────────────────────────────────────────────┘
    Think 若决定收尾就 `End`。这正是 LangGraph 03 的同构版本。

无 Agent 怎么做 ReAct：完全手写。Think 节点直接调 model、要一段 JSON（下一步动作），
自己 `parse_json` 校验；Act 节点用普通 Python 函数执行"工具"、把观察写回 state，再回到
Think。由图来驱动循环——这就是 `@agent.tool` 在底层替你做的事，现在摊开自己做。

- **依赖注入**：工具要用的外部数据（汇率表）通过 `deps` 注入，节点用 `ctx.deps` 读
  ——生产级写法，不靠全局变量。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.graph import Graph

from shared.model import ask, get_model, parse_json

model = get_model()

_MAX_STEPS = 6  # 兜底：防止模型陷入死循环


@dataclass
class Deps:
    """运行时依赖：用一个假的汇率表模拟「外部数据源」。"""

    rates_to_cny: dict[str, float]


class Action(BaseModel):
    """模型每一步要做的决定（结构化输出 = 程序可判断的离散信号）。"""

    reasoning: str = Field(description="一句话说明这一步为什么这么做")
    tool: Literal["convert_currency", "days_between", "finish"]
    args: dict[str, str] = Field(default_factory=dict)


_THINK_SYSTEM = (
    "你是出行助理，按 ReAct 方式一步步解决问题：每次只决定**下一个动作**，并只输出一个 JSON。"
    "可用动作：\n"
    '- 换算汇率：{"reasoning": "...", "tool": "convert_currency", '
    '"args": {"amount": "2000", "currency": "USD"}}\n'
    '- 计算天数：{"reasoning": "...", "tool": "days_between", '
    '"args": {"start": "2026-07-01", "end": "2026-07-15"}}\n'
    '- 收尾：{"reasoning": "...", "tool": "finish", "args": {"answer": "最终答案"}}\n'
    "信息齐了就用 finish。不要 markdown、不要编造数字。"
)


@dataclass
class ReactState:
    task: str
    observations: list[str] = field(default_factory=list)
    answer: str = ""
    steps: int = 0


def _dispatch(deps: Deps, tool: str, args: dict[str, str]) -> str:
    """真正执行工具的地方（对照 LangGraph 的 ToolNode）。"""
    if tool == "convert_currency":
        amount = float(args.get("amount", "0"))
        currency = args.get("currency", "").upper()
        rate = deps.rates_to_cny.get(currency)
        if rate is None:
            return f"未知货币 {currency}，支持：{list(deps.rates_to_cny)}"
        return f"{amount} {currency} = {amount * rate:.2f} CNY"
    if tool == "days_between":
        start = date.fromisoformat(args["start"])
        end = date.fromisoformat(args["end"])
        return f"{args['start']} 到 {args['end']} 相差 {(end - start).days} 天"
    return f"未知工具 {tool}"


@dataclass
class Think(BaseNode[ReactState, Deps, str]):
    """想：让模型选下一个动作。返回 `Act | End` = ReAct 的条件边。"""

    async def run(self, ctx: GraphRunContext[ReactState, Deps]) -> Act | End[str]:
        ctx.state.steps += 1
        history = "\n".join(ctx.state.observations) or "（暂无）"
        prompt = (
            f"任务：{ctx.state.task}\n已有观察：\n{history}\n\n请决定下一步动作（只输出 JSON）。"
        )
        action = parse_json(await ask(model, prompt, system=_THINK_SYSTEM), Action)
        print(f"  [想#{ctx.state.steps}] {action.reasoning} -> {action.tool}")

        if action.tool == "finish" or ctx.state.steps >= _MAX_STEPS:
            answer = action.args.get("answer") or "（已达最大步数，提前收尾）"
            ctx.state.answer = answer
            return End(answer)
        return Act(tool=action.tool, args=action.args)


@dataclass
class Act(BaseNode[ReactState, Deps]):
    """做：执行工具、把观察写回 state，然后无条件回到 Think（循环边）。"""

    tool: str
    args: dict[str, str]

    async def run(self, ctx: GraphRunContext[ReactState, Deps]) -> Think:
        observation = _dispatch(ctx.deps, self.tool, self.args)
        print(f"  [做]   {self.tool}({self.args}) -> {observation}")
        ctx.state.observations.append(f"{self.tool}({self.args}) = {observation}")
        return Think()


react_graph = Graph(nodes=(Think, Act), state_type=ReactState, run_end_type=str)


async def main() -> None:
    print("=== react_graph 结构（mermaid，看 Think<->Act 的循环）===")
    print(react_graph.mermaid_code(start_node=Think))

    deps = Deps(rates_to_cny={"USD": 7.18, "JPY": 0.047, "EUR": 7.76})
    question = (
        "我计划 2026-07-01 到 2026-07-15 去日本，预算 2000 USD。"
        "帮我算一下这是多少人民币，以及行程一共多少天。"
    )
    print(f"\n用户：{question}\n")

    result = await react_graph.run(Think(), state=ReactState(task=question), deps=deps)
    print("\n=== 最终回答 ===")
    print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
