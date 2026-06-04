"""0.3 ReAct Agent —— 工具调用与依赖注入

对照 LangGraph 的 ReAct（ToolNode + shouldContinue 循环）：
- LangGraph 里你要自己搭「模型节点 -> 判断有无 tool_calls -> ToolNode 执行 -> 回到模型」的循环。
- Pydantic AI 把这整个 ReAct 循环**内置**了：你只管用 `@agent.tool` 注册工具，
  模型决定调用 -> 框架执行 -> 结果回喂 -> 模型继续，直到产出最终答案。

两个关键概念：
- `@agent.tool`：注册一个工具，函数签名（参数 + 类型 + docstring）会自动转成给模型的 schema。
  docstring 很重要——模型靠它判断「何时、如何」调用。
- `RunContext[Deps]` + `deps_type`：依赖注入。把数据库连接、API client、当前用户等
  运行时依赖通过 `ctx.deps` 传进工具，而不是用全局变量。这就是生产级写法。
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from shared.model import get_model


@dataclass
class Deps:
    """运行时依赖：这里用一个假的汇率表模拟"外部数据源"。"""

    rates_to_cny: dict[str, float]


agent = Agent(
    get_model(),
    deps_type=Deps,
    instructions=(
        "你是出行助理。需要汇率时调用 convert_currency，"
        "需要当前天数差时调用 days_between。不要自己编造数字。"
    ),
)


@agent.tool
def convert_currency(ctx: RunContext[Deps], amount: float, currency: str) -> str:
    """把指定金额的外币换算成人民币（CNY）。

    Args:
        amount: 外币金额。
        currency: 货币代码，如 USD、JPY、EUR。
    """
    rate = ctx.deps.rates_to_cny.get(currency.upper())
    if rate is None:
        # 抛 ValueError 会被框架转成给模型的错误提示；也可用 ModelRetry 明确要求重试
        return f"未知货币 {currency}，支持：{list(ctx.deps.rates_to_cny)}"
    return f"{amount} {currency.upper()} = {amount * rate:.2f} CNY"


@agent.tool_plain
def days_between(start: str, end: str) -> str:
    """计算两个日期之间相差的天数。日期格式 YYYY-MM-DD。"""
    from datetime import date

    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    return f"{start} 到 {end} 相差 {(e - s).days} 天"


def main() -> None:
    deps = Deps(rates_to_cny={"USD": 7.18, "JPY": 0.047, "EUR": 7.76})
    question = (
        "我计划 2026-07-01 到 2026-07-15 去日本，预算 2000 USD。"
        "帮我算一下这是多少人民币，以及行程一共多少天。"
    )
    print(f"用户：{question}\n")

    result = agent.run_sync(question, deps=deps)
    print("=== 最终回答 ===")
    print(result.output)

    print("\n=== 完整消息轨迹（看 ReAct 循环里都发生了什么）===")
    for msg in result.all_messages():
        for part in msg.parts:
            kind = type(part).__name__
            preview = str(getattr(part, "content", getattr(part, "args", "")))[:80]
            print(f"  - {kind}: {preview}")


if __name__ == "__main__":
    main()
