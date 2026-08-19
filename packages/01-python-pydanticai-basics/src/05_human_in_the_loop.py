"""0.5 Human-in-the-Loop —— 在节点处「中断」，等人类拍板再恢复

对照 LangGraph 的 `interrupt()`（在节点暂停、配合 Checkpointer 存现场、人工决策后 resume）：
- pydantic-graph 用 `graph.iter()` + `run.next()` **逐节点驱动**图：你可以在任意一步停下来。
- 关键时机由 `run.next()` 的语义决定：它会运行「当前节点」并把**下一个待运行的节点**返回给你。
  所以当返回值是 `Approve`（高危操作前的审批节点）时，它**还没执行**——这就是天然的中断点。
  我们在此刻把图状态交给人类、收集决策、写回 State，再继续 `run.next()` 恢复执行。

本例：列文件随便看，但「删除文件」必须人工批准。审批用规则模拟（删 .env 一律拒绝，
其余批准）；换成 input() 就是真人交互。同时把快照写进 `FileStatePersistence`——
配合它，这套「暂停/恢复」就能跨进程、跨重启（真正的 durable interrupt）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_graph import BaseNode, End, GraphRunContext
from pydantic_graph.graph import Graph
from pydantic_graph.persistence.file import FileStatePersistence

_STORE = Path(__file__).resolve().parent / "graph_hitl.json"


@dataclass
class HitlState:
    request: str
    targets: list[str] = field(default_factory=list)  # 待删除的文件
    decisions: dict[str, bool] = field(default_factory=dict)  # 人类的批准结果
    log: list[str] = field(default_factory=list)


@dataclass
class Plan(BaseNode[HitlState]):
    """只读节点：列文件、并根据请求定出「待删除清单」（无需审批）。"""

    async def run(self, ctx: GraphRunContext[HitlState]) -> Approve:
        files = ["config.json", "README.md", ".env", "app.log"]
        ctx.state.log.append(f"列出文件：{', '.join(files)}")
        # demo 里简单地把请求里出现的文件名挑出来当删除目标
        ctx.state.targets = [f for f in files if f in ctx.state.request]
        return Approve()


@dataclass
class Approve(BaseNode[HitlState]):
    """审批节点 = 中断点。它本身不做事——人类的决策在「暂停」时已写进 state.decisions。"""

    async def run(self, ctx: GraphRunContext[HitlState]) -> Perform:
        return Perform()


@dataclass
class Perform(BaseNode[HitlState, None, str]):
    """按已批准的决策真正执行删除。"""

    async def run(self, ctx: GraphRunContext[HitlState]) -> End[str]:
        for target in ctx.state.targets:
            if ctx.state.decisions.get(target):
                ctx.state.log.append(f"已删除 {target}")
            else:
                ctx.state.log.append(f"拒绝删除 {target}（未获批准）")
        return End("\n".join(ctx.state.log))


hitl_graph = Graph(nodes=(Plan, Approve, Perform), state_type=HitlState)


def review(targets: list[str]) -> dict[str, bool]:
    """模拟人类审批：真实场景换成 UI 弹窗或 input()。"""
    decisions: dict[str, bool] = {}
    for target in targets:
        approved = target != ".env"  # .env 受保护，禁止删除
        mark = "批准" if approved else "拒绝"
        print(f"  [审批] {mark}删除 {target}")
        decisions[target] = approved
    return decisions


async def main() -> None:
    print("=== hitl_graph 结构（mermaid）===")
    print(hitl_graph.mermaid_code(start_node=Plan))

    if _STORE.exists():
        _STORE.unlink()

    request = "先列出文件，然后把 app.log 和 .env 都删掉。"
    print(f"\n用户：{request}\n")

    state = HitlState(request=request)
    persistence = FileStatePersistence(_STORE)
    persistence.set_graph_types(hitl_graph)

    async with hitl_graph.iter(Plan(), state=state, persistence=persistence) as run:
        # 逐步推进，直到「即将运行」审批节点 —— 这就是 interrupt 时机
        node = await run.next()  # 运行 Plan，返回下一个待运行节点
        while not isinstance(node, (Approve, End)):
            node = await run.next()

        if isinstance(node, Approve):
            print("=== 触发人工审批（图已暂停在 Approve 之前）===")
            print(f"  待删除：{state.targets}")
            # 收集人类决策，写回 State —— 相当于 LangGraph 的 Command(resume=...)
            state.decisions = review(state.targets)
            print("=== 恢复执行 ===")

        # 继续把图跑到 End
        while not isinstance(node, End):
            node = await run.next()

    assert run.result is not None  # 图已跑到 End，result 必有值
    print("\n=== 最终结果 ===")
    print(run.result.output)


if __name__ == "__main__":
    asyncio.run(main())
