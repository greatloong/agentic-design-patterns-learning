"""0.5 Human-in-the-Loop —— 危险操作前先让人类拍板

对照 LangGraph 的 interrupt（在节点处中断、等人工输入再恢复）：
- LangGraph 用 `interrupt()` 暂停图，配合 Checkpointer 保存现场，人工决策后 resume。
- Pydantic AI 用 **deferred tools（延迟工具）** 里的「需要审批」一类来实现，分两步：

  第 1 步（stop-the-world）：把高危工具标记 `requires_approval=True`，并把 Agent 的
  `output_type` 加上 `DeferredToolRequests`。当模型想调用该工具时，run **不会执行工具**，
  而是直接结束，返回一个 `DeferredToolRequests`（里面是待审批的工具调用清单）。

  第 2 步（resume）：你把清单拿给人类看，收集"批准/拒绝"，打包成 `DeferredToolResults`，
  连同**原对话的 message_history** 再次 run。批准的工具才真正执行，被拒的会把拒绝理由回喂模型。

本例：列文件随便看，但「删除文件」必须人工批准。为可自动跑通，审批环节用规则模拟
（删 .env 一律拒绝，其余批准）；把它换成 input() 就是真人交互。
"""

from __future__ import annotations

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ToolDenied,
)

from shared.model import get_model

# 关键：output_type 必须显式包含 DeferredToolRequests，
# 这样类型系统才知道一次 run 可能以"待审批"的形态结束。
agent = Agent(
    get_model(),
    output_type=[str, DeferredToolRequests],
    instructions="你是文件助手。可以列出文件和删除文件。删除前不要啰嗦，直接调用工具。",
)


@agent.tool_plain
def list_files() -> str:
    """列出当前目录下的文件（只读，无需审批）。"""
    return "config.json, README.md, .env, app.log"


@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    """删除指定文件。高危操作，必须经过人工审批。"""
    # 能执行到这里，说明审批已通过
    return f"已删除文件：{path}"


def review(requests: DeferredToolRequests) -> DeferredToolResults:
    """模拟人类审批。真实场景这里换成 UI 弹窗或 input()。"""
    results = DeferredToolResults()
    for call in requests.approvals:
        # call.args 可能是 dict 也可能是 JSON 字符串，args_as_dict() 统一成 dict
        path = call.args_as_dict().get("path")
        if call.tool_name == "delete_file" and path == ".env":
            print(f"  [审批] 拒绝删除 {path}（受保护文件）")
            results.approvals[call.tool_call_id] = ToolDenied(".env 受保护，禁止删除")
        else:
            print(f"  [审批] 批准 {call.tool_name}({path})")
            results.approvals[call.tool_call_id] = True
    return results


def main() -> None:
    prompt = "先列出文件，然后把 app.log 和 .env 都删掉。"
    print(f"用户：{prompt}\n")

    result = agent.run_sync(prompt)

    # 模型若想调用需审批的工具，run 会以 DeferredToolRequests 结束
    if isinstance(result.output, DeferredToolRequests):
        print("=== 触发人工审批 ===")
        for call in result.output.approvals:
            print(f"  待批准：{call.tool_name} args={call.args}")

        decisions = review(result.output)

        # 第 2 步：带着原 message_history + 审批结果，恢复运行
        result = agent.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=decisions,
        )

    print("\n=== 最终回答 ===")
    print(result.output)


if __name__ == "__main__":
    main()
