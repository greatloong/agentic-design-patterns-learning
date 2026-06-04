"""0.1 Agent 与结构化输出 —— Pydantic AI 的"地基"

对照 LangGraph 的 State / Node / Edge：
- LangGraph 用「图」做核心抽象：你显式声明 State、写 Node 函数、连 Edge。
- Pydantic AI 反过来，核心抽象是 **Agent**：一个把「模型 + 指令 + 工具 + 输出类型」
  打包起来的容器。它内部其实也跑一张 pydantic-graph 的状态机，但默认对你隐藏，
  你只需 `agent.run_sync(...)` 就能从头跑到尾。

本节只看两件最基础的事：
  (1) 最朴素的纯文本 Agent；
  (2) Pydantic AI 的招牌能力——用 `output_type` 强制模型返回**结构化、已校验**的数据。
      模型若给出不合 schema 的内容，框架会自动把校验错误回喂给模型让它重试。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from pydantic_ai import Agent, PromptedOutput

from shared.model import get_model


# ---------- (1) 最朴素：纯文本输出 ----------
# 不指定 output_type 时，默认就是 str。
text_agent = Agent(
    get_model(),
    instructions="你是一个简洁的助手，回答控制在一句话内。",
)


# ---------- (2) 结构化输出：用 Pydantic 模型当 output_type ----------
class CityInfo(BaseModel):
    """一个城市的结构化信息。字段的注释会作为 schema 描述传给模型。"""

    name: str = Field(description="城市名")
    country: str = Field(description="所属国家")
    population_million: float = Field(description="人口（百万为单位）")
    famous_for: list[str] = Field(description="3 个最知名的标签")


# 关于输出模式（output mode）的一个重要实战点：
# pydantic-ai 默认用「输出工具 + tool_choice=required」来强制结构化输出（ToolOutput 模式）。
# 但我们用的 deepseek-v4-pro 是**思考模型**，DashScope 在思考模式下不允许 tool_choice=required，
# 会直接报 400。所以这里改用 PromptedOutput 模式：把 JSON schema 写进提示词，让模型返回
# JSON 文本，pydantic-ai 再解析+校验。它不依赖 tool_choice，兼容任何模型（含思考模型）。
#   - 普通（非思考）模型：可省略 PromptedOutput，直接 output_type=CityInfo 即可。
city_agent = Agent(
    get_model(),
    output_type=PromptedOutput(CityInfo),  # 返回值会被校验并转成 CityInfo
    instructions="你是地理百科助手，根据用户问的城市给出结构化信息。",
)


def main() -> None:
    print("=== (1) 纯文本 Agent ===")
    r1 = text_agent.run_sync("用一句话介绍一下杭州。")
    print(r1.output)  # 类型是 str
    print(f"[usage] {r1.usage}\n")  # 新版里 usage 是属性，不是方法

    print("=== (2) 结构化输出 Agent ===")
    r2 = city_agent.run_sync("介绍一下杭州")
    info = r2.output  # 类型是 CityInfo（IDE/类型检查器都能识别）
    print(repr(info))
    print(f"人口：{info.population_million} 百万")
    print(f"标签：{', '.join(info.famous_for)}")


if __name__ == "__main__":
    main()
