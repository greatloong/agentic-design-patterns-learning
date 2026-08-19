"""统一的模型工厂 + **底层 model 调用助手**（无 Agent）。

本系列是"纯 pydantic-graph"实现：流程编排靠图，单步 LLM 调用**直接打到底层
`Model.request()`**，不经过 `Agent`。`Agent` 本质就是 `Model` 上的一层封装
（帮你管消息、工具、结构化输出、重试）；这里我们把那层拿掉，亲手做这些事，
好彻底展示"graph 不依赖 Agent"。

这个模块提供两类东西：
- `get_model()`：构造指向 DashScope（OpenAI 兼容端点）的 `OpenAIChatModel`。
- 一组**薄助手**（`ask` / `chat` / `stream_text` / `parse_json` 等）：对 `model.request()`
  的极薄封装，仅做"拼请求消息、抽取文本、解析 JSON"这些纯机械活，**不是 Agent**。

为什么能绕开思考模型的坑：我们不向模型声明任何工具、也不用 `tool_choice`，
结构化输出靠"提示词要 JSON + 自己解析校验"，所以 DashScope 思考模式下不会再报 400。
"""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPartDelta
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# 包根目录下的 .env（src/shared/model.py -> parents[2] 即包根）
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PACKAGE_ROOT / ".env")

_DEFAULT_MODEL = "deepseek-v4-pro"
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_model() -> OpenAIChatModel:
    """构造一个指向 DashScope DeepSeek 的模型实例。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 DASHSCOPE_API_KEY。请在 packages/01-python-pydanticai-basics/.env 中配置"
            "（可参考 .env.example）。"
        )

    base_url = os.environ.get("DASHSCOPE_BASE_URL", _DEFAULT_BASE_URL)
    model_name = os.environ.get("PYDANTICAI_MODEL", _DEFAULT_MODEL)

    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(base_url=base_url, api_key=api_key),
    )


# ---------------------------------------------------------------------------
# 底层 model 调用助手（对 Model.request / request_stream 的极薄封装，非 Agent）
# ---------------------------------------------------------------------------

# 默认请求参数：不声明工具、纯文本输出。这样思考模型不会触发 tool_choice=required 的 400。
_PARAMS = ModelRequestParameters()

T = TypeVar("T", bound=BaseModel)


def user_msg(text: str, *, system: str | None = None) -> ModelRequest:
    """把（可选的 system 指令 + 用户输入）打包成一条 ModelRequest。"""
    parts: list = []
    if system:
        parts.append(SystemPromptPart(content=system))
    parts.append(UserPromptPart(content=text))
    return ModelRequest(parts=parts)


def text_of(resp: ModelResponse) -> str:
    """从模型响应里抽取纯文本（思考模型会带 ThinkingPart，这里只取 TextPart）。"""
    return "".join(p.content for p in resp.parts if isinstance(p, TextPart))


async def chat(model: OpenAIChatModel, messages: Sequence[ModelMessage]) -> ModelResponse:
    """最底层的一次调用：把整段消息历史发给模型，拿回一个 ModelResponse。"""
    return await model.request(list(messages), None, _PARAMS)


async def ask(model: OpenAIChatModel, user: str, *, system: str | None = None) -> str:
    """单轮便捷调用：system + user -> 纯文本答案。"""
    resp = await chat(model, [user_msg(user, system=system)])
    return text_of(resp)


async def stream_text(
    model: OpenAIChatModel, messages: Sequence[ModelMessage]
) -> AsyncIterator[str]:
    """流式调用：逐段 yield 最终答案的文本增量（只吐 TextPart，过滤思维链）。

    文本会分两种事件到来：新段开始的 PartStartEvent（含首块内容）+ 后续的 PartDeltaEvent。
    两种都要接，否则会丢掉开头第一块。
    """
    async with model.request_stream(list(messages), None, _PARAMS) as stream:
        async for event in stream:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                if event.part.content:
                    yield event.part.content
            elif (
                isinstance(event, PartDeltaEvent)
                and isinstance(event.delta, TextPartDelta)
                and event.delta.content_delta
            ):
                yield event.delta.content_delta


def parse_json(text: str, model_cls: type[T]) -> T:
    """把模型返回的 JSON 文本解析+校验成给定的 Pydantic 模型。

    模型有时会用 ```json 包裹或在前后加解释，这里抓第一个 {...} 块再校验
    ——这就是 Agent 的"结构化输出"在底层替你做的事，现在我们自己做。
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        raise ValueError(f"模型未返回 JSON：{text!r}")
    return model_cls.model_validate_json(match.group(0))
