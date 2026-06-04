"""统一的模型工厂：所有示例都通过这里拿到一个指向 DashScope（阿里云百炼）的 pydantic-ai 模型。

对照 TS 仓库里 `src/shared/` 的做法——把"用哪个 LLM、怎么鉴权"收敛到一处，
示例文件只关心 pydantic-ai 的概念本身。

这里用 DashScope 提供的 DeepSeek 模型（兼容 OpenAI 协议）。
因为是自定义 base_url 的 OpenAI 兼容端点，所以用通用的 `OpenAIProvider`
（传 base_url + api_key），而不是内置的 DeepSeekProvider。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
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
