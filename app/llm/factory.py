from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.llm.deepseek_client import DeepSeekClient
from app.llm.mock_client import MockLLMClient
from app.llm.volcengine_client import VolcengineArkClient


def create_llm_client(provider: str | None = None):
    load_dotenv(Path.cwd() / ".env", override=False)
    provider = (provider or os.getenv("PATCHHARNESS_LLM_PROVIDER", "deepseek")).lower()
    if provider == "deepseek":
        return DeepSeekClient()
    if provider in {"volcengine", "ark", "volcengine_ark"}:
        return VolcengineArkClient()
    if provider == "mock":
        return MockLLMClient()
    raise ValueError(f"Unknown LLM provider: {provider}")
