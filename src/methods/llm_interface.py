from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from FERMI.src.utils.logging import get_logger

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore[misc,assignment]


class LLMClient:
    """
    OPRO/FERMI 的 LLM 呼叫介面。

    - 有設定 OPENAI key 時，使用 OpenAI chat.completions
    - 未設定 key 時，才 fallback 到本地 prompt 變體
    """

    def __init__(
        self,
        evaluator_model: str = "gpt-4.1-mini",
        optimizer_model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 2,
        request_timeout: float = 30.0,
    ) -> None:
        self.logger = get_logger("fermi.llm")

        self.evaluator_model = evaluator_model
        self.optimizer_model = optimizer_model
        self.api_key_env = api_key_env
        self.max_retries = max(0, int(max_retries))
        self.request_timeout = float(request_timeout)

        self.api_calls = 0
        self.fallback_calls = 0
        self.last_fallback_reason: Optional[str] = None

        self.api_key = os.environ.get(self.api_key_env)
        self.client = None
        if not self.api_key:
            self.last_fallback_reason = f"missing_env:{self.api_key_env}"
            self.logger.warning(
                "OPENAI API key missing (%s). LLM generation falls back to local template.",
                self.api_key_env,
            )
            return

        if OpenAI is None:
            raise RuntimeError(
                "openai SDK is not installed. Install with `pip install openai` before running OPRO/FERMI with API."
            )

        self.client = OpenAI(api_key=self.api_key, timeout=self.request_timeout)

    def generate(
        self,
        prompt: str,
        temperature: float = 1.0,
        max_tokens: int = 256,
        model: Optional[str] = None,
        role: str = "optimizer",
        allow_fallback: bool = True,
    ) -> str:
        if self.client is None:
            self.fallback_calls += 1
            if not allow_fallback:
                raise RuntimeError(self.last_fallback_reason or f"missing_env:{self.api_key_env}")
            return self._fallback(prompt)

        resolved_model = model or (self.optimizer_model if role == "optimizer" else self.evaluator_model)
        req: Dict[str, Any] = {
            "model": resolved_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**req)
                text = ""
                if getattr(resp, "choices", None):
                    text = str(resp.choices[0].message.content or "")
                text = text.strip()
                if not text:
                    raise RuntimeError("OpenAI returned empty response content.")
                self.api_calls += 1
                return text
            except Exception as exc:  # pragma: no cover - depends on runtime/network
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 4))

        self.last_fallback_reason = f"api_error:{type(last_exc).__name__}" if last_exc is not None else "api_error:unknown"
        if allow_fallback:
            self.fallback_calls += 1
            return self._fallback(prompt)

        raise RuntimeError(
            f"OpenAI completion failed for model={resolved_model} after {self.max_retries + 1} attempts"
        ) from last_exc

    def summary(self) -> Dict[str, Any]:
        return {
            "provider": "openai" if self.client is not None else "fallback",
            "api_key_env": self.api_key_env,
            "has_api_key": bool(self.api_key),
            "optimizer_model": self.optimizer_model,
            "evaluator_model": self.evaluator_model,
            "openai_max_retries": self.max_retries,
            "openai_request_timeout": self.request_timeout,
            "api_calls": self.api_calls,
            "fallback_calls": self.fallback_calls,
            "fallback_reason": self.last_fallback_reason,
        }

    @staticmethod
    def _fallback(prompt: str) -> str:
        # 可執行骨架 fallback：回傳可用的 prompt 變體
        return (
            "Refined prompt: prioritize user history consistency, keep output concise, "
            "and avoid unsupported assumptions.\n"
            f"Seed from: {prompt[:120]}"
        )
