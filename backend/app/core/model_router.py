import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from ..config import Settings, get_settings
from .model_policy import read_policy
from .model_capabilities import generation_options, forced_thinking


class ModelRole(str, Enum):
    router = "router"
    paper = "paper"
    code = "code"
    reproduction = "reproduction"
    audit = "audit"  # Compatibility for historical records only.
    reasoning = "reasoning"
    report = "report"


ROLE_MODEL_KIND: dict[ModelRole, str] = {
    ModelRole.router: "router",
    ModelRole.paper: "chat",
    ModelRole.code: "coder",
    ModelRole.reproduction: "coder",
    ModelRole.audit: "reasoning",
    ModelRole.reasoning: "reasoning",
    ModelRole.report: "chat",
}


@dataclass(frozen=True)
class ModelSelection:
    provider: str
    model: str
    base_url: str
    api_key: str
    logical_kind: str

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model)


@dataclass
class ModelCallResult:
    data: dict[str, Any]
    provider: str
    model: str
    usage: dict[str, int]
    fallback_used: bool = False
    error: str = ""


@dataclass
class ModelTextResult:
    text: str
    provider: str
    model: str
    usage: dict[str, int]
    error: str = ""
    finish_reason: str = ""


class ModelRouter:
    """统一多模型调度入口。

    Agent 只声明自己的能力角色，不直接读取密钥或模型 ID。比赛环境切换
    MODEL_PROVIDER=moma 后，同一套 Agent 图会改用 MoMA 网关及其模型池。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @staticmethod
    def _base_url(value: str) -> str:
        base = value.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        return base if base.endswith("/v1") else f"{base}/v1"

    def select(self, role: ModelRole) -> ModelSelection:
        provider = self.settings.model_provider.lower().strip()
        logical_kind = ROLE_MODEL_KIND[role]
        choices = read_policy(self.settings)['models']
        if provider == "moma":
            models = {
                "router": self.settings.moma_model_router or self.settings.moma_model_chat,
                "chat": self.settings.moma_model_chat,
                "coder": self.settings.moma_model_coder or self.settings.moma_model_chat,
                "reasoning": self.settings.moma_model_reasoning or self.settings.moma_model_chat,
            }
            return ModelSelection(
                provider="moma",
                model=choices.get(logical_kind) or models[logical_kind],
                base_url=self._base_url(self.settings.moma_base_url),
                api_key=self.settings.moma_api_key,
                logical_kind=logical_kind,
            )

        models = {
            "router": self.settings.llm_model_chat,
            "chat": self.settings.llm_model_chat,
            "coder": self.settings.llm_model_coder,
            "reasoning": self.settings.llm_model_reasoning or self.settings.llm_model_chat,
        }
        return ModelSelection(
            provider="openai_compatible",
            model=choices.get(logical_kind) or models[logical_kind],
            base_url=self._base_url(self.settings.llm_base_url),
            api_key=self.settings.llm_api_key,
            logical_kind=logical_kind,
        )

    async def complete_json(
        self,
        role: ModelRole,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 1800,
        model_override: str = "",
        extended_output: bool = False,
        non_thinking: bool = False,
        timeout_seconds: int | None = None,
        allow_fallback: bool = True,
    ) -> ModelCallResult:
        primary = self.select(role)
        if model_override:
            primary = ModelSelection(
                provider=primary.provider, model=model_override,
                base_url=primary.base_url, api_key=primary.api_key,
                logical_kind=primary.logical_kind,
            )
        max_tokens = min(max_tokens, max(self.settings.llm_max_tokens_per_call, 16000)
                         if extended_output and role == ModelRole.code
                         else self.settings.llm_max_tokens_per_call)
        if not primary.enabled:
            return ModelCallResult(
                data={}, provider=primary.provider, model=primary.model or "unconfigured",
                usage={}, fallback_used=True, error="model_not_configured",
            )

        candidates = [primary]
        fallback = self.select(ModelRole.paper)
        if allow_fallback and fallback.enabled and fallback.model != primary.model:
            candidates.append(fallback)

        last_error = ""
        for index, selection in enumerate(candidates):
            try:
                data, usage = await self._request_json(
                    selection, system_prompt, user_prompt, max_tokens=max_tokens,
                    non_thinking=non_thinking,
                    timeout_seconds=timeout_seconds or (max(self.settings.llm_timeout_seconds, 120)
                                     if extended_output and role == ModelRole.code
                                     else self.settings.llm_timeout_seconds),
                )
                return ModelCallResult(
                    data=data,
                    provider=selection.provider,
                    model=selection.model,
                    usage=usage,
                    fallback_used=index > 0,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        return ModelCallResult(
            data={}, provider=primary.provider, model=primary.model, usage={},
            fallback_used=True, error=last_error,
        )

    async def _request_json(
        self,
        selection: ModelSelection,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        non_thinking: bool = False,
        timeout_seconds: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        content, usage, finish_reason = await self._request_text(
            selection, system_prompt, user_prompt, max_tokens=max_tokens,
            non_thinking=non_thinking, timeout_seconds=timeout_seconds, json_mode=True,
        )
        if finish_reason == "length":
            raise ValueError("结构化响应被输出上限截断")
        return self._parse_json(content), usage

    async def complete_text(
        self, role: ModelRole, system_prompt: str, user_prompt: str, *,
        max_tokens: int = 1800, model_override: str = "", non_thinking: bool = False,
        timeout_seconds: int = 120,
    ) -> ModelTextResult:
        selection = self.select(role)
        if model_override:
            selection = ModelSelection(selection.provider, model_override, selection.base_url,
                                       selection.api_key, selection.logical_kind)
        if not selection.enabled:
            return ModelTextResult("", selection.provider, selection.model, {}, "model_not_configured")
        try:
            content, usage, finish_reason = await self._request_text(
                selection, system_prompt, user_prompt, max_tokens=min(max_tokens, 3000),
                non_thinking=non_thinking, timeout_seconds=timeout_seconds, json_mode=False,
            )
            return ModelTextResult(content, selection.provider, selection.model, usage,
                                   finish_reason=finish_reason)
        except Exception as exc:
            return ModelTextResult("", selection.provider, selection.model, {}, f"{type(exc).__name__}: {exc}")

    async def _request_text(
        self,
        selection: ModelSelection,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        non_thinking: bool = False,
        timeout_seconds: int | None = None,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, int], str]:
        payload = {
            "model": selection.model,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(generation_options(selection.model,max_tokens,non_thinking,self.settings.llm_max_tokens_per_call))
        headers = {
            "Authorization": f"Bearer {selection.api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(timeout_seconds or self.settings.llm_timeout_seconds, connect=15)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{selection.base_url}/chat/completions", headers=headers, json=payload
            )
            if response.status_code == 400:
                payload.pop("response_format", None)
                if not forced_thinking(selection.model):
                    payload.pop("thinking", None)
                    payload.pop("reasoning_effort", None)
                response = await client.post(
                    f"{selection.base_url}/chat/completions", headers=headers, json=payload
                )
            response.raise_for_status()
            body = response.json()
        choice = body["choices"][0]
        message = choice["message"]
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"模型返回空内容，finish_reason={choice.get('finish_reason', 'unknown')}"
            )
        usage = body.get("usage") or {}
        return content, {
            key: int(value) for key, value in usage.items() if isinstance(value, (int, float))
        }, str(choice.get("finish_reason") or "")

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for position, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(cleaned[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise ValueError("模型没有返回可解析的 JSON 对象")


_router: ModelRouter | None = None


def get_model_router() -> ModelRouter:
    global _router
    if _router is None:
        _router = ModelRouter()
    return _router
