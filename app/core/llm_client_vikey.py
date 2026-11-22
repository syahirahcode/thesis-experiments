"""
VIKEY (OpenAI-compatible) structured-generation client.

Drop-in for `GoogleGenAIClient`: same `generate_structured()` contract, same
tool-call loop and the same `StructuredGeneration` / `StructuredGenerationError`
telemetry objects. Only the transport differs (HTTP + OpenAI chat-completions).

`app/core/llm_client.py` is not modified; this module imports its dataclasses.

.env:
    LLM_OPTION=vikey
    VIKEY_URL=https://api.vikey.ai/v1
    VIKEY_KEY=vk-xxxx
    VIKEY_MODEL=deepseek-v4.1-flash
"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from app.core.config import Settings, get_settings
from app.core.llm_client import (
    GoogleGenAIClient,
    LLMClientProtocol,
    StructuredGeneration,
    StructuredGenerationError,
    ToolInvoker,
)
from app.schemas.analysis import LLMRoundTrace, ToolCallRecord

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _openai_parameters(value: Any) -> Any:
    """Gemini function declarations use UPPERCASE types; OpenAI needs lowercase."""
    if isinstance(value, dict):
        return {
            key: (
                str(item).lower()
                if key == "type" and isinstance(item, str)
                else _openai_parameters(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_openai_parameters(item) for item in value]
    return value


def openai_tools(declarations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": declaration["name"],
                "description": declaration.get("description", ""),
                "parameters": _openai_parameters(declaration.get("parameters", {})),
            },
        }
        for declaration in declarations
    ]


def extract_json(text: str) -> str:
    """Models often wrap JSON in ``` fences or add a sentence before it."""
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        if candidate.rstrip().endswith("```"):
            candidate = candidate.rstrip()[:-3]
        candidate = candidate.strip()
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].lstrip()
    if candidate.startswith("{") or candidate.startswith("["):
        return candidate
    start = min(
        (index for index in (candidate.find("{"), candidate.find("[")) if index >= 0),
        default=-1,
    )
    end = max(candidate.rfind("}"), candidate.rfind("]"))
    if start >= 0 and end > start:
        return candidate[start:end + 1]
    return candidate


class VikeyClient:
    """Structured generation through a VIKEY OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        timeout_seconds: float,
        max_retries: int,
        max_rounds: int,
        max_tokens: int = 8192,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self.session = session or requests.Session()

    # --- transport -------------------------------------------------------- #
    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=body,
                    timeout=self.timeout_seconds,
                )
                if response.status_code >= 400:
                    raise _HttpError(response.status_code, response.text[:400])
                return response.json()
            except Exception as error:
                if attempts > self.max_retries or not self._is_retryable(error):
                    raise
                time.sleep(min(0.5 * (2 ** (attempts - 1)), 4.0))

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(error, _HttpError):
            return error.status_code in RETRYABLE_STATUS
        text = str(error).casefold()
        return any(
            marker in text
            for marker in ("timeout", "timed out", "connection", "reset", "remotedisconnected")
        )

    # --- main loop -------------------------------------------------------- #
    def generate_structured(
        self,
        system: str,
        prompt: str,
        schema: type[Any],
        tools: ToolInvoker | None = None,
    ) -> StructuredGeneration[Any]:
        started = time.perf_counter()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        total_attempts = 0
        llm_rounds = 0
        input_tokens = 0
        output_tokens = 0
        tool_calls: list[ToolCallRecord] = []
        llm_round_details: list[LLMRoundTrace] = []
        final_text: str | None = None

        while llm_rounds < self.max_rounds:
            declarations = tools.declarations if tools else []
            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if declarations:
                body["tools"] = openai_tools(declarations)
                body["tool_choice"] = "auto"
            else:
                # json_object is the widely supported subset of structured output.
                body["response_format"] = {"type": "json_object"}

            try:
                payload = self._post(body)
            except _HttpError as error:
                # Some gateways reject response_format; retry once without it.
                if "response_format" in body and error.status_code in {400, 404, 422}:
                    body.pop("response_format", None)
                    payload = self._post(body)
                else:
                    raise

            total_attempts += 1
            llm_rounds += 1
            usage = payload.get("usage") or {}
            input_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            output_tokens += completion_tokens

            choices = payload.get("choices") or []
            choice = choices[0] if choices else {}
            llm_round_details.append(
                LLMRoundTrace(
                    round_number=llm_rounds,
                    finish_reason=choice.get("finish_reason"),
                    completion_tokens=completion_tokens,
                )
            )
            if not choices:
                error = ValueError(f"VIKEY returned no choices: {str(payload)[:200]}")
                raise StructuredGenerationError(
                    error,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    attempts=total_attempts,
                    llm_rounds=llm_rounds,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tool_calls=tool_calls,
                    llm_round_details=llm_round_details,
                ) from error
            message = choice.get("message") or {}
            calls = message.get("tool_calls") or []

            if not calls:
                final_text = message.get("content") or ""
                break
            if tools is None or not declarations:
                raise ValueError("LLM meminta tool tetapi tidak ada tool yang didaftarkan.")

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": calls,
                }
            )
            for call in calls:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                raw_args = function.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    args = {}
                tool_started = time.perf_counter()
                try:
                    result, latency_ms = tools.invoke(name, args)
                except Exception as error:
                    failed_calls = getattr(error, "tool_calls", None) or [
                        ToolCallRecord(
                            name=name,
                            args=args,
                            result_summary=f"ERROR: {error}",
                            latency_ms=round((time.perf_counter() - tool_started) * 1000, 3),
                        )
                    ]
                    raise StructuredGenerationError(
                        error,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        attempts=total_attempts,
                        llm_rounds=llm_rounds,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        tool_calls=[*tool_calls, *failed_calls],
                        llm_round_details=llm_round_details,
                    ) from error

                tool_calls.append(
                    ToolCallRecord(
                        name=name,
                        args=args,
                        result_summary=self._summarize_tool_result(name, result),
                        latency_ms=round(latency_ms, 3),
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        if final_text is None:
            error = RuntimeError(
                f"LLM melampaui batas {self.max_rounds} putaran tanpa respons akhir."
            )
            raise StructuredGenerationError(
                error,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                attempts=total_attempts,
                llm_rounds=llm_rounds,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
                llm_round_details=llm_round_details,
            ) from error

        try:
            parsed = json.loads(extract_json(final_text))
        except json.JSONDecodeError as error:
            parse_error = json.JSONDecodeError(
                f"VIKEY returned invalid JSON for {schema.__name__}: {error.msg}",
                error.doc,
                error.pos,
            )
            raise StructuredGenerationError(
                parse_error,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                attempts=total_attempts,
                llm_rounds=llm_rounds,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
                llm_round_details=llm_round_details,
            ) from error
        try:
            value = schema.model_validate(parsed)
        except Exception as error:
            raise StructuredGenerationError(
                error,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                attempts=total_attempts,
                llm_rounds=llm_rounds,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
                llm_round_details=llm_round_details,
            ) from error
        return StructuredGeneration(
            value=value,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            attempts=total_attempts,
            llm_rounds=llm_rounds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
            llm_round_details=llm_round_details,
        )

    @staticmethod
    def _summarize_tool_result(name: str, result: Any) -> str:
        if name == "search_products":
            return json.dumps([item.get("product_id") for item in result])
        if name == "get_season_rules":
            return json.dumps(list(result))
        return f"{type(result).__name__}"


class _HttpError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def build_llm_client(settings: Settings | None = None) -> LLMClientProtocol:
    """Return the client selected by LLM_OPTION (gemini | vikey)."""
    settings = settings or get_settings()
    if settings.LLM_OPTION == "vikey":
        return VikeyClient(
            api_key=settings.require_vikey_key(),
            base_url=settings.VIKEY_URL,
            model=settings.VIKEY_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
            max_rounds=settings.LLM_MAX_ROUNDS,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    return GoogleGenAIClient(
        api_key=settings.require_api_key(),
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
        max_rounds=settings.LLM_MAX_ROUNDS,
    )
