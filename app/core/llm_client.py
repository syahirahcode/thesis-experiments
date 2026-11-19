"""Structured Google Gen AI client with an explicit, observable function-call loop."""

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

from app.schemas.analysis import ToolCallRecord

T = TypeVar("T", bound=BaseModel)
GEMINI_REQUEST_DELAY_SECONDS = 6


def _gemini_response_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Remove JSON Schema features unsupported by the Gemini Developer API."""
    response_schema = deepcopy(schema.model_json_schema())

    def remove_additional_properties(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("additionalProperties", None)
            for nested in value.values():
                remove_additional_properties(nested)
        elif isinstance(value, list):
            for nested in value:
                remove_additional_properties(nested)

    remove_additional_properties(response_schema)
    return response_schema


class ToolInvoker(Protocol):
    @property
    def declarations(self) -> list[dict[str, Any]]: ...

    def invoke(self, name: str, args: dict[str, Any]) -> tuple[Any, float]: ...


class LLMClientProtocol(Protocol):
    def generate_structured(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        tools: ToolInvoker | None = None,
    ) -> "StructuredGeneration[T]": ...


@dataclass
class StructuredGeneration(Generic[T]):
    value: T
    latency_ms: float
    attempts: int
    llm_rounds: int
    input_tokens: int
    output_tokens: int
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


class StructuredGenerationError(RuntimeError):
    def __init__(
        self,
        error: Exception,
        *,
        latency_ms: float,
        attempts: int,
        llm_rounds: int,
        input_tokens: int,
        output_tokens: int,
        tool_calls: list[ToolCallRecord],
    ):
        super().__init__(str(error))
        self.original_error = error
        self.latency_ms = latency_ms
        self.attempts = attempts
        self.llm_rounds = llm_rounds
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.tool_calls = tool_calls


class GoogleGenAIClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float,
        timeout_seconds: float,
        max_retries: int,
        max_rounds: int,
        client: Any | None = None,
    ):
        if client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as error:
                raise RuntimeError(
                    "Google Gen AI SDK belum terpasang. Pasang dependensi dari requirements.txt."
                ) from error
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
            )
            self.types = types
        else:
            self.client = client
            self.types = None
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_rounds = max_rounds

    def generate_structured(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        tools: ToolInvoker | None = None,
    ) -> StructuredGeneration[T]:
        started = time.perf_counter()
        messages: list[Any] = [{"role": "user", "parts": [{"text": prompt}]}]
        total_attempts = 0
        llm_rounds = 0
        input_tokens = 0
        output_tokens = 0
        tool_calls: list[ToolCallRecord] = []
        final_text: str | None = None

        while llm_rounds < self.max_rounds:
            declarations = tools.declarations if tools else []
            response = self._generate_round(
                system, messages, declarations, schema
            )
            total_attempts += response[1]
            llm_rounds += 1
            model_response = response[0]
            usage = getattr(model_response, "usage_metadata", None)
            input_tokens += int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens += int(getattr(usage, "candidates_token_count", 0) or 0)

            content = self._response_content(model_response)
            calls = self._function_calls(content)
            if not calls:
                final_text = self._response_text(content)
                break
            if tools is None or not declarations:
                raise ValueError("LLM meminta tool tetapi tidak ada tool yang didaftarkan.")

            messages.append(content)
            tool_responses: list[dict[str, Any]] = []
            for call in calls:
                name = str(getattr(call, "name", "") or "")
                args = dict(getattr(call, "args", {}) or {})
                tool_started = time.perf_counter()
                try:
                    result, latency_ms = tools.invoke(name, args)
                except Exception as error:
                    failed_calls = getattr(error, "tool_calls", None)
                    if failed_calls is None:
                        failed_calls = [
                            ToolCallRecord(
                                name=name,
                                args=args,
                                result_summary=f"ERROR: {error}",
                                latency_ms=round(
                                    (time.perf_counter() - tool_started) * 1000, 3
                                ),
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
                    ) from error
                tool_calls.append(
                    ToolCallRecord(
                        name=name,
                        args=args,
                        result_summary=self._summarize_tool_result(name, result),
                        latency_ms=round(latency_ms, 3),
                    )
                )
                tool_responses.append({"name": name, "response": {"result": result}})
            messages.append(self._tool_response_content(tool_responses))

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
            ) from error
        try:
            parsed = json.loads(final_text)
        except json.JSONDecodeError as error:
            raise json.JSONDecodeError(
                f"Gemini returned invalid JSON for {schema.__name__}: {error.msg}",
                error.doc,
                error.pos,
            ) from error
        value = schema.model_validate(parsed)
        return StructuredGeneration(
            value=value,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            attempts=total_attempts,
            llm_rounds=llm_rounds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
        )

    def _generate_round(
        self,
        system: str,
        messages: list[Any],
        declarations: list[dict[str, Any]],
        schema: type[BaseModel],
    ) -> tuple[Any, int]:
        attempts = 0
        response_schema = _gemini_response_schema(schema)
        while True:
            attempts += 1
            try:
                if self.types is None:
                    config = {
                        "system_instruction": system,
                        "temperature": self.temperature,
                        "tools": declarations or None,
                        "response_mime_type": (
                            "application/json" if not declarations else None
                        ),
                        "response_schema": (
                            response_schema if not declarations else None
                        ),
                    }
                else:
                    sdk_tools = (
                        [
                            self.types.Tool(
                                function_declarations=[
                                    self.types.FunctionDeclaration(
                                        name=declaration["name"],
                                        description=declaration["description"],
                                        parameters=declaration["parameters"],
                                    )
                                    for declaration in declarations
                                ]
                            )
                        ]
                        if declarations
                        else None
                    )
                    config = self.types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=self.temperature,
                        tools=sdk_tools,
                        response_mime_type=(
                            "application/json" if not declarations else None
                        ),
                        response_schema=(
                            response_schema if not declarations else None
                        ),
                    )
                time.sleep(GEMINI_REQUEST_DELAY_SECONDS)
                response = self.client.models.generate_content(
                    model=self.model, contents=messages, config=config
                )
                return response, attempts
            except Exception as error:
                if attempts > self.max_retries or not self._is_retryable(error):
                    raise
                time.sleep(min(0.25 * (2 ** (attempts - 1)), 2.0))

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        status = getattr(error, "status_code", None) or getattr(error, "code", None)
        if status in {408, 429, 500, 502, 503, 504}:
            return True
        text = str(error).casefold()
        return any(
            marker in text
            for marker in ("timeout", "timed out", "connection reset", "connection error")
        )

    @staticmethod
    def _response_content(response: Any) -> Any:
        candidates = getattr(response, "candidates", None) or []
        if not candidates or getattr(candidates[0], "content", None) is None:
            finish_reason = (
                getattr(candidates[0], "finish_reason", None) if candidates else None
            )
            prompt_feedback = getattr(response, "prompt_feedback", None)
            block_reason = getattr(prompt_feedback, "block_reason", None)
            raise ValueError(
                "Gemini returned no response content "
                f"(finish_reason={finish_reason}, block_reason={block_reason})."
            )
        return candidates[0].content

    @staticmethod
    def _function_calls(content: Any) -> list[Any]:
        return [
            part.function_call
            for part in (getattr(content, "parts", None) or [])
            if getattr(part, "function_call", None) is not None
        ]

    @staticmethod
    def _response_text(content: Any) -> str:
        parts = getattr(content, "parts", None) or []
        text = "".join(str(part.text) for part in parts if getattr(part, "text", None))
        if not text.strip():
            raise ValueError("Gemini returned empty text despite JSON output being required.")
        return text

    def _tool_response_content(self, responses: list[dict[str, Any]]) -> Any:
        if self.types is None:
            return {"role": "user", "parts": responses}
        return self.types.Content(
            role="user",
            parts=[
                self.types.Part.from_function_response(
                    name=item["name"], response=item["response"]
                )
                for item in responses
            ],
        )

    @staticmethod
    def _summarize_tool_result(name: str, result: Any) -> str:
        if name == "search_products":
            return json.dumps([item.get("product_id") for item in result])
        if name == "get_season_rules":
            return json.dumps(list(result))
        return f"{type(result).__name__}"
