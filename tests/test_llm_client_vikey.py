import json

import pytest

from app.core.config import Settings
from app.core.llm_client import StructuredGenerationError
from app.core.llm_client_vikey import VikeyClient, build_llm_client
from app.schemas.analysis import ProductAgentOutput


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []

    def post(self, url, *, headers, json, timeout):
        self.requests.append({"url": url, "headers": headers, "json": json})
        return _Response(self.payloads.pop(0))


class _Tool:
    declarations = [
        {
            "name": "lookup",
            "parameters": {
                "type": "OBJECT",
                "properties": {},
                "required": [],
            },
        }
    ]

    def invoke(self, name, args):
        return {"ok": True}, 1.0


def _client(session, *, max_tokens=1024):
    return VikeyClient(
        api_key="test-key",
        base_url="https://example.test/v1",
        model="test-model",
        temperature=0,
        timeout_seconds=1,
        max_retries=0,
        max_rounds=4,
        max_tokens=max_tokens,
        session=session,
    )


def test_vikey_sends_max_tokens_and_records_each_round() -> None:
    session = _Session(
        [
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": ProductAgentOutput().model_dump_json()
                        },
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 9},
            },
        ]
    )

    result = _client(session, max_tokens=2048).generate_structured(
        "system", "prompt", ProductAgentOutput, tools=_Tool()
    )

    assert [request["json"]["max_tokens"] for request in session.requests] == [
        2048,
        2048,
    ]
    assert [detail.model_dump() for detail in result.llm_round_details] == [
        {"round_number": 1, "finish_reason": "tool_calls", "completion_tokens": 4},
        {"round_number": 2, "finish_reason": "stop", "completion_tokens": 9},
    ]
    assert result.output_tokens == 13


def test_invalid_json_is_not_retried_and_preserves_round_telemetry() -> None:
    session = _Session(
        [
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "not valid json"},
                    }
                ],
                "usage": {"prompt_tokens": 6, "completion_tokens": 3},
            }
        ]
    )

    with pytest.raises(StructuredGenerationError) as error:
        _client(session).generate_structured("system", "prompt", ProductAgentOutput)

    assert len(session.requests) == 1
    assert isinstance(error.value.original_error, json.JSONDecodeError)
    assert [detail.model_dump() for detail in error.value.llm_round_details] == [
        {"round_number": 1, "finish_reason": "stop", "completion_tokens": 3}
    ]


def test_build_vikey_client_uses_llm_max_tokens_setting() -> None:
    settings = Settings(
        LLM_OPTION="vikey",
        VIKEY_KEY="test-key",
        LLM_MAX_TOKENS=3072,
    )

    client = build_llm_client(settings)

    assert isinstance(client, VikeyClient)
    assert client.max_tokens == 3072
