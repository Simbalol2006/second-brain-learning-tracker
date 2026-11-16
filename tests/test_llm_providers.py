"""Provider tests. No network: a fake Anthropic client and httpx's MockTransport."""

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from secondbrain.config import Settings
from secondbrain.errors import LLMError
from secondbrain.llm import build_provider
from secondbrain.llm.anthropic_provider import AnthropicProvider
from secondbrain.llm.ollama_provider import OllamaProvider

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


# ---------- Anthropic ----------


class FakeMessages:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def anthropic_provider(result: Any) -> tuple[AnthropicProvider, FakeMessages]:
    messages = FakeMessages(result)
    client = SimpleNamespace(messages=messages)
    return AnthropicProvider("key", "claude-haiku-5-5", 4096, client=client), messages  # type: ignore[arg-type]


def message(text: str = '{"ok": true}', stop_reason: str = "end_turn", thinking: bool = False) -> Any:
    content = [SimpleNamespace(type="text", text=text)] if text else []
    if thinking:
        content.insert(0, SimpleNamespace(type="thinking", thinking="", signature="sig"))
    usage = SimpleNamespace(input_tokens=321, output_tokens=45)
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage)


def status_error(cls: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(code, request=request), body=None)


def test_anthropic_request_shape_and_response() -> None:
    provider, messages = anthropic_provider(message(thinking=True))
    response = provider.generate_json("SYS", "USER", SCHEMA)

    assert (response.text, response.input_tokens, response.output_tokens) == ('{"ok": true}', 321, 45)
    sent = messages.kwargs
    assert sent["model"] == "claude-haiku-5-5"
    assert sent["max_tokens"] == 4096
    assert sent["system"] == "SYS"
    assert sent["messages"] == [{"role": "user", "content": "USER"}]
    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert not {"temperature", "top_p", "top_k"} & set(sent)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (message(stop_reason="refusal"), "declined"),
        (message(stop_reason="max_tokens"), "LLM_MAX_OUTPUT_TOKENS"),
        (message(text=""), "no text block"),
        (status_error(anthropic.AuthenticationError, 401), "ANTHROPIC_API_KEY"),
        (status_error(anthropic.RateLimitError, 429), "API error 429"),
        (anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")), "Could not reach"),
    ],
)
def test_anthropic_problems_become_llm_errors(result: Any, expected: str) -> None:
    provider, _ = anthropic_provider(result)
    with pytest.raises(LLMError, match=expected):
        provider.generate_json("SYS", "USER", SCHEMA)


# ---------- Ollama ----------


def ollama_provider(handler: Any) -> OllamaProvider:
    client = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(handler))
    return OllamaProvider("http://localhost:11434/", "tiny:3b", num_ctx=5000, max_output_tokens=1000, client=client)


def test_ollama_request_shape_and_response() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": '{"ok": true}'}, "done_reason": "stop", "prompt_eval_count": 700, "eval_count": 30},
        )

    response = ollama_provider(handler).generate_json("SYS", "USER", SCHEMA)
    assert (response.text, response.input_tokens, response.output_tokens) == ('{"ok": true}', 700, 30)
    body = seen["body"]
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert body["model"] == "tiny:3b"
    assert body["stream"] is False
    assert body["format"] == SCHEMA
    assert body["options"] == {"temperature": 0, "num_ctx": 5000, "num_predict": 1000}
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"].startswith("SYS")
    assert json.dumps(SCHEMA) in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": "USER"}


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("slow", request=request)


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (_connect_error, "not reachable at http://localhost:11434. Start it with: ollama serve"),
        (_timeout, "did not answer"),
        (lambda r: httpx.Response(404, json={"error": "model not found"}), "ollama pull tiny:3b"),
        (lambda r: httpx.Response(500, text="kaboom"), "Ollama error 500: kaboom"),
        (lambda r: httpx.Response(200, text="<html>"), "not JSON"),
        (lambda r: httpx.Response(200, json={"message": {"content": ""}}), "empty response"),
        (lambda r: httpx.Response(200, json={"message": {"content": "{"}, "done_reason": "length"}), "output limit"),
    ],
)
def test_ollama_problems_become_llm_errors(handler: Any, expected: str) -> None:
    with pytest.raises(LLMError, match=expected):
        ollama_provider(handler).generate_json("SYS", "USER", SCHEMA)


def test_ollama_warns_when_prompt_may_be_cut_off(caplog: pytest.LogCaptureFixture) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": "{}"}, "prompt_eval_count": 4000})

    ollama_provider(handler).generate_json("SYS", "USER", SCHEMA)
    assert "may have been cut off" in caplog.text


# ---------- factory ----------


@pytest.mark.parametrize("key", [None, "", "   "])
def test_factory_needs_an_api_key_for_anthropic(key: str | None) -> None:
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY is not set"):
        build_provider(Settings(anthropic_api_key=key, _env_file=None))


def test_factory_builds_anthropic() -> None:
    provider = build_provider(Settings(anthropic_api_key="sk-test", _env_file=None))
    assert (provider.name, provider.model) == ("anthropic", "claude-haiku-5-5")


def test_factory_builds_ollama_with_context_for_input_plus_output() -> None:
    settings = Settings(
        llm_provider="ollama", llm_model="tiny:3b", llm_max_input_tokens=4000,
        llm_max_output_tokens=1024, _env_file=None,
    )
    provider = build_provider(settings)
    assert (provider.name, provider.model) == ("ollama", "tiny:3b")
    assert provider._num_ctx == 5024  # type: ignore[attr-defined]
