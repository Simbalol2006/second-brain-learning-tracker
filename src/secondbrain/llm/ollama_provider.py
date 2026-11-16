"""Local Ollama backend, using the /api/chat endpoint with a JSON Schema in 'format'."""

import json
import logging
from typing import Any

import httpx

from secondbrain.errors import LLMError
from secondbrain.llm.base import LLMResponse

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 600.0  # local models on a laptop can be slow


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        num_ctx: int,
        max_output_tokens: int,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._host = host.rstrip("/")
        self._num_ctx = num_ctx
        self._max_output_tokens = max_output_tokens
        self._client = client or httpx.Client(base_url=self._host, timeout=_TIMEOUT_SECONDS)

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        # Small models follow a schema better when they also see it in the prompt.
        system_with_schema = (
            f"{system}\n\nReturn only JSON that matches this JSON Schema:\n{json.dumps(schema)}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_with_schema},
                {"role": "user", "content": user},
            ],
            "format": schema,
            "stream": False,
            # num_ctx is set explicitly: Ollama's default context is small and
            # longer prompts would be cut off without any error.
            "options": {
                "temperature": 0,
                "num_ctx": self._num_ctx,
                "num_predict": self._max_output_tokens,
            },
        }
        data = self._post(payload)
        if data.get("done_reason") == "length":
            raise LLMError(
                "The local model hit its output limit before finishing. "
                "Raise LLM_MAX_OUTPUT_TOKENS in .env."
            )
        text = (data.get("message") or {}).get("content") or ""
        if not text:
            raise LLMError("Ollama returned an empty response.")
        prompt_tokens = data.get("prompt_eval_count")
        if isinstance(prompt_tokens, int) and prompt_tokens >= self._num_ctx - self._max_output_tokens:
            logger.warning(
                "Prompt used %d tokens of a %d-token context; it may have been cut off. "
                "Lower LLM_MAX_INPUT_TOKENS.",
                prompt_tokens,
                self._num_ctx,
            )
        return LLMResponse(text, prompt_tokens, data.get("eval_count"))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._client.post("/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise LLMError(
                f"Ollama is not reachable at {self._host}. Start it with: ollama serve"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama did not answer within {_TIMEOUT_SECONDS:.0f}s.") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        if response.status_code == 404:
            raise LLMError(
                f"Ollama does not have the model '{self.model}'. Run: ollama pull {self.model}"
            )
        if response.status_code >= 400:
            raise LLMError(f"Ollama error {response.status_code}: {response.text[:300]}")
        try:
            return dict(response.json())
        except ValueError as exc:
            raise LLMError("Ollama returned a response that is not JSON.") from exc
