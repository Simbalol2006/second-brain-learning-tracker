"""Anthropic API backend, using structured outputs (output_config.format)."""

import logging
from typing import Any

import anthropic

from secondbrain.errors import LLMError
from secondbrain.llm.base import LLMResponse

logger = logging.getLogger(__name__)


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_output_tokens: int,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self._max_output_tokens = max_output_tokens
        self._client = client or anthropic.Anthropic(api_key=api_key)

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        # No temperature/top_p: current Claude models expect those to be omitted.
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self._max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("Anthropic rejected the API key. Check ANTHROPIC_API_KEY in .env.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Anthropic API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc

        if message.stop_reason == "refusal":
            raise LLMError("The model declined to process this content.")
        if message.stop_reason == "max_tokens":
            raise LLMError(
                "The model ran out of output tokens before finishing (thinking counts toward "
                "the limit). Raise LLM_MAX_OUTPUT_TOKENS in .env."
            )
        text = "".join(block.text for block in message.content if block.type == "text")
        if not text:
            raise LLMError("The Anthropic response contained no text block.")
        return LLMResponse(text, message.usage.input_tokens, message.usage.output_tokens)
