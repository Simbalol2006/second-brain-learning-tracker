"""The one interface every LLM call in the app goes through."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMResponse:
    """Raw model output (expected to be a JSON document) plus token usage if reported."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(Protocol):
    """A backend that answers one prompt with JSON shaped by a JSON Schema.

    Providers ask the backend to enforce the schema where it can. Callers must
    still validate the returned text: enforcement differs between backends.
    Providers raise LLMError for anything that makes the response unusable.
    """

    name: str
    model: str

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> LLMResponse: ...
