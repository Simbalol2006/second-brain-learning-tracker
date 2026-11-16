"""LLM access. Use build_provider(); nothing else in the app talks to an LLM directly."""

from secondbrain.config import Settings
from secondbrain.errors import LLMError
from secondbrain.llm.anthropic_provider import AnthropicProvider
from secondbrain.llm.base import LLMProvider, LLMResponse
from secondbrain.llm.ollama_provider import OllamaProvider

__all__ = ["LLMProvider", "LLMResponse", "build_provider"]


def build_provider(settings: Settings) -> LLMProvider:
    """Create the provider named by LLM_PROVIDER, or explain what is missing."""
    if settings.llm_provider == "anthropic":
        key = settings.anthropic_api_key.get_secret_value().strip() if settings.anthropic_api_key else ""
        if not key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Add it to .env, or set LLM_PROVIDER=ollama "
                "to use a local model."
            )
        return AnthropicProvider(
            api_key=key,
            model=settings.llm_model_name,
            max_output_tokens=settings.llm_max_output_tokens,
        )
    return OllamaProvider(
        host=settings.ollama_host,
        model=settings.llm_model_name,
        num_ctx=settings.llm_max_input_tokens + settings.llm_max_output_tokens,
        max_output_tokens=settings.llm_max_output_tokens,
    )
