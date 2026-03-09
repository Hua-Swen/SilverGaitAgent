"""
Application configuration.
Reads settings from environment variables or .env file.
Supports Claude (Anthropic), OpenAI, and Gemini (Google) as LLM providers.
"""

import os
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

SUPPORTED_PROVIDERS = ("claude", "openai", "gemini")


def get_llm(provider: str | None = None) -> BaseChatModel:
    """
    Return a configured chat model for the given provider.

    Provider is resolved in this order:
      1. The `provider` argument (passed explicitly, e.g. from CLI)
      2. The LLM_PROVIDER env var
      3. Defaults to "claude"

    Supported values: "claude", "openai", "gemini"
    """
    resolved = (provider or os.getenv("LLM_PROVIDER", "claude")).lower().strip()

    if resolved not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{resolved}'. Choose from: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    if resolved == "claude":
        return _build_claude(temperature)
    elif resolved == "openai":
        return _build_openai(temperature)
    else:
        return _build_gemini(temperature)


def _build_claude(temperature: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    print(f"[Config] Provider: Claude  |  Model: {model}  |  Temperature: {temperature}")
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        anthropic_api_key=api_key,
        max_tokens=4096,
    )


def _build_openai(temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    print(f"[Config] Provider: OpenAI  |  Model: {model}  |  Temperature: {temperature}")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        max_tokens=4096,
    )


def _build_gemini(temperature: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    print(f"[Config] Provider: Gemini  |  Model: {model}  |  Temperature: {temperature}")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        max_output_tokens=4096,
    )
