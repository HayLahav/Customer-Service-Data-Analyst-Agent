"""
Centralised LLM factory.

Priority:
  1. NEBIUS_API_KEY  → Nebius (OpenAI-compatible, Llama 3.3 70B by default)
  2. OPENAI_API_KEY  → standard OpenAI  (gpt-4o-mini by default)

All agent modules call get_llm() so switching providers never requires
touching individual files.
"""

import os
from langchain_openai import ChatOpenAI

# ── Nebius defaults ────────────────────────────────────────────────────────────
NEBIUS_BASE_URL    = "https://api.studio.nebius.com/v1/"
NEBIUS_MODEL       = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

# ── OpenAI fallback default ────────────────────────────────────────────────────
OPENAI_MODEL       = "gpt-4o-mini"


def get_llm(model: str | None = None, temperature: float = 0) -> ChatOpenAI:
    """
    Return a LangChain ChatOpenAI instance pointed at the configured provider.

    Parameters
    ----------
    model       : Override the model name.  Pass None to use the provider default.
    temperature : Sampling temperature (default 0 = deterministic).
    """
    nebius_key = os.getenv("NEBIUS_API_KEY")

    if nebius_key:
        return ChatOpenAI(
            model=model or os.getenv("NEBIUS_MODEL", NEBIUS_MODEL),
            api_key=nebius_key,
            base_url=os.getenv("NEBIUS_BASE_URL", NEBIUS_BASE_URL),
            temperature=temperature,
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model=model or OPENAI_MODEL,
            api_key=openai_key,
            temperature=temperature,
        )

    raise EnvironmentError(
        "No LLM provider configured. "
        "Set NEBIUS_API_KEY (for Nebius/Llama) or OPENAI_API_KEY (for OpenAI)."
    )


def active_provider() -> str:
    """Return a human-readable string describing the active LLM provider."""
    if os.getenv("NEBIUS_API_KEY"):
        model = os.getenv("NEBIUS_MODEL", NEBIUS_MODEL)
        return f"Nebius ({model})"
    if os.getenv("OPENAI_API_KEY"):
        return f"OpenAI ({OPENAI_MODEL})"
    return "unconfigured"
