"""ESGF LLM - Natural language interface for climate data discovery."""

from .esgf_client import (
    ESGFClient,
    ESGFSearchResult,
    COMMON_VARIABLES,
    EXPERIMENTS,
)

__all__ = [
    "ESGFClient",
    "ESGFSearchResult",
    "COMMON_VARIABLES",
    "EXPERIMENTS",
]

# Lazy imports for optional LLM functionality
def create_assistant(api_key: str | None = None):
    """Create an LLM-powered ESGF assistant."""
    from .llm_interface import ESGFAssistant
    return ESGFAssistant(api_key=api_key)
