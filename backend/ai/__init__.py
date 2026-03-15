"""
AI agent module for AUTO MIXER Tubeslave.

Provides intelligent mixing assistance through a 3-tier architecture:
1. Rule-based engine for fast, deterministic decisions
2. Local LLM (Ollama) for complex reasoning
3. Cloud LLM (Perplexity) as final fallback

Also includes a RAG knowledge base for mixing best practices.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from .knowledge_base import KnowledgeBase
except ImportError as e:
    logger.warning(f"Could not import KnowledgeBase: {e}")
    KnowledgeBase = None  # type: ignore[assignment, misc]

try:
    from .rule_engine import RuleEngine
except ImportError as e:
    logger.warning(f"Could not import RuleEngine: {e}")
    RuleEngine = None  # type: ignore[assignment, misc]

try:
    from .llm_client import OllamaClient, PerplexityClient, FallbackChain
except ImportError as e:
    logger.warning(f"Could not import LLM clients: {e}")
    OllamaClient = None  # type: ignore[assignment, misc]
    PerplexityClient = None  # type: ignore[assignment, misc]
    FallbackChain = None  # type: ignore[assignment, misc]

try:
    from .agent import AIAgent
except ImportError as e:
    logger.warning(f"Could not import AIAgent: {e}")
    AIAgent = None  # type: ignore[assignment, misc]

__all__ = [
    "KnowledgeBase",
    "RuleEngine",
    "OllamaClient",
    "PerplexityClient",
    "FallbackChain",
    "AIAgent",
]
