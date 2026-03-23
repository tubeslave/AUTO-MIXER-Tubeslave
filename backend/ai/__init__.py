"""
AI Agent modules for AUTO-MIXER-Tubeslave.
Provides knowledge base, rule engine, LLM integration, and autonomous agent.
"""
try:
    from .knowledge_base import KnowledgeBase
    from .rule_engine import RuleEngine, Rule, RuleResult
    from .llm_client import LLMClient
    from .agent import MixingAgent
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"AI module import error (optional deps): {e}")
