"""
Optional skill modules for extending automation (soundcheck, analysis hooks).

This is a lightweight in-process registry — not a third-party runtime. Register
callables at startup or from tests; handlers and agents can resolve them by name.
"""

from .builtins import register_builtin_skills
from .registry import SkillContext, list_skills, register_skill, run_skill

__all__ = [
    "SkillContext",
    "list_skills",
    "register_builtin_skills",
    "register_skill",
    "run_skill",
]
