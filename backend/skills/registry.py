"""
In-process skill registry for pluggable automation steps.

Skills are plain async callables registered by name. Use for optional workflows
(e.g. custom soundcheck steps) without pulling in external agent frameworks.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SkillFn = Callable[["SkillContext", Dict[str, Any]], Awaitable[Any]]


@dataclass
class SkillContext:
    """Opaque bag for server-owned services passed into skills."""

    server: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)


_REGISTRY: Dict[str, SkillFn] = {}


def register_skill(name: str, fn: SkillFn) -> None:
    """Register or replace a skill under *name* (non-empty string)."""
    if not name or not isinstance(name, str):
        raise ValueError("skill name must be a non-empty string")
    if not asyncio.iscoroutinefunction(fn):
        raise TypeError(f"skill {name!r} must be an async callable")
    _REGISTRY[name] = fn
    logger.debug("Registered skill %r", name)


def get_skill(name: str) -> Optional[SkillFn]:
    """Return the registered skill or None."""
    return _REGISTRY.get(name)


def list_skills() -> List[str]:
    """Return sorted skill names."""
    return sorted(_REGISTRY.keys())


async def run_skill(
    name: str,
    ctx: SkillContext,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Execute a skill by name; raises KeyError if missing."""
    fn = _REGISTRY.get(name)
    if fn is None:
        raise KeyError(f"unknown skill: {name}")
    return await fn(ctx, params or {})


def clear_skills_for_tests() -> None:
    """Remove all skills (test helper only)."""
    _REGISTRY.clear()
