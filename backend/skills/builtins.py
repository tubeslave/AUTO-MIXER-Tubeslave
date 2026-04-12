"""Built-in example skills (optional registration)."""

from __future__ import annotations

import logging
from typing import Any, Dict

from .registry import SkillContext, register_skill

logger = logging.getLogger(__name__)


async def _skill_ping(_ctx: SkillContext, params: Dict[str, Any]) -> Dict[str, str]:
    """Health-check skill: returns ok and echoes optional message."""
    msg = params.get("message", "")
    return {"status": "ok", "echo": str(msg)}


def register_builtin_skills() -> None:
    """Register default skills (idempotent: overwrites same names)."""
    register_skill("ping", _skill_ping)
