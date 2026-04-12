"""Tests for backend/skills registry."""

import pytest

from skills.registry import (
    SkillContext,
    clear_skills_for_tests,
    list_skills,
    register_skill,
    run_skill,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_skills_for_tests()
    yield
    clear_skills_for_tests()


@pytest.mark.asyncio
async def test_register_and_run():
    async def demo(ctx: SkillContext, params: dict):
        return {"x": params.get("n", 0)}

    register_skill("demo", demo)
    assert "demo" in list_skills()
    out = await run_skill("demo", SkillContext(), {"n": 3})
    assert out == {"x": 3}


@pytest.mark.asyncio
async def test_unknown_skill_raises():
    with pytest.raises(KeyError):
        await run_skill("missing", SkillContext(), {})


def test_register_rejects_sync_callable():
    def bad(ctx: SkillContext, params: dict):
        return 1

    with pytest.raises(TypeError):
        register_skill("bad", bad)  # type: ignore[arg-type]
