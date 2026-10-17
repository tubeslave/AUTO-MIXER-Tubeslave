from backend.ml.research_model_registry import research_model_status
from backend.ml.sequential_stem_blending_policy import StemContext, build_blend_plan, acceptance_requirements


def test_sequential_plan_orders_rock_roles_and_keeps_context():
    stems = [
        StemContext("vox", "lead_vocal"),
        StemContext("bass", "bass"),
        StemContext("gtr", "rhythm_guitar"),
        StemContext("dr", "drums"),
    ]
    plan = build_blend_plan(stems)
    assert [s.stem.stem_id for s in plan] == ["dr", "bass", "gtr", "vox"]
    assert plan[0].prior_stem_ids == ()
    assert plan[-1].prior_stem_ids == ("dr", "bass", "gtr")
    assert all(step.auto_apply is False for step in plan)


def test_research_registry_does_not_overclaim_deployment():
    status = research_model_status()
    assert status["sequential_stem_blending"]["principle_integrated"] is True
    assert status["sequential_stem_blending"]["paper_model_deployed"] is False
    assert status["diffvox"]["paper_model_deployed"] is False
    assert status["diff2mix"]["paper_model_deployed"] is False


def test_acceptance_contract_requires_level_matched_ab():
    contract = acceptance_requirements()
    assert contract["level_matched_ab"] is True
    assert contract["auto_apply"] is False
