from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "Docs" / "AUTO_EQ_INTEGRATION.md"


def test_autoeq_integration_guide_preserves_safe_promotion_boundary():
    text = DOC_PATH.read_text(encoding="utf-8")

    required_phrases = [
        "audit-only, shadow-only",
        "proposal-only",
        "не должны вызывать OSC apply/reset",
        "## OSC адреса только для справки",
        "не являются инструкцией к прямому AutoEQ apply",
        "не дают AutoEQ\nправа на live write",
        "Product Layer proposal",
        "`backend/autoeq_audit.py`",
        "supervised gate",
        "physical pre-read",
        "bounded delta",
        "readback",
        "rollback",
        "cooldown",
        "preserved\nartifacts",
        "## Следующие безопасные срезы",
        "offline/replay материалах, не по live\n      console writes",
    ]

    for phrase in required_phrases:
        assert phrase in text

    assert "## TODO" not in text
