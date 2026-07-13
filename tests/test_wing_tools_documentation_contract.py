"""Keep the operator tooling canon aligned with executable WING addresses."""

from pathlib import Path

from wing_addresses import WING_OSC_ADDRESSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DOC = PROJECT_ROOT / "docs" / "TOOLS.md"


def _documentation_template(value: str) -> str:
    return (
        value.replace("{ch}", "{N}")
        .replace("{bus}", "{N}")
        .replace("{main}", "{N}")
    )


def test_wing_tools_documentation_uses_runtime_canonical_addresses():
    document = TOOLS_DOC.read_text(encoding="utf-8")
    channel = WING_OSC_ADDRESSES["channel"]
    expected = [
        channel["fdr"],
        channel["mute"],
        channel["pan"],
        channel["in"]["set"]["trim"],
        channel["dyn"]["att"],
        channel["dyn"]["rel"],
        channel["name"],
        channel["icon"],
        channel["color"],
        WING_OSC_ADDRESSES["bus"]["fdr"],
        WING_OSC_ADDRESSES["bus"]["mute"],
        WING_OSC_ADDRESSES["main"]["fdr"],
        WING_OSC_ADDRESSES["main"]["mute"],
    ]
    for address in expected:
        assert _documentation_template(address) in document
    for suffix in ("f", "g", "q"):
        assert f"/ch/{{N}}/eq/{{B}}{suffix}" in document

    for stale_address in (
        "/ch/{N}/mix/fader",
        "/ch/{N}/mix/on",
        "/ch/{N}/mix/pan",
        "/ch/{N}/preamp/trim",
        "/ch/{N}/dyn/attack",
        "/ch/{N}/dyn/release",
        "/main/mix/fader",
        "/bus/{N}/mix/fader",
    ):
        assert stale_address not in document
