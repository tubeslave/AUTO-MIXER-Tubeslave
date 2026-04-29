"""Tests for the AutoFOH channel-name classifier."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from autofoh_models import SourceRole, StemRole
from channel_recognizer import classify_channel_name


def test_common_aliases_resolve_to_expected_roles_and_stems():
    cases = [
        ("LD VOX", SourceRole.LEAD_VOCAL, StemRole.LEAD),
        ("Snare Btm", SourceRole.SNARE_BOTTOM, StemRole.SNARE),
        ("OH L", SourceRole.OH_L, StemRole.CYMBALS),
        ("Bass DI", SourceRole.BASS_DI, StemRole.BASS),
        ("AGTR", SourceRole.ACOUSTIC_GUITAR, StemRole.GUITARS),
        ("Trax", SourceRole.TRACKS, StemRole.PLAYBACK),
        ("FX Verb", SourceRole.REVERB_RETURN, StemRole.FX),
    ]

    for name, expected_role, expected_stem in cases:
        result = classify_channel_name(name)
        assert result.source_role == expected_role
        assert expected_stem in result.stem_roles
        assert result.confidence >= 0.8


def test_unknown_channel_names_do_not_receive_dangerous_controls():
    result = classify_channel_name("Mystery Percussion Pod")
    allowed = {control.value for control in result.allowed_controls}

    assert result.source_role == SourceRole.UNKNOWN
    assert result.confidence < 0.2
    assert "gain" not in allowed
    assert "eq" not in allowed
    assert "compressor" not in allowed
    assert "fader" not in allowed
    assert "feedback_notch" in allowed


def test_override_config_can_promote_custom_pattern_to_role():
    result = classify_channel_name(
        "Pastor Mic",
        classifier_config={
            "name_overrides": [
                {
                    "pattern": r"pastor",
                    "source_role": "talkback",
                    "confidence": 0.99,
                }
            ]
        },
    )

    assert result.source_role == SourceRole.TALKBACK
    assert result.match_type == "override"
    assert result.confidence == 0.99
