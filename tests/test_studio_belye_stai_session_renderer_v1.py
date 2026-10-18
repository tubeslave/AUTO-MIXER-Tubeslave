from __future__ import annotations

import hashlib

import numpy as np
import pytest
import soundfile as sf

from audio_workbench.belye_stai_session import (
    BelyeStaiRecipe,
    BelyeStaiSessionRenderer,
    EXPECTED_TRACKS,
    export_delivery_pcm24,
)
from audio_workbench.routed_contribution import SessionRender


def _fake_reverb(send, sr, room_size, damping):
    out = np.zeros_like(send)
    lag = max(1, round(.007 * sr))
    out[lag:] = send[:-lag] * np.float32(
        .4 + .1 * room_size - .05 * damping
    )
    return out


def _tracks(sr=44100, n=44100):
    t = np.arange(n) / sr
    frequencies = {
        "KICK": 70,
        "SNARE": 190,
        "TOM_1": 120,
        "TOM_2": 105,
        "FLOOR": 82,
        "HI_HAT": 5000,
        "BASS": 95,
        "GTR": 420,
        "VALERA_VOX": 260,
        "NIKITA_VOX": 330,
    }
    out = {}
    for i, name in enumerate(EXPECTED_TRACKS):
        if name in {"OH", "KEYS", "PLAYBACK"}:
            f = {"OH": 3200, "KEYS": 620, "PLAYBACK": 880}[name]
            out[name] = np.column_stack([
                .025 * np.sin(2 * np.pi * f * t),
                .023 * np.sin(2 * np.pi * (f * 1.017) * t),
            ]).astype("float32")
        else:
            f = frequencies[name]
            env = (
                .7 + .3 * np.sin(2 * np.pi * (1.2 + i * .07) * t) ** 2
            ).astype("float32")
            out[name] = (
                .035 * env * np.sin(2 * np.pi * f * t)
            ).astype("float32")
    return out


def _renderer(n=44100):
    recipe = BelyeStaiRecipe(
        sample_rate=44100,
        frames=n,
        accent_center_s=.5,
        accent_sigma_s=.03,
    )
    tracks = _tracks(recipe.sample_rate, n)
    return (
        BelyeStaiSessionRenderer(
            tracks,
            recipe=recipe,
            reverb_backend=_fake_reverb,
        ),
        tracks,
        recipe,
    )


def test_no_change_override_is_sample_identical_and_inputs_stay_immutable():
    renderer, tracks, _ = _renderer()
    before = {k: v.copy() for k, v in tracks.items()}
    a = renderer({})
    b = renderer({"VALERA_VOX": tracks["VALERA_VOX"].copy()})
    assert isinstance(a, SessionRender)
    np.testing.assert_array_equal(a.mix, b.mix)
    np.testing.assert_array_equal(a.vocal_bus, b.vocal_bus)
    np.testing.assert_array_equal(a.drums_bus, b.drums_bus)
    np.testing.assert_array_equal(a.early_room, b.early_room)
    for name in EXPECTED_TRACKS:
        np.testing.assert_array_equal(tracks[name], before[name])
    assert b.metadata["override_stage"] == (
        "post_local_track_processing_pre_context"
    )
    assert b.metadata["override_ids"] == ["VALERA_VOX"]
    assert b.metadata["baseline_promoted"] is False
    assert b.metadata["requires_human_listening"] is True


def test_processed_vocal_override_rerenders_dependent_full_mix():
    renderer, tracks, _ = _renderer()
    base = renderer({})
    candidate = (
        tracks["VALERA_VOX"] * .88
        + .006 * np.sin(
            2 * np.pi * 1800 * np.arange(len(tracks["VALERA_VOX"])) / 44100
        )
    ).astype("float32")
    changed = renderer({"VALERA_VOX": candidate})
    assert changed.mix.shape == base.mix.shape
    assert np.max(np.abs(changed.mix - base.mix)) > 1e-5
    assert np.max(np.abs(changed.vocal_bus - base.vocal_bus)) > 1e-5
    # The dry drum room send is unchanged, but shared MIX_GLUE is recomputed,
    # so the returned post-glue room contribution follows the full rerender.
    assert np.max(np.abs(changed.early_room - base.early_room)) > 0


def test_snare_override_changes_drums_and_early_room():
    renderer, tracks, _ = _renderer()
    base = renderer({})
    changed = renderer({
        "SNARE": (tracks["SNARE"] * .6).astype("float32")
    })
    assert np.max(np.abs(changed.drums_bus - base.drums_bus)) > 1e-5
    assert np.max(np.abs(changed.early_room - base.early_room)) > 1e-5
    assert np.max(np.abs(changed.mix - base.mix)) > 1e-5


def test_invalid_processed_track_contract_fails_closed():
    renderer, tracks, recipe = _renderer()
    with pytest.raises(ValueError):
        renderer({"UNKNOWN": tracks["KICK"]})
    with pytest.raises(ValueError):
        renderer({
            "VALERA_VOX": np.zeros((recipe.frames, 2), "float32")
        })
    bad = dict(tracks)
    bad.pop("BASS")
    with pytest.raises(ValueError):
        BelyeStaiSessionRenderer(
            bad,
            recipe=recipe,
            reverb_backend=_fake_reverb,
        )


def test_delivery_pcm24_export_is_deterministic(tmp_path):
    renderer, _, recipe = _renderer()
    mix = renderer({}).mix
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    export_delivery_pcm24(a, mix, recipe)
    export_delivery_pcm24(b, mix, recipe)
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(
        b.read_bytes()
    ).digest()
    x, sr = sf.read(a, dtype="float32")
    assert sr == recipe.sample_rate
    assert x.shape == mix.shape
