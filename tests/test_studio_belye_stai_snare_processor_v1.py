from dataclasses import replace
import json
import numpy as np
import pytest

from audio_workbench.belye_stai_snare import BelyeStaiSnareRecipe, prepare_snare, render_snare
from audio_workbench.belye_stai_vocal import _active_level, _bell, _expand, _filter, _gain, _legacy_compressor, _level
from audio_workbench.mixing.compression import CompressorConfig


def sources(seconds=4):
    sr = 44100
    t = np.arange(sr * seconds) / sr
    top = np.zeros_like(t)
    bottom = np.zeros_like(t)
    for i, p in enumerate(np.arange(.18, seconds, .22)):
        dt = np.maximum(t - p, 0)
        on = t >= p
        strength = .18 if i % 4 == 3 else .42
        top += strength * (np.sin(2*np.pi*190*dt) + .35*np.sin(2*np.pi*2400*dt)) * np.exp(-dt*22) * on
        # Raw bottom is opposite polarity in this synthetic fixture.
        bottom += -strength * .65 * np.sin(2*np.pi*1100*dt) * np.exp(-dt*25) * on
    # Low-level cymbal/bleed surrogate in both microphones.
    top += .006*np.sin(2*np.pi*6500*t)
    bottom += .004*np.sin(2*np.pi*5200*t)
    return top.astype('float32'), bottom.astype('float32')


def recipe(x):
    return replace(BelyeStaiSnareRecipe(), frames=len(x))


def manual(st, sb):
    sr = 44100
    sb = -sb
    st = _filter(st, sr, 85, 12500)
    sb = _filter(sb, sr, 220, 11500)
    st = _bell(_bell(st, sr, 380, -2.0), sr, 2500, 2.0, .7)
    sb, _ = _expand(sb, sr, 600, 7000, threshold_q=25, margin_db=9, floor=-8)
    match = _active_level(st, sr, 94) - _active_level(sb, sr, 94) - 11.
    pre = st + _gain(sb, match)
    y, _, _ = _legacy_compressor(pre, sr, 3.5, 3., 14, 110, 5.)
    y, _ = _level(y, sr, -23.8, 94)
    return pre, y, match


def test_no_change_exact_and_nonmutating():
    st, sb = sources(); a, b = st.copy(), sb.copy(); pre, expected, match = manual(st, sb)
    result = render_snare(st, sb, recipe=recipe(st))
    np.testing.assert_array_equal(result.pre_compression, pre)
    np.testing.assert_array_equal(result.processed, expected)
    np.testing.assert_array_equal(st, a); np.testing.assert_array_equal(sb, b)
    assert result.report['synchronized_mic_group']
    assert result.report['bottom_polarity_frozen']
    assert result.report['bottom_expansion_frozen']
    assert result.report['precompression']['polarity']['SN_B'] == -1
    assert result.report['precompression']['bottom_relative_match_gain_db'] == pytest.approx(match)
    json.dumps(result.report, allow_nan=False)


def test_prepare_is_exact_compressor_boundary():
    st, sb = sources(); pre, _, _, _ = prepare_snare(st, sb, recipe=recipe(st))
    np.testing.assert_array_equal(pre, render_snare(st, sb, recipe=recipe(st)).pre_compression)


def test_replacement_requires_bound_controls_and_freezes_bottom_and_gain():
    st, sb = sources(); r = recipe(st); base = render_snare(st, sb, recipe=r)
    cfg = CompressorConfig(threshold_dbfs=-34, ratio=3, attack_ms=18, release_ms=95, knee_db=5, max_gr_db=5, rms_ms=3)
    cand = render_snare(st, sb, recipe=r, compressor_config=cfg, frozen_controls=base.controls)
    assert cand.report['compressor_mode'] == 'replacement'
    assert cand.report['precompression']['bottom_relative_match_gain_db'] == base.controls.bottom_relative_match_gain_db
    assert cand.report['precompression']['snare_bottom']['processed_sha256'] == base.controls.bottom_processed_sha256
    assert cand.report['level']['gain_db'] == base.controls.final_gain_db
    assert not cand.report['level']['level_reestimated']
    assert cand.report['requires_full_session_rerender']
    assert not np.array_equal(cand.processed, base.processed)
    with pytest.raises(ValueError):
        render_snare(st, sb, recipe=r, compressor_config=cfg)
    with pytest.raises(ValueError):
        render_snare(st*.99, sb, recipe=r, compressor_config=cfg, frozen_controls=base.controls)


@pytest.mark.parametrize('which', ['stereo','length','silent','nan','rate','short'])
def test_invalid_group_fails_closed(which):
    st, sb = sources(); r = recipe(st)
    if which == 'stereo': st = np.column_stack([st, st])
    if which == 'length': sb = sb[:-1]
    if which == 'silent': sb[:] = 0
    if which == 'nan': st[10] = np.nan
    if which == 'rate': r = replace(r, sample_rate=48000)
    if which == 'short': st = st[:100]; sb = sb[:100]; r = replace(r, frames=100)
    with pytest.raises(ValueError): render_snare(st, sb, recipe=r)
