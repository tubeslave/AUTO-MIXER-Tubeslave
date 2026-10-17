from dataclasses import replace
import json
import numpy as np
import pytest

from audio_workbench.belye_stai_kick import BelyeStaiKickRecipe, prepare_kick, render_kick
from audio_workbench.belye_stai_vocal import _active_level, _bell, _filter, _gain, _legacy_compressor, _level
from audio_workbench.mixing.compression import CompressorConfig


def sources(seconds=3):
    sr = 44100
    t = np.arange(sr * seconds) / sr
    env = np.zeros_like(t)
    for point in np.arange(.2, seconds, .45):
        env += np.exp(-np.maximum(t - point, 0) * 18) * (t >= point)
    kick_in = (
        .35 * np.sin(2 * np.pi * 72 * t) * env
        + .12 * np.sin(2 * np.pi * 2400 * t)
        * np.exp(-np.maximum((t % .45) - .2, 0) * 70)
        * ((t % .45) >= .2)
    ).astype('float32')
    kick_out = (.30 * np.sin(2 * np.pi * 58 * t) * env).astype('float32')
    return kick_in, kick_out


def recipe(x):
    return replace(BelyeStaiKickRecipe(), frames=len(x))


def manual(kick_in, kick_out):
    sr = 44100
    kick_out = np.concatenate([kick_out[84:], np.zeros(84, np.float32)])
    kick_in = _filter(kick_in, sr, 28, 6500)
    for frequency, gain_db, q in ((65, 1.2, .8), (190, -2., .9), (2800, 1., .7)):
        kick_in = _bell(kick_in, sr, frequency, gain_db, q)
    kick_out = _filter(kick_out, sr, 28, 1400)
    kick_out = _bell(kick_out, sr, 190, -2.5, .8)
    match = _active_level(kick_in, sr, 92) - _active_level(kick_out, sr, 92) - 5
    pre = kick_in + _gain(kick_out, match)
    processed, _, _ = _legacy_compressor(pre, sr, 3., 3., 22, 115, 4.5)
    processed, _ = _level(processed, sr, -22.5, 92)
    return pre, processed


def test_no_change_exact_and_nonmutating():
    kick_in, kick_out = sources()
    before_in, before_out = kick_in.copy(), kick_out.copy()
    expected_pre, expected = manual(kick_in, kick_out)
    result = render_kick(kick_in, kick_out, recipe=recipe(kick_in))
    np.testing.assert_array_equal(result.pre_compression, expected_pre)
    np.testing.assert_array_equal(result.processed, expected)
    np.testing.assert_array_equal(kick_in, before_in)
    np.testing.assert_array_equal(kick_out, before_out)
    assert result.report['synchronized_mic_group']
    assert result.report['compression_stage_count'] == 1
    assert result.report['precompression']['phase']['KICK_OUT']['advance_samples'] == 84
    json.dumps(result.report, allow_nan=False)


def test_prepare_is_exact_compressor_boundary():
    kick_in, kick_out = sources()
    pre, _, _ = prepare_kick(kick_in, kick_out, recipe=recipe(kick_in))
    np.testing.assert_array_equal(
        pre,
        render_kick(kick_in, kick_out, recipe=recipe(kick_in)).pre_compression,
    )


def test_replacement_requires_source_bound_controls_and_freezes_mic_balance_and_gain():
    kick_in, kick_out = sources()
    frozen_recipe = recipe(kick_in)
    baseline = render_kick(kick_in, kick_out, recipe=frozen_recipe)
    config = CompressorConfig(
        threshold_dbfs=-30,
        ratio=3,
        attack_ms=30,
        release_ms=90,
        knee_db=5,
        max_gr_db=4.5,
        rms_ms=3,
    )
    candidate = render_kick(
        kick_in,
        kick_out,
        recipe=frozen_recipe,
        compressor_config=config,
        frozen_controls=baseline.controls,
    )
    assert candidate.report['compressor_mode'] == 'replacement'
    assert candidate.report['requires_full_session_rerender']
    assert candidate.report['precompression']['out_relative_match_gain_db'] == baseline.controls.out_relative_match_gain_db
    assert candidate.report['level']['gain_db'] == baseline.controls.final_gain_db
    assert not candidate.report['level']['level_reestimated']
    assert not np.array_equal(candidate.processed, baseline.processed)
    with pytest.raises(ValueError):
        render_kick(kick_in, kick_out, recipe=frozen_recipe, compressor_config=config)
    with pytest.raises(ValueError):
        render_kick(
            kick_in * .9,
            kick_out,
            recipe=frozen_recipe,
            compressor_config=config,
            frozen_controls=baseline.controls,
        )


@pytest.mark.parametrize('which', ['stereo', 'length', 'silent', 'nan', 'rate', 'short'])
def test_invalid_group_fails_closed(which):
    kick_in, kick_out = sources()
    frozen_recipe = recipe(kick_in)
    if which == 'stereo':
        kick_in = np.column_stack([kick_in, kick_in])
    if which == 'length':
        kick_out = kick_out[:-1]
    if which == 'silent':
        kick_out[:] = 0
    if which == 'nan':
        kick_in[10] = np.nan
    if which == 'rate':
        frozen_recipe = replace(frozen_recipe, sample_rate=48000)
    if which == 'short':
        kick_in = kick_in[:100]
        kick_out = kick_out[:100]
        frozen_recipe = replace(frozen_recipe, frames=100)
    with pytest.raises(ValueError):
        render_kick(kick_in, kick_out, recipe=frozen_recipe)
