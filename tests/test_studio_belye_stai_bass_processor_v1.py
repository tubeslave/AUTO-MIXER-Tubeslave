"""A bass-specific in-place insert, not a copy of the vocal compression chain."""
from dataclasses import replace
import json

import numpy as np
import pytest

from audio_workbench.belye_stai_bass import (
    BelyeStaiBassRecipe, _second_stage, prepare_bass, render_bass,
)
from audio_workbench.belye_stai_vocal import _bell, _filter, _gain, _legacy_compressor, _level, _ride
from audio_workbench.mixing.compression import CompressorConfig, LinkedCompressor
from audio_workbench.quality_loop import evaluate_candidate


def source():
    sr = 44100
    t = np.arange(sr * 3) / sr
    x = (.20 * np.sin(2 * np.pi * 73.416 * t) + .035 * np.sin(2 * np.pi * 440.5 * t))
    return (x * (.2 + .8 * np.exp(-4 * (t % .4)))).astype('float32')


def recipe(x):
    return replace(BelyeStaiBassRecipe(), frames=len(x))


def manual_frozen_recipe(x):
    sr = 44100
    y = _filter(x, sr, 33, 6000)
    y = _bell(_bell(y, sr, 175, -2., .7), sr, 850, 1.2, .7)
    pre, _ = _ride(y, sr, 50, 1100, 2.)
    y, _, _ = _legacy_compressor(pre, sr, 4., 3.5, 9, 130, 5.)
    y, _, _ = _legacy_compressor(y, sr, 1.5, 2., 35, 240, 2.8)
    y, _ = _level(y, sr, -26.8, 65)
    return pre, y


def test_exact_recipe_reproduction_input_immutable_and_two_stages():
    x = source(); original = x.copy()
    expected_pre, expected = manual_frozen_recipe(x)
    r = render_bass(x, recipe=recipe(x))
    np.testing.assert_array_equal(expected_pre, r.pre_compression)
    np.testing.assert_array_equal(expected, r.processed)
    np.testing.assert_array_equal(original, x)
    np.testing.assert_array_equal(render_bass(x, recipe=recipe(x)).processed, r.processed)
    assert r.report['compression_stage_count'] == 2
    assert not r.report['baseline_eligible']
    assert r.report['precompression']['ride_detector_band_hz'] == [50, 1100]
    assert r.report['first_compressor']['ratio'] == 3.5
    json.dumps(r.report, allow_nan=False)


def test_fixed_second_matches_original_recurrence_when_given_original_threshold():
    x = source()
    expected, report, _ = _legacy_compressor(x, 44100, 1.5, 2., 35, 240, 2.8)
    actual, fixed_report = _second_stage(x, 44100, report['threshold_dbfs'])
    np.testing.assert_array_equal(actual, expected)
    assert fixed_report['threshold_reestimated'] is False


def test_prepare_is_insert_boundary():
    x = source()
    pre, _ = prepare_bass(x, recipe=recipe(x))
    np.testing.assert_array_equal(pre, render_bass(x, recipe=recipe(x)).pre_compression)


def test_replacement_freezes_second_threshold_and_final_gain_not_envelope():
    x = source(); r = recipe(x)
    baseline = render_bass(x, recipe=r)
    config = CompressorConfig(threshold_dbfs=-24, attack_ms=35, release_ms=210, max_gr_db=4.)
    result = render_bass(x, recipe=r, first_stage_config=config, frozen_controls=baseline.controls)
    first, _ = LinkedCompressor(44100, config).process(baseline.pre_compression)
    second, _ = _second_stage(first, 44100, baseline.controls.second_threshold_dbfs)
    np.testing.assert_array_equal(result.processed, _gain(second, baseline.controls.final_gain_db))
    assert not np.array_equal(result.processed, baseline.processed)
    assert result.report['second_compressor']['threshold_dbfs'] == baseline.controls.second_threshold_dbfs
    assert result.report['second_compressor']['gain_trace_recomputed'] is True
    assert result.report['level']['gain_db'] == baseline.controls.final_gain_db
    assert not result.report['level']['level_reestimated']
    assert result.report['compression_stage_count'] == 2
    assert result.report['requires_full_session_rerender']
    assert result.report['requires_human_review']


def test_bypass_only_removes_first_stage_not_entire_bass_chain():
    x = source(); r = recipe(x)
    baseline = render_bass(x, recipe=r)
    candidate = render_bass(x, recipe=r, first_stage_config=CompressorConfig(bypass=True),
                            frozen_controls=baseline.controls)
    expected, _ = _second_stage(baseline.pre_compression, 44100, baseline.controls.second_threshold_dbfs)
    np.testing.assert_array_equal(candidate.processed, _gain(expected, baseline.controls.final_gain_db))
    assert candidate.report['first_compressor']['observed_max_gr_db'] == 0
    assert candidate.report['requires_human_review']


def test_cannot_override_without_source_bound_controls_or_reuse_wrong_source():
    x = source(); r = recipe(x)
    baseline = render_bass(x, recipe=r)
    with pytest.raises(ValueError):
        render_bass(x, recipe=r, first_stage_config=CompressorConfig())
    with pytest.raises(ValueError):
        render_bass(x * .9, recipe=r, first_stage_config=CompressorConfig(), frozen_controls=baseline.controls)
    with pytest.raises(ValueError):
        render_bass(x, recipe=r, frozen_controls=baseline.controls)


@pytest.mark.parametrize('field,value', [('final_gain_db', np.nan), ('final_gain_db', 31),
                                        ('second_threshold_dbfs', np.inf)])
def test_invalid_downstream_control_fails_closed(field, value):
    x = source(); baseline = render_bass(x, recipe=recipe(x))
    with pytest.raises(ValueError):
        render_bass(x, recipe=recipe(x), first_stage_config=CompressorConfig(),
                    frozen_controls=replace(baseline.controls, **{field: value}))


@pytest.mark.parametrize('invalid', ['stereo', 'length', 'nan', 'int', 'silent', 'rate', 'short'])
def test_invalid_audio_and_recipe(invalid):
    x = source(); r = recipe(x)
    if invalid == 'stereo': x = np.column_stack([x, x])
    if invalid == 'length': x = x[:-1]
    if invalid == 'nan': x[3] = np.nan
    if invalid == 'int': x = x.astype('int16')
    if invalid == 'silent': x[:] = 0
    if invalid == 'rate': r = replace(r, sample_rate=48000)
    if invalid == 'short': x = x[:100]; r = recipe(x)
    with pytest.raises(ValueError): render_bass(x, recipe=r)


def test_bass_compression_is_not_machine_promoted():
    x = source(); baseline = render_bass(x, recipe=recipe(x))
    result = render_bass(x, recipe=recipe(x), first_stage_config=CompressorConfig(), frozen_controls=baseline.controls)
    decision = evaluate_candidate({'confidence': {'cause': .9, 'intervention': .9}},
                                  {'id': 'bass-proposal', **result.report}, True, [], .95)
    assert not decision['accepted']
    assert decision['acceptance_state'] == 'pending_human_review'
