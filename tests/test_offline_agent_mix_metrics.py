import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import soundfile as sf


def load_offline_agent_mix():
    spec = importlib.util.spec_from_file_location(
        "offline_agent_mix_metrics_module",
        Path(__file__).resolve().parents[1] / "tools" / "offline_agent_mix.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_event_based_metrics_raise_tom_rms_above_bleed_only_average():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 8.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.012 * np.sin(2.0 * np.pi * 6500.0 * t)).astype(np.float32)
    audio = bleed.copy()
    for start_sec in (1.0, 3.4, 5.8):
        start = int(start_sec * sr)
        length = int(0.16 * sr)
        burst_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        audio[start:start + length] += (0.42 * np.sin(2.0 * np.pi * 145.0 * burst_t)).astype(np.float32) * env

    full_metrics = mod.metrics_for(audio, sr, instrument="custom")
    tom_metrics = mod.metrics_for(audio, sr, instrument="rack_tom")

    assert tom_metrics["analysis_mode"] == "event_based"
    assert tom_metrics["analysis_active_ratio"] < 0.25
    assert tom_metrics["rms_db"] > full_metrics["rms_db"] + 8.0


def test_event_based_metrics_focus_vocal_phrases_not_between_phrase_bleed():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 10.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.01 * np.sin(2.0 * np.pi * 7000.0 * t)).astype(np.float32)
    audio = bleed.copy()
    for start_sec in (0.8, 3.2, 6.4):
        start = int(start_sec * sr)
        length = int(0.9 * sr)
        phrase_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        phrase = (
            0.06 * np.sin(2.0 * np.pi * 220.0 * phrase_t)
            + 0.03 * np.sin(2.0 * np.pi * 880.0 * phrase_t)
        ).astype(np.float32)
        audio[start:start + length] += phrase * env

    full_metrics = mod.metrics_for(audio, sr, instrument="custom")
    vocal_metrics = mod.metrics_for(audio, sr, instrument="lead_vocal")

    assert vocal_metrics["analysis_mode"] == "event_based"
    assert 0.15 < vocal_metrics["analysis_active_ratio"] < 0.5
    assert vocal_metrics["rms_db"] > full_metrics["rms_db"] + 4.0


def test_event_based_expander_reduces_tom_bleed_between_hits():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 8.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.015 * np.sin(2.0 * np.pi * 6200.0 * t)).astype(np.float32)
    audio = bleed.copy()
    hit_ranges = []
    for start_sec in (1.0, 3.4, 5.8):
        start = int(start_sec * sr)
        length = int(0.18 * sr)
        burst_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        audio[start:start + length] += (0.46 * np.sin(2.0 * np.pi * 155.0 * burst_t)).astype(np.float32) * env
        hit_ranges.append((start, start + length))

    metrics = mod.metrics_for(audio, sr, instrument="rack_tom")
    plan = mod.ChannelPlan(
        path=Path("TOM.wav"),
        name="TOM",
        instrument="rack_tom",
        pan=0.0,
        hpf=65.0,
        target_rms_db=-25.0,
        metrics=metrics,
    )
    summary = mod.apply_event_based_dynamics({1: plan})
    processed, report = mod.apply_event_based_expander(audio, sr, plan)

    hit_mask = np.zeros(len(audio), dtype=bool)
    for start, end in hit_ranges:
        hit_mask[start:end] = True
    bleed_mask = ~hit_mask

    before_bleed_rms = np.sqrt(np.mean(np.square(audio[bleed_mask]))).item()
    after_bleed_rms = np.sqrt(np.mean(np.square(processed[bleed_mask]))).item()
    before_hit_rms = np.sqrt(np.mean(np.square(audio[hit_mask]))).item()
    after_hit_rms = np.sqrt(np.mean(np.square(processed[hit_mask]))).item()

    assert summary["enabled"] is True
    assert report["enabled"] is True
    assert report["mode"] == "event_based_expander"
    assert mod.amp_to_db(after_bleed_rms) < mod.amp_to_db(before_bleed_rms) - 5.0
    assert mod.amp_to_db(after_hit_rms) > mod.amp_to_db(before_hit_rms) - 1.5


def test_event_based_expander_softly_closes_vocal_mic_between_phrases():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 10.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.012 * np.sin(2.0 * np.pi * 6800.0 * t)).astype(np.float32)
    audio = bleed.copy()
    phrase_ranges = []
    for start_sec in (0.8, 3.2, 6.4):
        start = int(start_sec * sr)
        length = int(0.9 * sr)
        phrase_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        phrase = (
            0.07 * np.sin(2.0 * np.pi * 220.0 * phrase_t)
            + 0.035 * np.sin(2.0 * np.pi * 880.0 * phrase_t)
        ).astype(np.float32)
        audio[start:start + length] += phrase * env
        phrase_ranges.append((start, start + length))

    metrics = mod.metrics_for(audio, sr, instrument="lead_vocal")
    plan = mod.ChannelPlan(
        path=Path("Lead.wav"),
        name="Lead",
        instrument="lead_vocal",
        pan=0.0,
        hpf=90.0,
        target_rms_db=-20.0,
        metrics=metrics,
    )
    summary = mod.apply_event_based_dynamics({1: plan})
    processed, report = mod.apply_event_based_expander(audio, sr, plan)

    phrase_mask = np.zeros(len(audio), dtype=bool)
    for start, end in phrase_ranges:
        phrase_mask[start:end] = True
    gap_mask = ~phrase_mask

    before_gap_rms = np.sqrt(np.mean(np.square(audio[gap_mask]))).item()
    after_gap_rms = np.sqrt(np.mean(np.square(processed[gap_mask]))).item()
    before_phrase_rms = np.sqrt(np.mean(np.square(audio[phrase_mask]))).item()
    after_phrase_rms = np.sqrt(np.mean(np.square(processed[phrase_mask]))).item()

    assert summary["enabled"] is True
    assert report["enabled"] is True
    assert mod.amp_to_db(after_gap_rms) < mod.amp_to_db(before_gap_rms) - 2.5
    assert mod.amp_to_db(after_phrase_rms) > mod.amp_to_db(before_phrase_rms) - 1.2


def test_event_based_expander_uses_cached_raw_event_windows_after_trim():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 8.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.014 * np.sin(2.0 * np.pi * 6100.0 * t)).astype(np.float32)
    audio = bleed.copy()
    for start_sec in (1.2, 4.1):
        start = int(start_sec * sr)
        length = int(0.16 * sr)
        burst_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        audio[start:start + length] += (0.35 * np.sin(2.0 * np.pi * 130.0 * burst_t)).astype(np.float32) * env

    metrics = mod.metrics_for(audio, sr, instrument="floor_tom")
    plan = mod.ChannelPlan(
        path=Path("F TOM.wav"),
        name="F TOM",
        instrument="floor_tom",
        pan=0.0,
        hpf=55.0,
        target_rms_db=-24.5,
        trim_db=-9.0,
        fader_db=-1.0,
        lpf=5800.0,
        metrics=metrics,
        event_activity=mod._event_activity_ranges(audio, sr, "floor_tom") or {},
    )
    mod.apply_event_based_dynamics({1: plan})

    processed_input = audio * mod.db_to_amp(plan.trim_db + plan.fader_db)
    processed_input = mod.highpass(processed_input, sr, plan.hpf)
    processed_input = mod.lowpass(processed_input, sr, plan.lpf)
    _, report = mod.apply_event_based_expander(processed_input, sr, plan)

    assert report["enabled"] is True
    assert report["mode"] == "event_based_expander"


def test_event_based_expander_reduces_snare_bleed_between_hits():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 8.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bleed = (0.014 * np.sin(2.0 * np.pi * 7600.0 * t)).astype(np.float32)
    audio = bleed.copy()
    hit_ranges = []
    for start_sec in (0.9, 2.4, 4.8, 6.6):
        start = int(start_sec * sr)
        length = int(0.15 * sr)
        burst_t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        tone = (
            0.17 * np.sin(2.0 * np.pi * 210.0 * burst_t)
            + 0.08 * np.sin(2.0 * np.pi * 1800.0 * burst_t)
        ).astype(np.float32)
        audio[start:start + length] += tone * env
        hit_ranges.append((start, start + length))

    metrics = mod.metrics_for(audio, sr, instrument="snare")
    plan = mod.ChannelPlan(
        path=Path("SNARE.wav"),
        name="SNARE",
        instrument="snare",
        pan=0.0,
        hpf=110.0,
        target_rms_db=-23.0,
        metrics=metrics,
    )
    summary = mod.apply_event_based_dynamics({1: plan})
    processed, report = mod.apply_event_based_expander(audio, sr, plan)

    hit_mask = np.zeros(len(audio), dtype=bool)
    for start, end in hit_ranges:
        hit_mask[start:end] = True
    bleed_mask = ~hit_mask

    before_bleed_rms = np.sqrt(np.mean(np.square(audio[bleed_mask]))).item()
    after_bleed_rms = np.sqrt(np.mean(np.square(processed[bleed_mask]))).item()
    before_hit_rms = np.sqrt(np.mean(np.square(audio[hit_mask]))).item()
    after_hit_rms = np.sqrt(np.mean(np.square(processed[hit_mask]))).item()

    assert metrics["analysis_mode"] == "event_based"
    assert summary["enabled"] is True
    assert report["enabled"] is True
    assert mod.amp_to_db(after_bleed_rms) < mod.amp_to_db(before_bleed_rms) - 3.8
    assert mod.amp_to_db(after_hit_rms) > mod.amp_to_db(before_hit_rms) - 1.2


def test_ambient_mics_do_not_get_event_based_expander():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 6.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    audio = (
        0.04 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.02 * np.sin(2.0 * np.pi * 4200.0 * t)
    ).astype(np.float32)

    metrics = mod.metrics_for(audio, sr, instrument="overhead")
    plan = mod.ChannelPlan(
        path=Path("OH L.wav"),
        name="OH L",
        instrument="overhead",
        pan=-0.7,
        hpf=150.0,
        target_rms_db=-27.0,
        metrics=metrics,
    )
    summary = mod.apply_event_based_dynamics({1: plan})
    _, report = mod.apply_event_based_expander(audio, sr, plan)

    assert metrics["analysis_mode"] == "windowed_full_track"
    assert summary["enabled"] is False
    assert report["enabled"] is False


def test_ride_event_detection_survives_single_clipped_peak():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 6.0
    audio = np.zeros(int(sr * duration_sec), dtype=np.float32)

    audio[10] = 1.0

    for start_sec in (2.0, 2.8, 3.6):
        start = int(start_sec * sr)
        length = int(0.42 * sr)
        t = np.arange(length, dtype=np.float32) / sr
        env = np.hanning(length).astype(np.float32)
        burst = (
            0.11 * np.sin(2.0 * np.pi * 3200.0 * t)
            + 0.09 * np.sin(2.0 * np.pi * 6400.0 * t)
            + 0.06 * np.sin(2.0 * np.pi * 8800.0 * t)
        ).astype(np.float32)
        audio[start:start + length] += burst * env

    activity = mod._event_activity_ranges(audio, sr, "ride")

    assert activity is not None
    assert activity["ranges"]
    first_start_sec = activity["ranges"][0][0] / sr
    assert 1.7 < first_start_sec < 2.3


def test_autofoh_measurement_corrections_reduce_presence_masking_by_guitar():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 0.6
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    lead = (0.2 * np.sin(2.0 * np.pi * 2200.0 * t)).astype(np.float32)
    guitar = (0.9 * np.sin(2.0 * np.pi * 2800.0 * t)).astype(np.float32)

    plans = {
        1: mod.ChannelPlan(
            path=Path("Lead.wav"),
            name="Lead",
            instrument="lead_vocal",
            pan=0.0,
            hpf=90.0,
            target_rms_db=-20.0,
        ),
        2: mod.ChannelPlan(
            path=Path("Guitar.wav"),
            name="Guitar",
            instrument="electric_guitar",
            pan=0.0,
            hpf=90.0,
            target_rms_db=-24.0,
        ),
    }
    rendered = {
        1: mod.pan_mono(lead, 0.0),
        2: mod.pan_mono(guitar, 0.0),
    }

    report = mod.apply_autofoh_measurement_corrections(plans, rendered, sr)

    assert report["enabled"] is True
    assert any(item["label"] == "lead_masking" for item in report["detected_problems"])
    assert any(item["label"] == "lead_masking" and item["sent"] for item in report["applied_actions"])
    assert len(plans[2].eq_bands) >= 3
    assert plans[2].eq_bands[2][1] < 0.0


def test_autofoh_measurement_corrections_control_excess_sub_from_bass():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 0.6
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    bass = (0.95 * np.sin(2.0 * np.pi * 45.0 * t)).astype(np.float32)
    lead = (0.12 * np.sin(2.0 * np.pi * 2200.0 * t)).astype(np.float32)

    plans = {
        1: mod.ChannelPlan(
            path=Path("Bass.wav"),
            name="Bass",
            instrument="bass_guitar",
            pan=0.0,
            hpf=35.0,
            target_rms_db=-21.0,
        ),
        2: mod.ChannelPlan(
            path=Path("Lead.wav"),
            name="Lead",
            instrument="lead_vocal",
            pan=0.0,
            hpf=90.0,
            target_rms_db=-20.0,
        ),
    }
    rendered = {
        1: mod.pan_mono(bass, 0.0),
        2: mod.pan_mono(lead, 0.0),
    }

    report = mod.apply_autofoh_measurement_corrections(plans, rendered, sr)

    assert report["enabled"] is True
    assert any(item["label"] == "low_end" for item in report["detected_problems"])
    assert any(item["label"] == "low_end" and item["sent"] for item in report["applied_actions"])
    assert plans[1].eq_bands
    assert plans[1].eq_bands[0][1] < 0.0


def test_autofoh_measurement_corrections_raise_quieter_secondary_lead():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 0.8
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    anchor = (
        0.22 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.08 * np.sin(2.0 * np.pi * 2200.0 * t)
    ).astype(np.float32)
    quiet = (
        0.10 * np.sin(2.0 * np.pi * 240.0 * t)
        + 0.04 * np.sin(2.0 * np.pi * 2500.0 * t)
    ).astype(np.float32)

    plans = {
        1: mod.ChannelPlan(
            path=Path("Anchor.wav"),
            name="Anchor",
            instrument="lead_vocal",
            pan=0.0,
            hpf=90.0,
            target_rms_db=-20.0,
            trim_db=-4.0,
            fader_db=0.0,
            metrics={"analysis_active_ratio": 0.2},
        ),
        2: mod.ChannelPlan(
            path=Path("Secondary.wav"),
            name="Secondary",
            instrument="lead_vocal",
            pan=0.0,
            hpf=90.0,
            target_rms_db=-20.0,
            trim_db=-6.5,
            fader_db=0.0,
            metrics={"analysis_active_ratio": 0.15},
        ),
    }
    rendered = {
        1: mod.pan_mono(anchor, 0.0),
        2: mod.pan_mono(quiet, 0.0),
    }

    report = mod.apply_autofoh_measurement_corrections(plans, rendered, sr)

    assert any(item["label"] == "lead_handoff_balance" for item in report["detected_problems"])
    assert any(item["label"] == "lead_handoff_balance" and item["sent"] for item in report["applied_actions"])
    assert plans[2].trim_db > -6.5


def test_autofoh_measurement_corrections_tame_cymbal_buildup():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 0.8
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    lead = (
        0.05 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.02 * np.sin(2.0 * np.pi * 2400.0 * t)
    ).astype(np.float32)
    hihat = (
        0.16 * np.sin(2.0 * np.pi * 6500.0 * t)
        + 0.14 * np.sin(2.0 * np.pi * 9200.0 * t)
    ).astype(np.float32)
    overhead = (
        0.08 * np.sin(2.0 * np.pi * 5400.0 * t)
        + 0.08 * np.sin(2.0 * np.pi * 9800.0 * t)
    ).astype(np.float32)

    plans = {
        1: mod.ChannelPlan(
            path=Path("Lead.wav"),
            name="Lead",
            instrument="lead_vocal",
            pan=0.0,
            hpf=90.0,
            target_rms_db=-20.0,
        ),
        2: mod.ChannelPlan(
            path=Path("HiHat.wav"),
            name="HiHat",
            instrument="hi_hat",
            pan=0.0,
            hpf=180.0,
            target_rms_db=-24.0,
            fader_db=-2.0,
        ),
        3: mod.ChannelPlan(
            path=Path("OH.wav"),
            name="OH",
            instrument="overhead",
            pan=0.0,
            hpf=150.0,
            target_rms_db=-24.0,
            fader_db=-2.0,
        ),
    }
    rendered = {
        1: mod.pan_mono(lead, 0.0),
        2: mod.pan_mono(hihat, 0.0),
        3: mod.pan_mono(overhead, 0.0),
    }

    report = mod.apply_autofoh_measurement_corrections(plans, rendered, sr)

    assert any(item["label"] == "cymbal_buildup" for item in report["detected_problems"])
    assert any(item["label"] == "cymbal_buildup" and item["sent"] for item in report["applied_actions"])
    assert plans[2].fader_db < -2.0 or any(band[1] < 0.0 for band in plans[3].eq_bands)


def test_compressor_auto_makeup_restores_sustained_signal_rms():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 0.8
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    x = (0.8 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    without_makeup = mod.compressor(
        x,
        sr,
        threshold_db=-24.0,
        ratio=4.0,
        attack_ms=5.0,
        release_ms=90.0,
        auto_makeup=False,
    )
    with_makeup = mod.compressor(
        x,
        sr,
        threshold_db=-24.0,
        ratio=4.0,
        attack_ms=5.0,
        release_ms=90.0,
        auto_makeup=True,
    )

    input_rms_db = mod.amp_to_db(float(np.sqrt(np.mean(np.square(x))) + 1e-12))
    without_makeup_rms_db = mod.amp_to_db(float(np.sqrt(np.mean(np.square(without_makeup))) + 1e-12))
    with_makeup_rms_db = mod.amp_to_db(float(np.sqrt(np.mean(np.square(with_makeup))) + 1e-12))

    assert abs(with_makeup_rms_db - input_rms_db) < abs(without_makeup_rms_db - input_rms_db)
    assert abs(with_makeup_rms_db - input_rms_db) < 0.75


def test_render_channel_uses_trim_pre_processing_and_fader_post_pan():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 0.7
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    mono = (0.78 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "Tone.wav"
        sf.write(path, mono, sr, subtype="PCM_24")

        plan = mod.ChannelPlan(
            path=path,
            name="Tone",
            instrument="lead_vocal",
            pan=0.35,
            hpf=90.0,
            target_rms_db=-20.0,
            trim_db=-3.0,
            fader_db=-6.0,
            phase_invert=True,
            delay_ms=1.2,
            comp_threshold_db=-24.0,
            comp_ratio=4.0,
            comp_attack_ms=5.0,
            comp_release_ms=90.0,
        )

        rendered = mod.render_channel(path, plan, len(mono), sr)

        expected = mono.copy()
        expected = mod.declick_start(expected, sr, plan.input_fade_ms)
        expected = expected * mod.db_to_amp(plan.trim_db)
        expected = -expected
        expected = mod.delay_signal(expected, sr, plan.delay_ms)
        expected = mod.highpass(expected, sr, plan.hpf)
        expected, _ = mod.apply_event_based_expander(expected, sr, plan)
        expected = mod.compressor(
            expected,
            sr,
            threshold_db=plan.comp_threshold_db,
            ratio=plan.comp_ratio,
            attack_ms=plan.comp_attack_ms,
            release_ms=plan.comp_release_ms,
            makeup_db=0.0,
        )
        expected = mod.pan_mono(expected, plan.pan) * mod.db_to_amp(plan.fader_db)

        assert np.allclose(rendered, expected.astype(np.float32), atol=1e-5)


def test_reference_mix_guidance_applies_bounded_channel_adjustments():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 1.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    vocal = (
        0.07 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.03 * np.sin(2.0 * np.pi * 880.0 * t)
    ).astype(np.float32)
    guitar = (0.08 * np.sin(2.0 * np.pi * 1100.0 * t)).astype(np.float32)
    reference = np.column_stack([
        (0.4 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32),
        (0.25 * np.sin(2.0 * np.pi * 660.0 * t)).astype(np.float32),
    ])

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        vocal_path = tmpdir_path / "Lead.wav"
        guitar_path = tmpdir_path / "Guitar.wav"
        reference_path = tmpdir_path / "Reference.wav"
        sf.write(vocal_path, vocal, sr, subtype="PCM_24")
        sf.write(guitar_path, guitar, sr, subtype="PCM_24")
        sf.write(reference_path, reference, sr, subtype="PCM_24")

        plans = {
            1: mod.ChannelPlan(
                path=vocal_path,
                name="Lead",
                instrument="lead_vocal",
                pan=0.0,
                hpf=90.0,
                target_rms_db=-20.0,
                fader_db=-6.0,
                comp_threshold_db=-20.0,
                comp_ratio=2.5,
                metrics=mod.metrics_for(vocal, sr, instrument="lead_vocal"),
            ),
            2: mod.ChannelPlan(
                path=guitar_path,
                name="Guitar",
                instrument="guitar",
                pan=-0.2,
                hpf=100.0,
                target_rms_db=-24.0,
                fader_db=-8.0,
                comp_threshold_db=-18.0,
                comp_ratio=2.0,
                metrics=mod.metrics_for(guitar, sr, instrument="guitar"),
            ),
        }

        context = mod.prepare_reference_mix_context(reference_path)
        report = mod.apply_reference_mix_guidance(plans, sr, context)

        assert report["enabled"] is True
        assert report["reference_path"] == str(reference_path.resolve())
        assert report["applied_channel_count"] >= 1
        assert report["targets"]["tilt"]
        assert report["sections"]
        assert plans[1].fader_db >= -7.5
        assert plans[1].fader_db <= -4.5
        assert plans[2].fader_db >= -9.5
        assert plans[2].fader_db <= -6.5
        assert plans[1].expander_enabled is True
        assert plans[1].expander_threshold_db is not None
        assert plans[2].fx_send_db is not None
        assert any(action["pan"] is not None or action["fx_send"] is not None for action in report["actions"])


def test_prepare_reference_mix_context_merges_audio_directory():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 1.0
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    ref_a = np.column_stack([
        (0.4 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32),
        (0.3 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32),
    ])
    ref_b = np.column_stack([
        (0.35 * np.sin(2.0 * np.pi * 110.0 * t)).astype(np.float32),
        (0.25 * np.sin(2.0 * np.pi * 550.0 * t)).astype(np.float32),
    ])

    with TemporaryDirectory() as tmpdir:
        ref_dir = Path(tmpdir) / "reference"
        ref_dir.mkdir()
        sf.write(ref_dir / "A.wav", ref_a, sr, subtype="PCM_24")
        sf.write(ref_dir / "B.wav", ref_b, sr, subtype="PCM_24")

        context = mod.prepare_reference_mix_context(ref_dir)

    assert context is not None
    assert context.source_type == "audio_directory"
    assert context.sample_rate == sr
    assert len(context.source_paths) == 2
    assert context.audio is not None
    assert context.audio.ndim == 2
    assert len(context.audio) >= len(ref_a) * 2
    assert context.targets["tilt"]
    assert context.sections


def test_rock_genre_profile_tightens_vocals_and_centers_snare_layers():
    mod = load_offline_agent_mix()
    plans = {
        1: mod.ChannelPlan(
            path=Path("Vocal.wav"),
            name="Vocal",
            instrument="lead_vocal",
            pan=0.0,
            hpf=90.0,
            target_rms_db=-20.0,
            trim_db=0.5,
            comp_threshold_db=-23.0,
            comp_ratio=3.2,
            comp_attack_ms=5.0,
            comp_release_ms=120.0,
            expander_enabled=True,
            expander_range_db=4.5,
            expander_open_ms=18.0,
            expander_close_ms=140.0,
            expander_hold_ms=180.0,
            expander_threshold_db=-42.0,
        ),
        2: mod.ChannelPlan(
            path=Path("SNARE T.wav"),
            name="SNARE T",
            instrument="snare",
            pan=-0.02,
            hpf=90.0,
            target_rms_db=-22.0,
            comp_threshold_db=-20.0,
            comp_ratio=3.4,
        ),
        3: mod.ChannelPlan(
            path=Path("Snare B.wav"),
            name="Snare B",
            instrument="snare",
            pan=0.02,
            hpf=120.0,
            target_rms_db=-27.0,
            comp_threshold_db=-22.0,
            comp_ratio=3.5,
        ),
    }

    report = mod.apply_genre_mix_profile(plans, "rock")

    assert report["enabled"] is True
    assert sorted(report["snare_layers"]) == ["SNARE T.wav", "Snare B.wav"]
    assert plans[1].trim_db > 0.5
    assert plans[1].comp_ratio >= 4.8
    assert plans[1].comp_threshold_db <= -26.0
    assert plans[1].expander_close_ms >= 240.0
    assert plans[1].expander_hold_ms >= 300.0
    assert plans[2].pan == 0.0
    assert plans[3].pan == 0.0


def test_cross_adaptive_eq_protects_kick_low_band_and_cuts_bass(monkeypatch):
    mod = load_offline_agent_mix()
    sr = 48_000
    target_len = 2048

    class DummyProcessor:
        BAND_CENTERS = mod.CrossAdaptiveEQ.BAND_CENTERS

        def __init__(self, *args, **kwargs):
            pass

        def calculate_corrections(self, channel_band_energy, channel_priorities):
            return [
                SimpleNamespace(channel_id=1, frequency_hz=120.0, gain_db=-1.0, q_factor=4.0),
                SimpleNamespace(channel_id=2, frequency_hz=120.0, gain_db=-1.0, q_factor=4.0),
            ]

    monkeypatch.setattr(mod, "CrossAdaptiveEQ", DummyProcessor)
    monkeypatch.setattr(
        mod,
        "render_channel",
        lambda path, plan, target_len, sr: np.column_stack([
            np.full(target_len, 0.01, dtype=np.float32),
            np.full(target_len, 0.01, dtype=np.float32),
        ]),
    )

    plans = {
        1: mod.ChannelPlan(
            path=Path("KICK.wav"),
            name="KICK",
            instrument="kick",
            pan=0.0,
            hpf=35.0,
            target_rms_db=-20.0,
        ),
        2: mod.ChannelPlan(
            path=Path("Bass.wav"),
            name="Bass",
            instrument="bass_guitar",
            pan=0.0,
            hpf=35.0,
            target_rms_db=-21.0,
        ),
    }

    report = mod.apply_cross_adaptive_eq(plans, target_len, sr)

    assert report["enabled"] is True
    assert not any(item["channel"] == 1 for item in report["applied"])
    bass_actions = [item for item in report["applied"] if item["channel"] == 2]
    assert len(bass_actions) == 1
    assert bass_actions[0]["gain_db"] < 0.0


def test_kick_bass_hierarchy_boosts_kick_when_bass_overwhelms():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 3.0
    length = int(sr * duration_sec)
    t = np.arange(length, dtype=np.float32) / sr

    bass = (0.24 * np.sin(2.0 * np.pi * 72.0 * t)).astype(np.float32)
    kick = np.zeros_like(bass)
    for start_sec in (0.4, 1.0, 1.6, 2.2):
        start = int(start_sec * sr)
        hit_len = int(0.12 * sr)
        hit_t = np.arange(hit_len, dtype=np.float32) / sr
        env = np.hanning(hit_len).astype(np.float32)
        hit = (
            0.14 * np.sin(2.0 * np.pi * 64.0 * hit_t)
            + 0.03 * np.sin(2.0 * np.pi * 3200.0 * hit_t)
        ).astype(np.float32)
        kick[start:start + hit_len] += hit * env

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        kick_path = tmpdir_path / "KICK.wav"
        bass_path = tmpdir_path / "Bass.wav"
        sf.write(kick_path, kick, sr, subtype="PCM_24")
        sf.write(bass_path, bass, sr, subtype="PCM_24")

        kick_plan = mod.ChannelPlan(
            path=kick_path,
            name="KICK",
            instrument="kick",
            pan=0.0,
            hpf=35.0,
            target_rms_db=-20.0,
            fader_db=-2.0,
            eq_bands=[],
            comp_threshold_db=0.0,
            comp_ratio=1.0,
            metrics=mod.metrics_for(kick, sr, instrument="kick"),
        )
        bass_plan = mod.ChannelPlan(
            path=bass_path,
            name="Bass",
            instrument="bass_guitar",
            pan=0.0,
            hpf=35.0,
            target_rms_db=-21.0,
            fader_db=-1.0,
            eq_bands=[],
            comp_threshold_db=0.0,
            comp_ratio=1.0,
            metrics=mod.metrics_for(bass, sr, instrument="bass_guitar"),
        )

        report = mod.apply_kick_bass_hierarchy({1: kick_plan, 2: bass_plan}, length, sr, desired_kick_advantage_db=1.5)

    assert report["enabled"] is True
    assert report["measured_advantage_db"] < report["desired_kick_advantage_db"]
    assert report["kick_fader_after_db"] > report["kick_fader_before_db"]
    assert report["bass_fader_after_db"] < report["bass_fader_before_db"]
    assert report["kick_eq_added"]
    assert report["bass_eq_added"]


def test_stem_mix_verification_checks_slope_and_reseats_kick():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 3.0
    length = int(sr * duration_sec)
    t = np.arange(length, dtype=np.float32) / sr

    kick = np.zeros(length, dtype=np.float32)
    for start_sec in (0.35, 0.95, 1.55, 2.15):
        start = int(start_sec * sr)
        hit_len = int(0.12 * sr)
        hit_t = np.arange(hit_len, dtype=np.float32) / sr
        env = np.hanning(hit_len).astype(np.float32)
        hit = (
            0.12 * np.sin(2.0 * np.pi * 64.0 * hit_t)
            + 0.006 * np.sin(2.0 * np.pi * 3200.0 * hit_t)
        ).astype(np.float32)
        kick[start:start + hit_len] += hit * env

    bass = (0.04 * np.sin(2.0 * np.pi * 72.0 * t)).astype(np.float32)
    overhead = (0.14 * np.sin(2.0 * np.pi * 3500.0 * t)).astype(np.float32)

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        kick_path = tmpdir_path / "KICK.wav"
        bass_path = tmpdir_path / "Bass.wav"
        overhead_path = tmpdir_path / "OH L.wav"
        sf.write(kick_path, kick, sr, subtype="PCM_24")
        sf.write(bass_path, bass, sr, subtype="PCM_24")
        sf.write(overhead_path, overhead, sr, subtype="PCM_24")

        plans = {
            1: mod.ChannelPlan(
                path=kick_path,
                name="KICK",
                instrument="kick",
                pan=0.0,
                hpf=35.0,
                target_rms_db=-20.0,
                eq_bands=[],
                comp_threshold_db=-18.0,
                comp_ratio=4.0,
                comp_attack_ms=8.0,
                comp_release_ms=90.0,
                metrics=mod.metrics_for(kick, sr, instrument="kick"),
                event_activity=mod._event_activity_ranges(kick, sr, "kick") or {},
            ),
            2: mod.ChannelPlan(
                path=bass_path,
                name="Bass",
                instrument="bass_guitar",
                pan=0.0,
                hpf=35.0,
                target_rms_db=-21.0,
                eq_bands=[],
                comp_threshold_db=-22.0,
                comp_ratio=3.2,
                comp_attack_ms=18.0,
                comp_release_ms=180.0,
                metrics=mod.metrics_for(bass, sr, instrument="bass_guitar"),
            ),
            3: mod.ChannelPlan(
                path=overhead_path,
                name="OH L",
                instrument="overhead",
                pan=-0.7,
                hpf=150.0,
                target_rms_db=-27.0,
                eq_bands=[],
                comp_threshold_db=-18.0,
                comp_ratio=1.6,
                comp_attack_ms=25.0,
                comp_release_ms=300.0,
                metrics=mod.metrics_for(overhead, sr, instrument="overhead"),
            ),
        }

        report = mod.apply_stem_mix_verification(plans, length, sr, genre="rock")

    assert report["enabled"] is True
    assert report["applied"] is True
    assert report["before"]["stems"]
    assert report["before"]["band_hierarchy"]
    assert report["before"]["slope_conformity"]["reference_tilt_db_per_octave"] == 4.5
    assert report["before"]["tilt_conformity"]["MASTER"]["compensation_db_per_octave"] == 4.5
    assert report["before"]["slope_conformity"]["bass_deficit_db"] >= 0.0
    assert any(action["type"] == "kick_click_boost" for action in report["actions"])
    assert any(action["type"].startswith("master_slope_low_end_support") for action in report["actions"])
    assert plans[1].comp_ratio >= 5.4
    assert plans[1].comp_threshold_db <= -23.5
    assert report["after"]["kick_focus"]["kick_click_share_in_drums"] >= report["before"]["kick_focus"]["kick_click_share_in_drums"]


def test_master_process_uses_reference_mastering_when_reference_audio_present(monkeypatch):
    mod = load_offline_agent_mix()
    sr = 48_000
    mix = np.full((48_000, 2), 0.05, dtype=np.float32)
    reference = np.full((48_000, 2), 0.08, dtype=np.float32)
    captured = {}

    class DummyAutoMaster:
        def __init__(self, sample_rate, target_lufs, true_peak_limit):
            captured["init"] = {
                "sample_rate": sample_rate,
                "target_lufs": target_lufs,
                "true_peak_limit": true_peak_limit,
            }

        def master(self, audio, reference=None, sample_rate=None):
            captured["call"] = {
                "audio_shape": tuple(audio.shape),
                "reference_shape": tuple(reference.shape),
                "sample_rate": sample_rate,
            }
            return (audio * 1.4).astype(np.float32)

    monkeypatch.setattr(mod, "AutoMaster", DummyAutoMaster)
    context = mod.ReferenceMixContext(
        path=Path("/tmp/reference.wav"),
        source_type="audio",
        style_profile=mod.StyleProfile(name="ref", loudness_lufs=-60.0),
        audio=reference,
        sample_rate=sr,
    )

    mastered, report = mod.master_process(mix, sr, reference_context=context)

    assert mastered.shape == mix.shape
    assert report["reference_mastering"]["enabled"] is True
    assert report["reference_mastering"]["backend"] == "reference_audio_fallback"
    assert captured["call"]["reference_shape"] == tuple(reference.shape)


def test_master_process_rejects_reference_mastering_when_output_goes_too_quiet(monkeypatch):
    mod = load_offline_agent_mix()
    sr = 48_000
    mix = np.full((4096, 2), 0.08, dtype=np.float32)
    reference = np.full((4096, 2), 0.12, dtype=np.float32)

    class DummyAutoMaster:
        def __init__(self, sample_rate, target_lufs, true_peak_limit):
            pass

        def master(self, audio, reference=None, sample_rate=None):
            broken = np.zeros_like(audio, dtype=np.float32)
            broken[0, :] = 0.85
            return broken

    monkeypatch.setattr(mod, "AutoMaster", DummyAutoMaster)
    context = mod.ReferenceMixContext(
        path=Path("/tmp/reference-folder"),
        source_type="audio_directory",
        style_profile=mod.StyleProfile(name="ref", loudness_lufs=-13.0),
        audio=reference,
        sample_rate=sr,
        source_paths=[Path("/tmp/reference-a.wav"), Path("/tmp/reference-b.wav")],
    )

    mastered, report = mod.master_process(mix, sr, reference_context=context)

    assert mastered.shape == mix.shape
    assert report["reference_mastering"]["enabled"] is False
    assert report["reference_mastering"]["reason"] == "reference_mastering_rejected_low_loudness"
    assert np.count_nonzero(np.abs(mastered) > 1e-6) > 10
    assert not np.array_equal(mastered[0], np.array([0.85, 0.85], dtype=np.float32))


def test_master_process_conforms_reference_mastering_to_target_lufs(monkeypatch):
    mod = load_offline_agent_mix()
    sr = 48_000
    t = np.arange(48_000, dtype=np.float32) / sr
    mono_mix = (0.16 * np.sin(2.0 * np.pi * 110.0 * t)).astype(np.float32)
    mono_reference = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    mix = np.column_stack([mono_mix, mono_mix]).astype(np.float32)
    reference = np.column_stack([mono_reference, mono_reference]).astype(np.float32)

    class DummyAutoMaster:
        def __init__(self, sample_rate, target_lufs, true_peak_limit):
            self.target_lufs = target_lufs

        def master(self, audio, reference=None, sample_rate=None):
            return (audio * 3.0).astype(np.float32)

    monkeypatch.setattr(mod, "AutoMaster", DummyAutoMaster)
    context = mod.ReferenceMixContext(
        path=Path("/tmp/reference-folder"),
        source_type="audio_directory",
        style_profile=mod.StyleProfile(name="ref", loudness_lufs=-11.0),
        audio=reference,
        sample_rate=sr,
        source_paths=[Path("/tmp/reference-a.wav"), Path("/tmp/reference-b.wav")],
    )

    mastered, report = mod.master_process(mix, sr, target_lufs=-16.0, reference_context=context)

    assert mastered.shape == mix.shape
    assert report["reference_mastering"]["enabled"] is True
    assert report["reference_mastering"]["backend"] == "reference_audio_fallback"
    assert report["reference_mastering"]["target_lufs"] == -16.0
    assert report["reference_mastering"]["pre_target_conform_lufs"] != report["reference_mastering"]["lufs"]
    assert abs(report["post_master_lufs"] - (-16.0)) < 0.7


def test_apply_offline_fx_plan_uses_reference_send_override():
    mod = load_offline_agent_mix()
    sr = 48_000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    mono = (0.08 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    signal = np.column_stack([mono, mono]).astype(np.float32)
    reference = np.column_stack([
        (0.14 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
        (0.04 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
    ]).astype(np.float32)
    plans = {
        1: mod.ChannelPlan(
            path=Path("Lead.wav"),
            name="Lead",
            instrument="lead_vocal",
            pan=0.0,
            hpf=90.0,
            target_rms_db=-20.0,
            fx_send_db=-10.0,
        ),
    }
    context = mod.ReferenceMixContext(
        path=Path("/tmp/ref.wav"),
        source_type="audio",
        style_profile=mod.StyleTransfer().extract_style(reference, sr, name="ref"),
        audio=reference,
        sample_rate=sr,
    )
    mod._reference_targets_from_context(context)

    returns, report = mod.apply_offline_fx_plan({1: signal}, plans, sr, reference_context=context)

    assert report["enabled"] is True
    assert returns
    assert report["reference_send_overrides"]
    assert report["reference_send_overrides"][0]["target_db"] == -10.0
    assert report["reference_fx_overrides"]["enabled"] is True
    assert report["reference_fx_overrides"]["bus_adjustments"]


def test_stem_mix_verification_reports_reference_distance():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 1.4
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    lead = (0.06 * np.sin(2.0 * np.pi * 220.0 * t) + 0.02 * np.sin(2.0 * np.pi * 2600.0 * t)).astype(np.float32)
    bgv = (0.08 * np.sin(2.0 * np.pi * 330.0 * t) + 0.03 * np.sin(2.0 * np.pi * 1800.0 * t)).astype(np.float32)
    kick = (0.14 * np.sin(2.0 * np.pi * 62.0 * t) + 0.01 * np.sin(2.0 * np.pi * 3200.0 * t)).astype(np.float32)
    bass = (0.15 * np.sin(2.0 * np.pi * 72.0 * t)).astype(np.float32)
    reference = np.column_stack([
        (0.42 * np.sin(2.0 * np.pi * 220.0 * t) + 0.18 * np.sin(2.0 * np.pi * 3100.0 * t)).astype(np.float32),
        (0.24 * np.sin(2.0 * np.pi * 72.0 * t) + 0.07 * np.sin(2.0 * np.pi * 5200.0 * t)).astype(np.float32),
    ])

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        lead_path = tmp / "Lead.wav"
        bgv_path = tmp / "BGV.wav"
        kick_path = tmp / "Kick.wav"
        bass_path = tmp / "Bass.wav"
        reference_path = tmp / "Reference.wav"
        sf.write(lead_path, lead, sr, subtype="PCM_24")
        sf.write(bgv_path, bgv, sr, subtype="PCM_24")
        sf.write(kick_path, kick, sr, subtype="PCM_24")
        sf.write(bass_path, bass, sr, subtype="PCM_24")
        sf.write(reference_path, reference, sr, subtype="PCM_24")

        plans = {
            1: mod.ChannelPlan(
                path=lead_path,
                name="Lead",
                instrument="lead_vocal",
                pan=0.0,
                hpf=90.0,
                target_rms_db=-20.0,
                fader_db=-6.0,
                metrics=mod.metrics_for(lead, sr, instrument="lead_vocal"),
            ),
            2: mod.ChannelPlan(
                path=bgv_path,
                name="BGV",
                instrument="backing_vocal",
                pan=0.2,
                hpf=100.0,
                target_rms_db=-22.0,
                fader_db=-3.5,
                metrics=mod.metrics_for(bgv, sr, instrument="backing_vocal"),
            ),
            3: mod.ChannelPlan(
                path=kick_path,
                name="Kick",
                instrument="kick",
                pan=0.0,
                hpf=35.0,
                target_rms_db=-20.0,
                metrics=mod.metrics_for(kick, sr, instrument="kick"),
                event_activity=mod._event_activity_ranges(kick, sr, "kick") or {},
            ),
            4: mod.ChannelPlan(
                path=bass_path,
                name="Bass",
                instrument="bass_guitar",
                pan=0.0,
                hpf=35.0,
                target_rms_db=-21.0,
                metrics=mod.metrics_for(bass, sr, instrument="bass_guitar"),
            ),
        }
        context = mod.prepare_reference_mix_context(reference_path)
        report = mod.apply_stem_mix_verification(plans, len(lead), sr, genre="rock", reference_context=context)

    assert report["enabled"] is True
    assert report["reference_targets"]["hierarchy"]
    assert report["reference_distance"]["before"]["enabled"] is True
    assert "overall_distance" in report["reference_distance"]["after"]
    assert report["before"]["hierarchy_metrics"]
    assert report["after"]["hierarchy_metrics"]


def test_reference_vocal_fx_focus_only_adjusts_vocal_space():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 1.2
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    lead = (0.10 * np.sin(2.0 * np.pi * 220.0 * t) + 0.04 * np.sin(2.0 * np.pi * 2800.0 * t)).astype(np.float32)
    bgv = (0.035 * np.sin(2.0 * np.pi * 330.0 * t) + 0.01 * np.sin(2.0 * np.pi * 2200.0 * t)).astype(np.float32)
    reference = np.column_stack([
        (0.12 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32),
        (0.05 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32),
    ])

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        lead_path = tmp / "Lead.wav"
        bgv_path = tmp / "BGV.wav"
        kick_path = tmp / "Kick.wav"
        ref_path = tmp / "Ref.wav"
        sf.write(lead_path, lead, sr, subtype="PCM_24")
        sf.write(bgv_path, bgv, sr, subtype="PCM_24")
        sf.write(kick_path, 0.08 * np.sin(2.0 * np.pi * 60.0 * t).astype(np.float32), sr, subtype="PCM_24")
        sf.write(ref_path, reference, sr, subtype="PCM_24")

        plans = {
            1: mod.ChannelPlan(path=lead_path, name="Lead", instrument="lead_vocal", pan=0.0, hpf=90.0, target_rms_db=-20.0, fader_db=0.0),
            2: mod.ChannelPlan(path=bgv_path, name="BGV", instrument="backing_vocal", pan=0.3, hpf=100.0, target_rms_db=-24.0, fader_db=-4.0),
            3: mod.ChannelPlan(path=kick_path, name="Kick", instrument="kick", pan=0.0, hpf=35.0, target_rms_db=-20.0, fader_db=-1.0),
        }
        context = mod.prepare_reference_mix_context(ref_path)
        before_kick_fader = plans[3].fader_db
        report = mod.apply_reference_vocal_fx_focus(plans, len(lead), sr, context)

    assert report["enabled"] is True
    assert report["applied"] is True
    assert plans[3].fader_db == before_kick_fader
    assert plans[1].fx_bus_send_db
    assert report["after"]["lead_over_bgv_rms_db"] < report["before"]["lead_over_bgv_rms_db"]


def test_frequency_window_balance_clears_vocal_window_and_keeps_kick_untouched():
    mod = load_offline_agent_mix()
    sr = 48_000
    duration_sec = 1.6
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr

    lead = (
        0.035 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.03 * np.sin(2.0 * np.pi * 980.0 * t)
        + 0.02 * np.sin(2.0 * np.pi * 2900.0 * t)
    ).astype(np.float32)
    guitar = (
        0.06 * np.sin(2.0 * np.pi * 320.0 * t)
        + 0.08 * np.sin(2.0 * np.pi * 1100.0 * t)
        + 0.07 * np.sin(2.0 * np.pi * 3000.0 * t)
    ).astype(np.float32)
    bgv = (
        0.03 * np.sin(2.0 * np.pi * 260.0 * t)
        + 0.045 * np.sin(2.0 * np.pi * 1020.0 * t)
        + 0.03 * np.sin(2.0 * np.pi * 3100.0 * t)
    ).astype(np.float32)
    hat = (
        0.04 * np.sin(2.0 * np.pi * 7600.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * 9800.0 * t)
    ).astype(np.float32)
    kick = (0.08 * np.sin(2.0 * np.pi * 62.0 * t)).astype(np.float32)

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        lead_path = tmp / "Lead.wav"
        guitar_path = tmp / "Guitar.wav"
        bgv_path = tmp / "BGV.wav"
        hat_path = tmp / "Hat.wav"
        kick_path = tmp / "Kick.wav"
        sf.write(lead_path, lead, sr, subtype="PCM_24")
        sf.write(guitar_path, guitar, sr, subtype="PCM_24")
        sf.write(bgv_path, bgv, sr, subtype="PCM_24")
        sf.write(hat_path, hat, sr, subtype="PCM_24")
        sf.write(kick_path, kick, sr, subtype="PCM_24")

        plans = {
            1: mod.ChannelPlan(path=lead_path, name="Lead", instrument="lead_vocal", pan=0.0, hpf=90.0, target_rms_db=-20.0),
            2: mod.ChannelPlan(path=guitar_path, name="Guitar", instrument="electric_guitar", pan=-0.25, hpf=90.0, target_rms_db=-22.0),
            3: mod.ChannelPlan(path=bgv_path, name="BGV", instrument="backing_vocal", pan=0.25, hpf=100.0, target_rms_db=-23.0),
            4: mod.ChannelPlan(path=hat_path, name="Hat", instrument="hi_hat", pan=0.35, hpf=300.0, target_rms_db=-28.0),
            5: mod.ChannelPlan(path=kick_path, name="Kick", instrument="kick", pan=0.0, hpf=35.0, target_rms_db=-20.0),
        }
        kick_eq_before = list(plans[5].eq_bands)
        report = mod.apply_frequency_window_balance(plans, len(lead), sr)

    assert report["enabled"] is True
    assert report["applied"] is True
    action_windows = {item["window"] for item in report["actions"]}
    assert "vocal_conflict" in action_windows
    assert "air_sibilance" in action_windows
    assert plans[2].eq_bands
    assert plans[4].eq_bands
    assert plans[5].eq_bands == kick_eq_before
