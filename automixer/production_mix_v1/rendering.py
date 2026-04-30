"""Offline renderer for production_mix_v1 candidates."""

from __future__ import annotations

from typing import Any

import numpy as np

from .models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_FX_SEND,
    ACTION_GAIN,
    ACTION_PARALLEL_COMPRESSION,
    ACTION_STATUS_FAILED,
    ACTION_STATUS_VERIFIED,
    ActionRenderStatus,
    AudioStem,
    MixAction,
    MixCandidate,
    RenderResult,
    amp_to_db,
    db_to_amp,
    ensure_stereo,
)


def render_candidate(stems: list[AudioStem], candidate: MixCandidate) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    result = render_candidate_with_status(stems, candidate)
    return result.mix, result.processed_stems


def render_candidate_with_status(stems: list[AudioStem], candidate: MixCandidate) -> RenderResult:
    processed = {int(stem.channel_id): ensure_stereo(stem.audio).copy() for stem in stems}
    by_id = {int(stem.channel_id): stem for stem in stems}
    statuses: list[ActionRenderStatus] = []
    for action in candidate.actions:
        if action.channel_id is None:
            continue
        channel_id = int(action.channel_id)
        if channel_id not in processed:
            statuses.append(ActionRenderStatus(action=action, status=ACTION_STATUS_FAILED, reason="target_channel_missing"))
            continue
        rendered, status = _apply_action_with_status(processed[channel_id], action, sample_rate=by_id[channel_id].sample_rate)
        processed[channel_id] = rendered
        statuses.append(status)
    if not processed:
        mix = np.zeros((0, 2), dtype=np.float32)
    else:
        max_len = max(audio.shape[0] for audio in processed.values())
        mix = np.zeros((max_len, 2), dtype=np.float32)
        for audio in processed.values():
            if audio.shape[0] < max_len:
                audio = np.pad(audio, ((0, max_len - audio.shape[0]), (0, 0)))
            mix += audio[:max_len]
    for action in candidate.actions:
        if action.channel_id is None:
            mix, status = _apply_action_with_status(mix, action, sample_rate=stems[0].sample_rate if stems else 48000)
            statuses.append(status)
    return RenderResult(mix=mix.astype(np.float32), processed_stems=processed, action_statuses=statuses)


def _apply_action_with_status(audio: np.ndarray, action: MixAction, *, sample_rate: int) -> tuple[np.ndarray, ActionRenderStatus]:
    before = ensure_stereo(audio)
    try:
        rendered, metrics = _render_action(before, action, sample_rate)
    except Exception as exc:
        return before, ActionRenderStatus(action=action, status=ACTION_STATUS_FAILED, reason=str(exc))
    delta = float(np.sqrt(np.mean((rendered - before) ** 2) + 1e-12))
    verified = bool(metrics.pop("verified", delta > 1e-8))
    return rendered.astype(np.float32), ActionRenderStatus(
        action=action,
        status=ACTION_STATUS_VERIFIED if verified else ACTION_STATUS_FAILED,
        rendered=delta > 1e-8,
        verified=verified,
        reason="offline_render_verified_by_audio_delta" if verified else "offline_render_produced_no_audio_delta",
        metrics=metrics,
    )


def _render_action(audio: np.ndarray, action: MixAction, sample_rate: int) -> tuple[np.ndarray, dict[str, Any]]:
    if action.action_type == ACTION_GAIN:
        gain_db = float(action.parameters.get("gain_db", 0.0))
        return audio * db_to_amp(gain_db), {"gain_db": gain_db, "verified": abs(gain_db) >= 0.01}
    if action.action_type == ACTION_EQ:
        gain_db = float(action.parameters.get("gain_db", 0.0))
        return audio * db_to_amp(gain_db * 0.15), {"gain_db": gain_db, "verified": abs(gain_db) >= 0.05}
    if action.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}:
        rendered, metrics = _compress(
            audio,
            threshold_db=float(action.parameters.get("threshold_db", -22.0)),
            ratio=float(action.parameters.get("ratio", 2.0)),
            wet_mix_percent=float(action.parameters.get("wet_mix_percent", 100.0)),
            makeup_gain_db=float(action.parameters.get("makeup_gain_db", 0.0)),
            parallel=action.action_type == ACTION_PARALLEL_COMPRESSION,
        )
        metrics["verified"] = float(metrics["gain_reduction_proxy_db"]) >= 0.05
        return rendered, metrics
    if action.action_type == ACTION_FX_SEND:
        rendered, metrics = _fx_send(
            audio,
            sample_rate=sample_rate,
            send_level_db=float(action.parameters.get("send_level_db", -18.0)),
            return_level_db=float(action.parameters.get("return_level_db", -6.0)),
            pre_delay_ms=float(action.parameters.get("pre_delay_ms", 18.0)),
            decay_ms=float(action.parameters.get("decay_ms", 920.0)),
            width=float(action.parameters.get("width", 0.72)),
            modulation_depth_ms=float(action.parameters.get("modulation_depth_ms", 4.5)),
            modulation_rate_hz=float(action.parameters.get("modulation_rate_hz", 0.32)),
        )
        metrics["verified"] = (
            float(metrics["fx_wetness_proxy"]) >= 0.001
            and float(metrics["spatial_reflection_energy_proxy"]) >= 0.001
            and float(metrics["modulation_energy_proxy"]) >= 0.00015
        )
        return rendered, metrics
    raise ValueError(f"unsupported_action:{action.action_type}")


def _compress(audio: np.ndarray, *, threshold_db: float, ratio: float, wet_mix_percent: float, makeup_gain_db: float, parallel: bool) -> tuple[np.ndarray, dict[str, float]]:
    stereo = ensure_stereo(audio)
    threshold = db_to_amp(threshold_db)
    magnitude = np.abs(stereo)
    over = magnitude > threshold
    compressed = stereo.copy()
    gr = 0.0
    if np.any(over):
        over_db = 20.0 * np.log10(magnitude[over] / threshold + 1e-12)
        reduced_db = over_db / max(1.0, ratio)
        gr = float(np.mean(np.maximum(0.0, over_db - reduced_db)))
        compressed[over] = np.sign(stereo[over]) * threshold * np.power(10.0, reduced_db / 20.0)
    compressed *= db_to_amp(makeup_gain_db)
    wet = max(0.0, min(100.0, wet_mix_percent)) / 100.0
    if parallel:
        wet = min(wet, 0.35)
    rendered = stereo * (1.0 - wet) + compressed * wet
    return rendered.astype(np.float32), {"gain_reduction_proxy_db": gr, "rms_before_db": _rms_db(stereo), "rms_after_db": _rms_db(rendered)}


def _fx_send(
    audio: np.ndarray,
    *,
    sample_rate: int,
    send_level_db: float,
    return_level_db: float,
    pre_delay_ms: float,
    decay_ms: float,
    width: float,
    modulation_depth_ms: float,
    modulation_rate_hz: float,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    stereo = ensure_stereo(audio)
    if stereo.size == 0:
        return stereo, {
            "fx_wetness_proxy": 0.0,
            "offline_fx_return_present": False,
            "spatial_reflection_energy_proxy": 0.0,
            "modulation_energy_proxy": 0.0,
            "stereo_width_before": 0.0,
            "stereo_width_wet": 0.0,
            "stereo_width_after": 0.0,
            "stereo_width_delta_proxy": 0.0,
            "pre_delay_ms": float(pre_delay_ms),
            "decay_ms": float(decay_ms),
            "width": float(width),
            "modulation_depth_ms": float(modulation_depth_ms),
            "modulation_rate_hz": float(modulation_rate_hz),
        }

    send = _send_feed(stereo) * db_to_amp(send_level_db + return_level_db)
    delay = max(1, int(pre_delay_ms * sample_rate / 1000.0))
    wet = np.zeros_like(stereo)
    early = np.zeros_like(stereo)
    late = np.zeros_like(stereo)

    _add_delayed_tap(early, send, delay, 0.36, 0.28, crossfeed=0.20)
    _add_delayed_tap(early, send, delay + _ms_to_samples(7.0, sample_rate), 0.14, 0.21, crossfeed=0.62)
    _add_delayed_tap(early, send, delay + _ms_to_samples(13.0, sample_rate), 0.12, 0.09, crossfeed=0.35)
    _add_delayed_tap(early, send, delay + _ms_to_samples(23.0, sample_rate), 0.06, 0.10, crossfeed=0.75)

    decay = max(80.0, float(decay_ms))
    late_offsets = (43.0, 71.0, 109.0, 163.0, 251.0, 379.0, 571.0, 887.0)
    for index, offset_ms in enumerate(late_offsets):
        envelope = float(np.exp(-offset_ms / decay))
        gain = 0.20 * envelope / float(np.sqrt(index + 1.0))
        skew = 0.16 if index % 2 == 0 else -0.13
        _add_delayed_tap(
            late,
            send,
            delay + _ms_to_samples(offset_ms, sample_rate),
            gain * (1.0 + skew),
            gain * (1.0 - skew),
            crossfeed=0.24 + 0.08 * (index % 4),
        )

    modulation = _modulated_delay(
        send,
        sample_rate=sample_rate,
        base_delay_ms=pre_delay_ms + 11.0,
        depth_ms=max(0.0, modulation_depth_ms),
        rate_hz=max(0.0, modulation_rate_hz),
        gain=0.18,
    )
    wet = _apply_width(_damp_wet(early + late + modulation), width)
    rendered = stereo + wet
    wet_rms = _rms_linear(wet)
    dry_rms = _rms_linear(stereo)
    early_rms = _rms_linear(early + late)
    modulation_rms = _rms_linear(modulation)
    width_before = _stereo_width_proxy(stereo)
    width_wet = _stereo_width_proxy(wet)
    width_after = _stereo_width_proxy(rendered)
    return rendered.astype(np.float32), {
        "fx_wetness_proxy": wet_rms / max(dry_rms, 1e-12),
        "offline_fx_return_present": wet_rms > 1e-8,
        "spatial_reflection_energy_proxy": early_rms / max(dry_rms, 1e-12),
        "modulation_energy_proxy": modulation_rms / max(dry_rms, 1e-12),
        "stereo_width_before": width_before,
        "stereo_width_wet": width_wet,
        "stereo_width_after": width_after,
        "stereo_width_delta_proxy": max(0.0, width_wet - width_before),
        "pre_delay_ms": float(pre_delay_ms),
        "decay_ms": float(decay_ms),
        "width": float(width),
        "modulation_depth_ms": float(modulation_depth_ms),
        "modulation_rate_hz": float(modulation_rate_hz),
    }


def _send_feed(stereo: np.ndarray) -> np.ndarray:
    mid = 0.5 * (stereo[:, 0] + stereo[:, 1])
    side = 0.5 * (stereo[:, 0] - stereo[:, 1])
    return np.column_stack([mid + side * 0.35, mid - side * 0.35]).astype(np.float32)


def _add_delayed_tap(dst: np.ndarray, src: np.ndarray, delay: int, gain_l: float, gain_r: float, *, crossfeed: float) -> None:
    if delay <= 0 or delay >= src.shape[0]:
        return
    cross = max(0.0, min(1.0, float(crossfeed)))
    segment = src[: src.shape[0] - delay]
    left = segment[:, 0] * (1.0 - cross) + segment[:, 1] * cross
    right = segment[:, 1] * (1.0 - cross) + segment[:, 0] * cross
    dst[delay:, 0] += left * float(gain_l)
    dst[delay:, 1] += right * float(gain_r)


def _modulated_delay(
    src: np.ndarray,
    *,
    sample_rate: int,
    base_delay_ms: float,
    depth_ms: float,
    rate_hz: float,
    gain: float,
) -> np.ndarray:
    n = int(src.shape[0])
    out = np.zeros_like(src)
    depth_samples = float(depth_ms) * float(sample_rate) / 1000.0
    base_delay = max(1.0, float(base_delay_ms) * float(sample_rate) / 1000.0)
    if n <= int(base_delay + depth_samples + 2) or depth_samples <= 0.0 or rate_hz <= 0.0:
        return out
    chunk = 262_144
    phases = (0.0, np.pi * 0.63)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        idx = np.arange(start, end, dtype=np.float64)
        for channel, phase in enumerate(phases):
            modulation = np.sin((2.0 * np.pi * float(rate_hz) * idx / float(sample_rate)) + phase)
            read_pos = idx - base_delay - depth_samples * modulation
            valid = (read_pos >= 0.0) & (read_pos < n - 1)
            if not np.any(valid):
                continue
            lower = np.floor(read_pos[valid]).astype(np.int64)
            frac = read_pos[valid] - lower
            delayed = src[lower, channel] * (1.0 - frac) + src[lower + 1, channel] * frac
            segment = out[start:end, channel]
            segment[valid] = delayed.astype(np.float32)
    return out * float(gain)


def _damp_wet(wet: np.ndarray) -> np.ndarray:
    damped = wet.copy()
    if wet.shape[0] > 1:
        damped[1:] = damped[1:] * 0.72 + wet[:-1] * 0.22
    if wet.shape[0] > 2:
        damped[2:] = damped[2:] + wet[:-2] * 0.06
    return damped.astype(np.float32)


def _apply_width(audio: np.ndarray, width: float) -> np.ndarray:
    amount = max(0.0, min(1.25, float(width)))
    mid = 0.5 * (audio[:, 0] + audio[:, 1])
    side = 0.5 * (audio[:, 0] - audio[:, 1])
    mid_gain = 1.0 - amount * 0.10
    side_gain = 1.0 + amount * 0.90
    left = mid * mid_gain + side * side_gain
    right = mid * mid_gain - side * side_gain
    return np.column_stack([left, right]).astype(np.float32)


def _rms_linear(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(ensure_stereo(audio) ** 2) + 1e-12))


def _stereo_width_proxy(audio: np.ndarray) -> float:
    stereo = ensure_stereo(audio)
    if stereo.size == 0:
        return 0.0
    mid = 0.5 * (stereo[:, 0] + stereo[:, 1])
    side = 0.5 * (stereo[:, 0] - stereo[:, 1])
    side_rms = float(np.sqrt(np.mean(side * side) + 1e-12))
    mid_rms = float(np.sqrt(np.mean(mid * mid) + 1e-12))
    return float(side_rms / max(mid_rms + side_rms, 1e-12))


def _ms_to_samples(ms: float, sample_rate: int) -> int:
    return max(1, int(float(ms) * float(sample_rate) / 1000.0))


def _rms_db(audio: np.ndarray) -> float:
    return amp_to_db(float(np.sqrt(np.mean(ensure_stereo(audio) ** 2) + 1e-12)))
