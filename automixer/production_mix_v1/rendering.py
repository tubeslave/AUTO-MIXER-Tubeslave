"""Candidate rendering utilities."""

from __future__ import annotations

import numpy as np

from .models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_FX_SEND,
    ACTION_GAIN,
    ACTION_PAN,
    ACTION_PARALLEL_COMPRESSION,
    ACTION_STATUS_FAILED,
    ACTION_STATUS_RENDERED,
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


def render_candidate(
    stems: list[AudioStem],
    candidate: MixCandidate,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Render a candidate mix from stems and staged actions."""
    result = render_candidate_with_status(stems, candidate)
    return result.mix, result.processed_stems


def render_candidate_with_status(
    stems: list[AudioStem],
    candidate: MixCandidate,
) -> RenderResult:
    """Render a candidate and return per-action offline render evidence."""
    processed: dict[int, np.ndarray] = {
        int(stem.channel_id): _apply_pan(ensure_stereo(stem.audio).copy(), stem.base_pan)
        if abs(float(stem.base_pan)) > 1e-6
        else ensure_stereo(stem.audio).copy()
        for stem in stems
    }
    stem_by_channel = {int(stem.channel_id): stem for stem in stems}
    statuses: list[ActionRenderStatus] = []
    for action in candidate.actions:
        channel_id = action.channel_id
        if channel_id is None:
            continue
        if int(channel_id) not in processed:
            statuses.append(
                ActionRenderStatus(
                    action=action,
                    status=ACTION_STATUS_FAILED,
                    rendered=False,
                    verified=False,
                    reason="target_channel_missing_from_offline_render",
                )
            )
            continue
        rendered, status = _apply_action_with_status(
            processed[int(channel_id)],
            action,
            sample_rate=stem_by_channel[int(channel_id)].sample_rate,
        )
        processed[int(channel_id)] = rendered
        statuses.append(status)
    if not processed:
        return RenderResult(np.zeros((0, 2), dtype=np.float32), processed, statuses)
    target_len = max(audio.shape[0] for audio in processed.values())
    mix = np.zeros((target_len, 2), dtype=np.float32)
    for audio in processed.values():
        if audio.shape[0] < target_len:
            audio = np.pad(audio, ((0, target_len - audio.shape[0]), (0, 0)))
        mix += audio[:target_len]
    for action in candidate.actions:
        if action.channel_id is None:
            mix, status = _apply_action_with_status(
                mix,
                action,
                sample_rate=stems[0].sample_rate if stems else 48000,
            )
            statuses.append(status)
    return RenderResult(mix.astype(np.float32), processed, statuses)


def _apply_action(audio: np.ndarray, action: MixAction, *, sample_rate: int) -> np.ndarray:
    rendered, _status = _apply_action_with_status(audio, action, sample_rate=sample_rate)
    return rendered


def _apply_action_with_status(
    audio: np.ndarray,
    action: MixAction,
    *,
    sample_rate: int,
) -> tuple[np.ndarray, ActionRenderStatus]:
    before = ensure_stereo(audio)
    try:
        rendered, metrics = _render_action(before, action, sample_rate=sample_rate)
    except Exception as exc:
        return before, ActionRenderStatus(
            action=action,
            status=ACTION_STATUS_FAILED,
            rendered=False,
            verified=False,
            reason=f"offline_render_failed:{exc}",
        )
    rendered = ensure_stereo(rendered)
    changed = _changed(before, rendered)
    verified = bool(metrics.pop("verified", changed))
    if changed and verified:
        status = ACTION_STATUS_VERIFIED
        reason = "offline_render_verified_by_audio_delta"
    elif changed:
        status = ACTION_STATUS_RENDERED
        reason = "offline_render_changed_audio_but_metric_threshold_not_met"
    else:
        status = ACTION_STATUS_FAILED
        reason = "offline_render_produced_no_audio_delta"
    return rendered.astype(np.float32), ActionRenderStatus(
        action=action,
        status=status,
        rendered=changed,
        verified=changed and verified,
        reason=reason,
        metrics=metrics,
    )


def _render_action(
    audio: np.ndarray,
    action: MixAction,
    *,
    sample_rate: int,
) -> tuple[np.ndarray, dict[str, float | bool | str]]:
    if action.action_type == ACTION_GAIN:
        gain_db = float(action.parameters.get("gain_db", action.parameters.get("delta_db", 0.0)))
        rendered = (audio * db_to_amp(gain_db)).astype(np.float32)
        return rendered, {
            "gain_db": gain_db,
            "verified": abs(gain_db) >= 0.01,
        }
    if action.action_type == ACTION_EQ:
        gain_db = float(action.parameters.get("gain_db", 0.0))
        rendered = _apply_fft_eq(
            audio,
            sample_rate=sample_rate,
            frequency_hz=float(action.parameters.get("frequency_hz", 1000.0)),
            gain_db=gain_db,
            q=float(action.parameters.get("q", 1.0)),
        )
        return rendered, {
            "frequency_hz": float(action.parameters.get("frequency_hz", 1000.0)),
            "gain_db": gain_db,
            "verified": abs(gain_db) >= 0.05,
        }
    if action.action_type == ACTION_COMPRESSION:
        missing = _missing_compression_params(action)
        if missing:
            raise ValueError(f"missing_compression_parameters:{','.join(missing)}")
        rendered, metrics = _apply_compression(
            audio,
            threshold_db=float(action.parameters.get("threshold_db", -18.0)),
            ratio=float(action.parameters.get("ratio", 2.0)),
            makeup_gain_db=float(action.parameters.get("makeup_gain_db", 0.0)),
            wet_mix_percent=float(action.parameters.get("wet_mix_percent", 100.0)),
        )
        target = float(action.parameters.get("target_gain_reduction_db", 0.4))
        metrics["target_gain_reduction_db"] = target
        metrics["verified"] = float(metrics.get("gain_reduction_proxy_db", 0.0)) >= min(0.20, target)
        return rendered, metrics
    if action.action_type == ACTION_PARALLEL_COMPRESSION:
        missing = _missing_compression_params(action)
        if missing:
            raise ValueError(f"missing_compression_parameters:{','.join(missing)}")
        rendered, metrics = _apply_parallel_compression(
            audio,
            threshold_db=float(action.parameters.get("threshold_db", -24.0)),
            ratio=float(action.parameters.get("ratio", 3.0)),
            wet_mix_percent=float(action.parameters.get("wet_mix_percent", 22.0)),
            makeup_gain_db=float(action.parameters.get("makeup_gain_db", 0.0)),
        )
        target = float(action.parameters.get("target_gain_reduction_db", 0.4))
        metrics["target_gain_reduction_db"] = target
        metrics["verified"] = float(metrics.get("gain_reduction_proxy_db", 0.0)) >= min(0.15, target)
        return rendered, metrics
    if action.action_type == ACTION_FX_SEND:
        missing = _missing_fx_params(action)
        if missing:
            raise ValueError(f"missing_fx_parameters:{','.join(missing)}")
        if not bool(action.parameters.get("route_to_main", False)):
            raise ValueError("fx_return_not_routed_to_main")
        rendered, metrics = _apply_fx_send(
            audio,
            sample_rate=sample_rate,
            send_level_db=float(action.parameters.get("send_level_db", -18.0)),
            return_level_db=float(action.parameters.get("return_level_db", -12.0)),
            pre_delay_ms=float(action.parameters.get("pre_delay_ms", 18.0)),
            decay_ms=float(action.parameters.get("decay_ms", 900.0)),
            width=float(action.parameters.get("width", 0.70)),
        )
        wet_target = float(action.parameters.get("wet_target_percent", 4.0)) / 100.0
        metrics["wet_target_percent"] = float(action.parameters.get("wet_target_percent", 4.0))
        metrics["offline_fx_return_present"] = True
        metrics["verified"] = float(metrics.get("fx_wetness_proxy", 0.0)) >= min(0.002, wet_target)
        return rendered, metrics
    if action.action_type == ACTION_PAN:
        rendered = _apply_pan(audio, float(action.parameters.get("pan", 0.0)))
        return rendered, {
            "pan": float(action.parameters.get("pan", 0.0)),
            "verified": abs(float(action.parameters.get("pan", 0.0))) >= 0.01,
        }
    raise ValueError(f"unsupported_action_type:{action.action_type}")


def _apply_fft_eq(
    audio: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    gain_db: float,
    q: float,
) -> np.ndarray:
    if abs(gain_db) < 1e-6 or audio.size == 0:
        return audio
    stereo = ensure_stereo(audio)
    freq = max(20.0, min(float(sample_rate) / 2.0 - 1.0, frequency_hz))
    width_oct = max(0.25, min(2.5, 1.0 / max(0.25, q) * 1.5))
    result = np.zeros_like(stereo)
    for channel in range(stereo.shape[1]):
        spectrum = np.fft.rfft(stereo[:, channel])
        freqs = np.fft.rfftfreq(stereo.shape[0], 1.0 / sample_rate)
        log_distance = np.log2(np.maximum(freqs, 20.0) / freq)
        shape = np.exp(-0.5 * (log_distance / width_oct) ** 2)
        gain_curve = np.power(10.0, (gain_db * shape) / 20.0)
        result[:, channel] = np.fft.irfft(spectrum * gain_curve, n=stereo.shape[0]).astype(np.float32)
    return result.astype(np.float32)


def _apply_compression(
    audio: np.ndarray,
    *,
    threshold_db: float,
    ratio: float,
    makeup_gain_db: float,
    wet_mix_percent: float,
) -> tuple[np.ndarray, dict[str, float]]:
    stereo = ensure_stereo(audio)
    threshold = db_to_amp(threshold_db)
    ratio = max(1.0, float(ratio))
    wet = max(0.0, min(100.0, float(wet_mix_percent))) / 100.0
    magnitude = np.abs(stereo)
    over = magnitude > threshold
    compressed = stereo.copy()
    gain_reduction_proxy = 0.0
    if np.any(over):
        sign = np.sign(stereo[over])
        over_db = 20.0 * np.log10(magnitude[over] / threshold + 1e-12)
        reduced_db = over_db / ratio
        gain_reduction_proxy = float(np.mean(np.maximum(0.0, over_db - reduced_db)))
        compressed_mag = threshold * np.power(10.0, reduced_db / 20.0)
        compressed[over] = sign * compressed_mag
    compressed = compressed * db_to_amp(makeup_gain_db)
    rendered = stereo * (1.0 - wet) + compressed * wet
    return rendered.astype(np.float32), {
        "gain_reduction_proxy_db": gain_reduction_proxy,
        "wet_mix_percent": wet_mix_percent,
        "makeup_gain_db": makeup_gain_db,
        "rms_before_db": _rms_db(stereo),
        "rms_after_db": _rms_db(rendered),
    }


def _apply_parallel_compression(
    audio: np.ndarray,
    *,
    threshold_db: float,
    ratio: float,
    wet_mix_percent: float,
    makeup_gain_db: float,
) -> tuple[np.ndarray, dict[str, float]]:
    stereo = ensure_stereo(audio)
    wet = max(0.0, min(45.0, float(wet_mix_percent))) / 100.0
    compressed, metrics = _apply_compression(
        stereo,
        threshold_db=threshold_db,
        ratio=ratio,
        makeup_gain_db=makeup_gain_db,
        wet_mix_percent=100.0,
    )
    rendered = stereo * (1.0 - wet) + compressed * wet
    metrics.update(
        {
            "wet_mix_percent": wet_mix_percent,
            "parallel_blend": wet,
            "rms_before_db": _rms_db(stereo),
            "rms_after_db": _rms_db(rendered),
        }
    )
    return rendered.astype(np.float32), metrics


def _apply_fx_send(
    audio: np.ndarray,
    *,
    sample_rate: int,
    send_level_db: float,
    return_level_db: float,
    pre_delay_ms: float,
    decay_ms: float,
    width: float,
) -> tuple[np.ndarray, dict[str, float]]:
    stereo = ensure_stereo(audio)
    if stereo.size == 0:
        return stereo, {"fx_wetness_proxy": 0.0}
    send_amp = db_to_amp(min(-6.0, float(send_level_db)))
    return_amp = db_to_amp(min(0.0, float(return_level_db)))
    width = max(0.0, min(1.0, float(width)))
    pre = max(1, int(float(pre_delay_ms) * sample_rate / 1000.0))
    decay = max(1.0, float(decay_ms))
    taps_ms = [0.0, 31.0, 67.0, 113.0, 181.0]
    mono = np.mean(stereo, axis=1)
    wet_left = np.zeros_like(mono)
    wet_right = np.zeros_like(mono)
    for idx, tap_ms in enumerate(taps_ms):
        delay = pre + int(tap_ms * sample_rate / 1000.0)
        if delay >= mono.shape[0]:
            continue
        gain = float(np.exp(-(float(pre_delay_ms) + tap_ms) / decay)) / float(idx + 1)
        source = mono[: mono.shape[0] - delay] * gain
        if idx % 2 == 0:
            wet_left[delay:] += source
            wet_right[delay:] += source * (1.0 - 0.35 * width)
        else:
            wet_left[delay:] += source * (1.0 - 0.35 * width)
            wet_right[delay:] += source
    wet = np.column_stack([wet_left, wet_right]).astype(np.float32)
    wet_render = wet * send_amp * return_amp
    rendered = (stereo + wet_render).astype(np.float32)
    wet_rms = float(np.sqrt(np.mean(wet_render * wet_render) + 1e-12))
    dry_rms = float(np.sqrt(np.mean(stereo * stereo) + 1e-12))
    return rendered, {
        "fx_wetness_proxy": float(wet_rms / max(dry_rms, 1e-12)),
        "send_level_db": float(send_level_db),
        "return_level_db": float(return_level_db),
    }


def _apply_pan(audio: np.ndarray, pan: float) -> np.ndarray:
    pan = max(-1.0, min(1.0, float(pan)))
    mono = np.mean(ensure_stereo(audio), axis=1)
    angle = (pan + 1.0) * np.pi / 4.0
    left = np.cos(angle) * mono
    right = np.sin(angle) * mono
    return np.column_stack([left, right]).astype(np.float32)


COMPRESSION_REQUIRED_PARAMS = {
    "threshold_db",
    "ratio",
    "attack_ms",
    "release_ms",
    "knee_db",
    "makeup_gain_db",
    "wet_mix_percent",
    "target_gain_reduction_db",
    "reason",
    "expected_metric_change",
}

FX_REQUIRED_PARAMS = {
    "fx_slot",
    "fx_model",
    "send_level_db",
    "return_level_db",
    "send_mode",
    "route_to_main",
    "wet_target_percent",
    "reason",
}


def _missing_compression_params(action: MixAction) -> list[str]:
    return [
        key
        for key in sorted(COMPRESSION_REQUIRED_PARAMS)
        if key not in action.parameters or action.parameters.get(key) in ("", None)
    ]


def _missing_fx_params(action: MixAction) -> list[str]:
    return [
        key
        for key in sorted(FX_REQUIRED_PARAMS)
        if key not in action.parameters or action.parameters.get(key) in ("", None)
    ]


def _changed(before: np.ndarray, after: np.ndarray) -> bool:
    if before.shape != after.shape:
        return True
    if before.size == 0 and after.size == 0:
        return False
    delta = np.sqrt(np.mean((ensure_stereo(after) - ensure_stereo(before)) ** 2) + 1e-12)
    base = np.sqrt(np.mean(ensure_stereo(before) ** 2) + 1e-12)
    return bool(delta / max(base, 1e-12) > 1e-5)


def _rms_db(audio: np.ndarray) -> float:
    stereo = ensure_stereo(audio)
    rms = float(np.sqrt(np.mean(stereo * stereo) + 1e-12)) if stereo.size else 0.0
    return amp_to_db(rms)
