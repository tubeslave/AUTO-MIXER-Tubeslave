"""Virtual render -> objective -> decision guard -> acknowledgement.

Enabled explicitly through autofoh.shadow. Live permissions and emergency
handling remain in AutoFOHSafetyController/AutoSoundcheckEngine.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from autofoh_safety import ChannelEQMove, CompressorAdjust, HighPassAdjust, TypedCorrectionAction
from decision_guard import DecisionGuard, GuardConfig, ReviewDecision
from shadow_renderer import Compressor, EQBand, RenderChannel, RenderResult, ShadowRenderer, measure
from target_corridor import FEATURE_VERSION, TargetCorridor


def _flag(value: Any) -> bool:
    if value in (True, 1, "1", "ON", "on", "true"):
        return True
    if value in (False, 0, "0", "OFF", "off", "false"):
        return False
    raise ValueError("Unknown processing enable state")


def live_states(channels: Sequence[Any], config: Mapping[str, Any]) -> dict[int, RenderChannel]:
    """Conservative direct-sum adapter; incomplete console readback fails closed."""
    if config.get("tap") != "post_input_pre_processing":
        raise ValueError("An explicit post-input/pre-processing tap is required")
    if config.get("aligned_complete_direct_sum") is not True:
        raise ValueError("Complete aligned capture and direct routing are unverified")
    if config.get("unmodelled_processing_bypassed") is not True:
        raise ValueError("Inserts, shelves, bus/master DSP and FX must be verified bypassed")
    result = {}
    for channel in channels:
        raw = channel.raw_settings
        if channel.channel_id in result:
            raise ValueError("Duplicate capture channel")
        if _flag(raw["gate_enabled"]):
            raise ValueError("Gate model is not supported")
        if raw["active_bus_sends"] or raw["dca_assignments"]:
            raise ValueError("Bus/DCA routing is not modelled")
        send = raw["main_send"]
        if not _flag(send["on"]) or _flag(send["pre"]) or float(send["level_db"]) != 0:
            raise ValueError("Non-unity/non-post-fader main routing is not modelled")
        if len(raw["eq_bands"]) != 4:
            raise ValueError("All four EQ bands must have readback")
        bands = {i: EQBand(*map(float, band)) for i, band in enumerate(raw["eq_bands"], 1)}
        hpf_on = _flag(raw["hpf_enabled"])
        comp = None
        if _flag(raw["compressor_enabled"]):
            if config.get("compressor_model") != "reference_linked_peak":
                raise ValueError("Compressor model has not been specified")
            if float(raw["compressor_mix_pct"]) != 100:
                raise ValueError("Parallel compressor is not modelled")
            comp = Compressor(*(float(raw["compressor_" + name]) for name in
                                ("threshold_db", "ratio", "attack_ms", "release_ms", "makeup_db")))
        hpf_order = int(config["hpf_order"]) if hpf_on else int(config.get("hpf_order", 2))
        result[channel.channel_id] = RenderChannel(
            channel.channel_id, float(raw["fader_db"]), _flag(raw["eq_on"]), bands,
            float(raw["hpf_freq"]) if hpf_on else None, hpf_order, comp,
            _flag(raw["muted"]), channel.role, float(raw["pan"]) / 100,
        )
    return result


class ShadowPipeline:
    def __init__(self, config: Mapping[str, Any] | None = None,
                 corridor: TargetCorridor | None = None):
        self.config = dict(config or {})
        self.mode = self.config.get("mode", "off")
        if self.mode not in ("off", "shadow", "guarded"):
            raise ValueError("Shadow mode must be off, shadow or guarded")
        self.context = str(self.config.get("context", "unclassified"))
        self.guard = DecisionGuard(GuardConfig(**self.config.get("guard", {})))
        self.corridor = corridor
        if self.corridor is None and self.config.get("corridor_path"):
            self.corridor = TargetCorridor.load(self.config["corridor_path"])
        if self.corridor is not None and self.corridor.context != self.context:
            raise ValueError("Configured context does not match approved corridor")
        self.max_age_sec = float(self.config.get("max_render_age_sec", 2.0))
        self.model_uncertainty = float(self.config.get("model_uncertainty", 0.05))
        self.ceiling = float(self.config.get("true_peak_ceiling_dbtp", -1.25))
        if (not np.isfinite([self.max_age_sec, self.model_uncertainty, self.ceiling]).all()
                or self.max_age_sec <= 0 or self.model_uncertainty < 0 or self.ceiling > -1.0):
            raise ValueError("Unsafe shadow limits")
        self._audit_path = Path(self.config["journal_path"]) if self.config.get("journal_path") else None

    def _journal(self, report: dict) -> None:
        if self._audit_path:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            # Audit failure propagates BEFORE writes and therefore fails closed.
            with self._audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report, allow_nan=False, sort_keys=True) + "\n")

    def evaluate_live(self, channels: Sequence[Any], action: TypedCorrectionAction,
                      sample_rate: int) -> ReviewDecision:
        try:
            states = live_states(channels, self.config)
            buffers = {ch.channel_id: ch.audio for ch in channels}
            if isinstance(action, HighPassAdjust) and "hpf_order" not in self.config:
                raise ValueError("HPF order must be explicit before enabling/changing HPF")
            if isinstance(action, CompressorAdjust):
                if self.config.get("compressor_model") != "reference_linked_peak":
                    raise ValueError("Compressor model must be explicit before changing dynamics")
            protected = self.config.get("protected_eq_bands", {})
            if isinstance(action, ChannelEQMove):
                band = states[action.channel_id].eq_bands[action.band]
                if (action.band in protected.get(str(action.channel_id), [])
                        or (band.q >= 8 and band.gain_db <= -3)):
                    raise ValueError("Protected/manual/feedback EQ band")
            return self.evaluate(buffers, states, action, sample_rate)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            result = ReviewDecision(False, "invalid_live_model", {"detail": str(exc), "mode": self.mode})
            self._journal({"event": "shadow_rejected", **result.report, "reason": result.reason})
            return result

    def evaluate(self, buffers: Mapping[int, np.ndarray], states: Mapping[int, RenderChannel],
                 action: TypedCorrectionAction, sample_rate: int) -> ReviewDecision:
        verdict = self._evaluate(buffers, states, action, sample_rate)
        self._journal({"event": "shadow_review", "allowed": verdict.allowed,
                       "reason": verdict.reason, "mode": self.mode, **verdict.report})
        return verdict

    def _evaluate(self, buffers, states, action, sample_rate) -> ReviewDecision:
        start = time.monotonic()
        if self.mode == "off":
            return ReviewDecision(False, "shadow_pipeline_disabled")
        if self.corridor is None:
            return ReviewDecision(False, "no_approved_target_corridor")
        if (self.mode == "guarded" and (self.config.get("model_validated") is not True
                or self.corridor.source != "approved_mixes" or self.corridor.sample_count < 5)):
            return ReviewDecision(False, "model_or_target_not_validated_for_writes")
        try:
            buffers = {key: np.array(value, dtype=np.float64, copy=True)
                       for key, value in buffers.items()}
            states = deepcopy(dict(states))
            renderer = ShadowRenderer(sample_rate)
            # Independent baseline/candidate runs start with identical filter histories.
            baseline = renderer.render(buffers, states)
            proposed, changes = renderer.propose(states, action)
            candidate = renderer.render(buffers, proposed)
            before = self.corridor.loss(baseline.features, context=self.context)
            after = self.corridor.loss(candidate.features, context=self.context)
            if baseline.features["lufs"] < -70:
                return ReviewDecision(False, "insufficient_program_level")
            if (candidate.true_peak_dbtp > self.ceiling
                    or max(candidate.channel_true_peaks.values()) > self.ceiling):
                return ReviewDecision(False, "predicted_peak_ceiling", {
                    "mix_dbtp": candidate.true_peak_dbtp,
                    "channel_dbtp": max(candidate.channel_true_peaks.values())})
            if candidate.features["crest_db"] < baseline.features["crest_db"] - 1.5:
                return ReviewDecision(False, "excessive_crest_loss")
            if abs(candidate.features["lufs"] - baseline.features["lufs"]) > 1.0:
                return ReviewDecision(False, "excessive_loudness_change")
            uncertainty = self.model_uncertainty + self._window_dispersion(
                baseline, candidate, sample_rate)
            frame_hash = hashlib.sha256()
            for key in sorted(buffers):
                frame_hash.update(str(key).encode("ascii"))
                frame_hash.update(np.ascontiguousarray(buffers[key]).tobytes())
            if time.monotonic() - start > self.max_age_sec:
                return ReviewDecision(False, "render_expired")
            key = (self.context, action.action_type, action.channel_id, getattr(action, "band", None))
            signature = json.dumps({name: getattr(action, name) for name in ("freq_hz", "q")
                                    if hasattr(action, name)}, sort_keys=True)
            verdict = self.guard.review(key, changes=changes, before=before, after=after,
                                        uncertainty=uncertainty, frame_id=frame_hash.hexdigest(),
                                        now=time.monotonic(), signature=signature)
            verdict.report.update(mode=self.mode, feature_version=FEATURE_VERSION,
                                  before_features=baseline.features, after_features=candidate.features,
                                  true_peak_dbtp=candidate.true_peak_dbtp,
                                  elapsed_sec=time.monotonic() - start,
                                  rendered_action=asdict(action),
                                  peak_method="4x_polyphase_estimate")
            return verdict
        except (KeyError, TypeError, ValueError, FloatingPointError, OverflowError) as exc:
            return ReviewDecision(False, "invalid_render_evidence", {"detail": str(exc)})

    def _window_dispersion(self, before: RenderResult, after: RenderResult, sr: int) -> float:
        """Conservative temporal-variation penalty, NOT a calibrated confidence interval."""
        n = len(before.audio)
        chunks = min(4, int(n // (sr * 0.5)))
        if chunks < 2 or "lead_mask_db" in self.corridor.bands:
            # Masking objective lacks independent window estimates in this version.
            return 0.1
        improvements = []
        for indices in np.array_split(np.arange(n), chunks):
            lo, hi = int(indices[0]), int(indices[-1]) + 1
            a = measure(before.audio[lo:hi], sr)
            b = measure(after.audio[lo:hi], sr)
            improvements.append(self.corridor.loss(a, context=self.context)
                                - self.corridor.loss(b, context=self.context))
        return float(np.std(improvements, ddof=1))
