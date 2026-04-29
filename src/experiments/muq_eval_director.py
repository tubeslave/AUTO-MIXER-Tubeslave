"""Offline MuQ-Eval director experiment.

This module is intentionally isolated from live FOH control. It never imports a
mixer client and never sends OSC. The experiment renders candidate mix states to
offline WAV files, scores them with MuQ-Eval, and advances a beam search using
MuQ as the primary reward plus engineering safety terms.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Optional

import numpy as np
import soundfile as sf
import yaml

try:  # pragma: no cover - optional plotting dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import pyloudnorm as pyln
except Exception:  # pragma: no cover - tests can use RMS fallback
    pyln = None

try:
    from scipy.signal import butter, lfilter
except Exception:  # pragma: no cover
    butter = None
    lfilter = None


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from evaluation import MuQEvalService, MuQEvalResult
except Exception:  # pragma: no cover - constructor reports this as unavailable
    MuQEvalService = None  # type: ignore[assignment]
    MuQEvalResult = Any  # type: ignore[misc,assignment]

from output_paths import ai_logs_path, ai_mixing_path, ensure_ai_output_dirs


LOGGER = logging.getLogger(__name__)
MODE_NAME = "MUQ_EVAL_DIRECTOR_OFFLINE_TEST"
EQ_CENTERS_HZ = (60.0, 80.0, 120.0, 250.0, 500.0, 800.0, 1200.0, 2500.0, 3500.0, 6000.0, 8500.0, 12000.0)
CENTER_LOCKED = {"lead_vocal", "kick", "snare", "bass", "bass_guitar", "bass_di", "bass_mic"}
DEFAULT_MUQ_DIRECTOR_LOG_ROOT = ai_logs_path("muq_director")
DEFAULT_MUQ_DIRECTOR_AUDIO_ROOT = ai_mixing_path("muq_director")


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _amp_to_db(value: float) -> float:
    return float(20.0 * math.log10(max(float(value), 1e-12)))


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, float(value))))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    return value


@dataclass
class MuqDirectorConfig:
    """Config for the offline-only MuQ director experiment."""

    enabled: bool = False
    offline_only: bool = True
    send_osc: bool = False
    shadow_mode: bool = False
    window_sec: float = 10.0
    hop_sec: float = 5.0
    max_steps: int = 80
    candidates_per_step: int = 12
    beam_width: int = 3
    min_delta_accept: float = 0.005
    stop_if_no_improvement_steps: int = 8
    rollback_on_drop: bool = True
    save_every_candidate: bool = True
    save_best_mix_each_step: bool = True
    target_lufs: float = -14.0
    true_peak_ceiling_dbfs: float = -1.0
    max_gain_change_db: float = 1.0
    max_eq_gain_change_db: float = 1.5
    max_pan_change: float = 0.08
    max_compressor_threshold_change_db: float = 1.5
    max_ratio_change: float = 0.4
    allow_master_limiter: bool = True
    allow_aggressive_mode: bool = True
    log_jsonl: bool = True
    mode: str = "muq_safe"
    sample_rate: int = 48000
    require_available_muq: bool = True
    max_eval_windows: int = 4
    device: str = "auto"
    muq_eval_root: str = ""
    local_files_only: bool = True
    output_root: str = str(DEFAULT_MUQ_DIRECTOR_LOG_ROOT)
    audio_output_root: str = str(DEFAULT_MUQ_DIRECTOR_AUDIO_ROOT)
    genre: str = ""
    existing_automixer_path: str = ""

    @classmethod
    def from_mapping(cls, payload: Optional[dict[str, Any]] = None) -> "MuqDirectorConfig":
        data = dict(payload or {})
        if isinstance(data.get("muq_director"), dict):
            data = dict(data["muq_director"])
        known = set(cls.__dataclass_fields__)
        values = {key: data[key] for key in known if key in data}
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MuqDirectorConfig":
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_mapping(loaded)


@dataclass
class ChannelState:
    """Offline channel-strip state used by the director."""

    channel_id: int
    name: str
    path: str
    instrument: str
    gain_db: float = 0.0
    pan: float = 0.0
    hpf_hz: float = 0.0
    lpf_hz: float = 0.0
    eq_bands: list[dict[str, float]] = field(default_factory=list)
    comp_threshold_db: float = -24.0
    comp_ratio: float = 1.0
    comp_attack_ms: float = 20.0
    comp_release_ms: float = 160.0
    comp_enabled: bool = False
    change_count: int = 0


@dataclass
class DirectorAction:
    """One candidate mix action."""

    action_type: str
    channel_id: Optional[int] = None
    instrument: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class DirectorState:
    """Beam-search state for one offline mix version."""

    state_id: str
    step: int
    channels: dict[int, ChannelState]
    parent_state_id: str = ""
    last_action: Optional[DirectorAction] = None
    action_sequence: list[dict[str, Any]] = field(default_factory=list)
    muq_score: float = 0.0
    final_score: float = 0.0
    render_path: str = ""
    per_window_scores: list[float] = field(default_factory=list)
    no_improvement_steps: int = 0


@dataclass
class CandidateResult:
    """Evaluation result for a rendered action candidate."""

    step: int
    parent_state_id: str
    candidate_id: str
    action: DirectorAction
    state: DirectorState
    render_path: str
    muq_score_before: float
    muq_score_after: float
    delta_muq: float
    loudness_before: Optional[float]
    loudness_after: float
    peak_before: Optional[float]
    peak_after: float
    phase_score: float
    loudness_score: float
    peak_headroom_score: float
    anti_overprocessing_score: float
    final_score: float
    accepted: bool
    rejection_reason: str
    audio_window_scores: list[float]
    model_status: str = "unknown"
    audio_window_model_statuses: list[str] = field(default_factory=list)
    parameters_before: dict[str, Any] = field(default_factory=dict)
    parameters_after: dict[str, Any] = field(default_factory=dict)

    def to_log_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["action"] = _json_safe(self.action)
        record["state"] = self.state.state_id
        return _json_safe(record)


class MuqEvalDirectorOfflineTest:
    """Beam-search offline mixer with MuQ-Eval as the primary reward engine."""

    def __init__(
        self,
        config: MuqDirectorConfig | dict[str, Any] | None = None,
        *,
        muq_service: Any | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config if isinstance(config, MuqDirectorConfig) else MuqDirectorConfig.from_mapping(config)
        if not self.config.offline_only:
            raise ValueError("MUQ_EVAL_DIRECTOR_OFFLINE_TEST is offline-only")
        if self.config.send_osc:
            raise ValueError("MUQ_EVAL_DIRECTOR_OFFLINE_TEST forbids send_osc=true")
        if self.config.shadow_mode:
            raise ValueError("MUQ_EVAL_DIRECTOR_OFFLINE_TEST applies offline renders; shadow_mode must be false")

        self.session_id = f"muq_director_{int(time.time())}"
        if output_dir is None and (
            Path(self.config.output_root).expanduser() == DEFAULT_MUQ_DIRECTOR_LOG_ROOT
            or Path(self.config.audio_output_root).expanduser() == DEFAULT_MUQ_DIRECTOR_AUDIO_ROOT
        ):
            ensure_ai_output_dirs()
        self.output_dir = Path(output_dir).expanduser() if output_dir is not None else Path(self.config.output_root).expanduser() / self.session_id
        if output_dir is not None or (
            Path(self.config.output_root).expanduser() != DEFAULT_MUQ_DIRECTOR_LOG_ROOT
            and Path(self.config.audio_output_root).expanduser() == DEFAULT_MUQ_DIRECTOR_AUDIO_ROOT
        ):
            self.audio_dir = self.output_dir
        else:
            self.audio_dir = Path(self.config.audio_output_root).expanduser() / self.session_id
        self.candidate_dir = self.audio_dir / "candidate_renders"
        self.best_states_dir = self.output_dir / "best_states"
        self.steps_path = self.output_dir / "steps.jsonl"
        self.report_path = self.output_dir / "report.json"
        self.best_mix_path = self.audio_dir / "best_mix.wav"
        self.channel_audio: dict[int, np.ndarray] = {}
        self.sample_rate = int(self.config.sample_rate)
        self.initial_state: DirectorState | None = None
        self.best_state: DirectorState | None = None
        self.beams: list[DirectorState] = []
        self.accepted_actions: list[dict[str, Any]] = []
        self.rejected_actions: list[dict[str, Any]] = []
        self.score_curve: list[dict[str, Any]] = []
        self.warnings: list[str] = []
        self.stopping_reason = ""
        self._state_archive: dict[str, DirectorState] = {}

        self.muq_service = muq_service

    def load_session(self, multitrack_dir: str | Path) -> DirectorState:
        """Load a multitrack directory into an initial offline mix state."""

        input_dir = Path(multitrack_dir).expanduser().resolve()
        wavs = sorted(path for path in input_dir.glob("*.wav") if path.is_file())
        if not wavs:
            raise FileNotFoundError(f"No WAV files found in {input_dir}")

        channels: dict[int, ChannelState] = {}
        target_len = 0
        for index, path in enumerate(wavs, start=1):
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if int(sr) != self.sample_rate:
                audio = self._resample_audio(audio, int(sr), self.sample_rate)
            mono = self._as_mono(audio)
            target_len = max(target_len, len(mono))
            instrument = self._classify_instrument(path.stem)
            channels[index] = ChannelState(
                channel_id=index,
                name=path.stem,
                path=str(path),
                instrument=instrument,
                pan=self._initial_pan(path.stem, instrument),
                hpf_hz=self._initial_hpf(instrument),
                comp_threshold_db=self._initial_comp_threshold(instrument),
                comp_ratio=self._initial_comp_ratio(instrument),
                comp_enabled=instrument in {"lead_vocal", "backing_vocal", "kick", "snare", "bass", "bass_guitar", "bass_di"},
            )
            self.channel_audio[index] = mono

        for channel, audio in list(self.channel_audio.items()):
            if len(audio) < target_len:
                padded = np.zeros(target_len, dtype=np.float32)
                padded[: len(audio)] = audio
                self.channel_audio[channel] = padded

        state = DirectorState(
            state_id="state_0000",
            step=0,
            channels=channels,
        )
        self.initial_state = state
        self._state_archive[state.state_id] = copy.deepcopy(state)
        return state

    def render_candidate(self, state: DirectorState, action: DirectorAction | None) -> str:
        """Render an offline candidate WAV for `state` plus optional `action`."""

        render_state = self.apply_best_action(state, action) if action is not None else copy.deepcopy(state)
        candidate_id = render_state.state_id
        render_path = self.candidate_dir / f"{candidate_id}.wav"
        self._ensure_output_dirs()
        audio = self._render_state_audio(render_state)
        sf.write(str(render_path), audio, self.sample_rate, subtype="PCM_24")
        render_state.render_path = str(render_path)
        self._state_archive[render_state.state_id] = copy.deepcopy(render_state)
        return str(render_path)

    def evaluate_candidate(
        self,
        audio_path: str | Path,
        *,
        action: DirectorAction | None = None,
        state: DirectorState | None = None,
    ) -> dict[str, Any]:
        """Evaluate a rendered candidate with MuQ-Eval and engineering metrics."""

        self._ensure_muq_service()
        if not self._muq_available():
            raise RuntimeError("MuQ-Eval is unavailable")
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        audio = self._normalize_audio_shape(audio)
        window_scores: list[float] = []
        window_model_statuses: list[str] = []
        for start, end in self._evaluation_windows(audio, int(sr)):
            result = self.muq_service.evaluate(audio[start:end], int(sr))
            model_status = str(getattr(result, "model_status", "unknown") or "unknown")
            if self.config.require_available_muq and model_status != "available":
                raise RuntimeError("MuQ-Eval returned unavailable status")
            window_scores.append(float(result.quality_score))
            window_model_statuses.append(model_status)

        if not window_scores:
            raise RuntimeError("No audio windows available for MuQ-Eval")

        metrics = self._engineering_metrics(audio, int(sr), state=state, action=action)
        muq_score = float(np.mean(np.asarray(window_scores, dtype=np.float64)))
        final_score = self._final_score(
            muq_score=muq_score,
            loudness_score=metrics["loudness_score"],
            peak_headroom_score=metrics["peak_headroom_score"],
            phase_mono_score=metrics["phase_score"],
            anti_overprocessing_score=metrics["anti_overprocessing_score"],
        )
        return {
            "muq_score": muq_score,
            "audio_window_scores": window_scores,
            "audio_window_model_statuses": window_model_statuses,
            "model_status": self._aggregate_model_status(window_model_statuses),
            "final_score": final_score,
            **metrics,
        }

    def propose_actions(self, state: DirectorState) -> list[DirectorAction]:
        """Generate bounded candidate mix actions for a beam state."""

        actions: list[DirectorAction] = []
        priority = {
            "lead_vocal": 0,
            "kick": 1,
            "snare": 2,
            "bass": 3,
            "bass_guitar": 3,
            "electric_guitar": 4,
            "guitar": 4,
            "overhead": 5,
            "playback": 6,
            "backing_vocal": 7,
        }
        channels = sorted(
            state.channels.values(),
            key=lambda item: (priority.get(item.instrument, 20), item.channel_id),
        )

        for channel in channels:
            for delta in (-1.0, 1.0, -0.5, 0.5):
                if abs(delta) <= self.config.max_gain_change_db:
                    actions.append(
                        DirectorAction(
                            action_type="gain",
                            channel_id=channel.channel_id,
                            instrument=channel.instrument,
                            parameters={"delta_db": delta},
                            reason="bounded_channel_gain_probe",
                        )
                    )

            if channel.instrument not in CENTER_LOCKED:
                for delta_pan in (-self.config.max_pan_change, self.config.max_pan_change):
                    actions.append(
                        DirectorAction(
                            action_type="pan",
                            channel_id=channel.channel_id,
                            instrument=channel.instrument,
                            parameters={"delta": delta_pan},
                            reason="small_stereo_position_probe",
                        )
                    )

            for freq, gain in self._eq_probes_for(channel.instrument):
                actions.append(
                    DirectorAction(
                        action_type="eq",
                        channel_id=channel.channel_id,
                        instrument=channel.instrument,
                        parameters={
                            "freq_hz": freq,
                            "gain_db": _clamp(gain, -self.config.max_eq_gain_change_db, self.config.max_eq_gain_change_db),
                            "q": 1.2 if freq < 1000.0 else 1.6,
                        },
                        reason="bounded_eq_probe",
                    )
                )

            if channel.instrument in {"lead_vocal", "backing_vocal", "kick", "snare", "bass", "bass_guitar", "bass_di"}:
                for delta in (-self.config.max_compressor_threshold_change_db, self.config.max_compressor_threshold_change_db):
                    actions.append(
                        DirectorAction(
                            action_type="compressor_threshold",
                            channel_id=channel.channel_id,
                            instrument=channel.instrument,
                            parameters={"delta_db": delta},
                            reason="compression_threshold_probe",
                        )
                    )
                actions.append(
                    DirectorAction(
                        action_type="compressor_ratio",
                        channel_id=channel.channel_id,
                        instrument=channel.instrument,
                        parameters={"delta": self.config.max_ratio_change},
                        reason="compression_ratio_probe",
                    )
                )

        actions.extend(
            [
                DirectorAction(action_type="master_loudness", parameters={"target_lufs": self.config.target_lufs}, reason="target_lufs_probe"),
                DirectorAction(action_type="bus_glue", parameters={"enabled": True}, reason="gentle_bus_compression_probe"),
            ]
        )

        unique: list[DirectorAction] = []
        seen: set[str] = set()
        for action in actions:
            key = json.dumps(_json_safe(action), sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(action)

        if not unique:
            return []
        count = max(1, int(self.config.candidates_per_step))
        start = (state.step * 7 + sum(ord(ch) for ch in state.state_id)) % len(unique)
        rotated = unique[start:] + unique[:start]
        return rotated[:count]

    def select_best_candidate(self, candidates: Iterable[CandidateResult]) -> CandidateResult | None:
        """Return the highest scoring accepted candidate, preserving beam safety."""

        accepted = [candidate for candidate in candidates if candidate.accepted]
        if not accepted:
            return None
        return max(accepted, key=lambda item: (item.final_score, item.muq_score_after, item.delta_muq))

    def apply_best_action(self, state: DirectorState, best_action: DirectorAction | None) -> DirectorState:
        """Return a copied state with `best_action` applied offline."""

        new_state = copy.deepcopy(state)
        new_state.parent_state_id = state.state_id
        new_state.step = state.step + (1 if best_action is not None else 0)
        new_state.state_id = self._next_state_id(new_state.step)
        new_state.last_action = copy.deepcopy(best_action)
        if best_action is None:
            return new_state

        channel = new_state.channels.get(int(best_action.channel_id or 0))
        params = best_action.parameters
        if best_action.action_type == "gain" and channel is not None:
            channel.gain_db = _clamp(channel.gain_db + float(params.get("delta_db", 0.0)), -18.0, 12.0)
            channel.change_count += 1
        elif best_action.action_type == "pan" and channel is not None:
            if channel.instrument not in CENTER_LOCKED:
                channel.pan = _clamp(channel.pan + float(params.get("delta", 0.0)), -1.0, 1.0)
                channel.change_count += 1
        elif best_action.action_type == "eq" and channel is not None:
            channel.eq_bands.append(
                {
                    "freq_hz": float(params.get("freq_hz", 1000.0)),
                    "gain_db": _clamp(float(params.get("gain_db", 0.0)), -self.config.max_eq_gain_change_db, self.config.max_eq_gain_change_db),
                    "q": max(0.2, float(params.get("q", 1.0))),
                }
            )
            channel.change_count += 1
        elif best_action.action_type == "low_cut" and channel is not None:
            channel.hpf_hz = max(channel.hpf_hz, float(params.get("freq_hz", 80.0)))
            channel.change_count += 1
        elif best_action.action_type == "high_cut" and channel is not None:
            channel.lpf_hz = float(params.get("freq_hz", 12000.0))
            channel.change_count += 1
        elif best_action.action_type == "compressor_threshold" and channel is not None:
            channel.comp_enabled = True
            channel.comp_threshold_db = _clamp(
                channel.comp_threshold_db + float(params.get("delta_db", 0.0)),
                -48.0,
                -6.0,
            )
            channel.change_count += 1
        elif best_action.action_type == "compressor_ratio" and channel is not None:
            channel.comp_enabled = True
            channel.comp_ratio = _clamp(channel.comp_ratio + float(params.get("delta", 0.0)), 1.0, 8.0)
            channel.change_count += 1
        elif best_action.action_type == "compressor_preset" and channel is not None:
            self._apply_compressor_preset(channel, str(params.get("preset", "")))
            channel.change_count += 1

        new_state.action_sequence.append(_json_safe(best_action))
        return new_state

    def rollback(self, state: DirectorState) -> DirectorState | None:
        """Return the archived parent state for `state`, if one exists."""

        if not state.parent_state_id:
            return None
        parent = self._state_archive.get(state.parent_state_id)
        return copy.deepcopy(parent) if parent is not None else None

    def run(self, multitrack_dir: str | Path | None = None, *, dry_run: bool = False) -> dict[str, Any]:
        """Run the MuQ director beam search and write report artifacts."""

        self._ensure_output_dirs()
        if not self.config.enabled:
            self.stopping_reason = "disabled"
            return self.write_report(status="disabled")

        if multitrack_dir is not None:
            self.load_session(multitrack_dir)
        if self.initial_state is None:
            raise RuntimeError("load_session() must be called before run()")

        if dry_run:
            proposed = [asdict(action) for action in self.propose_actions(self.initial_state)]
            self.warnings.append("dry_run: no candidates rendered or evaluated")
            return self.write_report(status="dry_run", extra={"proposed_actions": proposed})

        self._ensure_muq_service()
        if not self._muq_available():
            self.stopping_reason = "muq_eval_unavailable"
            self.warnings.append("MuQ-Eval is unavailable; director search was not started")
            return self.write_report(status="failed")

        initial_path = self.render_candidate(self.initial_state, None)
        initial_metrics = self.evaluate_candidate(initial_path, state=self.initial_state)
        self.initial_state.render_path = initial_path
        self.initial_state.muq_score = float(initial_metrics["muq_score"])
        self.initial_state.final_score = float(initial_metrics["final_score"])
        self.initial_state.per_window_scores = list(initial_metrics["audio_window_scores"])
        self.initial_state.step = 0
        self.initial_state.state_id = "state_0000"
        self.initial_state.parent_state_id = ""
        self._state_archive[self.initial_state.state_id] = copy.deepcopy(self.initial_state)
        self.best_state = copy.deepcopy(self.initial_state)
        self.beams = [copy.deepcopy(self.initial_state)]
        self.score_curve.append(
            {
                "step": 0,
                "muq_score": self.initial_state.muq_score,
                "final_score": self.initial_state.final_score,
                "model_status": str(initial_metrics.get("model_status", self._model_status())),
            }
        )
        self._copy_best_mix(self.initial_state)

        no_improve = 0
        for step in range(1, max(1, int(self.config.max_steps)) + 1):
            step_candidates: list[CandidateResult] = []
            for parent in self.beams:
                for action in self.propose_actions(parent):
                    candidate = self._evaluate_action_candidate(parent, action, step)
                    step_candidates.append(candidate)
                    self._log_step(candidate)

            best_candidate = self.select_best_candidate(step_candidates)
            if best_candidate is None:
                self.stopping_reason = "all_candidates_rejected_or_worse"
                break

            accepted = sorted(
                [candidate for candidate in step_candidates if candidate.accepted],
                key=lambda item: (item.final_score, item.muq_score_after),
                reverse=True,
            )
            self.beams = [copy.deepcopy(candidate.state) for candidate in accepted[: max(1, int(self.config.beam_width))]]
            for beam in self.beams:
                self._state_archive[beam.state_id] = copy.deepcopy(beam)

            if self.best_state is None or best_candidate.final_score > self.best_state.final_score:
                improvement = best_candidate.final_score - (self.best_state.final_score if self.best_state else -1.0)
                self.best_state = copy.deepcopy(best_candidate.state)
                self._copy_best_mix(self.best_state)
                if self.config.save_best_mix_each_step:
                    self._write_state_json(self.best_state, self.best_states_dir / f"step_{step:03d}_{self.best_state.state_id}.json")
                no_improve = 0 if improvement >= self.config.min_delta_accept else no_improve + 1
            else:
                no_improve += 1

            self.score_curve.append(
                {
                    "step": float(step),
                    "muq_score": float(self.best_state.muq_score if self.best_state else 0.0),
                    "final_score": float(self.best_state.final_score if self.best_state else 0.0),
                }
            )
            if no_improve >= int(self.config.stop_if_no_improvement_steps):
                self.stopping_reason = "no_improvement"
                break
        else:
            self.stopping_reason = "max_steps"

        self._write_plots()
        return self.write_report(status="completed")

    def write_report(self, *, status: str = "completed", extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Write `report.json` and return the report payload."""

        best = self.best_state
        initial = self.initial_state
        comparison = self._compare_existing_automixer(best) if best is not None else {}
        final_metrics = self._metrics_for_path(best.render_path) if best and best.render_path else {}
        report = {
            "mode_name": MODE_NAME,
            "status": status,
            "session_id": self.session_id,
            "model_status": self._model_status(),
            "config": _json_safe(self.config),
            "best_muq_score": best.muq_score if best else None,
            "initial_muq_score": initial.muq_score if initial else None,
            "total_delta": (best.muq_score - initial.muq_score) if best and initial else None,
            "number_of_steps": int(best.step) if best else 0,
            "accepted_actions": self.accepted_actions,
            "rejected_actions": self.rejected_actions,
            "best_action_sequence": best.action_sequence if best else [],
            "score_curve": self.score_curve,
            "per_window_scores": best.per_window_scores if best else [],
            "final_lufs": final_metrics.get("lufs"),
            "final_peak_dbfs": final_metrics.get("peak_dbfs"),
            "phase_score": final_metrics.get("phase_score"),
            "warnings": self.warnings,
            "stopping_reason": self.stopping_reason,
            "comparison_with_existing_automixer_if_available": comparison,
            "artifact_paths": {
                "report": str(self.report_path),
                "steps_jsonl": str(self.steps_path),
                "audio_dir": str(self.audio_dir),
                "best_mix": str(self.best_mix_path),
                "best_states": str(self.best_states_dir),
                "candidate_renders": str(self.candidate_dir),
            },
        }
        if extra:
            report.update(extra)
        self._ensure_output_dirs()
        self.report_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def _evaluate_action_candidate(self, parent: DirectorState, action: DirectorAction, step: int) -> CandidateResult:
        candidate_state = self.apply_best_action(parent, action)
        candidate_id = f"step_{step:03d}_{candidate_state.state_id}"
        candidate_state.state_id = candidate_id
        render_path = self.candidate_dir / f"{candidate_id}.wav"
        audio = self._render_state_audio(candidate_state)
        if self.config.save_every_candidate:
            sf.write(str(render_path), audio, self.sample_rate, subtype="PCM_24")
        else:
            sf.write(str(render_path), audio, self.sample_rate, subtype="PCM_24")
        candidate_state.render_path = str(render_path)

        before = self._parameters_for_action(parent, action)
        after = self._parameters_for_action(candidate_state, action)
        try:
            metrics = self.evaluate_candidate(render_path, action=action, state=candidate_state)
            delta_muq = float(metrics["muq_score"] - parent.muq_score)
            rejection = self._candidate_rejection_reason(metrics, delta_muq, action, parent)
            accepted = rejection == ""
        except Exception as exc:
            metrics = {
                "muq_score": 0.0,
                "audio_window_scores": [],
                "final_score": 0.0,
                "lufs": math.nan,
                "peak_dbfs": math.inf,
                "phase_score": 0.0,
                "loudness_score": 0.0,
                "peak_headroom_score": 0.0,
                "anti_overprocessing_score": 0.0,
                "model_status": self._model_status(),
                "audio_window_model_statuses": [],
            }
            delta_muq = -parent.muq_score
            accepted = False
            rejection = f"evaluation_failed:{exc}"

        candidate_state.muq_score = float(metrics["muq_score"])
        candidate_state.final_score = float(metrics["final_score"])
        candidate_state.per_window_scores = list(metrics["audio_window_scores"])

        result = CandidateResult(
            step=step,
            parent_state_id=parent.state_id,
            candidate_id=candidate_id,
            action=action,
            state=candidate_state,
            render_path=str(render_path),
            muq_score_before=parent.muq_score,
            muq_score_after=float(metrics["muq_score"]),
            delta_muq=delta_muq,
            loudness_before=self._metrics_for_path(parent.render_path).get("lufs") if parent.render_path else None,
            loudness_after=float(metrics["lufs"]),
            peak_before=self._metrics_for_path(parent.render_path).get("peak_dbfs") if parent.render_path else None,
            peak_after=float(metrics["peak_dbfs"]),
            phase_score=float(metrics["phase_score"]),
            loudness_score=float(metrics["loudness_score"]),
            peak_headroom_score=float(metrics["peak_headroom_score"]),
            anti_overprocessing_score=float(metrics["anti_overprocessing_score"]),
            final_score=float(metrics["final_score"]),
            accepted=accepted,
            rejection_reason=rejection,
            audio_window_scores=list(metrics["audio_window_scores"]),
            model_status=str(metrics.get("model_status", self._model_status())),
            audio_window_model_statuses=list(metrics.get("audio_window_model_statuses", [])),
            parameters_before=before,
            parameters_after=after,
        )
        if accepted:
            self.accepted_actions.append(result.to_log_record())
        else:
            self.rejected_actions.append(result.to_log_record())
        return result

    def _candidate_rejection_reason(
        self,
        metrics: dict[str, Any],
        delta_muq: float,
        action: DirectorAction,
        parent: DirectorState,
    ) -> str:
        if float(metrics.get("clipping_ratio", 0.0)) > 0.0:
            return "clipping"
        if float(metrics.get("peak_dbfs", 0.0)) > float(self.config.true_peak_ceiling_dbfs) + 0.05:
            return "peak_ceiling"
        if float(metrics.get("crest_factor_db", 12.0)) < 4.5:
            return "overcompressed_low_crest"
        if delta_muq < float(self.config.min_delta_accept):
            return "below_min_delta_accept"
        if action.channel_id is not None:
            channel = parent.channels.get(int(action.channel_id))
            if channel is not None and channel.change_count >= 8:
                return "too_many_sequential_channel_changes"
        return ""

    def _render_state_audio(self, state: DirectorState) -> np.ndarray:
        target_len = max((len(audio) for audio in self.channel_audio.values()), default=0)
        mix = np.zeros((target_len, 2), dtype=np.float32)
        for channel_id, channel in state.channels.items():
            source = self.channel_audio[channel_id]
            rendered = self._process_channel(source, channel)
            mix += self._pan_mono(rendered, channel.pan)

        mix = self._apply_bus_glue(mix)
        mix = self._normalize_loudness(mix)
        if self.config.allow_master_limiter:
            mix = self._limit_peak(mix, self.config.true_peak_ceiling_dbfs)
        return np.nan_to_num(mix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def _process_channel(self, audio: np.ndarray, channel: ChannelState) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).copy()
        if channel.hpf_hz > 0.0:
            x = self._filter(x, channel.hpf_hz, "highpass")
        if channel.lpf_hz > 0.0:
            x = self._filter(x, channel.lpf_hz, "lowpass")
        for band in channel.eq_bands:
            x = self._peaking_eq(x, float(band["freq_hz"]), float(band["gain_db"]), float(band.get("q", 1.0)))
        if channel.comp_enabled and channel.comp_ratio > 1.01:
            x = self._compress(x, channel.comp_threshold_db, channel.comp_ratio)
        return (x * _db_to_amp(channel.gain_db)).astype(np.float32)

    def _filter(self, audio: np.ndarray, freq_hz: float, mode: str) -> np.ndarray:
        if butter is None or lfilter is None or freq_hz <= 0.0 or freq_hz >= self.sample_rate * 0.49:
            return audio.astype(np.float32)
        b, a = butter(2, freq_hz / (self.sample_rate * 0.5), btype=mode)
        return lfilter(b, a, audio).astype(np.float32)

    def _peaking_eq(self, audio: np.ndarray, freq_hz: float, gain_db: float, q: float) -> np.ndarray:
        if abs(gain_db) < 1e-6 or lfilter is None:
            return audio.astype(np.float32)
        a = _db_to_amp(gain_db)
        omega = 2.0 * math.pi * float(freq_hz) / float(self.sample_rate)
        alpha = math.sin(omega) / (2.0 * max(q, 0.05))
        cos_omega = math.cos(omega)
        b0 = 1.0 + alpha * a
        b1 = -2.0 * cos_omega
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * cos_omega
        a2 = 1.0 - alpha / a
        b = np.asarray([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        aa = np.asarray([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return lfilter(b, aa, audio).astype(np.float32)

    def _compress(self, audio: np.ndarray, threshold_db: float, ratio: float) -> np.ndarray:
        level_db = 20.0 * np.log10(np.maximum(np.abs(audio), 1e-8))
        over_db = np.maximum(0.0, level_db - float(threshold_db))
        gain_db = -over_db * (1.0 - 1.0 / max(1.0, float(ratio)))
        return (audio * np.power(10.0, gain_db / 20.0)).astype(np.float32)

    def _apply_bus_glue(self, mix: np.ndarray) -> np.ndarray:
        if not self.config.allow_aggressive_mode:
            return mix
        mono = np.mean(mix, axis=1)
        crest = self._crest_factor_db(mono)
        if crest < 7.0:
            return mix
        gain = self._compress(mono, -10.0, 1.25)
        denominator = np.maximum(np.abs(mono), 1e-7)
        gain_factor = np.clip(np.abs(gain) / denominator, 0.75, 1.0)
        return (mix * gain_factor[:, None]).astype(np.float32)

    def _normalize_loudness(self, mix: np.ndarray) -> np.ndarray:
        current = self._loudness_lufs(mix, self.sample_rate)
        if not np.isfinite(current):
            return mix
        gain_db = _clamp(float(self.config.target_lufs) - current, -12.0, 12.0)
        return (mix * _db_to_amp(gain_db)).astype(np.float32)

    def _limit_peak(self, mix: np.ndarray, ceiling_dbfs: float) -> np.ndarray:
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        ceiling = _db_to_amp(ceiling_dbfs)
        if peak <= ceiling or peak <= 0.0:
            return mix.astype(np.float32)
        # Offline director renders are meant to test mix decisions, not to mimic
        # a passive trim. Clamping peak overs preserves the loudness candidate
        # while still enforcing the configured ceiling for safety scoring.
        return np.clip(mix, -ceiling, ceiling).astype(np.float32)

    def _engineering_metrics(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        state: DirectorState | None = None,
        action: DirectorAction | None = None,
    ) -> dict[str, float]:
        arr = self._normalize_audio_shape(audio)
        mono = self._as_mono(arr)
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        peak_db = _amp_to_db(peak)
        lufs = self._loudness_lufs(arr, sr)
        phase_score = self._phase_mono_score(arr)
        crest = self._crest_factor_db(mono)
        clipping_ratio = float(np.mean(np.abs(arr) >= 0.999)) if arr.size else 0.0
        bright, sub = self._brightness_and_sub_ratios(mono, sr)
        loudness_score = _clamp(1.0 - abs(lufs - self.config.target_lufs) / 8.0, 0.0, 1.0) if np.isfinite(lufs) else 0.0
        peak_headroom_score = _clamp(1.0 - max(0.0, peak_db - self.config.true_peak_ceiling_dbfs) * 2.0, 0.0, 1.0)
        anti = _clamp(
            1.0
            - max(0.0, 6.0 - crest) / 8.0
            - max(0.0, bright - 0.42) * 0.8
            - max(0.0, sub - 0.38) * 0.8
            - self._action_size_penalty(action)
            - self._state_density_penalty(state),
            0.0,
            1.0,
        )
        return {
            "lufs": float(lufs),
            "peak_dbfs": float(peak_db),
            "phase_score": float(phase_score),
            "crest_factor_db": float(crest),
            "clipping_ratio": clipping_ratio,
            "brightness_ratio": float(bright),
            "sub_ratio": float(sub),
            "loudness_score": float(loudness_score),
            "peak_headroom_score": float(peak_headroom_score),
            "anti_overprocessing_score": float(anti),
        }

    def _final_score(
        self,
        *,
        muq_score: float,
        loudness_score: float,
        peak_headroom_score: float,
        phase_mono_score: float,
        anti_overprocessing_score: float,
    ) -> float:
        mode = str(self.config.mode or "muq_safe").strip().lower()
        if mode == "pure_muq":
            return float(muq_score)
        safe = (
            0.70 * float(muq_score)
            + 0.10 * float(loudness_score)
            + 0.10 * float(peak_headroom_score)
            + 0.05 * float(phase_mono_score)
            + 0.05 * float(anti_overprocessing_score)
        )
        if mode == "muq_genre":
            return float(0.95 * safe + 0.05 * self._genre_score_placeholder())
        return float(safe)

    def _evaluation_windows(self, audio: np.ndarray, sr: int) -> list[tuple[int, int]]:
        length = len(audio)
        window = max(1, int(round(float(self.config.window_sec) * sr)))
        hop = max(1, int(round(float(self.config.hop_sec) * sr)))
        if length <= window:
            return [(0, length)]
        candidates: list[tuple[float, int]] = []
        mono = self._as_mono(audio)
        for start in range(0, length - window + 1, hop):
            block = mono[start:start + window]
            candidates.append((float(np.sqrt(np.mean(block * block) + 1e-12)), start))
        max_windows = max(1, int(self.config.max_eval_windows))
        selected: list[int] = []
        for _, start in sorted(candidates, reverse=True):
            if all(abs(start - other) >= window for other in selected):
                selected.append(start)
            if len(selected) >= max_windows:
                break
        return [(start, min(length, start + window)) for start in sorted(selected or [0])]

    def _log_step(self, candidate: CandidateResult) -> None:
        if not self.config.log_jsonl:
            return
        self._ensure_output_dirs()
        with self.steps_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(candidate.to_log_record(), ensure_ascii=False) + "\n")

    def _copy_best_mix(self, state: DirectorState) -> None:
        if not state.render_path:
            return
        target = self.best_mix_path
        data, sr = sf.read(state.render_path, dtype="float32", always_2d=False)
        sf.write(str(target), data, sr, subtype="PCM_24")

    def _write_state_json(self, state: DirectorState, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(state), indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_plots(self) -> None:
        if plt is None or not self.score_curve:
            return
        steps = [item["step"] for item in self.score_curve]
        muq = [item["muq_score"] for item in self.score_curve]
        final = [item["final_score"] for item in self.score_curve]

        plt.figure(figsize=(7, 4))
        plt.plot(steps, muq, label="MuQ")
        plt.plot(steps, final, label="Final")
        plt.xlabel("Step")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "score_curve.png")
        plt.close()

        counts: dict[str, int] = {}
        for action in self.accepted_actions:
            label = str(action.get("action", {}).get("action_type", "unknown"))
            counts[label] = counts.get(label, 0) + 1
        if counts:
            plt.figure(figsize=(7, 4))
            plt.bar(list(counts), list(counts.values()))
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(self.output_dir / "action_timeline.png")
            plt.close()

        channel_counts: dict[str, int] = {}
        for action in self.accepted_actions:
            channel = str(action.get("action", {}).get("channel_id", "master"))
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
        if channel_counts:
            plt.figure(figsize=(7, 4))
            plt.bar(list(channel_counts), list(channel_counts.values()))
            plt.xlabel("Channel")
            plt.ylabel("Accepted changes")
            plt.tight_layout()
            plt.savefig(self.output_dir / "channel_change_summary.png")
            plt.close()

    def _metrics_for_path(self, path: str) -> dict[str, float]:
        if not path:
            return {}
        try:
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception:
            return {}
        metrics = self._engineering_metrics(audio, int(sr))
        return {
            "lufs": round(float(metrics["lufs"]), 3),
            "peak_dbfs": round(float(metrics["peak_dbfs"]), 3),
            "phase_score": round(float(metrics["phase_score"]), 6),
        }

    def _compare_existing_automixer(self, best: DirectorState | None) -> dict[str, Any]:
        path = Path(self.config.existing_automixer_path).expanduser() if self.config.existing_automixer_path else None
        if best is None or path is None or not path.exists():
            return {"available": False}
        try:
            existing = self.evaluate_candidate(path)
            return {
                "available": True,
                "existing_path": str(path),
                "existing_muq_score": existing["muq_score"],
                "director_muq_score": best.muq_score,
                "delta": best.muq_score - float(existing["muq_score"]),
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def _build_muq_service(self) -> Any:
        if MuQEvalService is None:
            return None
        return MuQEvalService(
            {
                "enabled": True,
                "device": self.config.device,
                "window_sec": self.config.window_sec,
                "hop_sec": self.config.hop_sec,
                "sample_rate": 24000,
                "fallback_enabled": False,
                "log_scores": False,
                "local_files_only": self.config.local_files_only,
                "muq_eval_root": self.config.muq_eval_root,
            }
        )

    def _muq_available(self) -> bool:
        return bool(self.muq_service is not None and getattr(self.muq_service, "model_status", "") == "available")

    def _model_status(self) -> str:
        if self.muq_service is None:
            return "not_initialized"
        return str(getattr(self.muq_service, "model_status", "unknown") or "unknown")

    @staticmethod
    def _aggregate_model_status(statuses: Iterable[str]) -> str:
        normalized = [str(status or "unknown") for status in statuses]
        if not normalized:
            return "unknown"
        unique = set(normalized)
        if unique == {"available"}:
            return "available"
        if len(unique) == 1:
            return normalized[0]
        return "mixed"

    def _ensure_muq_service(self) -> None:
        if self.muq_service is None:
            self.muq_service = self._build_muq_service()

    def _ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.best_states_dir.mkdir(parents=True, exist_ok=True)

    def _next_state_id(self, step: int) -> str:
        return f"state_{step:04d}_{len(self._state_archive) + 1:04d}"

    @staticmethod
    def _normalize_audio_shape(audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            return arr.T.astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    @classmethod
    def _as_mono(cls, audio: np.ndarray) -> np.ndarray:
        arr = cls._normalize_audio_shape(audio)
        if arr.ndim == 1:
            return arr.astype(np.float32, copy=False)
        return np.mean(arr, axis=1).astype(np.float32)

    @staticmethod
    def _resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        arr = MuqEvalDirectorOfflineTest._normalize_audio_shape(audio)
        if src_sr == dst_sr or len(arr) == 0:
            return arr.astype(np.float32, copy=True)
        dst_len = max(1, int(round(len(arr) * float(dst_sr) / float(src_sr))))
        src_x = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        if arr.ndim == 1:
            return np.interp(dst_x, src_x, arr).astype(np.float32)
        channels = [np.interp(dst_x, src_x, arr[:, idx]).astype(np.float32) for idx in range(arr.shape[1])]
        return np.column_stack(channels).astype(np.float32)

    @staticmethod
    def _classify_instrument(name: str) -> str:
        text = name.lower().replace("_", " ")
        if "kick" in text:
            return "kick"
        if "snare" in text:
            return "snare"
        if "bass" in text:
            return "bass"
        if "vocal" in text or "vox" in text:
            return "backing_vocal" if "back" in text or "bgv" in text else "lead_vocal"
        if "oh" in text or "overhead" in text:
            return "overhead"
        if "tom" in text:
            return "floor_tom" if "f tom" in text or "floor" in text else "rack_tom"
        if "guitar" in text:
            return "electric_guitar"
        if "playback" in text or "track" in text:
            return "playback"
        if "accordion" in text:
            return "accordion"
        return "other"

    @staticmethod
    def _initial_pan(name: str, instrument: str) -> float:
        text = name.lower()
        if instrument in CENTER_LOCKED:
            return 0.0
        if " l" in f" {text}" or text.endswith("l"):
            return -0.72
        if " r" in f" {text}" or text.endswith("r"):
            return 0.72
        if instrument == "overhead":
            return -0.65 if "l" in text else 0.65 if "r" in text else 0.0
        return 0.0

    @staticmethod
    def _initial_hpf(instrument: str) -> float:
        return {
            "kick": 30.0,
            "bass": 35.0,
            "snare": 80.0,
            "rack_tom": 60.0,
            "floor_tom": 50.0,
            "lead_vocal": 85.0,
            "backing_vocal": 100.0,
            "electric_guitar": 70.0,
            "accordion": 80.0,
            "overhead": 140.0,
            "playback": 30.0,
        }.get(instrument, 30.0)

    @staticmethod
    def _initial_comp_threshold(instrument: str) -> float:
        return {
            "lead_vocal": -24.0,
            "backing_vocal": -24.0,
            "kick": -18.0,
            "snare": -20.0,
            "bass": -22.0,
        }.get(instrument, -24.0)

    @staticmethod
    def _initial_comp_ratio(instrument: str) -> float:
        return {
            "lead_vocal": 3.0,
            "backing_vocal": 2.5,
            "kick": 3.0,
            "snare": 3.2,
            "bass": 3.0,
        }.get(instrument, 1.0)

    def _eq_probes_for(self, instrument: str) -> list[tuple[float, float]]:
        max_gain = float(self.config.max_eq_gain_change_db)
        if instrument == "kick":
            return [(60.0, 0.8), (250.0, -max_gain), (3500.0, 0.8)]
        if instrument in {"bass", "bass_guitar", "bass_di"}:
            return [(80.0, 0.7), (250.0, -1.0), (1200.0, 0.5)]
        if instrument == "snare":
            return [(250.0, -1.0), (3500.0, 0.8), (8500.0, -0.8)]
        if instrument in {"lead_vocal", "backing_vocal"}:
            return [(250.0, -0.8), (2500.0, 0.7), (6000.0, -0.7), (12000.0, 0.5)]
        if instrument in {"electric_guitar", "accordion", "playback"}:
            return [(250.0, -0.8), (500.0, -0.5), (2500.0, -0.7), (6000.0, -0.7)]
        if instrument == "overhead":
            return [(500.0, -0.5), (3500.0, -0.8), (8500.0, -0.6), (12000.0, 0.5)]
        return [(freq, -0.5) for freq in EQ_CENTERS_HZ[:4]]

    def _apply_compressor_preset(self, channel: ChannelState, preset: str) -> None:
        presets = {
            "vocal_smooth": (-26.0, 3.5, 8.0, 140.0),
            "drum_punch": (-18.0, 4.0, 6.0, 90.0),
            "bass_control": (-24.0, 3.2, 18.0, 180.0),
            "bus_glue": (-12.0, 1.6, 30.0, 240.0),
        }
        threshold, ratio, attack, release = presets.get(preset, presets["vocal_smooth"])
        channel.comp_enabled = True
        channel.comp_threshold_db = threshold
        channel.comp_ratio = ratio
        channel.comp_attack_ms = attack
        channel.comp_release_ms = release

    @staticmethod
    def _pan_mono(audio: np.ndarray, pan: float) -> np.ndarray:
        pan = _clamp(pan, -1.0, 1.0)
        theta = (pan + 1.0) * math.pi / 4.0
        left = math.cos(theta)
        right = math.sin(theta)
        return np.column_stack((audio * left, audio * right)).astype(np.float32)

    @staticmethod
    def _loudness_lufs(audio: np.ndarray, sr: int) -> float:
        arr = MuqEvalDirectorOfflineTest._normalize_audio_shape(audio)
        if pyln is not None and len(arr) >= sr // 2:
            try:
                return float(pyln.Meter(sr).integrated_loudness(arr))
            except Exception:
                pass
        mono = MuqEvalDirectorOfflineTest._as_mono(arr)
        return _amp_to_db(float(np.sqrt(np.mean(mono * mono) + 1e-12)))

    @staticmethod
    def _crest_factor_db(audio: np.ndarray) -> float:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12)) if audio.size else 0.0
        return _amp_to_db(peak) - _amp_to_db(rms)

    @staticmethod
    def _phase_mono_score(audio: np.ndarray) -> float:
        arr = MuqEvalDirectorOfflineTest._normalize_audio_shape(audio)
        if arr.ndim == 1 or arr.shape[1] < 2:
            return 1.0
        left = arr[:, 0]
        right = arr[:, 1]
        if np.std(left) < 1e-9 or np.std(right) < 1e-9:
            return 0.6
        corr = float(np.corrcoef(left, right)[0, 1])
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        side_ratio = float(np.sqrt(np.mean(side * side) + 1e-12) / (np.sqrt(np.mean(mid * mid) + 1e-12) + 1e-12))
        return _clamp(1.0 - max(0.0, -corr) * 0.8 - max(0.0, side_ratio - 0.9) * 0.6 - max(0.0, 0.03 - side_ratio) * 0.3, 0.0, 1.0)

    @staticmethod
    def _brightness_and_sub_ratios(audio: np.ndarray, sr: int) -> tuple[float, float]:
        mono = MuqEvalDirectorOfflineTest._as_mono(audio)
        if len(mono) < 256:
            return 0.0, 0.0
        size = min(32768, 2 ** int(math.floor(math.log2(len(mono)))))
        frame = mono[:size] * np.hanning(size)
        spectrum = np.abs(np.fft.rfft(frame)).astype(np.float64) ** 2 + 1e-12
        freqs = np.fft.rfftfreq(size, 1.0 / sr)
        total = float(np.sum(spectrum)) + 1e-12

        def ratio(lo: float, hi: float) -> float:
            mask = (freqs >= lo) & (freqs < hi)
            return float(np.sum(spectrum[mask]) / total) if np.any(mask) else 0.0

        return ratio(5000.0, 14000.0), ratio(20.0, 80.0)

    def _action_size_penalty(self, action: DirectorAction | None) -> float:
        if action is None:
            return 0.0
        params = action.parameters
        if action.action_type == "gain":
            return min(0.12, abs(float(params.get("delta_db", 0.0))) / 24.0)
        if action.action_type == "eq":
            return min(0.12, abs(float(params.get("gain_db", 0.0))) / 20.0)
        if action.action_type == "pan":
            return min(0.08, abs(float(params.get("delta", 0.0))) / 2.0)
        if action.action_type.startswith("compressor"):
            return 0.04
        return 0.0

    @staticmethod
    def _state_density_penalty(state: DirectorState | None) -> float:
        if state is None:
            return 0.0
        changes = sum(channel.change_count for channel in state.channels.values())
        eq_count = sum(len(channel.eq_bands) for channel in state.channels.values())
        return min(0.25, changes * 0.005 + eq_count * 0.004)

    @staticmethod
    def _genre_score_placeholder() -> float:
        return 0.7

    @staticmethod
    def _parameters_for_action(state: DirectorState, action: DirectorAction) -> dict[str, Any]:
        if action.channel_id is None:
            return {"master": True}
        channel = state.channels.get(int(action.channel_id))
        return _json_safe(channel) if channel is not None else {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MUQ_EVAL_DIRECTOR_OFFLINE_TEST.")
    parser.add_argument("--input", required=True, help="Path to a multitrack WAV directory.")
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "muq_director_test.yaml"))
    parser.add_argument("--mode", default="", choices=["", "pure_muq", "muq_safe", "muq_genre", "muq_vs_existing_agent"])
    parser.add_argument("--output", default="", help="Output session directory.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = MuqDirectorConfig.from_yaml(args.config)
    if args.mode:
        config.mode = args.mode
    output_dir = Path(args.output) if args.output else None
    director = MuqEvalDirectorOfflineTest(config, output_dir=output_dir)
    report = director.run(args.input, dry_run=bool(args.dry_run))
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=False))
    return 0 if report.get("status") in {"completed", "dry_run", "disabled"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
