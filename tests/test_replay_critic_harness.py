from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ai_mixing_pipeline.audio_utils import measure_audio_file, read_audio, write_audio
from ai_mixing_pipeline.critics.base import AudioCritic, standard_critic_result


@dataclass(frozen=True)
class _Snapshot:
    path: str
    audio_sha256: str
    duration_sec: float
    sample_rate: int


class _PumpingCritic(AudioCritic):
    name = "pumping"
    role = "dynamics_pumping_critic"

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        mono = _mono_audio(audio_path)
        envelope = _moving_rms(mono, window_size=256)
        std_ratio = float(np.std(envelope) / (np.mean(envelope) + 1e-12))
        score = float(max(0.0, min(1.0, 1.0 - (std_ratio / 1.8))))
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={"overall": score, "pumping_index": std_ratio},
            delta={},
            confidence=0.95,
            warnings=[],
            explanation="Lower envelope variance indicates less pumping.",
            model_available=False,
            metadata={"envelope_std_ratio": std_ratio},
        )


class _IntelligibilityCritic(AudioCritic):
    name = "intelligibility"
    role = "intelligibility_critic"

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        mono = _mono_audio(audio_path)
        score, detail = _intelligibility_score(mono)
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={"overall": score, "high_band_ratio": detail["high_band_ratio"], "mid_band_ratio": detail["mid_band_ratio"]},
            delta={},
            confidence=0.85,
            warnings=[],
            explanation="Higher high-frequency balance improves intelligibility proxy.",
            model_available=False,
            metadata=detail,
        )


class _GainStabilityCritic(AudioCritic):
    name = "gain_stability"
    role = "gain_stability_critic"

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = dict(context or {})
        target_delta = float(context.get("gain_delta_db", 0.0))
        score = float(max(0.0, min(1.0, 1.0 - abs(target_delta) / 16.0)))
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={"overall": score, "gain_delta_db": target_delta},
            delta={},
            confidence=0.9,
            warnings=[],
            explanation="Predictable gain deltas score higher than large jumps.",
            model_available=False,
            metadata={"requested_gain_delta_db": target_delta},
        )

    def compare(
        self,
        before_path: str,
        after_path: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        before = self.analyze(before_path, context=context)
        after = self.analyze(after_path, context=context)
        before_mono = _mono_audio(before_path)
        after_mono = _mono_audio(after_path)
        before_level = 20.0 * np.log10(max(1e-12, float(np.sqrt(np.mean(before_mono ** 2)))))
        after_level = 20.0 * np.log10(max(1e-12, float(np.sqrt(np.mean(after_mono ** 2)))))
        measured_delta_db = float(after_level - before_level)
        score_after = float(max(0.0, min(1.0, 1.0 - abs(measured_delta_db) / 16.0)))
        score_before = float(before["scores"].get("overall", 0.0))
        delta = {"overall": score_after - score_before, "measured_gain_delta_db": measured_delta_db}
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={"overall": score_after, "gain_delta_db": measured_delta_db},
            delta=delta,
            confidence=0.9,
            warnings=[],
            explanation="Gain delta is computed from measured RMS between before and after audio.",
            model_available=False,
            metadata={"measured_gain_delta_db": measured_delta_db},
        )


class _RoomCompensationCritic(AudioCritic):
    name = "room_compensation"
    role = "room_balance_critic"

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        mono = _mono_audio(audio_path)
        metrics = _band_energy(mono)
        target = 0.22
        low_mid = float(metrics["low_mid_band"])
        low_mid_error = abs(low_mid - target)
        score = float(max(0.0, min(1.0, 1.0 - (low_mid_error / max(target, 0.01)))))
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={"overall": score, "low_mid_band": low_mid},
            delta={},
            confidence=0.8,
            warnings=[],
            explanation="Lower low-mid mismatch indicates better room compensation balance.",
            model_available=False,
            metadata=metrics,
        )


class _AbruptTransitionCritic(AudioCritic):
    name = "abrupt_transition"
    role = "artifact_guard_critic"

    def analyze(self, audio_path: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        mono = _mono_audio(audio_path)
        level = np.abs(mono)
        level = _moving_rms(level, window_size=1024)
        step = float(np.max(np.abs(np.diff(level))))
        rms = float(np.sqrt(np.mean(level ** 2)) + 1e-12)
        abruptness = step / (rms + 1e-12)
        score = float(max(0.0, min(1.0, 1.0 - abruptness)))
        return standard_critic_result(
            critic_name=self.name,
            role=self.role,
            scores={"overall": score, "abruptness": abruptness},
            delta={},
            confidence=0.88,
            warnings=[],
            explanation="Smooth level envelope is rewarded over abrupt transitions.",
            model_available=False,
            metadata={"step_metric": step, "rms": rms},
        )


class _ReplayCriticHarness:
    def __init__(self, *, scenario_path: Path, output_dir: Path):
        self.scenario_path = Path(scenario_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.critics = {
            "pumping": _PumpingCritic(),
            "intelligibility": _IntelligibilityCritic(),
            "gain_stability": _GainStabilityCritic(),
            "room_compensation": _RoomCompensationCritic(),
            "abrupt_transition": _AbruptTransitionCritic(),
        }

    def run(self) -> dict[str, Any]:
        scenario = _load_scenario(self.scenario_path)
        run_output = {
            "scenario": scenario["scenario"],
            "events": [],
            "module_names": sorted(self.critics),
        }
        for index, event in enumerate(scenario["events"]):
            family = event["family"]
            critic = self.critics[family]
            before, after, sample_rate = _render_event_audio(event)
            render_dir = self.output_dir / f"run_{event['event_id']}"
            render_dir.mkdir(parents=True, exist_ok=True)
            before_path = render_dir / "before.wav"
            after_path = render_dir / "after.wav"
            write_audio(before_path, before, sample_rate)
            write_audio(after_path, after, sample_rate)
            context = {
                "event_id": event["event_id"],
                "family": family,
                "gain_delta_db": float(event.get("payload", {}).get("after_gain_db", 0.0))
                - float(event.get("payload", {}).get("base_gain_db", 0.0)),
            }
            before_result = critic.analyze(str(before_path), context=context)
            after_result = critic.analyze(str(after_path), context=context)
            compared = critic.compare(str(before_path), str(after_path), context=context)
            event_output = {
                "event_id": event["event_id"],
                "family": family,
                "index": index,
                "before": asdict(_snapshot(before_path)),
                "after": asdict(_snapshot(after_path)),
                "before_scores": before_result["scores"],
                "after_scores": after_result["scores"],
                "delta": compared["delta"],
                "warnings": sorted(set((before_result.get("warnings") or []) + (after_result.get("warnings") or []))),
                "context": context,
                "expectations": event.get("expectations", {}),
            }
            event_output["artifacts"] = self._persist_event_artifacts(render_dir, event_output)
            run_output["events"].append(event_output)

        return self._persist_run_artifacts(run_output)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> str:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return str(path)

    def _persist_event_artifacts(self, render_dir: Path, event_output: dict[str, Any]) -> dict[str, str]:
        snapshot_path = render_dir / "event_snapshot.json"
        diff_path = render_dir / "event_regression_diff.json"
        snapshot_diff = _json_diff(event_output["before"], event_output["after"])
        score_diff = _json_diff(event_output["before_scores"], event_output["after_scores"])
        diff_payload = {
            "event_id": event_output["event_id"],
            "family": event_output["family"],
            "snapshot_changes": snapshot_diff,
            "score_changes": score_diff,
            "delta": dict(event_output["delta"]),
            "expectations": dict(event_output.get("expectations", {})),
            "warnings": list(event_output.get("warnings", [])),
        }
        return {
            "snapshot": self._write_json(snapshot_path, event_output),
            "regression_diff": self._write_json(diff_path, diff_payload),
        }

    def _persist_run_artifacts(self, run_output: dict[str, Any]) -> dict[str, Any]:
        stable_snapshot = _stable_result(run_output)
        regression_diffs = {
            "scenario": run_output["scenario"],
            "module_names": list(run_output["module_names"]),
            "events": [
                {
                    "event_id": event["event_id"],
                    "family": event["family"],
                    "snapshot_changes": _json_diff(event["before"], event["after"]),
                    "score_changes": _json_diff(event["before_scores"], event["after_scores"]),
                    "delta": dict(event["delta"]),
                    "expectations": dict(event.get("expectations", {})),
                }
                for event in run_output["events"]
            ],
        }
        artifact_paths = {
            "run_output": self._write_json(self.output_dir / "replay_run_output.json", run_output),
            "stable_snapshot": self._write_json(self.output_dir / "replay_stable_snapshot.json", stable_snapshot),
            "regression_diffs": self._write_json(self.output_dir / "replay_regression_diffs.json", regression_diffs),
        }
        manifest = {
            "scenario": run_output["scenario"],
            "event_ids": [event["event_id"] for event in run_output["events"]],
            "artifacts": artifact_paths,
            "per_event_artifacts": {
                event["event_id"]: dict(event.get("artifacts", {}))
                for event in run_output["events"]
            },
        }
        artifact_paths["debug_manifest"] = self._write_json(
            self.output_dir / "replay_debug_manifest.json",
            manifest,
        )
        run_output["artifacts"] = artifact_paths
        return run_output


def _load_scenario(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise ValueError("Replay scenario payload must contain event list")
    return payload


def _snapshot(path: Path) -> _Snapshot:
    metrics = measure_audio_file(path)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return _Snapshot(
        path=str(path),
        audio_sha256=digest,
        duration_sec=float(metrics["duration_sec"]),
        sample_rate=int(metrics["sample_rate"]),
    )


def _mono_audio(path: str) -> np.ndarray:
    audio, _ = read_audio(path)
    return np.mean(np.asarray(audio, dtype=np.float32), axis=1)


def _moving_rms(signal: np.ndarray, window_size: int) -> np.ndarray:
    if signal.size == 0:
        return np.zeros(1, dtype=np.float32)
    kernel = np.ones(int(window_size), dtype=np.float32) / float(window_size)
    padded = np.pad(np.square(signal), (window_size - 1, 0), mode="edge")
    return np.sqrt(np.convolve(padded, kernel, mode="valid"))


def _intelligibility_score(signal: np.ndarray) -> tuple[float, dict[str, float]]:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1 / 48000)
    total = float(np.sum(spectrum[1:]) + 1e-12)
    high = float(np.sum(spectrum[(freqs >= 3000.0)]) / total)
    mid = float(np.sum(spectrum[(freqs >= 500.0) & (freqs < 3000.0)]) / total)
    score = float(max(0.0, min(1.0, high - (0.20 - 0.8 * (mid + 1e-12)) / 2.0 + 0.5)))
    return score, {"high_band_ratio": high, "mid_band_ratio": mid}


def _band_energy(signal: np.ndarray) -> dict[str, float]:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1 / 48000)
    total = float(np.sum(spectrum[1:]) + 1e-12)
    low_mid = float(np.sum(spectrum[(freqs >= 120.0) & (freqs < 420.0)] ) / total)
    return {"low_mid_band": low_mid}


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _render_base_tone(seed: int, duration_sec: float, sample_rate: int, frequency_hz: float = 220.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration_sec * sample_rate), dtype=np.float32) / float(sample_rate)
    harmonic = 0.35 * np.sin(2.0 * np.pi * float(frequency_hz) * t)
    bed = 0.03 * rng.normal(size=t.shape)
    return (harmonic + bed).astype(np.float32)


def _render_event_audio(event: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    sample_rate = int(event.get("payload", {}).get("sample_rate", event.get("sample_rate", 48000))) or 48000
    duration = float(event.get("duration_sec", 1.0))
    seed = int(event.get("seed", 0))
    family = event["family"]
    payload = dict(event.get("payload") or {})

    base = _render_base_tone(seed, duration, sample_rate, frequency_hz=float(payload.get("frequency_hz", 220.0)))
    base *= _db_to_amp(-24.0)

    if family == "pumping":
        before = base.copy()
        t = np.linspace(0.0, duration, num=base.shape[0], endpoint=False, dtype=np.float32)
        modulation = 1.0 + float(payload.get("modulation_depth", 0.35)) * np.sin(
            2.0 * np.pi * float(payload.get("modulation_rate_hz", 6.0)) * t
        )
        after = (base * modulation).astype(np.float32)
    elif family == "intelligibility":
        before = _lowpass(base, cutoff_hz=float(payload.get("lowpass_freq_hz", 2200.0)), sample_rate=sample_rate)
        t = np.linspace(0.0, duration, num=base.shape[0], endpoint=False, dtype=np.float32)
        high_band = 0.03 * np.sin(2.0 * np.pi * 3500.0 * t) * _db_to_amp(float(payload.get("highband_gain_db", 2.0)))
        after = (before + high_band).astype(np.float32)
    elif family == "gain_stability":
        before_gain = float(payload.get("base_gain_db", -20.0))
        after_gain = float(payload.get("after_gain_db", -20.0))
        before = base * _db_to_amp(before_gain)
        after = base * _db_to_amp(after_gain)
    elif family == "room_compensation":
        t = np.linspace(0.0, duration, num=base.shape[0], endpoint=False, dtype=np.float32)
        room_hum = 0.12 * np.sin(2.0 * np.pi * 220.0 * t)
        room_hum *= _db_to_amp(float(payload.get("room_hum_db", -8.0)))
        before = base + room_hum
        compensation = 0.12 * _db_to_amp(-float(payload.get("compensation_gain_db", 3.0)))
        after = (before - room_hum * compensation).astype(np.float32)
    elif family == "abrupt_transition":
        before = base.copy()
        after = base.copy()
        transition = float(payload.get("transition_ratio", 0.5))
        split = int(after.shape[0] * transition)
        after[:split] *= _db_to_amp(float(payload.get("attack_gain_db", -12.0)))
        after[split:] *= _db_to_amp(float(payload.get("release_gain_db", 0.0)))
    else:
        raise ValueError(f"Unknown replay family: {family}")

    before_stereo = np.stack([before, before * 0.9], axis=1).astype(np.float32)
    after_stereo = np.stack([after, after * 0.9], axis=1).astype(np.float32)
    return before_stereo, after_stereo, sample_rate


def _lowpass(signal: np.ndarray, *, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    if signal.size == 0:
        return signal
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1 / float(sample_rate))
    filtered = spectrum * (freqs <= float(cutoff_hz)).astype(np.float32)
    return np.fft.irfft(filtered).astype(np.float32)


def _sign(value: float) -> str:
    if value > 1e-6:
        return "positive"
    if value < -1e-6:
        return "negative"
    return "zero"


def _assert_delta_sign(value: float, expectation: str) -> None:
    actual = _sign(value)
    if expectation == "positive":
        assert actual == "positive"
    elif expectation == "non_positive":
        assert actual in {"negative", "zero"}
    elif expectation == "negative":
        assert actual == "negative"
    elif expectation == "non_negative":
        assert actual in {"positive", "zero"}
    else:
        raise ValueError(f"Unknown expectation: {expectation}")


def _stable_event(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "family": event["family"],
        "index": event["index"],
        "before": {
            "audio_sha256": event["before"]["audio_sha256"],
            "duration_sec": event["before"]["duration_sec"],
            "sample_rate": event["before"]["sample_rate"],
        },
        "after": {
            "audio_sha256": event["after"]["audio_sha256"],
            "duration_sec": event["after"]["duration_sec"],
            "sample_rate": event["after"]["sample_rate"],
        },
        "before_scores": event["before_scores"],
        "after_scores": event["after_scores"],
        "delta": event["delta"],
        "warnings": event["warnings"],
    }


def _stable_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario": result["scenario"],
        "module_names": result["module_names"],
        "events": [_stable_event(event) for event in result["events"]],
    }


def _json_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    current_path = path or "$"
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            next_path = key if not path else f"{path}.{key}"
            if key not in before:
                changes.append({"path": next_path, "before": None, "after": after[key]})
            elif key not in after:
                changes.append({"path": next_path, "before": before[key], "after": None})
            else:
                changes.extend(_json_diff(before[key], after[key], next_path))
        return changes
    if isinstance(before, list) and isinstance(after, list):
        if before == after:
            return []
        return [{"path": current_path, "before": before, "after": after}]
    if before == after:
        return []
    return [{"path": current_path, "before": before, "after": after}]


def _scenario_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "critic_replay_scenarios.json"


def test_replay_critic_harness_is_deterministic_for_all_families(tmp_path: Path):
    first = _stable_result(
        _ReplayCriticHarness(scenario_path=_scenario_path(), output_dir=tmp_path / "run_a").run()
    )
    second = _stable_result(
        _ReplayCriticHarness(scenario_path=_scenario_path(), output_dir=tmp_path / "run_b").run()
    )
    assert second == first


def test_replay_critic_family_scores_and_snapshot_comparisons(tmp_path: Path):
    harness = _ReplayCriticHarness(scenario_path=_scenario_path(), output_dir=tmp_path / "single")
    result = harness.run()

    assert result["scenario"] == "deterministic_critic_replay_v1"
    assert len(result["events"]) == 5

    for event in result["events"]:
        before = event["before"]
        after = event["after"]
        assert before["audio_sha256"] != after["audio_sha256"]
        assert before["sample_rate"] == after["sample_rate"]
        assert event["index"] in {0, 1, 2, 3, 4}
        expectation = event["expectations"]["delta_sign"]
        delta = float(event["delta"].get("overall", 0.0))
        _assert_delta_sign(delta, expectation)
        assert event["before_scores"]["overall"] >= 0.0
        assert event["after_scores"]["overall"] >= 0.0
        assert event["before_scores"]["overall"] <= 1.0
        assert event["after_scores"]["overall"] <= 1.0

    event_families = [event["family"] for event in result["events"]]
    assert event_families == [
        "pumping",
        "intelligibility",
        "gain_stability",
        "room_compensation",
        "abrupt_transition",
    ]


def test_replay_critic_harness_persists_debug_artifacts_and_regression_diffs(tmp_path: Path):
    harness = _ReplayCriticHarness(scenario_path=_scenario_path(), output_dir=tmp_path / "artifacts")
    result = harness.run()

    artifact_paths = result["artifacts"]
    stable_snapshot = json.loads(Path(artifact_paths["stable_snapshot"]).read_text(encoding="utf-8"))
    regression_diffs = json.loads(Path(artifact_paths["regression_diffs"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(artifact_paths["debug_manifest"]).read_text(encoding="utf-8"))

    assert stable_snapshot == _stable_result(result)
    assert regression_diffs["scenario"] == result["scenario"]
    assert len(regression_diffs["events"]) == len(result["events"])
    assert manifest["artifacts"]["run_output"] == artifact_paths["run_output"]
    assert set(manifest["per_event_artifacts"]) == {event["event_id"] for event in result["events"]}

    gain_event = next(event for event in regression_diffs["events"] if event["event_id"] == "E003")
    pumping_event = next(event for event in regression_diffs["events"] if event["event_id"] == "E001")
    assert any(change["path"] == "audio_sha256" for change in gain_event["snapshot_changes"])
    assert any(change["path"] == "overall" for change in pumping_event["score_changes"])

    per_event = next(event for event in result["events"] if event["event_id"] == "E001")
    event_snapshot = json.loads(Path(per_event["artifacts"]["snapshot"]).read_text(encoding="utf-8"))
    event_diff = json.loads(Path(per_event["artifacts"]["regression_diff"]).read_text(encoding="utf-8"))
    assert event_snapshot["event_id"] == "E001"
    assert event_diff["event_id"] == "E001"
    assert event_diff["snapshot_changes"]


@pytest.mark.parametrize("event_id,family", [("E001", "pumping"), ("E002", "intelligibility"), ("E004", "room_compensation")])
def test_replay_critic_harness_supports_family_specific_interfaces(event_id: str, family: str, tmp_path: Path):
    harness = _ReplayCriticHarness(scenario_path=_scenario_path(), output_dir=tmp_path / family)
    result = harness.run()
    by_id = {event["event_id"]: event for event in result["events"]}
    event = by_id[event_id]
    assert event["family"] == family
    assert event["context"]["family"] == family
    assert event["before_scores"]["overall"] != event["after_scores"]["overall"]
