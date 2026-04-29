"""CLI runner for offline decision/correction mixing."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any

from ai_mixing_pipeline.reports import write_decision_layer_reports

from .action_schema import CandidateActionSet, NoChangeAction
from .candidate_generator import CandidateGenerator
from .critic_bridge import CriticBridge
from .fallback_virtual_mixer import FallbackVirtualMixer
from .mixer_state import discover_multitrack, role_from_channel_map
from .pymixconsole_adapter import PyMixConsoleAdapter
from .sandbox_renderer import DecisionSandboxRenderer
from .decision_engine import CorrectionDecisionEngine
from .safety_governor import DecisionSafetyGovernor


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required for ai_decision_layer.yaml") from exc
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    payload.setdefault("optimizer", {})
    payload.setdefault("virtual_mixer", {})
    payload.setdefault("action_space", {})
    payload.setdefault("safety", {})
    payload.setdefault("critics", {})
    payload.setdefault("outputs", {})
    return payload


class OfflineCorrectionRunner:
    """End-to-end Current Mix -> candidates -> critics -> safety -> best mix."""

    def __init__(
        self,
        *,
        input_dir: str | Path,
        output_dir: str | Path,
        config_path: str | Path,
        mode: str = "offline_test",
        optimizer: str = "nevergrad",
        max_candidates: int = 20,
    ):
        self.input_dir = Path(input_dir).expanduser()
        self.output_root = Path(output_dir).expanduser()
        self.config_path = Path(config_path).expanduser()
        self.mode = mode
        self.optimizer = optimizer
        self.max_candidates = int(max_candidates)
        self.config = load_config(self.config_path)
        self.run_id = datetime.now().strftime("decision_correction_%Y%m%d_%H%M%S")
        self.run_dir = self.output_root / self.run_id
        self.reports_dir = self.run_dir / "reports"
        self.renders_dir = self.run_dir / "renders"

    def run(self) -> dict[str, Any]:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        multitrack_dir, channel_map, files = discover_multitrack(self.input_dir)
        channels = {path.stem: role_from_channel_map(path, channel_map) for path in files}
        mixer = self._build_mixer()
        mixer_state_before = mixer.load_project(multitrack_dir, channel_map)
        self._write_json(self.reports_dir / "mixer_state_before.json", mixer_state_before)

        initial_candidate = CandidateActionSet(
            "candidate_000_no_change",
            [NoChangeAction()],
            "Required no-change baseline.",
            "manual_rule",
            self.config.get("safety", {}),
        )
        technical_profile = {"muddiness_proxy": 0.25, "harshness_proxy": 0.0}
        generator = CandidateGenerator(self.config)
        candidates = generator.generate(
            channels,
            technical_profile,
            optimizer_name=self.optimizer,
            max_candidates=self.max_candidates,
        )
        if not any(candidate.is_no_change for candidate in candidates):
            candidates.insert(0, initial_candidate)

        governor = DecisionSafetyGovernor(self.config)
        action_safety: dict[str, dict[str, Any]] = {}
        rejected_pre_render: list[dict[str, Any]] = []
        renderable: list[CandidateActionSet] = []
        for candidate in candidates:
            verdict = governor.check_actions(candidate)
            action_safety[candidate.candidate_id] = verdict
            if verdict["passed"] or candidate.is_no_change:
                renderable.append(candidate)
            else:
                rejected_pre_render.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "actions": [action.to_dict() for action in candidate.actions],
                        "safety": verdict,
                        "stage": "action_level",
                    }
                )

        renderer = DecisionSandboxRenderer(
            mixer,
            self.run_dir,
            {
                **(self.config.get("virtual_mixer", {}) or {}),
                "safety": self.config.get("safety", {}),
                "ayaic_balance": self.config.get("ayaic_balance", {}),
            },
        )
        render_results = renderer.render_candidates(renderable)
        no_change_id = next((candidate.candidate_id for candidate in candidates if candidate.is_no_change), candidates[0].candidate_id)
        before_path = render_results[no_change_id]["path"]

        bridge = CriticBridge(self.config)
        critic_scores: dict[str, dict[str, dict[str, Any]]] = {}
        for candidate in candidates:
            render_path = render_results.get(candidate.candidate_id, {}).get("path")
            if not render_path:
                critic_scores[candidate.candidate_id] = {}
                continue
            critic_scores[candidate.candidate_id] = bridge.compare(
                before_path,
                render_path,
                context={"run_id": self.run_id, "candidate_id": candidate.candidate_id},
            )

        safety_results: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            if candidate.candidate_id not in render_results:
                safety_results[candidate.candidate_id] = action_safety[candidate.candidate_id]
                continue
            render_verdict = governor.check_render(candidate, render_results[candidate.candidate_id]["path"])
            critic_verdict = governor.check_critics(candidate, critic_scores.get(candidate.candidate_id, {}))
            safety_results[candidate.candidate_id] = governor.combine(
                candidate.candidate_id,
                action_safety[candidate.candidate_id],
                render_verdict,
                critic_verdict,
            )

        decision = CorrectionDecisionEngine(self.config).choose_best(
            self.run_id,
            candidates,
            critic_scores,
            safety_results,
        )
        generator.tell_results(
            {key: float(value) for key, value in (decision.get("final_scores", {}) or {}).items()},
            {
                candidate_id: {
                    "safety": safety_results.get(candidate_id, {}),
                    "selected": candidate_id == decision.get("selected_candidate_id"),
                }
                for candidate_id in decision.get("final_scores", {})
            },
        )
        selected_path = render_results.get(decision["selected_candidate_id"], render_results[no_change_id])["path"]
        best_mix_path = self.renders_dir / "best_mix.wav"
        shutil.copyfile(selected_path, best_mix_path)
        render_results["best_mix"] = {"path": str(best_mix_path), "candidate_id": decision["selected_candidate_id"]}

        mixer_state_after = mixer.export_state()
        mixer_state_after.update(
            {
                "selected_candidate_id": decision["selected_candidate_id"],
                "accepted_actions": decision["selected_actions"],
                "best_mix_path": str(best_mix_path),
                "state_applied": self.mode in {"offline_test", "assisted_offline"},
                "osc_midi_sent": False,
            }
        )
        self._write_json(self.reports_dir / "mixer_state_after.json", mixer_state_after)

        dependency_status = dependency_availability()
        module_status = bridge.module_status(critic_scores)
        module_status.setdefault("safety_governor", {"available": True, "participated": True, "warnings": [], "role": "final_protection_layer"})
        report_paths = write_decision_layer_reports(
            self.reports_dir,
            run_id=self.run_id,
            candidates=candidates,
            render_results=render_results,
            critic_scores=critic_scores,
            safety_results=safety_results,
            decision=decision,
            optimizer_status=generator.optimizer_status,
            mixer_status=mixer_state_after.get("metadata", {}),
            dependency_status=dependency_status,
            module_status=module_status,
            rejected_pre_render=rejected_pre_render,
        )
        return {
            "run_id": self.run_id,
            "output_dir": str(self.run_dir),
            "best_mix_path": str(best_mix_path),
            "selected_candidate_id": decision["selected_candidate_id"],
            "decision": decision,
            "reports": report_paths
            | {
                "candidate_manifest": str(self.reports_dir / "candidate_manifest.json"),
                "mixer_state_before": str(self.reports_dir / "mixer_state_before.json"),
                "mixer_state_after": str(self.reports_dir / "mixer_state_after.json"),
            },
            "dependency_status": dependency_status,
        }

    def _build_mixer(self):
        vm_config = {
            **(self.config.get("virtual_mixer", {}) or {}),
            "safety": self.config.get("safety", {}),
            "ayaic_balance": self.config.get("ayaic_balance", {}),
        }
        preferred = str(vm_config.get("preferred", "pymixconsole"))
        if preferred == "pymixconsole":
            return PyMixConsoleAdapter(vm_config)
        return FallbackVirtualMixer(vm_config)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def dependency_availability() -> dict[str, bool]:
    import importlib.util

    return {
        "nevergrad": importlib.util.find_spec("nevergrad") is not None,
        "optuna": importlib.util.find_spec("optuna") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "scipy": importlib.util.find_spec("scipy") is not None,
        "soundfile": importlib.util.find_spec("soundfile") is not None,
        "pyloudnorm": importlib.util.find_spec("pyloudnorm") is not None,
        "pandas": importlib.util.find_spec("pandas") is not None,
        "pyyaml": importlib.util.find_spec("yaml") is not None,
        "pymixconsole": importlib.util.find_spec("pymixconsole") is not None,
        "dasp_pytorch": importlib.util.find_spec("dasp_pytorch") is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run offline AI decision/correction mixing.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/ai_decision_layer.yaml")
    parser.add_argument("--mode", default="offline_test")
    parser.add_argument("--optimizer", default="nevergrad", choices=["nevergrad", "optuna", "manual"])
    parser.add_argument("--max-candidates", type=int, default=20)
    args = parser.parse_args(argv)
    optimizer = "nevergrad" if args.optimizer == "manual" else args.optimizer
    runner = OfflineCorrectionRunner(
        input_dir=args.input,
        output_dir=args.output,
        config_path=args.config,
        mode=args.mode,
        optimizer=optimizer,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(runner.run(), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
