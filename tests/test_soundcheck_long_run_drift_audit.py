from __future__ import annotations

import importlib.util
import json
from pathlib import Path


RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "soundcheck_long_run_drift_audit.py"
)
SPEC = importlib.util.spec_from_file_location("soundcheck_long_run_drift_audit", RUNNER_PATH)
assert SPEC is not None
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_long_run_drift_audit_passes_absolute_proposal_only_targets(tmp_path: Path) -> None:
    evidence = tmp_path / "safe.jsonl"
    _write_jsonl(
        evidence,
        [
            {
                "cycle": 1,
                "t": 0.0,
                "proposals": [
                    {
                        "id": "fader.ch1",
                        "target": "ch1.fdr",
                        "parameters": {"target_fader_db": -12.0, "delta_db": -0.5},
                    },
                    {
                        "id": "trim.ch2",
                        "target": "ch2",
                        "parameters": {"target_trim_db": -2.0, "delta_db": -0.25},
                    },
                ],
            },
            {
                "cycle": 2,
                "t": 601.0,
                "proposals": [
                    {
                        "id": "fader.ch1",
                        "target": "ch1.fdr",
                        "parameters": {"target_fader_db": -12.0, "delta_db": -0.5},
                    },
                    {
                        "id": "trim.ch2",
                        "target": "ch2",
                        "parameters": {"target_trim_db": -2.0, "delta_db": -0.25},
                    },
                ],
            },
        ],
    )

    summary = audit.build_audit_summary([evidence], min_duration_sec=600.0)

    assert summary["status"] == "passed"
    assert summary["duration_sec"] == 601.0
    assert summary["action_count"] == 4
    assert summary["live_authority_granted"] is False
    assert summary["readiness_closure_allowed"] is False


def test_long_run_drift_audit_fails_live_relative_positive_actions(tmp_path: Path) -> None:
    evidence = tmp_path / "unsafe.jsonl"
    _write_jsonl(
        evidence,
        [
            {
                "cycle": 1,
                "t": 0.0,
                "applied": [
                    {
                        "id": "trim.ch4",
                        "target": "ch4",
                        "parameters": {"gain_db": 1.0},
                        "metadata": {
                            "safety_gate": {"dry_run": False, "send_to_executor": True}
                        },
                    },
                    {
                        "id": "fader.ch1",
                        "target": "ch1.fdr",
                        "parameters": {"gain_db": -1.5},
                        "metadata": {
                            "safety_gate": {"dry_run": False, "send_to_executor": True}
                        },
                    },
                ],
            },
            {
                "cycle": 2,
                "t": 700.0,
                "applied": [
                    {
                        "id": "fader.ch1",
                        "target": "ch1.fdr",
                        "parameters": {"gain_db": -1.75},
                        "metadata": {
                            "safety_gate": {"dry_run": False, "send_to_executor": True}
                        },
                    }
                ],
            },
        ],
    )

    summary = audit.build_audit_summary(
        [evidence],
        min_duration_sec=600.0,
        max_cumulative_db=3.0,
    )

    assert summary["status"] == "failed"
    assert "live_executor_actions_present" in summary["failures"]
    assert "positive_gain_or_trim_boosts_present" in summary["failures"]
    assert "relative_only_targets_present" in summary["failures"]
    assert "cumulative_relative_drift_present" in summary["failures"]
    assert summary["positive_gain_or_trim_boost_count"] == 1
    assert summary["cumulative_drift_surface_count"] == 1


def test_long_run_drift_audit_requires_duration_and_actions(tmp_path: Path) -> None:
    evidence = tmp_path / "empty.jsonl"
    _write_jsonl(evidence, [{"cycle": 1, "t": 10.0}, {"cycle": 2, "t": 20.0}])

    summary = audit.build_audit_summary([evidence], min_duration_sec=600.0)

    assert summary["status"] == "failed"
    assert "duration_below_minimum" in summary["failures"]
    assert "proposal_actions_missing" in summary["failures"]


def test_long_run_drift_audit_does_not_count_eq_gain_as_trim_boost(tmp_path: Path) -> None:
    evidence = tmp_path / "eq.jsonl"
    _write_jsonl(
        evidence,
        [
            {
                "cycle": 1,
                "t": 0.0,
                "proposals": [
                    {
                        "id": "eq.ch1.b2",
                        "target": "ch1.eq.b2",
                        "action_type": "gain_correction",
                        "parameters": {"gain_db": 1.5},
                    }
                ],
            },
            {"cycle": 2, "t": 601.0, "proposals": []},
        ],
    )

    summary = audit.build_audit_summary([evidence], min_duration_sec=600.0)

    assert summary["status"] == "passed"
    assert summary["positive_gain_or_trim_boost_count"] == 0
    assert summary["relative_only_target_count"] == 0
