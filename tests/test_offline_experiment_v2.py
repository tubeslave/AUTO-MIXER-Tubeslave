from pathlib import Path

from automixer.experiments import ExperimentRunner


def test_offline_experiment_saves_json_and_markdown_reports(tmp_path):
    report = ExperimentRunner().run(
        {
            "analyzer_output": {
                "channels": [
                    {
                        "channel_id": 1,
                        "role": "lead_vocal",
                        "metrics": {
                            "lufs": -24.0,
                            "target_lufs": -20.0,
                            "true_peak_dbtp": -10.0,
                        },
                        "confidence": 0.9,
                    }
                ]
            },
            "current_state": {"channel:1": {"true_peak_dbtp": -10.0}},
        },
        tmp_path,
    )

    json_path = Path(report["artifacts"]["json"])
    md_path = Path(report["artifacts"]["markdown"])
    assert json_path.exists()
    assert md_path.exists()
    assert "rules_only" in report["variants"]
    assert "rules_critic_decision_engine" in report["variants"]
