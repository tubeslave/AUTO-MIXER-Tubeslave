import json
import subprocess
import sys

import numpy as np
import soundfile as sf


def _write_sine(path, freq, amp=0.2, sr=48000):
    t = np.arange(int(sr * 0.25), dtype=np.float32) / sr
    audio = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, audio, sr)


def test_offline_test_chain_cli_writes_required_artifacts(tmp_path):
    input_dir = tmp_path / "offline_test_input"
    output_dir = tmp_path / "offline_test_output"
    multitrack = input_dir / "multitrack"
    reference = input_dir / "reference"
    config_dir = input_dir / "config"
    multitrack.mkdir(parents=True)
    reference.mkdir(parents=True)
    config_dir.mkdir(parents=True)

    _write_sine(multitrack / "01_kick.wav", 60.0, 0.25)
    _write_sine(multitrack / "02_snare.wav", 180.0, 0.15)
    _write_sine(multitrack / "03_bass.wav", 90.0, 0.20)
    _write_sine(multitrack / "04_vocal.wav", 880.0, 0.12)
    _write_sine(reference / "reference_mix.wav", 440.0, 0.18)
    (config_dir / "channel_map.json").write_text(
        json.dumps(
            {
                "01_kick": {"role": "kick"},
                "02_snare": {"role": "snare"},
                "03_bass": {"role": "bass"},
                "04_vocal": {"role": "lead_vocal"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_mixing_pipeline.offline_test_runner",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--config",
            "configs/ai_mixing_roles.yaml",
            "--mode",
            "offline_test",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)

    assert payload["selected_candidate_id"]
    assert {
        "muq_eval",
        "audiobox_aesthetics",
        "mert",
        "clap",
        "essentia",
        "panns_or_beats",
        "demucs_or_openunmix",
        "automix_toolkit_fxnorm_diffmst_deepafx",
    }.issubset(payload["module_status"])
    assert all(
        payload["module_status"][name].get("participated", True)
        for name in [
            "muq_eval",
            "audiobox_aesthetics",
            "mert",
            "clap",
            "essentia",
            "panns_or_beats",
            "demucs_or_openunmix",
            "automix_toolkit_fxnorm_diffmst_deepafx",
        ]
    )
    assert (output_dir / "renders" / "000_initial_mix.wav").exists()
    assert (output_dir / "renders" / "001_candidate_gain_balance.wav").exists()
    assert (output_dir / "renders" / "002_candidate_eq_cleanup.wav").exists()
    assert (output_dir / "renders" / "003_candidate_compression.wav").exists()
    assert (output_dir / "renders" / "004_candidate_fx.wav").exists()
    assert (output_dir / "renders" / "005_best_mix.wav").exists()
    assert (output_dir / "reports" / "decision_log.jsonl").exists()
    assert (output_dir / "reports" / "summary_report.md").exists()
    assert (output_dir / "reports" / "critic_scores.csv").exists()
    assert (output_dir / "reports" / "accepted_actions.json").exists()
    assert (output_dir / "reports" / "rejected_actions.json").exists()
    assert (output_dir / "snapshots" / "mixer_state_before.json").exists()
    assert (output_dir / "snapshots" / "mixer_state_after.json").exists()

    rows = [
        json.loads(line)
        for line in (output_dir / "reports" / "decision_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["candidate_id"] == "000_initial_mix" for row in rows)
    assert any(row["accepted"] for row in rows)
    assert all(row["rejection_reason"] is None or row["rejection_reason"] for row in rows)
    summary = (output_dir / "reports" / "summary_report.md").read_text(encoding="utf-8")
    assert "Safety Governor" in summary
    assert "demucs_or_openunmix" in summary
    assert "automix_toolkit_fxnorm_diffmst_deepafx" in summary
