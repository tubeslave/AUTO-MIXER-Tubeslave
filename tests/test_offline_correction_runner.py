import json
import subprocess
import sys

import numpy as np
import soundfile as sf


def _write_sine(path, freq, amp=0.2, sr=48000):
    t = np.arange(int(sr * 0.1), dtype=np.float32) / sr
    sf.write(path, (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32), sr)


def test_offline_correction_runner_creates_best_mix_and_reports(tmp_path):
    input_dir = tmp_path / "offline_test_input"
    multitrack = input_dir / "multitrack"
    config_dir = input_dir / "config"
    output_dir = tmp_path / "offline_test_output"
    multitrack.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    _write_sine(multitrack / "01_kick.wav", 60.0, 0.2)
    _write_sine(multitrack / "02_bass.wav", 90.0, 0.15)
    _write_sine(multitrack / "03_vocal.wav", 440.0, 0.1)
    (config_dir / "channel_map.json").write_text(
        json.dumps(
            {
                "01_kick": {"role": "kick"},
                "02_bass": {"role": "bass"},
                "03_vocal": {"role": "vocal"},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "ai_decision_layer.yaml"
    config.write_text(
        """
optimizer:
  primary: nevergrad
  random_seed: 42
virtual_mixer:
  preferred: fallback_virtual_mixer
  sample_rate: 48000
  loudness_match: true
  target_lufs: -24.0
  prevent_clipping: true
safety:
  max_gain_change_db_per_step: 1.0
  max_eq_change_db_per_step: 1.5
  max_master_gain_boost_db: 0.5
  min_score_improvement: 0.03
  max_true_peak_dbfs: -1.0
  min_headroom_db: 1.0
  forbid_clipping: true
critics:
  muq_eval: {enabled: false, weight: 0.30}
  audiobox: {enabled: false, weight: 0.20}
  stem_critics: {enabled: false, weight: 0.15}
  clap: {enabled: false, weight: 0.10}
  essentia: {enabled: false, weight: 0.10}
  identity_bleed: {enabled: false, weight: 0.10}
outputs:
  save_reports: true
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ai_mixing_pipeline.decision_layer.offline_correction_runner",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--config",
            str(config),
            "--mode",
            "offline_test",
            "--optimizer",
            "nevergrad",
            "--max-candidates",
            "8",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    run_dir = output_dir / payload["run_id"]

    assert (run_dir / "renders" / "best_mix.wav").exists()
    assert (run_dir / "reports" / "summary_report.md").exists()
    assert (run_dir / "reports" / "decision_log.jsonl").exists()
    assert (run_dir / "reports" / "candidate_manifest.json").exists()
    assert (run_dir / "reports" / "mixer_state_after.json").exists()
