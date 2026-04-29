import json
import subprocess
import sys

import numpy as np
import soundfile as sf

from mix_agent.agent import analyze, apply_conservative
from mix_agent.reporting import render_markdown_report


def _write_stem(path, frequency, amplitude=0.2, sample_rate=48000, duration=1.0):
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    audio = (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)
    sf.write(path, audio, sample_rate)
    return audio


def test_offline_analyze_generates_metrics_issues_and_report(tmp_path):
    stems = tmp_path / "stems"
    stems.mkdir()
    _write_stem(stems / "lead_vocal.wav", 2500.0, amplitude=0.05)
    _write_stem(stems / "guitars.wav", 2600.0, amplitude=0.25)
    _write_stem(stems / "bass.wav", 90.0, amplitude=0.25)

    plan = analyze(stems=str(stems), genre="rock")
    report = render_markdown_report(plan)

    assert plan.analysis.mix.level["peak_dbfs"] <= 0.0
    assert plan.analysis.masking_matrix
    assert plan.dashboard.technical_health_score >= 0.0
    assert "Mix Agent Report" in report
    assert "Detected Issues" in report


def test_mix_agent_cli_writes_json_and_markdown(tmp_path):
    stems = tmp_path / "stems"
    stems.mkdir()
    _write_stem(stems / "kick.wav", 60.0, amplitude=0.4)
    _write_stem(stems / "bass.wav", 70.0, amplitude=0.4)
    out = tmp_path / "suggestions.json"
    md = tmp_path / "report.md"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mix_agent",
            "suggest",
            "--stems",
            str(stems),
            "--genre",
            "hip-hop",
            "--out",
            str(out),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "actions" in payload
    assert "issues" in payload
    assert "actions" in result.stdout

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mix_agent",
            "analyze",
            "--stems",
            str(stems),
            "--genre",
            "hip-hop",
            "--out",
            str(md),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "# Mix Agent Report" in md.read_text(encoding="utf-8")


def test_apply_conservative_renders_without_overwriting_stems(tmp_path):
    stems = tmp_path / "stems"
    stems.mkdir()
    original = _write_stem(stems / "guitars.wav", 300.0, amplitude=0.9)
    _write_stem(stems / "lead_vocal.wav", 1200.0, amplitude=0.05)
    out = tmp_path / "render.wav"

    plan = apply_conservative(stems=str(stems), out=str(out), genre="rock")

    assert out.exists()
    rendered, sr = sf.read(out, always_2d=True)
    reread, _ = sf.read(stems / "guitars.wav", always_2d=False)
    assert sr == 48000
    assert rendered.shape[1] == 2
    assert np.allclose(reread[: len(original)], original, atol=1e-4)
    assert plan.audit_trail
