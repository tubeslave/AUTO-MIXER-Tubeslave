import json
from pathlib import Path

from mix_agent.stem_training_loop import generate_fixture, run_training_loop


def test_stem_training_loop_generates_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_DISABLED", "true")
    input_path = generate_fixture(tmp_path / "fixture.wav", seconds=2.0)
    result = run_training_loop(input_path, tmp_path / "out", splitter_mode="mock")

    output_mix = Path(result.output_mix_path)
    report = Path(result.report_path)

    assert output_mix.exists()
    assert report.exists()
    assert result.splitter_mode == "mock"
    assert result.osc_disabled is True
    assert result.score_after >= 0.0
    assert result.score_before >= 0.0

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["osc_disabled"] is True
    assert "metrics_before" in data
    assert set(data["metrics_before"].keys()) == {"vocals", "drums", "bass", "other"}


def test_stem_training_loop_refuses_when_osc_not_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("OSC_DISABLED", "false")
    input_path = generate_fixture(tmp_path / "fixture.wav", seconds=1.0)
    try:
        run_training_loop(input_path, tmp_path / "out", splitter_mode="mock")
    except RuntimeError as exc:
        assert "OSC_DISABLED" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when OSC_DISABLED is false")
