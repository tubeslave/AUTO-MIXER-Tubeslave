import numpy as np
import soundfile as sf

from ai_mixing_pipeline.audio_utils import measure_audio_file
from ai_mixing_pipeline.decision_layer.action_schema import (
    CandidateActionSet,
    CompressorAction,
    EQAction,
    GainAction,
    NoChangeAction,
)
from ai_mixing_pipeline.decision_layer.fallback_virtual_mixer import FallbackVirtualMixer
from ai_mixing_pipeline.decision_layer.pymixconsole_adapter import PyMixConsoleAdapter


def _write_sine(path, amp=0.2, freq=440.0, sr=48000):
    t = np.arange(int(sr * 0.1), dtype=np.float32) / sr
    sf.write(path, (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32), sr)


def test_fallback_virtual_mixer_renders_gain_and_prevents_clipping(tmp_path):
    multitrack = tmp_path / "multitrack"
    multitrack.mkdir()
    _write_sine(multitrack / "vocal.wav", amp=0.9)
    mixer = FallbackVirtualMixer({"sample_rate": 48000, "prevent_clipping": True, "safety": {"max_true_peak_dbfs": -1.0}})
    mixer.load_project(multitrack)

    no_change = mixer.render(CandidateActionSet("no_change", [NoChangeAction()]), tmp_path / "no_change.wav")
    louder = mixer.render(CandidateActionSet("vocal_up", [GainAction("vocal", 1.0)]), tmp_path / "vocal_up.wav")

    no_change_level = measure_audio_file(no_change["path"])["level"]
    louder_level = measure_audio_file(louder["path"])["level"]
    assert louder_level["clip_count"] == 0
    assert louder_level["true_peak_dbtp"] <= -1.0
    assert louder_level["rms_dbfs"] >= no_change_level["rms_dbfs"]


def test_fallback_virtual_mixer_applies_eq_and_compressor(tmp_path):
    multitrack = tmp_path / "multitrack"
    multitrack.mkdir()
    _write_sine(multitrack / "vocal.wav", amp=0.35, freq=250.0)
    mixer = FallbackVirtualMixer({"sample_rate": 48000, "prevent_clipping": True})
    mixer.load_project(multitrack)

    candidate = CandidateActionSet(
        "eq_comp",
        [
            EQAction("vocal", "low_mid", 250.0, -1.5, 1.0),
            CompressorAction("vocal", threshold_db=-28.0, ratio=2.0, attack_ms=5.0, release_ms=80.0),
        ],
    )
    result = mixer.render(candidate, tmp_path / "eq_comp.wav")

    statuses = [item["status"] for item in result["audit"]]
    assert statuses == ["applied", "applied"]
    assert not any("unsupported" in warning for warning in result["warnings"])
    processed, _ = sf.read(result["path"], always_2d=True)
    baseline = np.mean(np.abs(sf.read(multitrack / "vocal.wav", always_2d=True)[0]))
    assert np.mean(np.abs(processed)) < baseline


def test_pymixconsole_adapter_uses_real_backend_or_fallback(tmp_path):
    multitrack = tmp_path / "multitrack"
    multitrack.mkdir()
    _write_sine(multitrack / "vocal.wav", amp=0.2)
    mixer = PyMixConsoleAdapter({"sample_rate": 48000, "prevent_clipping": True})
    mixer.load_project(multitrack)

    result = mixer.render(
        CandidateActionSet("vocal_up", [GainAction("vocal", 0.5)]),
        tmp_path / "pymixconsole_or_fallback.wav",
    )

    assert result["virtual_mixer"] in {"pymixconsole", "fallback_virtual_mixer"}
    assert result["osc_midi_sent"] is False
    assert (tmp_path / "pymixconsole_or_fallback.wav").exists()
