import numpy as np
import soundfile as sf

from ai_mixing_pipeline.critics import AudioboxAestheticsCritic, CLAPSemanticCritic
from ai_mixing_pipeline.technical_analyzers import EssentiaTechnicalAnalyzer


def _write_sine(path, freq=440.0, amp=0.2, sr=48000):
    t = np.arange(int(sr * 0.2), dtype=np.float32) / sr
    audio = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, audio, sr)


def _assert_standard_result(result):
    assert set(
        [
            "critic_name",
            "role",
            "scores",
            "delta",
            "confidence",
            "warnings",
            "explanation",
        ]
    ).issubset(result)
    assert isinstance(result["scores"], dict)
    assert isinstance(result["delta"], dict)
    assert 0.0 <= result["confidence"] <= 1.0


def test_audio_critic_analyze_and_compare_share_standard_shape(tmp_path):
    before = tmp_path / "before.wav"
    after = tmp_path / "after.wav"
    _write_sine(before, amp=0.2)
    _write_sine(after, amp=0.1)

    for critic in [
        AudioboxAestheticsCritic({"enabled": True}),
        CLAPSemanticCritic({"enabled": True, "prompts": ["clean lead vocal", "muddy low mids"]}),
        EssentiaTechnicalAnalyzer({"enabled": True}),
    ]:
        _assert_standard_result(critic.analyze(str(before)))
        compared = critic.compare(str(before), str(after))
        _assert_standard_result(compared)
        assert "overall" in compared["delta"]
