import numpy as np
import soundfile as sf

from ai_mixing_pipeline.critics import AudioboxAestheticsCritic, CLAPSemanticCritic, MuQEvalCritic
from ai_mixing_pipeline.stem_critics import MERTStemCritic
from ai_mixing_pipeline.technical_analyzers import EssentiaTechnicalAnalyzer, IdentityBleedCritic


def _write(path, freq=440.0, amp=0.2, sr=48000):
    t = np.arange(int(sr * 0.2), dtype=np.float32) / sr
    audio = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, audio, sr)


def test_missing_heavy_models_return_fallback_results(tmp_path):
    before = tmp_path / "before.wav"
    after = tmp_path / "after.wav"
    _write(before, amp=0.2)
    _write(after, amp=0.15)

    critics = [
        MuQEvalCritic({"enabled": True, "local_files_only": True, "max_window_seconds": 1}),
        AudioboxAestheticsCritic({"enabled": True}),
        MERTStemCritic(
            {
                "enabled": True,
                "backend": "mert",
                "model_name": "missing-local/mert-model",
                "local_files_only": True,
                "fallback_to_lightweight": True,
            }
        ),
        CLAPSemanticCritic({"enabled": True}),
        EssentiaTechnicalAnalyzer({"enabled": True}),
        IdentityBleedCritic({"enabled": True}),
    ]

    for critic in critics:
        result = critic.compare(str(before), str(after), context={"channel_name": "lead_vocal"})
        assert result["critic_name"] == critic.name
        assert "overall" in result["delta"]
        assert isinstance(result["warnings"], list)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["score_source"] in {"real_model", "proxy", "unavailable"}


def test_panns_beats_fallback_compare_is_neutral(tmp_path):
    before = tmp_path / "before.wav"
    after = tmp_path / "after.wav"
    _write(before, amp=0.2)
    _write(after, amp=0.1)

    result = IdentityBleedCritic({"enabled": True}).compare(str(before), str(after), context={"channel_name": "vocal"})

    assert result["score_source"] == "unavailable"
    assert result["delta"]["overall"] == 0.0
    assert result["scores"]["bleed_score"] == 0.0
    assert any("neutral" in warning.lower() for warning in result["warnings"])


def test_mert_compare_reports_embedding_distance(tmp_path):
    before = tmp_path / "before.wav"
    after = tmp_path / "after.wav"
    _write(before, freq=220.0, amp=0.2)
    _write(after, freq=880.0, amp=0.2)

    result = MERTStemCritic(
        {
            "enabled": True,
            "backend": "lightweight",
            "fallback_to_lightweight": True,
            "max_window_seconds": 1,
        }
    ).compare(str(before), str(after), context={"channel_name": "lead_vocal"})

    assert "embedding_cosine_distance" in result["delta"]
    assert result["delta"]["embedding_cosine_distance"] >= 0.0
    assert "embedding_similarity" in result["metadata"]
