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
