"""Audio loading and writing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import load_production_mix_config
from .models import AudioStem, ROLE_ACTION_ROLE_SPECIFIC, RoleDetectionResult, ensure_stereo
from .role_detection import detect_role


AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac"}


def load_audio_stems(input_dir: str | Path, *, role_overrides: Mapping[str, Any] | None = None) -> list[AudioStem]:
    import soundfile as sf

    config = load_production_mix_config()
    root = Path(input_dir).expanduser()
    files = sorted(path for path in root.iterdir() if path.suffix.lower() in AUDIO_EXTENSIONS)
    stems: list[AudioStem] = []
    for idx, path in enumerate(files, start=1):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        detection = detect_role(
            filename=path.name,
            channel_name=path.stem,
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=int(sample_rate),
            config=config,
            overrides=role_overrides or config.role_overrides,
        )
        stems.append(
            AudioStem(
                name=path.stem,
                audio=ensure_stereo(audio),
                sample_rate=int(sample_rate),
                role=detection.role,
                channel_id=idx,
                path=str(path),
                role_detection=detection,
            )
        )
    return stems


def align_stems(stems: list[AudioStem]) -> list[AudioStem]:
    if not stems:
        return []
    max_len = max(stem.audio.shape[0] for stem in stems)
    aligned = []
    for stem in stems:
        audio = ensure_stereo(stem.audio)
        if audio.shape[0] < max_len:
            audio = np.pad(audio, ((0, max_len - audio.shape[0]), (0, 0)))
        aligned.append(AudioStem(**{**stem.__dict__, "audio": audio[:max_len]}))
    return aligned


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> str:
    import soundfile as sf

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, ensure_stereo(audio), int(sample_rate))
    return str(output)


def generate_smoke_stems(sample_rate: int = 48000, seconds: float = 3.0) -> list[AudioStem]:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False, dtype=np.float32)
    specs = [
        ("Kick", "kick", 55.0, 0.20),
        ("Bass", "bass", 90.0, 0.12),
        ("Gtr", "electric_guitar", 440.0, 0.08),
        ("Lead Vox", "lead_vocal", 660.0, 0.08),
    ]
    stems = []
    for idx, (name, role, freq, amp) in enumerate(specs, start=1):
        audio = (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)
        detection = RoleDetectionResult(
            role=role,
            confidence=1.0,
            evidence={"filename_tokens": [name.lower()], "spectral_features": {}, "user_override": {"source": "generated_smoke"}},
            allowed_action_level=ROLE_ACTION_ROLE_SPECIFIC,
        )
        stems.append(AudioStem(name=name, role=role, audio=ensure_stereo(audio), sample_rate=sample_rate, channel_id=idx, role_detection=detection))
    return stems
