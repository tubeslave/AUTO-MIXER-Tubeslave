"""Audio loading, smoke generation, and WAV writing for `production_mix_v1`."""

from __future__ import annotations

from pathlib import Path
import wave
from typing import Any, Iterable, Mapping

import numpy as np

from .models import AudioStem, RoleDetectionResult, ensure_stereo
from .role_detection import detect_role


AUDIO_SUFFIXES = {".wav", ".wave", ".aif", ".aiff", ".flac", ".ogg"}


def generate_smoke_stems(
    *,
    sample_rate: int = 48000,
    duration_sec: float = 3.0,
) -> list[AudioStem]:
    """Generate deterministic synthetic stems for offline dry-run smoke tests."""
    rng = np.random.default_rng(1729)
    samples = int(sample_rate * duration_sec)
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)

    kick = _decaying_sine(t, 58.0, 0.34, sample_rate, period_sec=0.5)
    bass = 0.10 * np.sin(2.0 * np.pi * 92.0 * t) + 0.025 * np.sin(2.0 * np.pi * 184.0 * t)
    guitar = 0.045 * np.sin(2.0 * np.pi * 220.0 * t) + 0.018 * rng.normal(size=samples)
    vocal = 0.060 * np.sin(2.0 * np.pi * 330.0 * t)
    vocal += 0.028 * np.sin(2.0 * np.pi * 980.0 * t)
    vocal += 0.008 * rng.normal(size=samples)

    stems = [
        AudioStem(
            "Kick",
            ensure_stereo(kick),
            sample_rate,
            "kick",
            1,
            -10.0,
            0.0,
            role_detection=_trusted_role("kick", ["kick"]),
        ),
        AudioStem(
            "Bass",
            ensure_stereo(bass),
            sample_rate,
            "bass",
            2,
            -12.0,
            0.0,
            role_detection=_trusted_role("bass", ["bass"]),
        ),
        AudioStem(
            "Gtr",
            ensure_stereo(guitar),
            sample_rate,
            "electric_guitar",
            3,
            -14.0,
            -0.35,
            role_detection=_trusted_role("electric_guitar", ["gtr"]),
        ),
        AudioStem(
            "Lead Vox",
            ensure_stereo(vocal),
            sample_rate,
            "lead_vocal",
            4,
            -16.0,
            0.0,
            role_detection=_trusted_role("lead_vocal", ["lead", "vox"]),
        ),
    ]
    return stems


def load_audio_stems(
    input_dir: str | Path,
    *,
    target_sample_rate: int | None = None,
    max_duration_sec: float | None = None,
    role_overrides: Mapping[str, Any] | None = None,
) -> list[AudioStem]:
    """Load audio files from a directory as stems.

    `soundfile` is used when available. WAV files also work through stdlib
    `wave`, so the pipeline remains smoke-testable without optional audio I/O
    packages.
    """
    root = Path(input_dir).expanduser()
    files = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    stems: list[AudioStem] = []
    for index, path in enumerate(files, start=1):
        audio, sample_rate = _read_audio_file(path)
        if target_sample_rate and sample_rate != target_sample_rate:
            audio = _resample_linear(audio, sample_rate, target_sample_rate)
            sample_rate = int(target_sample_rate)
        if max_duration_sec is not None:
            max_samples = int(float(max_duration_sec) * sample_rate)
            audio = audio[:max_samples]
        role_result = detect_role(
            filename=path.name,
            channel_name=path.stem,
            audio=audio,
            sample_rate=sample_rate,
            overrides=role_overrides or {},
        )
        stems.append(
            AudioStem(
                name=path.stem,
                audio=ensure_stereo(audio),
                sample_rate=sample_rate,
                role=role_result.role,
                channel_id=index,
                current_fader_db=-12.0,
                base_pan=infer_pan_from_name(path.stem),
                path=str(path),
                role_detection=role_result,
            )
        )
    return stems


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> str:
    """Write a stereo WAV file and return its path."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stereo = np.clip(ensure_stereo(audio), -1.0, 1.0)
    try:
        import soundfile as sf

        sf.write(output_path, stereo, sample_rate)
        return str(output_path)
    except Exception:
        pcm = (stereo * 32767.0).astype("<i2")
        with wave.open(str(output_path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(int(sample_rate))
            handle.writeframes(pcm.tobytes())
        return str(output_path)


def infer_role_from_name(name: str) -> str:
    token = name.lower().replace("-", "_").replace(" ", "_")
    if "playback" in token or "track" in token:
        return "playback"
    if "back" in token or "bgv" in token:
        return "backing_vocal"
    if "kick" in token:
        return "kick"
    if "snare" in token:
        return "snare"
    if "tom" in token:
        return "toms"
    if "bass" in token:
        return "bass"
    if "vox" in token or "vocal" in token or "lead" in token:
        return "lead_vocal"
    if "gtr" in token or "guitar" in token:
        return "electric_guitar"
    if (
        "key" in token
        or "piano" in token
        or "synth" in token
        or "pad" in token
        or "accordion" in token
    ):
        return "keys"
    if "fx" in token or "verb" in token or "delay" in token:
        return "fx_return"
    if "oh" in token or "overhead" in token or "room" in token:
        return "overheads"
    return "unknown"


def _trusted_role(role: str, tokens: list[str]) -> RoleDetectionResult:
    return RoleDetectionResult(
        role=role,
        confidence=1.0,
        evidence={
            "filename_tokens": tokens,
            "spectral_features": {},
            "user_override": {"role": role, "confidence": 1.0, "source": "generated_smoke"},
        },
        is_ambiguous=False,
        allowed_action_level="role_specific",
    )


def infer_pan_from_name(name: str) -> float:
    """Infer safe static pan from common exported mono L/R stem names."""
    token = f" {name.lower().replace('-', ' ').replace('_', ' ')} "
    is_left = any(marker in token for marker in (" l ", " left "))
    is_right = any(marker in token for marker in (" r ", " right "))
    if not is_left and not is_right:
        return 0.0
    side = -1.0 if is_left else 1.0
    if "oh" in token or "overhead" in token or "playback" in token:
        return 0.80 * side
    if "guitar" in token:
        return 0.65 * side
    if "back" in token or "vox" in token:
        return 0.50 * side
    return 0.45 * side


def align_stems(stems: Iterable[AudioStem]) -> list[AudioStem]:
    """Pad/truncate stems so all have the same sample count and sample rate."""
    stem_list = list(stems)
    if not stem_list:
        return []
    sample_rate = stem_list[0].sample_rate
    target_len = max(int(stem.audio.shape[0]) for stem in stem_list)
    aligned = []
    for stem in stem_list:
        audio = stem.audio
        if stem.sample_rate != sample_rate:
            audio = _resample_linear(audio, stem.sample_rate, sample_rate)
        if audio.shape[0] < target_len:
            audio = np.pad(audio, ((0, target_len - audio.shape[0]), (0, 0)))
        else:
            audio = audio[:target_len]
        aligned.append(
            AudioStem(
                name=stem.name,
                audio=ensure_stereo(audio),
                sample_rate=sample_rate,
                role=stem.role,
                channel_id=stem.channel_id,
                current_fader_db=stem.current_fader_db,
                base_pan=stem.base_pan,
                path=stem.path,
                role_detection=stem.role_detection,
            )
        )
    return aligned


def _read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        return ensure_stereo(audio), int(sample_rate)
    except Exception:
        if path.suffix.lower() not in {".wav", ".wave"}:
            raise
        return _read_wav_stdlib(path)


def _read_wav_stdlib(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path}: stdlib WAV fallback supports 16-bit PCM only")
    pcm = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    return ensure_stereo(pcm), int(sample_rate)


def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    stereo = ensure_stereo(audio)
    if source_rate == target_rate or stereo.size == 0:
        return stereo
    duration = stereo.shape[0] / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=stereo.shape[0], endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    lanes = [np.interp(target_x, source_x, stereo[:, idx]) for idx in range(stereo.shape[1])]
    return np.stack(lanes, axis=1).astype(np.float32)


def _decaying_sine(
    t: np.ndarray,
    frequency_hz: float,
    level: float,
    sample_rate: int,
    *,
    period_sec: float,
) -> np.ndarray:
    period = max(1, int(period_sec * sample_rate))
    phase = np.arange(t.size) % period
    envelope = np.exp(-phase / float(max(1, int(0.08 * sample_rate))))
    return (level * envelope * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)
