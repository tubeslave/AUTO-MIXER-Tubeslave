"""Offline audio loader for mixes, stems and optional references."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import soundfile as sf

from mix_agent.models import AnalysisContext, AudioStem

AUDIO_SUFFIXES = {".wav", ".wave", ".flac", ".aif", ".aiff", ".ogg"}


@dataclass
class LoadedAudioContext:
    """Decoded audio buffers for one analysis pass."""

    context: AnalysisContext
    mix_audio: np.ndarray
    mix_sample_rate: int
    stems: Dict[str, np.ndarray] = field(default_factory=dict)
    stem_info: Dict[str, AudioStem] = field(default_factory=dict)
    reference_audio: Optional[np.ndarray] = None
    reference_sample_rate: int = 0


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if audio.size == 0:
        raise ValueError(f"{path} contains no audio")
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _audio_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES
    )


def infer_stem_role(name: str) -> str:
    """Infer a broad stem role from a filename or channel label."""
    label = Path(name).stem.lower().replace("-", "_").replace(" ", "_")
    tokens = {part for part in label.split("_") if part}
    if "synth" in tokens and {"perc", "percussion"} & tokens and {"pad", "pads"} & tokens:
        return "playback"
    checks = [
        ("lead_vocal", ("leadvox", "lead_vocal", "vocal_main", "main_vocal", "vox_lead")),
        ("backing_vocal", ("backing", "back_vox", "bgv", "bvox", "harmony")),
        ("vocal", ("vocal", "vox", "voice")),
        ("kick", ("kick", "bd", "bass_drum")),
        ("snare", ("snare", "sn", "clap")),
        ("floor_tom", ("floor_tom", "ftom", "f_tom")),
        ("tom", ("tom", "rack_tom")),
        ("hihat", ("hihat", "hi_hat", "hh", "hat")),
        ("ride", ("ride",)),
        ("percussion", ("percussion", "perc")),
        ("drums", ("drum", "oh", "overhead", "cymbal")),
        ("bass", ("bass", "sub", "808")),
        ("guitars", ("gtr", "guitar", "strat", "tele", "riff")),
        ("synth", ("synth", "lead_synth")),
        ("playback", ("playback", "track", "tracks", "music", "pad", "pads")),
        ("keys", ("keys", "piano", "organ", "rhodes")),
        ("accordion", ("accordion", "bayan")),
        ("fx", ("fx", "reverb", "delay", "return", "riser")),
    ]
    for role, tokens in checks:
        if any(token in label for token in tokens):
            return role
    return "unknown"


def _match_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) == length:
        return audio
    if len(audio) > length:
        return audio[:length]
    pad = np.zeros((length - len(audio), audio.shape[1]), dtype=np.float32)
    return np.vstack([audio, pad])


def _sum_stems(stems: Dict[str, np.ndarray]) -> np.ndarray:
    if not stems:
        raise ValueError("Cannot synthesize a mix without stems")
    length = max(len(audio) for audio in stems.values())
    channels = max(audio.shape[1] for audio in stems.values())
    mix = np.zeros((length, channels), dtype=np.float32)
    for audio in stems.values():
        current = audio
        if current.shape[1] == 1 and channels == 2:
            current = np.repeat(current, 2, axis=1)
        elif current.shape[1] != channels:
            current = current[:, :1]
            current = np.repeat(current, channels, axis=1)
        mix += _match_length(current, length)
    peak = float(np.max(np.abs(mix)) + 1e-12)
    if peak > 1.0:
        mix = mix / peak * 0.98
    return mix.astype(np.float32)


def load_audio_context(
    stems_path: str | None = None,
    mix_path: str | None = None,
    reference_path: str | None = None,
    genre: str = "neutral",
    target_platform: str = "streaming",
) -> LoadedAudioContext:
    """Load stems, an optional stereo mix and an optional reference track.

    If no mix path is supplied, a temporary analysis mix is synthesized from the
    stems.  This is analysis context only and is not treated as final loudness.
    """
    limitations: list[str] = []
    stems: Dict[str, np.ndarray] = {}
    stem_info: Dict[str, AudioStem] = {}
    sample_rate = 0

    if stems_path:
        root = Path(stems_path).expanduser()
        for path in _audio_files(root):
            audio, sr = _read_audio(path)
            if sample_rate and sr != sample_rate:
                limitations.append(f"Stem {path.name} has sample rate {sr}; expected {sample_rate}.")
                continue
            sample_rate = sample_rate or sr
            key = path.stem
            stems[key] = audio
            stem_info[key] = AudioStem(
                name=key,
                role=infer_stem_role(key),
                path=str(path),
                sample_rate=sr,
                duration_sec=len(audio) / sr,
            )

    mix_audio: Optional[np.ndarray] = None
    if mix_path:
        mix_audio, mix_sr = _read_audio(Path(mix_path).expanduser())
        if sample_rate and mix_sr != sample_rate:
            raise ValueError(f"Mix sample rate {mix_sr} does not match stems {sample_rate}")
        sample_rate = sample_rate or mix_sr
    elif stems:
        mix_audio = _sum_stems(stems)
        limitations.append("No stereo mix supplied; analysis mix was synthesized from stems.")
    else:
        raise ValueError("Provide at least --mix or --stems")

    reference_audio = None
    reference_sr = 0
    if reference_path:
        reference_audio, reference_sr = _read_audio(Path(reference_path).expanduser())
        if sample_rate and reference_sr != sample_rate:
            limitations.append(
                f"Reference sample rate {reference_sr} differs from context {sample_rate}; "
                "comparison uses raw samples without resampling."
            )

    context = AnalysisContext(
        stems_path=stems_path or "",
        mix_path=mix_path or "",
        reference_path=reference_path or "",
        genre=genre or "neutral",
        target_platform=target_platform or "streaming",
        sample_rate=sample_rate,
        mode="offline",
        limitations=limitations,
    )
    return LoadedAudioContext(
        context=context,
        mix_audio=mix_audio,
        mix_sample_rate=sample_rate,
        stems=stems,
        stem_info=stem_info,
        reference_audio=reference_audio,
        reference_sample_rate=reference_sr,
    )
