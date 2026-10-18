"""Frozen STUDIO session adapter for the delivered ``Belye Stai`` DSP mix.

The original delivery script was procedural and tied to one container path. This
module exposes the post-local-track-processing stage as an explicit immutable
session boundary and reruns every downstream dependency: drum bus, source-driven
masking/automation, sends/returns, shared mix glue, common edge/trim and the
accepted bounded drum-accent refinement.

This is conventional DSP. It does not infer, separate, generate, tune or time-
correct audio. The adapter is song/version specific on purpose: changing recipe
constants requires a new version and a new reproduction proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import soundfile as sf
from numba import njit
from scipy import ndimage, signal

from .routed_contribution import SessionRender


EXPECTED_TRACKS = (
    "KICK", "SNARE", "TOM_1", "TOM_2", "FLOOR", "OH", "HI_HAT",
    "BASS", "GTR", "KEYS", "PLAYBACK", "VALERA_VOX", "NIKITA_VOX",
)
STEREO_TRACKS = frozenset({"OH", "KEYS", "PLAYBACK"})


@dataclass(frozen=True)
class BelyeStaiRecipe:
    sample_rate: int = 44100
    frames: int = 9_128_700
    accent_center_s: float = 159.056
    accent_sigma_s: float = 0.055
    accent_reduction_db: float = 3.0
    delivery_dither_seed: int = 20_260_924


ReverbBackend = Callable[[np.ndarray, int, float, float], np.ndarray]


def _db(x: np.ndarray | float) -> np.ndarray | float:
    return 20 * np.log10(np.maximum(x, 1e-12))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-30))


def _gain(x: np.ndarray, db_value: np.ndarray | float) -> np.ndarray:
    gain = np.power(10, np.asarray(db_value) / 20).astype(np.float32)
    return (x * (gain[:, None] if np.ndim(gain) and x.ndim == 2 else gain)).astype(np.float32)


def _frame_db(x: np.ndarray, sr: int, hop_s: float = .01) -> np.ndarray:
    hop = max(1, round(sr * hop_s))
    usable = len(x) // hop * hop
    if usable == 0:
        raise ValueError("Belye Stai session stage is too short for frame analysis")
    frames = x[:usable].reshape(-1, hop, *x.shape[1:])
    power = np.mean(frames.astype(np.float64) ** 2, axis=1)
    if power.ndim > 1:
        power = np.mean(power, axis=1)
    return 10 * np.log10(np.maximum(power, 1e-24))


def _curve(values: np.ndarray, hop_s: float, sr: int, n: int) -> np.ndarray:
    return np.interp(
        np.arange(n) / sr,
        (np.arange(len(values)) + .5) * hop_s,
        values,
    ).astype(np.float32)


def _filter(x: np.ndarray, sr: int, low: float | None = None,
            high: float | None = None, order: int = 2) -> np.ndarray:
    y = x
    if low:
        y = signal.sosfilt(
            signal.butter(order, low, btype="highpass", fs=sr, output="sos"),
            y, axis=0,
        )
    if high:
        y = signal.sosfilt(
            signal.butter(order, high, btype="lowpass", fs=sr, output="sos"),
            y, axis=0,
        )
    return np.asarray(y, dtype=np.float32)


def _band(x: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    return signal.sosfiltfilt(
        signal.butter(2, [low, high], btype="bandpass", fs=sr, output="sos"),
        x, axis=0,
    ).astype(np.float32)


def _pan(x: np.ndarray, position: float) -> np.ndarray:
    theta = (position + 1) * np.pi / 4
    return np.column_stack([x * np.cos(theta), x * np.sin(theta)]).astype(np.float32)


def _active_level(x: np.ndarray, sr: int, percentile: float = 65) -> float:
    level = _frame_db(x, sr, .02)
    threshold = max(
        float(np.percentile(level, percentile)),
        float(np.max(level)) - 35,
    )
    active = level >= threshold
    return float(10 * np.log10(np.mean(10 ** (level[active] / 10))))


def _level(x: np.ndarray, sr: int, target: float,
           percentile: float = 65) -> np.ndarray:
    before = _active_level(x, sr, percentile)
    adjustment = float(np.clip(target - before, -30, 30))
    return _gain(x, adjustment)


@njit(cache=True)
def _smooth_gr(request: np.ndarray, sr: int, attack_ms: float,
               release_ms: float) -> np.ndarray:
    attack = np.exp(-1 / (sr * attack_ms / 1000))
    release = np.exp(-1 / (sr * release_ms / 1000))
    output = np.empty(len(request), np.float32)
    previous = 0.0
    for i in range(len(request)):
        coefficient = attack if request[i] > previous else release
        previous = coefficient * previous + (1 - coefficient) * request[i]
        output[i] = previous
    return output


def _recipe_compressor(x: np.ndarray, sr: int, desired: float, ratio: float,
                       attack_ms: float, release_ms: float, cap_db: float,
                       knee_db: float = 5.) -> tuple[np.ndarray, np.ndarray]:
    power = x.astype(np.float64) ** 2
    if power.ndim == 2:
        power = power.mean(1)
    alpha = np.exp(-1 / (sr * .003))
    power = signal.lfilter([1 - alpha], [1, -alpha], power)
    envelope = 10 * np.log10(np.maximum(power, 1e-20))
    step = max(1, round(sr * .01))
    sampled = envelope[::step]
    active = sampled[sampled >= np.percentile(sampled, 60)]
    threshold = float(
        np.percentile(active, 90) - desired / (1 - 1 / ratio)
    )
    over = envelope - threshold
    request = np.where(
        over <= -knee_db / 2,
        0,
        np.where(
            over >= knee_db / 2,
            over * (1 - 1 / ratio),
            ((over + knee_db / 2) ** 2) / (2 * knee_db) * (1 - 1 / ratio),
        ),
    )
    request = np.clip(request, 0, cap_db)
    gr = _smooth_gr(request, sr, attack_ms, release_ms)
    return _gain(x, -gr), gr


def _pedalboard_reverb(send: np.ndarray, sr: int, room_size: float,
                       damping: float) -> np.ndarray:
    from pedalboard import Reverb

    processor = Reverb(
        room_size=room_size,
        damping=damping,
        wet_level=1.,
        dry_level=0.,
        width=1.,
        freeze_mode=0.,
    )
    return processor(send.T.copy(), sr, reset=True).T.astype(np.float32)


def _reverb_aux(inp: np.ndarray, *, sr: int, n: int, room_size: float,
                damping: float, predelay_ms: float, low: float, high: float,
                relative_rms_db: float, backend: ReverbBackend) -> np.ndarray:
    predelay = round(predelay_ms * sr / 1000)
    send = np.pad(inp, ((predelay, 0), (0, 0)))[:n]
    send = _filter(send, sr, low, high)
    wet = np.asarray(backend(send, sr, room_size, damping), dtype=np.float32)
    if wet.shape != inp.shape or not np.isfinite(wet).all():
        raise ValueError("reverb backend returned invalid PCM")
    wet = _filter(wet, sr, low, high)
    factor = (
        10 ** (relative_rms_db / 20)
        * _rms(inp)
        / max(_rms(wet), 1e-10)
    )
    return (wet * np.float32(factor)).astype(np.float32)


def _validate_track(name: str, x: np.ndarray,
                    recipe: BelyeStaiRecipe) -> np.ndarray:
    audio = np.asarray(x)
    if audio.dtype.kind != "f" or not np.isfinite(audio).all():
        raise ValueError(f"{name} must be finite floating-point PCM")
    expected_channels = 2 if name in STEREO_TRACKS else 1
    if expected_channels == 1 and audio.ndim != 1:
        raise ValueError(f"{name} must be mono")
    if expected_channels == 2 and (audio.ndim != 2 or audio.shape[1] != 2):
        raise ValueError(f"{name} must be stereo")
    if len(audio) != recipe.frames:
        raise ValueError(f"{name} frame count differs from frozen recipe")
    return audio.astype(np.float32, copy=True)


class BelyeStaiSessionRenderer:
    """Reusable downstream renderer for the delivered Belye Stai DSP recipe.

    Overrides are *processed-track* inserts, after the original local EQ/dynamics/
    level stage and before dependent group/routing/effect processing. Raw-track
    replacement is intentionally not claimed here.
    """

    def __init__(self, processed_tracks: Mapping[str, np.ndarray], *,
                 recipe: BelyeStaiRecipe | None = None,
                 reverb_backend: ReverbBackend | None = None):
        self.recipe = recipe or BelyeStaiRecipe()
        if (
            isinstance(self.recipe.sample_rate, bool)
            or self.recipe.sample_rate < 16_000
        ):
            raise ValueError(
                "recipe sample rate is invalid for the frozen filter graph"
            )
        if self.recipe.frames <= 0:
            raise ValueError("recipe frames must be positive")
        missing = [
            name for name in EXPECTED_TRACKS if name not in processed_tracks
        ]
        extra = [
            name for name in processed_tracks if name not in EXPECTED_TRACKS
        ]
        if missing or extra:
            raise ValueError(
                f"processed-track set mismatch: missing={missing}, extra={extra}"
            )
        self._tracks = {
            name: _validate_track(name, processed_tracks[name], self.recipe)
            for name in EXPECTED_TRACKS
        }
        for audio in self._tracks.values():
            audio.flags.writeable = False
        self._reverb_backend = reverb_backend or _pedalboard_reverb

    @classmethod
    def from_processed_directory(
        cls,
        directory: str | Path,
        *,
        recipe: BelyeStaiRecipe | None = None,
        reverb_backend: ReverbBackend | None = None,
    ) -> "BelyeStaiSessionRenderer":
        recipe = recipe or BelyeStaiRecipe()
        root = Path(directory)
        tracks: dict[str, np.ndarray] = {}
        for name in EXPECTED_TRACKS:
            audio, sr = sf.read(root / f"{name}.wav", dtype="float32")
            if sr != recipe.sample_rate:
                raise ValueError(
                    f"{name} sample rate differs from frozen recipe"
                )
            tracks[name] = audio
        return cls(
            tracks,
            recipe=recipe,
            reverb_backend=reverb_backend,
        )

    def _working_tracks(
        self, overrides: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        unknown = [name for name in overrides if name not in EXPECTED_TRACKS]
        if unknown:
            raise ValueError(
                f"unknown Belye Stai processed-track override(s): {unknown}"
            )
        tracks = {name: audio for name, audio in self._tracks.items()}
        for name, value in overrides.items():
            tracks[name] = _validate_track(name, value, self.recipe)
        return tracks

    def __call__(self, overrides: Mapping[str, np.ndarray]) -> SessionRender:
        tracks = self._working_tracks(overrides)
        sr = self.recipe.sample_rate
        n = self.recipe.frames

        drums = _pan(tracks["KICK"], 0) + _pan(tracks["SNARE"], 0)
        drum_send = _pan(tracks["SNARE"], 0)
        for name, position in (
            ("TOM_1", -.45),
            ("TOM_2", .12),
            ("FLOOR", .52),
        ):
            panned = _pan(tracks[name], position)
            drums += panned
            drum_send += panned * .7
        drums += tracks["OH"]
        drums += _pan(tracks["HI_HAT"], -.45)
        drums, _ = _recipe_compressor(
            drums, sr, 1., 1.5, 30, 160, 1.6
        )
        drums = _level(drums, sr, -25.5, 65)

        buses: dict[str, np.ndarray] = {
            "DRUMS": drums,
            "BASS": _pan(tracks["BASS"], 0),
            "GUITAR": _pan(tracks["GTR"], -.23),
            "KEYS": tracks["KEYS"].copy(),
            "PLAYBACK": tracks["PLAYBACK"].copy(),
            "LEAD_VOX": _pan(tracks["VALERA_VOX"], 0.),
            "BACK_VOX": _pan(tracks["NIKITA_VOX"], .12),
        }

        vocal = buses["LEAD_VOX"]
        vocal_detector = _frame_db(
            _band(vocal, sr, 250, 3500), sr, .05
        )
        vocal_threshold = max(
            float(np.percentile(vocal_detector, 18)) + 10,
            float(np.percentile(vocal_detector, 75)) - 16,
        )
        activity = np.clip(
            (vocal_detector - vocal_threshold) / 6, 0, 1
        )
        activity = ndimage.gaussian_filter1d(activity, 4)
        full_activity = _curve(activity, .05, sr, n)
        buses["GUITAR"] = _gain(
            buses["GUITAR"], 1.2 * (1 - full_activity)
        )

        vocal_detector = _frame_db(
            _band(vocal, sr, 1400, 4200), sr, .05
        )
        for name, max_cut in (
            ("GUITAR", .9),
            ("KEYS", .7),
            ("PLAYBACK", 1.0),
        ):
            source = buses[name]
            component = _band(source, sr, 1400, 4200)
            masker = _frame_db(component, sr, .05)
            overlap = (
                np.clip((masker - vocal_detector + 7) / 8, 0, 1)
                * activity
            )
            gr = _curve(
                ndimage.gaussian_filter1d(overlap * max_cut, 3),
                .05,
                sr,
                n,
            )
            candidate = (
                source
                + component * (10 ** (-gr[:, None] / 20) - 1)
            )
            delta = float(_db(_rms(candidate)) - _db(_rms(source)))
            if abs(delta) < .6:
                buses[name] = candidate.astype(np.float32)

        bass = buses["BASS"]
        low = _band(bass, sr, 35, 115)
        kick_detector = _frame_db(tracks["KICK"], sr, .01)
        kick_activity = np.clip(
            (kick_detector - np.percentile(kick_detector, 80)) / 12,
            0,
            1,
        )
        bass_gr = _curve(
            ndimage.gaussian_filter1d(kick_activity, 3) * 1.2,
            .01,
            sr,
            n,
        )
        buses["BASS"] = (
            bass + low * (10 ** (-bass_gr[:, None] / 20) - 1)
        ).astype(np.float32)

        room = _reverb_aux(
            drum_send,
            sr=sr,
            n=n,
            room_size=.32,
            damping=.65,
            predelay_ms=12,
            low=220,
            high=7000,
            relative_rms_db=-20.,
            backend=self._reverb_backend,
        )
        chamber = _reverb_aux(
            buses["LEAD_VOX"] + buses["BACK_VOX"] * .4,
            sr=sr,
            n=n,
            room_size=.47,
            damping=.68,
            predelay_ms=29,
            low=230,
            high=7800,
            relative_rms_db=-19.5,
            backend=self._reverb_backend,
        )
        vocal_filtered = _filter(
            buses["LEAD_VOX"], sr, 300, 4800
        )
        slap = np.zeros_like(vocal_filtered)
        lag = round(.112 * sr)
        if lag < n:
            slap[lag:] = (
                vocal_filtered[:-lag] * 10 ** (-27 / 20)
            )
        buses["FX"] = (room + chamber + slap).astype(np.float32)

        mix = np.zeros((n, 2), np.float64)
        for bus in buses.values():
            mix += bus
        mix, mix_gr = _recipe_compressor(
            mix.astype(np.float32), sr, .65, 1.4, 35, 230, 1.25
        )
        for name in tuple(buses):
            buses[name] = _gain(buses[name], -mix_gr)
        room = _gain(room, -mix_gr)

        edge = np.ones(n, np.float32)
        fade_in = min(221, n)
        if fade_in:
            edge[:fade_in] = np.linspace(0, 1, fade_in)
        fade_out = min(22050, n)
        if fade_out:
            edge[-fade_out:] = np.linspace(1, 0, fade_out)
        mix *= edge[:, None]
        trim = min(0., -4 - float(_db(np.max(abs(mix)))))
        mix = _gain(mix, trim)
        for name in tuple(buses):
            buses[name] = _gain(buses[name], trim) * edge[:, None]
        room = _gain(room, trim) * edge[:, None]

        # Frozen post-mix accent refinement from the delivered project. The same
        # final DRUMS bus is subtracted/added, so routing remains sample-aligned.
        time = np.arange(n) / sr
        accent_gr = self.recipe.accent_reduction_db * np.exp(
            -.5
            * (
                (time - self.recipe.accent_center_s)
                / self.recipe.accent_sigma_s
            ) ** 2
        )
        new_drums = (
            buses["DRUMS"]
            * np.power(10, -accent_gr[:, None] / 20).astype(np.float32)
        )
        mix = (mix + (new_drums - buses["DRUMS"])).astype(np.float32)
        buses["DRUMS"] = new_drums.astype(np.float32)

        return SessionRender(
            mix=mix,
            vocal_bus=buses["LEAD_VOX"],
            drums_bus=buses["DRUMS"],
            early_room=room.astype(np.float32),
            metadata={
                "session": (
                    "Belye_Stai_DSP_Mix_v1_plus_accent_refinement"
                ),
                "adapter_schema": "belye-stai-session-renderer-v1",
                "override_stage": (
                    "post_local_track_processing_pre_context"
                ),
                "override_ids": sorted(overrides),
                "sample_rate": sr,
                "frames": n,
                "common_trim_db": float(trim),
                "accent_refinement_applied": True,
                "baseline_promoted": False,
                "requires_human_listening": True,
            },
        )


def export_delivery_pcm24(
    path: str | Path,
    mix: np.ndarray,
    recipe: BelyeStaiRecipe | None = None,
) -> None:
    """Write the frozen delivery quantizer used by final accent refinement."""
    recipe = recipe or BelyeStaiRecipe()
    audio = np.asarray(mix, dtype=np.float32)
    if audio.shape != (recipe.frames, 2) or not np.isfinite(audio).all():
        raise ValueError(
            "delivery PCM must match the frozen stereo session timeline"
        )
    rng = np.random.default_rng(recipe.delivery_dither_seed)
    dither = (
        rng.random(audio.shape) - rng.random(audio.shape)
    ) / 2 ** 24
    sf.write(
        path,
        audio + dither,
        recipe.sample_rate,
        subtype="PCM_24",
    )
