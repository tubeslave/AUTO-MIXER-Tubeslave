#!/usr/bin/env python3
"""
Спектральное сведение v2: многозонные характерные частоты каждого инструмента,
целевая кривая −4.5 дБ/окт, иерархия владения полосами, динамический EQ.

Каждый инструмент описан набором «спектральных зон»:
  - essential: частоты, которые НУЖНО сохранить/подчеркнуть (punch kick, crack snare…)
  - cut:       частоты, которые НУЖНО убрать (mud, boxiness, ring…)
  - secondary: второстепенные зоны — убираются, если мешают другим

Алгоритм (на каждый STFT-фрейм):
  1. Энергия каждого стема в 28 третьоктавных полосах.
  2. Для каждой полосы — кто «владелец» (по иерархии + essential-зоне).
  3. Если инструмент лезет в чужую essential-зону — срезать.
  4. Если у инструмента CUT-зона — всегда срезать.
  5. Составной спектр подгоняется к целевой кривой −4.5 дБ/окт.
  6. Gain-коррекции через STFT overlap-add.
  7. RMS-автоматизация уровней по каждому стему (не sidechain-ducking).
     Динамическое приглушение одних дорожек от других в музыке не применяется;
     ducking в смысле FOH — только для речи (вне этого скрипта).

Источники:
  pointprimerecordings.com — Kick Drum EQ Guide
  musicguymixing.com — Snare, Tom, Hi-Hat, Ride, Cymbal, Guitar, Backing Vocal EQ
  drummagazine.com — How to EQ Drums
  producerhive.com — Vocal EQ Chart
  demomentor.com — Vocal EQ for Clarity
  theguitarside.com — Guitar EQ Cheat Sheet
  mixinglessons.com — Drum EQ guide, EQ Electric Guitars
  musicproductionnerds.com — Ride Cymbal EQ
  musical-u.com — Cymbal Frequencies
  emastered.com — Hi-Hat EQ
  sonible.com — Spectral Balance
  mastering.com — Range Allocation
  Myk Eff — Spectrum Analyzer Slopes (−4.5 dB/oct)
  Mike Senior — Mixing Secrets (range allocation, subtractive EQ)
  Roey Izhaki — Mixing Audio (EQ, masking, hierarchy)

Зависимости: numpy, soundfile, scipy.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import get_window

# ─────────────────────────────────────────────────────────────
# 1. Spectral zone descriptors
# ─────────────────────────────────────────────────────────────

class Zone(NamedTuple):
    lo: float
    hi: float
    role: str          # "essential" | "cut" | "secondary" | "conditional_cut"
    weight: float      # strength: 1.0 = normal, >1 = more aggressive


INSTRUMENT_ZONES: Dict[str, List[Zone]] = {
    # ── KICK ──
    # Sources: pointprimerecordings.com, musicguymixing.com, drummagazine.com
    "kick": [
        Zone(40, 65, "essential", 1.0),     # sub punch
        Zone(60, 120, "essential", 1.2),     # body / thump (главная зона)
        Zone(200, 400, "cut", 1.0),          # mud / boxiness
        Zone(2500, 5000, "essential", 0.8),  # beater click / snap
    ],
    # ── SNARE ──
    # Sources: musicguymixing.com, politusic.com, soundshockaudio.com
    # crack/attack (2–5 kHz) — secondary, уступает вокалу в этом диапазоне
    "snare": [
        Zone(150, 250, "essential", 1.0),    # body / fatness
        Zone(400, 600, "cut", 0.8),          # boxiness
        Zone(600, 900, "cut", 0.6),          # ring
        Zone(2000, 5000, "secondary", 0.5),  # crack — уступить вокалу
        Zone(6000, 10000, "secondary", 0.4), # shimmer / texture
    ],
    # ── FLOOR TOM ──
    # Sources: productlondon.com, musicguymixing.com, drummagazine.com
    "floor_tom": [
        Zone(60, 100, "essential", 1.0),     # low body
        Zone(200, 400, "cut", 0.7),          # ring / boxiness
        Zone(4000, 6000, "essential", 0.7),  # stick click
    ],
    # ── MID/RACK TOM ──
    "mid_tom": [
        Zone(100, 250, "essential", 1.0),    # body / fullness
        Zone(300, 600, "cut", 0.7),          # boxiness
        Zone(4000, 7000, "essential", 0.7),  # attack click
    ],
    # ── HI-HAT ──
    # Sources: emastered.com, musicguymixing.com, mixinglessons.com
    "hihat": [
        Zone(200, 500, "cut", 0.8),          # mud / bleed cleanup
        Zone(6000, 9000, "essential", 1.0),  # shimmer / sizzle
        Zone(10000, 14000, "essential", 0.8), # air
        Zone(3000, 6000, "secondary", 0.5),  # crispness (уступает вокалу)
    ],
    # ── RIDE ──
    # Sources: musicproductionnerds.com, musicguymixing.com, musical-u.com
    "ride": [
        Zone(300, 800, "secondary", 0.4),    # body (уступает гитаре/вокалу)
        Zone(3000, 5000, "cut", 0.6),        # harshness
        Zone(8000, 12000, "essential", 1.0),  # sizzle / shimmer
    ],
    # ── OVERHEAD ──
    # Sources: soundshockaudio.com, drumsbydavid.com
    "overhead": [
        Zone(50, 250, "cut", 0.9),           # bleed / mud from kick
        Zone(4000, 8000, "essential", 0.7),   # cymbal presence
        Zone(8000, 16000, "essential", 0.9),  # air / shimmer
    ],
    # ── DRUM ROOM ──
    # Sources: audioproductionroom.com, mikesmixmaster.com
    "room": [
        Zone(50, 150, "cut", 0.8),           # rumble
        Zone(200, 600, "secondary", 0.3),     # ambience body (не мешать)
        Zone(3000, 5000, "secondary", 0.4),   # presence
        Zone(7000, 10000, "essential", 0.5),  # air / ambience shimmer
    ],
    # ── LEAD VOCAL ──
    # Sources: producerhive.com, demomentor.com, unison.audio
    # Вокал — приоритет #0, должен лидировать в широком речевом коридоре.
    "lead_vocal": [
        Zone(80, 120, "cut", 0.7),           # rumble / proximity (мягче)
        Zone(150, 350, "essential", 0.9),     # warmth / body
        Zone(300, 1250, "essential", 1.0),    # articulation / vocal body
        Zone(800, 1200, "conditional_cut", 0.35),  # honk — только если реально выпирает
        Zone(1500, 5000, "essential", 1.6),   # presence / intelligibility (ГЛАВНАЯ)
        Zone(5000, 7000, "secondary", 0.3),   # sibilance — мягкое ослабление, не рубить
        Zone(8000, 12000, "essential", 0.9),  # air / sheen
    ],
    # ── BACKING VOCAL ──
    # Sources: musicguymixing.com (backing vocal EQ, mix background vocals)
    "backing_vocal": [
        Zone(80, 300, "cut", 1.0),           # body — уступить лид-вокалу
        Zone(400, 1500, "essential", 0.6),    # гармоническое ядро (сужена от presence)
        Zone(1500, 5000, "secondary", 0.7),   # presence — УСТУПИТЬ лид-вокалу
        Zone(5500, 16000, "cut", 0.6),        # bite/air — зарезервировать для лида
    ],
    # ── ELECTRIC GUITAR ──
    # Sources: theguitarside.com, producerhive.com, musicguymixing.com
    # В зоне 2–5 кГц гитара УСТУПАЕТ вокалу (secondary) — вокал приоритет #0
    "guitar": [
        Zone(80, 150, "cut", 1.0),           # mud / rumble
        Zone(200, 500, "essential", 0.9),     # body / fullness (главная зона гитары)
        Zone(800, 2000, "essential", 0.7),    # mid-range bite (ниже вокал-presence)
        Zone(2000, 5000, "secondary", 0.6),   # presence — УСТУПИТЬ вокалу
        Zone(5000, 8000, "secondary", 0.4),   # harshness / fizz
        Zone(8000, 16000, "cut", 0.5),        # fizz верхний
    ],
    # ── ACCORDION ──
    # Sources: maciejewski.com (spectral analysis), practical-music-production.com
    # В зоне вокального presence (2–5 кГц) — уступает
    "accordion": [
        Zone(80, 200, "cut", 0.7),           # не нужный низ
        Zone(300, 800, "essential", 0.8),     # body reed тембр (расширена вниз)
        Zone(800, 1500, "essential", 0.7),    # mid reed harmonics
        Zone(1500, 5000, "secondary", 0.6),   # presence — УСТУПИТЬ вокалу
        Zone(5000, 10000, "secondary", 0.4),  # air — уступает
    ],
    # ── PLAYBACK (backing track) ──
    # Sources: mixingmusiclive.com, practical-music-production.com
    "playback": [
        Zone(30, 80, "cut", 0.9),            # sub — уступить kick
        Zone(100, 250, "secondary", 0.3),     # body — не мешать бас-группе
        Zone(250, 1000, "secondary", 0.3),    # mid — заполнение, не лидерство
        Zone(1000, 5000, "secondary", 0.3),   # presence — уступить вокалу
        Zone(5000, 12000, "secondary", 0.3),  # air — заполнение
    ],
}

# Вариант лид-вокала для «живой площадки»: меньше аналитического push в ВЧ,
# presence чуть мягче, верх — secondary вместо essential.
LEAD_VOCAL_ZONES_LIVE: List[Zone] = [
    Zone(80, 120, "cut", 0.65),
    Zone(150, 350, "essential", 0.95),
    Zone(300, 1250, "essential", 1.05),
    Zone(800, 1200, "conditional_cut", 0.3),
    Zone(1500, 5000, "essential", 1.15),
    Zone(5000, 8000, "secondary", 0.45),
    Zone(8000, 12000, "secondary", 0.5),
]


def instrument_zones_for_mode(foh_live: bool) -> Dict[str, List[Zone]]:
    z = {k: list(v) for k, v in INSTRUMENT_ZONES.items()}
    if not foh_live:
        return z
    z["lead_vocal"] = list(LEAD_VOCAL_ZONES_LIVE)
    z["kick"] = [
        Zone(40, 65, "essential", 1.05),
        Zone(60, 120, "essential", 1.35),
        Zone(200, 400, "cut", 1.0),
        Zone(2500, 5000, "essential", 0.85),
    ]
    z["snare"] = [
        Zone(150, 250, "essential", 1.1),
        Zone(400, 600, "cut", 0.8),
        Zone(600, 900, "cut", 0.6),
        Zone(2000, 5000, "secondary", 0.45),
        Zone(6000, 10000, "secondary", 0.45),
    ]
    z["floor_tom"] = [
        Zone(60, 100, "essential", 1.08),
        Zone(200, 400, "cut", 0.7),
        Zone(4000, 6000, "essential", 0.78),
    ]
    z["mid_tom"] = [
        Zone(100, 250, "essential", 1.08),
        Zone(300, 600, "cut", 0.7),
        Zone(4000, 7000, "essential", 0.78),
    ]
    return z


# ─────────────────────────────────────────────────────────────
# 2. Stem map: file → instrument type + mix settings
# ─────────────────────────────────────────────────────────────

class StemInfo(NamedTuple):
    instrument: str
    hierarchy: int    # 0 = highest priority
    pan: float        # -100..100
    base_db: float    # initial fader offset


STEM_MAP: Dict[str, StemInfo] = {
    "Vox.wav":          StemInfo("lead_vocal",    0,    0.0,   6.0),
    "Vox1.wav":         StemInfo("lead_vocal",    0,    0.0,   6.0),
    "Kick.wav":         StemInfo("kick",           1,    0.0,   0.0),
    "Snare.wav":        StemInfo("snare",          2,    5.0,  -1.0),
    "Floor Tom.wav":    StemInfo("floor_tom",      3,  -25.0,  -1.5),
    "Mid Tom.wav":      StemInfo("mid_tom",        3,   22.0,  -1.5),
    "Hi-Hat.wav":       StemInfo("hihat",          4,  -40.0,  -3.5),
    "Ride.wav":         StemInfo("ride",           4,   38.0,  -3.5),
    "Overhead L.wav":   StemInfo("overhead",       4, -100.0,  -4.5),
    "Overhead R.wav":   StemInfo("overhead",       4,  100.0,  -4.5),
    "Drum Room L.wav":  StemInfo("room",           4,  -95.0,  -7.0),
    "Drum Room R.wav":  StemInfo("room",           4,   95.0,  -7.0),
    "Guitar L.wav":     StemInfo("guitar",         5,  -95.0,  -2.5),
    "Guitar R.wav":     StemInfo("guitar",         5,   95.0,  -2.5),
    "Accordion.wav":    StemInfo("accordion",      6,    0.0,  -3.0),
    "Vox2.wav":         StemInfo("backing_vocal",  6,  -20.0,  -1.0),
    "Vox3.wav":         StemInfo("backing_vocal",  6,   20.0,  -1.0),
    "Playback L.wav":   StemInfo("playback",       7,  -90.0,  -6.5),
    "Playback R.wav":   StemInfo("playback",       7,   90.0,  -6.5),
}

VOCAL_EXCLUSIVE_GROUPS = [
    ("Vox.wav", "Vox1.wav"),
]


def resolve_active_stems(listing: set[str]) -> List[str]:
    active = set()
    for group in VOCAL_EXCLUSIVE_GROUPS:
        present = [name for name in group if name in listing]
        if present:
            active.add(present[0])
    for name in STEM_MAP:
        if name in listing and all(name not in group for group in VOCAL_EXCLUSIVE_GROUPS):
            active.add(name)
    return sorted(active)


def select_lead_stem(stem_names: List[str]) -> str | None:
    for candidate in ("Vox.wav", "Vox1.wav"):
        if candidate in stem_names:
            return candidate
    return None


# ─────────────────────────────────────────────────────────────
# 3. Third-octave bands (ISO 266 subset, 31.5 Hz – 16 kHz)
# ─────────────────────────────────────────────────────────────

BAND_CENTERS = np.array([
    31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
    3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000,
], dtype=np.float64)

N_BANDS = len(BAND_CENTERS)
_FACTOR = 2.0 ** (1.0 / 6.0)
BAND_LO = BAND_CENTERS / _FACTOR
BAND_HI = BAND_CENTERS * _FACTOR


def target_curve_db(centers: np.ndarray, slope: float = -4.5,
                    ref_hz: float = 1000.0, ref_db: float = -20.0) -> np.ndarray:
    return ref_db + slope * np.log2(centers / ref_hz)


# ─────────────────────────────────────────────────────────────
# 4. Pre-compute zone masks per instrument per band
# ─────────────────────────────────────────────────────────────

def build_zone_band_lookup(
    zones: List[Zone], band_centers: np.ndarray
) -> Dict[str, np.ndarray]:
    """For each zone role, return array of (n_bands,) weights.
    If a band center falls in the zone → weight, else 0."""
    out: Dict[str, np.ndarray] = {
        "essential": np.zeros(len(band_centers)),
        "cut": np.zeros(len(band_centers)),
        "secondary": np.zeros(len(band_centers)),
        "conditional_cut": np.zeros(len(band_centers)),
    }
    for z in zones:
        mask = (band_centers >= z.lo) & (band_centers <= z.hi)
        existing = out[z.role]
        existing[mask] = np.maximum(existing[mask], z.weight)
    return out


# ─────────────────────────────────────────────────────────────
# 5. STFT helpers
# ─────────────────────────────────────────────────────────────

def stft_analyze(x: np.ndarray, fft_size: int, hop: int,
                 window: np.ndarray) -> np.ndarray:
    pad_l = fft_size // 2
    pad_r = fft_size // 2 + (fft_size - (len(x) % hop)) % fft_size
    xp = np.pad(x, (pad_l, pad_r), mode="constant")
    n_frames = (len(xp) - fft_size) // hop + 1
    frames = np.lib.stride_tricks.as_strided(
        xp, shape=(n_frames, fft_size),
        strides=(xp.strides[0] * hop, xp.strides[0]),
    ).copy()
    frames *= window
    return np.fft.rfft(frames, n=fft_size, axis=1)


def stft_synthesize(spec: np.ndarray, fft_size: int, hop: int,
                    window: np.ndarray, out_len: int) -> np.ndarray:
    n_frames = spec.shape[0]
    frames = np.fft.irfft(spec, n=fft_size, axis=1)
    frames *= window
    total = fft_size + (n_frames - 1) * hop
    out = np.zeros(total, dtype=np.float64)
    w2 = np.zeros(total, dtype=np.float64)
    for i in range(n_frames):
        s = i * hop
        out[s:s + fft_size] += frames[i]
        w2[s:s + fft_size] += window ** 2
    w2 = np.maximum(w2, 1e-8)
    out /= w2
    off = fft_size // 2
    return out[off:off + out_len]


def build_band_masks(freqs: np.ndarray, band_lo: np.ndarray,
                     band_hi: np.ndarray) -> np.ndarray:
    masks = np.zeros((len(band_lo), len(freqs)), dtype=np.float64)
    for b in range(len(band_lo)):
        m = (freqs >= band_lo[b]) & (freqs < band_hi[b])
        masks[b, m] = 1.0
    return masks


def band_energy(mag: np.ndarray, bmasks: np.ndarray) -> np.ndarray:
    return bmasks.dot(mag ** 2)


# ─────────────────────────────────────────────────────────────
# 6. Per-frame spectral gain engine (v2: multi-zone)
# ─────────────────────────────────────────────────────────────

def compute_gains_v2(
    stem_energies: Dict[str, np.ndarray],
    target_db: np.ndarray,
    stem_infos: Dict[str, StemInfo],
    zone_lookups: Dict[str, Dict[str, np.ndarray]],
    max_boost: float = 6.0,
    max_cut: float = -12.0,
    hierarchy_adv: float = 4.5,
) -> Dict[str, np.ndarray]:
    eps = 1e-12
    names = list(stem_energies.keys())
    nb = len(target_db)

    composite = np.zeros(nb)
    for n in names:
        composite += stem_energies[n]
    comp_db = 10.0 * np.log10(composite + eps)
    delta = target_db - comp_db

    gains: Dict[str, np.ndarray] = {n: np.zeros(nb) for n in names}

    # Vocal energy detection: is the lead vocal active in this frame?
    vox_names = [n for n in names if stem_infos[n].instrument == "lead_vocal"]
    vox_total_e = sum(np.sum(stem_energies[n]) for n in vox_names)
    vox_active = vox_total_e > 1e-8
    vocal_protect_mask = (BAND_CENTERS >= 315.0) & (BAND_CENTERS <= 5000.0)

    for b in range(nb):
        active = [(n, stem_energies[n][b]) for n in names if stem_energies[n][b] > eps]
        if not active:
            continue

        total_e = sum(e for _, e in active)

        essential_list = []
        cut_list = []
        secondary_list = []
        conditional_cut_list = []
        neutral_list = []

        for n, e in active:
            inst = stem_infos[n].instrument
            zl = zone_lookups[inst]
            ess_w = zl["essential"][b]
            cut_w = zl["cut"][b]
            sec_w = zl["secondary"][b]
            cond_cut_w = zl["conditional_cut"][b]

            if cut_w > 0:
                cut_list.append((n, e, cut_w))
            elif cond_cut_w > 0:
                conditional_cut_list.append((n, e, cond_cut_w))
            elif ess_w > 0:
                essential_list.append((n, e, ess_w, stem_infos[n].hierarchy))
            elif sec_w > 0:
                secondary_list.append((n, e, sec_w))
            else:
                neutral_list.append((n, e))

        # 1) CUT zones — always cut
        for n, e, w in cut_list:
            share = e / (total_e + eps)
            gains[n][b] = np.clip(-3.0 * w - share * 4.0, max_cut, -0.5)

        # 1b) CONDITIONAL CUT — only when the source truly dominates the band.
        for n, e, w in conditional_cut_list:
            share = e / (total_e + eps)
            if share > 0.52:
                gains[n][b] = np.clip(-2.0 * w * (share - 0.52) * 5.0, -4.0, 0.0)

        # 2) Sort essential by hierarchy (lower = more important)
        essential_list.sort(key=lambda t: t[3])

        if essential_list:
            if vocal_protect_mask[b]:
                lead_candidates = [t for t in essential_list if stem_infos[t[0]].instrument == "lead_vocal"]
                if lead_candidates:
                    lead_name, lead_e, lead_w, lead_h = lead_candidates[0]
                    lead_share = lead_e / (total_e + eps)
                    if lead_share > 0.05 or vox_active:
                        essential_list = (
                            [(lead_name, lead_e, lead_w, lead_h)]
                            + [t for t in essential_list if t[0] != lead_name]
                        )

            owner_n, owner_e, owner_w, owner_h = essential_list[0]
            owner_share = owner_e / (total_e + eps)
            is_vocal_owner = stem_infos[owner_n].instrument == "lead_vocal"

            # Vocal gets stronger boost (it's the most important element)
            adv = hierarchy_adv * 1.7 if is_vocal_owner else hierarchy_adv
            boost = np.clip(
                delta[b] * 0.4 + adv * owner_w * (1.0 - owner_share),
                -0.5, max_boost
            )
            gains[owner_n][b] = max(gains[owner_n][b], boost)

            # Non-owner essentials get penalized harder
            for n, e, w, h in essential_list[1:]:
                share = e / (total_e + eps)
                # If vocal owns this band, others lose more
                extra = 2.0 if is_vocal_owner and vox_active else 1.0
                if share > 0.10:
                    penalty = np.clip(
                        -hierarchy_adv * 0.6 * w * extra,
                        max_cut, 0.0
                    )
                    gains[n][b] = min(
                        gains[n][b], penalty
                    ) if gains[n][b] != 0 else penalty

        # 3) Secondary zones — спектральное ослабление при маскировке
        for n, e, w in secondary_list:
            share = e / (total_e + eps)
            vox_is_owner = (essential_list
                            and stem_infos[essential_list[0][0]].instrument
                            == "lead_vocal")
            vocal_band_extra = 2.2 if vocal_protect_mask[b] and vox_active else 1.0
            extra = 1.8 * vocal_band_extra if vox_is_owner and vox_active else vocal_band_extra
            if essential_list and share > 0.06:
                gains[n][b] = np.clip(
                    -2.5 * w * share * 4.0 * extra, max_cut, -0.5
                )
            elif share > 0.25:
                gains[n][b] = np.clip(-2.0 * w * extra, max_cut, -0.5)

        # 4) Neutral (no zone defined) — cut if masking essential owner
        for n, e in neutral_list:
            share = e / (total_e + eps)
            if share > 0.15 and essential_list:
                extra = 1.8 if vocal_protect_mask[b] and vox_active else 1.0
                gains[n][b] = np.clip(-1.5 * share * 3.0 * extra, max_cut, 0.0)

    return gains


def apply_gains_to_spec(spec_frame: np.ndarray, freqs: np.ndarray,
                        gains_db: np.ndarray, band_lo: np.ndarray,
                        band_hi: np.ndarray) -> np.ndarray:
    g = np.ones(len(freqs), dtype=np.float64)
    for b in range(len(band_lo)):
        if gains_db[b] != 0.0:
            m = (freqs >= band_lo[b]) & (freqs < band_hi[b])
            g[m] = 10.0 ** (gains_db[b] / 20.0)
    return spec_frame * g


# ─────────────────────────────────────────────────────────────
# 7. Automation helpers
# ─────────────────────────────────────────────────────────────

def rms_env(x: np.ndarray, block: int) -> np.ndarray:
    nb = int(np.ceil(len(x) / block))
    env = np.zeros(nb)
    for i in range(nb):
        seg = x[i * block:min((i + 1) * block, len(x))]
        env[i] = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
    return env


def expand_env(env: np.ndarray, block: int, n: int) -> np.ndarray:
    raw = np.repeat(env, block)[:n]
    if len(raw) < n:
        raw = np.pad(raw, (0, n - len(raw)), mode="edge")
    k = max(3, block // 4)
    return np.convolve(raw, np.ones(k) / k, mode="same")


def pan_cp(pct: float, mono: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = float(np.clip(pct, -100.0, 100.0)) / 100.0
    a = (p + 1.0) * (np.pi / 4.0)
    return mono * np.cos(a), mono * np.sin(a)


# ─────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Spectral hierarchy mix v2: multi-zone instrument EQ, "
                    "-4.5 dB/oct target, dynamic corrections"
    )
    ap.add_argument("mix_dir", nargs="?",
                    default="/Users/dmitrijvolkov/Desktop/MIX")
    ap.add_argument("-o", "--output",
                    default="/Users/dmitrijvolkov/Desktop/MIX_SpectralHierarchy.wav")
    ap.add_argument("--slope", type=float, default=-4.5)
    ap.add_argument("--ref-db", type=float, default=-20.0)
    ap.add_argument("--peak-dbfs", type=float, default=-1.0)
    ap.add_argument("--fft-size", type=int, default=4096)
    ap.add_argument("--block-ms", type=float, default=50.0)
    ap.add_argument("--max-boost", type=float, default=6.0)
    ap.add_argument("--max-cut", type=float, default=-12.0)
    ap.add_argument(
        "--foh-live",
        action="store_true",
        help="FOH / live PA: плотнее ударные, теплее лид-вокал, мягче ВЧ-цель; без ducking",
    )
    args = ap.parse_args()

    if args.foh_live:
        args.slope = -3.3
        args.ref_db = -19.0

    mix_dir = os.path.abspath(args.mix_dir)
    listing = set(os.listdir(mix_dir))
    stem_names = resolve_active_stems(listing)
    if not stem_names:
        print("Нет поддерживаемых стемов в папке MIX", file=sys.stderr)
        return 1

    info = sf.info(os.path.join(mix_dir, stem_names[0]))
    sr = info.samplerate
    fft_size = args.fft_size
    hop = max(int(sr * args.block_ms / 1000.0), 256)
    window = get_window("hann", fft_size, fftbins=True).astype(np.float64)
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
    tgt_db = target_curve_db(BAND_CENTERS, slope=args.slope, ref_db=args.ref_db)
    bmasks = build_band_masks(freqs, BAND_LO, BAND_HI)

    zones_mode = instrument_zones_for_mode(args.foh_live)
    zone_lookups: Dict[str, Dict[str, np.ndarray]] = {}
    for inst, zones in zones_mode.items():
        zone_lookups[inst] = build_zone_band_lookup(zones, BAND_CENTERS)

    # ── Print zone map ──
    print("═══ Карта зон по инструментам ═══", file=sys.stderr)
    for inst, zones in zones_mode.items():
        parts = []
        for z in zones:
            tag = {
                "essential": "★",
                "cut": "✂",
                "secondary": "▽",
                "conditional_cut": "~",
            }[z.role]
            parts.append(f"{tag}{z.lo:.0f}-{z.hi:.0f}")
        print(f"  {inst:16s}: {', '.join(parts)}", file=sys.stderr)
    print(file=sys.stderr)

    # ── Load stems ──
    print("Загрузка стемов…", file=sys.stderr)
    raw: Dict[str, np.ndarray] = {}
    n_samp = None
    for name in stem_names:
        x, r = sf.read(os.path.join(mix_dir, name), dtype="float64")
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        if r != sr:
            print(f"SR mismatch: {name}", file=sys.stderr)
            return 1
        if n_samp is None:
            n_samp = len(x)
        else:
            n_samp = min(n_samp, len(x))
        raw[name] = x

    _drum_live_db = {
        "kick": 1.8,
        "snare": 1.5,
        "floor_tom": 1.2,
        "mid_tom": 1.2,
        "hihat": 0.6,
        "ride": 0.6,
        "overhead": -0.8,
        "room": -1.2,
    }
    for name in stem_names:
        raw[name] = raw[name][:n_samp]
        gain_db = STEM_MAP[name].base_db
        inst = STEM_MAP[name].instrument
        if args.foh_live:
            if inst == "lead_vocal":
                gain_db -= 2.0
            elif inst in _drum_live_db:
                gain_db += _drum_live_db[inst]
        raw[name] *= 10.0 ** (gain_db / 20.0)

    # ── STFT ──
    print("STFT-анализ…", file=sys.stderr)
    specs: Dict[str, np.ndarray] = {}
    for name in stem_names:
        specs[name] = stft_analyze(raw[name], fft_size, hop, window)
    n_frames = specs[stem_names[0]].shape[0]
    print(f"  {n_samp} семплов, {n_frames} фреймов, hop={hop}", file=sys.stderr)

    # ── Phase check (stereo pairs: report only, no auto-invert) ──
    print("Фазовая когерентность L/R пар:", file=sys.stderr)
    for la, ra in [("Overhead L.wav", "Overhead R.wav"),
                   ("Drum Room L.wav", "Drum Room R.wav"),
                   ("Guitar L.wav", "Guitar R.wav"),
                   ("Playback L.wav", "Playback R.wav")]:
        if la in raw and ra in raw:
            chunk = min(sr, n_samp)
            c = np.corrcoef(raw[la][:chunk], raw[ra][:chunk])
            corr = float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0
            print(f"  {la}/{ra}: corr={corr:+.3f}", file=sys.stderr)

    # ── Per-frame spectral correction ──
    print("Вычисление спектральных коррекций (v2: multi-zone)…", file=sys.stderr)
    corrected: Dict[str, np.ndarray] = {n: specs[n].copy() for n in stem_names}
    report_iv = max(1, n_frames // 20)

    for fi in range(n_frames):
        if fi % report_iv == 0:
            print(f"  {100*fi//n_frames}%", file=sys.stderr, end="\r")

        energies = {}
        for name in stem_names:
            energies[name] = band_energy(np.abs(specs[name][fi]), bmasks)

        g = compute_gains_v2(
            energies, tgt_db, STEM_MAP, zone_lookups,
            max_boost=args.max_boost, max_cut=args.max_cut,
        )

        for name in stem_names:
            corrected[name][fi] = apply_gains_to_spec(
                specs[name][fi], freqs, g[name], BAND_LO, BAND_HI
            )

    print("  100% — готово              ", file=sys.stderr)

    # ── ISTFT ──
    print("ISTFT синтез…", file=sys.stderr)
    stems_out: Dict[str, np.ndarray] = {}
    for name in stem_names:
        stems_out[name] = stft_synthesize(
            corrected[name], fft_size, hop, window, n_samp
        )

    # ── Level automation ──
    print("Автоматизация уровней…", file=sys.stderr)
    auto_blk = max(int(sr * 0.3), 4096)
    for name in stem_names:
        env = rms_env(stems_out[name], auto_blk)
        active = env[env > 1e-6]
        if len(active) == 0:
            continue
        med = float(np.median(active))
        ge = np.ones_like(env)
        for i in range(len(env)):
            if env[i] > med * 2.5:
                ge[i] = 0.82
            elif env[i] > med * 1.8:
                ge[i] = 0.90
            elif env[i] < med * 0.12 and env[i] > 1e-7:
                ge[i] = 1.12
        stems_out[name] *= expand_env(ge, auto_blk, n_samp)

    # ── Pan & sum ──
    print("Панорамирование и сумма…", file=sys.stderr)
    mix_lr = np.zeros((n_samp, 2), dtype=np.float64)
    for name in stem_names:
        l, r = pan_cp(STEM_MAP[name].pan, stems_out[name])
        mix_lr[:, 0] += l
        mix_lr[:, 1] += r

    # ── Normalize ──
    peak = float(np.max(np.abs(mix_lr))) + 1e-12
    mix_lr *= 10.0 ** (args.peak_dbfs / 20.0) / peak

    # ── Report ──
    final_rms = float(np.sqrt(np.mean(mix_lr ** 2)))
    final_peak = float(np.max(np.abs(mix_lr)))
    print(f"\n═══ Отчёт ═══", file=sys.stderr)
    if args.foh_live:
        print("  Режим: FOH-LIVE (без ducking дорожек)", file=sys.stderr)
    print(f"  Peak: {20*np.log10(final_peak+1e-12):.1f} dBFS", file=sys.stderr)
    print(f"  RMS:  {20*np.log10(final_rms+1e-12):.1f} dBFS", file=sys.stderr)
    print(f"  Кривая: {args.slope} дБ/окт @ {args.ref_db} дБ (1 кГц)",
          file=sys.stderr)

    print(f"\n  Иерархия (активные стемы):", file=sys.stderr)
    active_sorted = sorted(
        ((n, STEM_MAP[n]) for n in stem_names),
        key=lambda kv: kv[1].hierarchy,
    )
    for name, si in active_sorted:
        zones = zones_mode[si.instrument]
        ess = [z for z in zones if z.role == "essential"]
        cuts = [z for z in zones if z.role in ("cut", "conditional_cut")]
        ess_s = " + ".join(f"{z.lo:.0f}-{z.hi:.0f}" for z in ess)
        cut_s = " − ".join(f"{z.lo:.0f}-{z.hi:.0f}" for z in cuts) if cuts else "—"
        print(f"    [{si.hierarchy}] {si.instrument:16s}  "
              f"★ {ess_s:30s}  ✂ {cut_s}  ({name})",
              file=sys.stderr)

    # ── Write ──
    out = os.path.abspath(args.output)
    sf.write(out, mix_lr.astype(np.float32), sr, subtype="PCM_24")
    print(f"\nЗаписано: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
