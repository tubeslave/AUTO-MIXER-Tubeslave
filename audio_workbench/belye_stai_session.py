"""Deterministic SessionRender adapter for the delivered Belye Stai DSP mix v1.

This is a structural migration of the delivered conventional-DSP recipe. It keeps
its signal order and coefficients so a no-change render can be compared with the
stored premaster before any new Compression Director candidate is evaluated.
No learned audio model, mastering stage, file writing or baseline promotion lives
in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import ndimage, signal
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda fn: fn
from pedalboard import Reverb
import soundfile as sf

from .routed_contribution import SessionRender
from .mixing.compression import as_audio


SOURCE_IDS = (
    "BASS", "FLOOR", "GTR", "HI_HAT", "KEYS_L", "KEYS_R", "KICK_IN",
    "KICK_OUT", "NIKITA_VOX", "OH_R", "OHL", "PB_L", "PB_R", "SN_B",
    "SN_T", "TOM_1", "TOM_2", "VALERA_VOX",
)
RECIPE_ID = "belye-stai-dsp-mix-v1-session-adapter"


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-30))


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def _frame_db(x, sr, hop=.01):
    h = max(1, round(sr * hop))
    usable = len(x) // h * h
    if not usable:
        return np.array([], dtype=np.float64)
    a = x[:usable].reshape(-1, h, *x.shape[1:])
    p = np.mean(a.astype("float64") ** 2, axis=1)
    if p.ndim > 1:
        p = np.mean(p, axis=1)
    return 10 * np.log10(np.maximum(p, 1e-24))


def _curve(values, hop, n, sr):
    if not len(values):
        return np.zeros(n, np.float32)
    return np.interp(np.arange(n) / sr, (np.arange(len(values)) + .5) * hop, values).astype("float32")


def _gain(x, dd):
    g = np.power(10, np.asarray(dd) / 20).astype("float32")
    return (x * (g[:, None] if np.ndim(g) and x.ndim == 2 else g)).astype("float32")


def _filt(x, sr, lo=None, hi=None, order=2):
    y = x
    if lo:
        y = signal.sosfilt(signal.butter(order, lo, btype="highpass", fs=sr, output="sos"), y, axis=0)
    if hi:
        y = signal.sosfilt(signal.butter(order, hi, btype="lowpass", fs=sr, output="sos"), y, axis=0)
    return y.astype("float32")


def _bell(x, sr, f, dd, q=.85):
    A = 10 ** (dd / 40)
    w = 2 * np.pi * f / sr
    c = np.cos(w)
    alpha = np.sin(w) / (2 * q)
    b = np.array([1 + alpha * A, -2 * c, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * c, 1 - alpha / A])
    return signal.lfilter(b / a[0], a / a[0], x, axis=0).astype("float32")


def _band(x, sr, lo, hi):
    return signal.sosfiltfilt(
        signal.butter(2, [lo, hi], btype="bandpass", fs=sr, output="sos"), x, axis=0
    ).astype("float32")


def _active_level(x, sr, q=65):
    d = _frame_db(x, sr, .02)
    if not len(d):
        return -300.0
    threshold = max(float(np.percentile(d, q)), float(np.max(d)) - 35)
    a = d >= threshold
    return float(10 * np.log10(np.mean(10 ** (d[a] / 10))))


def _level(x, sr, target, q=65):
    before = _active_level(x, sr, q)
    g = float(np.clip(target - before, -30, 30))
    y = _gain(x, g)
    return y, {"active_percentile": q, "active_rms_before_dbfs": before,
               "target_active_rms_dbfs": target, "gain_db": g,
               "active_rms_after_dbfs": _active_level(y, sr, q)}


def _smooth_gr_python(req, sr, attack_ms, release_ms):
    a = np.exp(-1 / (sr * attack_ms / 1000))
    r = np.exp(-1 / (sr * release_ms / 1000))
    y = np.empty(len(req), np.float32)
    last = 0.0
    for i in range(len(req)):
        c = a if req[i] > last else r
        last = c * last + (1 - c) * req[i]
        y[i] = last
    return y


_smooth_gr_compiled = njit(cache=True)(_smooth_gr_python)


def _compressor(x, sr, desired=3., ratio=3., attack=15., release=150., cap=5., knee=5.):
    p = x.astype("float64") ** 2
    if p.ndim == 2:
        p = p.mean(1)
    a = np.exp(-1 / (sr * .003))
    p = signal.lfilter([1 - a], [1, -a], p)
    env = 10 * np.log10(np.maximum(p, 1e-20))
    stride = max(1, round(sr * .01))
    sample = env[::stride]
    act = sample[sample >= np.percentile(sample, 60)]
    threshold = float(np.percentile(act, 90) - desired / (1 - 1 / ratio))
    v = env - threshold
    req = np.where(v <= -knee / 2, 0, np.where(
        v >= knee / 2, v * (1 - 1 / ratio),
        ((v + knee / 2) ** 2) / (2 * knee) * (1 - 1 / ratio)))
    req = np.clip(req, 0, cap)
    gr = _smooth_gr_compiled(req, sr, attack, release)
    y = _gain(x, -gr)
    diag = {"ratio": ratio, "threshold_dbfs": threshold, "attack_ms": attack,
            "release_ms": release, "knee_db": knee, "hard_max_gr_db": cap,
            "observed_max_gr_db": float(gr.max()),
            "observed_p95_gr_db": float(np.percentile(gr[::stride], 95)),
            "stereo_linked": True}
    return y, diag, gr


def _expand(x, sr, n, lo, hi, *, threshold_q=25, margin_db=6., floor=-10., release=.16, hold=.10):
    e = _frame_db(_band(x, sr, lo, hi), sr)
    th = float(np.percentile(e, threshold_q) + margin_db)
    e = ndimage.maximum_filter1d(e, size=max(3, int(hold / .01) * 2 + 1))
    dd = np.clip((e - th) * 1.2, floor, 0)
    dd = ndimage.gaussian_filter1d(dd, max(1, release / .01 / 3))
    return _gain(x, _curve(dd, .01, n, sr)), {"threshold_dbfs": th, "floor_db": floor}


def _ride(x, sr, n, lo=180, hi=3800, depth=2.):
    d = _frame_db(_band(x, sr, lo, hi), sr, .05)
    s = ndimage.gaussian_filter1d(d, 8)
    th = max(float(np.percentile(s, 20) + 9), float(np.percentile(s, 75) - 18))
    active = s > th
    tar = float(np.median(s[active])) if active.any() else float(np.median(s))
    dd = np.where(active, np.clip((tar - s) * .50, -depth, depth), 0)
    dd = ndimage.gaussian_filter1d(dd, 6)
    return _gain(x, _curve(dd, .05, n, sr)), {"range_db": [float(dd.min()), float(dd.max())]}


def _deess(x, sr, n):
    h = _band(x, sr, 4500, 10000)
    m = _band(x, sr, 800, 3500)
    hd = _frame_db(h, sr, .005)
    md = _frame_db(m, sr, .005)
    ratio = hd - md
    th = max(float(np.percentile(ratio, 82)), -10.)
    gr = np.clip((ratio - th) * .6, 0, 2.8)
    gr[hd < np.percentile(hd, 55)] = 0
    gr = ndimage.gaussian_filter1d(gr, 2)
    g = 10 ** (_curve(-gr, .005, n, sr) / 20)
    return (x + h * (g - 1)).astype("float32"), {"max_gr_db": float(gr.max())}


def _advance(x, n):
    if n == 0:
        return x.copy()
    return np.concatenate([x[n:], np.zeros(n, np.float32)]) if n > 0 else np.concatenate([np.zeros(-n, np.float32), x[:n]])


def _pan(x, p):
    theta = (p + 1) * np.pi / 4
    return np.column_stack([x * np.cos(theta), x * np.sin(theta)]).astype("float32")


def _reverb_aux(inp, sr, n, room, damping, predelay_ms, low, high, rel):
    pre = round(predelay_ms * sr / 1000)
    send = np.pad(inp, ((pre, 0), (0, 0)))[:n]
    send = _filt(send, sr, low, high)
    rev = Reverb(room_size=room, damping=damping, wet_level=1., dry_level=0., width=1., freeze_mode=0.)
    wet = rev(send.T.copy(), sr, reset=True).T.astype("float32")
    wet = _filt(wet, sr, low, high)
    factor = 10 ** (rel / 20) * _rms(inp) / max(_rms(wet), 1e-10)
    wet *= np.float32(factor)
    return wet


def _render_graph(sources: Mapping[str, np.ndarray], sr: int) -> SessionRender:
    if set(sources) != set(SOURCE_IDS):
        missing = sorted(set(SOURCE_IDS) - set(sources)); extra = sorted(set(sources) - set(SOURCE_IDS))
        raise ValueError(f"Belye Stai source set mismatch; missing={missing}, extra={extra}")
    raw = {k: as_audio(np.asarray(sources[k])).copy() for k in SOURCE_IDS}
    n = len(raw[SOURCE_IDS[0]])
    if not n or any(x.ndim != 1 or len(x) != n for x in raw.values()):
        raise ValueError("Belye Stai requires equal-length non-empty mono sources")

    raw["KICK_OUT"] = _advance(raw["KICK_OUT"], 84)
    raw["SN_B"] = -raw["SN_B"]
    buses = {}; drum_parts = {}
    for name, hp, eq in [
        ("KICK_IN", 28, [(65, 1.2, .8), (190, -2., .9), (2800, 1.0, .7)]),
        ("KICK_OUT", 28, [(190, -2.5, .8)]),
    ]:
        y = _filt(raw.pop(name), sr, hp, 6500 if name == "KICK_IN" else 1400)
        for f, d, q in eq:
            y = _bell(y, sr, f, d, q)
        drum_parts[name] = y
    match = _active_level(drum_parts["KICK_IN"], sr, 92) - _active_level(drum_parts["KICK_OUT"], sr, 92) - 5
    kick = drum_parts.pop("KICK_IN") + _gain(drum_parts.pop("KICK_OUT"), match)
    kick, _, _ = _compressor(kick, sr, 3., 3., 22, 115, 4.5)
    kick, _ = _level(kick, sr, -22.5, 92)

    st = _filt(raw.pop("SN_T"), sr, 85, 12500); sb = _filt(raw.pop("SN_B"), sr, 220, 11500)
    st = _bell(_bell(st, sr, 380, -2.0), sr, 2500, 2.0, .7)
    sb, _ = _expand(sb, sr, n, 600, 7000, threshold_q=25, margin_db=9, floor=-8)
    match = _active_level(st, sr, 94) - _active_level(sb, sr, 94) - 11.
    snare = st + _gain(sb, match)
    snare, _, _ = _compressor(snare, sr, 3.5, 3., 14, 110, 5.)
    snare, _ = _level(snare, sr, -23.8, 94)
    drums = _pan(kick, 0) + _pan(snare, 0); drum_send = _pan(snare, 0)
    for name, hp, p in [("TOM_1", 65, -.45), ("TOM_2", 55, .12), ("FLOOR", 45, .52)]:
        y = _filt(raw.pop(name), sr, hp, 10500); y = _bell(y, sr, 420, -2., .8)
        y, _ = _expand(y, sr, n, 80, 350, threshold_q=90, margin_db=-5., floor=-16., release=.22, hold=.08)
        y, _, _ = _compressor(y, sr, 2.5, 2.6, 18, 160, 4.); y, _ = _level(y, sr, -27., 98)
        yp = _pan(y, p); drums += yp; drum_send += yp * .7
    oh = np.column_stack([raw.pop("OHL"), _gain(raw.pop("OH_R"), 3.)])
    oh = _filt(oh, sr, 190, 13500); oh = _bell(oh, sr, 6000, -1.2, .7); oh, _ = _level(oh, sr, -34., 65); drums += oh
    hh = _filt(raw.pop("HI_HAT"), sr, 700, 11000); hh, _ = _level(hh, sr, -43., 75); drums += _pan(hh, -.45)
    drums, _, _ = _compressor(drums, sr, 1., 1.5, 30, 160, 1.6); drums, _ = _level(drums, sr, -25.5, 65); buses["DRUMS"] = drums

    bass = _filt(raw.pop("BASS"), sr, 33, 6000); bass = _bell(bass, sr, 175, -2.0, .7); bass = _bell(bass, sr, 850, 1.2, .7)
    bass, _ = _ride(bass, sr, n, 50, 1100, 2.0); bass, _, _ = _compressor(bass, sr, 4., 3.5, 9, 130, 5.0); bass, _, _ = _compressor(bass, sr, 1.5, 2., 35, 240, 2.8)
    bass, _ = _level(bass, sr, -26.8, 65); buses["BASS"] = _pan(bass, 0)
    gtr = _filt(raw.pop("GTR"), sr, 90, 9000); gtr = _bell(gtr, sr, 310, -1.8, .8); gtr = _bell(gtr, sr, 3000, -.8, .8)
    gtr, _, _ = _compressor(gtr, sr, 1.5, 2., 25, 180, 2.8); gtr, _ = _level(gtr, sr, -28.3, 60); buses["GUITAR"] = _pan(gtr, -.23)
    for left, right, key, hp, target in [("KEYS_L", "KEYS_R", "KEYS", 115, -32.2), ("PB_L", "PB_R", "PLAYBACK", 90, -28.5)]:
        y = np.column_stack([raw.pop(left), raw.pop(right)]); y = _filt(y, sr, hp); y = _bell(y, sr, 350, -1.3, .8)
        y, _, _ = _compressor(y, sr, 1.2, 1.6, 30, 230, 2.0); y, _ = _level(y, sr, target, 60); buses[key] = y
    for name, key, hp, target, p in [("VALERA_VOX", "LEAD_VOX", 110, -22.6, 0.), ("NIKITA_VOX", "BACK_VOX", 135, -29.3, .12)]:
        y = _filt(raw.pop(name), sr, hp, 14500); y = _bell(y, sr, 230, -2.4, .8); y = _bell(y, sr, 700, -.7, .9); y = _bell(y, sr, 2400, 2.0, .7)
        y, _ = _expand(y, sr, n, 250, 3500, threshold_q=18, margin_db=8., floor=-12 if key == "BACK_VOX" else -8, release=.2, hold=.15)
        y, _ = _ride(y, sr, n, 220, 3800, 2.3); y, _, _ = _compressor(y, sr, 3.2, 3.2, 10, 105, 4.5); y, _, _ = _compressor(y, sr, 1.3, 2., 30, 220, 2.5)
        y, _ = _deess(y, sr, n); y, _ = _level(y, sr, target, 65); buses[key] = _pan(y, p)
    if raw:
        raise RuntimeError(f"unconsumed Belye Stai sources: {sorted(raw)}")

    voc = buses["LEAD_VOX"]
    vd = _frame_db(_band(voc, sr, 250, 3500), sr, .05); th = max(float(np.percentile(vd, 18)) + 10, float(np.percentile(vd, 75)) - 16)
    activity = np.clip((vd - th) / 6, 0, 1); activity = ndimage.gaussian_filter1d(activity, 4); full_act = _curve(activity, .05, n, sr)
    buses["GUITAR"] = _gain(buses["GUITAR"], 1.2 * (1 - full_act))
    vd = _frame_db(_band(voc, sr, 1400, 4200), sr, .05)
    for name, maxcut in [("GUITAR", .9), ("KEYS", .7), ("PLAYBACK", 1.0)]:
        x = buses[name]; component = _band(x, sr, 1400, 4200); md = _frame_db(component, sr, .05)
        overlap = np.clip((md - vd + 7) / 8, 0, 1) * activity
        gr = _curve(ndimage.gaussian_filter1d(overlap * maxcut, 3), .05, n, sr)
        y = x + component * (10 ** (-gr[:, None] / 20) - 1)
        delta = float(_db(_rms(y)) - _db(_rms(x)))
        if abs(delta) < .6:
            buses[name] = y.astype("float32")
    x = buses["BASS"]; low = _band(x, sr, 35, 115); kd = _frame_db(kick, sr, .01); kact = np.clip((kd - np.percentile(kd, 80)) / 12, 0, 1); gr = _curve(ndimage.gaussian_filter1d(kact, 3) * 1.2, .01, n, sr)
    buses["BASS"] = (x + low * (10 ** (-gr[:, None] / 20) - 1)).astype("float32")

    room = _reverb_aux(drum_send, sr, n, .32, .65, 12, 220, 7000, -20.)
    chamber = _reverb_aux(buses["LEAD_VOX"] + buses["BACK_VOX"] * .4, sr, n, .47, .68, 29, 230, 7800, -19.5)
    vmono = _filt(buses["LEAD_VOX"], sr, 300, 4800); slap = np.zeros_like(vmono); lag = round(.112 * sr); slap[lag:] = vmono[:-lag] * 10 ** (-27 / 20)
    buses["FX"] = (room + chamber + slap).astype("float32")

    mix = np.zeros((n, 2), np.float64)
    for y in buses.values():
        mix += y
    mix = mix.astype("float32"); mix, glue, gr = _compressor(mix, sr, .65, 1.4, 35, 230, 1.25)
    for key in buses:
        buses[key] = _gain(buses[key], -gr)
    edge = np.ones(n, np.float32)
    fade_in = min(221, n); edge[:fade_in] = np.linspace(0, 1, fade_in)
    fade_out = min(22050, n); edge[-fade_out:] = np.linspace(1, 0, fade_out)
    mix *= edge[:, None]
    trim = min(0., -4 - float(_db(np.max(abs(mix))))); mix = _gain(mix, trim)
    for key in buses:
        buses[key] = _gain(buses[key], trim) * edge[:, None]

    # The delivered premaster contains one documented bounded refinement after the
    # base mix: a 3 dB Gaussian reduction of the DRUMS bus around the 159.056 s
    # break accent. It is part of the authoritative delivery graph and therefore
    # must also be inside the reusable session renderer.
    t = np.arange(n) / sr
    accent_gr = 3.0 * np.exp(-.5 * ((t - 159.056) / .055) ** 2)
    refined_drums = buses["DRUMS"] * np.power(10, -accent_gr[:, None] / 20).astype("float32")
    mix = (mix + (refined_drums - buses["DRUMS"])).astype("float32")
    buses["DRUMS"] = refined_drums.astype("float32", copy=False)

    return SessionRender(
        mix=mix.astype("float32", copy=False),
        vocal_bus=buses["LEAD_VOX"].astype("float32", copy=False),
        drums_bus=buses["DRUMS"].astype("float32", copy=False),
        early_room=room.astype("float32", copy=False),
        metadata={"recipe_id": RECIPE_ID, "sample_rate": int(sr), "frames": int(n),
                  "mix_glue": glue, "common_trim_db": float(trim),
                  "accent_refinement": {"center_seconds": 159.056, "sigma_seconds": .055, "max_reduction_db": 3.0},
                  "source_dependent_routing": ["vocal_guitar_lift", "vocal_masking", "kick_bass_ducking", "send_returns", "shared_mix_glue", "documented_drum_accent_refinement"]},
    )


@dataclass
class BelyeStaiSessionRenderer:
    """Callable full-session renderer compatible with routed_contribution.SessionRenderer."""
    sources: Mapping[str, np.ndarray]
    sample_rate: int = 44100

    def __post_init__(self):
        clean = {k: as_audio(np.asarray(v)).copy() for k, v in self.sources.items()}
        if set(clean) != set(SOURCE_IDS):
            raise ValueError("Belye Stai renderer requires all 18 named sources")
        n = len(clean[SOURCE_IDS[0]])
        if not n or any(v.ndim != 1 or len(v) != n for v in clean.values()):
            raise ValueError("Belye Stai sources must be equal-length mono PCM")
        object.__setattr__(self, "sources", clean)

    def __call__(self, overrides: Mapping[str, np.ndarray]) -> SessionRender:
        unknown = set(overrides) - set(SOURCE_IDS)
        if unknown:
            raise ValueError(f"unknown Belye Stai override(s): {sorted(unknown)}")
        current = {k: v.copy() for k, v in self.sources.items()}
        for key, value in overrides.items():
            candidate = as_audio(np.asarray(value))
            if candidate.shape != current[key].shape:
                raise ValueError(f"override shape differs for {key}")
            current[key] = candidate.copy()
        return _render_graph(current, int(self.sample_rate))

    @classmethod
    def from_directory(cls, raw_dir: str | Path, *, sample_rate: int = 44100,
                       manifest: Mapping[str, str] | None = None):
        raw_dir = Path(raw_dir)
        sources = {}
        for path in sorted(raw_dir.glob("*.wav")):
            stem = path.stem
            source_id = stem.split("_", 1)[1] if stem[:2].isdigit() and "_" in stem else stem
            if source_id not in SOURCE_IDS:
                continue
            if manifest is not None:
                expected = manifest.get(path.name)
                if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"source hash mismatch: {path.name}")
            x, sr = sf.read(path, dtype="float32")
            if sr != sample_rate:
                raise ValueError(f"sample-rate mismatch: {path.name}")
            sources[source_id] = x
        return cls(sources=sources, sample_rate=sample_rate)
