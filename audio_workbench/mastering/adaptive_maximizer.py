from __future__ import annotations
import numpy as np
from scipy import signal, ndimage

BANDS = ((25, 120), (120, 900), (900, 5000), (5000, None))
RELEASE_MS = (180., 120., 75., 45.)

def _band_split(x, sr):
    lows = []
    for fc in (120, 900, 5000):
        sos = signal.butter(3, fc, btype="lowpass", fs=sr, output="sos")
        lows.append(signal.sosfiltfilt(sos, x, axis=0).astype("float32"))
    return [lows[0], lows[1] - lows[0], lows[2] - lows[1], x - lows[2]]

def _envelope(x, sr):
    return ndimage.maximum_filter1d(
        np.max(np.abs(x), axis=1) + 1e-12,
        size=max(3, int(.004 * sr)),
        mode="nearest",
    )

def _smooth_gain(req, sr, release_ms):
    req = ndimage.minimum_filter1d(req.astype("float32"), size=max(3, int(.004 * sr)), mode="nearest")
    n = max(3, int(sr * release_ms / 1000))
    sm = ndimage.uniform_filter1d(req, size=n, mode="nearest")
    return np.minimum(req, sm)

def diagnose(x, sr, target_band_gr_db=(.65, .55, .45, .35)):
    bands = _band_split(x, sr)
    plan = []
    for i, (b, tgr) in enumerate(zip(bands, target_band_gr_db)):
        e = _envelope(b, sr)
        p995 = float(np.percentile(e, 99.5))
        p90 = float(np.percentile(e, 90))
        crest = 20 * np.log10((p995 + 1e-12) / (p90 + 1e-12))
        threshold = p995 / (10 ** (float(tgr) / 20))
        plan.append({
            "band": BANDS[i],
            "threshold": float(threshold),
            "target_gr_db": float(tgr),
            "upper_envelope_crest_db": float(crest),
            "release_ms": RELEASE_MS[i],
        })
    return plan

def process(x, sr, ceiling_db=-1.0, drive_db=2.2, target_band_gr_db=(.65, .55, .45, .35)):
    driven = x * np.float32(10 ** (drive_db / 20))
    bands = _band_split(driven, sr)
    plan = diagnose(driven, sr, target_band_gr_db)
    out = np.zeros_like(x)
    stats = []
    for b, p in zip(bands, plan):
        env = _envelope(b, sr)
        thr = p["threshold"]
        req = np.minimum(1., thr / env)
        g = _smooth_gain(req, sr, p["release_ms"])
        # Never let one adaptive band overreact to rare outliers.
        g = np.maximum(g, 10 ** (-1.25 / 20)).astype("float32")
        out += b * g[:, None]
        stats.append({
            **p,
            "max_gr_db": float(-20 * np.log10(max(float(g.min()), 1e-8))),
            "p95_gr_db": float(np.percentile(-20 * np.log10(np.maximum(g, 1e-8)), 95)),
        })
    env = _envelope(out, sr)
    ceil = 10 ** (ceiling_db / 20)
    req = np.minimum(1., ceil / env)
    g = _smooth_gain(req, sr, 55.)
    out *= g[:, None]
    return out.astype("float32"), {
        "drive_db": drive_db,
        "ceiling_db": ceiling_db,
        "bands": stats,
        "final_max_gr_db": float(-20 * np.log10(max(float(g.min()), 1e-8))),
        "final_p95_gr_db": float(np.percentile(-20 * np.log10(np.maximum(g, 1e-8)), 95)),
    }
