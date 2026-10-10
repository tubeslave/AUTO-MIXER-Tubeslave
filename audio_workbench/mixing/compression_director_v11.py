"""Compression Director v1.1: measured-GR threshold calibration and timing censor evidence.

This layer wraps v1 without changing its public behaviour. It may change threshold
only; ratio, attack, release, knee, RMS integration and max-GR safety bounds remain
those proposed by v1. Every candidate remains a human-listening audition candidate.
"""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
from scipy import ndimage, signal

from .compression import CompressorConfig, LinkedCompressor, as_audio
from .compression_director import POLICIES, _frame_db, _linked_power, config_from_candidate, propose_candidates


def _active_sample_mask(x: np.ndarray, sr: int, hop_s: float = .02) -> np.ndarray:
    x = as_audio(x)
    if not len(x):
        return np.zeros(0, dtype=bool)
    hop = max(1, int(round(sr * hop_s)))
    starts = np.arange(0, len(x), hop, dtype=int)
    ends = np.minimum(starts + hop, len(x))
    counts = ends - starts
    power = _linked_power(x)
    integral = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    frame_power = (integral[ends] - integral[starts]) / counts
    if float(np.max(frame_power)) <= 1e-20:
        return np.zeros(len(x), dtype=bool)
    db = 10 * np.log10(np.maximum(frame_power, 1e-30))
    threshold = max(-100.0, float(np.max(db)) - 40.0, float(np.percentile(db, 55)))
    return np.repeat(db >= threshold, counts)[:len(x)]


def _active_p95_gr(x: np.ndarray, sr: int, config: CompressorConfig,
                   active_mask: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    x = as_audio(x)
    mask = _active_sample_mask(x, sr) if active_mask is None else np.asarray(active_mask, dtype=bool)
    if mask.shape != (len(x),):
        raise ValueError("active mask must match source frames")
    _, gr = LinkedCompressor(sr, config).process(x)
    values = gr[mask]
    return (float(np.percentile(values, 95)) if len(values) else 0.0), gr


def calibrate_threshold(x: np.ndarray, sr: int, config: CompressorConfig,
                        target_active_p95_gr_db: float, *, tolerance_db: float = .08,
                        max_iterations: int = 14,
                        threshold_bounds: tuple[float, float] = (-80.0, 6.0)
                        ) -> tuple[CompressorConfig, dict]:
    """Bisection-calibrate threshold against actual LinkedCompressor p95 GR.

    The comparison mask is fixed from the raw source. Safety parameters are never
    weakened. If the target is unreachable, the result says so explicitly.
    """
    x = as_audio(x); config.validate(sr)
    if not np.isfinite(target_active_p95_gr_db) or target_active_p95_gr_db < 0:
        raise ValueError("target_active_p95_gr_db must be finite and non-negative")
    if not np.isfinite(tolerance_db) or tolerance_db <= 0:
        raise ValueError("tolerance_db must be finite and positive")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    low, high = map(float, threshold_bounds)
    if not np.isfinite(low + high) or low >= high:
        raise ValueError("invalid threshold bounds")
    mask = _active_sample_mask(x, sr)
    if not np.any(mask):
        report = {"status":"insufficient_signal","target_active_p95_gr_db":float(target_active_p95_gr_db),
                  "actual_active_p95_gr_db":0.0,"error_db":float(-target_active_p95_gr_db),
                  "starting_threshold_dbfs":float(config.threshold_dbfs),"calibrated_threshold_dbfs":float(config.threshold_dbfs),
                  "iterations":0,"tolerance_db":float(tolerance_db),"max_gr_unchanged":True,
                  "measurement_scope":"fixed RAW-active sample mask from non-overlapping 20 ms RMS frames"}
        return config, report

    def measure(threshold: float) -> tuple[float, CompressorConfig]:
        cfg = replace(config, threshold_dbfs=float(threshold)); cfg.validate(sr)
        actual, _ = _active_p95_gr(x, sr, cfg, mask)
        return actual, cfg

    strong, strong_cfg = measure(low)
    weak, weak_cfg = measure(high)
    target = float(target_active_p95_gr_db)
    if target > strong + tolerance_db:
        best_cfg, actual, status, iterations = strong_cfg, strong, "unreachable_high", 2
    elif target < weak - tolerance_db:
        best_cfg, actual, status, iterations = weak_cfg, weak, "unreachable_low", 2
    else:
        best_cfg, actual = min(((strong_cfg,strong),(weak_cfg,weak)), key=lambda q: abs(q[1]-target))
        status = "best_effort"; iterations = 2
        lo, hi = low, high
        for _ in range(max_iterations):
            mid = (lo + hi) / 2
            value, cfg = measure(mid); iterations += 1
            if abs(value-target) < abs(actual-target):
                best_cfg, actual = cfg, value
            if abs(value-target) <= tolerance_db:
                best_cfg, actual, status = cfg, value, "converged"
                break
            if value > target:  # too much compression: raise threshold
                lo = mid
            else:
                hi = mid
    report = {"status":status,"target_active_p95_gr_db":target,"actual_active_p95_gr_db":float(actual),
              "error_db":float(actual-target),"starting_threshold_dbfs":float(config.threshold_dbfs),
              "calibrated_threshold_dbfs":float(best_cfg.threshold_dbfs),"iterations":int(iterations),
              "tolerance_db":float(tolerance_db),"max_gr_unchanged":best_cfg.max_gr_db==config.max_gr_db,
              "measurement_scope":"fixed RAW-active sample mask from non-overlapping 20 ms RMS frames"}
    return best_cfg, report


def timing_censor_evidence(x: np.ndarray, sr: int, role: str) -> dict:
    """Report when v1 attack/recovery estimates terminate at analysis boundaries."""
    x = as_audio(x)
    if role not in POLICIES:
        raise ValueError(f"unknown compression role: {role}")
    p = POLICIES[role]
    db, times, _ = _frame_db(x, sr, p.frame_ms, p.hop_ms)
    if not len(db) or float(np.max(db)) < -100:
        return {"event_count":0,"attack_censored_count":0,"recovery_censored_count":0,
                "attack_censored_fraction":None,"recovery_censored_fraction":None,
                "attack_reason_counts":{},"recovery_reason_counts":{},
                "measurement_scope":"same role-smoothed macro envelope as Compression Director v1"}
    sigma=max(.5,p.smooth_ms/(2.355*p.hop_ms)); env=ndimage.gaussian_filter1d(db,sigma)
    active_threshold=max(-100.0,float(np.max(env))-45.0,float(np.percentile(env,40)))
    hop_s=float(np.median(np.diff(times))) if len(times)>1 else p.hop_ms/1000
    peaks,_=signal.find_peaks(env,height=active_threshold,prominence=p.prominence_db,
        distance=max(1,int(round((p.min_event_gap_ms/1000)/hop_s))))
    max_back=max(1,int(round(.20/hop_s))); max_forward=max(1,int(round(1.20/hop_s)))
    ac=rc=0; ar={}; rr={}
    for i,peak in enumerate(peaks):
        six=env[peak]-6.0; left_limit=max(0,int(peak)-max_back); left=int(peak)
        while left>left_limit and env[left]>six: left-=1
        if left==left_limit and env[left]>six:
            ac+=1; reason="source_start" if left_limit==0 else "lookback_limit"; ar[reason]=ar.get(reason,0)+1
        next_limit=int(peaks[i+1]-1) if i+1<len(peaks) else len(env)-1
        window_limit=min(len(env)-1,int(peak)+max_forward); right_limit=min(window_limit,next_limit); right=int(peak)
        while right<right_limit and env[right]>six: right+=1
        if right==right_limit and env[right]>six:
            rc+=1
            if right_limit==next_limit and i+1<len(peaks): reason="next_event"
            elif right_limit==len(env)-1: reason="source_end"
            else: reason="max_window"
            rr[reason]=rr.get(reason,0)+1
    n=len(peaks)
    return {"event_count":int(n),"attack_censored_count":int(ac),"recovery_censored_count":int(rc),
            "attack_censored_fraction":float(ac/n) if n else None,"recovery_censored_fraction":float(rc/n) if n else None,
            "attack_reason_counts":ar,"recovery_reason_counts":rr,
            "measurement_scope":"same role-smoothed macro envelope as Compression Director v1"}


def propose_calibrated_candidates(x: np.ndarray, sr: int, role: str, *,
                                  tolerance_db: float=.08, max_iterations: int=14) -> dict:
    """Return v1 candidates with threshold only calibrated to real active-p95 GR."""
    x=as_audio(x); base=propose_candidates(x,sr,role); candidates=[]
    for item in base["candidates"]:
        cfg=config_from_candidate(item); target=float(item["requested_static_gr_db"])
        calibrated, evidence=calibrate_threshold(x,sr,cfg,target,tolerance_db=tolerance_db,max_iterations=max_iterations)
        out=dict(item); out["compressor"]=asdict(calibrated); out["target_active_p95_gr_db"]=target
        out["calibration"]=evidence; out["calibration_status"]=evidence["status"]
        out["evidence_basis"]="v1 macro timing + actual LinkedCompressor active-p95 threshold calibration"
        candidates.append(out)
    return {"schema":"compression-director-v1.1","role":role,"analysis":base["analysis"],
            "timing_censoring":timing_censor_evidence(x,sr,role),"candidates":candidates,
            "selection_policy":"candidate set only; no machine winner or baseline promotion",
            "requires_human_review":True,"requires_human_listening":True,"baseline_eligible":False,
            "calibration_scope":"threshold only; ratio/attack/release/knee/RMS/max-GR inherited unchanged from v1"}


def render_calibrated_core_candidate(x: np.ndarray, sr: int, candidate: dict) -> tuple[np.ndarray, dict]:
    x=as_audio(x); cfg=config_from_candidate(candidate); mask=_active_sample_mask(x,sr)
    y,gr=LinkedCompressor(sr,cfg).process(x); active=gr[mask]
    actual=float(np.percentile(active,95)) if len(active) else 0.0
    target=candidate.get("target_active_p95_gr_db")
    return y,{"schema":"compression-director-v1.1-render","candidate_id":str(candidate["id"]),
              "active_p95_gr_db":actual,"target_active_p95_gr_db":target,
              "target_error_db":None if target is None else float(actual-float(target)),
              "max_gr_db":float(np.max(gr)) if len(gr) else 0.0,
              "whole_track_p95_gr_db":float(np.percentile(gr,95)) if len(gr) else 0.0,
              "requires_human_review":True,"requires_human_listening":True,"baseline_eligible":False}
