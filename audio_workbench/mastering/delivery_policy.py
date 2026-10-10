from __future__ import annotations
from dataclasses import dataclass

@dataclass
class DeliveryPolicy:
    """Quality-first, then loudness. A master is not complete at premaster-like loudness."""
    min_loudness_gain_lu: float = 2.0
    preferred_loudness_gain_lu: float = 3.0
    max_final_gr_db: float = 1.5
    max_band_gr_db: float = 1.75
    max_crest_loss_db: float = 4.0
    ceiling_dbtp: float = -1.0

def needs_delivery_stage(premaster_lufs:float,quality_lufs:float,policy:DeliveryPolicy|None=None)->bool:
    p=policy or DeliveryPolicy()
    return quality_lufs-premaster_lufs < p.preferred_loudness_gain_lu

def accept_delivery(premaster:dict,quality:dict,candidate:dict,diag:dict,
                    policy:DeliveryPolicy|None=None)->dict:
    p=policy or DeliveryPolicy();fail=[]
    gain=candidate["lufs_i"]-premaster["lufs_i"]
    if gain < p.min_loudness_gain_lu: fail.append("insufficient_loudness_gain")
    if diag.get("final_max_gr_db",0)>p.max_final_gr_db: fail.append("excess_final_gr")
    if diag.get("max_band_gr_db",0)>p.max_band_gr_db: fail.append("excess_band_gr")
    if premaster["crest_db"]-candidate["crest_db"]>p.max_crest_loss_db: fail.append("excess_crest_loss")
    return {"accept":not fail,"failures":fail,"loudness_gain_lu":gain,
            "quality_baseline_lufs":quality["lufs_i"],"delivery_lufs":candidate["lufs_i"]}
