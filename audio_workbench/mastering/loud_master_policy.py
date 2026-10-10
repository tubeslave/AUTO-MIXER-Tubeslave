from __future__ import annotations
from dataclasses import dataclass

@dataclass
class LoudMasterPolicy:
    loud_section_target_lufs: float=-8.0
    limiter_max_gr_db: float=3.0
    bus_comp_max_gr_db: float=1.5
    multiband_peak_max_gr_db: float=1.5
    clipper_max_peak_reduction_db: float=2.5
    true_peak_ceiling_dbtp: float=-1.0

def stage_budget()->list[dict]:
    return [
      {"stage":"bus_density","purpose":"raise RMS/density before peak stages","max_gr_db":1.5},
      {"stage":"multiband_peak_conditioner","purpose":"reduce band-local peaks without broadband pumping","max_gr_db":1.5},
      {"stage":"oversampled_clipper","purpose":"round isolated transients before limiter","max_peak_reduction_db":2.5},
      {"stage":"final_limiter","purpose":"ceiling and last loudness step","max_gr_db":3.0},
    ]

def accept(metrics:dict,policy:LoudMasterPolicy|None=None)->dict:
    p=policy or LoudMasterPolicy();fail=[]
    if metrics["loud_section_lufs"] < p.loud_section_target_lufs-.35: fail.append("target_not_reached")
    if metrics["limiter_max_gr_db"] > p.limiter_max_gr_db+.05: fail.append("limiter_over_budget")
    if metrics.get("true_peak_dbtp",-99)>p.true_peak_ceiling_dbtp+.05: fail.append("true_peak")
    return {"accept":not fail,"failures":fail}
