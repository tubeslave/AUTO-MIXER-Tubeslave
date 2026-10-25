from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class EditingPolicy:
    mode:str="minimal_intervention"
    max_drum_nudge_ms:float=18.0
    max_music_nudge_ms:float=25.0
    phase_search_ms:float=8.0
    polarity_confidence_min:float=.15
    transient_outlier_sigma:float=3.0
    vocal_silence_floor_db:float=-52.0
    click_window_ms:float=8.0
    pitch_mode:str="diagnose_only"
    preserve_length:bool=True

def plan() -> dict[str,Any]:
    return {
      "order":[
        "integrity_and_sync",
        "phase_and_polarity",
        "click_pop_noise_detection",
        "drum_transient_map",
        "groove_outlier_detection",
        "instrument_timing_relationships",
        "vocal_timing_and_double_alignment",
        "pitch_diagnosis",
        "pre_mix_level_outliers",
        "regression_and_export",
      ],
      "principle":"diagnose first; edit only high-confidence local defects; preserve musical groove",
      "forbidden_by_default":[
        "global_quantization","automatic_full-vocal-tuning","time-stretching-entire-tracks",
        "noise-gating-every-track","sample-replacement-without-evidence"
      ],
    }

def accept_edit(before:dict,after:dict)->dict[str,Any]:
    fail=[]
    if before.get("frames")!=after.get("frames"): fail.append("length_changed")
    if after.get("new_clip_count",0)>0: fail.append("new_clipping")
    if after.get("timing_artifact_score",0)>.25: fail.append("timing_artifact")
    return {"accept":not fail,"failures":fail}
