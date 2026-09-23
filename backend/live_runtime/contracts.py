from __future__ import annotations
from dataclasses import dataclass,field
from enum import Enum
from typing import Any

class LiveMode(str,Enum):
    OBSERVE="observe"
    PROPOSE="propose"
    BENCH_TEST="bench_test"
    SUPERVISED="supervised"
    AUTO_SAFE="auto_safe"
    EMERGENCY="emergency"
    FREEZE="freeze"

class SoundcheckState(str,Enum):
    DISCOVER="discover"
    PATCH_VERIFY="patch_verify"
    LISTEN="listen"
    PROPOSE="propose"
    APPLY="apply"
    VERIFY="verify"
    HOLD="hold"

@dataclass(frozen=True)
class ChannelFeatures:
    channel:int
    name:str
    rms_dbfs:float
    peak_dbfs:float
    crest_db:float
    lufs_short:float|None=None
    spectral_centroid_hz:float|None=None
    low_mid_ratio_db:float|None=None
    harshness:float|None=None
    activity:float=0.
    confidence:float=1.

@dataclass(frozen=True)
class MixFeatures:
    channels:list[ChannelFeatures]
    main_rms_dbfs:float
    main_peak_dbfs:float
    main_crest_db:float
    stereo_width_db:float|None=None
    feedback_candidates_hz:list[float]=field(default_factory=list)
    timestamp_s:float=0.

@dataclass(frozen=True)
class MixerSnapshot:
    revision:str
    channels:dict[int,dict[str,Any]]
    buses:dict[int,dict[str,Any]]
    mains:dict[int,dict[str,Any]]
    routing:dict[str,Any]
    timestamp_s:float

@dataclass(frozen=True)
class ProposedAction:
    target:str
    parameter:str
    value:float|str|bool
    reason:str
    confidence:float
    max_step:float|None=None
    reversible:bool=True
    risk:str="low"

@dataclass(frozen=True)
class VerifiedAction:
    proposal:ProposedAction
    before:Any
    after:Any
    readback:Any
    accepted:bool
    rollback_value:Any|None=None
