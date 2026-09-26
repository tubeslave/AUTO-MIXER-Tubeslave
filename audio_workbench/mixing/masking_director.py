from __future__ import annotations
from .masking import BANDS

PAIR_RULES=(
 ("vocal","guitar","presence"),
 ("vocal","keys","presence"),
 ("vocal","playback","presence"),
 ("vocal","cymbals","upper_presence"),
 ("kick","bass","low_punch"),
 ("snare","guitar","presence"),
)

def proposed_depth_db(evidence:dict,band_name:str)->float:
    band=BANDS[band_name]
    if evidence["score"]<.18:return 0.
    # Bounded. Evidence changes depth, never exceeds the band's hard ceiling.
    return float(min(band.max_cut_db,.35+band.max_cut_db*evidence["score"]*.75))

def decision(target_role:str,masker_role:str,band_name:str,evidence:dict)->dict:
    depth=proposed_depth_db(evidence,band_name)
    return {"target_role":target_role,"masker_role":masker_role,"band":band_name,
            "depth_db":depth,"apply":depth>=.35,"evidence":evidence}
