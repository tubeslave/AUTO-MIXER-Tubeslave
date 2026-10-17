from __future__ import annotations
FEATURES=("tonal_balance","center_sides","vocal_prominence","drum_density","low_end","space_depth","crest","section_contrast")
def build_map(assignments:dict)->dict:
    out={}
    for feature,refs in assignments.items():
        if feature not in FEATURES:continue
        out[feature]={"references":list(refs),"mode":"feature_only"}
    return {"features":out,"rule":"Never copy a whole reference when the user assigned only one feature."}
