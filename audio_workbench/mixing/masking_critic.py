from __future__ import annotations

def evaluate(before:dict,after:dict)->dict:
    fail=[]
    # We want target/masker separation to improve, but reject excessive global tonal motion.
    for k,b in before.items():
        a=after.get(k,b)
        if a.get("separation_change_db",0)<-.15:fail.append(f"{k}:separation_worse")
    if after.get("mix_spectral_max_shift_db",0)>1.5:fail.append("mix_tonal_shift_excess")
    if after.get("mix_spectral_rms_shift_db",0)>.75:fail.append("mix_tonal_shift_excess_rms")
    return {"accept":not fail,"failures":fail}
