from __future__ import annotations

def choose_modules(a:dict)->dict:
    # Bounded, explainable v0.2 policy. No genre/reference target required.
    return {
      "stabilizer":True,
      "clarity":True,
      "bass_director": a["bands_db"]["sub"] > a["bands_db"]["mid"]+1.0 or a["bands_db"]["low"] > a["bands_db"]["mid"]+3.5,
      "stereo_director": a["correlation"]>.88 or a["correlation"]<.25,
      "impact": a["crest_db"]>13.0,
      "exciter": a["bands_db"]["air"] < a["bands_db"]["presence"]-4.0,
      "clipper":True,
      "maximizer":True,
    }

def accept(before:dict,after:dict)->dict:
    failures=[]
    if after["correlation"] < 0.0: failures.append("negative_stereo_correlation")
    if after["crest_db"] < before["crest_db"]-5.0: failures.append("excessive_crest_loss")
    if abs(after["side_mid_db"]-before["side_mid_db"])>2.0: failures.append("excessive_width_shift")
    return {"accept":not failures,"failures":failures}
