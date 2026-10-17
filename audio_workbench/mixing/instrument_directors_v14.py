from __future__ import annotations
POLICIES={
 "vocal":{"deess_band_hz":(5000,9500),"deess_max_gr_db":2.0,"fx":"section-aware-later"},
 "bass":{"kick_sidechain_max_gr_db":1.2,"release_ms":95},
 "drums":{"sample_reinforcement":"evidence_only","max_sample_blend_db":-9},
 "guitar":{"width":"preserve_source_or_arrangement","synthetic_double":False},
}
def policy(name): return POLICIES.get(name,{})
