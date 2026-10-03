from __future__ import annotations
import numpy as np
ROLE_POLICY={
 "lead_vocal":{"ratio":2.6,"attack_ms":10,"release_ms":85,"target_p95_gr_db":3.0,"max_gr_db":4.5},
 "back_vocal":{"ratio":2.2,"attack_ms":12,"release_ms":90,"target_p95_gr_db":2.0,"max_gr_db":3.5},
 "bass":{"ratio":2.4,"attack_ms":22,"release_ms":110,"target_p95_gr_db":2.0,"max_gr_db":3.5},
 "kick":{"ratio":2.0,"attack_ms":20,"release_ms":80,"target_p95_gr_db":1.2,"max_gr_db":2.5},
 "snare":{"ratio":2.0,"attack_ms":15,"release_ms":100,"target_p95_gr_db":1.5,"max_gr_db":3.0},
 "guitar":{"ratio":1.5,"attack_ms":25,"release_ms":120,"target_p95_gr_db":.8,"max_gr_db":1.8},
}
def crest_db(x):
    r=np.sqrt(np.mean(x.astype("float64")**2)+1e-20);p=np.max(np.abs(x))+1e-20
    return float(20*np.log10(p/r))
def policy_for(role): return ROLE_POLICY.get(role)
