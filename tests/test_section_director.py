import numpy as np
from audio_workbench.mastering.section_director import analyze_sections,parameters_for_density
def test_density_detects_louder_half():
 sr=1000
 a=np.zeros((sr*8,2),dtype="float32")
 rng=np.random.default_rng(1);a[:sr*4]=rng.normal(0,.02,(sr*4,2));a[sr*4:]=rng.normal(0,.15,(sr*4,2))
 r=analyze_sections(a,sr,.5)
 d=np.array(r["density"]);assert d[len(d)//2:].mean()>d[:len(d)//2].mean()
def test_parameters_bounded():
 lo=parameters_for_density(0);hi=parameters_for_density(1)
 assert .05<=lo["clarity_strength"]<=.15
 assert hi["impact_max_gr_db"]<=1.0
