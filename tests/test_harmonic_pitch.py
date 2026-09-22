import numpy as np
from audio_workbench.editing.harmonic_pitch import choose_contextual_target,correction_from_context
def test_context_can_resolve_ambiguous_note():
 s=np.ones(12)*.01;s[4]=.8
 r=choose_contextual_target(64.42,s)
 assert r["chosen"]["midi"]==64
def test_correction_bounded():
 r=correction_from_context(60.4,60,.5)
 assert abs(r["correction_cents"])<=35
