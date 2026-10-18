import numpy as np
from audio_workbench.editing.groove import infer_straight_grid,isolated_outliers
def test_isolated_only():
 step=60/86/4;t=np.arange(30)*step+.02
 t[12]+=.045
 g={"step_s":step,"phase_s":.02}
 o=isolated_outliers(t,g)
 assert len(o)==1 and abs(o[0]["nudge_ms"])<=12
