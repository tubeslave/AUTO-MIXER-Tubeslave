import numpy as np
import pytest
from audio_workbench.mastering import MasteringConfig, MasteringDirector
from audio_workbench.mastering import maximizer
from audio_workbench.mastering.analyzer import true_peak_dbtp


def config():
    return MasteringConfig(stabilizer=False,clarity=False,impact=False,clipper=False,
                           maximizer=True,maximizer_drive_db=0,ceiling_db=-1.2)


def test_real_maximizer_intersample_overshoot_is_attenuated_before_gate():
    t=np.arange(24000,dtype=np.float64)/48000
    mono=(1.5*np.sin(2*np.pi*6000*t+np.pi*.75)).astype('float32')
    x=np.column_stack([mono,mono*.8]); original=x.copy()
    before,limiter=maximizer.process(x,48000,-1.2,0)
    assert true_peak_dbtp(before)>-1.1
    y,report=MasteringDirector(config()).render(x,48000)
    safety=report['budget']['true_peak_safety']
    assert safety['attenuation_db']>0
    assert safety['mode']=='linked_static_attenuation'
    assert safety['adds_gain'] is False
    assert true_peak_dbtp(y)<=-1.19999
    assert report['budget']['maximizer']['final_max_gr_db']==pytest.approx(limiter['final_max_gr_db'])
    np.testing.assert_array_equal(x,original)
    np.testing.assert_allclose(y,before*10**(-safety['attenuation_db']/20),rtol=2e-6,atol=1e-7)


def test_true_peak_guard_does_not_boost_a_quiet_output():
    t=np.arange(24000)/48000
    m=(.03*np.sin(2*np.pi*997*t)).astype('float32');x=np.column_stack([m,m*.8])
    expected,_=maximizer.process(x,48000,-1.2,0)
    y,report=MasteringDirector(config()).render(x,48000)
    assert report['budget']['true_peak_safety']['attenuation_db']==0
    np.testing.assert_array_equal(y,expected)
