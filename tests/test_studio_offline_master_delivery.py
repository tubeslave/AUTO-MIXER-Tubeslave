import hashlib
import json
import numpy as np
import pytest
import soundfile as sf
from audio_workbench.mastering import MasteringTargetSearchConfig
from audio_workbench.mastering.offline import render_offline_master, deliver_master


def tone():
    t=np.arange(48000)/48000
    m=(.03*np.sin(2*np.pi*997*t)).astype('float32')
    return np.column_stack([m,m*.8])


def test_actual_controller_delivery(tmp_path):
    x=tone();source=tmp_path/'input.wav';sf.write(source,x,48000,subtype='FLOAT')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    report=deliver_master(source,tmp_path/'delivery',target_lufs=-22.)
    assert report['status']=='pending_human_review'
    assert report['baseline_eligible'] is False
    assert report['controller']['chosen']['machine_safe'] is True
    assert len(report['artifacts'])==2
    assert report['export_failures']==[]
    assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
    stored=json.loads((tmp_path/'delivery/Master_report.json').read_text())
    assert stored['source_file_sha256']==digest
    with pytest.raises(FileExistsError): deliver_master(source,tmp_path/'delivery')


def test_rejected_target_preserves_source_without_fake_master(tmp_path):
    x=tone();original=x.copy()
    config=MasteringTargetSearchConfig(target_lufs=-5.,pregain_grid_db=(0.,),maximizer_drive_grid_db=(0.,))
    y,report=render_offline_master(x,48000,target_lufs=-5.,search_config=config)
    assert report['status']=='rejected'
    assert report['controller']['rolled_back_to_source']
    assert report['source_sha256']==report['candidate_sha256']
    np.testing.assert_array_equal(y,original);np.testing.assert_array_equal(x,original)
    source=tmp_path/'input.wav';sf.write(source,x,48000,subtype='FLOAT')
    delivered=deliver_master(source,tmp_path/'rejected',target_lufs=-5.,search_config=config)
    assert delivered['artifacts']=={}
    assert not list((tmp_path/'rejected').glob('*.wav'))


@pytest.mark.parametrize('kind',['nan','mono','short','silence'])
def test_invalid_audio_fails_closed(kind):
    x=tone()
    if kind=='nan': x[200,0]=np.nan
    if kind=='mono': x=x[:,0]
    if kind=='short': x=x[:100]
    if kind=='silence': x[:]=0
    with pytest.raises(ValueError): render_offline_master(x,48000)
