#!/usr/bin/env python3
"""One measured presence hypothesis on a fresh Ptitsa mix, followed by mastering."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import numpy as np
from scipy import signal
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_workbench.mixing import perceptual_critic
from audio_workbench.mastering.analyzer import analyze
from audio_workbench.mastering.offline import audio_digest, deliver_master


def presence_cut(x: np.ndarray, sr: int, depth_db: float = .8) -> np.ndarray:
    """Subtractive zero-phase broad-band control, identical on both channels."""
    if not 0 <= depth_db <= 1:
        raise ValueError('presence hypothesis exceeds its 1 dB bound')
    sos=signal.butter(3,[2500,6500],btype='bandpass',fs=sr,output='sos')
    component=signal.sosfiltfilt(sos,x,axis=0)
    return (x+component*(10**(-depth_db/20)-1)).astype('float32')


def run(source_dir: str | Path, output_dir: str | Path, *, target_lufs: float=-15.5,
        ceiling_dbtp: float=-1.3) -> dict:
    source_dir=Path(source_dir);out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=False)
    x,sr=sf.read(source_dir/'Ptitsa_Fresh_Premaster.wav',dtype='float32',always_2d=True)
    source_hash=audio_digest(x,sr)
    mix_report=json.loads((source_dir/'mix_report.json').read_text())
    references={k:sf.read(source_dir/'buses'/f'{k}.wav',dtype='float32',always_2d=True)[0] for k in ('lead','drums','room')}
    def section_levels(audio):
        hop=int(4*sr)
        return [float(20*np.log10(np.sqrt(np.mean(audio[i:i+hop].astype('float64')**2))+1e-15))
                for i in range(0,len(audio),hop)]
    before=perceptual_critic.snapshot(x,sr,vocal=references['lead'],drums=references['drums'],early_room=references['room'],section_rms_db=section_levels(x))
    y=presence_cut(x,sr)
    filtered={k:presence_cut(v,sr) for k,v in references.items()}
    after=perceptual_critic.snapshot(y,sr,vocal=filtered['lead'],drums=filtered['drums'],early_room=filtered['room'],section_rms_db=section_levels(y))
    verdict=perceptual_critic.accept_candidate(before,after,'harshness')
    chosen=y if verdict['machine_decision']!='rejected' else x.copy()
    report={'schema':'ptitsa-presence-hypothesis-v1','source_sha256':source_hash,
            'candidate_sha256':audio_digest(y,sr),'chosen_sha256':audio_digest(chosen,sr),
            'hypothesis':{'target':'harshness','band_hz':[2500,6500],'max_cut_db':.8,
                          'basis':'fresh-mix sustained spectral prominence proxy .7617; no taste score'},
            'before':asdict(before),'after':asdict(after),'perceptual_gate':verdict,
            'rolled_back':verdict['machine_decision']=='rejected',
            'baseline_promoted':False,'human_review':'pending',
            'source_unchanged':audio_digest(x,sr)==source_hash,
            'selection_scope':'audition candidate only, not an accepted baseline'}
    print('PRESENCE HYPOTHESIS',report,flush=True)
    sf.write(out/'Ptitsa_Fresh_Premaster.wav',chosen,sr,subtype='PCM_24')
    report['chosen_metrics']=analyze(chosen,sr,include_true_peak=True,include_loudness=True)
    (out/'refinement_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (out/'mix_report.json').write_text(json.dumps(mix_report,indent=2,allow_nan=False))
    del x,y,chosen,references,filtered;gc.collect()
    mastering=deliver_master(out/'Ptitsa_Fresh_Premaster.wav',out/'master',
                            name='Ptitsa_Fresh_Master',target_lufs=target_lufs,ceiling_dbtp=ceiling_dbtp)
    print('MASTER COMPLETE',mastering['status'],mastering['post_master'],flush=True)
    return mastering


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source_dir');p.add_argument('output_dir')
    p.add_argument('--target-lufs',type=float,default=-15.5)
    p.add_argument('--ceiling-dbtp',type=float,default=-1.3)
    a=p.parse_args();run(a.source_dir,a.output_dir,target_lufs=a.target_lufs,ceiling_dbtp=a.ceiling_dbtp)
