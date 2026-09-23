"""File-oriented STUDIO mastering: controller selection is not human acceptance."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import soundfile as sf

from .analyzer import analyze
from .pipeline import MasteringConfig
from .target_controller import MasteringTargetController, MasteringTargetSearchConfig


def audio_digest(audio: np.ndarray, sample_rate: int) -> str:
    data = np.ascontiguousarray(audio, dtype='<f4')
    digest = hashlib.sha256()
    digest.update(json.dumps({'sample_rate': sample_rate, 'shape': data.shape}).encode())
    digest.update(memoryview(data).cast('B'))
    return digest.hexdigest()


def render_offline_master(
    audio: np.ndarray, sample_rate: int, *, target_lufs: float = -14.5,
    ceiling_dbtp: float = -1.2, search_config: MasteringTargetSearchConfig | None = None,
    base_config: MasteringConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the real controller, keeping a rejected candidate out of delivery."""
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError('offline mastering requires samples x two stereo channels')
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate < 12000:
        raise ValueError('sample rate must support the mastering crossover bands')
    if len(source) < int(.4 * sample_rate) or not np.isfinite(source).all():
        raise ValueError('at least 400 ms of finite audio is required')
    if not np.isfinite(target_lufs) or not -40 <= target_lufs <= -5:
        raise ValueError('target_lufs must be finite and in [-40, -5]')
    if not np.isfinite(ceiling_dbtp) or not -12 <= ceiling_dbtp <= -1:
        raise ValueError('ceiling must preserve at least 1 dB true-peak headroom')
    if not np.any(np.abs(source) > 1e-12):
        raise ValueError('source is silent')
    original_hash = audio_digest(source, sample_rate)
    before = analyze(source, sample_rate, include_true_peak=True, include_loudness=True)
    measured = before.get('integrated_lufs')
    if measured is None or not np.isfinite(measured):
        raise ValueError('source integrated loudness is unmeasurable or silent')
    if search_config is None:
        center = float(np.clip(target_lufs - measured, -11.0, 11.0))
        search_config = MasteringTargetSearchConfig(
            target_lufs=target_lufs, tolerance_lu=.5,
            pregain_grid_db=(center - 1., center, center + 1.),
            maximizer_drive_grid_db=(0.,), max_candidates=3,
        )
    base_config = base_config or MasteringConfig(
        stabilizer=False, clarity=False, impact=False, clipper=False,
        maximizer=True, ceiling_db=ceiling_dbtp,
    )
    if abs(float(search_config.target_lufs) - target_lufs) > 1e-9:
        raise ValueError('search target differs from delivery target')
    if abs(float(base_config.ceiling_db) - ceiling_dbtp) > 1e-9:
        raise ValueError('controller ceiling differs from delivery ceiling')
    candidate, controller = MasteringTargetController(base_config, search_config).search(source, sample_rate)
    if audio_digest(source, sample_rate) != original_hash:
        raise RuntimeError('controller mutated the source')
    after = analyze(candidate, sample_rate, include_true_peak=True, include_loudness=True)
    if controller['rolled_back_to_source'] and not np.array_equal(candidate, source):
        raise RuntimeError('controller rollback changed source samples')
    report = {
        'schema': 'studio-offline-master-v1', 'source_sha256': original_hash,
        'candidate_sha256': audio_digest(candidate, sample_rate),
        'sample_rate': int(sample_rate), 'frames': len(source), 'channels': 2,
        'pre_master': before, 'post_master': after,
        'controller': controller, 'base_config': asdict(base_config),
        'status': controller['status'], 'baseline_eligible': False,
        'requires_human_listening': not controller['rolled_back_to_source'],
        'input_unchanged': True, 'live_control_used': False,
    }
    return candidate, report


def deliver_master(source_path: str | Path, output_dir: str | Path, *,
                   name: str = 'Master', target_lufs: float = -14.5,
                   ceiling_dbtp: float = -1.2,
                   search_config: MasteringTargetSearchConfig | None = None) -> dict[str, Any]:
    """Export only technically feasible masters, then validate actual WAV and MP3.

    A failed controller produces a report but no misleading 'mastered' audio.
    Post-encoding failures are kept as diagnostic files and clearly rejected.
    Existing output directories are never overwritten.
    """
    source_path = Path(source_path).resolve()
    out = Path(output_dir)
    if Path(name).name != name or name in {'', '.', '..'}:
        raise ValueError('name must be a plain filename stem')
    if out.exists():
        raise FileExistsError(f'output directory already exists: {out}')
    x, sr = sf.read(source_path, dtype='float32', always_2d=True)
    source_file_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    y, report = render_offline_master(x, sr, target_lufs=target_lufs,
                                      ceiling_dbtp=ceiling_dbtp, search_config=search_config)
    out.mkdir(parents=True, exist_ok=False)
    report_path = out / f'{name}_report.json'
    report['source_file_sha256'] = source_file_hash
    report['artifacts'] = {}
    if report['status'] != 'rejected':
        wav = out / f'{name}.wav'
        mp3 = out / f'{name}_320.mp3'
        sf.write(wav, y, sr, subtype='PCM_24')
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                        '-i', str(wav), '-c:a', 'libmp3lame', '-b:a', '320k', str(mp3)], check=True)
        decoded = subprocess.check_output(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                                          '-i', str(mp3), '-f', 'f32le', '-acodec', 'pcm_f32le', '-'])
        pcm = np.frombuffer(decoded, dtype='<f4').reshape(-1, 2)
        wav_audio, wav_sr = sf.read(wav, dtype='float32', always_2d=True)
        report['export_measurements'] = {
            'wav': analyze(wav_audio, wav_sr, include_true_peak=True, include_loudness=True),
            'mp3_decoded': analyze(pcm, sr, include_true_peak=True, include_loudness=True),
        }
        failures = []
        for fmt, metrics in report['export_measurements'].items():
            if not np.isfinite(metrics['true_peak_dbtp']) or metrics['true_peak_dbtp'] > -1.0:
                failures.append(f'{fmt}_true_peak_exceeds_minus_1_dbtp')
            lu = metrics['integrated_lufs']
            if lu is None or not np.isfinite(lu) or abs(lu-target_lufs) > .6:
                failures.append(f'{fmt}_loudness_target_missed')
        if len(wav_audio) != len(x) or len(pcm) != len(x):
            failures.append('export_length_mismatch')
        report['export_failures'] = failures
        if failures:
            report['status'] = 'rejected_export_validation'
        report['artifacts'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (wav, mp3)}
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_file_hash:
        raise RuntimeError('source file changed during mastering')
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report
