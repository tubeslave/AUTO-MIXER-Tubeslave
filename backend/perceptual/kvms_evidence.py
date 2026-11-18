"""Read-only analysis of blinded listening evidence for KVMS.

This module DOES NOT score subjective audio quality, optimize DSP, or provide
mixer actions. It verifies evidence bookkeeping and produces a research summary.
No dependence on audio-thread or network services.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping
import argparse
import json
import math

SCHEMA = 'kvms-blind-result/1.1'


class InvalidEvidence(ValueError):
    """Evidence is incomplete or internally inconsistent."""


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise InvalidEvidence(f'{label} must be a number')
    if not math.isfinite(float(value)):
        raise InvalidEvidence(f'{label} must be finite')
    return float(value)


@dataclass(frozen=True)
class EvidenceSummary:
    experiment_id: str
    track: str
    sections: int
    independent_listeners: int
    counts_by_mix: dict[str, int]
    unanimous_choice: str | None
    loudness_match_bound_lu: float
    integrity_verified: bool
    study_class: str
    can_auto_apply: bool
    candidate_for_future_validation: str | None
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_evidence(record: Mapping[str, Any]) -> EvidenceSummary:
    """Validate a KVMS blind-result record and summarize within-study votes.

    Strict validation prevents accidental inversion of the blind key. A single
    listener across many song sections is *one* independent listener, never N.
    No result from this function authorizes DSP or console commands.
    """
    if not isinstance(record, Mapping) or record.get('schema_version') != SCHEMA:
        raise InvalidEvidence(f'Expected schema {SCHEMA}')
    experiment = record.get('experiment_id')
    if not isinstance(experiment, str) or not experiment.strip():
        raise InvalidEvidence('experiment_id missing')
    track = record.get('track')
    if not isinstance(track, str) or not track.strip():
        raise InvalidEvidence('track missing')
    n_listeners = record.get('listener_n')
    if isinstance(n_listeners, bool) or not isinstance(n_listeners, int) or n_listeners < 1:
        raise InvalidEvidence('listener_n must be positive int')
    if record.get('blinding') != 'key_revealed_after_listener_choices':
        raise InvalidEvidence('Missing confirmation that choices preceded key reveal')
    blinded = record.get('blind_key')
    choices = record.get('revealed_choices')
    if not isinstance(blinded, Mapping) or not isinstance(choices, list) or not choices:
        raise InvalidEvidence('Missing blind key or choices')
    if len(choices) != len(blinded):
        raise InvalidEvidence('Section count mismatch between blind key and choices')
    seen: set[str] = set()
    results: list[str] = []
    for item in choices:
        if not isinstance(item, Mapping):
            raise InvalidEvidence('Choice must be an object')
        name = item.get('id')
        letter = item.get('selected_letter')
        mix = item.get('selected_mix')
        if not isinstance(name, str) or not name or name in seen or name not in blinded:
            raise InvalidEvidence('Unknown, blank, or duplicate section')
        seen.add(name)
        key = blinded[name]
        if not isinstance(key, Mapping) or not key:
            raise InvalidEvidence('Invalid mapping for section')
        if not isinstance(letter, str) or letter not in key:
            raise InvalidEvidence('Selected letter absent from blind key')
        if not isinstance(mix, str) or key[letter] != mix:
            raise InvalidEvidence(f'Blind key contradicts recorded choice in {name}')
        if len(set(key.values())) != len(key):
            raise InvalidEvidence(f'Duplicate version in {name} blind key')
        start = _finite_number(item.get('start_s'), f'{name}.start_s')
        end = _finite_number(item.get('end_s'), f'{name}.end_s')
        if start < 0 or end <= start:
            raise InvalidEvidence(f'Invalid time window {name}')
        results.append(mix)
    data = record.get('levels')
    if not isinstance(data, Mapping):
        raise InvalidEvidence('Missing level-match evidence')
    bound = _finite_number(data.get('actual_max_within_section_lufs_difference_below'), 'level_match_bound')
    if bound < 0:
        raise InvalidEvidence('Negative level difference bound')
    integrity = record.get('signal_integrity', {})
    integrity_ok = (
        integrity.get('audio_hashes_verified') is True
        or integrity.get('all_nine_mp3_hashes_match_existing_verification') is True
    )
    matched = bound <= 0.1  # Evidence-quality criterion, NOT auditory threshold.
    counts = Counter(results)
    unanimous = results[0] if len(counts) == 1 else None
    flags = []
    if not integrity_ok:
        flags.append('Missing verified audio integrity')
    if not matched:
        flags.append('Loudness matching insufficient for clean comparison')
    flags.extend([
        'Repeated song sections from one listener are not independent listeners',
        'No calibrated, multi-song or independent-listener preference model',
        'Observed preference does not establish a causal auditory mechanism',
    ])
    status = 'exploratory_consistent' if unanimous and integrity_ok and matched else 'exploratory_inconclusive'
    # NOTE: Invariant: never auto-apply, regardless of count or confidence.
    return EvidenceSummary(
        experiment_id=experiment, track=track, sections=len(results),
        independent_listeners=n_listeners,
        counts_by_mix=dict(sorted(counts.items())), unanimous_choice=unanimous,
        loudness_match_bound_lu=bound, integrity_verified=integrity_ok,
        study_class=status, can_auto_apply=False,
        candidate_for_future_validation=unanimous if matched and integrity_ok else None,
        limitations=tuple(flags),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate and summarize KVMS blinded listening data; never touch mixer')
    parser.add_argument('input_json', type=Path)
    args = parser.parse_args()
    try:
        record = json.loads(args.input_json.read_text(encoding='utf-8'))
        report = summarize_evidence(record)
    except (InvalidEvidence, OSError, json.JSONDecodeError) as exc:
        parser.exit(status=2, message=f'KVMS evidence rejected: {exc}\n')
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
