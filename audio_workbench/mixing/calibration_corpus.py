"""Append-only calibration evidence corpus for STUDIO human/machine replay.

This module is deliberately evidence-only.  It cannot change Perceptual Critic
thresholds, waive protected regressions, change Autonomous Iteration decisions,
or promote an audio baseline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .perceptual_calibration import classify_human_machine_disagreement


_SCHEMA = "studio-calibration-corpus-record-v1"
_REPLAY_FIELDS = (
    "classification",
    "calibration_action",
    "machine_failures_preserved",
    "protected_regressions_preserved",
    "metric_hypotheses",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _record_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload.pop("record_id", None)
    return payload


def calibration_record_id(record: dict[str, Any]) -> str:
    """Return the SHA-256 digest of the canonical record payload."""
    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary")
    encoded = _canonical_json(_record_payload(record)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_key(source_evidence: dict[str, Any]) -> str:
    if not isinstance(source_evidence, dict) or not source_evidence:
        raise ValueError("source_evidence must be a non-empty dictionary")
    return hashlib.sha256(_canonical_json(source_evidence).encode("utf-8")).hexdigest()


def build_calibration_record(
    *,
    source_evidence: dict[str, Any],
    subject: dict[str, Any],
    critic_result: dict[str, Any],
    human_review: dict[str, Any],
    previous_record_id: str | None = None,
) -> dict[str, Any]:
    """Build one deterministic evidence-only calibration record."""
    if previous_record_id is not None:
        previous_record_id = str(previous_record_id).strip()
        if len(previous_record_id) != 64:
            raise ValueError("previous_record_id must be a SHA-256 hex digest or None")
        try:
            int(previous_record_id, 16)
        except ValueError as exc:
            raise ValueError("previous_record_id must be a SHA-256 hex digest or None") from exc
    if not isinstance(subject, dict) or not subject:
        raise ValueError("subject must be a non-empty dictionary")
    if not isinstance(critic_result, dict) or not critic_result:
        raise ValueError("critic_result must be a non-empty dictionary")
    if not isinstance(human_review, dict) or not human_review:
        raise ValueError("human_review must be a non-empty dictionary")

    replay = classify_human_machine_disagreement(critic_result, human_review)
    record = {
        "schema": _SCHEMA,
        "previous_record_id": previous_record_id,
        "source_evidence": dict(source_evidence),
        "source_key": _source_key(source_evidence),
        "subject": dict(subject),
        "critic": dict(critic_result),
        "human_review": dict(human_review),
        "replay_result": {
            field: replay[field]
            for field in _REPLAY_FIELDS
        },
        "evidence_only": True,
        "production_threshold_update_allowed": False,
        "protected_gate_override_allowed": False,
        "baseline_promotion_allowed": False,
        "requires_human_listening": True,
    }
    record["record_id"] = calibration_record_id(record)
    return record


def replay_calibration_record(
    record: dict[str, Any],
    *,
    expected_previous_record_id: str | None,
) -> dict[str, Any]:
    """Validate one record and deterministically replay its classification."""
    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary")
    if record.get("schema") != _SCHEMA:
        raise ValueError("unsupported calibration corpus record schema")
    if record.get("previous_record_id") != expected_previous_record_id:
        raise ValueError("calibration corpus previous-record chain mismatch")
    actual_id = str(record.get("record_id", "")).strip()
    if actual_id != calibration_record_id(record):
        raise ValueError("calibration corpus record digest mismatch")
    if not record.get("evidence_only"):
        raise ValueError("calibration corpus record must be evidence-only")
    if record.get("production_threshold_update_allowed") is not False:
        raise ValueError("calibration corpus cannot update production thresholds")
    if record.get("protected_gate_override_allowed") is not False:
        raise ValueError("calibration corpus cannot override protected gates")
    if record.get("baseline_promotion_allowed") is not False:
        raise ValueError("calibration corpus cannot promote baselines")
    if record.get("requires_human_listening") is not True:
        raise ValueError("calibration corpus must preserve human-listening requirement")

    source = record.get("source_evidence")
    if record.get("source_key") != _source_key(source):
        raise ValueError("calibration corpus source evidence digest mismatch")
    critic = record.get("critic")
    human = record.get("human_review")
    if not isinstance(critic, dict) or not isinstance(human, dict):
        raise ValueError("calibration corpus critic/human evidence is missing")

    replay = classify_human_machine_disagreement(critic, human)
    stored = record.get("replay_result")
    if not isinstance(stored, dict):
        raise ValueError("calibration corpus replay_result is missing")
    expected = {field: replay[field] for field in _REPLAY_FIELDS}
    if stored != expected:
        raise ValueError("calibration corpus replay classification drift")
    return replay


def replay_calibration_corpus(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Replay a complete prefix and reject duplicate source evidence."""
    previous: str | None = None
    seen_sources: set[str] = set()
    count = 0
    classifications: dict[str, int] = {}
    for record in records:
        replay = replay_calibration_record(
            record, expected_previous_record_id=previous
        )
        source_key = str(record["source_key"])
        if source_key in seen_sources:
            raise ValueError("duplicate calibration corpus source evidence")
        seen_sources.add(source_key)
        previous = str(record["record_id"])
        count += 1
        name = str(replay["classification"])
        classifications[name] = classifications.get(name, 0) + 1

    return {
        "schema": "studio-calibration-corpus-replay-v1",
        "records": count,
        "last_record_id": previous,
        "classifications": classifications,
        "evidence_only": True,
        "production_threshold_update_allowed": False,
        "protected_gate_override_allowed": False,
        "baseline_promotion_allowed": False,
        "requires_human_listening": True,
    }


def load_calibration_corpus(path: str | Path) -> list[dict[str, Any]]:
    """Load canonical JSONL records; malformed or blank interior lines fail closed."""
    corpus = Path(path)
    if not corpus.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        text = corpus.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read calibration corpus: {corpus}") from exc
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank calibration corpus line at {lineno}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid calibration corpus JSON at line {lineno}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"calibration corpus line {lineno} must be an object")
        if line != _canonical_json(value):
            raise ValueError(f"calibration corpus line {lineno} is not canonical JSON")
        records.append(value)
    return records


def append_calibration_record(
    path: str | Path,
    *,
    source_evidence: dict[str, Any],
    subject: dict[str, Any],
    critic_result: dict[str, Any],
    human_review: dict[str, Any],
) -> dict[str, Any]:
    """Replay the full prefix, then append exactly one canonical record.

    Existing bytes are never rewritten.  If validation or duplicate detection
    fails, no write is attempted.
    """
    corpus = Path(path)
    existing = load_calibration_corpus(corpus)
    summary = replay_calibration_corpus(existing)
    source_key = _source_key(source_evidence)
    if any(str(record.get("source_key")) == source_key for record in existing):
        raise ValueError("duplicate calibration corpus source evidence")

    record = build_calibration_record(
        source_evidence=source_evidence,
        subject=subject,
        critic_result=critic_result,
        human_review=human_review,
        previous_record_id=summary["last_record_id"],
    )
    corpus.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(record) + "\n"
    with corpus.open("a", encoding="utf-8", newline="") as handle:
        handle.write(encoded)
        handle.flush()

    replayed = load_calibration_corpus(corpus)
    replay_calibration_corpus(replayed)
    return record
