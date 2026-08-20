#!/usr/bin/env python3
"""Validate the frozen Codex Security disposition ledger without executing it."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_COUNT = 143
EXPECTED_SEVERITIES = {"high": 77, "medium": 59, "low": 7}
EXPECTED_OCCURRENCE_SET_SHA256 = (
    "80146278415ba2ef4108d7998f7eff70a061a27df1b7afadbcbb926c9c6a57fd"
)
ALLOWED_DISPOSITIONS = {
    "proof_task_required",
    "existing_local_evidence_revalidation_required",
    "locally_verified_unsealed",
    "not_applicable_locally_verified_unsealed",
    "blocked",
}
EXPECTED_SCHEMA = "automixer.codex_security_disposition/v1"


class LedgerValidationError(ValueError):
    """Raised when the security disposition ledger is incomplete or malformed."""


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise LedgerValidationError(f"blank ledger line: {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerValidationError(
                    f"invalid JSON at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, Mapping):
                raise LedgerValidationError(f"ledger row is not an object: {line_number}")
            rows.append(row)
    return rows


def _occurrence_digest(occurrence_ids: Iterable[str]) -> str:
    payload = "".join(f"{occurrence_id}\n" for occurrence_id in sorted(occurrence_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_ledger(path: Path) -> dict[str, Any]:
    """Validate ledger identity, schema, severity, and fail-closed status fields.

    Args:
        path: JSONL ledger path.

    Returns:
        A compact validation summary.

    Raises:
        LedgerValidationError: If any completeness or schema invariant fails.
    """

    rows = _load_rows(path)
    if len(rows) != EXPECTED_COUNT:
        raise LedgerValidationError(
            f"unexpected row count: {len(rows)} != {EXPECTED_COUNT}"
        )

    occurrence_ids: list[str] = []
    severities: Counter[str] = Counter()
    dispositions: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        if row.get("schema") != EXPECTED_SCHEMA:
            raise LedgerValidationError(f"invalid schema at line {index}")
        occurrence_id = row.get("occurrence_id")
        if not isinstance(occurrence_id, str) or not occurrence_id.startswith("occ_"):
            raise LedgerValidationError(f"invalid occurrence_id at line {index}")
        severity = row.get("severity")
        if severity not in EXPECTED_SEVERITIES:
            raise LedgerValidationError(f"invalid severity at line {index}")
        disposition = row.get("disposition")
        if disposition not in ALLOWED_DISPOSITIONS:
            raise LedgerValidationError(f"invalid disposition at line {index}")
        if not isinstance(row.get("title"), str) or not row["title"].strip():
            raise LedgerValidationError(f"missing title at line {index}")
        if not isinstance(row.get("locations"), list) or not row["locations"]:
            raise LedgerValidationError(f"missing locations at line {index}")
        if row.get("closure_authority") != "fresh_full_scan":
            raise LedgerValidationError(f"invalid closure authority at line {index}")
        if row.get("hardware_contacted") is not False:
            raise LedgerValidationError(f"hardware claim must remain false at line {index}")
        if row.get("live_authority_granted") is not False:
            raise LedgerValidationError(f"live authority must remain false at line {index}")
        occurrence_ids.append(occurrence_id)
        severities[severity] += 1
        dispositions[disposition] += 1

    if len(set(occurrence_ids)) != EXPECTED_COUNT:
        raise LedgerValidationError("duplicate occurrence_id")
    occurrence_digest = _occurrence_digest(occurrence_ids)
    if occurrence_digest != EXPECTED_OCCURRENCE_SET_SHA256:
        raise LedgerValidationError("occurrence set digest mismatch")
    if dict(severities) != EXPECTED_SEVERITIES:
        raise LedgerValidationError(
            f"severity histogram mismatch: {dict(severities)!r}"
        )

    return {
        "status": "valid",
        "row_count": len(rows),
        "severity_counts": dict(severities),
        "disposition_counts": dict(dispositions),
        "occurrence_set_sha256": occurrence_digest,
        "executes_ledger_commands": False,
        "hardware_contacted": False,
        "live_authority_granted": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("Docs/reports/CODEX_SECURITY_DISPOSITION_LEDGER_20260819.jsonl"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the data-only ledger validation CLI."""

    args = _build_parser().parse_args(argv)
    try:
        summary = validate_ledger(args.ledger)
    except (LedgerValidationError, OSError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
