import json
from pathlib import Path

import pytest

from tools.security_disposition_check import (
    EXPECTED_COUNT,
    EXPECTED_OCCURRENCE_SET_SHA256,
    LedgerValidationError,
    main,
    validate_ledger,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = (
    REPO_ROOT / "Docs/reports/CODEX_SECURITY_DISPOSITION_LEDGER_20260819.jsonl"
)


def test_frozen_security_ledger_has_exact_sealed_identity_and_denies_authority():
    summary = validate_ledger(LEDGER_PATH)

    assert summary["row_count"] == EXPECTED_COUNT
    assert summary["severity_counts"] == {"high": 77, "medium": 59, "low": 7}
    assert summary["occurrence_set_sha256"] == EXPECTED_OCCURRENCE_SET_SHA256
    assert summary["executes_ledger_commands"] is False
    assert summary["hardware_contacted"] is False
    assert summary["live_authority_granted"] is False


def test_security_ledger_rejects_replaced_occurrence_even_with_same_row_count(tmp_path):
    rows = [json.loads(line) for line in LEDGER_PATH.read_text().splitlines()]
    rows[0]["occurrence_id"] = "occ_" + ("f" * 32)
    candidate = tmp_path / "ledger.jsonl"
    candidate.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(LedgerValidationError, match="occurrence set digest mismatch"):
        validate_ledger(candidate)


def test_security_ledger_cli_fails_closed_on_missing_input(tmp_path, capsys):
    result = main(["--ledger", str(tmp_path / "missing.jsonl")])

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "invalid"
