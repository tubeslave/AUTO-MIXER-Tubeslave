# Proposal: source-identity-memory-trust-promotion-record-safety-20260606

## Short Thesis

Promotion-contract records should be self-contained safety evidence, not just
children of a safe contract envelope.

## What I Understood

The current source identity memory trust promotion contract is read-only and
transport-free. The small remaining ambiguity is representational: a UI,
report, or downstream read model may inspect a single `contract_records` row
without also reading the parent `safety` object.

## Proposed Solution

- Add explicit record-level safety flags:
  `read_only=true`, `live_write=false`, `mutates_console_state=false`,
  `queries_console=false`, `rollback_evidence_required=true`, and
  `live_widening_allowed=false`.
- Extend focused backend and Product Layer tests to assert the fields.
- Add a project-memory lesson and ADR for the evidence-shape decision.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_trust_promotion_contract.py`
- `tests/test_source_identity_calibration_memory_trust_promotion_contract.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-memory-trust-promotion-record-safety-20260606.md`

## Alternatives Considered

- Rely on the parent contract `safety` block only. Rejected because record-level
  consumers may render or export rows independently.
- Promote the records into executable actions. Rejected because this slice is
  evidence-only and still lacks an audited executor and rollback apply path.

## Risks

The change adds more repeated false-authority fields. That repetition is
intentional here because safety evidence should survive partial rendering.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_contract.py tests/test_operator_product_state.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_contract.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi may argue the parent safety envelope is enough. I would keep the record
flags because this is a read-model contract and records can be consumed apart
from their parent summary.
