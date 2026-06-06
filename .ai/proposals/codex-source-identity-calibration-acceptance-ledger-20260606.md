# Proposal: source-identity-calibration-acceptance-ledger-20260606

## Short Thesis

Add a read-only acceptance ledger for source identity calibration so operator
decisions become auditable evidence before any separate persistence or
downstream-priority contract exists.

## What I Understood

The project needs reliable automatic source typing before source-specific EQ,
dynamics, FX, pan, balance, and adaptation decisions can be trusted. Replay
calibration already creates review candidates, but accepted/rejected operator
decisions need their own fail-closed evidence layer that cannot be mistaken for
model, memory, priority, queue, or live authority.

## Proposed Solution

- Add `source_identity_calibration_acceptance_ledger.py`.
- Validate acceptance records against current replay-calibration candidates.
- Reject missing candidates, mismatched source types, mismatched trust states,
  unknown decisions, missing reviewers, non-UTC timestamps, and unsafe authority
  claims.
- Expose ledger status in `main_pipeline_preview` without adding proposals to
  the decision queue.
- Register the ledger as `shadow_only` source-identity evidence in the
  processing registry and capability matrix.

## Likely Files To Touch

- `backend/source_identity_calibration_acceptance_ledger.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_acceptance_ledger.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-calibration-acceptance-ledger-20260606.md`

## Alternatives Considered

- Let accepted replay calibration mutate memory immediately. Rejected because
  persistence needs a separate operator-reviewed write contract.
- Import accepted calibration into the operator proposal queue. Rejected because
  calibration acceptance is evidence, not a supervised fader/gain apply action.
- Keep acceptance records external. Rejected because downstream priority and
  long-term memory need auditable, testable evidence before future promotion.

## Risks

- Preview users may assume an accepted calibration record already changed source
  trust state. Mitigate with explicit false flags for model, memory,
  downstream-priority, queue, and live mutations.
- Bad acceptance records could poison future persistence work. Mitigate by
  making malformed or unsafe records force replacement-required status.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_acceptance_ledger.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue that acceptance records should only appear after persistence is
implemented. I would land the ledger first because it creates the fail-closed
audit boundary that persistence must consume later.
