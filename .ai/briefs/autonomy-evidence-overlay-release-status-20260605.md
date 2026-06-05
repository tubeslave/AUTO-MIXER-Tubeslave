# Brief: Autonomy Evidence Overlay Release Status

Task id: autonomy-evidence-overlay-release-status-20260605

## Context

`evaluate_supervised_proof_artifacts.py` already builds an
`autonomy_evidence_overlay` that maps current supervised fader/gain proof
artifacts to autonomy promotion targets. The no-WING release summary and
frontend module-status adapter did not yet make that dynamic overlay visible.

## Goal

Expose the overlay as read-only release/operator evidence so the cockpit can
show whether fader/gain promotion targets have no evidence, dry-run evidence,
blocked artifacts, or physical candidates. The Product Layer should read the
latest supervised proof report artifact and surface a sanitized overlay in
`get_processing_pipeline_snapshot`.

## Constraints

- Do not create a live executor.
- Do not update safety rows automatically.
- Keep Full Autonomous blocked.
- Keep all overlay rows `canWriteLive=false` in frontend status output.
- Validate the overlay before trusting it in no-WING release summaries.
- Normalize proof-report artifact safety before exposing it to the cockpit.

## Likely Files

- `tools/no_wing_release_check.py`
- `backend/operator_product_state.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_operator_product_state.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `.ai/memory/project.md`
- `Docs/adr/autonomy-evidence-overlay-release-status-20260605.md`
