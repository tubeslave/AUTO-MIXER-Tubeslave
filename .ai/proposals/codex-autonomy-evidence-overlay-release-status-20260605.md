# Codex Proposal: Autonomy Evidence Overlay Release Status

## Short Thesis

Treat `autonomy_evidence_overlay` as a first-class read-only evidence surface in
release summaries and frontend module status rows, while preserving the current
supervised fader/gain-only live boundary.

## What I Understood

The project goal requires visible, explainable progress toward autonomous
mixing. Proof evidence currently exists in the proof artifact report, but the
release runner and cockpit adapter do not fully surface it as target-specific
operator status.

## Proposed Solution

- Add a no-WING schema validator for `autonomy_evidence_overlay`.
- Add a no-WING summary block for overlay status, target ids, blockers, and
  safety flags.
- Add a Product Layer proof evidence snapshot that reads the latest supervised
  proof report artifact without generating artifacts.
- Normalize `autonomy_evidence_overlay` safety before exposing it through
  `get_processing_pipeline_snapshot`.
- Include both standard no-WING release roots and the current-cycle proof root
  as Product Layer read inputs.
- Add frontend adapter rows for `autonomy_evidence_overlay.rows`.
- Add a frontend source-status row for `supervised_proof_evidence` so operators
  can see whether a proof report artifact is available before inspecting target
  overlay rows.
- Add freshness metadata so stale proof reports are shown as review-required
  evidence rather than current promotion evidence.
- Keep frontend rows `canWriteLive=false` and `liveWriteKinds=[]`.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`

## Alternatives Considered

- Regenerate proof reports from Product Layer runtime. Rejected: snapshot calls
  must not create artifacts or inspect live mixers.
- Promote safety coverage automatically from overlay candidates. Rejected:
  physical proof still requires manual safety review.

## Risks

- UI could make dry-run evidence look like live authority. Mitigation: adapter
  rows use separate `supervised_proof_evidence` and
  `autonomy_evidence_overlay` domains and force `canWriteLive=false`.
- A malicious or stale artifact could claim live authority. Mitigation: Product
  Layer normalizes safety flags to read-only, non-live, non-promoting output.
- Stale proof artifacts could look current. Mitigation: Product Layer reports
  `latest_report_age_sec`, `stale_after_sec`, and `latest_report_stale`; the UI
  marks stale artifacts separately.
- Release summaries could overclaim autonomous readiness. Mitigation: validator
  requires `full_autonomous_ready=false` and no automatic safety update.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py tests/test_autonomy_evidence_overlay.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q`
- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `python3 tools/memory/memory_lint.py`

## Where The Other Agent May Disagree

Kimi may prefer a separate websocket route for proof evidence instead of adding
it to `get_processing_pipeline_snapshot`. Keeping it on the processing snapshot
is acceptable for this slice because module-status rows already consume that
payload.
