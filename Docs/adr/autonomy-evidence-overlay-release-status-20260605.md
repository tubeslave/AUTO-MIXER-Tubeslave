# ADR: Autonomy Evidence Overlay Release Status

## Context

The supervised proof artifact report already includes
`autonomy_evidence_overlay`, a read-only mapping from fader/gain proof artifacts
to autonomy promotion targets. The no-WING release summary, Product Layer
processing snapshot, and frontend module status adapter did not expose this
target-level evidence.

## Options Considered

1. Leave the overlay only inside the raw proof report.
2. Surface the overlay in release summaries and frontend adapter rows.
3. Read the latest supervised proof report artifact in Product Layer and expose
   a sanitized overlay through `get_processing_pipeline_snapshot`.
4. Use the overlay to automatically update safety coverage.

## Decision

Use options 2 and 3. The overlay becomes visible read-only release/operator
evidence and is also available in the backend processing snapshot after safety
normalization. It does not grant apply authority, update safety rows, or unblock
Full Autonomous mode.

## Why This Won

The project needs explainable progress toward autonomy. Showing target-specific
proof evidence helps operators see what remains missing without confusing
dry-run artifacts with live readiness.

## Rejected Alternatives

- Raw-report only: too easy for operator-facing readiness to miss the evidence.
- Product Layer artifact generation: snapshot calls must only read existing
  artifacts and must not create proof runs.
- Automatic safety update: unsafe without manual review and physical WING
  proof.

## Implementation Plan

- Validate overlay schema in `tools/no_wing_release_check.py`.
- Embed a compact overlay section into no-WING `summary.json` and `summary.md`.
- Add a read-only Product Layer supervised proof evidence snapshot that reads
  existing report artifacts and normalizes unsafe source flags.
- Include `artifacts/no_wing_release_check_current_cycle` alongside the
  standard no-WING release artifact roots so current-cycle proof evidence is
  visible without extra runtime configuration.
- Report proof artifact freshness (`latest_report_age_sec`, `stale_after_sec`,
  `latest_report_stale`) and mark stale source rows separately in the cockpit.
- Adapt `supervised_proof_evidence` and `autonomy_evidence_overlay.rows` into
  frontend module status records, separating proof-source availability from
  target-level evidence.
- Add focused backend and frontend contract tests.

## Test Plan

- Run no-WING release-check unit tests.
- Run autonomy overlay tests.
- Run Product Layer operator snapshot tests.
- Run frontend adapter contract tests.
- Run memory lint.

## Risks And Rollback

Risk: overlay rows could be interpreted as live controls. Mitigation: every row
is read-only, `canWriteLive=false`, and the summary/source rows require
`full_autonomous_ready=false`. Risk: untrusted artifacts could claim live
authority. Mitigation: Product Layer and frontend rows force `live_write=false`,
`automatic_safety_update_allowed=false`, and clears autonomous-ready targets.
Risk: old proof artifacts could be mistaken for current proof. Mitigation:
source rows expose age and stale state before target-level evidence is reviewed.
Rollback is to remove the summary/helper, Product Layer loader, and frontend
overlay row mapping; the underlying proof report remains unchanged.
