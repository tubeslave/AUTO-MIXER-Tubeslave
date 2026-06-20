# M078 Replay Evidence Trace Detail Blocker Guard

## Context

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`: real WING Rack read-only
evidence and operator review are still required before physical readback or
supervised live proof. Replay Evidence already blocks its trace status badge,
latest-run status/id, policy row, channel row, report path and start action
while offline integration-bundle blockers are visible.

The Replay Evidence `Trace` detail text still renders raw replay trace counts
or a raw decision-trace action/delta. While the global blocker is visible, that
can make stale or unavailable trace proof look like current promotion evidence
even though replay start is blocked and no backend replay request is allowed
from this surface.

## Scope

- Derive a local Replay Evidence trace-detail display from
  `integrationBundleApplyBlocked`.
- Show `No trace proof` with danger styling while blockers are visible.
- Preserve the raw replay trace count, decision-trace action/delta or
  `No decision trace` when the blocker is absent.
- Add a frontend contract assertion and a WING MVP rail for the visible guard.
- Update roadmap, capability matrix, completion gate, current state, heartbeat,
  last report and run ledger.

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No replay run input mutation.
- No decision trace input mutation.
- No channel evidence input mutation.
- No task data mutation.
- No queue mutation.
- No replay start.
- No supervised apply opening.
- No dispatch of supervised apply.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

- M078-CP01: Replay Evidence derives `Trace` detail label and class from
  `integrationBundleApplyBlocked`.
- M078-CP02: Frontend and WING rails prove the trace-detail surface remains
  presentation-only and cannot grant live authority.

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m078_replay_evidence_trace_detail_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
