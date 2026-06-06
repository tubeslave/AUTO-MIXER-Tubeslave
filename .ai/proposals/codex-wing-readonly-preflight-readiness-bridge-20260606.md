# Codex Proposal: WING Read-Only Preflight Readiness Bridge

## Short Thesis

Wire the read-only WING preflight evidence into the existing autonomy readiness
snapshot as a non-authoritative overlay, so real evidence can drive manual
readiness review without widening live authority.

## What I Understood

The project needs the supervised WING Rack MVP to move through concrete proof
blockers. The previous slice created a transport-free evaluator for
`wing_readonly_preflight.v1`, but `autonomy_promotion_readiness` still did not
surface that evidence in the readiness audit used by downstream gates and
reports.

## Proposed Solution

- Import the transport-free preflight evidence builder into
  `backend/autonomy_promotion_readiness.py`.
- Add a `wing_readonly_preflight_evidence` summary overlay.
- Report candidate blocker ids only when the evidence is valid, invalid count
  is zero, and no closure/live/autonomous flags are set.
- Keep `PromotionReadinessRow`, `SafetyCoverageRow`, and live-write policy
  untouched.
- Add tests for missing evidence and valid candidate evidence.

## Likely Files To Touch

- `backend/autonomy_promotion_readiness.py`
- `tests/test_autonomy_promotion_readiness.py`
- `.ai/memory/project.md`
- `Docs/INDEX.md`
- `Docs/adr/wing-readonly-preflight-readiness-bridge-20260606.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_READINESS_BRIDGE_20260606.md`

## Alternatives Considered

- Update safety coverage rows automatically from valid preflight evidence.
  Rejected because read-only preflight is not apply, rollback, or emergency-stop
  proof.
- Add another standalone readiness method. Rejected because the existing
  promotion readiness snapshot is the canonical audit surface.

## Risks

A valid preflight candidate could be mistaken for completed hardware proof. The
overlay therefore keeps `closed_blocker_ids=[]`, live authority false, and a
plain claim that it supports manual readiness review only.

## Test Plan

- `python3 -m py_compile backend/autonomy_promotion_readiness.py backend/wing_readonly_preflight_evidence.py backend/processing_registry.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_autonomy_promotion_readiness.py tests/test_wing_readonly_preflight_evidence.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_processing_registry.py::test_registry_snapshot_can_hide_full_method_inventory tests/test_no_wing_release_check.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may argue that readiness should remain artifact-agnostic until hardware
proof exists. I would keep this overlay because it does not close proof; it
makes the exact missing or candidate preflight state visible in the canonical
readiness surface.
