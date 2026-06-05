# Proposal: no-wing-read-remediation-summary-gate-20260605

## 1. Short thesis

no-WING release artifacts should validate that the remediation plan agrees with
the physical readback target reported by fixture evidence.

## 2. What I understood

The project goal needs complete console read coverage before autonomous control.
We now expose the next readback action in backend/cockpit, but release summaries
do not yet prove that this action is synchronized with fixture evidence.

## 3. Proposed solution

Add `validate_read_surface_remediation_plan_snapshot()` to
`tools/no_wing_release_check.py`. It builds the fixture evidence snapshot, builds
the remediation plan from that snapshot, checks safety flags, verifies physical
gap IDs, and ensures `next_physical_readback_action_id` equals
`remediate_<next_physical_readback_surface_id>`. It also rejects ambiguous plans
where more than one action is flagged as `next_physical_readback_target`. Embed
the resulting summary in `summary.json` and `summary.md`.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/no-wing-read-remediation-summary-gate-20260605.md`
- `Docs/adr/no-wing-read-remediation-summary-gate-20260605.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Trust cockpit/frontend tests. Rejected because release artifacts should be
  self-validating.
- Fold remediation into fixture evidence summary. Rejected because fixture
  coverage and remediation planning are separate evidence layers.

## 6. Risks

The release summary may look like a work queue. Mitigation: all fields remain
read-only and non-executing, and the gate validates no live-write claim.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- Focused remediation target/action mismatch and duplicate-next-target tests.
- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py tests/test_read_surface_remediation_plan.py tests/test_processing_registry.py -q`
- `python3 tools/memory/memory_lint.py`

## 8. Where the other agent may disagree

Kimi may prefer only validating fixture evidence. I would add the remediation
gate because autonomous progress needs a proof target and its next action to
stay synchronized in release artifacts.
