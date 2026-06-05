# Proposal: read-surface-remediation-physical-target-20260605

## 1. Short thesis

Read-surface remediation should plan against physical fixture evidence, not only
inventory row status.

## 2. What I understood

The project goal requires full console observability before broad autonomous
control. `read_surface_fixture_evidence` already reports the next physical
readback gap, but `read_surface_remediation_plan` can omit a cached surface such
as `raw_mixer_state_cache` because cached inventory status is not considered
incomplete.

## 3. Proposed solution

Have `build_read_surface_remediation_plan()` consume the transport-free fixture
evidence snapshot by default. Create actions for any surface that is either
inventory-incomplete or has `physical_readback_fixture_missing`. Add summary
fields for physical gap IDs, next physical readback target, and the target
action ID. For physical-only gaps, use `add_physical_readback_fixture`.

## 4. Likely files to touch

- `backend/read_surface_remediation_plan.py`
- `tests/test_read_surface_remediation_plan.py`
- `.ai/briefs/read-surface-remediation-physical-target-20260605.md`
- `Docs/adr/read-surface-remediation-physical-target-20260605.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Keep remediation based only on inventory status. Rejected because the next
  physical target can be hidden.
- Read latest no-WING artifact from disk in the plan. Rejected because registry
  snapshots should be deterministic and transport/file-path agnostic.
- Promote cached surfaces to `partial`. Rejected because cache coverage and
  physical proof are separate axes.

## 6. Risks

Operators may read a physical fixture action as runtime capability. Mitigation:
the action stays read-only, non-mutating, and uses explicit physical-proof
contract names.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_remediation_plan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_inventory.py tests/test_read_surface_fixture_evidence.py tests/test_processing_registry.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Where the other agent may disagree

Kimi may prefer leaving cached surfaces out of remediation actions. I would keep
them in when fixture evidence names a physical gap, because autonomous readiness
needs both cache coverage and physical proof.
