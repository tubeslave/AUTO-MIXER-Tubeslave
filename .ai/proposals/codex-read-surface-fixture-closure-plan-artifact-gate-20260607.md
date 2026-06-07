# Proposal: Read-Surface Fixture Closure Plan Artifact Gate

## 1. Short thesis

Make manual fixture-closure planning a generated, validated no-WING artifact
without letting it close fixtures or widen live authority.

## 2. What I understood

The project needs complete read-surface proof before broader console autonomy.
After operators capture and accept physical-readback artifacts, the fixture
closure plan is the next boundary. That plan must remain manual and fail-closed,
but it should be machine-readable in release runs.

## 3. Proposed solution

- Add `tools/build_read_surface_physical_readback_fixture_closure_plan.py`.
- Generate Markdown for operators and JSON for no-WING release checks.
- Validate the generated JSON artifact for safety flags, summary counts,
  manual-only closure semantics, candidate safety flags, operator sequence, and
  focused validation command.
- Add the artifact result to `summary.json` and `summary.md`.

## 4. Likely files to touch

- `tools/build_read_surface_physical_readback_fixture_closure_plan.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_fixture_closure_plan.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-fixture-closure-plan-artifact-gate-20260607.md`
- `Docs/adr/read-surface-fixture-closure-plan-artifact-gate-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_FIXTURE_CLOSURE_PLAN_ARTIFACT_GATE_20260607.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Keep the closure plan as an in-memory schema check only.
- Fold closure planning into the acceptance-template gate.
- Automatically close fixture gaps when a closure candidate exists.

## 6. Risks

The main risk is mistaking a closure plan for fixture closure. The artifact must
state that fixture updates, closure, live authority, and autonomous readiness
remain false.

## 7. Test plan

- py_compile for the no-WING tool and new builder.
- Focused tests for the builder and valid/missing/unsafe artifacts.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint, diff check, and full backend pytest.

## 8. Where the other agent may disagree

Kimi may argue that another offline artifact gate delays real hardware work.
The reason to keep it is that fixture closure is a safety boundary: real
physical artifacts should not be allowed to change readiness unless the closure
handoff is itself explicit, file-backed, and fail-closed.
