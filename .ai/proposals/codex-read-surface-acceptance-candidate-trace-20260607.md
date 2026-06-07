# Proposal: Read-Surface Acceptance Candidate Trace

## 1. Short thesis

Acceptance records should carry a read-only `candidate_trace` so an operator's
manual decision can be checked against the exact review-queue candidate it came
from.

## 2. What I understood

The physical-readback flow is still an offline/no-WING operator gate. The review
queue can now preserve artifact source ids, but the acceptance template only
copies the selected artifact id. That is not enough to detect an edited record
whose candidate status, source list, or safety flags no longer match the queue.

## 3. Proposed solution

- Add `candidate_trace` to generated acceptance-template records.
- Include queue method/status, candidate status, intake status, artifact source
  ids, candidate counts, target consistency, missing-source status, and explicit
  false authority flags.
- Extend the fixture acceptance ledger to compare each record trace against the
  current queue candidate and count missing or mismatched traces.
- Extend no-WING schema validation and summary output for the trace counters.
- Cover valid, missing-trace, and mismatched-trace paths in tests.

## 4. Likely files to touch

- `tools/build_read_surface_physical_readback_acceptance_template.py`
- `backend/read_surface_physical_readback_fixture_acceptance_ledger.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_fixture_acceptance_template.py`
- `tests/test_read_surface_physical_readback_fixture_acceptance_ledger.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/read-surface-acceptance-candidate-trace-20260607.md`
- `Docs/adr/read-surface-acceptance-candidate-trace-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_ACCEPTANCE_CANDIDATE_TRACE_20260607.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Trust `artifact_source_id` alone.
- Recompute all candidate context from fixtures during ledger validation.
- Let mismatched traces pass and rely on later fixture closure gates.

## 6. Risks

The main risk is making the operator template look more authoritative than it
is. Every generated trace field must keep fixture closure, live authority, and
full autonomous readiness false, and no-WING must continue to fail closed.

## 7. Test plan

- py_compile for the acceptance-template builder, acceptance ledger, and
  no-WING runner.
- Focused acceptance-template, acceptance-ledger, and no-WING tests.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint, diff check, and full backend pytest.

## 8. Where the other agent may disagree

Kimi may prefer keeping acceptance records minimal and validating only the
artifact id. The trace is worth the small extra JSON surface because it catches
stale or hand-edited records before any later closure decision sees them.
