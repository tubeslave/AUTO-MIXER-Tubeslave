# ADR: Read-Surface Acceptance Candidate Trace

## Context

The read-surface physical-readback acceptance template renders pending operator
records from review-queue candidates. After source ids were preserved in the
queue, the next gap was traceability between a generated acceptance record and
the candidate it represents. A record with only `artifact_source_id` can prove
which artifact was selected, but not whether the queue candidate status,
candidate counts, source list, and safety flags still match.

## Options considered

1. Keep only `artifact_source_id` in acceptance records.
2. Let the acceptance ledger trust a manually edited record if the source id
   matches.
3. Copy a read-only candidate trace into the acceptance template and have the
   ledger verify it against the queue candidate.

## Decision

Use option 3. Generated acceptance records now include `candidate_trace`, and
the fixture acceptance ledger rejects missing or mismatched traces.

## Why this won

The acceptance template is the operator handoff. It should be self-describing
enough to audit what was accepted or rejected, while the ledger should still
verify the trace against the current queue before treating a record as valid.
This keeps the physical-readback path deterministic and fail-closed without
querying any console or widening live authority.

## Rejected alternatives

- Trusting `artifact_source_id` alone would miss stale candidate status or
  changed source-list context.
- Recomputing context only in the ledger would make the operator-facing
  acceptance record less auditable.
- Deferring trace checks to fixture closure would let invalid acceptance records
  advance too far.

## Implementation plan

1. Add `candidate_trace` generation to acceptance-template records.
2. Validate trace source ids, statuses, candidate counts, consistency flags, and
   false authority flags in the fixture acceptance ledger.
3. Add ledger summary counts for missing and mismatched traces.
4. Extend no-WING validation and summary Markdown for candidate-trace counts.
5. Update focused tests, docs, and the documentation index.

## Test plan

- `python3 -m py_compile tools/build_read_surface_physical_readback_acceptance_template.py backend/read_surface_physical_readback_fixture_acceptance_ledger.py tools/no_wing_release_check.py`
- Focused acceptance-template, acceptance-ledger, and no-WING pytest.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

This change does not query hardware, mutate console state, update fixtures,
close read-surface gaps, or grant live authority. Rollback is to remove the
trace field, ledger trace validation, no-WING summary fields, tests, and docs
row; no mixer transport path is affected.
