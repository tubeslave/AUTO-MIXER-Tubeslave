# Brief: WING MVP Proof Plan Artifact Root Trace

## Task

Carry accepted read-only preflight candidate `artifact_root` into the next supervised WING proof-plan summary and fail closed when the trace is missing.

## Context

The WING read-only path now validates runtime `artifact_root`, evidence digests, review-decision digests, and decision-review report rendering. The next downstream boundary is `backend/supervised_wing_proof_plan.py`: it already carries accepted source ids and accepted-candidate prestart diagnostics, but physical proof planning could still be summarized without an explicit accepted-candidate artifact root.

## Constraints

- No live mixer transport.
- No live-write authority.
- No positive gain or fader movement.
- Keep the first WING MVP blocker open until a real WING read-only candidate is captured and reviewed.
- Validate through focused proof-plan and no-WING tests.

## Done When

- The supervised proof-plan summary exposes candidate and expected accepted-candidate `artifact_root` diagnostics.
- Missing `artifact_root` on an accepted candidate blocks physical proof steps.
- The no-WING release check rejects stale proof-plan summaries that omit or misreport the root trace.
- The decision is recorded in ADR/report docs and indexed.
