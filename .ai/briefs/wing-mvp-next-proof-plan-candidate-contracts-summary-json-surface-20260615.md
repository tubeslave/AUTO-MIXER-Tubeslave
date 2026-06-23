# Brief: WING MVP Next Proof Plan Candidate Contracts Summary JSON Surface

Date: 2026-06-15

## Task

Expose `next_wing_proof_plan` accepted-candidate prestart and artifact-root
contract verdicts as top-level no-WING `summary.json` fields instead of
requiring machine consumers to parse the nested proof-plan object.

## Constraints

- Keep `WING read-only connection proof` as the first open blocker.
- Do not contact WING, query a console, start a backend/WING session, or widen
  live authority.
- Preserve offline proof-plan evidence as non-authoritative for readiness
  closure.
- Keep the change limited to no-WING reporting, tests, and Council evidence.

## Expected Evidence

- Top-level `summary.json` fields for accepted-candidate prestart contract
  validity, accepted-candidate artifact-root contract validity, expected
  contract violation maps, candidate source/root lists, and invalid records.
- The projected candidate-contract surface remains offline and explicitly
  denies readiness closure and live authority.
- Focused and standard tests pass.
