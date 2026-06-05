# Raw Mixer State Physical Readback Artifact Intake

## Task

Add a read-only intake snapshot for raw mixer-state physical readback artifacts
captured outside Automixer by an operator.

## Context

The raw mixer-state physical readback contract defines the expected artifact
shape, and the capture plan names the next target. The system still needs a
safe way to surface whether an operator-captured artifact exists, whether it
validates against the contract, and whether manual review is still required.

## Constraints

- The intake must not query a mixer or capture hardware itself.
- The intake must not execute tests or mutate console state.
- A valid artifact is only a physical fixture candidate, not fixture closure.
- `fixture_closure_allowed_by_intake`, `live_write`, `live_authority_granted`,
  and `full_autonomous_ready` must remain false.
- Invalid candidates should fail the no-WING intake schema check.

## Expected Evidence

- Backend intake builder with file and inline-payload validation.
- Processing registry and project-goal capability visibility.
- no-WING summary and Product Layer normalization.
- Frontend Module Status row that separates contract presence from artifact
  intake status.
- Focused tests covering missing, valid, invalid, and unsafe source claims.
