# Brief: WING MVP Decision Review Prestart Ready Gate

## Context

The active MVP blocker remains `WING read-only connection proof`. The supervised
proof path already requires accepted read-only preflight source ids to have a
valid accepted-candidate backend prestart trace before any fader/gain proof
step can be planned.

The standalone decision-review tool still risked presenting an accepted
operator decision as ready before that accepted-candidate prestart trace was
valid.

## Task

Make the read-only preflight review decision ledger and decision-review CLI
fail closed unless accepted source ids also have a valid accepted-candidate
prestart trace.

## Constraints

- Do not contact WING hardware.
- Do not stop, start, or reconfigure local backend listeners.
- Do not grant live authority or mutate readiness files.
- Preserve the first MVP blocker boundary.

## Validation

- Focused review decision, no-WING release, and supervised proof-plan tests.
- Project Director and direction guard remain on the WING read-only blocker.
- `git diff --check`, memory lint, and full backend pytest before commit.
