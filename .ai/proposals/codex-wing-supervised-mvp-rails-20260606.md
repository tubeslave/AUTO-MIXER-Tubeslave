# Proposal: WING Supervised MVP Rails

## Short thesis

Add a small set of durable MVP rail files so Codex always knows the current
goal, first open blocker, allowed next action, and forbidden detours.

## What I understood

The project needs fewer broad abstractions and more direct movement toward a
working supervised WING Rack proof.

## Proposed solution

- Create a canonical MVP contract in `Docs/MVP_WING_SUPERVISED.md`.
- Create a live blocker table in `.ai/project-layer/wing_mvp_blockers.md`.
- Create runbooks for read-only preflight and supervised fader/gain proof.
- Create a latest progress report.
- Strengthen the direction guard and prompt to use these files.

## Likely files to touch

- `Docs/MVP_WING_SUPERVISED.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`
- `tools/codex_autoloop/*`
- `tests/*mvp*`

## Alternatives considered

- Add only memory notes. Rejected because Codex needs an operational table.
- Add another daemon. Rejected because the existing autoloop can run the guard.

## Risks

- The table can become stale if agents do not update it after real artifacts.
- Codex may still over-focus on report tooling unless the guard enforces the
  first open blocker.

## Test plan

- Focused tests for MVP docs and progress updater.
- Direction guard tests.
- Memory lint.

## Where the other agent may disagree

Another agent may want the blocker table generated from code only. Manual table
plus tests is the lower-risk starting point because hardware proof acceptance
still requires operator review.
