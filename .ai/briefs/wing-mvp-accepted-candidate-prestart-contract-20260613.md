# Brief: WING MVP accepted-candidate prestart trace contract

Task id: `wing-mvp-accepted-candidate-prestart-contract-20260613`

## Context

The supervised WING Rack MVP is still blocked on real WING read-only
connection proof. Before any supervised fader/gain proof, an accepted
read-only preflight candidate must carry a valid no-WING prestart trace bound
by path and SHA256.

## Problem

The accepted-candidate prestart trace check was implemented separately in the
review decision ledger and the supervised proof plan. A drift between those two
checks could let one surface report a candidate as ready while another keeps
the supervised proof blocked.

## Goal

Move the accepted-candidate prestart trace gate into the shared
`backend/wing_readonly_preflight_contract.py` contract and make both
downstream surfaces use the same helper.

## Non-goals

- No WING hardware contact.
- No live-write authority.
- No readiness closure.
- No change to supervised proof execution.

## Validation

- Focused unit tests for the shared helper, review decision ledger, supervised
  proof plan, runner, and evidence intake.
- no-WING release check.
- Direction guard and project director checks.
