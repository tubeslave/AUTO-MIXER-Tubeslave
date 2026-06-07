# Brief: WING Supervised Proof Physical Readback Only - 2026-06-07

## Context

The supervised WING proof runner is the narrow physical write path for the WING
Rack MVP. It must prove physical console behavior before any readiness update.

## Problem

The runner accepted `verified_from_cache` as sufficient proof for a physical
`--execute` apply/rollback path. That weakens the MVP evidence chain because a
cache-confirmed result does not prove the console physically moved and rolled
back.

## Scope

- Require physical pre-read for `--execute`.
- Require physical apply readback.
- Require physical rollback readback.
- Keep dry-run planning allowed to use cache.
- Add a no-WING static gate that fails if cache is reintroduced as physical
  proof evidence.

## Out Of Scope

- Running WING hardware.
- Widening live writes beyond fader/gain proof.
- Updating readiness rows from offline evidence.
