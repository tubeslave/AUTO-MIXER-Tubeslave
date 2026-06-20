# ADR: Pan Spatial Read-Only Context Gates

## Context

The project goal requires automatic pan decisions and digital-console panorama
control. Pan changes are high risk because they can break mono compatibility,
move center anchors out of the mix, reduce intelligibility, or hide phase
problems behind a stereo image.

M001 still waits for real WING read-only evidence. This ADR does not grant live
authority or replace hardware/operator validation.

## Options Considered

1. Keep pan proposals usable when current pan or mono/phase context is missing.
2. Require fresh read-only/no-authority pan context and, for non-center pan,
   fresh read-only/no-authority mono/phase safety context before proposal
   emission.
3. Disable pan proposals entirely until WING hardware proof exists.

## Decision

Use option 2. Pan proposal candidates require fresh read-only/no-authority
current pan context. Non-center pan candidates also require fresh
read-only/no-authority mono/phase safety context before they can become
operator-facing proposals.

## Why This Won

It advances the pan/stereo placement part of the AI Mixing Engineer goal while
preserving the current live boundary. It also handles real failure modes: stale
pan context can trust the wrong old value, and missing or mutable mono/phase
evidence can produce a wide stereo suggestion that collapses badly in mono.

## Rejected Alternatives

- Allowing missing context was rejected because pan decisions need current
  readback, source identity, stereo-pair or stage-position evidence, mono/phase
  safety, rollback evidence and operator review.
- Disabling the layer entirely was rejected because offline proposal gating and
  tests can improve the future assisted workflow without touching hardware.

## Implementation Plan

1. Record the existing pan proposal freshness gates as M090.
2. Require fresh current pan context for proposal emission.
3. Require no-authority pan context and no-authority mono/phase context.
4. Block non-center pan when mono/phase evidence is missing, stale, unsafe or
   live-authority-bearing.
5. Record M090 rail coverage without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_pan_spatial_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m090_pan_spatial_readonly_context_gates_stay_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M090 commit, which
restores the previous pan/stereo rail documentation and M090 state/report rows.
