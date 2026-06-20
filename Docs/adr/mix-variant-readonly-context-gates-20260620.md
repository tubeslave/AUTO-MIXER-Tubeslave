# ADR: Mix Variant Read-Only Context Gates

## Context

The project goal requires automatically forming mix variants for different
genres and tasks. A mix-variant profile can fan out into bus balance, FX,
master bus and scene/preset proposal families, so stale or live-authority-
bearing context can multiply an upstream mistake across many downstream
surfaces.

M001 still waits for real WING read-only evidence. This ADR does not grant live
authority or replace hardware/operator validation.

## Options Considered

1. Keep mix-variant proposals usable when current genre/task context is missing.
2. Require fresh read-only/no-authority mix-variant context, trusted source
   identity, reference metrics and arrangement context before proposal emission.
3. Disable mix-variant proposals entirely until WING hardware proof exists.

## Decision

Use option 2. Mix-variant proposal candidates require fresh read-only/no-
authority current mix-variant context, trusted source identity, fresh reference
metrics, arrangement context and follow-up proposal fan-out before they can
become operator-facing proposals.

## Why This Won

It advances the genre/task variant part of the AI Mixing Engineer goal while
preserving the current live boundary. It also handles the real fan-out risk:
wrong genre, stale reference metrics or missing arrangement context can create
plausible but unsafe downstream balance, FX, master-bus or preset suggestions.

## Rejected Alternatives

- Allowing missing context was rejected because mix variants need current
  readback, source identity, reference metrics, arrangement context, headroom
  evidence and operator intent before downstream fan-out.
- Disabling the layer entirely was rejected because offline proposal gating and
  tests can improve future assisted workflows without touching hardware.

## Implementation Plan

1. Record the mix-variant profile freshness gates as M092.
2. Require fresh current mix-variant context for proposal emission.
3. Require no-authority mix-variant context.
4. Keep missing, stale, live-authority, source-untrusted, reference-missing or
   arrangement-missing candidates analysis-only.
5. Record M092 rail coverage without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_mix_variant_profile_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_project_goal_capability_matrix.py::test_genre_variants_are_tracked_as_proposal_layer_without_live_widening -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m092_mix_variant_readonly_context_gates_stay_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M092 commit, which
restores the previous mix-variant rail documentation and M092 state/report
rows.
