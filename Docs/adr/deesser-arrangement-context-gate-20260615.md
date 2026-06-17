# ADR: De-esser Arrangement Context Gate

Date: 2026-06-15
Status: accepted

## Context

Automatic de-essing is an explicit project requirement. The existing
`deesser_vocal_sibilance` method was transport-free and proposal-only, but it
used source role and sibilance evidence without a separate arrangement-context
gate. Arrangement context can change how aggressive a de-esser should be:
lead vocals can tolerate direct sibilance control, while tucked/backing vocals
should usually be treated more conservatively.

## Options considered

1. Keep the de-esser source-role-only.
2. Add arrangement role and freshness evidence to the proposal-only de-esser.
3. Promote de-esser proposals into queueable/live dynamics changes.

## Decision

Use option 2. `DeesserAnalysisInput` now accepts `arrangement_role` and
`arrangement_context_cache_freshness`. The analyzer records arrangement evidence,
uses a less aggressive profile for tucked/backing/double/harmony vocals, and
blocks proposal-only output when provided arrangement context freshness is not
fresh.

## Why this won

It moves a real DSP capability closer to the long-term AI Mixing Engineer goal
without widening live writes. The recommendation remains transport-free,
non-queueable, and blocked from live apply until separate hardware readback and
rollback proof exists.

## Rejected alternatives

Source-role-only processing was rejected because a vocal's role in the
arrangement changes the correct processing intensity. Live promotion was
rejected because dynamics/dynamic-EQ readback and rollback proof are still
missing.

## Implementation plan

1. Add arrangement fields to the de-esser analysis input.
2. Normalize arrangement context freshness into evidence.
3. Add `deesser_arrangement_context_cache_not_fresh` for stale provided context.
4. Tune tucked/backing vocal proposals to use lower confidence, lower ratio, and
   less sensitive threshold than lead vocals.
5. Update proposal contract metadata, capability wording, and backlog docs.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_deesser_vocal_sibilance.py tests/test_project_goal_capability_matrix.py`
- Related registry/proposal contract tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`.

## Risks and rollback

Risk: some backing-vocal sibilance cases become analysis-only or less
aggressive. Roll back by removing arrangement profile scaling and the freshness
blocker. No live WING, OSC, or console mutation behavior is introduced.
