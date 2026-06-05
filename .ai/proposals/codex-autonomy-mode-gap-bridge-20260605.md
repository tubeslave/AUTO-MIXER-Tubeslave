# Proposal: autonomy-mode-gap-bridge-20260605

## Short thesis

Bridge the new Product Layer promotion gap summary into the mode transition
gate so the four-mode architecture can explain why Full Autonomous remains
blocked in concrete evidence categories.

## What I understood

The project needs Analysis, Assisted, Supervised AutoMix, and Full Autonomous
modes. Current code correctly keeps Full Autonomous blocked, but the mode gate
should point to the same safety/read/write/proof gaps that readiness and
Settings now expose.

## Proposed solution

Extend `AutonomyModeGateRow` with:

- `readiness_gap_bucket_counts`
- `readiness_gap_target_ids_by_bucket`

Use `summary.promotion_gap_summary` from readiness. For Auto mode, append
`readiness_gap_<bucket>` blockers for every nonzero bucket. For other modes,
report the counts for explainability without changing their availability.

## Likely files to touch

- `backend/autonomy_mode_transition_gate.py`
- `tests/test_autonomy_mode_transition_gate.py`
- `.ai/briefs/autonomy-mode-gap-bridge-20260605.md`
- `Docs/adr/autonomy-mode-gap-bridge-20260605.md`

## Alternatives considered

Only show the gap rows in frontend Settings. That helps operators, but backend
mode decisions remain less explainable to agents and reports.

Use a separate remediation module. That may be useful later, but this bridge is
a direct read-only derivation of already available readiness summary data.

## Risks

The main risk is making Supervised appear blocked by broad autonomous gaps. The
implementation should carry the counts but only use them as hard blockers for
Full Autonomous.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_mode_transition_gate.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_mode_transition_gate.py tests/test_autonomy_promotion_readiness.py tests/test_processing_registry.py -q`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may argue that mode gate rows should not grow more fields. I would keep
the fields because they are explicit evidence, not behavior, and the mode gate
is the canonical place to answer "why is Auto blocked?"
