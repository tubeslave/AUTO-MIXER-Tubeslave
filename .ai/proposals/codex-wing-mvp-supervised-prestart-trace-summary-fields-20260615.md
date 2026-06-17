# Codex Proposal: WING MVP Supervised Prestart Trace Summary Fields

## Short Thesis

Make all supervised prestart trace prerequisites visible as top-level no-WING
summary fields so release checks do not need nested JSON parsing to verify the
gate.

## What I Understood

The current WING MVP remains on the read-only connection proof blocker. The
offline handoff rail now carries a structured supervised prestart gate, but only
the expected-artifact trace prerequisite was mirrored as a top-level summary
field.

## Proposed Solution

- Derive flat fields for accepted-candidate prestart trace and artifact-root
  trace from `supervised_prestart_gate`.
- Render the flat fields in `summary.json` and operator-readable `summary.md`.
- Add focused test assertions and Project Director baseline markers.
- Document the decision as no-authority validation.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`

## Alternatives Considered

- Leave the prerequisites nested only. Rejected because release summaries should
  be operator-readable and grep-friendly.
- Add a broad summary-surface registry refactor. Rejected because it is larger
  than needed for this gate.

## Risks

- A flat field could drift from the structured gate. Mitigation: derive from the
  structured gate and assert JSON/Markdown parity.

## Test Plan

- Compile touched Python files.
- Run focused no-WING and Project Director tests.
- Run no-WING quick validation.
- Run the full pytest suite.
- Run memory lint and diff whitespace check.

## Where The Other Agent May Disagree

Kimi may prefer a generated list of gate projections. That may be worthwhile
later, but this narrow projection keeps the current rail moving with less blast
radius.
