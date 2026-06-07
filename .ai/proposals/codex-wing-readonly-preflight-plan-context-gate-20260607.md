# Proposal: WING Read-Only Preflight Plan Context Gate

## 1. Short thesis

The offline WING preflight plan should fail schema validation unless it carries
the blocker, manual-review, fail-closed, and post-run review context required to
run the first real WING MVP step safely.

## 2. What I understood

The current first open blocker is WING read-only connection proof. No real
artifact exists in `artifacts/wing_readonly_preflight/`, so this cycle cannot
close the blocker. The useful safe improvement is to make the existing plan
artifact harder to misuse before the operator runs the real read-only preflight.

## 3. Proposed solution

- Add required fail-closed conditions to `tools/no_wing_release_check.py`.
- Validate `selected_mvp_blocker`, blocker status, `operator_required`, and
  manual-review boundary fields.
- Validate `post_run_review_commands` against
  `wing_readonly_preflight_review_command()` and
  `wing_readonly_preflight_review_decision_command()`.
- Validate handoff packet artifacts against the referenced offline preflight
  plan artifact, failing closed if the companion plan is missing or invalid.
- Expose the new fields in summary JSON and Markdown.
- Extend positive and unsafe plan-artifact tests.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-readonly-preflight-plan-context-gate-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-plan-context-gate-20260607.md`
- `Docs/adr/wing-readonly-preflight-plan-context-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PLAN_CONTEXT_GATE_20260607.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Leave the runner tests as the only enforcement.
- Duplicate the expected post-run commands as string literals.
- Skip the gate because the real WING artifact still must be reviewed later.

## 6. Risks

The main risk is overstating progress. This gate only hardens an offline plan;
it does not prove WING connectivity, physical readback, or readiness.

## 7. Test plan

- py_compile for `tools/no_wing_release_check.py`.
- Focused plan-artifact, handoff-artifact, and summary tests.
- no-WING quick run.
- memory lint, diff check, and full backend pytest.

## 8. Where the other agent may disagree

Kimi may argue no more offline tooling is needed. This is still directly tied to
the first blocker because it prevents an incomplete preflight plan from passing
the release gate before a real WING run.
