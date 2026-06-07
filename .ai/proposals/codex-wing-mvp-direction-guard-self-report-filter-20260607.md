# Proposal: WING MVP Direction Guard Self-Report Filter - 2026-06-07

## 1. Short thesis

Keep the direction guard strict, but stop counting its own safety warnings and
generated evidence tables as new drift evidence.

## 2. What I understood

The project is still on the supervised WING Rack MVP. The immediate blocker is
real WING read-only preflight proof. The direction guard must stop broad work
when the repo drifts toward memory/queue/result-consumer/report-only layers, but
it must not self-trigger on sections that explicitly forbid those paths.

## 3. Proposed solution

Add a small Markdown-section filter used only for drift scoring:

- always ignore `## Forbidden Next Actions` and `## Forbidden Scope`;
- for generated `codex_direction_guard_*.md` reports, also ignore
  `## Target Evidence`, `## Drift Evidence`, and `## Mixed Evidence`;
- keep target scoring on the full text;
- keep drift scoring active for real drift outside ignored sections.

## 4. Likely files to touch

- `tools/codex_autoloop/codex_direction_guard.py`
- `tests/test_codex_direction_guard.py`
- `.ai/briefs/wing-mvp-direction-guard-self-report-filter-20260607.md`
- `Docs/adr/wing-mvp-direction-guard-self-report-filter-20260607.md`
- `Docs/reports/AUTOMIXER_WING_MVP_DIRECTION_GUARD_SELF_REPORT_FILTER_20260607.md`
- `Docs/INDEX.md`

## 5. Alternatives considered

- Lower the drift score penalty. Rejected because it would make real drift less
  visible.
- Remove generated reports from evidence. Rejected because recent reports still
  carry useful WING-proof context.
- Ignore all `Drift Evidence` sections in every report. Rejected because
  hand-written reports may include real evidence under that title.

## 6. Risks

- Over-filtering could hide real drift if it is placed inside a forbidden
  policy section. This is acceptable because those sections are normative
  guidance, not action evidence.
- Generated guard reports could still inflate target scoring. This change is
  scoped to false drift; target weighting can be tuned separately if needed.

## 7. Test plan

- Compile the guard.
- Run focused direction guard tests.
- Run the guard against the current repo and confirm `drift_item_count=0` and
  `mixed_item_count=0`.
- Run memory lint, diff check, and the full pytest suite before commit.

## 8. Where the other agent may disagree

Kimi may argue to remove generated guard reports entirely from evidence. I would
keep them for now because they provide useful target context and the current bug
is specifically recursive drift classification.
