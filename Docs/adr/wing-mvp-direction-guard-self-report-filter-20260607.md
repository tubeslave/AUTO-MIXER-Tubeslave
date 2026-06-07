# ADR: WING MVP Direction Guard Self-Report Filter - 2026-06-07

## 1. Context

The supervised WING Rack MVP uses `tools/codex_autoloop/codex_direction_guard.py`
as a stop-the-line check. Recent runs reported `mixed_needs_review` even with no
standalone drift items because generated reports contained forbidden policy
language and stale generated evidence tables.

The first open MVP blocker remains real WING read-only connection proof. This
ADR changes only the direction guard's evidence classification.

## 2. Options considered

1. Keep current behavior and accept manual interpretation.
2. Lower drift penalties.
3. Exclude all generated reports from evidence.
4. Ignore policy-only forbidden sections and generated guard evidence tables
   only during drift scoring.

## 3. Decision

Use option 4.

## 4. Why this won

It removes the observed false-positive loop without weakening real drift
detection. Target scoring still sees the full report text, while drift scoring
ignores sections that are policy guidance or generated diagnostics rather than
work evidence.

## 5. Rejected alternatives

Manual interpretation keeps the autoloop stop signal noisy. Lowering penalties
weakens real drift detection. Excluding generated reports entirely loses useful
recent WING-proof context.

## 6. Implementation plan

- Add Markdown heading helpers.
- Strip `Forbidden Next Actions` and `Forbidden Scope` for drift scoring.
- Strip generated direction-guard `Target Evidence`, `Drift Evidence`, and
  `Mixed Evidence` sections for drift scoring.
- Keep full-text target matching unchanged.
- Add regression tests for ignored policy text, ignored generated guard evidence
  tables, and real drift outside ignored sections.

## 7. Test plan

- `python3 -m py_compile tools/codex_autoloop/codex_direction_guard.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## 8. Risks and rollback

The main risk is hiding real drift if someone writes action evidence inside a
forbidden policy section. Rollback is to revert the filter and restore raw text
drift matching.
