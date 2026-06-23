# Automixer WING MVP no-WING Manifest Cross-Surface Safety Denial

Date: 2026-06-15

## Summary

The no-WING manifest cross-surface check now verifies that both `summary.json`
and operator-readable `summary.md` keep manifest cross-surface readiness closure
and live authority denied. A surface that renders those safety fields as `true`
now fails the aggregate manifest cross-surface check with explicit mismatch
names.

## Safety

This is an offline no-WING reporting and validation change. It does not contact
WING, query a console, mutate mixer state, close readiness, or grant live
authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_when_manifest_summary_surface_missing_line tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_markdown_renders_manifest_missing_line_self_check tests/test_no_wing_release_check.py::test_no_wing_artifact_manifest_contract_cross_surface_check_detects_mismatch tests/test_no_wing_release_check.py::test_no_wing_artifact_manifest_cross_surface_check_denies_live_authority_mismatch tests/test_no_wing_release_check.py::test_no_wing_release_summary_fails_when_manifest_cross_surface_digest_is_stale
```

Result:

```text
6 passed in 1.24s
```

Focused WING/no-WING/Director validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
```

Result:

```text
318 passed, 8 skipped in 3.62s
```

no-WING quick gate:

```bash
PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend
```

Result: passed. The generated no-WING artifacts include all four new safety
denial verdicts in `summary.md` and `artifact-manifest.json`, with
`artifact_manifest_contract_cross_surface_match: true`,
`artifact_manifest_contract_cross_surface_readiness_closure_allowed: false`,
and `artifact_manifest_contract_cross_surface_live_authority_granted: false`.

Project Director:

```bash
PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print
```

Result: `guard_status: on_target`; selected blocker remains
`WING read-only connection proof`; selected blocker status remains
`tooling_ready_waits_real_wing`; selected evidence file count is `176`.

Repository hygiene:

```bash
PYTHONPATH=backend python tools/memory/memory_lint.py
/opt/homebrew/bin/git diff --check
```

Result: both passed.

Full standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3393 passed, 9 skipped in 52.78s
```

## Readiness Impact

No blocker was closed. The first MVP blocker remains `WING read-only connection
proof`, which still requires a real WING read-only preflight capture and manual
operator review.
