# Automixer WING MVP no-WING Manifest Cross-Surface Subcheck Render

Date: 2026-06-15

## Summary

The no-WING primary summary surfaces now expose individual manifest
cross-surface subcheck verdicts for contract id, contract digest, and
digest-match flags. The operator-readable `summary.md` renders those values
after cross-surface validation computes them, so a stale manifest contract can
be diagnosed without relying only on the aggregate mismatch list.

## Safety

This is an offline no-WING reporting change. It does not contact WING, query a
console, mutate mixer state, close readiness, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_when_manifest_summary_surface_missing_line tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_markdown_renders_manifest_missing_line_self_check tests/test_no_wing_release_check.py::test_no_wing_artifact_manifest_contract_cross_surface_check_detects_mismatch tests/test_no_wing_release_check.py::test_no_wing_release_summary_fails_when_manifest_cross_surface_digest_is_stale
```

Result:

```text
5 passed in 1.20s
```

Focused WING/no-WING/Director validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
```

Result:

```text
317 passed, 8 skipped in 3.45s
```

no-WING quick gate:

```bash
PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend
```

Result: passed. The generated `summary.json` and operator-readable
`summary.md` both expose the individual cross-surface subcheck verdict fields,
with no missing surface fields or missing markdown lines. The gate still reports
`artifact_manifest_readiness_closure_allowed: false` and
`artifact_manifest_live_authority_granted: false`.

Project Director:

```bash
PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print
```

Result: `guard_status: on_target`; selected blocker remains
`WING read-only connection proof`; selected blocker status remains
`tooling_ready_waits_real_wing`; selected evidence file count is `175`.

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
3392 passed, 9 skipped in 50.76s
```

## Readiness Impact

No blocker was closed. The first MVP blocker remains `WING read-only connection
proof`, which still requires a real WING read-only preflight capture and manual
operator review.
