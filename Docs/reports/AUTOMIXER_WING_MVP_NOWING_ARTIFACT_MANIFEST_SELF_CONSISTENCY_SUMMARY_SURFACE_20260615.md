# Automixer WING MVP no-WING Artifact Manifest Self-Consistency Summary Surface

Date: 2026-06-15

## Summary

The no-WING `summary.json` and operator-readable `summary.md` now expose the
manifest self-consistency verdict that was previously only visible in
`artifact-manifest.json`. This makes the handoff entrypoint show whether the
manifest sidecar is expected to match its declared artifact IDs, paths, SHA
fields, SHA availability, and readiness/live-authority denial flags.

## Safety

This is an offline no-WING artifact-surface change. It does not contact WING,
query a console, mutate mixer state, close readiness, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_when_manifest_summary_surface_missing_line tests/test_no_wing_release_check.py::test_no_wing_release_check_fails_when_manifest_contract_digest_line_missing tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_markdown_renders_manifest_missing_line_self_check
```

Result:

```text
4 passed in 1.16s
```

no-WING quick validation:

```bash
PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend
```

Latest generated artifact root:
`artifacts/no_wing_release_check/20260614T233929Z_no_wing_release_check`

Observed results:

```text
summary.json status: passed
summary.json artifact_manifest_self_consistency_match: true
summary.md artifact_manifest_self_consistency_match line rendered: true
artifact-manifest.json artifact_manifest_self_consistency_match: true
artifact_manifest_summary_surface_missing_fields: []
artifact_manifest_summary_surface_missing_field_lines: []
```

Focused WING/no-WING guard validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
```

Result:

```text
319 passed, 8 skipped in 3.52s
```

Project Director guard:

```bash
PYTHONPATH=backend python tools/codex_autoloop/project_director_agent.py --format json --print
```

Result:

```text
guard_status: on_target
guard_score: 90
selected_blocker: WING read-only connection proof
selected_blocker_status: tooling_ready_waits_real_wing
selected_evidence_file_count: 178
```

Repository hygiene:

```bash
PYTHONPATH=backend python tools/memory/memory_lint.py
/opt/homebrew/bin/git diff --check
```

Result:

```text
memory_lint: PASS
git diff --check: PASS
```

Full standard validation:

```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

Result:

```text
3394 passed, 9 skipped in 51.02s
```

## Readiness Impact

No blocker was closed. The first MVP blocker remains `WING read-only connection
proof`, which still requires a real WING read-only preflight capture and manual
operator review.
