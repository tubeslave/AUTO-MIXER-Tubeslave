# Automixer WING MVP no-WING Artifact Manifest Self-Consistency

Date: 2026-06-15

## Summary

The no-WING `artifact-manifest.json` now includes a self-consistency verdict for
its declared artifact IDs, paths, required SHA fields, top-level SHA values, and
readiness/live-authority denial flags. The detached manifest also now writes the
required artifact SHA field list it previously only implied through summary
surfaces.

## Safety

This is an offline no-WING artifact validation change. It does not contact WING,
query a console, mutate mixer state, close readiness, or grant live authority.

## Validation

Targeted validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan tests/test_no_wing_release_check.py::test_no_wing_artifact_manifest_contract_cross_surface_check_detects_mismatch tests/test_no_wing_release_check.py::test_no_wing_artifact_manifest_cross_surface_check_denies_live_authority_mismatch tests/test_no_wing_release_check.py::test_no_wing_artifact_manifest_self_consistency_detects_stale_artifact_rows tests/test_no_wing_release_check.py::test_no_wing_release_summary_fails_when_manifest_cross_surface_digest_is_stale
```

Result:

```text
5 passed in 1.22s
```

Focused WING/no-WING guard validation:

```bash
PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py tests/test_project_director_agent.py tests/test_wing_mvp_rails.py
```

Result:

```text
319 passed, 8 skipped in 3.54s
```

no-WING quick validation:

```bash
PYTHONPATH=backend python tools/no_wing_release_check.py --quick --skip-frontend
```

Latest generated artifact root:
`artifacts/no_wing_release_check/20260614T232906Z_no_wing_release_check`

Observed results:

```text
summary.json status: passed
artifact-manifest.json artifact_manifest_self_consistency_match: true
artifact-manifest.json artifact_manifest_self_consistency_mismatches: []
artifact-manifest.json artifact_manifest_self_consistency_readiness_closure_allowed: false
artifact-manifest.json artifact_manifest_self_consistency_live_authority_granted: false
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
selected_evidence_file_count: 177
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
3394 passed, 9 skipped in 50.66s
```

## Readiness Impact

No blocker was closed. The first MVP blocker remains `WING read-only connection
proof`, which still requires a real WING read-only preflight capture and manual
operator review.
