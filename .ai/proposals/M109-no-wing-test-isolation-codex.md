# Codex proposal — M109 no-WING contract-test isolation

## 1. Short thesis

Patch only the three static contract tests so they validate runner constants
against an explicitly empty evidence inventory; preserve the production
fail-closed artifact aggregate unchanged.

## 2. What I understood

The regression was introduced by adding truthful failed real-WING attempts to
the canonical artifact root. The release validator correctly rejects an
aggregate containing invalid candidates. The affected tests are not testing
artifact acceptance; they are testing runner/evidence constant agreement, but
they accidentally inherit mutable repository evidence.

## 3. Proposed solution

Add a small test-local helper that monkeypatches
`release_check.build_wing_readonly_preflight_evidence` to return
`build_wing_readonly_preflight_evidence(artifact_payloads=(), artifact_roots=())`.
Use it in exactly these tests:

- `test_no_wing_release_check_tracks_wing_preflight_proof_runner_contract`
- `test_no_wing_release_check_tracks_wing_preflight_run_summary_contract`
- `test_no_wing_release_check_tracks_wing_preflight_evidence_allowed_value_types`

Do not change the validator, the evidence builder, or real artifacts.

## 4. Likely files to touch

- `tests/test_no_wing_release_check.py`
- `.ai/*` and `Docs/adr/*` process records only

## 5. Alternatives considered

- Delete failed artifacts: rejected because they are truthful hardware
  evidence.
- Ignore invalid candidates in production: rejected because it weakens the
  release rail.
- Move all failed attempts to another root: deferred; it changes evidence
  provenance and does not solve test hermeticity.
- Patch every contract test independently: valid but unnecessarily repetitive.

## 6. Risks

- A helper could accidentally hide artifact-dependent assertions. Limit its use
  to the three constant-contract tests and retain existing explicit invalid-
  evidence tests unchanged.
- Full-suite runs may generate unrelated reports. Preserve and report them
  separately.

## 7. Test plan

1. Run the three affected tests together.
2. Run `tests/test_no_wing_release_check.py` in full.
3. Run `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`.
4. Inspect `git diff --check` and the final diff.

## 8. Where the other agent may disagree

An alternative reviewer may prefer changing the production aggregate to select
only the latest accepted candidate. That would solve the immediate failure but
would silently alter a deliberate fail-closed rule and is too broad for this
regression.
