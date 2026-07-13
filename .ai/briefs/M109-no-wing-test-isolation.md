# M109 — No-WING contract-test isolation

## Evidence update — 2026-07-13

After the first three tests were repaired, the complete module reached a
fourth positive aggregate test,
`test_no_wing_release_check_runs_gap_rollup_schema_after_report`, which also
expected an all-passed synthetic release run while implicitly discovering the
same canonical 13/6/7 physical evidence. The acceptance scope is therefore
four explicit positive tests. Negative tests, production discovery and
canonical artifacts remain unpatched.

## Problem

The standard test command on HEAD `abf6b5d5e494bfa5343fab40424d456ea4339ec9`
is red because three static runner-contract tests and one positive aggregate
orchestration test call the no-WING validator against the repository's real
canonical WING artifact root. Seven intentionally preserved failed preflight
attempts make the aggregate validator fail closed even though the synthetic
contracts under test are valid.

## Evidence

- Standard suite: `1 failed, 2273 passed, 1 skipped` before `-x` stopped it.
- The first three affected tests fail together; the fourth appears when the
  complete module proceeds beyond them.
- The only runtime violation is
  `wing_readonly_preflight_invalid_artifacts_present`.
- `artifact_count=13`, `valid_candidate_count=6`,
  `invalid_candidate_count=7`.
- The production validator is behaving as designed and must remain fail closed.

## Required outcome

Make the three static contract tests and one positive aggregate orchestration
test artifact-history-hermetic by injecting an empty evidence snapshot while
leaving production discovery, failed hardware artifacts, and safety semantics
unchanged.

## In scope

- `tests/test_no_wing_release_check.py`
- A small shared fixture/helper if it removes duplication without changing
  production behavior.
- Targeted tests, the complete `tests/test_no_wing_release_check.py`, and the
  standard full suite.

## Out of scope

- Deleting or relocating real WING artifacts.
- Weakening `invalid_candidate_count` handling.
- WING/backend contact or any console write.
- DSP reset, live-authority expansion, and decision-queue transport changes.

## Acceptance criteria

1. All four positive synthetic/contract tests pass regardless of repository
   artifact history.
2. Tests that intentionally inject invalid evidence still prove fail-closed
   behavior.
3. No production file changes.
4. Standard full suite passes.

## Dirty-state note

The shared checkout contains three unrelated, byte-identical generated files
produced during validation:

- `reports/master_mix_eval_20260713T092520.json`;
- `reports/master_mix_eval_20260713T100257.json`;
- `reports/master_mix_eval_20260713T100607.json`.

They are preserved pending provenance review and are excluded from M109.
