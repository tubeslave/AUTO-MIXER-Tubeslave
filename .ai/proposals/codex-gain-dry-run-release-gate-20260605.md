# Codex Proposal: gain-dry-run-release-gate-20260605

## 1. Short thesis

Fold the existing fake-mixer gain proof runner into the no-WING release check
and validate that its report row stays dry-run-only.

## 2. What I understood

The project goal wants broader autonomous console control, but current safety
policy allows only a narrow supervised fader/gain executor. `gain_supervised`
is close to that boundary but still lacks enough evidence. We can make progress
by strengthening offline proof capture without claiming live readiness.

## 3. Proposed solution

- Add a `gain_dry_run_proof` no-WING check that runs
  `tools/gain_proof_dry_run_runner.py` under the release run directory.
- Pass both `offline_validation` and `gain_proof_dry_run` artifact roots to
  `tools/evaluate_supervised_proof_artifacts.py`.
- Add a report schema validation that requires at least one gain
  `dry_run_ready` row with fake-mixer pre-read/apply/rollback evidence,
  operator approval, idempotency evidence, and no promotion/autonomous claim.
- Add a non-executing next-WING-proof-plan schema summary so reviewers can see
  the operator-required physical proof step after dry-run evidence is captured.
- Publish the result in `summary.json` and `summary.md`.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/OFFLINE_NO_WING_VALIDATION_RUNBOOK.md`
- `.ai/memory/project.md`

## 5. Alternatives considered

- Update `safety_coverage_matrix.py` to mark gain proof fields passed. Rejected:
  no-WING fake-mixer proof is not physical WING proof and should not promote the
  safety row.
- Run `wing_operator_proposal_validation_runner.py`. Rejected: this is a no-WING
  path and must not require hardware.

## 6. Risks

- A fake-mixer artifact could be overread as live readiness. The schema check
  explicitly rejects promotion/autonomous claims.
- A physical proof plan could be mistaken for execution authority. The summary
  exposes only plan metadata and requires `execution_allowed_by_plan=false`.
- Release checks could become slower. The gain dry-run runner is local and
  currently completes in seconds.

## 7. Test plan

- Focused pytest for no-WING release check, gain proof runner, supervised proof
  evaluator, and supervised proof report.
- Real quick no-WING run into `/tmp` to prove the new check executes and the
  generated report contains the gain dry-run row.
- `git diff --check`.

## 8. Where the other agent may disagree

Kimi may argue this should change the safety matrix. I would keep the matrix
unchanged until physical pre-read/apply/rollback evidence exists; the new
evidence belongs in the report bundle, not the live authority contract.
