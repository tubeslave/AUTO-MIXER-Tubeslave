# ADR: gain-dry-run-release-gate-20260605

## 1. Context

`gain_trim_supervised_promotion` is one of the closest targets to the existing
supervised fader/gain executor. The repository already has a fake-mixer gain
proof runner, but no-WING release checks did not require that runner or verify
that the supervised proof report consumed its artifact.

## 2. Options considered

- Keep the gain proof runner manual-only.
- Add the runner to no-WING release checks as dry-run evidence.
- Promote `gain_supervised` safety fields based on fake-mixer evidence.

## 3. Decision

Add a `gain_dry_run_proof` check to the no-WING release bundle and validate the
resulting supervised proof report for a read-only gain dry-run row. Also expose
the next physical WING proof plan as read-only, operator-required, and
non-executing summary metadata.

## 4. Why this won

It creates repeatable evidence for the Product Layer gain path while preserving
the live boundary. It proves proposal creation, apply readback, rollback,
cooldown blocking, and emergency-stop blocking locally, then keeps the proof
classified as `dry_run_ready` rather than `supervised_candidate`.

## 5. Rejected alternatives

- Manual-only runner: too easy to omit from release evidence.
- Safety matrix promotion: too strong without physical WING proof.

## 6. Implementation plan

- Add a gain proof artifact root under each no-WING run directory.
- Run `tools/gain_proof_dry_run_runner.py` before
  `tools/evaluate_supervised_proof_artifacts.py`.
- Pass both offline validation and gain proof roots to the evaluator.
- Add a schema check for fake-mixer gain dry-run evidence with no live,
  promotion, or autonomous claim.
- Add a schema check for the next WING proof plan that preserves
  `execution_allowed_by_plan=false` and `operator_required=true`.
- Expose the result in release summary JSON and Markdown.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py tests/test_gain_proof_dry_run_runner.py tests/test_supervised_proof_artifact_evaluator.py tests/test_supervised_proof_artifact_report.py -q`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-gain-dry-run --command-timeout-sec 300`
- `git diff --check`

## 8. Risks and rollback

Risk: fake-mixer evidence is mistaken for physical readiness. Mitigation: the
schema check requires `promotion_allowed=false`, `autonomous_ready=false`, and
the physical-proof blocker. Rollback is to remove `gain_dry_run_proof` from
`build_checks()` and remove the schema summary section.
