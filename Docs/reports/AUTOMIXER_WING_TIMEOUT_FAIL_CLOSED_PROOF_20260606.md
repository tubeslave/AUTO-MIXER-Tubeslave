# Automixer WING Timeout Fail-Closed Proof - 2026-06-06

## Summary

This slice reduces the supervised WING Rack MVP blocker for
connection-loss / timeout fail-closed proof. It does not prove a new physical
WING run and does not widen live write authority.

The WING operator proposal validation runner now records whether a transport
failure happened before or after a live apply request:

- before `apply_operator_proposal`: artifact is blocked with
  `fail_closed=true`;
- after `apply_operator_proposal`: artifact is `manual_review_required` and
  must not be used as promotion evidence until the operator verifies physical
  console state and rollback.

## Why This Matters

A generic timeout error is not enough evidence for a supervised hardware proof.
For live sound, the runner must distinguish two very different states:

- no live apply request left the runner, so the proof failed closed;
- a live apply request may have reached the backend/console, so the system must
  require manual inspection instead of claiming safety.

This makes future WING proof artifacts more honest and prevents an inconclusive
network failure from being promoted as readiness evidence.

## Files Changed

- `tools/wing_operator_proposal_validation_runner.py`
- `backend/supervised_wing_proof_plan.py`
- `tests/test_wing_operator_proposal_validation_runner.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_supervised_proof_artifact_evaluator.py`
- `tests/test_server.py`
- `.ai/memory/project.md`
- `.ai/briefs/wing-operator-proposal-timeout-fail-closed-20260606.md`
- `.ai/proposals/codex-wing-operator-proposal-timeout-fail-closed-20260606.md`
- `Docs/adr/wing-operator-proposal-timeout-fail-closed-20260606.md`

## Safety Boundary

- No live WING/OSC action was run for this change.
- Tests use fake WebSocket or dummy server paths only.
- Physical mutation remains behind `--execute`.
- Physical mutation still goes only through `apply_operator_proposal`.
- Fader/gain remain the only supervised live write kinds.
- EQ, dynamics, gate, routing, mute, FX, scenes, matrices, DCA, and autonomous
  live mode remain blocked.

## Proof Semantics

| Failure point | Artifact status | Promotion evidence allowed |
| --- | --- | --- |
| WebSocket connection fails before any request | `blocked` | no |
| Timeout before `apply_operator_proposal` | `blocked`, `fail_closed=true` | no |
| Timeout after `apply_operator_proposal` | `manual_review_required` | no |
| Rollback pre-read timeout | `manual_write_rollback_blocked` | no |

The supervised proof artifact evaluator also treats timeout summaries as
non-promotable even if a future artifact includes otherwise complete readback
fields. `manual_review_required`, `blocked`, `failed`, and `error` runner
summaries cannot become `supervised_candidate` rows.

The report tool and promotion-evidence bridge preserve that decision:
timeout artifacts remain in `blocked_proof_ids`, and they do not create
`safety_row_review_candidate_ids`.

## Validation

```bash
python3 -m py_compile tools/wing_operator_proposal_validation_runner.py backend/supervised_wing_proof_plan.py
```

Result: PASS

```bash
python3 -m py_compile backend/supervised_proof_artifact_evaluator.py tools/evaluate_supervised_proof_artifacts.py tools/wing_operator_proposal_validation_runner.py backend/supervised_wing_proof_plan.py
```

Result: PASS

```bash
PYTHONPATH=backend python3 -m pytest -q \
  tests/test_wing_operator_proposal_validation_runner.py \
  tests/test_supervised_wing_proof_plan.py \
  tests/test_server.py::TestWingDeploymentWriteBoundary::test_rollback_last_manual_write_blocks_when_pre_read_times_out
```

Result: `18 passed`

```bash
PYTHONPATH=backend python3 -m pytest -q \
  tests/test_supervised_proof_artifact_evaluator.py \
  tests/test_supervised_proof_artifact_report.py \
  tests/test_wing_operator_proposal_validation_runner.py \
  tests/test_supervised_wing_proof_plan.py \
  tests/test_server.py::TestWingDeploymentWriteBoundary::test_rollback_last_manual_write_blocks_when_pre_read_times_out
```

Result: `38 passed`

```bash
python3 tools/memory/memory_lint.py
```

Result: PASS

```bash
/opt/homebrew/bin/git diff --check
```

Result: PASS

```bash
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
```

Result: `2837 passed, 1 skipped, 1 warning`

## Remaining WING MVP Blockers

This closes no hardware proof by itself. The next physical cycle still needs:

1. WING read-only connection proof.
2. Physical console readback proof.
3. Supervised fader write proof refresh with explicit operator approval.
4. Supervised gain-down / trim proof with explicit operator approval.
5. Rollback proof from physical readback.
6. Emergency stop proof from physical run.
7. Real connection-loss / timeout artifact from a controlled physical run.
8. Readiness matrix update based on real proof.

## Operator Guidance

For a physical proof run, use the existing proposal-based runner only when the
operator is present, the backend is connected to the intended WING Rack, and the
exact bounded action is approved. If a timeout happens after
`apply_operator_proposal`, inspect the console and rollback manually before
using any artifact for readiness decisions.
