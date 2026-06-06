# Codex Proposal: WING Operator Proposal Timeout Fail-Closed

## Short thesis

Make the physical WING proposal validation runner explicit about timeout
authority: fail closed before live apply, require manual review after an apply
request.

## What I understood

The current project priority is supervised WING Rack MVP proof. The next useful
slice must reduce a named blocker, especially connection-loss / timeout
fail-closed proof, without widening live write authority.

## Proposed solution

- Track sent request types in the runner artifact recorder.
- On `TimeoutError`, `OSError`, or `ConnectionError`, write structured summary
  evidence.
- If no `apply_operator_proposal` request was sent, mark the artifact blocked
  with `fail_closed=true`.
- If an apply request was sent, mark the artifact `manual_review_required` and
  explicitly reject it as promotion evidence until the operator verifies the
  console state and rollback.
- Teach the proof evaluator/report path that timeout/manual-review summaries are
  non-promotable even when readback-shaped fields exist.
- Change the read-only proof-plan command prefix from `python` to `python3`.

## Likely files to touch

- `tools/wing_operator_proposal_validation_runner.py`
- `backend/supervised_wing_proof_plan.py`
- `tests/test_wing_operator_proposal_validation_runner.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_supervised_proof_artifact_evaluator.py`
- `tests/test_supervised_proof_artifact_report.py`

## Alternatives considered

- Add a new runner: rejected because a runner already exists.
- Treat every timeout as fail-closed: rejected because timeout after an apply
  request leaves the physical console state unknown.

## Risks

- Artifacts may become more conservative and block promotion evidence more
  often. That is acceptable for live-sound safety.

## Test plan

- Focused pytest for the WING proposal runner and proof plan.
- Focused pytest for the proof evaluator/report bridge.
- `python3 -m py_compile` for touched Python files.
- `python3 tools/memory/memory_lint.py`.

## Where the other agent may disagree

Kimi may prefer a larger end-to-end hardware runbook update. This slice keeps
the diff small and directly improves the artifact semantics used by that run.
