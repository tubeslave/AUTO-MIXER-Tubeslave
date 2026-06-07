# Codex Proposal: WING Read-Only Preflight Session Confirmation Gate - 2026-06-07

## Short thesis

Physical read-only WING preflight should fail before WebSocket unless the
operator explicitly confirms that the intended backend/WING session is ready for
read-only inspection.

## What I understood

The next allowed real hardware action is a read-only preflight against an
operator-prepared WING Rack session. The runner must not acquire new live
authority, start a session on its own, or make physical-looking evidence easier
to accept than the operator workflow allows.

## Proposed solution

Add `--operator-confirm-session-ready` to
`tools/wing_readonly_preflight_runner.py`. Require it whenever
`--require-physical-readback` is used, include it in offline plans and captured
summaries, and add a fail-closed plan condition for missing session
confirmation.

Extend evidence intake and `tools/no_wing_release_check.py` so stale plans,
capture commands, handoff packets, and summaries without the flag or summary
fields are rejected before manual readiness review.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-session-confirmation-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_SESSION_CONFIRMATION_GATE_20260607.md`

## Alternatives considered

- Keep session readiness as prose in the runbook. Rejected because it is not
  visible in copied commands or machine-validated artifacts.
- Use an interactive prompt. Rejected because the preflight path is
  artifact-driven and should remain repeatable in runbooks and release checks.

## Risks

Older physical preflight command examples without the new flag now fail closed.
That is intentional because no accepted physical read-only proof has closed the
MVP blocker yet.

## Test plan

- Compile changed Python modules.
- Run focused tests for the runner, evidence validator, and no-WING gate.
- Run quick no-WING release check.
- Run direction guard, memory lint, diff check, and full pytest before commit.

## Where the other agent may disagree

Kimi may prefer a terminal prompt for operator confirmation. A CLI flag is the
better fit here because the proof workflow is generated, copied, hashed,
validated, and reviewed as files.
