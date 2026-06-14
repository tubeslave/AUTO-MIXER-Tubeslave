# Codex Proposal: WING MVP Prestart Backend No-Authority Detail

## Short thesis

Require the no-WING backend-port prestart result to carry explicit `false`
authority flags, and make the read-only preflight runner fail closed when those
flags are absent or stale.

## What I understood

The current first WING MVP task is still a real read-only connection proof. The
offline prestart gate can only prepare that run. It must not let a minimal or
old no-WING summary become the evidence that unlocks a backend WebSocket
attempt.

## Proposed solution

- Add a shared tuple of required `false` backend-port detail fields to
  `backend/wing_readonly_preflight_contract.py`.
- Populate those fields in `tools/no_wing_release_check.py` for the normal
  free-port path, busy-port path, and operator-confirmed intended-backend path.
- Validate those fields in `tools/wing_readonly_preflight_runner.py` as part of
  the prestart summary and expected-plan contract.
- Preserve `frontend/test-results/.last-run.json` around the frontend E2E
  release check.
- Add regression tests for missing safety detail flags and preserved command
  artifacts.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/no_wing_release_check.py`
- `tools/wing_readonly_preflight_runner.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- Council trail and report files for this task.

## Alternatives considered

- Leave missing backend-port detail fields as acceptable. Rejected because
  absence is weaker evidence than an explicit no-authority verdict.
- Only document the required fields. Rejected because the real runner needs an
  executable fail-closed check before WebSocket.

## Risks

- Older no-WING prestart summaries without the explicit detail fields will now
  be rejected. This is intentional; operators should rerun the current prestart
  gate before hardware work.

## Test plan

- Focused `pytest` for read-only preflight runner and backend-port checks.
- `git diff --check`.
- no-WING quick validation with `--skip-frontend` in the restricted sandbox.

## Where the other agent may disagree

Kimi may argue that missing fields should be treated as legacy-compatible. The
safer boundary is to require the current explicit no-authority contract before
opening WebSocket for a physical WING session.
