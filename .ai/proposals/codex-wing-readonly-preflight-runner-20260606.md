# Codex Proposal: WING Read-Only Preflight Runner

## Short thesis

Add a dedicated read-only preflight runner so the first WING proof steps can be
captured without using the proposal apply path or direct mixer writes.

## What I understood

The active project priority is the supervised WING Rack MVP. The next useful
work should reduce named blockers: read-only connection proof, physical console
readback proof, and operator-visible evidence.

## Proposed solution

- Create `tools/wing_readonly_preflight_runner.py`.
- Connect only to the backend WebSocket; do not discover or connect WING.
- Send only `get_operator_mode`, `get_state`, and `query_mixer_parameter`.
- Support selected channels and `fader` / `gain` readback.
- Mark the artifact blocked if backend is not in `wing` mode.
- Mark the artifact blocked if `--require-physical-readback` is set and any
  queried parameter is not `verified_from_physical_readback`.
- Record sent request types and explicitly list the full hard-safety forbidden
  request set, including fader/gain/trim/EQ/dynamics/routing/snapshot/reset/
  mute/insert/FX writes, proposal apply/create, rollback, emergency stop,
  operator-mode mutation, connect, and disconnect messages.

## Likely files to touch

- `tools/wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `.ai/memory/project.md`
- `Docs/INDEX.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_RUNNER_20260606.md`

## Alternatives considered

- Reuse `wing_operator_proposal_validation_runner.py` dry-run mode. Rejected:
  that runner is tied to later proposal apply proof and does not isolate the
  first read-only preflight step.
- Add a backend handler. Rejected: existing read-only handlers already expose
  the needed protocol.

## Risks

The runner can only prove current backend-observed state. It cannot claim
physical proof unless the backend returns physical readback evidence from the
connected WING.

## Test plan

- Focused fake-WebSocket tests.
- `python3 -m py_compile` for the runner.
- `python3 tools/memory/memory_lint.py`.
- `git diff --check`.

## Where the other agent may disagree

Kimi may prefer starting with a live operator run. This implementation keeps the
tooling safe and ready for that run without requiring hardware in this coding
turn.
