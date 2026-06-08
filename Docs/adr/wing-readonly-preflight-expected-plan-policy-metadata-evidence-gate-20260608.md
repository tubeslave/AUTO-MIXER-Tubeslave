# ADR: WING Read-Only Preflight Expected Plan Policy Metadata Evidence Gate

## Context

The supervised WING Rack MVP remains on the first non-closed blocker: WING
read-only connection proof. A real preflight summary must reference the offline
expected-plan JSON generated before the hardware run. The artifact evaluator
already checked the plan file hash, request sequence, summary fields, and core
no-contact booleans.

The remaining gap was that a tampered or stale plan file could still omit or
change policy metadata such as the selected MVP blocker, allowed/forbidden
request lists, required fail-closed conditions, and operator-required boundary
without being rejected by the evidence gate.

## Options considered

1. Rely only on the no-WING release gate to validate plan policy metadata.
2. Duplicate the policy metadata checks inside the real artifact evidence gate.
3. Remove expected-plan policy metadata from the preflight plan.

## Decision

Use option 2. The WING read-only preflight evidence validator now rejects a
referenced expected-plan file unless it preserves the selected MVP blocker,
operator-required boundary, plan output path, allowed request types, forbidden
request types, and required fail-closed conditions.

## Why this won

The real hardware artifact review should be self-contained. If an operator
reviews a captured `summary.json` and linked plan file, that linked plan must
still prove it belongs to the current WING read-only blocker and cannot claim a
broader or unsafe execution boundary.

The same cycle also fixed the handoff packet builder so `python3
tools/build_wing_readonly_preflight_handoff_packet.py ...` works when invoked as
a repo-local script path by no-WING release checks. That keeps the operator
handoff command executable, not only importable from tests.

## Rejected alternatives

Relying only on the release gate was weaker because the real artifact review may
happen after plan generation. Removing metadata would make the operator handoff
less auditable.

## Implementation plan

- Add expected-plan policy constants to
  `backend/wing_readonly_preflight_evidence.py`.
- Validate `selected_mvp_blocker`, blocker status, `operator_required`,
  `plan_output`, allowed request types, forbidden request types, and
  fail-closed conditions when reviewing linked expected-plan files.
- Update the test fixture and add a tampered-plan regression in
  `tests/test_wing_readonly_preflight_evidence.py`.
- Make `tools/build_wing_readonly_preflight_handoff_packet.py` insert the repo
  root on `sys.path` before importing repo-local `tools` modules.
- Add a CLI regression in `tests/test_wing_readonly_preflight_handoff_packet.py`
  that runs the builder as a script path.

## Test plan

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py tools/wing_readonly_preflight_runner.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_runner.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root artifacts/no_wing_release_check_current_cycle --command-timeout-sec 300`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk is low because this only tightens offline artifact validation and fixes a
repo-local script import path. It does not open WebSocket, query WING, or send
OSC/mixer commands. Rollback is to remove the additional expected-plan metadata
checks, the CLI import fix, and related tests, but that would make captured
artifacts easier to misread and the handoff command easier to regress during
manual review preparation.
