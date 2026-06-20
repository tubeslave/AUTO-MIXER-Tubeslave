# ADR: Project Director Post-M104 Intake Loop Guard

## Context

M095 through M104 added useful WING read-only intake guards: reserved roots,
declared paths, canonical roots, source paths, run-id shape and capture-time
ordering. That work tightened the M001 evidence intake. It also created a new
direction risk: while M001 still waits for a real WING artifact, Goal-mode can
keep selecting another small intake-hardening milestone and appear busy without
advancing to M002 physical readback.

## Options Considered

1. Continue adding WING read-only intake guards whenever a theoretical forged
   artifact shape can be imagined.
2. Stop all Goal-mode work until hardware evidence arrives.
3. Keep Goal mode active, but make Project Director detect the post-M104
   intake-hardening streak and narrow further autonomous work to real external
   validation, an explicit operator request or a named unverified contract gap.

## Decision

Use option 3. Project Director now derives recent commit subjects from its
existing repo evidence, detects the consecutive WING read-only intake guard
streak after M104, and marks the cycle with
`pre_hardware_intake_hardening_loop_guard: true`. When that guard is active,
the allowed autonomous scope is `external_validation_wait_or_named_gap_only`.

## Why This Won

This preserves the useful safety work already done while preventing the project
from endlessly improving the same WING read-only intake surface. It still allows
legitimate work if a real artifact arrives, the operator asks for a specific
handoff/intake change, or a concrete unverified contract gap is named.

## Rejected Alternatives

Option 1 was rejected because it can convert the external WING blocker into an
infinite local-hardening loop. Option 2 was rejected because control-plane
state, handoff preservation and named-gap fixes can still be valuable while
waiting for hardware.

## Implementation Plan

1. Add Project Director helpers that extract recent commit subjects from
   direction-guard evidence.
2. Detect a consecutive WING read-only intake guard streak that includes the
   M104 capture-time-order guard.
3. Emit directive fields for the loop guard and narrow allowed/forbidden scope.
4. Record M105 rail coverage without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_project_director_agent.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m105_project_director_post_m104_intake_loop_guard_stays_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M105 commit, which
restores the previous Project Director scope and M105 state/report rows. The
risk is an overly strict direction guard; the escape hatch is intentionally
explicit: a real artifact, operator request or named unverified contract gap can
justify further handoff/intake work.
