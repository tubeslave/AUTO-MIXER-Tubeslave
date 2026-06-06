# ADR: WING Read-Only Preflight Handoff Packet

## Context

The first WING MVP blocker still requires a real read-only preflight run against
the intended backend/WING session. The code now has separate capture, artifact
review, pending decision template, and decision-review commands. Running them
out of order or with mismatched channels would create weak evidence before the
first supervised fader/gain proof.

## Options considered

1. Keep the runbook as the only operator handoff surface.
2. Add an offline packet builder that emits the exact command sequence and
   safety boundary for the operator.
3. Add a wrapper that runs the hardware preflight automatically.

## Decision

Add `tools/build_wing_readonly_preflight_handoff_packet.py`. The tool writes a
Markdown or JSON packet with the read-only capture command, artifact review
command, decision-review command, focused validation command, expected
artifacts, required checklist ids, forbidden live request types, and the
blockers that remain open.

## Why this won

The operator gets one concrete execution packet without adding automation over
the real WING session. The packet builder is offline-only: it does not query
WING, execute tests, grant live authority, or update readiness rows.

## Rejected alternatives

- Runbook-only handoff leaves more room for channel or command drift.
- Automatic hardware execution would blur the current first blocker and could
  turn an operator review step into an unreviewed live-session action.

## Implementation plan

1. Add the offline packet builder and unit tests.
2. Validate channel parsing and keep `--operator-confirm-channels` equal to
   `--channels` inside the capture command.
3. Expose the packet tool and command in no-WING release summaries.
4. Fail no-WING release checks if the packet tool is missing or its channel
   argument diverges from the read-only capture command.
5. Document the packet in the WING read-only preflight runbook and report index.

## Test plan

- `python3 -m py_compile tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-handoff-packet --command-timeout-sec 300`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk is limited to offline operator guidance. Rollback is to remove the packet
builder, no-WING summary fields, tests, and runbook/report references. No live
WING transport or write path is affected.
