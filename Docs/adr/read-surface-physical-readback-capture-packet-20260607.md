# ADR: Read Surface Physical Readback Capture Packet

## Context

The long-term Automixer goal requires reading and understanding all relevant
console state: channels, groups, matrices, buses, FX, scenes, presets, and raw
mixer state. The repository already has read-surface physical-readback
contracts and a non-executing capture plan, but operators still need a concrete
packet artifact for the next physical fixture target.

## Options considered

1. Keep the capture plan available only as a backend Python function.
2. Add an offline CLI packet builder that renders the current capture plan for
   the operator.
3. Add an automated hardware capture runner.

## Decision

Add `tools/build_read_surface_physical_readback_capture_packet.py`. The tool
renders the current `read_surface_physical_readback_capture_plan` as Markdown or
JSON, including safety flags, next target surface, remediation action, required
artifact contracts, operator sequence, forbidden actions, and focused
validation.

## Why this won

The operator gets a concrete packet for closing read-surface fixture gaps
without adding console transport or live-read authority. This moves the broad
"read all console state" goal forward while preserving the current WING proof
boundary.

## Rejected alternatives

- Backend-only plan output is too easy to miss during a physical capture
  session.
- Automated hardware capture would be premature until the current real WING
  read-only proof is captured and reviewed.

## Implementation plan

1. Add an offline packet builder CLI.
2. Add unit tests for safety flags, current next target, Markdown/JSON output,
   and transport-free source.
3. Run the packet builder inside `tools/no_wing_release_check.py` as a smoke
   command.
4. Keep the packet non-authoritative: it cannot close fixtures, update
   readiness, or grant live authority.

## Test plan

- `python3 -m py_compile tools/build_read_surface_physical_readback_capture_packet.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_capture_packet.py tests/test_no_wing_release_check.py::test_no_wing_release_check_quick_mode_still_runs_write_surface_scan`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-read-surface-capture-packet --command-timeout-sec 300`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk is limited to offline operator guidance. Rollback is to remove the packet
builder, no-WING command check, tests, and documentation. No live WING/OSC/MIDI
transport or write path is affected.
