# Automixer Read Surface Physical Readback Capture Packet - 2026-06-07

## Summary

This slice adds an offline operator packet for the broader read-surface physical
fixture path. It supports the long-term requirement to read all console state
without widening live writes.

`tools/build_read_surface_physical_readback_capture_packet.py` renders the
current `read_surface_physical_readback_capture_plan` into Markdown or JSON. The
packet includes:

- the next physical readback target surface;
- the remediation action id;
- required artifact contracts;
- the operator sequence;
- forbidden live/mutation actions;
- focused validation command.

`tools/no_wing_release_check.py` now runs this packet builder as an offline
smoke command in the release-check run directory.

## Safety Boundary

- No live WING/OSC/MIDI/TCP/WebSocket action was run.
- No write, proposal, rollback, emergency-stop, connect, or disconnect path was
  added.
- The packet does not query a console.
- The packet does not execute hardware tests.
- The packet does not close fixture gaps.
- The packet does not grant live authority or update autonomy readiness.

## Files Changed

- `tools/build_read_surface_physical_readback_capture_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_capture_packet.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/read-surface-physical-readback-capture-packet-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PACKET_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/build_read_surface_physical_readback_capture_packet.py tools/no_wing_release_check.py`
  passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_capture_packet.py tests/test_read_surface_physical_readback_capture_plan.py tests/test_no_wing_release_check.py::test_no_wing_release_check_quick_mode_still_runs_write_surface_scan`
  passed: `10 passed`.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-read-surface-capture-packet --command-timeout-sec 300`
  passed with the packet builder included in the executed command plan.
- `python3 tools/build_read_surface_physical_readback_capture_packet.py --format json --output /tmp/automixer-read-surface-capture-packet-smoke.json --quiet`
  passed; the packet type was `read_surface_physical_readback_capture_packet`,
  the current next target was `raw_mixer_state_cache`, and
  `live_authority_granted` was `false`.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_capture_packet.py tests/test_no_wing_release_check.py`
  passed: `148 passed`.
- `python3 tools/memory/memory_lint.py` passed.
- `/opt/homebrew/bin/git diff --check` passed.

## Remaining Gap

The packet only prepares the operator path. Physical readback fixtures still
need to be captured from the intended console, reviewed manually, and accepted
before any fixture gap or readiness row can be closed.
