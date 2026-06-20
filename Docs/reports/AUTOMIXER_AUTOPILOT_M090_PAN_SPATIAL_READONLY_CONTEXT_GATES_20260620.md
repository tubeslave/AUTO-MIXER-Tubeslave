# M090: Pan spatial read-only context gates

Generated: 2026-06-20T11:55:57Z

## Decision

M090 moves the project toward the real AI Mixing Engineer goal by hardening the
pan/stereo placement proposal layer:

- current pan context must be fresh;
- current pan context must be read-only;
- current pan context must carry no live-write authority;
- non-center pan requires fresh mono/phase safety evidence;
- mono/phase safety context must carry no live-write authority;
- missing, stale, unsafe or live-authority context keeps results analysis-only.

## Safety Boundary

- No backend WebSocket was opened.
- No WING or OSC transport was contacted.
- No pan write, proposal apply, rollback, emergency-stop or readiness update
  was sent.
- No hardware evidence was created.
- Pan recommendations remain proposal-only / analysis-only evidence surfaces.
- `live_authority_granted` remains `false`.
- `readiness_closure_allowed` remains `false`.

## Validation Commands

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_pan_spatial_proposal.py -q`:
  `16 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m090_pan_spatial_readonly_context_gates_stay_readonly -q`:
  `1 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`:
  `103 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`:
  `4007 passed, 1 skipped, 1 warning`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`: PASS
- JSON validation for `ops/autopilot/STATE.json`: PASS
- JSONL validation for `ops/autopilot/RUN_LEDGER.jsonl`: PASS

## Why This Advances The Goal

M083 hardened source and arrangement context. M084 hardened downstream
EQ/frequency/balance proposals. M085 hardened effects and sends. M086 hardened
master-bus/output optimization. M087 hardened concert-time performance
adaptation. M088 hardened console routing. M089 hardened console mute/unmute.
M090 extends the same safety pattern to pan/stereo placement, a separate
mixing and console-control capability required by the project goal.

## Next Required External Step

M001 and M002 remain `WAITING_FOR_EXTERNAL_VALIDATION`. The next WING-specific
proof is still a human-supervised read-only preflight artifact. M090 does not
replace that proof; it strengthens offline pan evidence for future assisted
and replay-gated console-control decisions.
