# M089: Mute management read-only context gates

Generated: 2026-06-20T11:38:10Z

## Decision

M089 moves the project toward the real AI Mixing Engineer goal by hardening the
console-mute proposal layer:

- current mute context must be fresh;
- the context must be read-only;
- the context must carry no live-write authority;
- missing, stale or live-authority context keeps results analysis-only;
- proposal evidence now records `current_mute_cache_no_live_authority`.

Missing, stale or live-authority context keeps mute management analysis-only,
even when source activity, source identity, intent and persistence evidence
otherwise look valid.

## Safety Boundary

- No backend WebSocket was opened.
- No WING or OSC transport was contacted.
- No mute write, proposal apply, rollback, emergency-stop or readiness update
  was sent.
- No hardware evidence was created.
- Mute/unmute recommendations remain proposal-only / analysis-only evidence
  surfaces.
- `live_authority_granted` remains `false`.
- `readiness_closure_allowed` remains `false`.

## Validation Commands

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_mute_management_proposal.py -q`:
  `15 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m089_mute_management_readonly_context_gates_stay_readonly -q`:
  `1 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`:
  `102 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`:
  `4006 passed, 1 skipped, 1 warning`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`: PASS
- JSON validation for `ops/autopilot/STATE.json`: PASS
- JSONL validation for `ops/autopilot/RUN_LEDGER.jsonl`: PASS

## Why This Advances The Goal

M083 hardened source and arrangement context. M084 hardened downstream
EQ/frequency/balance proposals. M085 hardened effects and sends. M086 hardened
master-bus/output optimization. M087 hardened concert-time performance
adaptation. M088 hardened console routing. M089 extends the same safety pattern
to console mute/unmute, a separate console-control capability required by the
project goal.

## Next Required External Step

M001 and M002 remain `WAITING_FOR_EXTERNAL_VALIDATION`. The next WING-specific
proof is still a human-supervised read-only preflight artifact. M089 does not
replace that proof; it strengthens offline mute evidence for future assisted
and replay-gated console-control decisions.
