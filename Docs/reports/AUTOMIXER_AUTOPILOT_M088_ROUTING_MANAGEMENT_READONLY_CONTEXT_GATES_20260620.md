# M088: Routing management read-only context gates

Generated: 2026-06-20T11:28:50Z

## Decision

M088 moves the project toward the real AI Mixing Engineer goal by hardening the
console-routing proposal layer:

- current route context must be fresh;
- the context must be read-only;
- the context must carry no live-write authority;
- missing, stale or live-authority context keeps results analysis-only;
- proposal evidence now records `current_route_cache_no_live_authority`.

Missing, stale or live-authority context keeps routing management analysis-only,
even when route diff, source verification, rollback and scope evidence
otherwise look valid.

## Safety Boundary

- No backend WebSocket was opened.
- No WING or OSC transport was contacted.
- No write, proposal apply, rollback, emergency-stop or readiness update was
  sent.
- No hardware evidence was created.
- Routing proposals remain proposal-only / analysis-only evidence surfaces.
- `live_authority_granted` remains `false`.
- `readiness_closure_allowed` remains `false`.

## Validation Commands

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_routing_management_proposal.py -q`:
  `14 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m088_routing_management_readonly_context_gates_stay_readonly -q`:
  `1 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`:
  `101 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`:
  `4003 passed, 1 skipped, 1 warning`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`: PASS
- JSON validation for `ops/autopilot/STATE.json`: PASS
- JSONL validation for `ops/autopilot/RUN_LEDGER.jsonl`: PASS

## Why This Advances The Goal

M083 hardened source and arrangement context. M084 hardened downstream
EQ/frequency/balance proposals. M085 hardened effects and sends. M086 hardened
master-bus/output optimization. M087 hardened concert-time performance
adaptation. M088 extends the same safety pattern to console routing, a separate
console-control capability required by the project goal.

## Next Required External Step

M001 and M002 remain `WAITING_FOR_EXTERNAL_VALIDATION`. The next WING-specific
proof is still a human-supervised read-only preflight artifact. M088 does not
replace that proof; it strengthens offline routing evidence for future assisted
and replay-gated console-control decisions.
