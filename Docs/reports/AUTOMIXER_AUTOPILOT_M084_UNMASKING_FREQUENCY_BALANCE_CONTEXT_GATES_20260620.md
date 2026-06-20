# M084: Unmasking, frequency and balance context gates

Generated: 2026-06-20T07:20:47Z

## Decision

M084 moves the project toward the real AI Mixing Engineer goal by hardening
three downstream mix-decision layers:

- cross-channel unmasking;
- source/stem frequency separation;
- bus/group/DCA balance.

These proposal families now require fresh read-only/no-authority context before
emitting proposal candidates. Cross-channel unmasking also requires usable
arrangement-priority evidence before choosing the protected and moving channels.
Missing, stale or live-authority context keeps the result analysis-only.

## Safety Boundary

- No backend WebSocket was opened.
- No WING or OSC transport was contacted.
- No write, proposal apply, rollback, emergency-stop or readiness update was
  sent.
- No hardware evidence was created.
- Unmasking, frequency separation and bus/group balance remain proposal-only /
  analysis-only evidence surfaces.
- `live_authority_granted` remains `false`.
- `readiness_closure_allowed` remains `false`.

## Validation Commands

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_cross_adaptive_eq.py tests/test_frequency_separation_proposal.py tests/test_bus_group_balance_proposal.py -q`:
  `36 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m084_unmasking_frequency_balance_context_gates_stay_readonly -q`:
  `1 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`:
  `97 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`:
  `4212 passed, 1 skipped, 1 warning`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`: `PASS`
- JSON validation for `ops/autopilot/STATE.json`: passed
- JSONL validation for `ops/autopilot/RUN_LEDGER.jsonl`: passed

## Why This Is Not Looping

M081 and M082 hardened WING handoff/review proof. M083 hardened source identity
and arrangement roles. M084 now uses that upstream evidence in downstream mix
decisions. This is a dependency chain toward the project goal, not repeated
improvement of one feature.

## Next Required External Step

M001 and M002 remain `WAITING_FOR_EXTERNAL_VALIDATION`. The next WING-specific
proof is still a human-supervised read-only preflight artifact. M084 does not
replace that proof; it strengthens offline downstream mix-decision evidence for
future assisted and replay-gated mix decisions.
