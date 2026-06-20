# M085: FX send monitor and return safety gates

Generated: 2026-06-20T07:35:10Z

## Decision

M085 moves the project toward the real AI Mixing Engineer goal by hardening the
effects/send proposal layer:

- FX send/return context must be fresh and carry no live authority;
- FX parameter schema hints must be mapped and no-authority;
- monitor-exposed sends require fresh monitor-safety context;
- return level, feedback and intelligibility safety are explicit evidence;
- old-value and rollback readback sources are surfaced before any promotion.

Missing, stale, generic, mismatched or live-authority context keeps FX send
results analysis-only.

## Safety Boundary

- No backend WebSocket was opened.
- No WING or OSC transport was contacted.
- No write, proposal apply, rollback, emergency-stop or readiness update was
  sent.
- No hardware evidence was created.
- FX send proposals remain proposal-only / analysis-only evidence surfaces.
- `live_authority_granted` remains `false`.
- `readiness_closure_allowed` remains `false`.

## Validation Commands

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_fx_send_proposal.py -q`:
  `24 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m085_fx_send_monitor_return_safety_gates_stay_readonly -q`:
  `1 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`:
  `98 passed`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`:
  `3991 passed, 1 skipped, 1 warning`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`: PASS
- JSON validation for `ops/autopilot/STATE.json`: PASS
- JSONL validation for `ops/autopilot/RUN_LEDGER.jsonl`: PASS

## Why This Advances The Goal

M083 hardened source and arrangement context. M084 used that context for
downstream EQ/frequency/balance proposals. M085 extends the same safety pattern
to effects and sends, a separate capability required by the project goal.

## Next Required External Step

M001 and M002 remain `WAITING_FOR_EXTERNAL_VALIDATION`. The next WING-specific
proof is still a human-supervised read-only preflight artifact. M085 does not
replace that proof; it strengthens offline FX/send evidence for future assisted
and replay-gated mix decisions.
