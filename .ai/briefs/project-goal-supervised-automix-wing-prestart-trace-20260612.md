# Project Goal Supervised AutoMix WING Prestart Trace

## Context

The proof-plan registry now exposes the accepted-candidate prestart trace gate. The higher-level project goal capability matrix still described Supervised AutoMix progress as generic fake-mixer and physical proof before widening.

## Task

Update `mode_supervised_automix` so the long-term goal matrix explicitly requires accepted-candidate prestart trace from read-only WING preflight review before WING physical proof steps can support widening.

## Constraints

- Matrix/test/docs only.
- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- Keep current live-write kinds unchanged.

## Test Plan

- Focused project goal matrix test.
- Affected matrix/registry/WING proof tests.
- Full backend pytest suite.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
