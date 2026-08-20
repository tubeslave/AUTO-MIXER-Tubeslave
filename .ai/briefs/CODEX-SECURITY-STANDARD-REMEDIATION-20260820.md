# CODEX-SECURITY-STANDARD-REMEDIATION-20260820

## Goal

Fix all 13 findings in completed standard scan
`b6b64252-e684-44a9-abb6-c7d4360fc105` without creating new live-console
authority or overwriting unrelated changes in the dirty collaborative checkout.

## Scope

- WING readback provenance, WebSocket authentication/admission, mixer transport
  ownership, and Mixing Station trust.
- Gain/pan and mix-console JSONL resource bounds.
- Sandbox-render and replay evidence truthfulness.
- Safety configuration finiteness.
- Virtual Mixer loopback exposure and numeric parsing.
- GPT control bridge connection and input-size bounds.

## Safety boundary

Offline source changes and tests only. Do not discover, connect to, or send to
WING, dLive, Mixing Station, Ableton, or audio hardware. Unauthenticated OSC
readback cannot become physical proof. Software tests do not establish console
motion, acoustic safety, or concert readiness.

## Acceptance

Each finding needs a negative regression test and a positive bounded control.
Run focused tests, owning-module tests, no-WING release checks, full pytest,
syntax/import checks, and `git diff --check`. Preserve unrelated dirty-tree
work and report any external physical proof as still absent.
