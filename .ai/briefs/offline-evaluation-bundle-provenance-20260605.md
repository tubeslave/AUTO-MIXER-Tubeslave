# Brief: offline-evaluation-bundle-provenance-20260605

## Context

Recent offline ABX work can require safety-audit evidence and candidate-render
checksum provenance. The remaining gap is the larger bundle boundary: a render,
the safety audit, the ABX artifact, and the ABX gate report can still live as
separate JSON/files without one strict manifest tying their checksums together.

## Task

Add an offline-only evaluation bundle validator that checks one local manifest.
It must prove that candidate render, safety audit, ABX artifact, ABX gate
report, and tool metadata agree on checksums and offline/live policy.

## Safety constraints

- Do not import WING, OSC, mixer clients, websocket servers, or DSP apply paths.
- Do not execute live writes or widen live authority.
- Treat the bundle as offline evidence only.
- Fail closed if a checksum, passing audit, ABX candidate link, or gate-report
  source link is missing.

## Acceptance criteria

- A complete fixture bundle passes and reports `offline_only=True` and
  `live_write=False`.
- A safety audit for a different render fails.
- A stale ABX gate report that names a different ABX artifact checksum fails.
- Focused tests and memory lint pass.
