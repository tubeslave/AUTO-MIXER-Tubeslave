# Council Decision: System Inspector Phase 0 and Snapshot Replay

Codex and Kimi agree on an isolated offline package, deterministic canonical
replay, safe YAML, explicit `UNKNOWN`, imported-only Dante evidence and a
static ban on live/write dependencies.

The synthesis retains three master-prompt requirements that Kimi proposed
deferring:

- runtime write exception plus redacted security audit;
- repository CLI family `automixer inspector ...`;
- readiness score, but only with minimum coverage and explicit confidence,
  unknowns and freshness.

The accepted architecture, rejected options, tests and rollback are recorded
in `Docs/adr/system-inspector-phase0-replay-20260727.md`.

