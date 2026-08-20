# M123 — Instrument Space decision integration

Status: IN PROGRESS

Date: 2026-08-04

## Objective

Complete the Canvas-inspired Instrument Space software path: measured channel
objects become deterministic, explainable decision evidence that participates
in Automixer proposal review for WING and dLive while preserving all existing
console-specific safety and hardware gates.

## Checkpoints

| Checkpoint | Outcome | Evidence | Status |
|---|---|---|---|
| M123-CP01 | Goal, brief and ADR establish the no-direct-authority boundary | brief, proposal, ADR | complete |
| M123-CP02 | Deterministic relations, bounded source-bound history, conflicts and runtime drift are implemented | backend unit tests | complete |
| M123-CP03 | Product Layer and soundcheck proposal review consume fresh evidence with measurement provenance | integration tests | complete |
| M123-CP04 | Operator UI displays conflicts, drift, measurements, provenance and next analyzers without apply gestures | frontend contracts/build | complete |
| M123-CP05 | Targeted and broad safe validation is recorded in project truth | test logs and autopilot ledger | in progress |

## Correction-decision policy

Instrument Space can:

- flag coarse spatial/spectral/level competition;
- rank which existing analyzer should run next;
- add evidence and review context to a proposal;
- report snapshot drift.

Instrument Space cannot:

- calculate a final console parameter from display coordinates;
- import or apply a proposal by itself;
- bypass source identity, ERB masking, mono/phase, headroom or Safety Kernel;
- reuse WING hardware evidence for dLive or vice versa;
- grant live authority.

## Done criteria

- Stable versioned DTO and deterministic pair scoring.
- Missing or stale measurements fail closed to advisory/invalid outcomes.
- Runtime history is bounded in memory, resets on mixer/audio source identity
  changes and never persists or grants authority.
- Product Layer includes decision support with immutable authority denial.
- Soundcheck proposal values and transaction plans are unchanged; annotations
  are evidence-only and stale/undated scene evidence is rejected.
- Frontend exposes review items, drift, measurement provenance, evidence quality
  and next analyzers.
- Targeted backend/frontend tests and build pass.
- Full-suite or release-rail blockers are recorded without weakening CP05.
- dLive hardware parity remains explicitly waiting for a fresh attended proof.

## Current software evidence

- Focused backend/runtime regression: `43 passed`.
- Broad Product Layer/M123/soundcheck regression: `337 passed`.
- Full frontend contract suite: `215 passed`; production Vite build passed.
- Bounded 64-channel diagnostic benchmark: 2,016 evaluated pairs, 128 emitted
  relations, 200 runs, 16.677 ms mean, 18.590 ms p95, 19.882 ms max. This is
  local diagnostic evidence, not a production SLA.
- Final CP05 truth/no-console/full-suite validation remains pending after the
  last documentation and ledger write.

## Rollback

Remove the M123 module, Product Layer field, proposal annotation and UI section.
No mixer state, proposal queue state or console transport must require cleanup.
