# ADR: Mixing Learning deep review 2026-10-03 18:00 MSK

- **Status:** accepted research-base update
- **Update:** `ML-2026-10-03-1800`
- **Decision scope:** source-card depth, artifact audits, bounded rule reconciliation and unrun experiments only

## Decision

Upgrade `conference:dafx:2026:DAFx26_paper_04` and `conference:dafx:2026:DAFx26_paper_01` from limited-excerpt queue entries to full-text-reviewed source cards. Preserve their mathematical and objective engineering contributions while rejecting any interpretation that orthogonality, IACC, pole histograms, RMS continuity or mean CPU establish perceptual quality or production readiness.

Record both companion repositories as non-reproducible end to end: the FDN site provides demonstrations without implementation/tests/license; the CDC repository provides extensive raw outputs without the renderer/plugin/build source or license.

Reconcile the findings with existing reverb and real-time rules rather than duplicating them. Add one narrowly scoped dynamic-IR transition candidate rule, `auto_apply:false`, and two blocked `not_run` experiments. Do not alter audio, DSP, runtime, models or production behavior.

## Rationale

The FDN paper proves a useful structured matrix family and reports a focused implementation benchmark, but lacks perceptual validation and an open implementation. The CDC paper supplies inspectable objective data and a plausible single-state transition design, but compares against one baseline and explicitly leaves perceptual testing for future work. State continuity and RMS continuity are not equivalent to transparent sound.

## Consequences

- Both queued literature sources are now full-text reviewed; `citation_check:partial` remains.
- `RULE-ML-20260919-1800-01` and `RULE-ML-20260919-1800-04` gain explicit transition/raw-versus-matched and CPU-tail gates.
- `RULE-ML-20261003-1800-01` remains a medium-confidence candidate with `auto_apply:false`.
- `EXP-ML-20261003-1800-01` and `EXP-ML-20261003-1800-02` remain `not_run` and blocked by missing implementations/licensing.
- Video queue remains 84: 7 transcript-complete, 0 partial, 77 metadata-only.
- Execution order remains `EXP-ML-20261001-1800-01`, then `EXP-ML-20261002-1500-01`.
