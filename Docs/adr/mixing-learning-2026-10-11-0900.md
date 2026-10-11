# ADR: keep restoration-derived stems outside automatic mixing

- **ID:** ADR-ML-2026-10-11-0900
- **Date:** 2026-10-11
- **Status:** accepted for research governance
- **Related update:** [ML-2026-10-11-0900](../mixing_learning_updates/ML-2026-10-11-0900.md)

## Context

arXiv:2610.08284 reports a modular mixture-restoration, separation and stem-restoration pipeline. The formulation explicitly omits bus-level processing, uses validation and test sets from independent collections, reports metric disagreement, and includes no subjective listening test. The paper advertises code and models, but its linked GitHub repository was empty when checked on 2026-10-11.

Project policy also requires incoming claims to be tested on a project-owned or appropriately licensed multitrack before acceptance.

## Decision

Restoration-derived stems remain **shadow evidence only**. They must not replace project originals, drive automatic mix decisions, or become training truth. Any future evaluation must:

1. pin source, license, environment, checkpoint and configuration;
2. hold the separator and all non-target factors fixed;
3. compare raw outputs against true clean project stems;
4. create separate BS.1770-matched copies for blind listening;
5. protect identity, attacks, phase, room/bleed, low-end timing, mono and reconstruction coherence;
6. record Dmitry's verdict before considering rule promotion.

`RULE-ML-20261011-0900-01` remains `auto_apply:false`. `EXP-ML-20261011-0900-01` remains `not_run_blocked` while the advertised repository is empty.

## Consequences

The Stage-1 result is retained as a bounded ablation hypothesis, not as a mixing prescription. Metric improvements cannot override a listener regression or a protected-quality failure. Generated or restored stems cannot be called the original multitrack. The next executable work remains the existing `Пена` listener protocol.
