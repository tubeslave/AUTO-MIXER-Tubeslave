# ADR: keep cue-coupled distance as a bounded experiment, not an automatic rule

- **Date:** 2026-10-03
- **Update:** `ML-2026-10-03-0900`
- **Status:** accepted

## Context

The DAFx-26 Distance Space demo couples direct level, reverb wet level, spectral tilt, pre-delay, and low-mid compensation under one distance control. In one remote-headphone test with 18 participants and a shared three-source scene, the authors report better distance-placement metrics and unanimous composite-quality preference over their uncoupled manual baseline. The three-page paper does not provide enough detail on baseline construction, randomization, repeat structure, individual data, uncertainty, or inferential testing to generalize its exact mappings. Some mappings are explicitly hyper-real production choices.

## Decision

1. Add the paper and repository as source-grounded cards with exact reading depth and version identifiers.
2. Treat correlated distance cues as a testable production hypothesis, not a universal preset or proof that a single macro is always preferable.
3. Keep `RULE-ML-20261003-0900-01` at low-to-moderate hypothesis confidence and `auto_apply:false`.
4. Permit only a controlled project-owned A/B in which the single factor is a preregistered deterministic distance-macro value; freeze all other processing and globally loudness-match the complete excerpts.
5. Do not integrate or modify DSP/runtime code from the external repository without a separate source, asset, dependency, and build audit.

## Consequences

- `EXP-ML-20261003-0900-01` is `not_run` and follows the already-prioritized parallel-drum experiment.
- The paper's 1–20 m labels and parameter table remain implementation facts, not recommended settings.
- Lead vocal is not the default test subject; choose a source whose musical role is intentionally rear-plane.
- The next deep review must audit repository implementation parity and the fuller evaluation materials before confidence can increase.

