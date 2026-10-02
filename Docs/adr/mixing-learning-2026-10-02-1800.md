# ADR: block modal-reverb adoption until artifact and validation gaps close

- **Date:** 2026-10-02
- **Update:** `ML-2026-10-02-1800`
- **Status:** accepted

## Context

The DAFx-26 modal-reverberator paper provides a clear synthesis pipeline, but its reported regression scores are in-sample diagnostics on IRLS inliers. Features are extracted from resynthesised IRs, generated presets are checked in the same descriptor space used to construct them, and no held-out, cross-validated, external-corpus, or listener evaluation is described. The paper's companion GitHub URL is currently unavailable and absent from the public organisation listing.

## Decision

1. Retain the paper as a source-grounded design hypothesis and domain-boundary reference.
2. Downgrade `RULE-ML-20261002-1500-01` to low confidence and block it until a licensed, version-pinned implementation or approved local prototype exists.
3. Require fixed random seed, identical mode bank, fixed normalisation path, and an identical-setting null test before any modal-control A/B.
4. Do not represent corpus-cloud placement or inlier `R²rob` as perceptual or out-of-sample validation.
5. Keep the UAD-derived pre-reverb-compression experiment executable, but require matching both the reverb feed and the final excerpts.

## Consequences

- No modal-reverb DSP implementation or automatic rule is authorised.
- `EXP-ML-20261002-1800-01` is `not_run_blocked`.
- `EXP-ML-20261002-1500-01` remains `not_run` with a stricter two-stage loudness-matching protocol.
- The next executable project test remains `EXP-ML-20261001-1800-01`.
