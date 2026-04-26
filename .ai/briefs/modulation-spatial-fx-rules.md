# Modulation And Spatial FX Rules

## Task

Add source-grounded rules for modulation and spatial effects, then use the
updated rule set in the next offline mix pass.

## Constraints

- Do not change live OSC behavior by default.
- Keep new FX guidance source-grounded and inspectable in JSONL logs.
- Treat numeric settings as bounded audition starts, not fixed artistic truth.
- Preserve kick/bass low-end safety and vocal intelligibility.

## Decision

Encode the new guidance as advisory source-knowledge rules covering shared
filtered returns, role-based pre-delay, delay as a lower-masking depth tool,
ducked FX returns, modulation on support layers, modulation/reverb order, and
early/late reflection density.

## Test Plan

- Validate all source IDs in `sources.yaml`.
- Verify the new FX/modulation rules are retrievable.
- Run the project pytest command before the mix pass.
