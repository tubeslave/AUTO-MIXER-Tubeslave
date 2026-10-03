# STUDIO Routed Contribution Renderer v1

## Decision

Compression candidates that alter a source must be evaluated by rerendering the real offline session graph, not by subtracting a presumed dry contribution from an already rendered premaster.

A fixed additive source contribution is not generally well-defined once the source feeds sends/returns, source-dependent automation, masking logic, sidechains, or a shared nonlinear mix bus. The authoritative object is therefore the complete deterministic session render under a source override.

## Contract

`audio_workbench.routed_contribution.SessionRender` carries the full stereo mix plus optional rerendered vocal, drums and early-room buses. A caller supplies a deterministic `render_session(overrides)` callback. The evaluator performs four renders: baseline with the original source, an identical no-change rerender, the level-matched compression candidate, and a source-muted counterfactual.

The source-muted difference is labelled a context-dependent marginal only. It is never treated as an additive stem when shared nonlinear processing exists.

## Safety

The no-change rerender must be deterministic. Missing anchor buses, invalid layouts, level-match failures, audition failures, or rerender nondeterminism are objective vetoes and force rollback through the existing Compression Iteration / Perceptual Critic gate. Even a machine-safe survivor can only become a level-matched audition and remains pending human listening. Baseline promotion is forbidden here.

## Validation plan

Synthetic integration uses a real routing graph fixture with a dry path, delayed wet return and shared nonlinear bus. It verifies that the authoritative candidate differs from naive dry subtraction, that no-change is exact, nondeterminism is rejected, source arrays stay immutable, and anchor-bus absence fails closed. Exact-branch CI must pass before merge.

## Real-session limitation

The existing Belye Stai delivery recipe is a procedural Python render with hard-coded project paths and top-level execution. Inspection confirms source-dependent vocal masking/automation, reverb/slap sends and shared mix-bus compression, so direct subtraction is specifically invalid there. This task creates the renderer boundary, but does not claim that the old recipe has already been migrated into the callback contract. A dedicated Belye Stai session wrapper is the next bounded task.
