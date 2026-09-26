# ADR: Routed Compression Family v1

## Context

Compression Director v1.1 creates exactly three bounded compressor hypotheses, while Routed Contribution Renderer v1 evaluates one source candidate through an authoritative full-session rerender. The missing orchestration layer was the complete family boundary: every candidate must receive the same routing/context gates, machine vetoes may remove unsafe or non-improving candidates, and no machine path may silently rank or promote a subjective winner.

The accepted Belye Stai SessionRenderer inserts at the **post-local-track-processing / pre-context-routing** boundary. That distinction is essential. A compressor placed there is an additional processed-stage hypothesis; it does not replace the compressor already used while building the processed track.

## Decision

Add `audio_workbench/routed_compression_family.py` as a STUDIO-only coordinator. It requires the exact `preserve_transient / balanced / control` family from `prepare_compression_iteration`, evaluates every candidate with `evaluate_routed_compression_context`, and returns only two machine outcomes: rejected, or a level-matched audition that has reached the human-listening gate. `winner` and `ranking` are always `None`; baseline promotion is forbidden.

The family coordinator reuses the existing authoritative session renderer, Perceptual Critic and Autonomous Iteration gates rather than creating a second acceptance policy.

## Validation

PR #126, tested head `9d50dc760128ab5e08c674f91d51f2e7ec7692a1`, passed workflow `35958901206` on Python 3.10, 3.11 and 3.12, including compilation, STUDIO offline-boundary validation and the full test step.

A real 207 s Belye Stai experiment reconstructed the frozen premaster with SHA-256 `9cf6c5a9ffbd02d7685d6f16e76dd98525662499ae33364277d1c75ed39c8e76` exactly and zero no-change float error. Fresh Compression Director v1.1 candidates were generated from the processed `VALERA_VOX` track and rerendered through the complete downstream session graph. All three were rejected by the existing vocal-intelligibility target gate: improvements were approximately +0.00844, +0.01132 and +0.01521, below the required +0.02. Therefore no audition pair was exported and no musical winner was inferred.

## Consequence

The orchestration boundary is accepted, but this real experiment also exposes the next architectural limitation: evaluating extra compression after the already-compressed local vocal chain is not the same intervention as replacing the original vocal compressor. The next task is a deterministic raw/local vocal processor adapter with explicit compressor replacement hooks and a no-change reproduction proof before full-session candidate evaluation.

Human listening remains mandatory for subjective acceptance. No learned audio processing, paid services, live/OSC changes, mastering changes or baseline promotion are part of this decision.
