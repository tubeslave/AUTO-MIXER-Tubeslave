# AI Mixing Offline Chain

## Context

The project already has live AutoFOH control, a `mix_agent` offline facade,
MuQ-Eval support, MERT-like perceptual evaluation, and several larger offline
experiments. The new requirement is a unified offline test chain where AI/ML
critics evaluate rendered candidates while a safety layer remains the final
arbiter.

## Options considered

1. Extend `tools/offline_agent_mix.py`.
2. Add routes to the backend server and reuse live handlers.
3. Add a separate `ai_mixing_pipeline` package that adapts existing modules.

## Decision

Use a separate `ai_mixing_pipeline` package and keep it offline-only.

## Why this won

This keeps live sound paths untouched, preserves existing experiments, and gives
each requested technology a clear role. Existing modules remain useful through
adapters instead of being rewritten or deleted.

## Rejected alternatives

- `tools/offline_agent_mix.py` extension: too broad for the first unified chain.
- Backend server integration: risks accidental OSC/MIDI coupling.
- Mandatory heavy ML dependencies: breaks CI and violates graceful fallback.

## Implementation plan

1. Add role-specific packages and common dataclasses.
2. Implement standard `AudioCritic` adapters with fallbacks.
3. Load multitrack/reference input, analyze, and render conservative candidates
   with a pre-safety peak margin below the Safety Governor limit.
4. Score candidates, renormalize available critic weights, and select a best
   candidate or `no_change`.
5. Run the Safety Governor before accepting the result.
6. Write JSONL, CSV, Markdown, JSON action files, snapshots, and rendered WAVs.

## Test plan

- Unit tests for config, critic interface, fallbacks, renderer, decision engine,
  and safety governor.
- End-to-end offline test with synthetic stems and no installed heavy models.
- Existing targeted tests for MuQ/perceptual/mix_agent compatibility.

## Risks and rollback

Fallback critic scores are only proxies, so reports must expose model
availability. Rollback is simple: do not call the new CLI; no live control path
depends on it.
