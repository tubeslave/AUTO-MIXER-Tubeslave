# Codex Proposal: AI Mixing Offline Chain

## Short Thesis

Build the requested chain as an offline-only orchestration package that adapts
the existing analysis, rendering, MuQ, and perceptual modules instead of
rewriting the live mixer stack.

## What I Understood

The repository already controls live sound and has several partial offline
systems. The new chain must unify open-source AI/ML/analysis roles, but it must
not send OSC/MIDI, block the audio callback, or let critics directly change
gain/EQ/compressor settings.

## Proposed Solution

- Add `ai_mixing_pipeline/` with the requested role directories.
- Implement a common `AudioCritic` interface returning one dict shape.
- Provide adapters with deterministic fallbacks:
  MuQ-Eval via `backend/evaluation` when available, MERT via
  `backend/perceptual`, and lightweight technical proxies for unavailable
  Audiobox/CLAP/Essentia/PANNs/BEATs/Demucs.
- Reuse `mix_agent.analysis.loader`/metrics and conservative DSP helpers for
  offline stem loading, analysis, and rendering.
- Generate bounded candidate actions, render them in a sandbox, loudness-match
  candidates, score them, renormalize weights for available critics, then pass
  the chosen candidate through a Safety Governor.
- Write every decision to JSONL and summarize accepted/rejected actions.

## Likely Files To Touch

- `ai_mixing_pipeline/**`
- `configs/ai_mixing_roles.yaml`
- `tests/test_ai_mixing_roles_config.py`
- `tests/test_audio_critic_interface.py`
- `tests/test_offline_test_chain.py`
- `tests/test_sandbox_renderer.py`
- `tests/test_decision_engine.py`
- `tests/test_safety_governor.py`
- `tests/test_missing_model_fallbacks.py`
- `docs/AI_MIXING_OFFLINE_TEST_CHAIN.md`
- `Docs/adr/ai-mixing-offline-chain.md`

## Alternatives Considered

- Extending `tools/offline_agent_mix.py`: rejected for now because it is large,
  already has many specialized passes, and would make the new role architecture
  harder to test.
- Wiring into `backend/server.py`: rejected because offline tests must not
  touch live WebSocket/OSC/MIDI behavior.
- Making heavy models required dependencies: rejected for safety, CI speed, and
  operator reliability.

## Risks

- Fallback critics are not substitutes for real model judgments; reports must
  label them clearly.
- Candidate rendering is intentionally conservative and will not represent every
  future automix-toolkit/Diff-MST action.
- Reference source separation is a stub unless Demucs/Open-Unmix is installed.

## Test Plan

- Validate config loading and role weights.
- Verify each critic returns the standard dict and survives missing models.
- Verify `no_change` candidate exists and renderer creates WAVs.
- Verify dangerous gain/EQ/compression/clipping candidates are rejected.
- Verify decision weights renormalize when critics are unavailable.
- Verify full offline test writes required artifacts.
- Run targeted tests, then the repository standard pytest command if feasible.

## Where Another Agent May Disagree

Another reviewer may prefer to fold this into `mix_agent` immediately. I would
keep the first version separate because it is safer, easier to test, and leaves
the existing offline/live systems undisturbed.
