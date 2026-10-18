# ADR: keep Aura as a blocked, matched-compute research hypothesis

- Status: accepted_for_research_workflow
- Date: 2026-10-18
- Update: [ML-2026-10-18-1800](../mixing_learning_updates/ML-2026-10-18-1800.md)
- Source search: [ML-2026-10-18-1500](../mixing_learning_updates/ML-2026-10-18-1500.md)
- Scope: research and experiment planning only
- Runtime effect: none

## Context

Aura reports reference-conditioned mono-to-stereo generation with improved spatial-cue metrics and an exploratory five-listener study. The official demonstrator and Stable Audio 3 base dependency are public, but the Aura-specific sampler/config, exact model revisions, critical reward/optimizer/STFT parameters, listener-level data and a frozen evaluation manifest are not available. MUSDB18-HQ is educational-only and source-rights constrained.

The prior reward-off versus reward-on plan removed the reward phase on A. That also changes compute and latent trajectory, so it is not a clean single-factor comparison.

## Decision

1. Retain Aura only as a low-confidence, human-gated, research-only candidate.
2. Keep RULE-ML-20261018-1500-01 rev.2 at auto_apply:false.
3. Keep EXP-ML-20261018-1500-01 rev.2 at not_run_blocked.
4. For any future A/B, execute identical second-phase passes on both sides and vary only g_reward=0 versus g_reward=1.
5. Treat kappa=1 as the unchanged-reference coherence target, not as reward disabled.
6. Preserve raw renders and create separate attenuation-only BS.1770-matched blind copies with logged compensation.
7. Require rights-cleared/project-owned inputs, a frozen reference, exact model/code/environment hashes and Dmitry's blinded decision.
8. If the official implementation/config remains unavailable, label any implementation Aura-like rather than an Aura reproduction.
9. Do not reuse the official demo FLACs as project raw audio or a commercial reference set.

## Consequences

- The base dependency audit is more complete: public MIT code and a lock file exist.
- Execution remains blocked by Aura-specific reproducibility, gated model access/terms, rights, compute approval and test-design gaps.
- Objective IC/ILD metrics are diagnostic only and cannot override content, mono-fold, transient or listening gates.
- No DSP, model, runtime or repository implementation changes follow from this ADR.
