# ADR: Gate generative mono-to-stereo behind content and human checks

- Status: `accepted_for_research_workflow`
- Date: 2026-10-18
- Update: [ML-2026-10-18-1500](../mixing_learning_updates/ML-2026-10-18-1500.md)
- Scope: research and experiment planning only
- Runtime effect: none

## Context

Aura treats mono-to-stereo upmixing as an ill-posed generative inverse problem. Its reward matches ERB-band inter-channel coherence and level-difference distributions to one reference excerpt and can steer width with `kappa`. The method reports stronger spatial metrics and the best non-reference spatial listening score among its tested systems.

The evidence does not justify production use. Evaluation uses short, vocal-removed MUSDB18-HQ excerpts; Aura is not mono-preserving; restoration improves spatial/high-frequency measures while reducing mid-signal fidelity; the listener test has five participants and no reported uncertainty analysis; and no runnable, licensed Aura artifact was verified. The reward and several evaluation metrics also share IC/ILD cue families, so human quality and content preservation must remain independent gates.

## Decision

Add one shadow-only rule for genuinely mono, rights-cleared/project-owned material when the artistic brief explicitly calls for invented stereo space. Require measurement consistency before spatial reward, immutable provenance, raw plus attenuation-only loudness-matched copies, downmix/content and transient diagnostics, mono-fold checks and blinded human evaluation.

Plan one A/B in which the only changed factor is the spatial reward stage: measurement-consistency-only versus the same frozen pipeline with the reward enabled. `kappa=1.0` and five reward iterations are retained only as paper provenance, not defaults. The experiment remains blocked until code, checkpoint, licence, environment, compute, reference, test material and acceptance criteria are frozen.

Do not apply Aura to existing stereo mixes, vocals by default, mastering or automatic restoration. All rules remain `auto_apply:false`.

## Consequences

- One fresh preprint Source Card, one studied Video Source Card and nine Knowledge Cards are added.
- One bounded candidate rule and one blocked one-factor A/B are added.
- Three rediscovered video IDs are removed as duplicates.
- No multitrack/reference, audio, code, model, DSP or runtime change occurs.
- The next executable experiment remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
- Citation checking remains partial; absence of a discovered correction or contradiction is not confirmation.
