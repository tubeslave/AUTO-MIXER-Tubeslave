# Mix Agent Facade

`mix_agent` is a research-engineering facade for automatic mixing decisions. It
does not replace a mix engineer. It analyzes stems, a stereo mix and an optional
reference, then produces explainable recommendations and conservative reversible
offline renders.

## Philosophy

One metric does not define a good mix. LUFS, true peak, spectral centroid,
stereo width, masking proxies and reference distance are useful only as a
dashboard. The system reports independent components and explains why an action
is suggested, what evidence supports it, what risk it carries and why a
listening check is still required.

Every change must serve one of these goals:

- improve readability;
- reduce masking;
- support groove or transient intent;
- improve translation;
- fix a technical defect;
- move closer to genre/reference context without copying it.

If a metric improves but subjective clarity, emotional function or musical
intent worsens, the change should not be applied automatically.

## CLI

```bash
python -m mix_agent analyze \
  --stems ./stems \
  --mix ./mix.wav \
  --reference ./reference.wav \
  --genre pop \
  --out ./report.md

python -m mix_agent suggest \
  --stems ./stems \
  --genre hip-hop \
  --out ./suggestions.json

python -m mix_agent apply \
  --stems ./stems \
  --suggestions ./suggestions.json \
  --out ./renders/conservative_mix.wav \
  --report ./renders/conservative_mix.md
```

If no stereo mix is supplied, the analyzer creates an analysis-only sum from
the stems and marks that as a limitation. If no reference is supplied, the
agent uses the genre profile and generic engineering rules.

## How To Read The Report

- `Quality Dashboard` shows independent scores. Do not treat the aggregate as
  proof that the mix is good.
- `Detected Issues` uses `Issue`, `Evidence`, `Suggested action`, `Risk` and
  `Confidence` sections so the recommendation can be audited.
- `Reference Comparison` is loudness-normalized and should be read as a
  tolerance guide, not an instruction to imitate a protected recording.
- `Limitations` calls out approximations such as synthesized mix context,
  unavailable reference, mono-only input or fallback loudness/true-peak methods.

## Offline Apply

The conservative offline renderer supports only simple reversible operations:

- gain cuts;
- high-pass filter;
- small parametric EQ moves;
- pan adjustment.

Dynamic EQ, sidechain, compression, stereo-width, saturation, reverb and delay
recommendations remain placeholders unless a concrete processor chain is wired
in. The original stems are never overwritten.

## Backend / Real Console

`mix_agent.backend_bridge.MixAgentBackendBridge` translates supported
recommendations into existing AutoFOH typed actions and delegates application to
`AutoFOHSafetyController`. This preserves runtime policy, fader ceilings, rate
limits and mixer-specific bounds. Unsupported or ambiguous actions are returned
as blocked/advisory items instead of being sent to the console.

## Genre Profiles

Profiles live in `mix_agent/config/genre_profiles.yaml`. They are priors and
tolerances, not laws. A profile can suggest expected loudness, dynamics,
low-end behavior, vocal prominence, stereo width, transient priority and
reference tolerance.

## Ethical And Copyright Constraints

Reference matching is used for broad engineering comparison only. The system
must not download, reproduce or imitate protected source material or trademarked
engineer styles.

## Final Evaluation

Subjective listening remains mandatory. Use loudness-matched A/B checks, mono
checks, small-speaker checks and, for competing versions, a MUSHRA-like or
pairwise preference protocol before treating a mix as approved.
