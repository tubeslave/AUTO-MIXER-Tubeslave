# AI Decision/Correction Layer

The decision/correction layer turns critic scores into safe offline mix
variants. Critics answer "is this better?"; they do not directly change gain,
EQ, compression, pan, or FX. The new layer proposes bounded actions, renders
them in a virtual mixer, evaluates the renders, and accepts only candidates
that pass the Safety Governor.

## Roles

- Action Planner: creates conservative manual candidates such as vocal +/-0.5
  dB, low-mid cleanup, harshness reduction, bass/kick balance, light bus
  compression, and FX wet/dry probes.
- Nevergrad: optional primary optimizer. It proposes small bounded parameters
  for offline render candidates. If missing, manual candidates still run.
- Optuna: optional experimental batch optimizer. It is not required for the
  default workflow.
- pymixconsole: preferred headless console backend. It is installed from the
  upstream GitHub repository and used for gain, pan, EQ, and compression when
  available; the adapter falls back automatically if it cannot be imported.
  Default pymixconsole delay/reverb busses are disabled in offline correction
  unless FX sends are modeled explicitly.
- fallback_virtual_mixer: dependency-light WAV/FLAC/AIFF loader and summing
  mixer. It supports gain, pan, EQ, compression, gate/expander, master trim,
  and clipping prevention. FX sends remain logged as unsupported in fallback.
- dasp-pytorch: future trainable DSP option. It is optional and disabled unless
  installed manually.
- CriticBridge: calls MuQ-Eval, Audiobox, MERT, CLAP, Essentia, PANNs/BEATs or
  their existing fallbacks.
- Score Normalizer: separates real model scores from proxy/fallback scores.
  Real critics use the configured weight, proxy scores are down-weighted by
  `proxy_weight_multiplier`, and unavailable critics are excluded from the
  normalized weights.
- Safety Governor: final protection layer. It can reject the best-scoring
  candidate for clipping, headroom, phase, excessive gain/EQ/compression, vocal
  clarity drop, or bleed risk.

## Install Optional Dependencies

```bash
source .venv/bin/activate
scripts/install_ai_decision_layer.sh
```

Missing optional packages do not break the pipeline. `pymixconsole` is pinned
to its upstream source:

```bash
pip install "pymixconsole @ git+https://github.com/csteinmetz1/pymixconsole"
```

`dasp-pytorch` exists on PyPI as `dasp-pytorch==0.0.1`, but it may conflict
with local PyTorch stacks, so it is intentionally manual.

## Run Offline Correction

```bash
python -m ai_mixing_pipeline.decision_layer.offline_correction_runner \
  --input offline_test_input \
  --output offline_test_output \
  --config configs/ai_decision_layer.yaml \
  --mode offline_test \
  --optimizer nevergrad \
  --max-candidates 20
```

Output is written to:

```text
offline_test_output/<run_id>/
  renders/
    candidate_000_no_change.wav
    candidate_001_*.wav
    best_mix.wav
  reports/
    summary_report.md
    decision_log.jsonl
    critic_scores.csv
    candidate_manifest.json
    accepted_actions.json
    rejected_actions.json
    optimizer_history.json
    mixer_state_before.json
    mixer_state_after.json
```

## Reading Reports

- `summary_report.md`: dependencies, virtual mixer, optimizer, candidate counts,
  critic scores, selected candidate, and safety explanation.
- `decision_log.jsonl`: one auditable row per candidate.
- `critic_scores.csv`: compact critic breakdown with `score_source` and
  `proxy_score`.
- `model_usage_report.json`: real models used, fallback/proxy scores used,
  unavailable critics, selected effective weights, and confidence level.
- `candidate_manifest.json`: render metadata from the sandbox renderer.
- `accepted_actions.json` and `rejected_actions.json`: final action audit.

## Why No-Change Is Mandatory

The optimizer must prove improvement against a stable baseline. If every
candidate is unsafe or the best improvement is smaller than
`min_score_improvement`, the Decision Engine selects no-change and still writes
reports.

## Safety Notes

The layer is offline-only. It does not send OSC/MIDI, does not touch live mixer
state, and does not run in the real-time audio thread. Loudness matching is used
before critic evaluation so candidates are not selected merely because they are
louder.
