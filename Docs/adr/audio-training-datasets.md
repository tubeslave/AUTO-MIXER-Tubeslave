# Task audio-training-datasets

## Context

The research report in `/Users/dmitrijvolkov/Downloads/deep-research-report.md` concludes that public action logs for professional DAW mixing are scarce. The strongest immediately usable public sources are compact metadata/dialogue/parameter datasets such as MixAssist, MixParams, MixologyDB, and Hugging Face mirrors around mix-evaluation data. Large Cambridge/SOS REAPER projects are useful later, but they are hundreds of MB and license-limited.

The repository already contains a standalone `trainer_agent`, but it was only a command runner and was not connected to a dataset-discovery entrypoint.

## Options considered

1. Download all candidate datasets from the report immediately.
2. Wire the trainer agent into the existing backend `AgentTrainingService`.
3. Add a small dataset-discovery bootstrap script and let `trainer_agent` invoke it.

## Decision

Use option 3. Add a dedicated bootstrap script for Hugging Face dataset discovery/download and expose it through `TrainerAgent` as `discover_datasets`.

## Why this won

This keeps the first step safe, testable, and reversible. It downloads only compact public dataset files and records a manifest, without touching live-sound runtime behavior or starting model training on partially understood schemas.

## Rejected alternatives

Downloading Cambridge/SOS REAPER archives now was rejected because those archives are large and their terms need explicit operator review before bulk ingestion.

Directly running backend training was rejected because MixAssist and MixParams need schema adaptation before they can become supervised targets for the existing `train_gain_predictor` or `train_mix_console` functions.

## Implementation plan

- Add `scripts/discover_training_datasets.py`.
- Extend `src/agent_ops/trainer_agent.py` with a `discover_datasets` task.
- Extend `scripts/run_agents.py` with a `--task discover_datasets` mode.
- Ignore local downloaded dataset payloads under `models/training_datasets/`.
- Run the trainer agent to download compact initial datasets and write the manifest.

## Test plan

- Unit-test Hugging Face file selection and safe path handling.
- Unit-test `TrainerAgent` command construction for dataset discovery.
- Run focused tests first, then the standard repository test command when feasible.

## Risks and rollback

The first risk is schema mismatch: downloaded Parquet/JSON data is useful for future ingestion, but not yet a direct training input for existing model trainers. Roll back by deleting the manifest and downloaded local files.

The second risk is licensing ambiguity for audio-bearing corpora. The bootstrap avoids bulk audio archives and records source URLs so a human can review terms before expanding the corpus.
