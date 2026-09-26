# Audio Workbench MCP v0.1

Stateful, auditable audio-analysis MCP for Automixer. It is deliberately not a "magic mix score".

## What v0.1 does

- fingerprints every render with SHA-256, so measurements cannot silently belong to an old file;
- registers mandatory verification domains;
- performs deterministic signal analysis: sample peak, RMS, crest, clipping count, DC, broad-band energy, stereo correlation and Side/Mid ratio;
- can invoke FFmpeg EBU R128 analysis;
- marks dependent checks stale after gain/EQ/compression/stereo/reverb/edit/routing/master changes;
- blocks a render from being called final while mandatory checks are missing or stale;
- persists hypothesis -> action -> outcome in SQLite;
- exposes the workflow over MCP stdio.

## Install

From repository root:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install "mcp[cli]>=1.2"

FFmpeg must be available on PATH.

## Run

    python -m audio_workbench

MCP client config example:

    {
      "mcpServers": {
        "audio-workbench": {
          "command": "/ABSOLUTE/PATH/.venv/bin/python",
          "args": ["-m", "audio_workbench"],
          "cwd": "/ABSOLUTE/PATH/AUTO-MIXER-Tubeslave"
        }
      }
    }

## Tool contract

`inspect_audio` registers a render and returns its immutable identity plus signal measurements.

`analyze_audio` analyzes a file but never declares a mix artistically good.

`record_check` stores a check against one exact SHA-256 render.

`invalidate_after_change` makes affected checks stale. This is the anti-forgetting mechanism.

`get_coverage` exposes missing/stale checks and whether finalization is allowed.

`get_next_task` forces the agent back to the next unverified domain.

`record_decision` stores hypotheses, actions and outcomes for future preference learning.

## Mandatory domains

integrity, loudness, dynamics, spectrum, stereo_phase, transients, song_context, delivery.

The first implementation intentionally separates measurements from judgments. A future learned listener may add evidence, but it must not overwrite deterministic checks or silently convert an embedding score into "quality".

## Next milestones

1. section-aware timeline and activity masks;
2. pairwise masking graph between tracks/stems;
3. A/B loudness-matched comparison with invariant ordering tests;
4. render/rollback adapter for DAW;
5. calibrated audio-language observer;
6. learned preference comparator trained on accepted/rejected A/B decisions.


## DAWless / chat-first mode

Audio Workbench can work from a plain folder of aligned stems or renders without Ableton/Cubase.

New tools:
- `create_dawless_project(project_root, audio_dir, title)` scans supported audio files, records sample rate/channel/duration metadata, guesses broad musical roles from filenames, and writes `audio_workbench_project.json`.
- `get_project_context(project_root)` returns the persistent song manifest.
- `set_project_context(project_root, sections, references, notes)` stores song sections, reference files and mix-intent notes.

This is the preferred first workflow for chat-driven offline mixing. Ableton is treated as an optional execution backend for plugin automation, routing and session-native rendering, not as a requirement for analysis.


## Experiment engine

v0.1 now includes a reversible candidate renderer for explicit hypotheses. Every experiment automatically includes a bypass candidate, copies the baseline, renders candidates into an experiment directory, registers immutable render hashes, computes signal deltas, and waits for a separate evaluation step. Source audio is never overwritten.

Initial bounded processors: gain, one peaking-EQ biquad and feed-forward envelope compression. These are controlled probes, not claims that this small DSP set is sufficient for production mixing.

Additional diagnostics:
- generic transient-event analysis;
- role-priority priors with explicit project-context overrides;
- section analysis, suspicious-window scan and pairwise overlap graph;
- deterministic blind A/B labels.

The architecture deliberately separates detection, hypothesis, intervention, verification and selection.
