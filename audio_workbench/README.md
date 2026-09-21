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
