# Drum Phase Time Alignment Rule

## Context

The project already had a general `phase.multimic_check_first` rule. New AES
and Sound On Sound sources add a more specific drum-mixing constraint: phase
and time alignment should be checked before tonal processing, but delay
alignment is not inherently correct just because waveforms line up visually.

## Options considered

1. Update the existing phase rule only.
2. Add a dedicated drum time-alignment rule in source knowledge.
3. Wire automatic channel delay changes into the mixer path.

## Decision

Add `phase.drum_time_alignment_audition` as a dedicated source-grounded rule.
Keep it in `shadow` mode and make it emit/log `phase_alignment_candidate`
actions for offline or reviewed workflows.

## Why this won

It preserves the existing broad phase rule while making the drum-specific
guidance searchable and traceable. It also keeps live behavior unchanged: delay
changes can affect monitoring, groove feel, and bleed relationships, so they
must not become automatic OSC actions at this stage.

## Rejected alternatives

Updating only the old rule would hide the new AES/SOS source lineage. Direct
automatic delay changes were rejected because the project needs bounded,
auditioned behavior before any live control path uses channel delay.

## Implementation plan

- Register AES/SOS source metadata.
- Add a one-line JSONL rule with source IDs, candidate variants, bounds, and
  safety notes.
- Add a retrieval/validation test.
- Update the research notes document.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_source_knowledge.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is low because this only changes source-knowledge retrieval and logging.
Rollback is removing the new source rows, the JSONL rule, and this ADR/brief.
