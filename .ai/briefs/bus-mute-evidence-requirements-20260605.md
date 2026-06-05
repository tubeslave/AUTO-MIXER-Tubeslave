# bus-mute-evidence-requirements-20260605

## Context

Bus/group/matrix and mute promotion targets now have explicit fail-closed safety
coverage rows. Their readiness rows still lack machine-readable
`evidence_requirement_ids`, so the promotion gap rollup cannot name the proof
artifacts required for these action families.

## Goal

Add read-only evidence requirement IDs for bus/group/matrix and mute promotion
targets while preserving blocked live authority.

## Non-goals

- Do not enable bus, DCA, matrix, master, mute, or unmute writes.
- Do not mark physical proof complete.
- Do not change Product Layer supervised apply scope.

## Safety constraints

- The only supervised live target remains channel fader.
- Evidence IDs are planning metadata, not apply permission.
- Bus/mute rows remain blocked by safety, proof, read-surface, and write-surface
  gaps.

## Expected evidence

- Bus/group/matrix readiness serializes required freshness, readback, headroom,
  physical fixture, and rollback proof IDs.
- Mute readiness serializes required mute-state freshness, intent, manual
  authority, physical fixture, rollback, and emergency proof IDs.
- Gap summaries expose both action families under
  `evidence_requirement_ids_by_action_family`.
