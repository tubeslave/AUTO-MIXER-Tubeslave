# Scene / Preset Management Proposal Brief

Task id: scene-preset-management-proposal-20260604
Date: 2026-06-04

## Context

The project goal requires control of scenes and user presets, but direct scene,
snapshot, and preset operations can overwrite broad console state. Existing live
authority remains limited to supervised fader/gain through the Product Layer.

## Objective

Add a transport-free scene/preset proposal layer that can analyze scoped
scene/preset intent and emit operator-review evidence without loading scenes,
saving presets, mutating queues, or widening live console writes.

## Safety Boundary

- No live scene, snapshot, preset, routing, OSC, MIDI, or mixer-client calls.
- No `requested_change` payloads for scene/preset actions.
- No queue import or live apply until a separate promotion proves full state
  readback, scoped diffs, rollback, scope limits, and hardware behavior.
- Keep direct scene/snapshot/preset writes quarantined.

## Expected Files

- `backend/scene_preset_management_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_scene_preset_management_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/scene-preset-management-proposal-20260604.md`

## Test Plan

- Compile touched Python modules.
- Run focused scene/preset proposal tests.
- Run processing registry, proposal adapter, and project-goal matrix tests.
- Check that the new module is transport-free and emits no live apply payload.
