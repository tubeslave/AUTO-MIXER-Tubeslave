# Codex Proposal: autonomy-mode-contract-20260603

## 1. Short thesis

Make the operator mode policy a stable contract for the project's four target
autonomy modes while preserving the current safe execution boundary.

## 2. What I understood

The long-term target is a professional AI mixing engineer with four modes from
read-only analysis through full autonomous console control. The current repo is
not approved for broad autonomous live writes; WING deployment must remain
fail-closed except for the existing supervised fader/gain gate.

## 3. Proposed solution

Extend `backend/operator_mode_policy.py` with mode aliases, autonomy tier
metadata, target architecture names, and separate target-vs-current write
scope. `auto` remains the stable UI key, but goal-language aliases such as
`Full Autonomous Mixing Mode` normalize to it. Current allowed live write kinds
remain `fader` and `gain` only where the existing gate already allows them.

## 4. Likely files to touch

- `backend/operator_mode_policy.py`
- `tests/test_operator_mode_policy.py`
- `.ai/briefs/autonomy-mode-contract-20260603.md`
- `Docs/adr/autonomy-mode-contract-20260603.md`

## 5. Alternatives considered

- Rename `auto` to `full_autonomous`: rejected because it would churn the UI and
  existing websocket contract.
- Implement broad autonomous writes now: rejected because there is no hardware
  proof, readback, rollback, and safety coverage for those surfaces.

## 6. Risks

- UI code may ignore new fields. This is acceptable because existing fields stay
  unchanged.
- Operators may misread target surfaces as currently allowed. Mitigate by
  exposing `current_live_write_kinds` separately from `target_live_write_surfaces`.

## 7. Test plan

- Run `PYTHONPATH=backend python -m pytest tests/test_operator_mode_policy.py -q`.
- Run relevant existing policy tests if the dirty worktree allows it.

## 8. Where the other agent may disagree

Kimi may argue that mode metadata should live in a separate product schema
module. The smaller change is preferable here because all status construction
already passes through `operator_mode_policy.py`.
