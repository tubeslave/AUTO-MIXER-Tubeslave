# autonomy-mode-contract-20260603

## 1. Context

The project goal defines four target modes: Analysis Mode, Assisted Mixing
Mode, Supervised AutoMix Mode, and Full Autonomous Mixing Mode. The product
runtime already uses `analyze`, `assist`, `supervised`, and `auto`, but the
status payload did not explicitly identify the target architecture mode or
separate long-term target write surfaces from current proven live write scope.

## 2. Options considered

- Keep the existing status unchanged and document the mapping only in prose.
- Rename product keys to the long target mode names.
- Extend the existing policy contract with target mode metadata and aliases.

## 3. Decision

Extend `backend/operator_mode_policy.py` so every operator mode status includes
the target project mode number/name, autonomy tier, human gate, proposal policy,
target live write surfaces, and current live write kinds.

## 4. Why this won

It moves the architecture toward the requested autonomous system without
breaking existing frontend keys or widening live writes. The backend can now
accept goal-language aliases while still returning the stable UI mode keys.

## 5. Rejected alternatives

Renaming mode keys would create frontend/API churn without adding safety.
Unlocking broad autonomous live writes would skip required WING proof,
readback, rollback, emergency-stop, and safety coverage.

## 6. Implementation plan

1. Add four-mode metadata and aliases to `operator_mode_policy.py`.
2. Keep current allowed live write kinds limited to `fader` and `gain`.
3. Keep full autonomous apply locked on WING deployment even when the runtime
   unlock flag is set.
4. Add focused tests for mode options, aliases, supervised scope, and auto lock.

## 7. Test plan

- `PYTHONPATH=backend python -m pytest tests/test_operator_mode_policy.py -q`
- Relevant existing product-state and offline-validation tests when the current
  dirty worktree is ready for broader validation.

## 8. Risks and rollback

The primary risk is confusing target surfaces with current allowed writes. The
status payload keeps these fields separate. Rollback is a clean revert of the
policy metadata and the focused test file; no mixer transport or live write path
is touched.
