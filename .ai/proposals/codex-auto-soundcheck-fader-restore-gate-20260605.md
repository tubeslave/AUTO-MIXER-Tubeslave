# Codex Proposal: AutoSoundcheck fader restore gate

## 1. Short thesis

AutoSoundcheck restore fader writes should be blocked evidence, not an alternate live executor.

## 2. What I understood

The project target is autonomous mixing, but current proof only covers narrow supervised fader/gain through Product Layer. A direct `set_fader` in AutoSoundcheck restore bypasses that proof path even though the target surface is allowed elsewhere.

## 3. Proposed solution

Replace the direct restore `set_fader` call with `_record_blocked_console_write("set_fader", ...)`. Include `manual_gate="apply_operator_proposal"` and the target `value_db` in the blocked call payload.

## 4. Likely files to touch

- `backend/auto_soundcheck_engine.py`
- `tests/test_auto_soundcheck_engine.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- focused registry/matrix tests

## 5. Alternatives considered

- Route through `AutoFOHSafetyController` in `ROLLBACK` state. This still writes directly to the mixer and bypasses Product Layer approval.
- Keep direct restore as an emergency fallback. That keeps a hidden live executor and contradicts the current proof boundary.

## 6. Risks

Restore no longer performs an immediate fader mutation. The safety tradeoff is intentional: the system should preserve exact rollback intent but require supervised application until the restore path has its own proof.

## 7. Test plan

- Focused AutoSoundcheck restore/static tests.
- Registry/matrix/static scan tests.
- `git diff --check`.
- Full standard pytest.

## 8. Where the other agent may disagree

Kimi may argue that restore should remain direct for emergency recovery. I would require a separate emergency restore ADR with physical proof, readback, cooldown bypass rules, and explicit operator activation before allowing that.
