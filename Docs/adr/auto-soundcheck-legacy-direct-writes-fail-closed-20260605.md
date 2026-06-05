# ADR: AutoSoundcheck legacy direct writes fail closed

## 1. Context

AutoSoundcheck is a core path for the long-term autonomous mixer goal. Current live safety policy still allows only narrow supervised fader/gain writes. Static inspection found legacy AutoSoundcheck calls for EQ enable, compressor threshold, and mute restore that bypassed the proposal/evidence boundary.

## 2. Options considered

- Keep the direct writes as-is.
- Convert them to new `autofoh_safety.py` typed actions.
- Convert them to AutoSoundcheck-local blocked evidence.

## 3. Decision

Convert `set_eq_on`, `set_compressor_threshold`, and `set_mute` to AutoSoundcheck-local blocked evidence using the project-standard `proposal_only_quarantined` policy vocabulary.

## 4. Why this won

This keeps the autonomous intent visible without widening live console authority. It also preserves the exact would-be command payloads for later promotion work while avoiding a premature shared abstraction.

## 5. Rejected alternatives

Keeping direct writes leaves non-fader/gain automation live-capable without proof. Adding formal typed actions is useful later, but too broad for this patch because the immediate defect is a legacy bypass, not a missing public action contract.

## 6. Implementation plan

- Add `blocked_console_write_history` and `_record_blocked_console_write` to AutoSoundcheck.
- Replace legacy `set_eq_on`, `set_compressor_threshold`, and `set_mute` calls with blocked evidence.
- Update tests to assert no mixer mutation and exact evidence.
- Update inventory/matrix docs to reflect the closed AutoSoundcheck legacy path.

## 7. Test plan

- Focused AutoSoundcheck tests.
- Static scan for remaining direct writes.
- Registry, write-surface, and project-goal matrix tests.
- Full standard pytest.

## 8. Risks and rollback

The main risk is that a restore workflow no longer remutes channels automatically. That is intentional until mute write proof exists. Rollback is to revert this ADR and patch, but that would re-open unsupervised non-fader/gain writes.
