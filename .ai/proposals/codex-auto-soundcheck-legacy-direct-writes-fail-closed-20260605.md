# Codex Proposal: AutoSoundcheck legacy direct writes fail closed

## 1. Short thesis

AutoSoundcheck should keep producing useful correction intent, but legacy non-fader/gain console calls must become blocked evidence instead of direct mixer mutations.

## 2. What I understood

The system is moving toward a fully autonomous AI Mixing Engineer. The live boundary, however, is still intentionally narrow: channel fader/gain are the only proven live write families. AutoSoundcheck still contains older direct calls for EQ enable, compressor threshold, and mute restore.

## 3. Proposed solution

Add a small AutoSoundcheck-local blocked write recorder with the same policy vocabulary used by other quarantined surfaces:

- `blocked=True`
- `blocked_reason=auto_soundcheck_direct_apply_surface_quarantined`
- `write_policy=proposal_only_quarantined`
- `live_write_allowed=False`
- `can_apply_live=False`
- `mutates_mixer=False`

Replace `set_eq_on`, `set_compressor_threshold`, and `set_mute` calls with this recorder. Keep the values in `blocked_calls` so later promotion work has exact evidence.

## 4. Likely files to touch

- `backend/auto_soundcheck_engine.py`
- `tests/test_auto_soundcheck_engine.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`

## 5. Alternatives considered

- Add new typed actions to `autofoh_safety.py` for EQ enable, compressor threshold, and mute. This is heavier and blurs AutoFOH typed correction with legacy AutoSoundcheck restore behavior.
- Leave direct writes in place because some are fallback/restore oriented. This contradicts the current proof boundary for non-fader/gain surfaces.

## 6. Risks

- Tests that assumed local fake mixer mutation must be rewritten to check evidence.
- Blocking mute restore means restore is not complete for muted channels, but that is safer than unsupervised mute automation.
- Direct fader restore remains a separate decision; this proposal does not settle broader autonomous fader promotion.

## 7. Test plan

- Run py_compile on touched Python files.
- Run focused AutoSoundcheck tests.
- Run registry/project matrix/static scan tests.
- Run `git diff --check`.
- Run `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`.

## 8. Where the other agent may disagree

Kimi may argue for adding formal typed actions for these calls immediately. I would keep this patch smaller: record exact blocked evidence first, then decide whether a common typed abstraction is warranted after all remaining legacy writes are inventoried.
