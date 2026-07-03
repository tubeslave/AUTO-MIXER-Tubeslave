# DSP Dry-Run Report: EQ Decision Logic

- Created: 2026-05-08T14:48:03Z
- Repository: `/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main`
- Branch observed: `feature/mcp-director`
- Task id: `task_20260508T143224Z_d3a621be`
- Mode: dry-run repository analysis
- Source edits: none

## Executive Summary

The repository has three separate EQ decision surfaces:

1. `automixer/production_mix_v1`: offline corrective EQ and the newer explainable AutoEQ engine.
2. `backend`: live/server AutoEQ, soundcheck EQ, feedback notches, and WING client writes.
3. `ai_mixing_pipeline`: offline research candidate planning and safety scoring.

The strongest EQ decision logic is currently in `automixer/production_mix_v1/autoeq.py` and `automixer/production_mix_v1/ayaic_offline_pipeline.py`. It is explainable, bounded, test-covered, and mostly conservative when run offline. The live/backend path is weaker: some entrypoints are broken or bypass the newer live-apply safety gate, and the WING EQ OSC address model is inconsistent across code/docs/tests.

## Findings

### 1. High: multi-channel live AutoEQ start path is broken

`backend/server.py` passes `mode=mode` into `MultiChannelAutoEQController.start_multi_channel(...)`:

- `backend/server.py:2523-2530`

But the controller signature does not accept `mode`:

- `backend/auto_eq.py:2734-2739`

Dry-run introspection confirmed:

```text
(self, device_id: int, channels_config: List[Dict], on_spectrum_callback: Callable = None, on_corrections_callback: Callable = None, on_status_callback: Callable = None) -> bool
```

Impact: `start_multi_channel_auto_eq` fails before EQ analysis starts. This affects batch/soundcheck AutoEQ decision generation.

### 2. High: live EQ writes bypass the live-apply/readback safety gate

`LiveApplyService` only supports `set_fader` and `set_gain`:

- `backend/live_apply.py:15`
- `backend/live_apply.py:93-129`

EQ writes still go directly to the mixer client:

- Single-channel AutoEQ apply: `backend/auto_eq.py:2484-2518`
- Multi-channel AutoEQ apply: `backend/auto_eq.py:3012-3045`
- Soundcheck adaptive EQ: `backend/auto_soundcheck_engine.py:1086-1091`
- Feedback notch: `backend/auto_soundcheck_engine.py:1692-1696`
- ODA EQ agent: `backend/agents/eq_agent.py:88-99`

Impact: EQ live writes are not consistently covered by default dry-run, explicit confirmation, readback verification, or panic-stop policy. This is the main live-safety gap.

### 3. High: WING EQ OSC address model is inconsistent

The docs list slash-separated EQ band parameters:

- `Docs/TOOLS.md:32-35`: `/ch/{N}/eq/{B}/f`, `/g`, `/q`

Some agent code also reads/writes slash-separated state keys:

- `backend/agents/eq_agent.py:52-57`
- `backend/agents/eq_agent.py:94-98`

But `WingClient` and `OscManager` write compact paths like `/ch/{channel}/eq/{band}g`:

- `backend/wing_client.py:807-812`
- `backend/osc_manager.py:166-174`

Impact: depending on the active WING transport/address map, EQ commands may target the wrong node or fail silently. Any live EQ work needs readback-confirmed address tests before enabling writes.

### 4. Medium: backend AutoEQ can apply broad target-curve changes without headroom validation

`backend/auto_eq.py` calculates corrections by normalizing the current spectrum, comparing it to a target curve, and allowing up to +/-12 dB before WING-range clamps:

- `backend/auto_eq.py:1758-1762`
- `backend/auto_eq.py:1818-1855`

It then assigns the first correction to low shelf and last correction to high shelf regardless of whether the detected move was a narrow peak/resonance:

- `backend/auto_eq.py:1883-1897`
- `backend/auto_eq.py:1947-1960`

There is no true-peak/LUFS validation before live `apply_to_mixer()`:

- `backend/auto_eq.py:2460-2528`

Impact: this path can make more aggressive and less localized changes than the newer production AutoEQ safety model. For live use, it should be treated as suggest/preview-only until guarded by safety and readback.

### 5. Medium: `contextual_deep_eq` is now the offline default and intentionally allows large moves

The offline Ayaic path defaults to `contextual_deep_eq`:

- `automixer/production_mix_v1/ayaic_offline_pipeline.py:70`

That method can allow deep cuts and boosts, with role limits up to large positive boosts:

- `automixer/production_mix_v1/ayaic_offline_pipeline.py:2337-2349`
- `automixer/production_mix_v1/ayaic_offline_pipeline.py:2514-2531`
- `automixer/production_mix_v1/ayaic_offline_pipeline.py:2645-2654`

It logs critic notes and regret scores, but it does not hard-block all headroom-risk boosts. The master stage also defaults to no peak ceiling:

- `automixer/production_mix_v1/ayaic_offline_pipeline.py:67-68`
- `automixer/production_mix_v1/ayaic_offline_pipeline.py:5880-5899`

Impact: this is acceptable for offline experiments, but final render review should include true-peak checks, especially when `master_ceiling_dbfs` remains `None`.

### 6. Medium: `--autoeq-osc-apply` only changes report command mode in production AutoEQ

The production AutoEQ command adapter builds abstract commands and does not send them:

- `automixer/production_mix_v1/autoeq.py:850-856`
- `automixer/production_mix_v1/autoeq.py:914-956`

The CLI maps `--autoeq-osc-apply` into `osc_dry_run=false` and `osc_apply=true`:

- `automixer/production_mix_v1/__main__.py:74-80`

Impact: the flag name can be misread as live application. In this code path it exports apply-mode commands in the report; it does not itself perform WING writes.

### 7. Medium: CLI import path is fragile without `backend` on `PYTHONPATH`

Running the documented module entrypoint from repo root failed:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m automixer.production_mix_v1 --help
ModuleNotFoundError: No module named 'autogain_contract'
```

The failure comes through:

- `automixer/production_mix_v1/arrangement_input_levels.py:18-35`
- `backend/arrangement_automation/live_input_trim_controller.py:19-23`

With `PYTHONPATH=/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/backend`, the same `--help` command passed. Tests mask this because `tests/conftest.py:7-8` inserts `backend` onto `sys.path`.

Impact: dry-run users may not be able to invoke the offline EQ CLI unless the environment is prepared.

## Positive Observations

- `automixer/production_mix_v1/autoeq.py` has a real decision pipeline: spectral analysis, masking analysis, static/dynamic/group/master decisions, safety limiting, validation/rollback, report output, and dry-run command generation.
- The production AutoEQ safety limiter rejects low-confidence decisions, clamps live mode to smaller limits, blocks boosts near peak limit, caps filter count, and limits Q.
- `project_corrective_hybrid` is conservative: cuts only, max 4 filters per channel, dynamic EQ report-only, and downscaled frequency/bleed evidence.
- Existing tests cover the core offline EQ behavior and safety limiter.

## Verification Run

Passed:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q \
  tests/test_autoeq_engine.py \
  tests/test_ayaic_offline_pipeline.py::test_corrective_eq_methods_are_ported_from_offline_pipeline \
  tests/test_ayaic_offline_pipeline.py::test_contextual_deep_eq_allows_large_moves_with_critic_notes \
  tests/test_ayaic_offline_pipeline.py::test_project_corrective_eq_uses_primary_role_and_group_diagnostics \
  tests/test_ayaic_offline_pipeline.py::test_project_corrective_hybrid_merges_evidence_with_safety_limits \
  -p no:cacheprovider

13 passed in 1.39s
```

CLI check:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m automixer.production_mix_v1 --help
failed: ModuleNotFoundError: No module named 'autogain_contract'

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main/backend \
  python3 -m automixer.production_mix_v1 --help
passed
```

## Suggested Tests

1. Add a server contract test for `start_multi_channel_auto_eq` proving it does not pass unsupported kwargs to `MultiChannelAutoEQController.start_multi_channel`.
2. Add EQ live-apply tests mirroring `tests/test_live_apply_contract.py`, covering `set_eq_band`, low/high shelf, dry-run default, confirmation requirement, readback, and panic stop.
3. Add WING EQ address contract tests that assert the canonical address shape and verify readback after a dry-run/manual console check.
4. Add backend AutoEQ tests for headroom-sensitive boosts and for first/last correction mapping so narrow resonance cuts do not become shelves by accident.
5. Add a CLI smoke test for `python -m automixer.production_mix_v1 --help` from repo root without relying on pytest path injection.
6. Add final-render true-peak assertions for `contextual_deep_eq` with no master ceiling and with a configured ceiling.

## Safe Branch Plan

Use a separate branch after manual review, for example:

```bash
git switch -c fix/eq-decision-live-safety
```

Recommended implementation order:

1. Fix the `mode` kwarg mismatch in multi-channel AutoEQ and add the server contract test.
2. Choose and document one canonical WING EQ address format, then verify it against readback before broader use.
3. Extend `LiveApplyService` or add an equivalent `EQApplyService` for EQ writes.
4. Keep backend AutoEQ in suggest/confirm mode until EQ write safety has readback coverage.
5. Add true-peak/ceiling checks around offline deep-EQ renders before treating them as approved deliverables.

## Dry-Run Limitations

- No source files were edited.
- No console/WING hardware writes were attempted.
- No full offline render was run; this was a repository code-path and focused-test analysis.
