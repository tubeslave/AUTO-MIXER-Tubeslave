# TUB-365 WING Replay Readiness Bundle

## Scope completed in this change

- Added `backend/wing_replay_readiness_bundle.py`.
- The bundle merges three existing evidence streams:
  - shadow-mode replay reports from `ai_mixing_pipeline/reports/shadow_report_writer.py`
  - WING telemetry summaries/diffs from `backend/wing_telemetry_compare.py`
  - supervised live runner summaries from `backend/tub345_supervised_write_runner.py`
- The bundle emits:
  - `wing_replay_readiness_bundle.json`
  - `wing_replay_readiness_bundle.md`

## Safety behavior

- Fails closed if a shadow report is not dry-run only.
- Fails closed if a shadow report indicates live mixer writes.
- Fails closed if shadow safety evidence is missing from both the comparison
  report and the promotion-readiness report.
- Fails closed if observation-only telemetry contains any `write_sent` event.
- Fails closed if the supervised summary is missing telemetry artifact paths.
- Preserves the exact blocker list in the bundle output for review.
- Emits an explicit `no_clipping` evidence section even when measured proof is missing, so downstream trust gates can distinguish `missing`, `measured+clear`, and `measured+not_clear`.

## Regression proof added

- New test file: `tests/test_wing_replay_readiness_bundle.py`
- Coverage:
  - safe path: shadow-ready bundle + observation-only telemetry with blocked writes only
  - unsafe path: observation-only telemetry containing `write_sent` must force `ready=false`
  - unsafe path: missing shadow safety evidence must force `ready=false`
  - explicit no-clipping evidence path: supervised summary evidence is copied into the bundle and markdown output

## Targeted verification

- `python3 -m pytest -q tests/test_wing_replay_readiness_bundle.py tests/test_wing_telemetry_compare.py tests/test_wing_telemetry_recorder.py tests/test_shadow_mode_report_writer.py`
- Result: `7 passed`

## Usage

```bash
python3 backend/wing_replay_readiness_bundle.py \
  --output-dir artifacts/wing_replay_readiness_bundle/latest \
  --shadow-report-dir <shadow-report-dir> \
  --live-telemetry-session <live-session-dir-or-events.jsonl> \
  --replay-telemetry-session <replay-session-dir-or-events.jsonl> \
  --supervised-summary-path <supervised-summary.json>
```

## Remaining input needed for a real bundle artifact

- a checked-in or freshly produced `shadow_mode_summary.json` report directory
- the selected live/replay telemetry session pair to compare
- the supervised live summary JSON to stamp into the bundle
