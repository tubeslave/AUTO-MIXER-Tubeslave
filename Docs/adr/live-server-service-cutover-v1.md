# Live server/service cutover v1

Date: 2026-09-24
Status: software cutover implemented, published and focused-tested; physical HIL outstanding
Baseline: 69da8b91f5a68894d72940047c579a257e43b468
Implementation commit: bc2b1b8cdbccb5fdeee8a1adcb44b25be34aa410
Working branch: live-soundcheck-renovation-run11
Integration target: live-soundcheck-renovation, not master

## Decision and implemented scope

The server and soundcheck handlers manage soundcheck lifecycle/status through `LiveSoundcheckService`. They no longer import, retain, stop or inspect `AutoSoundcheckEngine`, and no longer copy its physical mixer/audio handles into independent legacy controller paths.

- The server retains `_live_soundcheck_service`, not `auto_soundcheck_engine`.
- `_sync_runtime_from_live_soundcheck()` refreshes display flags and returns status data only. It does not assign the server mixer, capture, connection mode or agent mixer.
- Cleanup delegates to the service before independent server-owned audio/mixer teardown, retaining the service reference for audit/status even after an error.
- `selected_channels` is copied from the explicit request. When a configured request selects all inputs, the role map supplies input IDs; reserved Main-return channels are not inferred as inputs.
- Existing WebSocket message names and compatibility channel observations remain available. Handlers do not retain the session/engine handle returned by `start()`.
- AST/import guards reject reintroduction of direct soundcheck-engine ownership in the server and handler.

This is a bounded soundcheck composition-root cutover. It does not make the entire server legacy-free: independent AutoEQ, AutoFader and AutoCompressor imports and controllers remain outside this change. Unconfigured compatibility sessions still use the existing lazy legacy adapter.

## Migration classification

KEEP_CORE: mixer/audio transport, discovery, routing, readback, and validated metrics/DSP primitives; unchanged here.

ADAPT: server composition root, soundcheck handlers, and canonical service status API.

ARCHIVE candidate: the legacy soundcheck orchestration module and its lazy compatibility adapter after all remaining references and HIL gates are closed. The obsolete direct server import/alias and hardware-copying synchronization have been replaced and covered by software tests; no legacy module has been moved or deleted.

DELETE_AFTER_PROOF: no legacy-module deletion. The temporary one-shot migration script/workflow were removed after their purpose was fulfilled; they are not runtime infrastructure.

## Executed software evidence

Initial proof UTC: 2026-09-24T17:06:50.930339+00:00.

Run `36031991398`, job `107742744774`, applied baseline-blob-checked edits and executed the complete focused file list with `OSC_DISABLED=true` and real-model loading disabled. JUnit: **185 tests, 0 failures, 0 errors, 0 skipped**. This comprises 183 live/runtime/handler tests and 2 stem tests. The offline mock stem loop and OSC-disabled report assertions also passed.

New coverage includes status-only ownership isolation across modes, missing/inactive service, channel-selection/metric forwarding, service-first cleanup including failure, detached status selections, configured role selection and source guards. Existing service, handler, feature, control/readback and rollback tests remain in the focused suite.

Artifact `10823215856` (`live-server-cutover-evidence`) retains the source patches, JUnit XML and pytest log for the initial run's retention period.

## Publication failure and resolution

The initial workflow's final push failed, although all test/evidence steps passed. Its restricted `GITHUB_TOKEN` was rejected by repository rules: protected-ref updates, PR/signature/code-scanning requirements and missing workflow-write permission. That workflow is therefore **failure**, not success.

The tested commit object was readable from GitHub. The user-authorized connector then published that exact commit to the working branch with a non-forced fast-forward. Repository rules were not edited, disabled or weakened; master was not modified.

A fresh normal **Stem Offline Test** run `36032335830` on published commit `bc2b1b8cdbccb5fdeee8a1adcb44b25be34aa410` completed **success**, independently confirming the published tree.

Commit `633f5884c000cc580333db9add0353254d2022e9` additionally enables the existing full Python 3.10/3.11/3.12 workflow on renovation work-branch pushes and PRs targeting renovation. Full-matrix success is not claimed here; it must be read from the corresponding run.

## Safety and remaining gate

No hardware was contacted, no WING mutation was performed and BENCH_TEST was not selected. Mode, snapshot/readback, freeze and production policies were not relaxed. No new decision functionality was added to legacy `auto_*` modules.

Integration into the renovation branch remains separate from publishing this work branch. Physical WING HIL and severing remaining compatibility/controller references remain prerequisites for archiving their legacy modules. The next live goal is an evidence-backed audit of those remaining callers, not another parallel decision authority.
