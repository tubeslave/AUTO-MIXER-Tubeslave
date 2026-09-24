# Live server/service cutover v1

Date: 2026-09-24
Status: software cutover tested; physical HIL still outstanding
Baseline: 69da8b91f5a68894d72940047c579a257e43b468

## Scope

Remove the server composition root's direct soundcheck-engine dependency. The server and soundcheck handlers must stop/status the canonical LiveSoundcheckService, not retain an engine alias or copy its physical mixer/audio handles into the server's legacy controller paths.

## Required changes

- Remove the direct AutoSoundcheckEngine import and server.auto_soundcheck_engine alias.
- Use _live_soundcheck_service for soundcheck lifecycle and status.
- Replace _sync_runtime_from_auto_soundcheck with status-only synchronization; never export mixer/audio ownership to legacy controllers.
- Expose selected_channels through the service status contract.
- Preserve websocket message names and channel observations for compatibility.
- Add source/import guards and behavioral lifecycle/status tests before declaring the cutover complete.

## Migration classification

KEEP_CORE: mixer/audio transport, discovery, routing, readback, and validated metrics/DSP primitives.
ADAPT: server composition root, soundcheck handlers, and canonical service status API.
ARCHIVE candidate: direct legacy engine import/alias and engine-to-server hardware synchronization, after references are severed and replacement evidence passes. The legacy module itself remains available only through the existing lazy compatibility adapter until its remaining references and HIL gates are closed.
DELETE_AFTER_PROOF: no module deletion in this change.

## Safety and evidence

No hardware access or WING writes are required for this software migration. It does not authorize BENCH_TEST or relax any production policy. master is not modified. Test and commit evidence will be recorded after implementation and execution; until then this document records the agreed migration, not completed work.

## Executed software evidence

UTC: 2026-09-24T17:06:50.930339+00:00

The branch-local one-shot migration applied only baseline-checked source edits. The complete focused test file list from stem_offline_test.yml, including the new server boundary tests, ran with OSC_DISABLED=true and real-model loading disabled. JUnit totals: {'tests': 185, 'failures': 0, 'errors': 0, 'skipped': 0}. The offline stem loop and report OSC-disabled assertions also passed before this evidence step. No console was contacted.

Server and handlers now retain only the service reference. Selected input IDs are copied from the explicit request (or configured input roles), and legacy observations remain status-only data. The removed synchronization cannot reassign the server's mixer, audio capture, connection mode or agent mixer. Cleanup delegates to the service before independent server-owned audio teardown and retains the service for audit. Existing unconfigured sessions remain behind the lazy legacy adapter.

ARCHIVE candidates remain candidates: no legacy module was moved or deleted and no HIL success is claimed. Full Python-matrix CI is separate from this focused evidence. The temporary migration runner/workflow are removed by the resulting source commit.
