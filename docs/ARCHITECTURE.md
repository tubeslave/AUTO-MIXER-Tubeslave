# Архитектура AUTO-MIXER-Tubeslave

## Статус документа

Этот документ описывает текущее архитектурное состояние репозитория и должен читаться вместе с:

- `docs/runtime_ownership_matrix.md`
- `docs/current_architecture_map.md`
- `docs/console_write_safety_policy.md`

Если между этим документом и кодом есть расхождение, источником истины для runtime ownership и console-write semantics считаются документы выше.

## Краткий обзор

- Canonical live coordinator: `backend/server.py`
- Canonical live input trim path: `backend/server.py -> backend/arrangement_automation/live_input_trim_controller.py`
- `automixer/production_mix_v1` — offline-first pipeline, не live runtime
- `backend/auto_soundcheck_engine.py` — active but overlapping path, не canonical live runtime
- Panic stop is not system-wide

Этот репозиторий больше не следует описывать как один единый canonical runtime. В нём есть несколько пересекающихся live, offline, research и legacy surfaces.

## Live / Offline Split

### Active Live Runtime

- `frontend/src/services/websocket.js`: основной UI transport к backend runtime
- `backend/server.py`: основной live runtime coordinator
- `backend/handlers/`: WebSocket handler surface
- `backend/audio_capture.py`: shared live audio input service
- `backend/live_apply.py`: текущий gated write service для approved manual scope
- `backend/arrangement_automation/live_input_trim_controller.py`: canonical live input trim path
- `backend/wing_client.py` и `backend/osc/enhanced_osc_client.py`: live console transport layer

### Active Offline Runtime

- `automixer/production_mix_v1/__main__.py`
- `automixer/production_mix_v1/pipeline.py`
- `automixer/production_mix_v1/osc_queue.py`

Этот путь считается offline-first и не должен описываться как live runtime.

### Active But Overlapping

- `start_soundcheck.py -> backend/auto_soundcheck_engine.py`
- `backend/auto_fader.py`
- `backend/arrangement_automation/controller.py`

Эти поверхности активны в коде, но не являются canonical live runtime и не должны автоматически считаться approved live-control paths.

### Legacy / Unclear

- snapshot/load/reset utility scripts
- `backend/handlers/snapshot_handlers.py` direct utility surface
- `backend/auto_fader_v2/` ownership remains unclear
- `mix_agent/` remains unclear/incomplete in the current workspace snapshot

### Research / Experimental

- `ai_mixing_pipeline/`
- `backend/ai/`
- `backend/ml/`
- parts of `backend/auto_fader_v2/`

Research or experimental presence в репозитории не означает approval for live console control.

## Canonical Live Runtime

### Main Coordinator

`backend/server.py` является canonical live coordinator.

Он отвечает за:

- создание mixer client;
- регистрацию handler surface;
- lifecycles controller-ов;
- audio capture lifecycle;
- voice and automation entrypoints;
- integration surface между frontend UI и live console control.

### Canonical Live Input Trim

Canonical live input trim path:

`backend/server.py -> backend/arrangement_automation/live_input_trim_controller.py`

Этот path считается canonical только для current live input trim ownership. Это не означает, что весь live write surface вокруг него автоматически approved.

## Console-Write Boundaries

### Approved Gated Path

Текущий approved gated path узкий:

- manual `set_fader`
- manual `set_gain`
- only through `LiveApplyService`

Этот approved scope не должен описываться шире без отдельного approval.

### Blocked-At-Gate Paths

- manual `set_eq`
- manual `set_compressor`

Эти операции имеют handler surface, но не должны описываться как approved real-write paths.

### Direct Bypass Paths

- `backend/auto_fader.py`
- `backend/arrangement_automation/controller.py`
- write-capable sections of `backend/auto_soundcheck_engine.py`
- snapshot/load/save utility and script surfaces

Direct bypass path не должен описываться как gated только потому, что он использует существующий mixer client или соседствует с approved runtime path.

### Emergency Exception Candidate

- `backend/feedback_detector.py`
- `backend/auto_soundcheck_engine.py::_handle_feedback_event`

Feedback detector path должен описываться только как safety-critical emergency exception candidate. Это не approved exception.

### Operator-Only Legacy Tools

- `backend/restore_channels.py`
- `backend/load_snap.py`
- `backend/load_snap_v2.py`
- `backend/load_snap_final.py`
- `backend/find_and_load_snap.py`
- `backend/scan_and_load_snap.py`
- `backend/reset_all_channels.py`
- `backend/reset_modules_trim_faders.py`
- `backend/reset_mixer.sh`

Эти paths следует считать operator-only legacy tools, а не частью approved live runtime.

## Safety Semantics

### Panic Stop

- Panic stop is not system-wide
- Current documented panic-stop semantics cover only manual `LiveApplyService` paths

Нельзя описывать текущую систему так, будто все live writes already protected panic-stop logic.

### Feedback Detector

- feedback detector is a safety-critical emergency exception candidate
- it is not an approved exception
- it requires separate policy review before any live-scope expansion

### Offline Sender Attachment

- offline real sender attachment is not approved without unified console-write gate and separate approval
- `automixer/production_mix_v1` не является live runtime
- offline pipelines не должны описываться как approved for real console use только потому, что у них есть queue or sender plumbing

## Architectural Notes

### Frontend And Handler Surface

- UI входит через WebSocket layer
- handler surface остаётся важной частью live runtime map
- presence of a handler does not equal approval for real console writes

### Audio Path

- `backend/audio_capture.py` остаётся shared capture service for live analyzers and automation
- live modules должны reuse this service instead of creating parallel capture ownership

### Transport Layer

- Wing transport remains the main live-console target surface
- `backend/osc/enhanced_osc_client.py` is a transport wrapper, not a policy gate
- transport availability does not imply approved write scope

## Open Decisions

Следующие формулировки намеренно не закрыты этим документом и остаются open decisions:

- whether feedback reaction path should ever become an approved emergency exception
- whether operator-only legacy utility tools should remain available or later move into archival/removal planning
- whether any offline pipeline may ever attach to a real sender, and under which unified gate model
- whether canonical live input trim should later become explicitly approved write scope
- whether a true global panic-stop capability should ever be designed and approved separately

## Related Documents

- `docs/runtime_ownership_matrix.md`
- `docs/current_architecture_map.md`
- `docs/console_write_safety_policy.md`
