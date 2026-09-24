# ADR: Explicit live session composition configuration v1

Date: 2026-09-24
Status: accepted for R3 software cutover; hardware HIL still required

## Context

`LiveSoundcheckService` already owns the canonical `LiveAudioCaptureBridge`, post-console Main evidence, PATCH_VERIFY and the Director/Critic loop. The websocket composition root still created `LiveStartRequest` without a capture bridge, so a production WING start could not supply channel roles or the exact Main-tap route contract needed by the new runtime.

The missing configuration must not be reconstructed from legacy `auto_*` channel classifiers, optimistic WING caches or assumed USB/Main routing. Those would create a second, implicit decision/configuration authority during the repository renovation.

## Decision

Add `backend/live_runtime/session_config.py` as the canonical resolver for composition-root live capture configuration and call it from `backend/handlers/soundcheck_handlers.py`.

The resolver is WING-specific for now because the current PATCH_VERIFY contract and native physical Main-meter evidence are WING-specific. A missing or explicitly disabled capture-bridge section returns no bridge. Once `enabled: true` is declared, validation is fail-closed.

Example shape:

```yaml
live_soundcheck:
  capture_bridge:
    enabled: true
    roles:
      "1": lead_vocal
      "2": guitar
    channel_names:
      "1": Lead Vocal
      "2": Guitar
    window_frames: 2048
    analysis_interval_s: 0.1
    main_tap:
      left_channel: 47
      right_channel: 48
      routes:
        - usb_slot: 47
          source_group: MAIN
          source_channel: 1
        - usb_slot: 48
          source_group: MAIN
          source_channel: 1
```

The route values above are examples only. The runtime supplies no default Main source channel, tap slot or musical role. The exact declared routes are later proved against fresh WING readback by PATCH_VERIFY.

## Validation contract

When the bridge is enabled:

- at least one explicit channel role is required;
- every UI-selected input channel must have an explicit role;
- all role/name/tap/route channel numbers must be within the 48-channel USB feature contract;
- reserved post-console Main tap channels cannot be selected as input channels or receive input roles/names;
- the route list must exactly cover the reserved Main tap slots;
- only `MAIN` route sources are accepted by `MainTapRouteExpectation`;
- invalid explicit configuration blocks the session before `LiveSoundcheckService.start()`;
- the handler does not invent a Main tap when configuration is absent;
- an enabled canonical bridge forces the underlying `AudioCapture` transport to 48 channels even when the operator selected fewer controllable inputs, so the reserved Main-return slots remain present.

The selected-channel list is therefore a decision-authority/input-role selection, not a request to shrink the physical WING USB capture stream.

## Mode and write safety

This ADR does not change the live mode contract. `BENCH_TEST` is only entered when explicitly declared. Missing mode still resolves to read-only `OBSERVE`; explicit legacy `observe_only: false` remains `SUPERVISED`, not BENCH_TEST.

Configuration resolution performs no WING writes. PATCH_VERIFY remains read-only. Autonomous Main mutation remains blocked until physical HIL proves the native Main meter point, USB post-console tap, timing/ballistics corridor and write/readback protections on target hardware.

## Migration classification

- **KEEP_CORE:** `backend/audio_capture.py`, WING transport/query/readback primitives and the 48-channel USB transport contract.
- **ADAPT:** `backend/handlers/soundcheck_handlers.py` as a temporary compatibility composition root; it now delegates roles/Main-tap composition to `live_runtime` and retains legacy event aliases only.
- **ADAPT:** legacy `AutoSoundcheckEngine` only for discovery/connection/AudioCapture ownership until those infrastructure pieces are extracted.
- **ARCHIVE:** legacy heuristic channel classification, soundcheck FSM, instrument presets and AutoFOH/AutoFader/AutoEQ decision/evaluation/rollback paths after runtime references are severed and HIL evidence passes.
- **DELETE_AFTER_PROOF:** no new candidates in this change.

## Software evidence

Focused tests cover:

- absence/disable behavior without invented routing;
- explicit role/name/Main contract resolution;
- exact route coverage and MAIN-only source validation;
- reserved Main channel exclusion;
- selected-channel role completeness;
- websocket composition of the resolved bridge into `LiveStartRequest`;
- forced 48-channel physical capture when the canonical bridge is enabled;
- fail-closed blocking before service start for invalid explicit configuration.

`tests/test_live_session_config.py` and `tests/test_live_soundcheck_handler_composition.py` are part of the focused `Stem Offline Test` workflow.

## Consequence

With an explicit site/session configuration, the production composition path is now structurally continuous:

`websocket start -> explicit roles/Main contract -> 48ch AudioCapture -> service-owned bridge -> coherent post-console Main -> PATCH_VERIFY -> LISTEN -> Director/Critic -> verified control plane`.

Without that explicit configuration the handler does not guess the missing physical/musical evidence, so the new autonomous feature path remains unopened rather than silently inheriting legacy policy.
