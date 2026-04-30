# Project cleanup inventory

Branch: `chore/project-structure-v2`  
Baseline branch: `master`  
Baseline commit used for this first pass: `44283494008ed0c2d28cfdd8e0ec038a0e77a227`

This document is the first safe inventory pass. It intentionally does **not** delete, move, or rewrite code. The goal is to map the current project, identify live/offline boundaries, and define safe refactor steps before any structural changes.

## Current confirmed entry points and high-risk modules

| path | purpose | status | imported by / relation | zone | deletion risk | recommendation |
|---|---|---:|---|---|---:|---|
| `README.md` | Project overview, install/run instructions, architecture notes for Wing OSC, Mix Agent, MuQ-Eval, training, offline director | used | Human/project documentation | docs | low | keep; later update after refactor |
| `backend/server.py` | Main WebSocket backend entry point. Bridges frontend, mixer clients, audio capture, controllers, AutoSoundcheckEngine, training service, and live/offline control endpoints | used | Backend runtime entry point | live/backend | critical | keep; later split into `backend/app/`, handlers, lifecycle, services |
| `backend/wing_client.py` | Wing OSC client used by backend server | used | imported by `backend/server.py` as `from wing_client import WingClient` | live/mixer | critical | move later to `backend/mixers/wing/wing_client.py` with compatibility shim |
| `backend/wing_addresses.py` | Wing OSC address map/reference | used/likely | documented in README; likely used by Wing client/control modules | live/mixer | high | move later to `backend/mixers/wing/wing_addresses.py` with import updates |
| `backend/dlive_client.py` | Allen & Heath dLive client/bridge | used/likely | imported by `backend/server.py` | live/mixer | high | move later to `backend/mixers/dlive/dlive_client.py` |
| `backend/mixing_station_client.py` | Mixing Station bridge/client | used | imported by `backend/server.py` | live/visualization/mixer | high | move later to `backend/mixers/mixing_station/` |
| `backend/osc/enhanced_osc_client.py` | Enhanced OSC transport wrapper | used | imported by `backend/server.py` | live/mixer transport | critical | keep; later place under `backend/mixers/transport/` or `backend/mixers/osc/` |
| `backend/auto_soundcheck_engine.py` | Headless auto-mixing/soundcheck engine | used | initialized by `backend/server.py` | live/backend | critical | keep; later move under `backend/core/decisions/` or `backend/app/services/` after dependency scan |
| `backend/observation_mixer.py` | Dry-run/observation proxy for soundcheck | used | imported by `backend/server.py` | live/safety/dry-run | high | keep; later move under `backend/core/safety/` or `backend/mixers/observation/` |
| `backend/handlers/` | WebSocket message handlers | used | `register_all_handlers(self)` in `backend/server.py` | live/backend | critical | keep; later align with `backend/app/handlers/` |
| `backend/ws_transport.py` | WebSocket send/broadcast helpers | used | imported by `backend/server.py` | live/backend | high | keep; later move to `backend/app/transport/` |
| `backend/audio_capture.py` | Unified audio capture service | used | imported and started by `backend/server.py` | live/audio input | critical | keep; later move to `backend/audio/input/` |
| `backend/audio_devices.py` | Audio device discovery/listing | used | imported by `backend/server.py` | live/audio input | high | keep; later move to `backend/audio/input/` |
| `backend/lufs_gain_staging.py` | LUFS-based gain staging and SafeGainCalibrator | used | imported by `backend/server.py` | live/audio decisions | critical | keep; later split analysis vs action parts |
| `backend/channel_recognizer.py` | Instrument/channel recognition | used | imported by `backend/server.py` | live/audio analysis | high | keep; later move to `backend/audio/analysis/` |
| `backend/auto_eq.py` | Auto-EQ controllers/profiles | used | imported by `backend/server.py` | live/processing decisions | critical | keep; later split profiles, analysis, actions |
| `backend/phase_alignment.py` | Phase alignment controller | used | imported by `backend/server.py` | live/audio analysis/actions | critical | keep; later ensure safety guards before OSC |
| `backend/system_measurement.py` | Master/system measurement controller | used | imported by `backend/server.py` | live/audio analysis | high | keep; later move to `backend/audio/analysis/` |
| `backend/auto_fader.py` | Auto-fader controller | used | imported by `backend/server.py` | live/decisions/actions | critical | keep; later wrap all output as typed actions |
| `backend/auto_compressor.py` | Auto-compressor controller | used | imported by `backend/server.py` | live/decisions/actions | critical | keep; later wrap all output as typed actions |
| `backend/bleed_service.py` | Bleed detection service | used | instantiated by `backend/server.py` | live/audio analysis | high | keep; later move to `backend/audio/analysis/bleed.py` or service module |
| `backend/feedback_detector.py` | Feedback detection | used | imported by `backend/server.py` | live/safety/audio analysis | critical | keep; later safety integration |
| `backend/backup_channels.py` | Channel backup/snapshot | used | imported by `backend/server.py` | live/safety/rollback | high | keep; later move to `backend/core/safety/rollback.py` |
| `backend/restore_channels.py` | Restore from backup | used | imported by `backend/server.py` | live/safety/rollback | high | keep; later move to `backend/core/safety/rollback.py` |
| `mix_agent/` | Offline/backend facade for analyze/suggest/apply and bridge to AutoFOH typed actions | used/likely | described in README; should not bypass safety | offline/backend bridge | high | keep; later move or mirror under `backend/agents/mix_agent/` only after import scan |
| `config/default_config.json` | Main default configuration | used | loaded by `backend/server.py` | config | critical | keep |
| `config/automixer.yaml` | Overlay config for AI, agent, audio, mixer, websocket, safety, training, MuQ-Eval | used | loaded/merged by `backend/server.py` | config | critical | keep |
| `config/muq_eval.yaml` | MuQ-Eval quality layer config | used/likely | documented in README | config/offline/critic | high | keep |
| `config/muq_director_test.yaml` | Offline-only MuQ director test config | used/likely | documented in README | config/offline experiment | medium | keep; mark offline-only |
| `frontend/src/App.js` | Main React frontend entry | used | frontend runtime | frontend | critical | keep; later split UI by mixer/meters/safety/logs |
| `sessions/` | Session logs/reports/renders | generated | runtime artifacts | generated | low for code, high for user data | keep ignored/generated; do not commit large artifacts |
| `external/` | External models/toolkits/submodules | mixed | external dependencies, research integrations | external | high | do not modify without dedicated task |
| `archive/` | Future holding area for legacy code | planned | cleanup destination only | archive | low | create only when archiving with explanation |

## Confirmed backend imports from `backend/server.py`

The backend server currently imports a large number of modules directly from the backend root. This is the main structural smell: too many live services and controllers are coupled to a single file.

Confirmed direct imports include:

```python
from wing_client import WingClient
from dlive_client import DLiveClient
from osc.enhanced_osc_client import EnhancedOSCClient
from mixing_station_client import MixingStationClient, discover_mixing_station
from audio_devices import get_audio_devices
from dante_routing_config import get_routing_as_dict, get_module_signal_info
from voice_control import VoiceControl
from voice_control_v2 import VoiceControlV2
from lufs_gain_staging import LUFSGainStagingController, SafeGainCalibrator
from channel_recognizer import scan_and_recognize, recognize_instrument_spectral_fallback, AVAILABLE_PRESETS
from auto_eq import AutoEQController, InstrumentProfiles, MultiChannelAutoEQController
from phase_alignment import PhaseAlignmentController
from system_measurement import SystemMeasurementController, TargetBus
from auto_fader import AutoFaderController
from auto_compressor import AutoCompressorController
from backup_channels import backup_channel
from restore_channels import restore_from_backup_using_client
from bleed_service import BleedService
from audio_capture import AudioCapture
from feedback_detector import FeedbackDetector
from auto_soundcheck_engine import AutoSoundcheckEngine
from observation_mixer import ObservationMixerClient
from mixer_discovery import discover_mixers, discover_mixer_auto, DiscoveredMixer
from handlers import register_all_handlers
from user_config_store import load_user_config, save_user_config
from ws_transport import broadcast_json, is_connection_closed_error, send_json
```

## Safety boundary inventory

### Must remain live-capable

- `backend/server.py`
- mixer clients and OSC transport
- auto soundcheck engine
- gain staging, EQ, fader, compressor controllers
- feedback detector
- backup/restore/rollback code
- WebSocket handlers

### Must remain offline-only unless explicitly promoted through safety gates

- MuQ-Eval director
- candidate renderers
- offline mix experiments
- critic-only modules
- training/study/discovery routines
- benchmark/evaluation scripts

## Known safety rule

No module under the future paths below may directly send OSC:

- `backend/agents/critics/`
- `backend/experiments/`
- `backend/audio/rendering/`
- MuQ-Eval director/offline director modules

Allowed live path:

```text
Decision Engine
  -> Typed Action
  -> AutoFOHSafetyController / safety guards
  -> Mixer Client
  -> OSC transport
  -> log + rollback snapshot
```

## Next inventory tasks before moving code

1. Run local repo scan:
   ```bash
   find . -type f | sort > docs/file_list.txt
   python scripts/generate_inventory.py
   ```
2. Build import graph:
   ```bash
   python -m pip install grimp import-linter || true
   python scripts/generate_import_graph.py
   ```
3. Search direct OSC bypasses:
   ```bash
   grep -R "send_message\|send_osc\|udp_client\|SimpleUDPClient\|pythonosc" -n backend mix_agent src || true
   ```
4. Search offline/director modules:
   ```bash
   grep -R "MuQ\|muq\|candidate\|director\|render" -n backend mix_agent src config || true
   ```
5. Only after these scans, start file moves with compatibility shims and tests.

## First-pass recommendation

Do not move code in this pass. The backend has a large root-level import surface and direct runtime imports from `backend/server.py`. Moving files without a complete import graph risks breaking live startup. The first safe PR should add this inventory and architecture docs, then the next PR should move Wing files with backward-compatible shims.
