# CLAUDE.md - AUTO MIXER Tubeslave

## Project Overview

AUTO MIXER Tubeslave is an automatic mixing application for the **Behringer Wing Rack** (firmware 3.0.5) digital mixer. It communicates over OSC (Open Sound Control) to provide real-time automated gain staging, EQ, fader management, compression, phase alignment, voice control, and bleed protection. The project is primarily written in Russian (comments, README, docs).

**Architecture:** Electron desktop app with a Python async backend and React frontend, communicating via WebSocket. The backend controls the Wing mixer over OSC/UDP.

```
React Frontend  <--WebSocket (8765)-->  Python Backend  <--OSC UDP (2222/2223)-->  Behringer Wing
```

## Repository Structure

```
AUTO-MIXER-Tubeslave/
├── backend/                    # Python backend (main application logic)
│   ├── server.py               # Main entry point - WebSocket server (~4900 lines)
│   ├── wing_client.py          # OSC client for Behringer Wing
│   ├── wing_addresses.py       # OSC address reference (fw 3.0.5)
│   ├── lufs_gain_staging.py    # LUFS-based gain staging (ITU-R BS.1770)
│   ├── auto_eq.py              # Auto EQ with spectrum analysis
│   ├── auto_fader.py           # Auto fader controller (v1)
│   ├── auto_compressor.py      # Adaptive compressor with profiles
│   ├── voice_control.py        # Voice commands (Faster-Whisper STT)
│   ├── voice_control_v2.py     # Voice control v2
│   ├── voice_control_sherpa.py # Voice control (Sherpa-ONNX)
│   ├── phase_alignment.py      # Phase alignment detection/correction
│   ├── bleed_service.py        # Microphone bleed protection
│   ├── mixing_station_client.py# Multi-mixer support via Mixing Station
│   ├── channel_recognizer.py   # Instrument/source recognition
│   ├── backup_channels.py      # Channel state backup
│   ├── restore_channels.py     # Channel state restore
│   ├── audio_devices.py        # Audio device enumeration
│   ├── signal_analysis.py      # Signal analysis utilities
│   ├── compressor_profiles.py  # Compressor instrument profiles
│   ├── compressor_presets.py   # Compressor presets
│   ├── auto_fader_v2/          # Advanced auto fader system (v2)
│   │   ├── controller.py       # Main v2 controller orchestrating all components
│   │   ├── bridge/             # C++ DSP bridge (cpp_bridge.py, metrics_receiver.py)
│   │   ├── core/               # Analysis modules (acoustic, bleed, activity, spectral)
│   │   ├── balance/            # Control algorithms (PID, fuzzy, hierarchical, static)
│   │   ├── profiles/           # Genre profiles (Pop/Rock, Jazz, Electronic, etc.)
│   │   └── ml/                 # ML data collection framework
│   ├── native/                 # C++ native DSP extensions
│   │   ├── CMakeLists.txt      # CMake build (C++17, SIMD, optional Dante/FFTW3)
│   │   └── src/                # C++ sources (LUFS meter, spectral, shared memory)
│   ├── requirements.txt        # Python dependencies
│   └── test_*.py / *.py        # Test scripts and utilities (50+)
├── frontend/                   # React + Electron desktop app
│   ├── src/
│   │   ├── App.js              # Main React component with tab management
│   │   ├── App.css             # Application styles
│   │   ├── components/         # Feature tab components (8 tabs)
│   │   │   ├── GainStagingTab.js/.css
│   │   │   ├── AutoEQTab.js/.css
│   │   │   ├── AutoFaderTab.js/.css
│   │   │   ├── AutoCompressorTab.js/.css
│   │   │   ├── VoiceControlTab.js/.css
│   │   │   ├── PhaseAlignmentTab.js/.css
│   │   │   ├── AutoSoundcheckTab.js/.css
│   │   │   └── SettingsTab.js/.css
│   │   └── services/
│   │       └── websocket.js    # Singleton WebSocket service (auto-reconnect)
│   ├── public/
│   │   ├── electron.js         # Electron main process (spawns backend)
│   │   └── preload.js          # Electron preload script
│   └── package.json            # npm config (React 18, Electron 28)
├── config/
│   └── default_config.json     # Master configuration (connections, automation params)
├── presets/                    # Saved mix presets (gitignored)
├── Docs/                       # Documentation and Behringer Wing protocol PDFs
│   ├── TECHNICAL.md
│   └── WING Remote Protocols v3.0.5.pdf
├── README.md                   # Project readme (Russian)
├── BUILD.md                    # macOS build instructions (Russian)
├── build.sh                    # Full build script (backend + frontend + .app)
├── build_backend.sh            # Backend-only build (PyInstaller)
├── start_backend.sh            # Backend start script
├── start_frontend.sh           # Frontend start script
└── .gitignore
```

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Frontend | React | 18.2.0 |
| Desktop | Electron | 28.0.0 |
| Backend | Python (asyncio) | 3.9+ |
| Native DSP | C++ | C++17 |
| Mixer Protocol | OSC (UDP) | python-osc |
| Frontend-Backend | WebSocket | websockets 11+ |
| Build (Python) | PyInstaller | 6.0+ |
| Build (Desktop) | electron-builder | 24.9.1 |
| Build (C++) | CMake | 3.15+ |

## Development Setup

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

The WebSocket server starts on `ws://localhost:8765`.

### Frontend

```bash
cd frontend
npm install
npm start              # React dev server on http://localhost:3000
npm run electron:dev   # Full Electron dev mode with hot-reload
```

### C++ Native Extensions

```bash
cd backend/native
mkdir build && cd build
cmake ..
make
```

CMake options: `-DUSE_DANTE=ON/OFF`, `-DUSE_SIMD=ON/OFF`, `-DBUILD_TESTS=ON/OFF`.

## Build & Distribution

Full macOS .app build:

```bash
./build.sh
```

Steps: PyInstaller bundles backend -> npm builds React -> electron-builder creates .app/DMG.

Output: `frontend/dist/AUTO MIXER Tubeslave.app`

## Key Commands

| Command | Purpose |
|---------|---------|
| `python backend/server.py` | Start backend WebSocket server |
| `cd frontend && npm start` | Start React dev server |
| `cd frontend && npm run electron:dev` | Full Electron dev mode |
| `cd frontend && npm run dist` | Build production .app + DMG |
| `./build.sh` | Full production build |
| `./build_backend.sh` | Build backend executable only |
| `cd frontend && npm test` | Run React tests |
| `python backend/test_wing_connection.py` | Test OSC connection to Wing |

## Testing

Backend tests are individual scripts in `backend/` prefixed with `test_`. They require a live Wing mixer connection for most tests. Run them directly:

```bash
cd backend
python test_wing_connection.py      # Basic OSC connectivity
python test_gain_staging_dante.py   # LUFS meter validation
python test_scan_mixer_websocket.py # Mixer state scanning
python test_voice_control.py        # Voice control
```

Frontend tests use React testing library: `cd frontend && npm test`.

C++ tests (when built with `-DBUILD_TESTS=ON`):

```bash
cd backend/native/build
ctest
```

## Architecture & Communication

### Execution Flow

1. Electron main process (`electron.js`) spawns `backend/server.py` as a subprocess
2. Backend starts WebSocket server on port 8765
3. Electron opens window loading the React app
4. React connects to backend via WebSocket (`websocket.js` singleton)
5. User initiates Wing connection from frontend
6. Backend creates `WingClient` and connects via OSC (UDP ports 2222/2223)
7. Bidirectional sync: Wing state changes -> backend -> frontend (and reverse)

### Communication Protocols

**WebSocket (Frontend <-> Backend):** JSON messages with `type` field:

```json
{"type": "connect_wing", "ip": "192.168.1.100", "send_port": 2222, "receive_port": 2223}
{"type": "set_fader", "channel": 1, "value": 0.75}
{"type": "wing_update", "address": "/ch/01/mix/fader", "values": [0.75]}
{"type": "connection_status", "connected": true}
```

**OSC (Backend <-> Wing Mixer):** UDP with standard OSC addresses:

```
/xremote                    # Subscribe to updates
/ch/01/mix/fader <float>    # Channel fader
/ch/01/preamp/trim <float>  # Channel gain/trim
/ch/01/eq/1/f <float>       # EQ band frequency
/ch/01/dyn/thr <float>      # Compressor threshold
```

Full OSC address reference: `backend/wing_addresses.py` and `Docs/WING Remote Protocols v3.0.5.pdf`.

### Connection Modes

The backend supports two connection modes via `AutoMixerServer`:

1. **Direct Wing OSC** (`WingClient`) - Direct UDP connection to Wing mixer
2. **Mixing Station** (`MixingStationClient`) - Via Mixing Station app for multi-mixer support

## Configuration

`config/default_config.json` is the master configuration file with sections:

- `wing` - Mixer connection defaults (IP, ports, firmware version, model)
- `server` - WebSocket host/port
- `automation` - All automation module parameters:
  - `lufs_gain_staging` - Target LUFS, true peak limit, attack/release/hold, presets
  - `auto_eq` - Mud range, threshold, max cut, Q factor
  - `auto_fader` - PID controller params, osc throttle, channel priorities, ducking, spectral masking
  - `bleed_protection` - Spectral/temporal modes, dominance threshold
  - `auto_compressor` - Soundcheck duration, target gain reduction
  - `auto_soundcheck` - Duration per stage
  - `safe_gain_calibration` - Per-instrument target LUFS and peak limits
- `safety` - Max fader, max gain, limits enable

User overrides go in `config/user_config.json` (gitignored).

## Code Conventions

### Python (Backend)

- **Style:** snake_case for functions/variables, PascalCase for classes
- **Async:** asyncio coroutines for WebSocket handling and main server loop
- **Threading:** Background threads for OSC receiver, voice control, audio analysis
- **Logging:** Standard `logging` module, `logger = logging.getLogger(__name__)`
- **Type hints:** Used throughout (`typing` module)
- **Error handling:** try-except with logging, graceful degradation
- **Comments/docs:** Primarily in Russian
- **NumPy types:** Must be converted to native Python types before JSON serialization (see `convert_numpy_types` in server.py)
- **OSC throttle:** WingClient limits send rate per address (default 10 Hz) to prevent flooding

### JavaScript (Frontend)

- **Style:** camelCase for functions/variables, PascalCase for React components
- **State:** React `useState` hooks for UI state
- **WebSocket:** Singleton service (`websocket.js`) with event listener pattern (`on`/`off`/`send`)
- **Components:** Each feature has a dedicated Tab component with paired CSS file
- **CSS:** Component-scoped `.css` files, dark theme (`#1a1a1a` background)

### C++ (Native)

- **Standard:** C++17
- **Build:** CMake 3.15+
- **Optimizations:** `-O3`, `-ffast-math`, SIMD (SSE4.2/AVX on x86, NEON on ARM)
- **Optional deps:** FFTW3 (FFT), Dante Application Library (audio routing)

## Key Backend Modules

| Module | Purpose |
|--------|---------|
| `server.py` | Central WebSocket server, orchestrates all features, bridges frontend-mixer |
| `wing_client.py` | Low-level OSC client with bidirectional sync, state tracking, throttling |
| `lufs_gain_staging.py` | LUFS-based automatic gain control (K-weighting, True Peak, ITU-R BS.1770) |
| `auto_eq.py` | Spectrum analysis (Essentia), resonance detection, instrument profiles, multi-channel EQ |
| `auto_fader.py` | Real-time fader automation with LUFS metering and genre profiles (v1) |
| `auto_fader_v2/` | Advanced fader system: C++ DSP bridge, PID/fuzzy/hierarchical controllers, ML |
| `auto_compressor.py` | Adaptive compressor with threshold/ratio/knee/attack/release control |
| `voice_control.py` | Faster-Whisper STT, fuzzy command matching (rapidfuzz) |
| `phase_alignment.py` | Phase detection and correction between microphones |
| `bleed_service.py` | Microphone bleed detection and compensation (spectral + temporal) |
| `channel_recognizer.py` | Instrument/source recognition from audio signals |

## Auto Fader V2 Architecture

The `auto_fader_v2/` module is a modular system:

- **bridge/** - C++ DSP metrics integration (`CppBridge`, `MetricsReceiver`)
- **core/** - Analysis: `AcousticAnalyzer`, `ChannelClassifier`, `ActivityDetector`, `BleedDetector`, `BleedCompensator`, `VocalActivityDetector`, `SpectralMaskingDetector`, `RollingIntegratedLufs`
- **balance/** - Control algorithms: `StaticBalancer`, `DynamicMixer`, `FuzzyFaderController`, `PIDLoudnessController`, `HierarchicalMixer`
- **profiles/** - Genre-based mixing profiles (`GenreProfile`, `GenreType`)
- **ml/** - ML data collection for training (`MLDataCollector`)
- **controller.py** - Orchestrator with operation modes: STOPPED, STATIC, DYNAMIC, HIERARCHICAL, SOUNDCHECK

## Important Patterns

1. **Server.py is the central hub** - All automation modules are instantiated and managed from `AutoMixerServer`. Changes to automation logic usually touch both the module and `server.py`.

2. **WebSocket message routing** - Messages have a `type` field. The server dispatches based on type to handler methods.

3. **WingClient state dictionary** - `wing_client.state` is a dict mapping OSC addresses to their current values. All modules read from and write through this state.

4. **Callback subscriptions** - Modules register callbacks on the WingClient for OSC address changes.

5. **Configuration-driven** - Behavior is controlled by `default_config.json`. Modules read their config section at startup.

6. **Safety limits** - `safety.max_fader`, `safety.max_gain`, and `safety.enable_limits` prevent dangerous output levels.

## Files to Never Commit

Per `.gitignore`:
- `__pycache__/`, `*.pyc` - Python bytecode
- `venv/`, `env/` - Virtual environments
- `node_modules/` - npm packages
- `dist/`, `build/` - Build output
- `*.app`, `*.dmg` - macOS bundles
- `presets/*.json` - User presets
- `config/user_config.json` - User config overrides
- `*.log` - Log files
- `.DS_Store` - macOS metadata

## Wing Mixer Reference

- **Supported models:** Wing Rack, Wing Full, Wing Compact
- **Firmware:** 3.0.5
- **OSC send port:** 2222 (send `WING?` to initiate connection)
- **OSC receive port:** 2223
- **Channels:** 48 input channels
- **Protocol docs:** `Docs/WING Remote Protocols v3.0.5.pdf`
- **Address reference:** `backend/wing_addresses.py`

## Dependencies

### Python (backend/requirements.txt)

- `python-osc` - OSC protocol
- `websockets` - WebSocket server
- `numpy`, `scipy` - Numerical/DSP
- `pyaudio` - Audio capture
- `aiohttp` - Async HTTP
- `faster-whisper` - Speech-to-text
- `rapidfuzz` - Fuzzy string matching
- `pyloudnorm` - LUFS measurement
- `essentia` - Audio analysis/MIR
- `pedalboard` - Audio effects
- `librosa` - Audio analysis
- `pyinstaller` - Build tool

### Node.js (frontend/package.json)

- `react` 18.2.0, `react-dom`, `react-scripts` 5.0.1
- `electron` 28.0.0, `electron-builder` 24.9.1
- `concurrently`, `wait-on` (dev)
