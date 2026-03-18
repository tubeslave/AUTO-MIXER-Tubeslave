# AGENTS.md

## Cursor Cloud specific instructions

### Services overview

| Service | Port | Start command |
|---------|------|---------------|
| Backend (WebSocket) | 8765 | `cd backend && PYTHONPATH=backend python3 server.py` |
| Frontend (React dev) | 3000 | `cd frontend && BROWSER=none npm start` |
| Virtual Mixer (OSC) | 2222 UDP | `python3 virtual_mixer/virtual_mixer.py` |

### Running tests

See `CLAUDE.md` for test commands. Key command:
```
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
```

### Running lint

Frontend ESLint: `cd frontend && npx eslint src/`

### Non-obvious caveats

- **`python` vs `python3`**: The VM only has `python3` on PATH, not `python`. Always use `python3`.
- **`essentia` package**: Not available for Python 3.12. It is optional and skipped during install. This does not affect tests or normal operation.
- **`faster-whisper`**: Required by `backend/voice_control.py` but not listed in root `requirements.txt`. Must be installed separately (it is in `backend/requirements.txt`).
- **System deps**: `portaudio19-dev`, `libsndfile1`, and `python3-dev` are required for PyAudio and audio libraries to compile/install.
- **Backend uses legacy websockets API**: The server imports `WebSocketServerProtocol` from `websockets.server` (legacy). A deprecation warning is expected.
- **No mixer needed to start backend**: The backend WebSocket server starts and accepts frontend connections without a physical mixer or virtual mixer running. Mixer connection is initiated from the UI.
- **Pre-existing test failure**: `tests/test_ml/test_differentiable_console.py::TestDifferentiableMixingConsole::test_forward_returns_mix_and_processed` fails due to shape mismatch (pre-existing, not environment-related).
- **`~/.local/bin` on PATH**: pip installs scripts to `~/.local/bin`, which may not be on PATH by default. The update script handles this.
