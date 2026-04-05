# AUTO-MIXER-Tubeslave

AI-driven automatic mixer for live sound. See `CLAUDE.md` for full project overview, quick-start commands, and safety rules.

## Cursor Cloud specific instructions

### System dependencies (pre-installed in snapshot)

- `portaudio19-dev`, `libsndfile1-dev`, `python3-dev` — required for `pyaudio` and `soundfile` to compile/install.

### Python backend

- Install from root `requirements.txt` (not `backend/requirements.txt` which includes `essentia` — unavailable on Python 3.12).
- `essentia` does not publish wheels for Python 3.12; it is skipped. All 400+ tests pass without it.
- PyTorch is installed CPU-only (`--index-url https://download.pytorch.org/whl/cpu`) to save disk/time.
- `faster-whisper` is required at runtime (imported by `voice_control.py`) but not listed in the root `requirements.txt`; the update script installs it explicitly.
- Run backend: `cd backend && python3 server.py` (WebSocket on `localhost:8765`).

### Frontend (React)

- Run dev server: `cd frontend && BROWSER=none npm start` (serves on `http://localhost:3000`).
- The frontend connects to the backend via WebSocket at `ws://localhost:8765`.

### Tests

- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` — runs all tests.
- 1 pre-existing test failure in `test_differentiable_console.py` (`test_forward_returns_mix_and_processed`) — not caused by environment setup.
- No external services (mixer hardware, Ollama, ChromaDB server) are needed for tests; everything is self-contained with mocks and synthetic signals.

### WebSocket message format

- Messages use `{"type": "<handler_name>"}` (not `"action"`).
- Handler types are registered in `backend/handlers/__init__.py` from 14 sub-modules.

### Audio in headless VM

- No audio devices are available in the cloud VM (no Dante/SoundGrid). The backend starts and serves WebSocket requests fine without audio hardware.
- For audio-related testing, use `virtual_mixer/virtual_mixer.py` which simulates the WING OSC API.
