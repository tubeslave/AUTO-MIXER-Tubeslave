# Инструменты, API и протоколы

## Behringer WING Rack — OSC Protocol (fw 3.0.5)

### Подключение
```
1. Отправить 'WING?' на UDP port 2222 (handshake)
2. OSC-команды на UDP port 2223
3. Keepalive: отправлять XREMOTE каждые 5 секунд
4. Rate limit: ≤ 10 команд/сек на адрес (throttle в wing_client.py)
```

### Маппинг фейдеров
```
OSC float → dB:
  0.0     = -∞ (mute)
  0.7498  = 0 dB
  1.0     = +10 dB

dB → OSC float:
  Линейная интерполяция по таблице WING (нелинейная шкала)
  См. wing_client.py: db_to_fader(), fader_to_db()
```

### Основные OSC-адреса
```
Channels (1-40):
  /ch/{N}/mix/fader     float [0.0-1.0]    Fader level
  /ch/{N}/mix/on        int [0,1]          Mute on/off
  /ch/{N}/mix/pan       float [0.0-1.0]    Pan (0.5 = center)
  /ch/{N}/preamp/trim   float              Trim/Gain
  /ch/{N}/eq/{B}/f      float              EQ band frequency (B=1-4)
  /ch/{N}/eq/{B}/g      float              EQ band gain
  /ch/{N}/eq/{B}/q      float              EQ band Q
  /ch/{N}/eq/{B}/type   int                EQ band type (0=peaking, etc.)
  /ch/{N}/eq/on         int [0,1]          EQ on/off
  /ch/{N}/gate/on       int [0,1]          Gate on/off
  /ch/{N}/gate/thr      float              Gate threshold
  /ch/{N}/dyn/on        int [0,1]          Compressor on/off
  /ch/{N}/dyn/thr       float              Compressor threshold
  /ch/{N}/dyn/ratio     int                Ratio (index в WING_RATIO_VALUES)
  /ch/{N}/dyn/attack    float              Attack time
  /ch/{N}/dyn/release   float              Release time
  /ch/{N}/dyn/knee      float              Knee
  /ch/{N}/config/name   string             Channel name
  /ch/{N}/config/icon   int                Channel icon
  /ch/{N}/config/color  int                Channel color

Main:
  /main/mix/fader       float              Main fader
  /main/mix/on          int                Main mute

Buses (1-16):
  /bus/{N}/mix/fader    float              Bus fader
  /bus/{N}/mix/on       int                Bus mute

Snapshots:
  /‐snap/load/{N}       —                  Load snapshot N
  /‐snap/name/{N}       string             Snapshot name (read-only cache!)
```

### WING Ratio Values (индекс → ratio)
```python
WING_RATIO_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0]
WING_RATIO_STRINGS = ["1:1", "1.5:1", "2:1", "2.5:1", "3:1", "4:1", "5:1", "7:1", "10:1", "20:1"]
```

### Полная документация
`Docs/WING Remote Protocols v3.0.5.pdf` — официальный справочник (все адреса, типы, диапазоны).

---

## Allen & Heath dLive — MIDI over TCP

### Подключение
```
TCP socket на port 51328 (plain) или 51329 (TLS)
Протокол: стандартный MIDI поверх TCP stream
SysEx header: F0 00 00 1A 50 10 01 00
SysEx end: F7
```

### Маппинг фейдеров (NRPN 14-bit)
```
0x0000 (0)     = -∞
0x2AAA (10922) = 0 dB
0x3FFF (16383) = +10 dB
```

### NRPN: EQ (Input channels)
```
Band 1: freq=0x40, gain=0x41, Q=0x42
Band 2: freq=0x44, gain=0x45, Q=0x46
Band 3: freq=0x48, gain=0x49, Q=0x4A
Band 4: freq=0x4C, gain=0x4D, Q=0x4E
HPF:    freq=0x50, on/off=0x51
```

### Channel Types → MIDI channel offset
```python
CHANNEL_TYPE_OFFSET = {
    "input": 0,
    "mono_group": 1, "stereo_group": 1,
    "mono_aux": 2, "stereo_aux": 2, "mono_fx_send": 2, "stereo_fx_send": 2,
    "mono_matrix": 3, "stereo_matrix": 3,
    "dca": 4,
}
```

### Документация
`Docs/` — Allen & Heath dLive MIDI Over TCP Protocol V2.0

---

## Audio Interfaces

### Dante Virtual Soundcard
- Auto-detect через `audio_devices.py`
- Sample rate: 48000 Hz (стандарт для live)
- Latency: настраивается (1-5 мс)
- Каналы: до 64x64

### Waves SoundGrid
- Auto-detect через `audio_devices.py`
- Sample rate: 44100/48000/96000 Hz
- Low-latency mode

### Конфигурация
```yaml
# config/automixer.yaml
audio:
  sample_rate: 48000
  block_size: 1024
  channels: 40
  source: "dante"  # dante, sounddevice, test_sine, test_pink_noise, silence
```

---

## AI / LLM

### Ollama (локальный)
```yaml
ai:
  llm_backend: "ollama"
  llm_model: "llama3"
  ollama_url: "http://localhost:11434"
```
- Используется для real-time решений при концерте (100-500 мс latency на M3 Max)
- Рекомендуемая модель: llama3.3:70b

### Perplexity API
```yaml
ai:
  llm_backend: "perplexity"
  perplexity_api_key: "pplx-..."
```
- Используется при саундчеке (доступ к актуальным знаниям)
- Function calling для управления пультом

### ChromaDB (Knowledge Base)
```python
# backend/ai/knowledge_base.py
# Embedding: sentence-transformers/all-MiniLM-L6-v2
# Collections: mixing_rules, instrument_profiles, troubleshooting, wing_osc_reference
```

---

## Библиотеки Python

### Core
| Библиотека | Назначение |
|-----------|-----------|
| `python-osc` | OSC клиент/сервер (WING) |
| `websockets` | WebSocket сервер (frontend ↔ backend) |
| `numpy` | Числовые вычисления, DSP |
| `scipy` | Signal processing (фильтры, FFT) |
| `pyloudnorm` | LUFS по ITU-R BS.1770-4 |
| `sounddevice` | Audio I/O |
| `librosa` | Audio analysis (spectral features) |

### ML
| Библиотека | Назначение |
|-----------|-----------|
| `torch` | Neural networks (classifier, predictor, style transfer) |
| `scikit-learn` | Feature preprocessing, классические ML |
| `joblib` | Model serialization |

### AI
| Библиотека | Назначение |
|-----------|-----------|
| `chromadb` | Vector database для RAG |
| `sentence-transformers` | Embeddings для knowledge base |
| `httpx` | HTTP client (Ollama, Perplexity API) |

### Infrastructure
| Библиотека | Назначение |
|-----------|-----------|
| `structlog` | Structured logging (JSON/human-readable) |
| `prometheus-client` | Metrics export |
| `watchdog` | Config hot-reload |
| `matchering` | Auto-mastering по референсу |
| `pyyaml` | YAML config parsing |

---

## CI/CD

### GitHub Actions (`.github/workflows/test.yml`)
- Trigger: push/PR на `main`
- Matrix: Python 3.10, 3.11, 3.12
- Steps: install → compile check → pytest
- System deps: `portaudio19-dev`, `libsndfile1`

### Локальный запуск тестов
```bash
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```
