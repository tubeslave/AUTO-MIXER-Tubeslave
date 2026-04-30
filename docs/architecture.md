# AUTO-MIXER-Tubeslave target architecture

This document defines the safe target structure for the automixer refactor. The purpose is to make the project easier to reason about without changing live behavior.

## Current problem

The current backend has a large root-level import surface. `backend/server.py` acts as:

- WebSocket server
- lifecycle manager
- mixer connector
- audio capture manager
- controller registry
- training service bootstrapper
- soundcheck coordinator
- cleanup/shutdown coordinator

This is functional but hard to maintain. The refactor should split responsibilities while preserving behavior.

## Target top-level layout

```text
backend/
  app/
  core/
    state/
    decisions/
    safety/
    logging/
  audio/
    input/
    analysis/
    rendering/
  mixers/
    base_client.py
    wing/
    dlive/
    mixing_station/
  agents/
    mix_agent/
    critics/
    directors/
  experiments/
  tests/

frontend/
config/
docs/
scripts/
sessions/
external/
archive/
```

## Layer responsibilities

### `backend/app/`

Runtime application shell:

- WebSocket server startup
- client registration
- message dispatch
- lifecycle/shutdown
- service wiring

It should not contain mixing algorithms.

### `backend/core/state/`

State models:

- mixer state
- channel state
- session state
- soundcheck state
- last known fader/EQ/dynamics values

### `backend/core/decisions/`

Decision layer:

- rule engine
- policy engine
- typed action definitions
- action routing
- conflict resolution

This layer should produce typed actions, not raw OSC.

### `backend/core/safety/`

Safety gate:

- AutoFOH safety controller
- limits
- rollback
- dry-run / observation wrappers
- phase guard
- feedback guard
- headroom guard

All live corrections must pass through this layer.

### `backend/audio/input/`

Audio input and device management:

- Dante/SoundGrid/sounddevice/file input
- device discovery
- audio buffer lifecycle

### `backend/audio/analysis/`

Measurement only:

- LUFS/RMS/peak
- spectral balance
- dynamics
- phase
- bleed
- feedback detection
- instrument recognition

This layer should not directly change the mixer.

### `backend/audio/rendering/`

Offline rendering only:

- candidate renders
- stem bus renders
- reference comparisons
- best mix export

This layer must not send OSC.

### `backend/mixers/`

Mixer abstraction:

- base mixer interface
- Wing implementation
- dLive implementation
- Mixing Station visualization bridge
- OSC mapping
- capability definitions

### `backend/agents/`

AI/rules/critic integration:

- Mix Agent facade
- MuQ-Eval critic
- fallback critic
- directors/policies

Critics can score and explain. They must not directly send OSC.

### `backend/experiments/`

Offline and research-only code:

- MuQ-Eval director tests
- candidate rendering experiments
- spectral ceiling EQ tests
- old research prototypes

Nothing in this folder should be imported by the live server except through explicit offline command handlers.

## Live control path

```text
Audio input
  -> audio analysis
  -> decision engine
  -> typed action
  -> safety controller
  -> mixer client
  -> OSC transport
  -> event log
  -> rollback snapshot
```

## Forbidden live bypasses

The following are forbidden:

```text
critic -> OSC
experiment -> OSC
offline renderer -> OSC
frontend -> raw OSC without backend safety
MuQ-Eval director -> live mixer client
```

## Compatibility strategy

Move code in stages. Each stage should keep compatibility shims when existing imports are root-level.

Example:

```python
# backend/wing_client.py
from mixers.wing.wing_client import *
```

This allows old imports to keep working while new imports use the target structure.

## Refactor stages

1. Add docs and inventory.
2. Add package directories and `__init__.py` files.
3. Move Wing files with shims.
4. Move mixer abstraction files.
5. Move safety/rollback files.
6. Move audio input/analysis files.
7. Move critic/offline-only files.
8. Add boundary tests for OSC bypass.
9. Split `backend/server.py` into app/lifecycle/handlers/service wiring.

## Non-goals for this refactor

- Do not change EQ/gain/compression algorithms.
- Do not change numerical thresholds.
- Do not change live soundcheck behavior.
- Do not promote MuQ-Eval to live director.
- Do not delete legacy code without inventory evidence.
