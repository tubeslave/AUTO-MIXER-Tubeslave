# Wing OSC mapping plan

This document defines how Wing-specific OSC code should be isolated during the project structure refactor.

## Current role

The project targets Behringer Wing Rack fw 3.0.5 through OSC. The current README documents example addresses such as:

```text
/ch/01/mix/fader
/ch/01/preamp/trim
/ch/01/eq/1/f
/ch/01/dyn/thr
/xremote
```

The current backend server imports `WingClient` from the backend root:

```python
from wing_client import WingClient
```

## Target location

Wing-specific code should move to:

```text
backend/mixers/wing/
  __init__.py
  wing_client.py
  wing_addresses.py
  wing_mapper.py
  wing_capabilities.py
```

## Compatibility shim

During migration, keep root-level compatibility files so existing imports do not break immediately.

```python
# backend/wing_client.py
from mixers.wing.wing_client import *
```

```python
# backend/wing_addresses.py
from mixers.wing.wing_addresses import *
```

Only remove shims after all imports are updated and tests pass.

## Mapper responsibility

The Wing mapper converts typed actions into Wing OSC commands.

Example:

```text
ChannelFaderMove(channel_id=1, value=-5.0 dB)
  -> /ch/01/mix/fader <normalized_value>
```

```text
ChannelEQMove(channel_id=1, band=1, frequency_hz=120, gain_db=-2.0, q=1.2)
  -> /ch/01/eq/1/f <frequency>
  -> /ch/01/eq/1/g <gain>
  -> /ch/01/eq/1/q <q>
```

The decision engine should never build these addresses directly.

## Capability map

`wing_capabilities.py` should describe what is safe/available:

- channel count
- fader range
- trim/preamp control availability
- EQ band count
- dynamics parameters
- mute availability
- bus send support
- unsupported or firmware-sensitive features

## Safety notes

Wing OSC writes must be downstream of safety approval.

Allowed:

```text
TypedAction -> SafetyController.approve() -> WingMapper -> WingClient.send
```

Forbidden:

```text
AutoEQ -> WingClient.send
MuQ -> WingClient.send
Experiment -> WingClient.send
Frontend -> raw OSC write
```

## Migration order

1. Create `backend/mixers/wing/` package.
2. Move `wing_addresses.py` first.
3. Move `wing_client.py` second.
4. Add root compatibility shims.
5. Update imports in `backend/server.py` and tests.
6. Add a smoke test that imports both old and new paths.
7. Add boundary test that only `backend/mixers/` and `backend/osc/` may own raw OSC sends.
