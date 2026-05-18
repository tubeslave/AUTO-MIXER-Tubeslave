# Channel Inserts and FX Modules - Usage Guide

## Overview

The `WingClient` now automatically scans all channels (1-40) and their inserts when connected to the Wing mixer. This allows you to:

1. **Read** all channel settings including inserts and FX modules
2. **Send correction parameters** to any FX module on any channel

## Automatic Scanning

When you connect to the Wing mixer, the system automatically:

1. Performs initial scan of basic channel parameters
2. Starts background scan of all 40 channels
3. For each channel, checks for pre-inserts and post-inserts
4. For each insert, scans the FX module to get:
   - FX model name (e.g., "P-BASS", "PCORR", "C5-CMB")
   - All FX parameters (numbered 1-32)
   - FX status (on/off, `fxmix` level)

The scan runs in the background and typically completes in 5-8 seconds.

## Usage Examples

### 1. Get All Channels with Inserts

```python
from wing_client import WingClient
import time

client = WingClient('192.168.1.102', 2223)
client.connect()

# Wait for scan to complete
time.sleep(8)

# Get all channels with inserts
channels = client.get_all_channels_with_inserts()

for ch_num, ch_info in channels.items():
    print(f"Channel {ch_num}:")
    if ch_info['inserts']['pre_insert']:
        fx = ch_info['inserts']['pre_insert']['fx_module']
        print(f"  Pre Insert: {fx['slot']} - {fx['model']}")
    if ch_info['inserts']['post_insert']:
        fx = ch_info['inserts']['post_insert']['fx_module']
        print(f"  Post Insert: {fx['slot']} - {fx['model']}")
```

### 2. Get Specific Channel Inserts

```python
# Get inserts for channel 10
inserts = client.get_channel_inserts(10)

if inserts['pre_insert']:
    fx = inserts['pre_insert']['fx_module']
    print(f"Pre Insert: {fx['slot']}")
    print(f"  Model: {fx['model']}")
    print(f"  Parameters: {fx['parameters']}")
```

### 3. Read FX Module Parameter

```python
# Read parameter 1 from FX13
param_value = client.get_fx_parameter('FX13', 1)
print(f"FX13 Parameter 1: {param_value}")
```

### 4. Send Correction Parameter to FX Module

```python
# Set parameter 1 (intensity) on FX13 to -1.0 dB
client.set_fx_parameter('FX13', 1, -1.0)

# Set parameter 2 (bass gain) on FX13 to -15.0 dB
client.set_fx_parameter('FX13', 2, -15.0)

# Set parameter 3 (crossover frequency) on FX13 to 80 Hz
client.set_fx_parameter('FX13', 3, 80.0)
```

### 5. Control FX Module

```python
# Turn FX module on/off
client.set_fx_on('FX13', 1)  # 1 = on, 0 = off

# Set FX mix level (0-100%)
client.set_fx_mix('FX13', 75.0)

# Change FX model (if needed)
client.set_fx_model('FX13', 'P-BASS')
```

## FX Module Information Structure

When you get FX module information, it contains:

```python
{
    'slot': 'FX13',           # FX slot name
    'model': 'P-BASS',        # FX model name
    'on': 1,                  # On/off status (0/1)
    'mix': 100.0,             # Mix level (0-100%)
    'node_type': '$esrc',     # Node type indicator
    'parameters': {           # Parameter dictionary
        1: -2.0,              # Parameter 1 value
        2: -16.5,             # Parameter 2 value
        3: 61.9,              # Parameter 3 value
        4: 0                  # Parameter 4 value
        # ... more parameters
    }
}
```

## Common FX Models and Parameters

### P-BASS (Psycho Bass)
- Parameter 1: Intensity (dB, range: -24 to +6)
- Parameter 2: Bass Gain (dB, range: -60 to 0)
- Parameter 3: Crossover Frequency (Hz, range: 32 to 200)
- Parameter 4: Solo (0/1)

### PCORR (Phase Corrector)
- Parameter 1: Low Frequency (Hz)
- Parameter 2: High Frequency (Hz)
- Parameter 3: Center Frequency (Hz)
- Parameters 4-15: Additional settings

### C5-CMB (C5 Compressor)
- Multiple parameters (1-32) for compressor settings

## Logging

The system logs scan progress:

```
INFO:wing_client:Starting full channel scan with inserts...
INFO:wing_client:Full channel scan complete. Total parameters: 464
INFO:wing_client:Found 6 channels with inserts:
INFO:wing_client:  Channel 10: Pre: FX13 (P-BASS)
INFO:wing_client:  Channel 12: Pre: FX12 (C5-CMB)
INFO:wing_client:  Channel 13: Post: FX4 (PCORR)
```

## Notes

- The scan runs automatically in the background after connection
- Wait 5-8 seconds after connection before querying channel inserts
- All FX parameters are stored in the client's state dictionary
- Parameter numbers vary by FX model - check WING documentation for specific models
- Use `get_fx_parameter()` to read current values before sending corrections

## See Also

- `example_fx_correction.py` - Complete example script
- `check_channel_10.py` - Example of detailed channel inspection
- `Docs/Behringer_WING_Official/pdfs/WING_Remote_Protocols_FW_3.1_MediaValet.pdf` - Full FX parameter documentation
- `Docs/Behringer_WING_Official/WING_RACK_OSC_FX_GUIDE.md` - local OSC/FX summary for this repo
