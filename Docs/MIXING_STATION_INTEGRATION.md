# Mixing Station Integration

This module mirrors Automixer safety decisions into Mixing Station Desktop so
you can visualize faders, mute, pan, and future discovered parameters for WING
Rack and dLive profiles.

## Safety Defaults

- `config/mixing_station.yaml` starts with `enabled: false`.
- `dry_run: true` is the default.
- `live_control: false` is the default.
- `config/mixing_station/safety.yaml` blocks scene recall, phantom power,
  destructive actions, and emergency-stop operation.
- Unknown Mixing Station dataPaths are not sent. They are logged as
  `needs_discovery`.

## Enable Mixing Station APIs

In Mixing Station Desktop:

1. Open `Global App Settings`.
2. Enable the REST API and/or OSC API.
3. Open the local API Explorer at `http://localhost:<configured_port>`.
4. Use the Explorer to confirm real endpoints and dataPath values for the
   selected mixer profile.

The default config assumes:

```yaml
host: "127.0.0.1"
rest_port: 8080
osc_port: 9000
dry_run: true
```

## Choose Profile

For WING Rack:

```yaml
console_profile: "wing_rack"
capabilities_file: "config/mixing_station/wing_rack_capabilities.yaml"
mapping_file: "config/mixing_station/maps/wing_rack.yaml"
```

For dLive:

```yaml
console_profile: "dlive"
capabilities_file: "config/mixing_station/dlive_capabilities.yaml"
mapping_file: "config/mixing_station/maps/dlive.yaml"
```

dLive notes: Mixing Station connects to the MixRack, scenes are treated as
read-only, and cue lists, soft-keys, and actions are blocked.

WING Rack notes: Remote Lock may make the console effectively read-only; verify
parameter access in the Mixing Station API Explorer.

## Commands

Check REST API:

```bash
python scripts/test_mixing_station_connection.py --host 127.0.0.1 --port 8080
```

Discover available paths:

```bash
python scripts/discover_mixing_station_paths.py \
  --console wing_rack \
  --host 127.0.0.1 \
  --port 8080 \
  --out logs/wing_rack_paths.json
```

Send a dry-run fader correction:

```bash
python scripts/send_test_corrections_to_mixing_station.py \
  --console wing_rack \
  --channel 0 \
  --parameter fader \
  --value -5 \
  --unit db \
  --dry-run true
```

Replay a log:

```bash
python scripts/replay_automix_log_to_mixing_station.py \
  --log logs/automix_to_mixing_station.jsonl \
  --speed 1.0 \
  --dry-run true
```

Emergency stop:

```bash
python scripts/mixing_station_emergency_stop.py stop
python scripts/mixing_station_emergency_stop.py status
python scripts/mixing_station_emergency_stop.py clear
```

## Mapped Parameters

Currently mapped for both `wing_rack` and `dlive`:

- `fader` -> `ch.{channel_index}.mix.lvl`
- `mute` -> `ch.{channel_index}.mix.mute`
- `pan` -> `ch.{channel_index}.mix.pan`

These are intended for offline/dry-run visualization first. Before non-dry-run
REST/WebSocket writes, configure the real write endpoint discovered in Mixing
Station API Explorer:

- `rest_command_endpoint`
- `websocket_command_path_template`

OSC fallback can send known dataPaths as `/con/v/{dataPath}` or
`/con/n/{dataPath}`.

## Needs Discovery

These are intentionally blocked until the API Explorer confirms dataPaths:

- `gain`
- `hpf.enabled`
- `hpf.frequency`
- `peq.band*.enabled/frequency/gain/q/type`
- `send.aux*.level`
- `send.fx*.level`
- `compressor.*`
- `gate.threshold`
- `channel.name`
- `channel.color`
- bus, DCA, and main strip paths

## Logs

Every correction attempt is appended to:

```text
logs/automix_to_mixing_station.jsonl
```

Each row contains console profile, channel, parameter, requested/sent value,
transport, dataPath, safety status, dry-run flag, success/error, and reason.

## Avoid Accidental Live Commands

Keep this for visualization:

```yaml
enabled: true
mode: "offline_visualization"
dry_run: true
live_control: false
```

Only after offline testing, path discovery, and operator approval should you set
`dry_run: false`. Keep `live_control: false` unless you explicitly intend real
control and have reviewed `config/mixing_station/safety.yaml`.
