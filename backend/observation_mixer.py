"""Observation-mode mixer proxy."""

from __future__ import annotations

import time
from collections import ChainMap
from typing import Any, Callable, Dict, List, Optional


class ObservationMixerClient:
    """Wrap a real mixer client and intercept all writes."""

    _MUTATING_PREFIXES = ("set_", "reset_", "load_", "recall_", "route_", "save_")

    def __init__(self, base_client: Any, on_command: Optional[Callable[[Dict[str, Any]], None]] = None):
        self._base_client = base_client
        self._on_command = on_command
        self._shadow_state: Dict[str, Any] = {}
        self._operations: List[Dict[str, Any]] = []
        self.state = ChainMap(self._shadow_state, getattr(base_client, "state", {}))
        self.callbacks = getattr(base_client, "callbacks", {})

    @property
    def is_connected(self) -> bool:
        return bool(getattr(self._base_client, "is_connected", False))

    def connect(self, *args, **kwargs):
        return self._base_client.connect(*args, **kwargs)

    def disconnect(self):
        return self._base_client.disconnect()

    def subscribe(self, *args, **kwargs):
        return self._base_client.subscribe(*args, **kwargs)

    def get_state(self) -> Dict[str, Any]:
        return dict(self.state)

    def _normalize(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): self._normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize(v) for v in value]
        return repr(value)

    def _record(self, method: str, args: tuple[Any, ...], kwargs: Dict[str, Any], updates: Optional[Dict[str, Any]] = None):
        if updates:
            self._shadow_state.update(updates)

        channel = kwargs.get("channel")
        if channel is None and args and isinstance(args[0], int):
            channel = args[0]

        normalized_args = self._normalize(list(args))
        normalized_kwargs = self._normalize(kwargs)
        arg_items = [repr(v) for v in normalized_args[1:]] if channel is not None else [repr(v) for v in normalized_args]
        kw_items = [f"{k}={repr(v)}" for k, v in normalized_kwargs.items()]
        payload = ", ".join(arg_items + kw_items)
        prefix = f"Ch {channel}: " if channel is not None else ""
        message = f"{prefix}{method}({payload})" if payload else f"{prefix}{method}()"

        operation = {
            "seq": len(self._operations) + 1,
            "timestamp": time.time(),
            "channel": channel,
            "method": method,
            "args": normalized_args,
            "kwargs": normalized_kwargs,
            "message": message,
        }
        self._operations.append(operation)
        if self._on_command:
            self._on_command(operation)
        return True

    def get_operations(self) -> List[Dict[str, Any]]:
        return list(self._operations)

    def get_summary(self) -> Dict[str, Any]:
        channels: Dict[str, Dict[str, Any]] = {}
        for op in self._operations:
            key = str(op["channel"]) if op["channel"] is not None else "global"
            entry = channels.setdefault(key, {"count": 0, "methods": [], "last_message": ""})
            entry["count"] += 1
            if op["method"] not in entry["methods"]:
                entry["methods"].append(op["method"])
            entry["last_message"] = op["message"]
        return {"total_operations": len(self._operations), "channels": channels}

    def _normalize_fx_slot(self, fx_slot: str) -> str:
        slot = str(fx_slot).upper()
        return slot[2:] if slot.startswith("FX") else slot

    def send(self, address: str, *args):
        if not args:
            return self._base_client.send(address)
        return self._record("send", (address, *args), {}, {address: args[-1]})

    def set_channel_gain(self, channel: int, value: float):
        return self._record("set_channel_gain", (channel, value), {}, {f"/ch/{channel}/in/set/trim": value})

    def get_channel_gain(self, channel: int):
        key = f"/ch/{channel}/in/set/trim"
        if key in self._shadow_state:
            return self._shadow_state[key]
        return self._base_client.get_channel_gain(channel)

    def set_gain(self, channel: int, value_db: float):
        return self.set_channel_gain(channel, value_db)

    def set_channel_fader(self, channel: int, value: float):
        return self._record("set_channel_fader", (channel, value), {}, {f"/ch/{channel}/fdr": value})

    def get_channel_fader(self, channel: int):
        key = f"/ch/{channel}/fdr"
        if key in self._shadow_state:
            return self._shadow_state[key]
        return self._base_client.get_channel_fader(channel)

    def set_fader(self, channel: int, value_db: float):
        return self.set_channel_fader(channel, value_db)

    def get_fader(self, channel: int):
        key = f"/ch/{channel}/fdr"
        if key in self._shadow_state:
            return self._shadow_state[key]
        getter = getattr(self._base_client, "get_fader", None)
        if getter is not None:
            return getter(channel)
        return self.get_channel_fader(channel)

    def set_channel_mute(self, channel: int, value: int):
        return self._record("set_channel_mute", (channel, value), {}, {f"/ch/{channel}/mute": value})

    def set_mute(self, channel: int, muted: bool):
        return self.set_channel_mute(channel, 1 if muted else 0)

    def get_mute(self, channel: int):
        key = f"/ch/{channel}/mute"
        if key in self._shadow_state:
            return bool(self._shadow_state[key])
        return self._base_client.get_mute(channel)

    def set_channel_pan(self, channel: int, value: float):
        return self._record("set_channel_pan", (channel, value), {}, {f"/ch/{channel}/pan": value})

    def set_pan(self, channel: int, pan: float):
        return self.set_channel_pan(channel, pan)

    def set_eq_on(self, channel: int, on: int):
        return self._record("set_eq_on", (channel, on), {}, {f"/ch/{channel}/eq/on": on})

    def get_eq_on(self, channel: int):
        key = f"/ch/{channel}/eq/on"
        if key in self._shadow_state:
            return self._shadow_state[key]
        return self._base_client.get_eq_on(channel)

    def set_eq_band_gain(self, channel: int, band: str, gain: float):
        return self._record("set_eq_band_gain", (channel, band, gain), {}, {f"/ch/{channel}/eq/{band}": gain})

    def get_eq_band_gain(self, channel: int, band: str):
        key = f"/ch/{channel}/eq/{band}"
        if key in self._shadow_state:
            return self._shadow_state[key]
        return self._base_client.get_eq_band_gain(channel, band)

    def set_eq_band_frequency(self, channel: int, band: str, frequency: float):
        return self._record("set_eq_band_frequency", (channel, band, frequency), {}, {f"/ch/{channel}/eq/{band}": frequency})

    def get_eq_band_frequency(self, channel: int, band: str):
        key = f"/ch/{channel}/eq/{band}"
        if key in self._shadow_state:
            return self._shadow_state[key]
        return self._base_client.get_eq_band_frequency(channel, band)

    def set_eq_band(self, channel: int, band: int, freq: float = None, gain: float = None, q: float = None):
        updates = {}
        if freq is not None:
            updates[f"/ch/{channel}/eq/{band}f"] = freq
        if gain is not None:
            updates[f"/ch/{channel}/eq/{band}g"] = gain
        if q is not None:
            updates[f"/ch/{channel}/eq/{band}q"] = q
        return self._record("set_eq_band", (channel, band, freq, gain, q), {}, updates)

    def set_eq_low_shelf(self, channel: int, gain: float = None, freq: float = None, q: float = None, eq_type: str = None):
        updates = {}
        if gain is not None:
            updates[f"/ch/{channel}/eq/lg"] = gain
        if freq is not None:
            updates[f"/ch/{channel}/eq/lf"] = freq
        if q is not None:
            updates[f"/ch/{channel}/eq/lq"] = q
        if eq_type is not None:
            updates[f"/ch/{channel}/eq/leq"] = eq_type
        return self._record("set_eq_low_shelf", (channel, gain, freq, q, eq_type), {}, updates)

    def set_eq_high_shelf(self, channel: int, gain: float = None, freq: float = None, q: float = None, eq_type: str = None):
        updates = {}
        if gain is not None:
            updates[f"/ch/{channel}/eq/hg"] = gain
        if freq is not None:
            updates[f"/ch/{channel}/eq/hf"] = freq
        if q is not None:
            updates[f"/ch/{channel}/eq/hq"] = q
        if eq_type is not None:
            updates[f"/ch/{channel}/eq/heq"] = eq_type
        return self._record("set_eq_high_shelf", (channel, gain, freq, q, eq_type), {}, updates)

    def set_low_cut(self, channel: int, enabled: int, frequency: float = None, slope: str = None):
        updates = {f"/ch/{channel}/flt/lc": enabled}
        if frequency is not None:
            updates[f"/ch/{channel}/flt/lcf"] = frequency
        if slope is not None:
            updates[f"/ch/{channel}/flt/lcs"] = slope
        return self._record("set_low_cut", (channel, enabled, frequency, slope), {}, updates)

    def set_high_cut(self, channel: int, enabled: int, frequency: float = None, slope: str = None):
        updates = {f"/ch/{channel}/flt/hc": enabled}
        if frequency is not None:
            updates[f"/ch/{channel}/flt/hcf"] = frequency
        if slope is not None:
            updates[f"/ch/{channel}/flt/hcs"] = slope
        return self._record("set_high_cut", (channel, enabled, frequency, slope), {}, updates)

    def set_hpf(self, channel: int, freq: float, enabled: bool = True):
        updates = {f"/ch/{channel}/flt/lc": 1 if enabled else 0, f"/ch/{channel}/flt/lcf": freq}
        return self._record("set_hpf", (channel, freq, enabled), {}, updates)

    def set_compressor_on(self, channel: int, on: int):
        return self._record("set_compressor_on", (channel, on), {}, {f"/ch/{channel}/dyn/on": on})

    def get_compressor_on(self, channel: int):
        key = f"/ch/{channel}/dyn/on"
        if key in self._shadow_state:
            return self._shadow_state[key]
        return self._base_client.get_compressor_on(channel)

    def set_compressor_gain(self, channel: int, gain: float):
        return self._record("set_compressor_gain", (channel, gain), {}, {f"/ch/{channel}/dyn/gain": gain})

    def get_compressor_gain(self, channel: int):
        key = f"/ch/{channel}/dyn/gain"
        if key in self._shadow_state:
            return self._shadow_state[key]
        return self._base_client.get_compressor_gain(channel)

    def set_compressor(self, channel: int, **kwargs):
        updates = {}
        if "threshold_db" in kwargs:
            updates[f"/ch/{channel}/dyn/thr"] = kwargs["threshold_db"]
        elif "threshold" in kwargs:
            updates[f"/ch/{channel}/dyn/thr"] = kwargs["threshold"]
        if "makeup_db" in kwargs:
            updates[f"/ch/{channel}/dyn/gain"] = kwargs["makeup_db"]
        elif "gain" in kwargs:
            updates[f"/ch/{channel}/dyn/gain"] = kwargs["gain"]
        if "enabled" in kwargs:
            updates[f"/ch/{channel}/dyn/on"] = 1 if kwargs["enabled"] else 0
        return self._record("set_compressor", (channel,), kwargs, updates)

    def get_compressor_gr(self, channel: int):
        getter = getattr(self._base_client, "get_compressor_gr", None)
        return getter(channel) if getter is not None else None

    def set_channel_phase_invert(self, channel: int, value: int):
        return self._record("set_channel_phase_invert", (channel, value), {}, {f"/ch/{channel}/in/set/inv": value})

    def set_polarity(self, channel: int, inverted: bool):
        return self.set_channel_phase_invert(channel, 1 if inverted else 0)

    def set_channel_delay(self, channel: int, value: float, mode: str = "MS"):
        updates = {
            f"/ch/{channel}/in/set/dlymode": mode,
            f"/ch/{channel}/in/set/dly": value,
            f"/ch/{channel}/in/set/dlyon": 1,
        }
        return self._record("set_channel_delay", (channel, value, mode), {}, updates)

    def set_delay(self, channel: int, delay_ms: float, enabled: bool = True):
        updates = {
            f"/ch/{channel}/in/set/dly": delay_ms,
            f"/ch/{channel}/in/set/dlyon": 1 if enabled else 0,
        }
        return self._record("set_delay", (channel, delay_ms, enabled), {}, updates)

    def set_send_level(self, channel: int, send_bus: int, level_db: float, channel_type: str = "input"):
        return self._record("set_send_level", (channel, send_bus, level_db, channel_type), {})

    def set_main_eq_on(self, main: int, on: int):
        return self._record("set_main_eq_on", (main, on), {}, {f"/main/{main}/eq/on": on})

    def set_main_eq_band(self, main: int, band: int, freq: float = None, gain: float = None, q: float = None):
        updates = {}
        if freq is not None:
            updates[f"/main/{main}/eq/{band}f"] = freq
        if gain is not None:
            updates[f"/main/{main}/eq/{band}g"] = gain
        if q is not None:
            updates[f"/main/{main}/eq/{band}q"] = q
        return self._record("set_main_eq_band", (main, band), {"freq": freq, "gain": gain, "q": q}, updates)

    def set_bus_eq_on(self, bus: int, on: int):
        return self._record("set_bus_eq_on", (bus, on), {}, {f"/bus/{bus}/eq/on": on})

    def set_bus_eq_band(self, bus: int, band: int, freq: float = None, gain: float = None, q: float = None):
        updates = {}
        if freq is not None:
            updates[f"/bus/{bus}/eq/{band}f"] = freq
        if gain is not None:
            updates[f"/bus/{bus}/eq/{band}g"] = gain
        if q is not None:
            updates[f"/bus/{bus}/eq/{band}q"] = q
        return self._record("set_bus_eq_band", (bus, band), {"freq": freq, "gain": gain, "q": q}, updates)

    def set_matrix_eq_on(self, matrix: int, on: int):
        return self._record("set_matrix_eq_on", (matrix, on), {}, {f"/mtx/{matrix}/eq/on": on})

    def set_matrix_eq_band(self, matrix: int, band: int, freq: float = None, gain: float = None, q: float = None):
        updates = {}
        if freq is not None:
            updates[f"/mtx/{matrix}/eq/{band}f"] = freq
        if gain is not None:
            updates[f"/mtx/{matrix}/eq/{band}g"] = gain
        if q is not None:
            updates[f"/mtx/{matrix}/eq/{band}q"] = q
        return self._record("set_matrix_eq_band", (matrix, band), {"freq": freq, "gain": gain, "q": q}, updates)

    def set_fx_model(self, fx_slot: str, model: str):
        fx_num = self._normalize_fx_slot(fx_slot)
        return self._record("set_fx_model", (fx_slot, model), {}, {f"/fx/{fx_num}/mdl": model})

    def set_fx_on(self, fx_slot: str, on: int):
        fx_num = self._normalize_fx_slot(fx_slot)
        return self._record("set_fx_on", (fx_slot, on), {}, {f"/fx/{fx_num}/on": on})

    def set_fx_mix(self, fx_slot: str, mix: float):
        fx_num = self._normalize_fx_slot(fx_slot)
        return self._record("set_fx_mix", (fx_slot, mix), {}, {f"/fx/{fx_num}/fxmix": mix})

    def set_fx_parameter(self, fx_slot: str, parameter: int, value: Any):
        fx_num = self._normalize_fx_slot(fx_slot)
        return self._record("set_fx_parameter", (fx_slot, parameter, value), {}, {f"/fx/{fx_num}/{parameter}": value})

    def set_insert(self, target_type: str, target: int, position: str, slot: str = "NONE", on: Optional[int] = None, mode: Optional[str] = None):
        target_map = {
            "channel": "ch",
            "ch": "ch",
            "aux": "aux",
            "bus": "bus",
            "main": "main",
            "matrix": "mtx",
            "mtx": "mtx",
        }
        pos_map = {"pre": "preins", "post": "postins"}
        base = f"/{target_map[str(target_type).lower()]}/{int(target)}/{pos_map[str(position).lower()]}"
        updates: Dict[str, Any] = {f"{base}/ins": str(slot).upper()}
        if on is not None:
            updates[f"{base}/on"] = on
        if mode is not None and str(position).lower() == "post":
            updates[f"{base}/mode"] = mode
        return self._record("set_insert", (target_type, target, position, slot), {"on": on, "mode": mode}, updates)

    def __getattr__(self, name: str):
        attr = getattr(self._base_client, name)
        if not callable(attr):
            return attr
        if name.startswith(self._MUTATING_PREFIXES):
            return lambda *args, **kwargs: self._record(name, args, kwargs)
        return attr
