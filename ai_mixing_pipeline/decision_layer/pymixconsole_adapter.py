"""Optional pymixconsole adapter for offline sandbox rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_mixing_pipeline.audio_utils import (
    db_to_amp,
    ensure_stereo,
    limit_peak,
    match_length,
    measure_audio,
    to_mono,
    write_audio,
)

from .action_schema import CandidateActionSet
from .fallback_virtual_mixer import FallbackVirtualMixer
from .virtual_mixer_base import VirtualMixer


class PyMixConsoleAdapter(VirtualMixer):
    """Use pymixconsole as a headless offline console when it is installed."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.warnings: list[str] = []
        self.block_size = int(self.config.get("block_size", 512) or 512)
        self.num_busses = int(self.config.get("num_busses", 0) or 0)
        self._fallback = FallbackVirtualMixer(self.config)
        try:
            import pymixconsole  # type: ignore

            self._pymixconsole = pymixconsole
            self.available = True
            self.warnings.append("pymixconsole import succeeded; using real headless console for supported DSP.")
        except Exception as exc:
            self._pymixconsole = None
            self.available = False
            self.warnings.append(f"pymixconsole unavailable; fallback_virtual_mixer used: {exc}")

    def load_project(self, multitrack_dir: str | Path, channel_map: dict[str, Any] | None = None) -> dict[str, Any]:
        state = self._fallback.load_project(multitrack_dir, channel_map)
        state.setdefault("metadata", {})["preferred_virtual_mixer"] = "pymixconsole"
        state["metadata"]["actual_virtual_mixer"] = "pymixconsole" if self.available else "fallback_virtual_mixer"
        state["metadata"]["warnings"] = list(self.warnings)
        return state

    def render(self, actions: CandidateActionSet, output_path: str | Path) -> dict[str, Any]:
        if not self.available:
            return self._render_with_fallback(actions, output_path)
        if not self._fallback._audio:
            raise RuntimeError("load_project must be called before render")
        try:
            return self._render_with_pymixconsole(actions, output_path)
        except Exception as exc:
            self.warnings.append(f"pymixconsole render failed; fallback_virtual_mixer used: {exc}")
            return self._render_with_fallback(actions, output_path)

    def export_state(self) -> dict[str, Any]:
        state = self._fallback.export_state()
        state.setdefault("metadata", {})["warnings"] = list(self.warnings)
        state["metadata"]["actual_virtual_mixer"] = "pymixconsole" if self.available else "fallback_virtual_mixer"
        return state

    def import_state(self, state: dict[str, Any]) -> None:
        self._fallback.import_state(state)

    def _render_with_fallback(self, actions: CandidateActionSet, output_path: str | Path) -> dict[str, Any]:
        result = self._fallback.render(actions, output_path)
        result["virtual_mixer"] = "fallback_virtual_mixer"
        result.setdefault("warnings", []).extend(self.warnings)
        return result

    def _render_with_pymixconsole(self, actions: CandidateActionSet, output_path: str | Path) -> dict[str, Any]:
        assert self._pymixconsole is not None
        channel_ids = list(self._fallback._audio.keys())
        length = max(len(audio) for audio in self._fallback._audio.values())
        channel_index = {channel_id: index for index, channel_id in enumerate(channel_ids)}
        console = self._pymixconsole.Console(
            block_size=self.block_size,
            sample_rate=self._fallback.sample_rate,
            num_channels=len(channel_ids),
            num_busses=self.num_busses,
        )
        self._mute_default_fx_sends(console)
        for channel_id, index in channel_index.items():
            channel_state = self._fallback.state.channels[channel_id]
            state_gain = float(channel_state.gain_db)
            post_gain = self._get_processor(console.channels[index].post_processors, "post-gain")
            self._set_parameter(post_gain, "gain", state_gain)
            try:
                panner = self._get_processor(console.channels[index].post_processors, "panner")
                pan = max(-1.0, min(1.0, float(channel_state.pan)))
                self._set_parameter(panner, "pan", (pan + 1.0) / 2.0)
            except Exception as exc:
                self.warnings.append(f"pymixconsole panner unavailable for {channel_id}; default pan not applied: {exc}")
        warnings = list(self.warnings)
        action_audit: list[dict[str, Any]] = []
        master_gain_db = 0.0
        master_actions: list[Any] = []
        action_by_channel: dict[str, list[Any]] = {}
        pre_console_actions: dict[str, list[Any]] = {}

        for action in actions.actions:
            channel_id = getattr(action, "channel_id", "mix")
            if action.action_type == "no_change":
                action_audit.append({"action": action.to_dict(), "status": "no_change"})
                continue
            if channel_id == "master":
                if action.action_type == "gain":
                    master_gain_db += float(getattr(action, "gain_db", 0.0))
                    action_audit.append({"action": action.to_dict(), "status": "applied_master"})
                else:
                    master_actions.append(action)
                continue
            if channel_id not in channel_index:
                warnings.append(f"{channel_id} not found by pymixconsole adapter; action skipped.")
                action_audit.append({"action": action.to_dict(), "status": "skipped_missing_channel"})
                continue
            if action.action_type == "gate_expander":
                pre_console_actions.setdefault(channel_id, []).append(action)
                action_audit.append({"action": action.to_dict(), "status": "applied_pre_pymixconsole"})
                continue
            action_by_channel.setdefault(channel_id, []).append(action)

        for channel_id, channel_actions in action_by_channel.items():
            channel = console.channels[channel_index[channel_id]]
            for action in channel_actions:
                status, warning = self._apply_channel_action(channel, action)
                if warning:
                    warnings.append(warning)
                action_audit.append({"action": action.to_dict(), "status": status})

        for action in master_actions:
            status, warning = self._apply_master_action(console, action)
            if warning:
                warnings.append(warning)
            action_audit.append({"action": action.to_dict(), "status": status})

        input_matrix = np.zeros((length, len(channel_ids)), dtype=np.float32)
        for index, channel_id in enumerate(channel_ids):
            channel_audio = match_length(self._fallback._audio[channel_id], length)
            for action in pre_console_actions.get(channel_id, []):
                channel_audio = self._fallback._apply_gate_expander_action(channel_audio, action)
            input_matrix[:, index] = to_mono(channel_audio)

        padded_length = int(np.ceil(length / float(self.block_size)) * self.block_size)
        if padded_length > length:
            padding = np.zeros((padded_length - length, input_matrix.shape[1]), dtype=np.float32)
            input_matrix = np.vstack([input_matrix, padding])

        output = np.zeros((input_matrix.shape[0], 2), dtype=np.float32)
        for start in range(0, input_matrix.shape[0], self.block_size):
            stop = start + self.block_size
            block = input_matrix[start:stop, :]
            processed = console.process_block(block)
            output[start:stop, :] = ensure_stereo(processed).astype(np.float32)
        mix = output[:length, :]

        for action in master_actions:
            if action.action_type == "eq" and self._processor_missing(console.master.processors, "master-eq"):
                mix = self._fallback._apply_eq_action(mix, action)
                action_audit.append({"action": action.to_dict(), "status": "applied_master_fallback_eq"})
            elif action.action_type == "compressor" and self._processor_missing(
                console.master.processors, "master-compressor"
            ):
                mix = self._fallback._apply_compressor_action(mix, action)
                action_audit.append({"action": action.to_dict(), "status": "applied_master_fallback_compressor"})

        if master_gain_db:
            mix = mix * db_to_amp(master_gain_db)
        output_gain_db = 0.0
        if self._fallback.prevent_clipping:
            mix, output_gain_db = limit_peak(mix, self._fallback.peak_ceiling_dbfs)
        path = write_audio(output_path, mix, self._fallback.sample_rate)
        metrics = measure_audio(mix, self._fallback.sample_rate)
        return {
            "candidate_id": actions.candidate_id,
            "path": str(path),
            "sample_rate": self._fallback.sample_rate,
            "duration_sec": round(len(mix) / float(max(1, self._fallback.sample_rate)), 3),
            "output_gain_db": output_gain_db,
            "warnings": warnings,
            "audit": action_audit,
            "metrics": metrics,
            "virtual_mixer": "pymixconsole",
            "osc_midi_sent": False,
        }

    def _apply_channel_action(self, channel: Any, action: Any) -> tuple[str, str | None]:
        if action.action_type == "gain":
            processor = self._get_processor(channel.post_processors, "post-gain")
            self._offset_parameter(processor, "gain", float(getattr(action, "gain_db", 0.0)))
            return "applied_pymixconsole", None
        if action.action_type == "pan":
            processor = self._get_processor(channel.post_processors, "panner")
            pan = max(-1.0, min(1.0, float(getattr(action, "pan", 0.0))))
            self._set_parameter(processor, "pan", (pan + 1.0) / 2.0)
            return "applied_pymixconsole", None
        if action.action_type == "eq":
            processor = self._get_processor(channel.processors, "eq")
            self._apply_eq_to_processor(processor, action)
            return "applied_pymixconsole", None
        if action.action_type == "compressor":
            processor = self._get_processor(channel.processors, "compressor")
            self._apply_compressor_to_processor(processor, action)
            return "applied_pymixconsole", None
        if action.action_type == "gate_expander":
            return "skipped_unsupported", "gate_expander unsupported by pymixconsole adapter; use fallback mixer for gate DSP."
        return "skipped_unsupported", f"{action.action_type} unsupported by pymixconsole adapter; logged but skipped."

    @staticmethod
    def _mute_default_fx_sends(console: Any) -> None:
        """Keep pymixconsole's default delay/reverb busses silent unless explicitly modeled."""

        for bus in getattr(console, "busses", []):
            for _, parameter in bus.parameters:
                parameter.value = -120.0

    def _apply_master_action(self, console: Any, action: Any) -> tuple[str, str | None]:
        if action.action_type == "eq":
            processor = self._get_processor(console.master.processors, "master-eq")
            self._apply_eq_to_processor(processor, action)
            return "applied_master_pymixconsole", None
        if action.action_type == "compressor":
            processor = self._get_processor(console.master.processors, "master-compressor")
            self._apply_compressor_to_processor(processor, action)
            return "applied_master_pymixconsole", None
        return "skipped_unsupported", f"{action.action_type} unsupported on pymixconsole master; logged but skipped."

    @staticmethod
    def _get_processor(processor_list: Any, name: str) -> Any:
        return processor_list.get(name)

    @staticmethod
    def _processor_missing(processor_list: Any, name: str) -> bool:
        try:
            processor_list.get(name)
            return False
        except Exception:
            return True

    @staticmethod
    def _set_parameter(processor: Any, name: str, value: float) -> None:
        getattr(processor.parameters, name).value = float(value)

    @staticmethod
    def _offset_parameter(processor: Any, name: str, delta: float) -> None:
        parameter = getattr(processor.parameters, name)
        parameter.value = float(parameter.value) + float(delta)

    def _apply_eq_to_processor(self, processor: Any, action: Any) -> None:
        freq = float(getattr(action, "freq_hz", 1000.0))
        gain = float(getattr(action, "gain_db", 0.0))
        q = max(0.1, min(10.0, float(getattr(action, "q", 0.7))))
        filter_type = str(getattr(action, "filter_type", "peaking"))
        if filter_type in {"low_shelf", "lowshelf"} or freq < 180.0:
            self._offset_parameter(processor, "low_shelf_gain", gain)
            self._set_parameter(processor, "low_shelf_freq", max(20.0, min(1000.0, freq)))
        elif filter_type in {"high_shelf", "highshelf"} or freq >= 8000.0:
            self._offset_parameter(processor, "high_shelf_gain", gain)
            self._set_parameter(processor, "high_shelf_freq", max(8000.0, min(20000.0, freq)))
        elif freq < 700.0:
            self._offset_parameter(processor, "first_band_gain", gain)
            self._set_parameter(processor, "first_band_freq", max(200.0, min(5000.0, freq)))
            self._set_parameter(processor, "first_band_q", q)
        elif freq < 2500.0:
            self._offset_parameter(processor, "second_band_gain", gain)
            self._set_parameter(processor, "second_band_freq", max(500.0, min(6000.0, freq)))
            self._set_parameter(processor, "second_band_q", q)
        else:
            self._offset_parameter(processor, "third_band_gain", gain)
            self._set_parameter(processor, "third_band_freq", max(2000.0, min(10000.0, freq)))
            self._set_parameter(processor, "third_band_q", q)

    def _apply_compressor_to_processor(self, processor: Any, action: Any) -> None:
        self._set_parameter(processor, "threshold", max(-80.0, min(0.0, float(getattr(action, "threshold_db", -18.0)))))
        self._set_parameter(processor, "ratio", max(1.0, min(100.0, float(getattr(action, "ratio", 1.5)))))
        self._set_parameter(processor, "attack_time", max(0.001, min(500.0, float(getattr(action, "attack_ms", 15.0)))))
        self._set_parameter(processor, "release_time", max(0.0, min(1000.0, float(getattr(action, "release_ms", 140.0)))))
        self._set_parameter(processor, "makeup_gain", max(-12.0, min(24.0, float(getattr(action, "makeup_gain_db", 0.0)))))
