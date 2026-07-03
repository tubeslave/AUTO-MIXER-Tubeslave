"""Clean Ayaic offline render pipeline.

Pipeline order:
1. Bleed-aware primary signal detection.
2. Arrangement-aware input levels measured on primary signal windows.
3. Global bleed/correlation phase-delay alignment.
4. Offline HPF/LPF correction.
5. Optional offline Corrective EQ.
6. Optional contextual compression.
7. Musical panning after dynamics and before final output balance.
8. Musical output balance measured after filters/EQ/compression/panning.
9. Snare top/bottom coherence guard.
10. Snare top/bottom relative balance.
11. Post-balance panorama width restoration.
12. Ayaic master level.

BUS/group level correction is available as an explicit opt-in, but the fixed clean
pipeline uses channel/pair levels and the master level only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np

from .arrangement_input_levels import ARRANGEMENT_INPUT_LEVEL_STAGE, apply_arrangement_input_levels
from .autoeq import AutoEQEngine, AutoEQSettings, AutoEQTrack
from .contextual_compression import (
    CONTEXTUAL_COMPRESSION_STAGE,
    DEFAULT_CONTEXTUAL_COMPRESSION_STYLE,
    apply_contextual_compression,
)
from .musical_panning import (
    DEFAULT_MUSICAL_PANNING_STYLE,
    MUSICAL_PANNING_STAGE,
    apply_musical_panning,
)
from .musical_panorama_width import (
    MUSICAL_PANORAMA_WIDTH_STAGE,
    apply_musical_panorama_width,
)
from .musical_output_balance import (
    DEFAULT_MUSICAL_BALANCE_STYLE,
    MUSICAL_OUTPUT_BALANCE_STAGE,
    apply_musical_output_balance,
)
from .snare_top_bottom_balance import (
    SNARE_TOP_BOTTOM_BALANCE_STAGE,
    apply_snare_top_bottom_balance,
)
from .snare_pair_coherence import (
    SNARE_PAIR_COHERENCE_STAGE,
    apply_snare_pair_coherence,
)
from .models import amp_to_db, db_to_amp, ensure_stereo, jsonable


AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac"}
DEFAULT_MASTER_TARGET_LUFS = -20.0
DEFAULT_MASTER_CEILING_DBFS = None
DEFAULT_MAX_PHASE_DELAY_MS = 10.0
DEFAULT_CORRECTIVE_EQ_METHOD = "contextual_deep_eq"
CORRECTIVE_EQ_METHODS = (
    "none",
    "profile",
    "bleed_control",
    "cross_adaptive",
    "frequency_window",
    "frequency_cross_profile",
    "project_corrective",
    "project_corrective_hybrid",
    "contextual_deep_eq",
)
AYAIC_BUS_GROUP_TARGETS = {
    "drums": -25.0,
    "snare": -25.0,
    "bass": -25.0,
    "playback": -25.0,
    "accordion": -25.0,
    "guitars": -25.0,
    "vocals": -22.0,
    "back_vocals": -28.0,
}
AYAIC_BUS_GROUP_ORDER = ("drums", "snare", "bass", "playback", "accordion", "guitars", "vocals", "back_vocals")
AYAIC_BUS_GROUP_TARGET_SOURCE = {
    "name": "User-defined SONG REPA Ayaic bus targets",
    "url": None,
    "notes": [
        "Drums bus: -25 LUFS",
        "Snare pair bus: -25 LUFS",
        "Bass bus: -25 LUFS",
        "Playback bus: -25 LUFS",
        "Accordion bus: -25 LUFS",
        "Guitars bus: -25 LUFS",
        "Vocals bus: -22 LUFS",
        "Back vocals bus: -28 LUFS",
    ],
}
OFFLINE_HPF_TARGETS_HZ = {
    "kick": 35.0,
    "bass": 35.0,
    "bass_guitar": 35.0,
    "snare_top": 90.0,
    "snare_bottom": 120.0,
    "snare": 110.0,
    "rack_tom": 65.0,
    "floor_tom": 55.0,
    "hi_hat": 180.0,
    "ride": 170.0,
    "overhead": 150.0,
    "room": 80.0,
    "lead_vocal": 90.0,
    "backing_vocal": 110.0,
    "electric_guitar": 90.0,
    "accordion": 100.0,
    "playback": 30.0,
    "custom": 80.0,
}
OFFLINE_LPF_TARGETS_HZ = {
    "kick": 6800.0,
    "snare": 7600.0,
    "snare_top": 7600.0,
    "snare_bottom": 7600.0,
    "rack_tom": 5800.0,
    "floor_tom": 5800.0,
    "hi_hat": 10500.0,
    "ride": 10500.0,
    "overhead": 9000.0,
    "room": 6500.0,
    "backing_vocal": 9500.0,
}
OFFLINE_HPF_LPF_TARGET_SOURCE = {
    "name": "offline_agent_mix classify_track/apply_codex_bleed_control",
    "source_branch": "refs/heads/chore/cleanup-inventory",
    "source_path": "tools/offline_agent_mix.py",
    "notes": [
        "Base HPF values follow offline classify_track.",
        "LPF values follow the offline cymbal/bleed-control correction rules.",
        "Final channel/pair Ayaic levels compensate filter- and EQ-induced loudness changes.",
    ],
}
OFFLINE_PROFILE_EQ_TARGET_SOURCE = {
    "name": "offline_agent_mix classify_track eq_bands",
    "source_branch": "refs/heads/master",
    "source_path": "tools/offline_agent_mix.py",
    "source_function": "classify_track",
}
OFFLINE_BLEED_CONTROL_EQ_TARGET_SOURCE = {
    "name": "offline_agent_mix apply_codex_bleed_control EQ cuts only",
    "source_branch": "refs/heads/master",
    "source_path": "tools/offline_agent_mix.py",
    "source_function": "apply_codex_bleed_control",
    "notes": [
        "Only append_eq_band corrective cuts are used here.",
        "Offline fader, compressor, and LPF edits from this method are intentionally excluded.",
    ],
}
OFFLINE_CROSS_ADAPTIVE_EQ_TARGET_SOURCE = {
    "name": "offline_agent_mix priority cross_adaptive_eq",
    "source_branch": "refs/heads/master",
    "source_path": "tools/offline_agent_mix.py",
    "source_function": "apply_cross_adaptive_eq",
}
OFFLINE_FREQUENCY_WINDOW_EQ_TARGET_SOURCE = {
    "name": "offline_agent_mix frequency_window_balance",
    "source_branch": "refs/heads/master",
    "source_path": "tools/offline_agent_mix.py",
    "source_function": "apply_frequency_window_balance",
}
OFFLINE_FREQUENCY_CROSS_PROFILE_EQ_TARGET_SOURCE = {
    "name": "offline Corrective EQ chain: frequency_window -> cross_adaptive -> profile",
    "source_branch": "refs/heads/master",
    "source_path": "tools/offline_agent_mix.py",
    "source_functions": [
        "apply_frequency_window_balance",
        "apply_cross_adaptive_eq",
        "classify_track",
    ],
    "notes": [
        "Applies method 4, then method 3, then method 1 as one Corrective EQ block.",
        "Channel/pair Ayaic levels are applied once after the full EQ chain.",
    ],
}
PROJECT_CORRECTIVE_EQ_TARGET_SOURCE = {
    "name": "Project Corrective EQ",
    "source": "local project_corrective role-aware method",
    "notes": [
        "Diagnostics use primary-signal audio after phase/delay and HPF/LPF.",
        "Channel and group spectra drive bounded role-aware corrections.",
        "Cuts are preferred over boosts; channel/pair Ayaic is reapplied after EQ.",
    ],
}
PROJECT_CORRECTIVE_HYBRID_EQ_TARGET_SOURCE = {
    "name": "Hybrid Project Corrective EQ",
    "source": "local project_corrective_hybrid decision layer",
    "notes": [
        "Uses Project Corrective EQ as the primary decision source.",
        "Uses frequency-window balance as broad-zone diagnostic evidence, not as a chained EQ block.",
        "Adds selective drum bleed-control cuts only on close drum microphones.",
        "Applies one shared safety/dedup limiter; boosts are not applied.",
        "Dynamic EQ candidates are reported only; this offline render applies static EQ bands.",
    ],
}
CONTEXTUAL_DEEP_EQ_TARGET_SOURCE = {
    "name": "Contextual Deep EQ",
    "source": "local role-aware, masking-aware deep EQ optimizer",
    "references": [
        {
            "name": "W3C Audio EQ Cookbook",
            "url": "https://www.w3.org/TR/audio-eq-cookbook/",
            "use": "stable peaking biquad coefficient model",
        },
        {
            "name": "Autonomous multitrack equalization based on masking reduction",
            "url": "https://www.researchgate.net/publication/277935370_Autonomous_Multitrack_Equalization_Based_on_Masking_Reduction",
            "use": "cross-channel masking reduction as an automatic mixing strategy",
        },
        {
            "name": "Cross-adaptive processing as musical intervention",
            "url": "https://www.frontiersin.org/journals/digital-humanities/articles/10.3389/fdigh.2018.00017/full",
            "use": "cross-adaptive listening between channels",
        },
    ],
    "notes": [
        "Allows larger corrective moves than project_corrective_hybrid.",
        "Hard limits only reject invalid/unstable filter geometry.",
        "Large or risky EQ moves are applied with critic notes and regret_score instead of being silently blocked.",
    ],
}
OFFLINE_PROFILE_CORRECTIVE_EQ_BANDS = {
    "kick": [(60.0, 3.0, 0.9), (320.0, -3.0, 1.3), (4200.0, 2.0, 1.2)],
    "snare_bottom": [(220.0, 1.0, 1.0), (5200.0, 2.0, 1.2), (850.0, -2.0, 2.0)],
    "snare_top": [(200.0, 2.0, 1.0), (850.0, -2.5, 2.0), (5200.0, 3.0, 1.0)],
    "snare": [(220.0, -2.0, 1.1), (750.0, -2.5, 1.7), (4800.0, 2.0, 1.0)],
    "floor_tom": [(95.0, 2.0, 1.0), (360.0, -2.5, 1.5), (4200.0, 1.5, 1.2)],
    "rack_tom": [(120.0, 2.0, 1.0), (380.0, -2.0, 1.5), (4300.0, 1.5, 1.2)],
    "hi_hat": [(450.0, -1.5, 1.2), (6500.0, 1.5, 1.0), (9500.0, 1.0, 0.8)],
    "ride": [(420.0, -1.2, 1.2), (4800.0, 1.2, 1.0), (9000.0, 1.0, 0.8)],
    "overhead": [(350.0, -1.5, 1.2), (3500.0, -1.0, 1.5), (10500.0, 1.5, 0.8)],
    "room": [(250.0, -1.5, 1.2), (2500.0, -1.0, 1.4), (8500.0, 0.8, 1.0)],
    "bass_guitar": [(80.0, 2.0, 0.9), (250.0, -2.5, 1.2), (750.0, 1.2, 1.0)],
    "electric_guitar": [(250.0, -2.0, 1.2), (2500.0, 1.5, 1.0), (6200.0, -1.0, 1.0)],
    "accordion": [(350.0, -1.5, 1.3), (2300.0, 1.3, 1.0), (7000.0, 0.8, 1.0)],
    "playback": [(180.0, -0.8, 1.0), (3500.0, 0.8, 1.0)],
    "backing_vocal": [(240.0, -2.0, 1.3), (2800.0, 1.5, 1.0), (9000.0, 1.0, 0.8)],
    "lead_vocal": [(250.0, -2.5, 1.4), (3100.0, 2.5, 1.0), (10500.0, 1.5, 0.8)],
    "custom": [(300.0, -1.0, 1.2), (3000.0, 0.8, 1.0)],
}
FREQUENCY_WINDOW_DEFINITIONS = (
    {
        "id": "low_end_foundation",
        "label": "Low End Foundation",
        "low_hz": 20.0,
        "high_hz": 120.0,
        "focus": "kick, bass, sub stability, mono foundation",
        "action_mode": "report_only",
    },
    {
        "id": "warmth_mud",
        "label": "Warmth / Mud",
        "low_hz": 120.0,
        "high_hz": 500.0,
        "focus": "body, mud, room build-up, low-mid masking",
        "action_mode": "cleanup_music",
        "center_hz": 320.0,
        "q": 1.15,
    },
    {
        "id": "core_mids",
        "label": "Core Mids",
        "low_hz": 500.0,
        "high_hz": 1000.0,
        "focus": "musical skeleton, note readability, center integrity",
        "action_mode": "report_only",
    },
    {
        "id": "vocal_conflict",
        "label": "Vocal Conflict",
        "low_hz": 700.0,
        "high_hz": 1500.0,
        "focus": "lead vocal against guitars, keys, playback, backing stack",
        "action_mode": "clear_vocal_space",
        "center_hz": 1100.0,
        "q": 1.3,
    },
    {
        "id": "presence_harshness",
        "label": "Presence / Harshness",
        "low_hz": 1500.0,
        "high_hz": 6000.0,
        "focus": "presence, attack, intelligibility, harshness buildup",
        "action_mode": "clear_vocal_space",
        "center_hz": 2900.0,
        "q": 1.45,
    },
    {
        "id": "air_sibilance",
        "label": "Air / Sibilance",
        "low_hz": 6000.0,
        "high_hz": 16000.0,
        "focus": "hats, cymbals, sibilance, air integration",
        "action_mode": "cymbal_control",
        "center_hz": 8500.0,
        "q": 1.2,
    },
)
WINDOW_SPACE_COMPETITOR_INSTRUMENTS = {
    "accordion",
    "backing_vocal",
    "guitar",
    "electric_guitar",
    "lead_guitar",
    "rhythm_guitar",
    "acoustic_guitar",
    "keys",
    "piano",
    "organ",
    "synth",
    "pad",
    "lead_synth",
    "playback",
}
CYMBAL_INSTRUMENTS = {"hi_hat", "ride", "overhead", "oh_l", "oh_r", "cymbals"}
DRUM_INSTRUMENTS = {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "overhead", "room", "percussion"}


@dataclass
class AyaicStem:
    name: str
    path: str
    audio: np.ndarray
    sample_rate: int
    channel_id: int
    target_lufs: float
    group: str
    track_gain_db: float = 0.0
    group_gain_db: float = 0.0
    phase_invert: bool = False
    delay_ms: float = 0.0
    pan: float = 0.0
    primary_loudness_ranges: list[tuple[int, int]] = field(default_factory=list)
    bleed_analysis: dict[str, Any] = field(default_factory=dict)
    notes: list[dict[str, Any]] = field(default_factory=list)
    pan_notes: list[dict[str, Any]] = field(default_factory=list)
    corrective_eq_notes: list[dict[str, Any]] = field(default_factory=list)
    compression_notes: list[dict[str, Any]] = field(default_factory=list)
    output_balance_notes: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AyaicPipelineResult:
    output_wav: str
    output_mp3: str | None
    report: str
    summary: dict[str, Any]


def run_ayaic_offline_pipeline(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    output_name: str = "ayaic_offline_mix",
    master_target_lufs: float = DEFAULT_MASTER_TARGET_LUFS,
    master_ceiling_dbfs: float | None = DEFAULT_MASTER_CEILING_DBFS,
    max_phase_delay_ms: float = DEFAULT_MAX_PHASE_DELAY_MS,
    write_mp3: bool = True,
    drums_only: bool = False,
    skip_group_levels: bool = True,
    skip_master_level: bool = False,
    corrective_eq_method: str = DEFAULT_CORRECTIVE_EQ_METHOD,
    autoeq_enabled: bool = False,
    autoeq_mode: str = "offline",
    autoeq_report_only: bool = False,
    autoeq_config: dict[str, Any] | None = None,
    compression_enabled: bool = True,
    compression_style: str = DEFAULT_CONTEXTUAL_COMPRESSION_STYLE,
    compression_bpm: float | None = None,
    compression_report_only: bool = False,
    panning_enabled: bool = True,
    panning_style: str = DEFAULT_MUSICAL_PANNING_STYLE,
    panning_report_only: bool = False,
    output_balance_enabled: bool = True,
    output_balance_style: str = DEFAULT_MUSICAL_BALANCE_STYLE,
    output_balance_report_only: bool = False,
) -> AyaicPipelineResult:
    corrective_eq_method = _normalize_corrective_eq_method(corrective_eq_method)
    stems = _load_stems(input_dir)
    if drums_only:
        stems = [stem for stem in stems if _drum_instrument(stem.name) is not None]
    if not stems:
        raise ValueError(f"no audio stems found in {input_dir}")

    sample_rate = stems[0].sample_rate
    _align_stems(stems)
    bleed_report = _analyze_bleed_activity(stems)
    pre_phase_channel_report = apply_arrangement_input_levels(
        stems,
        stage=ARRANGEMENT_INPUT_LEVEL_STAGE,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
        primary_audio_fn=_stem_primary_loudness_audio,
        band_energy_fn=_band_energy_for_rendered,
        db_to_amp_fn=db_to_amp,
    )
    phase_stage_report = _apply_phase_delay_from_overheads(stems, max_delay_ms=max_phase_delay_ms)
    primary_shift_report = _shift_primary_loudness_ranges_for_applied_delays(stems)
    snare_pair_coherence_report = apply_snare_pair_coherence(stems)
    hpf_lpf_report = _apply_hpf_lpf_correction(stems)
    autoeq_report = _apply_autoeq_engine(
        stems,
        enabled=autoeq_enabled,
        mode=autoeq_mode,
        report_only=autoeq_report_only,
        config=autoeq_config,
    )
    corrective_eq_report = _apply_corrective_eq(stems, method=corrective_eq_method)
    compression_report = apply_contextual_compression(
        stems,
        style=compression_style,
        bpm=compression_bpm,
        report_only=compression_report_only or not compression_enabled,
        role_fn=_project_channel_role,
        primary_audio_fn=_stem_primary_loudness_audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
    ) if compression_enabled or compression_report_only else {
        "enabled": False,
        "applied": False,
        "stage": CONTEXTUAL_COMPRESSION_STAGE,
        "reason": "contextual_compression_disabled",
        "channels": [],
    }
    if panning_enabled or panning_report_only:
        panning_report = apply_musical_panning(
            stems,
            style=panning_style,
            report_only=panning_report_only or not panning_enabled,
            role_fn=_project_channel_role,
            primary_audio_fn=_stem_primary_loudness_audio,
            lufs_fn=_lufs,
            peak_fn=_peak_dbfs,
            band_energy_fn=_band_energy_for_rendered,
        )
    else:
        panning_report = {
            "enabled": False,
            "applied": False,
            "stage": MUSICAL_PANNING_STAGE,
            "reason": "musical_panning_disabled",
            "channel_pans": [],
            "pair_pans": [],
        }
    if output_balance_enabled or output_balance_report_only:
        output_balance_report = apply_musical_output_balance(
            stems,
            style=output_balance_style,
            report_only=output_balance_report_only or not output_balance_enabled,
            role_fn=_project_channel_role,
            primary_audio_fn=_stem_primary_loudness_audio,
            lufs_fn=_lufs,
            peak_fn=_peak_dbfs,
            band_energy_fn=_band_energy_for_rendered,
        )
    else:
        output_balance_report = {
            "enabled": False,
            "applied": False,
            "stage": MUSICAL_OUTPUT_BALANCE_STAGE,
            "reason": "musical_output_balance_disabled",
            "channel_offsets": [],
            "bus_offsets": [],
        }
    snare_top_bottom_report = apply_snare_top_bottom_balance(
        stems,
        primary_audio_fn=_stem_primary_loudness_audio,
        lufs_fn=_lufs,
        peak_fn=_peak_dbfs,
        report_only=output_balance_report_only or not output_balance_enabled,
    )
    if panning_report.get("enabled"):
        panorama_width_report = apply_musical_panorama_width(
            stems,
            report_only=panning_report_only or not panning_enabled,
            role_fn=_project_channel_role,
            lufs_fn=_lufs,
            peak_fn=_peak_dbfs,
        )
    else:
        panorama_width_report = {
            "enabled": False,
            "applied": False,
            "stage": MUSICAL_PANORAMA_WIDTH_STAGE,
            "reason": "musical_panning_disabled",
            "channel_width": [],
        }
    post_phase_channel_report: list[dict[str, Any]] = []
    pre_corrective_pair_report: list[dict[str, Any]] = []
    post_corrective_channel_report: list[dict[str, Any]] = []
    post_corrective_pair_report: list[dict[str, Any]] = []
    post_compression_channel_report = output_balance_report.get("channel_offsets", [])
    post_compression_pair_report = output_balance_report.get("bus_offsets", [])
    final_pair_report = post_compression_pair_report
    if skip_group_levels:
        group_report = [
            {
                "enabled": False,
                "reason": "skip_group_levels",
                "note": "BUS level correction is disabled; only channel/pair levels and master LUFS are applied.",
            }
        ]
    else:
        group_report = _apply_group_levels(stems)
    pre_master_mix = _sum_stems(stems)
    if skip_master_level:
        final_mix = pre_master_mix
        master_report = {
            "enabled": False,
            "reason": "skip_master_level",
            "loudness_method": _lufs_method(),
            "pre_master_lufs": round(_lufs(pre_master_mix, sample_rate), 3),
            "pre_master_peak_dbfs": round(_peak_dbfs(pre_master_mix), 3),
            "final_lufs": round(_lufs(final_mix, sample_rate), 3),
            "final_peak_dbfs": round(_peak_dbfs(final_mix), 3),
        }
    else:
        master_report, final_mix = _apply_master_level(
            pre_master_mix,
            sample_rate=sample_rate,
            target_lufs=master_target_lufs,
            ceiling_dbfs=master_ceiling_dbfs,
        )
    phase_stage_name = (
        "snare_bleed_phase_alignment"
        if phase_stage_report.get("snare_bleed_alignment", {}).get("enabled")
        else "global_bleed_phase_alignment"
    )

    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    wav_path = out / f"{output_name}.wav"
    report_path = out / f"{output_name}_report.json"
    _write_wav(wav_path, final_mix, sample_rate)
    mp3_path = _write_mp3(wav_path) if write_mp3 else None

    summary = {
        "run_id": f"ayaic_offline.{int(time.time() * 1000)}",
        "source_dir": str(Path(input_dir).expanduser()),
        "pipeline_stages": [
            "bleed_primary_signal_detection",
            ARRANGEMENT_INPUT_LEVEL_STAGE,
            phase_stage_name,
            *([SNARE_PAIR_COHERENCE_STAGE] if snare_pair_coherence_report.get("enabled") else []),
            "offline_hpf_lpf_correction",
            *(["autoeq_engine"] if autoeq_enabled else []),
            *(
                [f"corrective_eq_{corrective_eq_method}"]
                if corrective_eq_method != "none"
                else []
            ),
            *([CONTEXTUAL_COMPRESSION_STAGE] if compression_report.get("enabled") else []),
            *([MUSICAL_PANNING_STAGE] if panning_report.get("enabled") else []),
            *([MUSICAL_OUTPUT_BALANCE_STAGE] if output_balance_report.get("enabled") else []),
            *([SNARE_TOP_BOTTOM_BALANCE_STAGE] if snare_top_bottom_report.get("enabled") else []),
            *([MUSICAL_PANORAMA_WIDTH_STAGE] if panorama_width_report.get("enabled") else []),
            *(["ayayc_instrument_group_levels"] if not skip_group_levels else []),
            *(["ayaic_master_level"] if not skip_master_level else []),
        ],
        "outputs": {
            "wav": str(wav_path),
            "mp3": str(mp3_path) if mp3_path else None,
            "report": str(report_path),
        },
        "settings": {
            "master_target_lufs": float(master_target_lufs),
            "master_ceiling_dbfs": float(master_ceiling_dbfs) if master_ceiling_dbfs is not None else None,
            "max_phase_delay_ms": float(max_phase_delay_ms),
            "drums_only": bool(drums_only),
            "skip_group_levels": bool(skip_group_levels),
            "skip_master_level": bool(skip_master_level),
            "corrective_eq_method": corrective_eq_method,
            "corrective_eq_sequence": _corrective_eq_method_sequence(corrective_eq_method),
            "autoeq_enabled": bool(autoeq_enabled),
            "autoeq_mode": autoeq_mode,
            "autoeq_report_only": bool(autoeq_report_only),
            "contextual_compression_enabled": bool(compression_enabled),
            "contextual_compression_report_only": bool(compression_report_only),
            "contextual_compression_style": str(compression_style),
            "contextual_compression_bpm": (
                float(compression_bpm) if compression_bpm is not None else None
            ),
            "musical_panning_enabled": bool(panning_enabled),
            "musical_panning_report_only": bool(panning_report_only),
            "musical_panning_style": str(panning_style),
            "musical_output_balance_enabled": bool(output_balance_enabled),
            "musical_output_balance_report_only": bool(output_balance_report_only),
            "musical_output_balance_style": str(output_balance_style),
            "snare_pair_coherence_enabled": bool(snare_pair_coherence_report.get("enabled")),
            "snare_top_bottom_balance_enabled": bool(snare_top_bottom_report.get("enabled")),
            "musical_panorama_width_enabled": bool(panorama_width_report.get("enabled")),
            "loudness_method": _lufs_method(),
            "input_leveling_method": "arrangement_aware",
            "track_loudness_scope": "primary signal windows when bleed detector is available",
        },
        "bleed_detection": bleed_report,
        "primary_loudness_range_shift": primary_shift_report,
        "hpf_lpf_correction": hpf_lpf_report,
        "autoeq": autoeq_report,
        "autoeq_settings": autoeq_report.get("autoeq_settings", {}),
        "autoeq_channel_analysis": autoeq_report.get("autoeq_channel_analysis", []),
        "autoeq_decisions": autoeq_report.get("autoeq_decisions", []),
        "autoeq_rejected_decisions": autoeq_report.get("autoeq_rejected_decisions", []),
        "autoeq_group_decisions": autoeq_report.get("autoeq_group_decisions", []),
        "autoeq_master_decisions": autoeq_report.get("autoeq_master_decisions", []),
        "autoeq_validation_summary": autoeq_report.get("autoeq_validation_summary", {}),
        "autoeq_osc_commands": autoeq_report.get("autoeq_osc_commands", []),
        "corrective_eq": corrective_eq_report,
        "contextual_compression": compression_report,
        "compression": compression_report,
        "musical_panning": panning_report,
        "panning": panning_report,
        "musical_output_balance": output_balance_report,
        "output_balance": output_balance_report,
        "snare_pair_coherence": snare_pair_coherence_report,
        "snare_top_bottom_balance": snare_top_bottom_report,
        "musical_panorama_width": panorama_width_report,
        "panorama_width": panorama_width_report,
        "pre_phase_channel_levels": pre_phase_channel_report,
        "post_phase_channel_levels": post_phase_channel_report,
        "post_phase_filter_channel_levels": post_phase_channel_report,
        "pre_corrective_eq_pair_levels": pre_corrective_pair_report,
        "post_corrective_eq_channel_levels": post_corrective_channel_report,
        "post_corrective_eq_pair_levels": post_corrective_pair_report,
        "post_compression_channel_levels": post_compression_channel_report,
        "post_compression_pair_levels": post_compression_pair_report,
        "channel_levels": [
            *pre_phase_channel_report,
            *post_phase_channel_report,
            *post_corrective_channel_report,
            *post_compression_channel_report,
        ],
        "pair_levels": final_pair_report,
        "track_levels": [
            *pre_phase_channel_report,
            *post_phase_channel_report,
            *pre_corrective_pair_report,
            *post_corrective_channel_report,
            *post_corrective_pair_report,
            *post_compression_channel_report,
        ],
        "phase_delay_order": phase_stage_report["order"],
        "overhead_pair_alignment": phase_stage_report["overhead_pair_alignment"],
        "phase_alignment": phase_stage_report["phase_alignment"],
        "global_phase_alignment": phase_stage_report.get("global_phase_alignment", {}),
        "snare_bleed_alignment": phase_stage_report.get("snare_bleed_alignment", {}),
        "drum_pan_rule": phase_stage_report["drum_pan_rule"],
        "group_levels": group_report,
        "master_level": master_report,
        "stems": [_stem_report(stem) for stem in stems],
    }
    report_path.write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return AyaicPipelineResult(str(wav_path), str(mp3_path) if mp3_path else None, str(report_path), summary)


def _load_stems(input_dir: str | Path) -> list[AyaicStem]:
    import soundfile as sf

    root = Path(input_dir).expanduser()
    files = sorted(path for path in root.iterdir() if path.suffix.lower() in AUDIO_EXTENSIONS)
    stems: list[AyaicStem] = []
    for channel_id, path in enumerate(files, start=1):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        name = path.stem
        stems.append(
            AyaicStem(
                name=name,
                path=str(path),
                audio=ensure_stereo(audio),
                sample_rate=int(sample_rate),
                channel_id=channel_id,
                target_lufs=_track_target_lufs(name),
                group=_instrument_group(name),
                pan=_initial_pan(name),
            )
        )
    return stems


def _align_stems(stems: list[AyaicStem]) -> None:
    max_len = max(stem.audio.shape[0] for stem in stems)
    for stem in stems:
        if stem.audio.shape[0] < max_len:
            stem.audio = np.pad(stem.audio, ((0, max_len - stem.audio.shape[0]), (0, 0)))
        else:
            stem.audio = stem.audio[:max_len]


def _analyze_bleed_activity(stems: list[AyaicStem]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for stem in stems:
        instrument = _bleed_analysis_instrument(stem.name)
        mono = _mono(stem.audio)
        activity = _event_activity_ranges(mono, stem.sample_rate, instrument)
        ranges = list(activity.get("ranges") or []) if activity else []
        fallback_ranges: list[tuple[int, int]] = []
        if ranges:
            loudness_ranges = ranges
            mode = "event_based_primary_signal"
        elif activity:
            fallback_ranges = [_analysis_block_range(mono, stem.sample_rate)]
            loudness_ranges = fallback_ranges
            mode = "windowed_full_track_fallback"
        else:
            loudness_ranges = []
            mode = "full_track_no_bleed_detector"

        stem.primary_loudness_ranges = _merge_ranges(loudness_ranges, gap=0)
        active_samples = sum(end - start for start, end in stem.primary_loudness_ranges)
        if not stem.primary_loudness_ranges:
            active_samples = len(mono)
        bleed_metrics = _activity_bleed_metrics(mono, stem.sample_rate, activity)
        stem.bleed_analysis = {
            "enabled": instrument is not None,
            "instrument": instrument,
            "analysis_mode": mode,
            "analysis_active_sec": round(active_samples / stem.sample_rate, 3) if stem.sample_rate else 0.0,
            "analysis_active_ratio": round(active_samples / max(1, len(mono)), 4),
            "analysis_threshold_db": activity.get("threshold_db") if activity else None,
            "analysis_threshold_relaxed": bool(activity.get("threshold_relaxed")) if activity else False,
            "event_range_count": len(ranges),
            "primary_loudness_range_count": len(stem.primary_loudness_ranges),
            "primary_loudness_ranges_preview_sec": _ranges_preview_seconds(stem.primary_loudness_ranges, stem.sample_rate),
            **bleed_metrics,
        }
        report.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                **stem.bleed_analysis,
            }
        )
    return report


def _stem_primary_loudness_audio(stem: AyaicStem) -> np.ndarray:
    return _extract_ranges_audio(stem.audio, stem.primary_loudness_ranges)


def _combined_primary_loudness_audio(stems: list[AyaicStem]) -> tuple[np.ndarray, dict[str, Any]]:
    summed = _sum_audio([stem.audio for stem in stems])
    ranges = _merge_ranges(
        [
            (start, end)
            for stem in stems
            for start, end in stem.primary_loudness_ranges
        ],
        gap=0,
    )
    if not ranges:
        return summed, {"scope": "full_track_no_primary_ranges", "active_ratio": 1.0, "range_count": 0}
    active_samples = sum(end - start for start, end in ranges)
    return (
        _extract_ranges_audio(summed, ranges),
        {
            "scope": "primary_signal_union",
            "active_ratio": round(active_samples / max(1, summed.shape[0]), 4),
            "range_count": len(ranges),
        },
    )


def _extract_ranges_audio(audio: np.ndarray, ranges: list[tuple[int, int]]) -> np.ndarray:
    arr = ensure_stereo(audio)
    if not ranges:
        return arr
    segments = [arr[max(0, start):min(arr.shape[0], end)] for start, end in ranges if end > start]
    if not segments:
        return arr
    return np.concatenate(segments, axis=0).astype(np.float32)


def _apply_hpf_lpf_correction(stems: list[AyaicStem]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for stem in stems:
        plan = _offline_hpf_lpf_plan(stem.name)
        before_primary = _lufs(_stem_primary_loudness_audio(stem), stem.sample_rate)
        before_full = _lufs(stem.audio, stem.sample_rate)
        filtered = ensure_stereo(stem.audio).copy()
        for channel_index in range(filtered.shape[1]):
            lane = filtered[:, channel_index]
            if plan["hpf_hz"] > 0.0:
                lane = _highpass(lane, stem.sample_rate, plan["hpf_hz"])
            if plan["lpf_hz"] > 0.0:
                lane = _lowpass(lane, stem.sample_rate, plan["lpf_hz"])
            filtered[:, channel_index] = lane.astype(np.float32)
        stem.audio = filtered.astype(np.float32)
        after_primary = _lufs(_stem_primary_loudness_audio(stem), stem.sample_rate)
        after_full = _lufs(stem.audio, stem.sample_rate)
        report.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "instrument": plan["instrument"],
                "hpf_hz": round(plan["hpf_hz"], 3),
                "lpf_hz": round(plan["lpf_hz"], 3) if plan["lpf_hz"] > 0.0 else None,
                "source": OFFLINE_HPF_LPF_TARGET_SOURCE,
                "loudness_method": _lufs_method(),
                "pre_primary_lufs": round(before_primary, 3),
                "post_primary_lufs": round(after_primary, 3),
                "primary_loudness_delta_lu": round(after_primary - before_primary, 3),
                "pre_full_track_lufs": round(before_full, 3),
                "post_full_track_lufs": round(after_full, 3),
                "full_track_loudness_delta_lu": round(after_full - before_full, 3),
            }
        )
    return report


def _apply_autoeq_engine(
    stems: list[AyaicStem],
    *,
    enabled: bool,
    mode: str,
    report_only: bool,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    settings = AutoEQSettings.from_mapping(
        config or {},
        enabled=enabled,
        mode=mode,
        report_only=report_only,
    )
    tracks = [
        AutoEQTrack(
            channel=stem.channel_id,
            file=Path(stem.path).name,
            instrument=_autoeq_instrument(stem.name),
            audio=stem.audio,
            sample_rate=stem.sample_rate,
            group=stem.group,
        )
        for stem in stems
    ]
    master_audio = _sum_stems(stems) if stems else None
    result = AutoEQEngine(settings).run(tracks, master_audio=master_audio)
    if enabled and not report_only:
        for stem in stems:
            processed = result.processed_audio_by_channel.get(stem.channel_id)
            if processed is not None:
                stem.audio = ensure_stereo(processed).astype(np.float32)
    return result.report


def _autoeq_instrument(name: str) -> str:
    key = _name_key(name)
    if key in {"oh l", "overhead l"}:
        return "overhead_l"
    if key in {"oh r", "overhead r"}:
        return "overhead_r"
    if key in {"snare t", "snare top"}:
        return "snare_top"
    if key in {"snare b", "snare bottom"}:
        return "snare_bottom"
    if key in {"guitar l", "gtr l"}:
        return "guitar_l"
    if key in {"guitar r", "gtr r"}:
        return "guitar_r"
    if key in {"playback l"}:
        return "playback_l"
    if key in {"playback r", "playbacks r"}:
        return "playback_r"
    instrument = _offline_filter_instrument(name)
    aliases = {
        "bass_guitar": "bass",
        "electric_guitar": "guitar",
        "backing_vocal": "back_vocal",
        "lead_vocal": "lead_vocal",
        "rack_tom": "tom",
        "overhead": "overhead_pair",
    }
    return aliases.get(instrument, instrument)


def _apply_corrective_eq(stems: list[AyaicStem], *, method: str) -> dict[str, Any]:
    method = _normalize_corrective_eq_method(method)
    sequence = _corrective_eq_method_sequence(method)
    source = _corrective_eq_source(method)
    if method == "none":
        return {
            "enabled": False,
            "method": method,
            "reason": "corrective_eq_method_none",
            "available_methods": list(CORRECTIVE_EQ_METHODS),
        }
    if len(sequence) > 1:
        sample_rate = stems[0].sample_rate if stems else 48_000
        pre_mix_lufs = _lufs(_sum_stems(stems), sample_rate)
        stages = [_apply_corrective_eq(stems, method=child_method) for child_method in sequence]
        post_mix_lufs = _lufs(_sum_stems(stems), sample_rate)
        actions = [
            {**action, "chain_method": stage["method"]}
            for stage in stages
            for action in (stage.get("actions") or [])
        ]
        return {
            "enabled": True,
            "method": method,
            "method_sequence": sequence,
            "source": source,
            "loudness_method": _lufs_method(),
            "applied": any(bool(stage.get("applied")) for stage in stages),
            "applied_channel_count": len({int(action["channel"]) for action in actions if "channel" in action}),
            "applied_band_count": sum(int(stage.get("applied_band_count") or 0) for stage in stages),
            "pre_mix_lufs": round(pre_mix_lufs, 3),
            "post_mix_lufs": round(post_mix_lufs, 3),
            "mix_loudness_delta_lu": round(post_mix_lufs - pre_mix_lufs, 3),
            "stages": stages,
            "actions": actions,
        }

    sample_rate = stems[0].sample_rate if stems else 48_000
    pre_mix_lufs = _lufs(_sum_stems(stems), sample_rate)
    bands_by_channel, analysis = _corrective_eq_bands_by_channel(stems, method)
    actions: list[dict[str, Any]] = []

    for stem in stems:
        bands = bands_by_channel.get(stem.channel_id, [])
        if not bands:
            continue
        before_primary = _lufs(_stem_primary_loudness_audio(stem), stem.sample_rate)
        before_full = _lufs(stem.audio, stem.sample_rate)
        processed = ensure_stereo(stem.audio).copy()
        for band in bands:
            for channel_index in range(processed.shape[1]):
                processed[:, channel_index] = _peaking_eq(
                    processed[:, channel_index],
                    stem.sample_rate,
                    float(band["frequency_hz"]),
                    float(band["gain_db"]),
                    float(band["q"]),
                )
        stem.audio = processed.astype(np.float32)
        after_primary = _lufs(_stem_primary_loudness_audio(stem), stem.sample_rate)
        after_full = _lufs(stem.audio, stem.sample_rate)
        note = {
            "method": method,
            "source": source,
            "bands": [_round_eq_band(band) for band in bands],
            "pre_primary_lufs": round(before_primary, 3),
            "post_primary_lufs": round(after_primary, 3),
            "primary_loudness_delta_lu": round(after_primary - before_primary, 3),
            "pre_full_track_lufs": round(before_full, 3),
            "post_full_track_lufs": round(after_full, 3),
            "full_track_loudness_delta_lu": round(after_full - before_full, 3),
        }
        stem.corrective_eq_notes.append(note)
        actions.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "instrument": _corrective_eq_instrument(stem.name),
                **note,
            }
        )

    if method == "frequency_window":
        analysis["after"] = _frequency_window_snapshot(stems)["windows"]
    if method in {"project_corrective", "project_corrective_hybrid", "contextual_deep_eq"}:
        analysis["after"] = _project_corrective_snapshot(stems)

    return {
        "enabled": True,
        "method": method,
        "source": source,
        "loudness_method": _lufs_method(),
        "applied": bool(actions),
        "applied_channel_count": len(actions),
        "applied_band_count": sum(len(item["bands"]) for item in actions),
        "pre_mix_lufs": round(pre_mix_lufs, 3),
        "post_mix_lufs": round(_lufs(_sum_stems(stems), sample_rate), 3),
        "mix_loudness_delta_lu": round(_lufs(_sum_stems(stems), sample_rate) - pre_mix_lufs, 3),
        "analysis": analysis,
        "actions": actions,
    }


def _normalize_corrective_eq_method(method: str) -> str:
    key = (method or "none").strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "disabled": "none",
        "static": "profile",
        "classify_track": "profile",
        "instrument_profile": "profile",
        "codex_bleed_control": "bleed_control",
        "bleed": "bleed_control",
        "cross": "cross_adaptive",
        "crossadaptive": "cross_adaptive",
        "frequency": "frequency_window",
        "window": "frequency_window",
        "frequency_window_balance": "frequency_window",
        "431": "frequency_cross_profile",
        "4_3_1": "frequency_cross_profile",
        "frequency_window_cross_adaptive_profile": "frequency_cross_profile",
        "frequency_cross_adaptive_profile": "frequency_cross_profile",
        "frequency_cross_profile": "frequency_cross_profile",
        "project": "project_corrective",
        "project_corrective_eq": "project_corrective",
        "projecteq": "project_corrective",
        "hybrid_project": "project_corrective_hybrid",
        "project_hybrid": "project_corrective_hybrid",
        "project_corrective_hybrid_eq": "project_corrective_hybrid",
        "deep": "contextual_deep_eq",
        "deep_eq": "contextual_deep_eq",
        "contextual": "contextual_deep_eq",
        "contextual_eq": "contextual_deep_eq",
        "contextual_deep": "contextual_deep_eq",
        "arrangement_eq": "contextual_deep_eq",
    }
    key = aliases.get(key, key)
    if key not in CORRECTIVE_EQ_METHODS:
        raise ValueError(f"unknown corrective EQ method {method!r}; expected one of {', '.join(CORRECTIVE_EQ_METHODS)}")
    return key


def _corrective_eq_method_sequence(method: str) -> list[str]:
    method = _normalize_corrective_eq_method(method)
    if method == "frequency_cross_profile":
        return ["frequency_window", "cross_adaptive", "profile"]
    if method == "none":
        return []
    return [method]


def _corrective_eq_source(method: str) -> dict[str, Any] | None:
    if method == "profile":
        return OFFLINE_PROFILE_EQ_TARGET_SOURCE
    if method == "bleed_control":
        return OFFLINE_BLEED_CONTROL_EQ_TARGET_SOURCE
    if method == "cross_adaptive":
        return OFFLINE_CROSS_ADAPTIVE_EQ_TARGET_SOURCE
    if method == "frequency_window":
        return OFFLINE_FREQUENCY_WINDOW_EQ_TARGET_SOURCE
    if method == "frequency_cross_profile":
        return OFFLINE_FREQUENCY_CROSS_PROFILE_EQ_TARGET_SOURCE
    if method == "project_corrective":
        return PROJECT_CORRECTIVE_EQ_TARGET_SOURCE
    if method == "project_corrective_hybrid":
        return PROJECT_CORRECTIVE_HYBRID_EQ_TARGET_SOURCE
    if method == "contextual_deep_eq":
        return CONTEXTUAL_DEEP_EQ_TARGET_SOURCE
    return None


def _corrective_eq_bands_by_channel(
    stems: list[AyaicStem],
    method: str,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    if method == "profile":
        return _profile_corrective_eq_bands(stems)
    if method == "bleed_control":
        return _bleed_control_corrective_eq_bands(stems)
    if method == "cross_adaptive":
        return _cross_adaptive_corrective_eq_bands(stems)
    if method == "frequency_window":
        return _frequency_window_corrective_eq_bands(stems)
    if method == "project_corrective":
        return _project_corrective_eq_bands(stems)
    if method == "project_corrective_hybrid":
        return _project_corrective_hybrid_eq_bands(stems)
    if method == "contextual_deep_eq":
        return _contextual_deep_eq_bands(stems)
    return {}, {}


def _profile_corrective_eq_bands(stems: list[AyaicStem]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    bands_by_channel: dict[int, list[dict[str, Any]]] = {}
    for stem in stems:
        instrument = _corrective_eq_profile_key(stem.name)
        bands_by_channel[stem.channel_id] = [
            {
                "frequency_hz": freq,
                "gain_db": gain,
                "q": q,
                "reason": "offline classify_track instrument profile EQ",
            }
            for freq, gain, q in OFFLINE_PROFILE_CORRECTIVE_EQ_BANDS.get(
                instrument,
                OFFLINE_PROFILE_CORRECTIVE_EQ_BANDS["custom"],
            )
        ]
    return bands_by_channel, {"profile_source": "classify_track", "channel_count": len(bands_by_channel)}


def _bleed_control_corrective_eq_bands(stems: list[AyaicStem]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    bands_by_channel: dict[int, list[dict[str, Any]]] = {}
    for stem in stems:
        instrument = _corrective_eq_instrument(stem.name)
        bands: list[dict[str, Any]] = []
        if instrument in {"kick", "snare", "rack_tom", "floor_tom"}:
            bands.extend(
                [
                    {
                        "frequency_hz": 6500.0,
                        "gain_db": -2.0,
                        "q": 1.1,
                        "reason": "presence cut targets cymbal bleed in close drum mics",
                    },
                    {
                        "frequency_hz": 9500.0,
                        "gain_db": -3.0,
                        "q": 0.9,
                        "reason": "air-band cut reduces repeated cymbal spill across close drum mics",
                    },
                ]
            )
        elif instrument == "overhead":
            bands.append(
                {
                    "frequency_hz": 6500.0,
                    "gain_db": -1.5,
                    "q": 1.0,
                    "reason": "presence cut reduces harsh cymbal build-up",
                }
            )
        if bands:
            bands_by_channel[stem.channel_id] = bands
    return bands_by_channel, {
        "ported_changes": "append_eq_band only",
        "excluded_changes": ["fader_db", "compressor", "lpf"],
        "channel_count": len(bands_by_channel),
    }


def _cross_adaptive_corrective_eq_bands(stems: list[AyaicStem]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    try:
        from backend.cross_adaptive_eq import CrossAdaptiveEQ
    except Exception as exc:
        return {}, {"enabled": False, "reason": f"CrossAdaptiveEQ import failed: {exc}"}

    preview = {stem.channel_id: stem.audio for stem in stems}
    channel_band_energy = {
        stem.channel_id: _band_energy_for_rendered(stem.audio, stem.sample_rate)
        for stem in stems
    }
    channel_priorities = {
        stem.channel_id: _priority_for_instrument(_corrective_eq_instrument(stem.name))
        for stem in stems
    }
    stem_by_channel = {stem.channel_id: stem for stem in stems}
    processor = CrossAdaptiveEQ(
        min_band_level_db=-82.0,
        overlap_tolerance_db=8.0,
        max_cut_db=-2.5,
        max_boost_db=1.2,
    )
    raw_adjustments = processor.calculate_corrections(channel_band_energy, channel_priorities)
    aggregated: dict[tuple[int, float], dict[str, Any]] = {}
    skipped = 0

    for adjustment in raw_adjustments:
        channel = int(getattr(adjustment, "channel_id"))
        stem = stem_by_channel.get(channel)
        if stem is None:
            skipped += 1
            continue
        frequency = float(getattr(adjustment, "frequency_hz"))
        gain = float(getattr(adjustment, "gain_db"))
        q = float(getattr(adjustment, "q_factor"))
        instrument = _corrective_eq_instrument(stem.name)
        band = _cross_band_for_frequency(frequency)
        priority = channel_priorities.get(channel, 3)

        if band == "air":
            skipped += 1
            continue
        if instrument == "kick" and gain < 0.0 and band in {"sub", "bass", "low_mid"}:
            skipped += 1
            continue
        if gain < 0.0 and priority == 1:
            skipped += 1
            continue
        if gain > 0.0 and priority != 1:
            skipped += 1
            continue
        if gain > 0.0 and band in {"sub", "bass", "low_mid"}:
            skipped += 1
            continue

        if band in {"mid", "high_mid", "high"}:
            scale = 0.36 if gain < 0.0 else 0.32
        elif band in {"low_mid", "bass"}:
            scale = 0.24 if gain < 0.0 else 0.20
        else:
            scale = 0.20
        if instrument == "bass_guitar" and band in {"sub", "bass"} and gain < 0.0:
            scale = 0.36

        key = (channel, frequency)
        item = aggregated.setdefault(
            key,
            {
                "frequency_hz": frequency,
                "gain_db": 0.0,
                "q": q,
                "band": band,
                "priority": priority,
                "source_count": 0,
                "reason": "priority cross-adaptive anti-mask EQ",
            },
        )
        item["gain_db"] += gain * scale
        item["source_count"] += 1

    by_channel: dict[int, list[dict[str, Any]]] = {}
    applied_preview: list[dict[str, Any]] = []
    for (channel, _frequency), item in aggregated.items():
        priority = int(item["priority"])
        instrument = _corrective_eq_instrument(stem_by_channel[channel].name)
        gain = float(item["gain_db"])
        if gain < 0.0:
            if instrument == "bass_guitar" and item["band"] in {"sub", "bass"}:
                lower = -1.8
            else:
                lower = -2.5 if priority >= 4 else (-2.0 if priority == 3 else -1.0)
            gain = max(lower, gain)
        else:
            gain = min(0.9, gain)
        if abs(gain) < 0.35:
            skipped += 1
            continue
        item["gain_db"] = round(gain, 2)
        item["q"] = round(float(item["q"]), 2)
        by_channel.setdefault(channel, []).append(item)

    for channel, items in list(by_channel.items()):
        items.sort(
            key=lambda item: (
                0 if item["band"] in {"mid", "high_mid", "high"} else 1,
                -abs(float(item["gain_db"])),
            )
        )
        by_channel[channel] = items[:3]
        for item in by_channel[channel]:
            stem = stem_by_channel[channel]
            applied_preview.append(
                {
                    "channel": channel,
                    "file": Path(stem.path).name,
                    "instrument": _corrective_eq_instrument(stem.name),
                    **_round_eq_band(item),
                    "band": item["band"],
                    "priority": item["priority"],
                    "source_count": item["source_count"],
                }
            )

    applied_preview.sort(key=lambda item: (item["priority"], item["channel"], item["frequency_hz"]))
    return by_channel, {
        "enabled": True,
        "raw_adjustments": len(raw_adjustments),
        "applied_adjustments": len(applied_preview),
        "skipped_adjustments": skipped,
        "analysis_preview_sec": 18.0,
        "channel_priorities": {str(channel): priority for channel, priority in sorted(channel_priorities.items())},
        "applied": applied_preview,
        "notes": [
            "Lower numeric priority is protected first.",
            "Lead vocals keep highest EQ priority; lower-priority accompaniment receives most anti-mask cuts.",
        ],
    }


def _frequency_window_corrective_eq_bands(stems: list[AyaicStem]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    snapshot_before = _frequency_window_snapshot(stems)
    by_id = snapshot_before.get("by_id") or {}
    config_by_id = {item["id"]: item for item in FREQUENCY_WINDOW_DEFINITIONS}
    stem_by_channel = {stem.channel_id: stem for stem in stems}
    bands_by_channel: dict[int, list[dict[str, Any]]] = {}
    actions: list[dict[str, Any]] = []

    def _candidate_channels(window_id: str, *, instruments: set[str], min_share: float = 0.06) -> list[dict[str, Any]]:
        window = by_id.get(window_id) or {}
        return [
            item for item in (window.get("channels") or [])
            if item.get("instrument") in instruments and float(item.get("share", 0.0)) >= min_share
        ]

    def _apply_window_cut(window_id: str, *, channels: list[dict[str, Any]], base_cut_db: float, reason: str) -> None:
        if not channels:
            return
        config = config_by_id.get(window_id) or {}
        center_hz = float(config.get("center_hz", 1000.0))
        q = float(config.get("q", 1.2))
        strongest_share = max(float(channels[0].get("share", 0.0)), 1e-6)
        for item in channels[:3]:
            channel = int(item["channel"])
            stem = stem_by_channel.get(channel)
            if stem is None:
                continue
            scale = math.sqrt(max(float(item.get("share", 0.0)), 1e-6) / strongest_share)
            cut_db = float(np.clip(base_cut_db * scale, 0.25, base_cut_db))
            band = {
                "frequency_hz": center_hz,
                "gain_db": -cut_db,
                "q": q,
                "window": window_id,
                "share_before": float(item.get("share", 0.0)),
                "reason": reason,
            }
            bands_by_channel.setdefault(channel, []).append(band)
            actions.append(
                {
                    "type": "frequency_window_eq_cut",
                    "window": window_id,
                    "channel": channel,
                    "file": Path(stem.path).name,
                    "instrument": _corrective_eq_instrument(stem.name),
                    **_round_eq_band(band),
                    "share_before": round(float(item.get("share", 0.0)), 4),
                    "reason": reason,
                }
            )

    warmth = by_id.get("warmth_mud") or {}
    warmth_shares = warmth.get("family_shares") or {}
    warmth_music_share = float(warmth_shares.get("music", 0.0) + warmth_shares.get("backing_vocal", 0.0))
    warmth_lead_share = float(warmth_shares.get("lead_vocal", 0.0))
    if warmth_music_share > 0.58 and warmth_lead_share < 0.24:
        warmth_cut_db = float(np.clip(0.35 + (warmth_music_share - max(warmth_lead_share, 0.16)) * 0.85, 0.35, 0.95))
        _apply_window_cut(
            "warmth_mud",
            channels=_candidate_channels("warmth_mud", instruments=WINDOW_SPACE_COMPETITOR_INSTRUMENTS),
            base_cut_db=warmth_cut_db,
            reason="120-500 Hz window is crowded by accompaniment body; broad low-mid cleanup restores depth without thinning the full mix.",
        )

    vocal_conflict = by_id.get("vocal_conflict") or {}
    vocal_shares = vocal_conflict.get("family_shares") or {}
    vocal_competition_share = float(vocal_shares.get("music", 0.0) + vocal_shares.get("backing_vocal", 0.0))
    vocal_lead_share = float(vocal_shares.get("lead_vocal", 0.0))
    vocal_window_advantage = vocal_competition_share - vocal_lead_share
    if (
        vocal_competition_share > 0.54
        and (
            vocal_lead_share < 0.26
            or str(vocal_conflict.get("dominant_family") or "") != "lead_vocal"
            or vocal_window_advantage > 0.045
        )
    ):
        vocal_cut_db = float(np.clip(0.4 + max(vocal_window_advantage, 0.0) * 1.55, 0.4, 1.2))
        _apply_window_cut(
            "vocal_conflict",
            channels=_candidate_channels("vocal_conflict", instruments=WINDOW_SPACE_COMPETITOR_INSTRUMENTS),
            base_cut_db=vocal_cut_db,
            reason="700-1500 Hz window hides the lead behind accompaniment; this carve clears the speaking range instead of just raising the vocal.",
        )

    presence = by_id.get("presence_harshness") or {}
    presence_shares = presence.get("family_shares") or {}
    presence_competition_share = float(presence_shares.get("music", 0.0) + presence_shares.get("backing_vocal", 0.0))
    presence_lead_share = float(presence_shares.get("lead_vocal", 0.0))
    if presence_competition_share > 0.52 and presence_lead_share < 0.24:
        presence_cut_db = float(np.clip(0.35 + (presence_competition_share - max(presence_lead_share, 0.18)) * 0.95, 0.35, 1.0))
        _apply_window_cut(
            "presence_harshness",
            channels=_candidate_channels("presence_harshness", instruments=WINDOW_SPACE_COMPETITOR_INSTRUMENTS, min_share=0.05),
            base_cut_db=presence_cut_db,
            reason="1.5-6 kHz window is overfilled by accompaniment presence; broad presence carving helps lyric intelligibility without global brightening.",
        )

    air = by_id.get("air_sibilance") or {}
    air_shares = air.get("family_shares") or {}
    cymbal_share = float(air_shares.get("cymbals", 0.0))
    if cymbal_share > 0.30:
        air_cut_db = float(np.clip(0.35 + (cymbal_share - 0.30) * 2.0, 0.35, 1.15))
        _apply_window_cut(
            "air_sibilance",
            channels=_candidate_channels("air_sibilance", instruments=CYMBAL_INSTRUMENTS, min_share=0.05),
            base_cut_db=air_cut_db,
            reason="6-16 kHz window is dominated by cymbal wash; a narrow upper-air trim preserves sparkle while reducing constant hiss.",
        )

    return bands_by_channel, {
        "enabled": True,
        "applied": bool(actions),
        "actions": actions,
        "notes": [
            "Frequency-window balancing analyses broad musical windows rather than chasing narrow resonances.",
            "Low end stays report-only because kick/bass hierarchy has its own measured level pass.",
        ],
        "before": snapshot_before["windows"],
    }


def _project_corrective_eq_bands(stems: list[AyaicStem]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    snapshot_before = _project_corrective_snapshot(stems)
    channel_diagnostics = {
        int(item["channel"]): item
        for item in snapshot_before["channel_diagnostics"]
    }
    stem_by_channel = {stem.channel_id: stem for stem in stems}
    bands_by_channel: dict[int, list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    def add_band(
        stem: AyaicStem,
        *,
        frequency_hz: float,
        gain_db: float,
        q: float,
        reason: str,
        confidence: float,
        source: str,
        target: str,
        metric: str,
    ) -> None:
        diagnostic = channel_diagnostics.get(stem.channel_id, {})
        _project_add_eq_band(
            bands_by_channel,
            rejected,
            stem,
            diagnostic=diagnostic,
            frequency_hz=frequency_hz,
            gain_db=gain_db,
            q=q,
            reason=reason,
            confidence=confidence,
            source=source,
            target=target,
            metric=metric,
        )

    lead_vocals = [
        stem for stem in stems
        if channel_diagnostics.get(stem.channel_id, {}).get("role") == "lead_vocal"
    ]
    music_bed = [
        stem for stem in stems
        if channel_diagnostics.get(stem.channel_id, {}).get("role") in {"guitar", "accordion", "keys", "playback", "back_vocal"}
    ]
    if lead_vocals:
        vocal_presence = max(float(channel_diagnostics[stem.channel_id]["presence_db"]) for stem in lead_vocals)
        vocal_mid = max(float(channel_diagnostics[stem.channel_id]["mid_db"]) for stem in lead_vocals)
        vocal_name = ", ".join(Path(stem.path).name for stem in lead_vocals[:2])
        for stem in music_bed:
            diagnostic = channel_diagnostics.get(stem.channel_id, {})
            role = str(diagnostic.get("role") or "")
            presence_db = float(diagnostic.get("presence_db", -100.0))
            mid_db = float(diagnostic.get("mid_db", -100.0))
            presence_margin = presence_db - (vocal_presence - 9.0)
            if presence_margin > 0.0:
                cut_db = -(0.55 + min(1.25, presence_margin * 0.12))
                confidence = float(np.clip(0.66 + presence_margin * 0.025, 0.66, 0.9))
                add_band(
                    stem,
                    frequency_hz=3100.0,
                    gain_db=cut_db,
                    q=1.25 if role != "back_vocal" else 1.05,
                    reason=f"{role} presence overlaps lead vocal clarity band ({vocal_name}); carve the masker instead of boosting vocal",
                    confidence=confidence,
                    source="lead_vocal_masking",
                    target="2.5-4 kHz vocal intelligibility space",
                    metric=f"masker_presence_db={presence_db:.2f}, vocal_presence_db={vocal_presence:.2f}",
                )
                conflicts.append(
                    {
                        "type": "lead_vocal_presence_masking",
                        "masked_source": vocal_name,
                        "masker_source": Path(stem.path).name,
                        "suggested_band_hz": [2500.0, 4000.0],
                        "suggested_action": "static bell cut on masker",
                        "confidence": round(confidence, 3),
                        "reason": "masker presence is too close to lead vocal presence after channel Ayaic",
                    }
                )
            mid_margin = mid_db - (vocal_mid - 7.0)
            if mid_margin > 0.0 and role != "back_vocal":
                cut_db = -(0.45 + min(0.85, mid_margin * 0.09))
                confidence = float(np.clip(0.62 + mid_margin * 0.02, 0.62, 0.82))
                add_band(
                    stem,
                    frequency_hz=1100.0,
                    gain_db=cut_db,
                    q=1.15,
                    reason=f"{role} core mids crowd the lead vocal speaking range; create space before master leveling",
                    confidence=confidence,
                    source="lead_vocal_mid_masking",
                    target="700-1500 Hz vocal body space",
                    metric=f"masker_mid_db={mid_db:.2f}, vocal_mid_db={vocal_mid:.2f}",
                )

    kick_stems = [
        stem for stem in stems
        if channel_diagnostics.get(stem.channel_id, {}).get("role") == "kick"
    ]
    bass_stems = [
        stem for stem in stems
        if channel_diagnostics.get(stem.channel_id, {}).get("role") == "bass"
    ]
    if kick_stems and bass_stems:
        kick_low = max(float(channel_diagnostics[stem.channel_id]["low_end_db"]) for stem in kick_stems)
        for bass_stem in bass_stems:
            bass_diag = channel_diagnostics[bass_stem.channel_id]
            bass_low = float(bass_diag["low_end_db"])
            overlap = bass_low - (kick_low - 6.0)
            if overlap > 0.0:
                cut_db = -(0.55 + min(0.95, overlap * 0.08))
                confidence = float(np.clip(0.64 + overlap * 0.018, 0.64, 0.82))
                add_band(
                    bass_stem,
                    frequency_hz=85.0,
                    gain_db=cut_db,
                    q=0.95,
                    reason="bass low-end overlaps the kick punch zone; keep kick transient readable while preserving bass sustain",
                    confidence=confidence,
                    source="kick_bass_low_end_separation",
                    target="50-100 Hz kick/bass separation",
                    metric=f"bass_low_end_db={bass_low:.2f}, kick_low_end_db={kick_low:.2f}",
                )
                conflicts.append(
                    {
                        "type": "kick_bass_low_end_conflict",
                        "masked_source": "kick",
                        "masker_source": Path(bass_stem.path).name,
                        "suggested_band_hz": [50.0, 100.0],
                        "suggested_action": "small bass cut in kick punch zone",
                        "confidence": round(confidence, 3),
                        "reason": "bass energy is close to or above the kick low-end reference",
                    }
                )

    overheads = [
        stem for stem in stems
        if channel_diagnostics.get(stem.channel_id, {}).get("role") == "overhead"
    ]
    if overheads:
        overhead_high = max(float(channel_diagnostics[stem.channel_id]["high_db"]) for stem in overheads)
        overhead_mid = max(float(channel_diagnostics[stem.channel_id]["mid_db"]) for stem in overheads)
        overhead_low_mid = max(float(channel_diagnostics[stem.channel_id]["low_mid_db"]) for stem in overheads)
        harsh_margin = overhead_high - (overhead_mid + 2.5)
        wash_margin = overhead_low_mid - (overhead_mid - 5.0)
        if harsh_margin > 0.0:
            cut_db = -(0.45 + min(0.85, harsh_margin * 0.09))
            confidence = float(np.clip(0.62 + harsh_margin * 0.018, 0.62, 0.82))
            for stem in overheads:
                add_band(
                    stem,
                    frequency_hz=7600.0,
                    gain_db=cut_db,
                    q=1.05,
                    reason="linked overhead harshness trim keeps cymbal top controlled without changing stereo image",
                    confidence=confidence,
                    source="linked_overhead_harshness",
                    target="5-9 kHz cymbal harshness",
                    metric=f"overhead_high_db={overhead_high:.2f}, overhead_mid_db={overhead_mid:.2f}",
                )
        if wash_margin > 0.0:
            cut_db = -(0.35 + min(0.75, wash_margin * 0.08))
            confidence = float(np.clip(0.6 + wash_margin * 0.016, 0.6, 0.78))
            for stem in overheads:
                add_band(
                    stem,
                    frequency_hz=360.0,
                    gain_db=cut_db,
                    q=1.0,
                    reason="linked overhead low-mid wash trim reduces cymbal/body buildup while preserving image",
                    confidence=confidence,
                    source="linked_overhead_wash",
                    target="250-500 Hz overhead wash",
                    metric=f"overhead_low_mid_db={overhead_low_mid:.2f}, overhead_mid_db={overhead_mid:.2f}",
                )

    for stem in stems:
        diagnostic = channel_diagnostics.get(stem.channel_id, {})
        role = str(diagnostic.get("role") or "music")
        low_mid_db = float(diagnostic.get("low_mid_db", -100.0))
        mid_db = float(diagnostic.get("mid_db", -100.0))
        presence_db = float(diagnostic.get("presence_db", -100.0))
        high_db = float(diagnostic.get("high_db", -100.0))
        air_db = float(diagnostic.get("air_db", -100.0))
        low_end_db = float(diagnostic.get("low_end_db", -100.0))

        if role == "kick":
            mud_margin = low_mid_db - (low_end_db - 8.0)
            if mud_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=285.0,
                    gain_db=-(0.5 + min(1.2, mud_margin * 0.08)),
                    q=1.1,
                    reason="kick low-mid body is high after cleanup; trim boxiness without reducing fundamental",
                    confidence=float(np.clip(0.62 + mud_margin * 0.02, 0.62, 0.82)),
                    source="kick_role_cleanup",
                    target="220-350 Hz kick boxiness",
                    metric=f"low_mid_db={low_mid_db:.2f}, low_end_db={low_end_db:.2f}",
                )
        elif role in {"snare_top", "snare_bottom", "snare"}:
            box_margin = max(low_mid_db, mid_db) - (presence_db - 5.0)
            if box_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=780.0 if role != "snare_bottom" else 900.0,
                    gain_db=-(0.55 + min(1.25, box_margin * 0.07)),
                    q=1.35,
                    reason=f"{role} boxiness is high relative to attack; trim body resonance cautiously",
                    confidence=float(np.clip(0.64 + box_margin * 0.018, 0.64, 0.84)),
                    source="snare_role_cleanup",
                    target="600-950 Hz snare boxiness",
                    metric=f"box_margin_db={box_margin:.2f}",
                )
            harsh_margin = presence_db - (mid_db + 4.0)
            if harsh_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=4200.0,
                    gain_db=-(0.4 + min(0.9, harsh_margin * 0.06)),
                    q=1.2,
                    reason=f"{role} attack/bleed is sharp; small presence trim before final Ayaic",
                    confidence=float(np.clip(0.6 + harsh_margin * 0.015, 0.6, 0.78)),
                    source="snare_harshness_control",
                    target="2-5 kHz snare harshness",
                    metric=f"presence_db={presence_db:.2f}, mid_db={mid_db:.2f}",
                )
        elif role in {"tom", "floor_tom"}:
            mud_margin = low_mid_db - (low_end_db - 7.0)
            if mud_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=340.0 if role == "tom" else 300.0,
                    gain_db=-(0.55 + min(1.15, mud_margin * 0.08)),
                    q=1.15,
                    reason=f"{role} low-mid buildup masks drum tone; trim mud while keeping fundamental",
                    confidence=float(np.clip(0.62 + mud_margin * 0.018, 0.62, 0.82)),
                    source="tom_role_cleanup",
                    target="250-400 Hz tom mud",
                    metric=f"low_mid_db={low_mid_db:.2f}, low_end_db={low_end_db:.2f}",
                )
            if high_db > mid_db + 4.0:
                add_band(
                    stem,
                    frequency_hz=6500.0,
                    gain_db=-0.75,
                    q=1.0,
                    reason=f"{role} close mic carries cymbal spill; light high trim keeps tom hits cleaner",
                    confidence=0.66,
                    source="tom_bleed_control",
                    target="5-8 kHz cymbal spill",
                    metric=f"high_db={high_db:.2f}, mid_db={mid_db:.2f}",
                )
        elif role in {"guitar", "accordion", "keys", "playback"}:
            mud_margin = low_mid_db - (mid_db - 3.5)
            if mud_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=320.0,
                    gain_db=-(0.45 + min(1.0, mud_margin * 0.08)),
                    q=1.05,
                    reason=f"{role} low-mid buildup crowds the arrangement; broad cleanup before final channel/pair Ayaic",
                    confidence=float(np.clip(0.6 + mud_margin * 0.018, 0.6, 0.82)),
                    source="music_bed_mud_control",
                    target="250-500 Hz arrangement mud",
                    metric=f"low_mid_db={low_mid_db:.2f}, mid_db={mid_db:.2f}",
                )
            fizz_margin = high_db - (presence_db + 3.5)
            if fizz_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=6500.0,
                    gain_db=-(0.35 + min(0.85, fizz_margin * 0.07)),
                    q=1.0,
                    reason=f"{role} upper band is bright compared with presence; reduce fizz instead of boosting vocals",
                    confidence=float(np.clip(0.58 + fizz_margin * 0.018, 0.58, 0.78)),
                    source="music_bed_fizz_control",
                    target="5-8 kHz accompaniment fizz",
                    metric=f"high_db={high_db:.2f}, presence_db={presence_db:.2f}",
                )
        elif role in {"lead_vocal", "back_vocal"}:
            mud_margin = low_mid_db - (mid_db - 5.0)
            if mud_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=260.0,
                    gain_db=-(0.45 + min(0.85, mud_margin * 0.07)),
                    q=1.05,
                    reason=f"{role} low-mid buildup reduces clarity; trim mud without adding presence boost",
                    confidence=float(np.clip(0.62 + mud_margin * 0.016, 0.62, 0.8)),
                    source="vocal_mud_control",
                    target="180-350 Hz vocal mud",
                    metric=f"low_mid_db={low_mid_db:.2f}, mid_db={mid_db:.2f}",
                )
            sibilance_margin = max(high_db, air_db) - (presence_db + 2.5)
            if sibilance_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=7300.0,
                    gain_db=-(0.35 + min(0.9, sibilance_margin * 0.07)),
                    q=1.15,
                    reason=f"{role} sibilance/air is high relative to presence; controlled de-ess style trim",
                    confidence=float(np.clip(0.6 + sibilance_margin * 0.018, 0.6, 0.8)),
                    source="vocal_sibilance_control",
                    target="5-9 kHz vocal sibilance",
                    metric=f"high_air_db={max(high_db, air_db):.2f}, presence_db={presence_db:.2f}",
                )

    actions = []
    for channel, bands in sorted(bands_by_channel.items()):
        stem = stem_by_channel.get(channel)
        if stem is None:
            continue
        actions.extend(
            {
                "channel": channel,
                "file": Path(stem.path).name,
                "instrument": _autoeq_instrument(stem.name),
                **_round_eq_band(band),
            }
            for band in bands
        )

    return bands_by_channel, {
        "enabled": True,
        "method": "project_corrective",
        "before": snapshot_before,
        "channel_diagnostics": snapshot_before["channel_diagnostics"],
        "group_diagnostics": snapshot_before["group_diagnostics"],
        "detected_conflicts": conflicts,
        "rejected_candidates": rejected,
        "actions_preview": actions,
        "limits": {
            "max_filters_per_channel": 5,
            "min_confidence_to_apply": 0.55,
            "cuts_preferred_over_boosts": True,
            "boosts_used": False,
            "overhead_lr_policy": "linked stereo EQ for overhead pair decisions",
        },
        "notes": [
            "Project Corrective EQ is role-aware and arrangement-aware, not a flat-spectrum matcher.",
            "Primary-signal windows are used so detected bleed does not drive channel EQ.",
            "Lead vocal masking is handled by cutting lower-priority maskers before any vocal boost.",
            "Final channel/pair Ayaic levels are expected immediately after this block.",
        ],
    }


def _project_corrective_hybrid_eq_bands(stems: list[AyaicStem]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    project_bands, project_analysis = _project_corrective_eq_bands(stems)
    frequency_bands, frequency_analysis = _frequency_window_corrective_eq_bands(stems)
    bleed_bands, bleed_analysis = _bleed_control_corrective_eq_bands(stems)
    snapshot_before = project_analysis.get("before") or _project_corrective_snapshot(stems)
    channel_diagnostics = {
        int(item["channel"]): item
        for item in snapshot_before["channel_diagnostics"]
    }
    stem_by_channel = {stem.channel_id: stem for stem in stems}
    bands_by_channel: dict[int, list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    evidence_count = {
        "project_corrective": 0,
        "frequency_window_diagnostic": 0,
        "selective_drum_bleed_control": 0,
    }

    def add_hybrid_band(
        stem: AyaicStem,
        band: dict[str, Any],
        *,
        component: str,
        gain_scale: float = 1.0,
        max_cut_db: float | None = None,
        reason_prefix: str = "",
    ) -> None:
        diagnostic = channel_diagnostics.get(stem.channel_id, {})
        role = str(diagnostic.get("role") or _project_channel_role(stem.name))
        gain = min(0.0, float(band.get("gain_db", 0.0)) * gain_scale)
        if max_cut_db is not None:
            gain = max(gain, -abs(float(max_cut_db)))
        source = str(band.get("source") or band.get("window") or component)
        reason = str(band.get("reason") or "")
        if reason_prefix:
            reason = f"{reason_prefix}: {reason}"
        accepted = _project_hybrid_add_eq_band(
            bands_by_channel,
            rejected,
            stem,
            diagnostic=diagnostic,
            frequency_hz=float(band["frequency_hz"]),
            gain_db=gain,
            q=float(band.get("q", 1.0)),
            reason=reason,
            confidence=float(band.get("confidence", 0.72)),
            component=component,
            source=source,
            target=str(band.get("target") or band.get("window") or "hybrid corrective EQ"),
            metric=str(band.get("metric") or ""),
        )
        if accepted:
            evidence_count[component] = evidence_count.get(component, 0) + 1

    for channel, bands in sorted(project_bands.items()):
        stem = stem_by_channel.get(channel)
        if stem is None:
            continue
        for band in bands:
            source = str(band.get("source") or "")
            role = str(channel_diagnostics.get(channel, {}).get("role") or _project_channel_role(stem.name))
            max_cut = 1.5
            if source == "lead_vocal_masking":
                max_cut = 1.35
            elif source == "lead_vocal_mid_masking":
                max_cut = 0.85
            elif role == "overhead":
                max_cut = 1.1
            add_hybrid_band(
                stem,
                band,
                component="project_corrective",
                max_cut_db=max_cut,
                reason_prefix="hybrid primary decision",
            )

    frequency_window_limits = {
        "warmth_mud": 0.6,
        "vocal_conflict": 0.55,
        "presence_harshness": 0.65,
        "air_sibilance": 0.55,
    }
    for channel, bands in sorted(frequency_bands.items()):
        stem = stem_by_channel.get(channel)
        if stem is None:
            continue
        for band in bands:
            window = str(band.get("window") or "")
            if window not in frequency_window_limits:
                continue
            add_hybrid_band(
                stem,
                {
                    **band,
                    "confidence": 0.62,
                    "source": "frequency_window_diagnostic",
                    "target": f"{window} broad arrangement window",
                },
                component="frequency_window_diagnostic",
                gain_scale=0.45,
                max_cut_db=frequency_window_limits[window],
                reason_prefix="hybrid broad-window evidence",
            )

    for channel, bands in sorted(bleed_bands.items()):
        stem = stem_by_channel.get(channel)
        if stem is None:
            continue
        diagnostic = channel_diagnostics.get(channel, {})
        role = str(diagnostic.get("role") or _project_channel_role(stem.name))
        if role not in {"kick", "snare", "snare_top", "snare_bottom", "tom", "floor_tom"}:
            continue
        for band in bands:
            frequency = float(band.get("frequency_hz", 0.0))
            if frequency < 5500.0:
                continue
            add_hybrid_band(
                stem,
                {
                    **band,
                    "confidence": 0.64,
                    "source": "selective_drum_bleed_control",
                    "target": "close drum cymbal bleed control",
                },
                component="selective_drum_bleed_control",
                gain_scale=0.32,
                max_cut_db=0.85,
                reason_prefix="hybrid selective close-drum bleed trim",
            )

    for channel, bands in list(bands_by_channel.items()):
        bands.sort(key=lambda item: (0 if abs(float(item["frequency_hz"]) - 3100.0) < 700.0 else 1, -abs(float(item["gain_db"]))))
        role = str(channel_diagnostics.get(channel, {}).get("role") or "")
        limit = 3 if role == "overhead" else 4
        if len(bands) > limit:
            for band in bands[limit:]:
                rejected.append(
                    {
                        "channel": channel,
                        "file": Path(stem_by_channel[channel].path).name,
                        "frequency_hz": round(float(band["frequency_hz"]), 2),
                        "gain_db": round(float(band["gain_db"]), 2),
                        "reason": "hybrid_post_dedup_filter_limit",
                        "component": band.get("component"),
                        "source": band.get("source"),
                    }
                )
            bands_by_channel[channel] = bands[:limit]

    actions = []
    for channel, bands in sorted(bands_by_channel.items()):
        stem = stem_by_channel.get(channel)
        if stem is None:
            continue
        actions.extend(
            {
                "channel": channel,
                "file": Path(stem.path).name,
                "instrument": _autoeq_instrument(stem.name),
                **_round_eq_band(band),
            }
            for band in bands
        )

    dynamic_candidates = _project_hybrid_dynamic_candidates(stems, project_analysis, channel_diagnostics)

    return bands_by_channel, {
        "enabled": True,
        "method": "project_corrective_hybrid",
        "before": snapshot_before,
        "channel_diagnostics": snapshot_before["channel_diagnostics"],
        "group_diagnostics": snapshot_before["group_diagnostics"],
        "detected_conflicts": project_analysis.get("detected_conflicts", []),
        "dynamic_candidates_report_only": dynamic_candidates,
        "rejected_candidates": rejected,
        "actions_preview": actions,
        "components": {
            "project_corrective": {
                "candidate_channel_count": len(project_bands),
                "candidate_band_count": sum(len(bands) for bands in project_bands.values()),
                "accepted_evidence_count": evidence_count["project_corrective"],
            },
            "frequency_window_diagnostic": {
                "candidate_channel_count": len(frequency_bands),
                "candidate_band_count": sum(len(bands) for bands in frequency_bands.values()),
                "accepted_evidence_count": evidence_count["frequency_window_diagnostic"],
                "analysis": frequency_analysis.get("before", []),
            },
            "selective_drum_bleed_control": {
                "candidate_channel_count": len(bleed_bands),
                "candidate_band_count": sum(len(bands) for bands in bleed_bands.values()),
                "accepted_evidence_count": evidence_count["selective_drum_bleed_control"],
                "excluded_changes": bleed_analysis.get("excluded_changes", []),
            },
        },
        "limits": {
            "max_filters_per_channel": 4,
            "max_filters_per_overhead_channel": 3,
            "max_project_vocal_masking_cut_db": -1.35,
            "max_frequency_window_assist_cut_db": -0.65,
            "max_selective_bleed_cut_db": -0.85,
            "boosts_used": False,
            "dynamic_eq_applied": False,
            "dynamic_eq_mode": "report_only",
        },
        "notes": [
            "Hybrid method mixes decision evidence, not serial EQ processing.",
            "Project Corrective EQ remains the primary source of channel decisions.",
            "Frequency-window and bleed-control candidates are downscaled and deduplicated.",
            "Dynamic EQ candidates are written for review but not applied in this offline render.",
        ],
    }


def _contextual_deep_eq_bands(stems: list[AyaicStem]) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    snapshot_before = _project_corrective_snapshot(stems)
    channel_diagnostics = {
        int(item["channel"]): item
        for item in snapshot_before["channel_diagnostics"]
    }
    stem_by_channel = {stem.channel_id: stem for stem in stems}
    bands_by_channel: dict[int, list[dict[str, Any]]] = {}
    rejected: list[dict[str, Any]] = []
    critic_notes: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    lead_vocal_channels = {
        ch
        for ch, item in channel_diagnostics.items()
        if item.get("role") == "lead_vocal"
    }
    lead_presence = max(
        (float(channel_diagnostics[ch].get("presence_db", -100.0)) for ch in lead_vocal_channels),
        default=-100.0,
    )
    lead_mid = max(
        (float(channel_diagnostics[ch].get("mid_db", -100.0)) for ch in lead_vocal_channels),
        default=-100.0,
    )

    def add_band(
        stem: AyaicStem,
        *,
        frequency_hz: float,
        gain_db: float,
        q: float,
        reason: str,
        confidence: float,
        source: str,
        target: str,
        metric: str,
        allow_boost: bool = True,
    ) -> None:
        diagnostic = channel_diagnostics.get(stem.channel_id, {})
        _contextual_add_eq_band(
            bands_by_channel,
            rejected,
            critic_notes,
            stem,
            diagnostic=diagnostic,
            frequency_hz=frequency_hz,
            gain_db=gain_db,
            q=q,
            reason=reason,
            confidence=confidence,
            source=source,
            target=target,
            metric=metric,
            allow_boost=allow_boost,
        )

    for stem in stems:
        diagnostic = channel_diagnostics.get(stem.channel_id, {})
        role = str(diagnostic.get("role") or _project_channel_role(stem.name))
        low_end_db = float(diagnostic.get("low_end_db", -100.0))
        low_mid_db = float(diagnostic.get("low_mid_db", -100.0))
        mid_db = float(diagnostic.get("mid_db", -100.0))
        presence_db = float(diagnostic.get("presence_db", -100.0))
        high_db = float(diagnostic.get("high_db", -100.0))
        air_db = float(diagnostic.get("air_db", -100.0))
        primary_lufs = float(diagnostic.get("primary_lufs", -120.0))

        if role in {"guitar", "accordion", "keys", "playback", "back_vocal"} and lead_vocal_channels:
            presence_margin = presence_db - (lead_presence - 10.0)
            if presence_margin > 0.0:
                gain = -(1.2 + min(5.2, presence_margin * 0.28))
                add_band(
                    stem,
                    frequency_hz=3100.0,
                    gain_db=gain,
                    q=1.25 if role != "back_vocal" else 1.05,
                    reason=f"{role} masks lead vocal presence; carve foreground space instead of raising the vocal first",
                    confidence=float(np.clip(0.68 + presence_margin * 0.025, 0.68, 0.96)),
                    source="contextual_lead_vocal_masking",
                    target="1.5-4 kHz lead vocal intelligibility",
                    metric=f"masker_presence_db={presence_db:.2f}, lead_presence_db={lead_presence:.2f}, margin_db={presence_margin:.2f}",
                )
                conflicts.append(
                    {
                        "type": "lead_vocal_presence_masking",
                        "masked_channel": sorted(lead_vocal_channels),
                        "masker_channel": stem.channel_id,
                        "masker_file": Path(stem.path).name,
                        "margin_db": round(presence_margin, 3),
                    }
                )
            mid_margin = mid_db - (lead_mid - 7.0)
            if mid_margin > 0.0 and role != "back_vocal":
                add_band(
                    stem,
                    frequency_hz=1050.0,
                    gain_db=-(0.9 + min(3.4, mid_margin * 0.22)),
                    q=1.05,
                    reason=f"{role} crowds vocal body/core mids; broad cut improves lead readability",
                    confidence=float(np.clip(0.62 + mid_margin * 0.02, 0.62, 0.9)),
                    source="contextual_vocal_body_masking",
                    target="700-1500 Hz vocal body space",
                    metric=f"masker_mid_db={mid_db:.2f}, lead_mid_db={lead_mid:.2f}, margin_db={mid_margin:.2f}",
                )

        if role == "lead_vocal":
            mud_margin = low_mid_db - (mid_db - 4.0)
            if mud_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=260.0,
                    gain_db=-(0.9 + min(4.2, mud_margin * 0.24)),
                    q=1.0,
                    reason="lead vocal low-mid buildup reduces intelligibility; clean mud before presence shaping",
                    confidence=float(np.clip(0.66 + mud_margin * 0.018, 0.66, 0.9)),
                    source="contextual_vocal_mud",
                    target="180-350 Hz vocal mud",
                    metric=f"low_mid_db={low_mid_db:.2f}, mid_db={mid_db:.2f}, mud_margin_db={mud_margin:.2f}",
                )
            clarity_deficit = max(0.0, mid_db - presence_db - 2.0)
            if clarity_deficit > 0.0:
                add_band(
                    stem,
                    frequency_hz=2850.0,
                    gain_db=0.8 + min(5.2, clarity_deficit * 0.22),
                    q=0.95,
                    reason="lead vocal presence is low relative to body; broad boost restores intelligibility after masking cuts",
                    confidence=float(np.clip(0.62 + clarity_deficit * 0.018, 0.62, 0.86)),
                    source="contextual_vocal_presence_restore",
                    target="2-4 kHz vocal clarity",
                    metric=f"mid_db={mid_db:.2f}, presence_db={presence_db:.2f}, deficit_db={clarity_deficit:.2f}",
                )
            sibilance_margin = max(high_db, air_db) - (presence_db + 3.0)
            if sibilance_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=7200.0,
                    gain_db=-(0.8 + min(3.8, sibilance_margin * 0.22)),
                    q=1.35,
                    reason="lead vocal sibilance/air is high relative to presence; de-ess style static trim",
                    confidence=float(np.clip(0.64 + sibilance_margin * 0.02, 0.64, 0.9)),
                    source="contextual_vocal_sibilance",
                    target="5-9 kHz vocal sibilance",
                    metric=f"high_air_db={max(high_db, air_db):.2f}, presence_db={presence_db:.2f}, margin_db={sibilance_margin:.2f}",
                )

        elif role in {"back_vocal"}:
            mud_margin = low_mid_db - (mid_db - 4.0)
            if mud_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=280.0,
                    gain_db=-(0.9 + min(3.2, mud_margin * 0.2)),
                    q=1.0,
                    reason="backing vocal low mids make the stack cloudy; keep support below lead vocal",
                    confidence=float(np.clip(0.62 + mud_margin * 0.018, 0.62, 0.86)),
                    source="contextual_back_vocal_cleanup",
                    target="200-400 Hz backing vocal mud",
                    metric=f"low_mid_db={low_mid_db:.2f}, mid_db={mid_db:.2f}, margin_db={mud_margin:.2f}",
                )

        elif role == "kick":
            punch_deficit = max(0.0, low_mid_db - low_end_db - 3.0)
            if punch_deficit > 0.0:
                add_band(
                    stem,
                    frequency_hz=285.0,
                    gain_db=-(1.0 + min(4.5, punch_deficit * 0.25)),
                    q=1.1,
                    reason="kick boxiness is high relative to low fundamental; cut mud before any attack boost",
                    confidence=float(np.clip(0.66 + punch_deficit * 0.02, 0.66, 0.9)),
                    source="contextual_kick_boxiness",
                    target="220-350 Hz kick boxiness",
                    metric=f"low_mid_db={low_mid_db:.2f}, low_end_db={low_end_db:.2f}, margin_db={punch_deficit:.2f}",
                )
            attack_deficit = max(0.0, low_end_db - presence_db - 9.0)
            if attack_deficit > 0.0 and primary_lufs > -80.0:
                add_band(
                    stem,
                    frequency_hz=4200.0,
                    gain_db=0.7 + min(4.0, attack_deficit * 0.18),
                    q=1.0,
                    reason="kick attack is low relative to low end; add beater definition if headroom allows",
                    confidence=float(np.clip(0.58 + attack_deficit * 0.015, 0.58, 0.8)),
                    source="contextual_kick_attack_restore",
                    target="3-5 kHz kick attack",
                    metric=f"low_end_db={low_end_db:.2f}, presence_db={presence_db:.2f}, deficit_db={attack_deficit:.2f}",
                )

        elif role == "bass":
            bass_mud = low_mid_db - (low_end_db - 5.0)
            if bass_mud > 0.0:
                add_band(
                    stem,
                    frequency_hz=260.0,
                    gain_db=-(1.0 + min(4.4, bass_mud * 0.24)),
                    q=0.95,
                    reason="bass low-mid buildup masks kick and vocal body; broad cut keeps foundation clear",
                    confidence=float(np.clip(0.66 + bass_mud * 0.02, 0.66, 0.9)),
                    source="contextual_bass_low_mid_control",
                    target="180-350 Hz bass mud",
                    metric=f"low_mid_db={low_mid_db:.2f}, low_end_db={low_end_db:.2f}, margin_db={bass_mud:.2f}",
                )
            definition_deficit = max(0.0, low_end_db - mid_db - 10.0)
            if definition_deficit > 0.0:
                add_band(
                    stem,
                    frequency_hz=850.0,
                    gain_db=0.6 + min(3.8, definition_deficit * 0.18),
                    q=0.85,
                    reason="bass note definition is weak compared with low end; broad mid lift helps translation",
                    confidence=float(np.clip(0.58 + definition_deficit * 0.015, 0.58, 0.8)),
                    source="contextual_bass_definition",
                    target="700-1100 Hz bass note definition",
                    metric=f"low_end_db={low_end_db:.2f}, mid_db={mid_db:.2f}, deficit_db={definition_deficit:.2f}",
                )

        elif role in {"snare", "snare_top", "snare_bottom"}:
            box_margin = max(low_mid_db, mid_db) - (presence_db - 4.0)
            if box_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=780.0 if role != "snare_bottom" else 920.0,
                    gain_db=-(0.9 + min(4.2, box_margin * 0.22)),
                    q=1.25,
                    reason=f"{role} boxiness is high against attack; cut body resonance enough to matter",
                    confidence=float(np.clip(0.64 + box_margin * 0.018, 0.64, 0.88)),
                    source="contextual_snare_boxiness",
                    target="600-950 Hz snare boxiness",
                    metric=f"box_margin_db={box_margin:.2f}",
                )
            attack_deficit = max(0.0, mid_db - presence_db - 3.0)
            if attack_deficit > 0.0:
                add_band(
                    stem,
                    frequency_hz=4500.0,
                    gain_db=0.7 + min(4.0, attack_deficit * 0.18),
                    q=1.0,
                    reason=f"{role} attack is dull relative to body; restore crack after boxiness control",
                    confidence=float(np.clip(0.58 + attack_deficit * 0.016, 0.58, 0.82)),
                    source="contextual_snare_attack_restore",
                    target="3-6 kHz snare attack",
                    metric=f"mid_db={mid_db:.2f}, presence_db={presence_db:.2f}, deficit_db={attack_deficit:.2f}",
                )

        elif role in {"tom", "floor_tom"}:
            mud_margin = low_mid_db - (low_end_db - 6.0)
            if mud_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=330.0 if role == "tom" else 285.0,
                    gain_db=-(0.9 + min(4.0, mud_margin * 0.22)),
                    q=1.0,
                    reason=f"{role} low-mid buildup masks shell tone; deep broad cut is allowed when evidence is strong",
                    confidence=float(np.clip(0.62 + mud_margin * 0.018, 0.62, 0.86)),
                    source="contextual_tom_mud",
                    target="250-400 Hz tom mud",
                    metric=f"low_mid_db={low_mid_db:.2f}, low_end_db={low_end_db:.2f}, margin_db={mud_margin:.2f}",
                )
            cymbal_bleed = high_db - (mid_db + 3.0)
            if cymbal_bleed > 0.0:
                add_band(
                    stem,
                    frequency_hz=6500.0,
                    gain_db=-(0.8 + min(4.0, cymbal_bleed * 0.22)),
                    q=1.1,
                    reason=f"{role} close mic has cymbal spill; cut high bleed more decisively than hybrid mode",
                    confidence=float(np.clip(0.6 + cymbal_bleed * 0.018, 0.6, 0.84)),
                    source="contextual_tom_bleed",
                    target="5-8 kHz cymbal spill",
                    metric=f"high_db={high_db:.2f}, mid_db={mid_db:.2f}, margin_db={cymbal_bleed:.2f}",
                )

        elif role in {"overhead", "room"}:
            wash_margin = low_mid_db - (mid_db - 4.0)
            if wash_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=360.0,
                    gain_db=-(0.9 + min(3.4, wash_margin * 0.2)),
                    q=0.9,
                    reason=f"{role} low-mid wash muddies drum image; linked-style cleanup per stem",
                    confidence=float(np.clip(0.62 + wash_margin * 0.016, 0.62, 0.84)),
                    source="contextual_overhead_wash",
                    target="250-500 Hz cymbal/room wash",
                    metric=f"low_mid_db={low_mid_db:.2f}, mid_db={mid_db:.2f}, margin_db={wash_margin:.2f}",
                )
            harsh_margin = max(high_db, air_db) - (presence_db + 2.5)
            if harsh_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=7600.0,
                    gain_db=-(0.9 + min(4.2, harsh_margin * 0.22)),
                    q=1.0,
                    reason=f"{role} harshness/air is excessive relative to presence; control cymbal top",
                    confidence=float(np.clip(0.62 + harsh_margin * 0.018, 0.62, 0.86)),
                    source="contextual_overhead_harshness",
                    target="5-9 kHz cymbal harshness",
                    metric=f"high_air_db={max(high_db, air_db):.2f}, presence_db={presence_db:.2f}, margin_db={harsh_margin:.2f}",
                )

        elif role in {"guitar", "accordion", "keys", "playback"}:
            mud_margin = low_mid_db - (mid_db - 3.0)
            if mud_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=320.0,
                    gain_db=-(0.9 + min(4.6, mud_margin * 0.24)),
                    q=0.95,
                    reason=f"{role} low-mid buildup crowds arrangement; deep cleanup allowed after ADI/input staging",
                    confidence=float(np.clip(0.62 + mud_margin * 0.018, 0.62, 0.88)),
                    source="contextual_music_bed_mud",
                    target="250-500 Hz accompaniment mud",
                    metric=f"low_mid_db={low_mid_db:.2f}, mid_db={mid_db:.2f}, margin_db={mud_margin:.2f}",
                )
            fizz_margin = high_db - (presence_db + 3.0)
            if fizz_margin > 0.0:
                add_band(
                    stem,
                    frequency_hz=6500.0,
                    gain_db=-(0.8 + min(4.0, fizz_margin * 0.22)),
                    q=1.0,
                    reason=f"{role} top end is fizzy relative to presence; reduce abrasive layer",
                    confidence=float(np.clip(0.6 + fizz_margin * 0.018, 0.6, 0.84)),
                    source="contextual_music_bed_fizz",
                    target="5-8 kHz accompaniment fizz",
                    metric=f"high_db={high_db:.2f}, presence_db={presence_db:.2f}, margin_db={fizz_margin:.2f}",
                )
            presence_deficit = max(0.0, mid_db - presence_db - 6.0)
            if presence_deficit > 0.0 and role in {"accordion", "keys", "guitar"}:
                add_band(
                    stem,
                    frequency_hz=2200.0,
                    gain_db=0.5 + min(3.6, presence_deficit * 0.16),
                    q=0.85,
                    reason=f"{role} lacks articulation after cleanup; broad boost is allowed but critic will flag excess",
                    confidence=float(np.clip(0.55 + presence_deficit * 0.015, 0.55, 0.78)),
                    source="contextual_music_bed_articulation",
                    target="1.8-3 kHz instrument articulation",
                    metric=f"mid_db={mid_db:.2f}, presence_db={presence_db:.2f}, deficit_db={presence_deficit:.2f}",
                )

    _contextual_link_stereo_pairs(bands_by_channel, stems, critic_notes)
    actions_preview = []
    for channel, bands in sorted(bands_by_channel.items()):
        stem = stem_by_channel.get(channel)
        if stem is None:
            continue
        actions_preview.extend(
            {
                "channel": channel,
                "file": Path(stem.path).name,
                "instrument": _autoeq_instrument(stem.name),
                **_round_eq_band(band),
            }
            for band in bands
        )

    return bands_by_channel, {
        "enabled": True,
        "method": "contextual_deep_eq",
        "before": snapshot_before,
        "channel_diagnostics": snapshot_before["channel_diagnostics"],
        "group_diagnostics": snapshot_before["group_diagnostics"],
        "detected_conflicts": conflicts,
        "rejected_candidates": rejected,
        "critic_notes": critic_notes,
        "actions_preview": actions_preview,
        "limits": {
            "max_cut_db": 12.0,
            "max_boost_db": 9.0,
            "max_filters_per_channel": 7,
            "max_q": 6.0,
            "hard_blocks": ["invalid_frequency", "unstable_q", "boost_headroom_extreme"],
            "critic_after_db": 5.0,
        },
        "notes": [
            "Contextual Deep EQ is not a flat target matcher.",
            "Large EQ moves are allowed when role/masking evidence is strong.",
            "Regret score and critic notes flag risky decisions without over-constraining the method.",
            "Post-corrective channel/pair levels remain responsible for level compensation.",
        ],
    }


def _project_hybrid_add_eq_band(
    bands_by_channel: dict[int, list[dict[str, Any]]],
    rejected: list[dict[str, Any]],
    stem: AyaicStem,
    *,
    diagnostic: dict[str, Any],
    frequency_hz: float,
    gain_db: float,
    q: float,
    reason: str,
    confidence: float,
    component: str,
    source: str,
    target: str,
    metric: str,
) -> bool:
    role = str(diagnostic.get("role") or _project_channel_role(stem.name))
    limits = _project_hybrid_role_limits(role)
    if confidence < 0.55:
        rejected.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "frequency_hz": round(float(frequency_hz), 2),
                "gain_db": round(float(gain_db), 2),
                "reason": "hybrid_confidence_below_threshold",
                "confidence": round(float(confidence), 3),
                "component": component,
                "source": source,
                "target": target,
            }
        )
        return False
    if gain_db >= 0.0:
        rejected.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "frequency_hz": round(float(frequency_hz), 2),
                "gain_db": round(float(gain_db), 2),
                "reason": "hybrid_boosts_disabled",
                "confidence": round(float(confidence), 3),
                "component": component,
                "source": source,
                "target": target,
            }
        )
        return False

    gain = max(float(gain_db), -limits["max_cut_db"])
    if abs(gain) < 0.22:
        rejected.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "frequency_hz": round(float(frequency_hz), 2),
                "gain_db": round(float(gain), 2),
                "reason": "hybrid_candidate_below_minimum_move",
                "confidence": round(float(confidence), 3),
                "component": component,
                "source": source,
                "target": target,
            }
        )
        return False

    bands = bands_by_channel.setdefault(stem.channel_id, [])
    for existing in bands:
        if abs(math.log2(float(existing["frequency_hz"]) / float(frequency_hz))) < 0.24:
            existing.setdefault("evidence", [])
            existing["evidence"].append({"component": component, "source": source, "gain_db": round(float(gain), 3)})
            if abs(gain) > abs(float(existing["gain_db"])):
                existing["gain_db"] = round(gain, 3)
                existing["q"] = round(float(np.clip(q, 0.6, limits["max_q"])), 3)
                existing["reason"] = reason
                existing["confidence"] = round(float(confidence), 3)
                existing["component"] = component
                existing["source"] = source
                existing["target"] = target
                existing["metric"] = metric
            return True

    if len(bands) >= int(limits["max_filters"]):
        rejected.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "frequency_hz": round(float(frequency_hz), 2),
                "gain_db": round(float(gain), 2),
                "reason": "hybrid_max_filters_per_channel_reached",
                "confidence": round(float(confidence), 3),
                "component": component,
                "source": source,
                "target": target,
            }
        )
        return False

    bands.append(
        {
            "frequency_hz": float(frequency_hz),
            "gain_db": round(gain, 3),
            "q": round(float(np.clip(q, 0.6, limits["max_q"])), 3),
            "reason": reason,
            "confidence": round(float(confidence), 3),
            "role": role,
            "component": component,
            "source": source,
            "target": target,
            "metric": metric,
            "evidence": [{"component": component, "source": source, "gain_db": round(float(gain), 3)}],
        }
    )
    return True


def _project_hybrid_role_limits(role: str) -> dict[str, float]:
    if role in {"overhead", "room"}:
        return {"max_cut_db": 1.1, "max_filters": 3.0, "max_q": 2.0}
    if role in {"lead_vocal", "back_vocal"}:
        return {"max_cut_db": 1.2, "max_filters": 3.0, "max_q": 2.0}
    if role in {"kick", "bass"}:
        return {"max_cut_db": 1.45, "max_filters": 4.0, "max_q": 2.1}
    if role in {"snare", "snare_top", "snare_bottom", "tom", "floor_tom"}:
        return {"max_cut_db": 1.6, "max_filters": 4.0, "max_q": 2.2}
    return {"max_cut_db": 1.6, "max_filters": 4.0, "max_q": 2.0}


def _contextual_add_eq_band(
    bands_by_channel: dict[int, list[dict[str, Any]]],
    rejected: list[dict[str, Any]],
    critic_notes: list[dict[str, Any]],
    stem: AyaicStem,
    *,
    diagnostic: dict[str, Any],
    frequency_hz: float,
    gain_db: float,
    q: float,
    reason: str,
    confidence: float,
    source: str,
    target: str,
    metric: str,
    allow_boost: bool,
) -> bool:
    role = str(diagnostic.get("role") or _project_channel_role(stem.name))
    limits = _contextual_role_limits(role)
    freq = float(frequency_hz)
    requested_gain = float(gain_db)
    requested_q = float(q)
    if freq <= 20.0 or freq >= min(float(stem.sample_rate) * 0.48, 20_000.0):
        rejected.append(_contextual_rejection(stem, freq, requested_gain, requested_q, "invalid_frequency", confidence, source, target))
        return False
    if requested_q < 0.25 or requested_q > 18.0:
        rejected.append(_contextual_rejection(stem, freq, requested_gain, requested_q, "unstable_q", confidence, source, target))
        return False
    if requested_gain > 0.0 and not allow_boost:
        rejected.append(_contextual_rejection(stem, freq, requested_gain, requested_q, "boost_not_allowed_for_this_decision", confidence, source, target))
        return False

    gain = float(np.clip(requested_gain, -limits["max_cut_db"], limits["max_boost_db"]))
    peak_dbfs = float(diagnostic.get("peak_dbfs", _peak_dbfs(stem.audio)))
    if gain > 0.0 and peak_dbfs + gain > 6.0:
        rejected.append(_contextual_rejection(stem, freq, gain, requested_q, "boost_headroom_extreme", confidence, source, target))
        return False
    if gain > 0.0 and peak_dbfs + gain > -1.0:
        critic_notes.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "severity": "high",
                "reason": "boost_may_reduce_headroom_before_post_levels",
                "frequency_hz": round(freq, 2),
                "gain_db": round(gain, 3),
                "peak_dbfs": round(peak_dbfs, 3),
                "note": "Allowed for offline render, but master/post-level safety must absorb the headroom cost.",
            }
        )

    if abs(gain) < 0.25:
        rejected.append(_contextual_rejection(stem, freq, gain, requested_q, "candidate_below_minimum_audible_move", confidence, source, target))
        return False

    q_limited = float(np.clip(requested_q, 0.35, limits["max_q"]))
    regret_score, notes = _contextual_regret_score(
        role=role,
        gain_db=gain,
        q=q_limited,
        frequency_hz=freq,
        confidence=confidence,
        peak_dbfs=peak_dbfs,
    )
    for note in notes:
        critic_notes.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "severity": note["severity"],
                "reason": note["reason"],
                "frequency_hz": round(freq, 2),
                "gain_db": round(gain, 3),
                "q": round(q_limited, 3),
                "regret_score": round(regret_score, 4),
                "note": note["note"],
            }
        )

    bands = bands_by_channel.setdefault(stem.channel_id, [])
    for existing in bands:
        if abs(math.log2(float(existing["frequency_hz"]) / freq)) < 0.2:
            existing.setdefault("evidence", [])
            existing["evidence"].append({"source": source, "gain_db": round(gain, 3), "regret_score": round(regret_score, 4)})
            if abs(gain) > abs(float(existing["gain_db"])):
                existing.update(
                    {
                        "gain_db": round(gain, 3),
                        "q": round(q_limited, 3),
                        "reason": reason,
                        "confidence": round(float(confidence), 3),
                        "role": role,
                        "source": source,
                        "target": target,
                        "metric": metric,
                        "regret_score": round(regret_score, 4),
                        "critic_notes": notes,
                    }
                )
            return True

    if len(bands) >= int(limits["max_filters"]):
        weakest = min(bands, key=lambda item: abs(float(item["gain_db"])) * float(item.get("confidence", 0.5)))
        incoming_strength = abs(gain) * float(confidence)
        weakest_strength = abs(float(weakest["gain_db"])) * float(weakest.get("confidence", 0.5))
        if incoming_strength <= weakest_strength:
            rejected.append(_contextual_rejection(stem, freq, gain, q_limited, "max_filters_kept_stronger_existing_bands", confidence, source, target))
            return False
        critic_notes.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "severity": "medium",
                "reason": "max_filters_replaced_weaker_band",
                "frequency_hz": round(freq, 2),
                "gain_db": round(gain, 3),
                "note": "Deep EQ kept the stronger move and removed a weaker existing band.",
            }
        )
        bands.remove(weakest)

    bands.append(
        {
            "frequency_hz": freq,
            "gain_db": round(gain, 3),
            "q": round(q_limited, 3),
            "reason": reason,
            "confidence": round(float(confidence), 3),
            "role": role,
            "source": source,
            "target": target,
            "metric": metric,
            "regret_score": round(regret_score, 4),
            "critic_notes": notes,
            "evidence": [{"source": source, "gain_db": round(gain, 3), "regret_score": round(regret_score, 4)}],
        }
    )
    return True


def _contextual_rejection(
    stem: AyaicStem,
    frequency_hz: float,
    gain_db: float,
    q: float,
    reason: str,
    confidence: float,
    source: str,
    target: str,
) -> dict[str, Any]:
    return {
        "channel": stem.channel_id,
        "file": Path(stem.path).name,
        "frequency_hz": round(float(frequency_hz), 2),
        "gain_db": round(float(gain_db), 3),
        "q": round(float(q), 3),
        "reason": reason,
        "confidence": round(float(confidence), 3),
        "source": source,
        "target": target,
    }


def _contextual_role_limits(role: str) -> dict[str, float]:
    if role in {"overhead", "room"}:
        return {"max_cut_db": 10.0, "max_boost_db": 4.0, "max_filters": 6.0, "max_q": 4.5}
    if role in {"lead_vocal", "back_vocal"}:
        return {"max_cut_db": 10.0, "max_boost_db": 7.0, "max_filters": 7.0, "max_q": 5.0}
    if role in {"kick", "bass"}:
        return {"max_cut_db": 12.0, "max_boost_db": 7.0, "max_filters": 7.0, "max_q": 5.0}
    if role in {"snare", "snare_top", "snare_bottom", "tom", "floor_tom"}:
        return {"max_cut_db": 12.0, "max_boost_db": 8.0, "max_filters": 7.0, "max_q": 5.5}
    return {"max_cut_db": 12.0, "max_boost_db": 6.0, "max_filters": 7.0, "max_q": 5.0}


def _contextual_regret_score(
    *,
    role: str,
    gain_db: float,
    q: float,
    frequency_hz: float,
    confidence: float,
    peak_dbfs: float,
) -> tuple[float, list[dict[str, str]]]:
    abs_gain = abs(float(gain_db))
    score = 0.0
    score += max(0.0, (abs_gain - 4.0) / 8.0) * 0.42
    score += max(0.0, (float(q) - 3.0) / 5.0) * 0.2
    score += max(0.0, (0.7 - float(confidence)) / 0.7) * 0.18
    if gain_db > 0.0:
        score += 0.12
        if peak_dbfs + gain_db > -1.0:
            score += 0.22
    if role in {"overhead", "room"} and gain_db > 0.0:
        score += 0.2
    if frequency_hz > 5000.0 and gain_db > 4.0:
        score += 0.16
    if frequency_hz < 120.0 and gain_db > 4.0:
        score += 0.18
    score = float(np.clip(score, 0.0, 1.0))

    notes: list[dict[str, str]] = []
    if abs_gain >= 6.0:
        notes.append(
            {
                "severity": "high" if abs_gain >= 9.0 else "medium",
                "reason": "large_eq_move",
                "note": "Large EQ move allowed; verify that it solves a role/masking problem rather than forcing a target curve.",
            }
        )
    if gain_db > 0.0 and peak_dbfs + gain_db > -1.0:
        notes.append(
            {
                "severity": "high",
                "reason": "boost_headroom_risk",
                "note": "Boost can push the channel above normal headroom before post-corrective leveling.",
            }
        )
    if q > 4.5:
        notes.append(
            {
                "severity": "medium",
                "reason": "narrow_filter",
                "note": "High-Q EQ can sound resonant; use only when the evidence points to a narrow problem.",
            }
        )
    if role in {"overhead", "room"} and gain_db > 0.0:
        notes.append(
            {
                "severity": "medium",
                "reason": "overhead_boost_risk",
                "note": "Boosting cymbal/room channels can exaggerate bleed and harshness.",
            }
        )
    return score, notes


def _contextual_link_stereo_pairs(
    bands_by_channel: dict[int, list[dict[str, Any]]],
    stems: list[AyaicStem],
    critic_notes: list[dict[str, Any]],
) -> None:
    pairs = (
        ("Guitar", ("guitar l", "guitar r")),
        ("Playback", ("playback l", "playback r")),
        ("Back Vox", ("back vox l", "back vox r")),
        ("Overheads", ("oh l", "oh r")),
    )
    by_key = {_name_key(stem.name): stem for stem in stems}
    for label, keys in pairs:
        members = [by_key[key] for key in keys if key in by_key]
        if len(members) != 2:
            continue
        first_bands = bands_by_channel.get(members[0].channel_id, [])
        second_bands = bands_by_channel.get(members[1].channel_id, [])
        if not first_bands and not second_bands:
            continue
        merged = _contextual_merge_pair_bands([first_bands, second_bands])
        for stem in members:
            bands_by_channel[stem.channel_id] = [
                {**band, "linked_group": label, "reason": f"{band.get('reason', '')} Linked stereo pair policy."}
                for band in merged
            ]
        critic_notes.append(
            {
                "channel": [stem.channel_id for stem in members],
                "file": [Path(stem.path).name for stem in members],
                "severity": "low",
                "reason": "linked_stereo_pair_eq",
                "note": f"{label} EQ bands were mirrored to preserve stereo image.",
            }
        )


def _contextual_merge_pair_bands(band_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for band in [item for bands in band_sets for item in bands]:
        for existing in merged:
            if abs(math.log2(float(existing["frequency_hz"]) / float(band["frequency_hz"]))) < 0.18:
                if abs(float(band["gain_db"])) > abs(float(existing["gain_db"])):
                    existing.update(dict(band))
                existing.setdefault("evidence", [])
                existing["evidence"].extend(band.get("evidence", []))
                break
        else:
            merged.append(dict(band))
    return sorted(merged, key=lambda item: float(item["frequency_hz"]))[:7]


def _project_hybrid_dynamic_candidates(
    stems: list[AyaicStem],
    project_analysis: dict[str, Any],
    channel_diagnostics: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    stem_by_file = {Path(stem.path).name: stem for stem in stems}
    lead_vocal = next((stem for stem in stems if channel_diagnostics.get(stem.channel_id, {}).get("role") == "lead_vocal"), None)
    candidates: list[dict[str, Any]] = []
    for conflict in project_analysis.get("detected_conflicts", []):
        conflict_type = str(conflict.get("type") or "")
        masker_file = str(conflict.get("masker_source") or "")
        masker = stem_by_file.get(masker_file)
        if conflict_type == "lead_vocal_presence_masking" and masker is not None and lead_vocal is not None:
            candidates.append(
                {
                    "channel": masker.channel_id,
                    "file": Path(masker.path).name,
                    "decision": "dynamic_bell_cut",
                    "frequency_hz": 3100.0,
                    "q": 1.2,
                    "max_gain_db": -1.4,
                    "sidechain_source": Path(lead_vocal.path).name,
                    "trigger_band_hz": [2500.0, 4000.0],
                    "reason": "Use dynamic cut only while lead vocal is active; static hybrid cut is intentionally capped.",
                    "applied": False,
                    "mode": "report_only",
                }
            )
        elif conflict_type == "kick_bass_low_end_conflict" and masker is not None:
            candidates.append(
                {
                    "channel": masker.channel_id,
                    "file": Path(masker.path).name,
                    "decision": "dynamic_bell_cut",
                    "frequency_hz": 85.0,
                    "q": 0.9,
                    "max_gain_db": -1.2,
                    "sidechain_source": "kick",
                    "trigger_band_hz": [50.0, 100.0],
                    "reason": "Low-end control can be dynamic if bass only masks kick during hits.",
                    "applied": False,
                    "mode": "report_only",
                }
            )
    overheads = [stem for stem in stems if channel_diagnostics.get(stem.channel_id, {}).get("role") == "overhead"]
    if overheads:
        high_scores = [float(channel_diagnostics.get(stem.channel_id, {}).get("harshness_score", 0.0)) for stem in overheads]
        if max(high_scores, default=0.0) > 0.0:
            for stem in overheads:
                candidates.append(
                    {
                        "channel": stem.channel_id,
                        "file": Path(stem.path).name,
                        "decision": "linked_dynamic_bell_cut",
                        "frequency_hz": 7600.0,
                        "q": 1.0,
                        "max_gain_db": -1.1,
                        "sidechain_source": "overhead_pair",
                        "trigger_band_hz": [5000.0, 9000.0],
                        "reason": "Prefer linked dynamic overhead harshness control for live mode.",
                        "applied": False,
                        "mode": "report_only",
                    }
                )
    return candidates


def _project_corrective_snapshot(stems: list[AyaicStem]) -> dict[str, Any]:
    return {
        "channel_diagnostics": [_project_channel_diagnostic(stem) for stem in stems],
        "group_diagnostics": _project_group_diagnostics(stems),
    }


def _project_channel_diagnostic(stem: AyaicStem) -> dict[str, Any]:
    primary_audio = _stem_primary_loudness_audio(stem)
    band_energy = _band_energy_for_rendered(primary_audio, stem.sample_rate)
    role = _project_channel_role(stem.name)
    low_end_db = _project_band_db(band_energy, ("sub", "bass"))
    low_mid_db = float(band_energy.get("low_mid", -100.0))
    mid_db = float(band_energy.get("mid", -100.0))
    presence_db = _project_band_db(band_energy, ("high_mid", "high"))
    high_db = float(band_energy.get("high", -100.0))
    air_db = float(band_energy.get("air", -100.0))
    return {
        "channel": stem.channel_id,
        "file": Path(stem.path).name,
        "instrument": _autoeq_instrument(stem.name),
        "role": role,
        "group": stem.group,
        "measurement_scope": stem.bleed_analysis.get("analysis_mode", "full_track_no_bleed_detector"),
        "primary_active_ratio": stem.bleed_analysis.get("post_phase_analysis_active_ratio", stem.bleed_analysis.get("analysis_active_ratio")),
        "primary_lufs": round(_lufs(primary_audio, stem.sample_rate), 3),
        "full_track_lufs": round(_lufs(stem.audio, stem.sample_rate), 3),
        "peak_dbfs": round(_peak_dbfs(stem.audio), 3),
        "band_energy_db": {key: round(float(value), 2) for key, value in band_energy.items()},
        "low_end_db": round(low_end_db, 2),
        "low_mid_db": round(low_mid_db, 2),
        "mid_db": round(mid_db, 2),
        "presence_db": round(presence_db, 2),
        "high_db": round(high_db, 2),
        "air_db": round(air_db, 2),
        "mud_score": round(max(0.0, low_mid_db - (mid_db - 4.0)), 3),
        "harshness_score": round(max(0.0, high_db - (presence_db + 2.0)), 3),
        "cymbal_wash_score": round(max(0.0, air_db - (mid_db + 1.0)), 3),
    }


def _project_group_diagnostics(stems: list[AyaicStem]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups = sorted({stem.group for stem in stems}, key=lambda group: (AYAIC_BUS_GROUP_ORDER.index(group) if group in AYAIC_BUS_GROUP_ORDER else 999, group))
    for group in groups:
        members = [stem for stem in stems if stem.group == group]
        if not members:
            continue
        audio, measurement = _combined_primary_loudness_audio(members)
        band_energy = _band_energy_for_rendered(audio, members[0].sample_rate)
        low_end_db = _project_band_db(band_energy, ("sub", "bass"))
        low_mid_db = float(band_energy.get("low_mid", -100.0))
        mid_db = float(band_energy.get("mid", -100.0))
        presence_db = _project_band_db(band_energy, ("high_mid", "high"))
        high_db = float(band_energy.get("high", -100.0))
        air_db = float(band_energy.get("air", -100.0))
        out.append(
            {
                "group": group,
                "members": [Path(stem.path).name for stem in members],
                "measurement_scope": measurement["scope"],
                "primary_active_ratio": measurement["active_ratio"],
                "primary_lufs": round(_lufs(audio, members[0].sample_rate), 3),
                "band_energy_db": {key: round(float(value), 2) for key, value in band_energy.items()},
                "low_end_db": round(low_end_db, 2),
                "low_mid_db": round(low_mid_db, 2),
                "mid_db": round(mid_db, 2),
                "presence_db": round(presence_db, 2),
                "high_db": round(high_db, 2),
                "air_db": round(air_db, 2),
                "mud_score": round(max(0.0, low_mid_db - (mid_db - 4.0)), 3),
                "harshness_score": round(max(0.0, high_db - (presence_db + 2.0)), 3),
                "low_end_excess_score": round(max(0.0, low_end_db - (mid_db + 5.0)), 3),
            }
        )
    return out


def _project_channel_role(name: str) -> str:
    instrument = _autoeq_instrument(name)
    if instrument in {"overhead_l", "overhead_r", "overhead_pair", "hi_hat", "ride"}:
        return "overhead"
    if instrument == "room":
        return "room"
    if instrument in {"snare_top", "snare_bottom"}:
        return instrument
    if instrument in {"tom", "rack_tom"}:
        return "tom"
    if instrument in {"floor_tom"}:
        return "floor_tom"
    if instrument in {"bass", "bass_guitar"}:
        return "bass"
    if instrument in {"guitar_l", "guitar_r", "guitar", "electric_guitar"}:
        return "guitar"
    if instrument in {"playback_l", "playback_r", "playback"}:
        return "playback"
    if instrument in {"back_vocal", "backing_vocal"}:
        return "back_vocal"
    if instrument in {"lead_vocal"}:
        return "lead_vocal"
    if instrument in {"accordion", "keys"}:
        return instrument
    if instrument == "kick":
        return "kick"
    if instrument == "snare":
        return "snare"
    return "music"


def _project_band_db(band_energy: dict[str, float], names: tuple[str, ...]) -> float:
    values = [float(band_energy.get(name, -100.0)) for name in names]
    if not values:
        return -100.0
    powers = [db_to_amp(value) ** 2 for value in values]
    return amp_to_db(float(math.sqrt(sum(powers) / max(1, len(powers)))))


def _project_add_eq_band(
    bands_by_channel: dict[int, list[dict[str, Any]]],
    rejected: list[dict[str, Any]],
    stem: AyaicStem,
    *,
    diagnostic: dict[str, Any],
    frequency_hz: float,
    gain_db: float,
    q: float,
    reason: str,
    confidence: float,
    source: str,
    target: str,
    metric: str,
) -> None:
    role = str(diagnostic.get("role") or _project_channel_role(stem.name))
    limits = _project_role_limits(role)
    if confidence < 0.55:
        rejected.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "frequency_hz": round(float(frequency_hz), 2),
                "gain_db": round(float(gain_db), 2),
                "reason": "confidence_below_project_corrective_threshold",
                "confidence": round(float(confidence), 3),
                "source": source,
                "target": target,
            }
        )
        return

    gain = float(gain_db)
    if 0.55 <= confidence < 0.8:
        gain *= 0.65
    if gain > 0.0:
        peak_dbfs = float(diagnostic.get("peak_dbfs", _peak_dbfs(stem.audio)))
        if peak_dbfs > -3.0:
            rejected.append(
                {
                    "channel": stem.channel_id,
                    "file": Path(stem.path).name,
                    "frequency_hz": round(float(frequency_hz), 2),
                    "gain_db": round(float(gain), 2),
                    "reason": "boost_rejected_due_to_limited_peak_headroom",
                    "confidence": round(float(confidence), 3),
                    "source": source,
                    "target": target,
                }
            )
            return
    gain = float(np.clip(gain, -limits["max_cut_db"], limits["max_boost_db"]))
    if abs(gain) < 0.25:
        rejected.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "frequency_hz": round(float(frequency_hz), 2),
                "gain_db": round(float(gain), 2),
                "reason": "candidate_below_minimum_audible_project_move",
                "confidence": round(float(confidence), 3),
                "source": source,
                "target": target,
            }
        )
        return

    bands = bands_by_channel.setdefault(stem.channel_id, [])
    for existing in bands:
        if abs(math.log2(float(existing["frequency_hz"]) / float(frequency_hz))) < 0.22:
            if abs(gain) > abs(float(existing["gain_db"])):
                existing["gain_db"] = round(gain, 3)
                existing["q"] = round(float(np.clip(q, 0.6, limits["max_q"])), 3)
                existing["reason"] = reason
                existing["confidence"] = round(float(confidence), 3)
                existing["source"] = source
                existing["target"] = target
                existing["metric"] = metric
            return
    if len(bands) >= int(limits["max_filters"]):
        rejected.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "frequency_hz": round(float(frequency_hz), 2),
                "gain_db": round(float(gain), 2),
                "reason": "max_project_corrective_filters_per_channel_reached",
                "confidence": round(float(confidence), 3),
                "source": source,
                "target": target,
            }
        )
        return

    bands.append(
        {
            "frequency_hz": float(frequency_hz),
            "gain_db": round(gain, 3),
            "q": round(float(np.clip(q, 0.6, limits["max_q"])), 3),
            "reason": reason,
            "confidence": round(float(confidence), 3),
            "role": role,
            "source": source,
            "target": target,
            "metric": metric,
        }
    )


def _project_role_limits(role: str) -> dict[str, float]:
    if role in {"overhead", "room"}:
        return {"max_cut_db": 1.4, "max_boost_db": 0.0, "max_filters": 3.0, "max_q": 2.0}
    if role in {"lead_vocal", "back_vocal"}:
        return {"max_cut_db": 1.6, "max_boost_db": 0.0, "max_filters": 4.0, "max_q": 2.2}
    if role in {"kick", "bass"}:
        return {"max_cut_db": 2.0, "max_boost_db": 0.0, "max_filters": 4.0, "max_q": 2.2}
    if role in {"snare", "snare_top", "snare_bottom", "tom", "floor_tom"}:
        return {"max_cut_db": 2.2, "max_boost_db": 0.0, "max_filters": 4.0, "max_q": 2.4}
    return {"max_cut_db": 2.2, "max_boost_db": 0.0, "max_filters": 5.0, "max_q": 2.2}


def _round_eq_band(band: dict[str, Any]) -> dict[str, Any]:
    out = {
        "frequency_hz": round(float(band["frequency_hz"]), 2),
        "gain_db": round(float(band["gain_db"]), 2),
        "q": round(float(band["q"]), 2),
    }
    for key in (
        "reason",
        "window",
        "band",
        "priority",
        "source_count",
        "share_before",
        "confidence",
        "role",
        "component",
        "source",
        "target",
        "metric",
        "evidence",
        "regret_score",
        "critic_notes",
        "linked_group",
    ):
        if key in band:
            value = band[key]
            out[key] = round(float(value), 4) if isinstance(value, float) else value
    return out


def _offline_hpf_lpf_plan(name: str) -> dict[str, Any]:
    instrument = _offline_filter_instrument(name)
    hpf = float(OFFLINE_HPF_TARGETS_HZ.get(instrument, OFFLINE_HPF_TARGETS_HZ["custom"]))
    lpf = float(OFFLINE_LPF_TARGETS_HZ.get(instrument, 0.0))
    return {"instrument": instrument, "hpf_hz": hpf, "lpf_hz": lpf}


def _offline_filter_instrument(name: str) -> str:
    key = _name_key(name)
    drum = _drum_instrument(name)
    if drum == "snare":
        if key in {"snare b", "snare bottom"}:
            return "snare_bottom"
        if key in {"snare t", "snare top"}:
            return "snare_top"
        return "snare"
    if drum:
        return drum
    if _is_backing_vocal_key(key):
        return "backing_vocal"
    if "vox" in key or "vocal" in key:
        return "lead_vocal"
    if "bass" in key:
        return "bass_guitar"
    if "guitar" in key or "gtr" in key:
        return "electric_guitar"
    if "accordion" in key:
        return "accordion"
    if "playback" in key:
        return "playback"
    if "room" in key:
        return "room"
    return "custom"


def _corrective_eq_profile_key(name: str) -> str:
    return _offline_filter_instrument(name)


def _corrective_eq_instrument(name: str) -> str:
    instrument = _offline_filter_instrument(name)
    if instrument in {"snare_top", "snare_bottom"}:
        return "snare"
    return instrument


def _priority_for_instrument(instrument: str) -> int:
    if instrument == "lead_vocal":
        return 1
    if instrument == "kick":
        return 2
    if instrument in {"snare", "bass_guitar"}:
        return 3
    if instrument in {"electric_guitar", "accordion", "playback", "backing_vocal"}:
        return 4
    if instrument in {"overhead", "room", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
        return 5
    return 4


def _analysis_block(x: np.ndarray, sample_rate: int, window_sec: float = 18.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    window = min(len(arr), max(1024, int(window_sec * sample_rate)))
    if len(arr) <= window:
        return arr
    hop = max(512, window // 3)
    best_start = 0
    best_energy = -1.0
    for start in range(0, len(arr) - window, hop):
        energy = float(np.mean(np.square(arr[start:start + window])))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return arr[best_start:best_start + window]


def _band_energy_for_rendered(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    block = _analysis_block(_mono(audio), sample_rate)
    if len(block) < 1024:
        return {}
    windowed = block * np.hanning(len(block))
    spec = np.abs(np.fft.rfft(windowed)) + 1e-12
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sample_rate)
    bands = {
        "sub": (20.0, 60.0),
        "bass": (60.0, 250.0),
        "low_mid": (250.0, 500.0),
        "mid": (500.0, 2000.0),
        "high_mid": (2000.0, 4000.0),
        "high": (4000.0, 8000.0),
        "air": (8000.0, 14000.0),
    }
    out: dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs < hi)
        if not np.any(idx):
            out[name] = -100.0
            continue
        band_rms = float(np.sqrt(np.sum(np.square(spec[idx]))) / len(windowed))
        out[name] = amp_to_db(band_rms)
    return out


def _cross_band_for_frequency(freq: float) -> str:
    centers = {
        "sub": 40.0,
        "bass": 120.0,
        "low_mid": 375.0,
        "mid": 1250.0,
        "high_mid": 3000.0,
        "high": 6000.0,
        "air": 12000.0,
    }
    return min(centers.keys(), key=lambda name: abs(centers[name] - freq))


def _frequency_window_snapshot(stems: list[AyaicStem]) -> dict[str, Any]:
    if not stems:
        return {"windows": [], "by_id": {}}

    spectrum_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    family_audio: dict[str, np.ndarray] = {}
    for stem in stems:
        family = _frequency_window_family(_corrective_eq_instrument(stem.name))
        if family not in family_audio:
            family_audio[family] = np.zeros_like(stem.audio)
        family_audio[family] += ensure_stereo(stem.audio)

    def _cached_band_metrics(audio: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> tuple[float, float]:
        key = id(audio)
        cached = spectrum_cache.get(key)
        if cached is None:
            block = _analysis_block(_mono(audio), sample_rate, window_sec=18.0)
            if len(block) < 256:
                cached = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
            else:
                windowed = block * np.hanning(len(block))
                spec = np.abs(np.fft.rfft(windowed)) + 1e-12
                cached = (
                    np.fft.rfftfreq(len(windowed), 1.0 / sample_rate),
                    np.square(spec),
                )
            spectrum_cache[key] = cached

        freqs, power = cached
        if freqs.size == 0:
            return 1e-12, -100.0
        idx = (freqs >= low_hz) & (freqs < high_hz)
        if not np.any(idx):
            return 1e-12, -100.0
        band_power = float(np.sum(power[idx]) + 1e-12)
        band_rms = float(np.sqrt(np.mean(power[idx])) + 1e-12)
        return band_power, amp_to_db(band_rms)

    windows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for config in FREQUENCY_WINDOW_DEFINITIONS:
        low_hz = float(config["low_hz"])
        high_hz = float(config["high_hz"])
        total_power = sum(
            _cached_band_metrics(stem.audio, stem.sample_rate, low_hz, high_hz)[0]
            for stem in stems
        )
        total_power = max(total_power, 1e-12)

        top_channels = []
        for stem in stems:
            power, band_db = _cached_band_metrics(stem.audio, stem.sample_rate, low_hz, high_hz)
            instrument = _corrective_eq_instrument(stem.name)
            top_channels.append(
                {
                    "channel": stem.channel_id,
                    "file": Path(stem.path).name,
                    "instrument": instrument,
                    "family": _frequency_window_family(instrument),
                    "band_db": round(band_db, 2),
                    "share": round(float(power / total_power), 4),
                }
            )
        top_channels.sort(key=lambda item: item["share"], reverse=True)

        family_metrics = []
        family_shares: dict[str, float] = {}
        for family, audio in family_audio.items():
            sample_rate = stems[0].sample_rate
            power, band_db = _cached_band_metrics(audio, sample_rate, low_hz, high_hz)
            share = float(power / total_power)
            family_shares[family] = round(share, 4)
            family_metrics.append(
                {
                    "family": family,
                    "band_db": round(band_db, 2),
                    "share": round(share, 4),
                }
            )
        family_metrics.sort(key=lambda item: item["share"], reverse=True)

        report = {
            "id": config["id"],
            "label": config["label"],
            "range_hz": [round(low_hz, 1), round(high_hz, 1)],
            "focus": config["focus"],
            "action_mode": config["action_mode"],
            "family_shares": family_shares,
            "dominant_family": family_metrics[0]["family"] if family_metrics else "",
            "runner_up_family": family_metrics[1]["family"] if len(family_metrics) > 1 else "",
            "families": family_metrics[:6],
            "channels": top_channels[:8],
        }
        windows.append(report)
        by_id[config["id"]] = report

    return {"windows": windows, "by_id": by_id}


def _frequency_window_family(instrument: str) -> str:
    if instrument == "lead_vocal":
        return "lead_vocal"
    if instrument == "backing_vocal":
        return "backing_vocal"
    if instrument in {"bass", "bass_guitar"}:
        return "bass"
    if instrument == "kick":
        return "kick"
    if instrument in CYMBAL_INSTRUMENTS:
        return "cymbals"
    if instrument in DRUM_INSTRUMENTS:
        return "drums"
    return "music"


def _shift_primary_loudness_ranges_for_applied_delays(stems: list[AyaicStem]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for stem in stems:
        if not stem.primary_loudness_ranges or stem.delay_ms <= 0.0:
            continue
        shift = _samples_from_ms(stem.delay_ms, stem.sample_rate)
        shifted = _merge_ranges(
            [
                (
                    min(stem.audio.shape[0], start + shift),
                    min(stem.audio.shape[0], end + shift),
                )
                for start, end in stem.primary_loudness_ranges
            ],
            gap=0,
        )
        active_samples = sum(end - start for start, end in shifted)
        stem.primary_loudness_ranges = shifted
        stem.bleed_analysis["primary_loudness_range_delay_shift_ms"] = round(stem.delay_ms, 3)
        stem.bleed_analysis["post_phase_analysis_active_ratio"] = round(active_samples / max(1, stem.audio.shape[0]), 4)
        report.append(
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "delay_ms": round(stem.delay_ms, 3),
                "shift_samples": shift,
                "primary_loudness_range_count": len(shifted),
                "post_phase_active_ratio": stem.bleed_analysis["post_phase_analysis_active_ratio"],
                "primary_loudness_ranges_preview_sec": _ranges_preview_seconds(shifted, stem.sample_rate),
            }
        )
    return report


def _apply_track_levels(stems: list[AyaicStem], *, stage: str = "ayaic_channel_levels") -> list[dict[str, Any]]:
    report = []
    for stem in stems:
        measurement_audio = _stem_primary_loudness_audio(stem)
        before = _lufs(measurement_audio, stem.sample_rate)
        gain = stem.target_lufs - before
        stem.audio = stem.audio * db_to_amp(gain)
        stem.track_gain_db += gain
        post_measurement_audio = _stem_primary_loudness_audio(stem)
        report.append(
            {
                "stage": stage,
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "target_lufs": round(stem.target_lufs, 3),
                "loudness_method": _lufs_method(),
                "measurement_scope": stem.bleed_analysis.get("analysis_mode", "full_track"),
                "primary_active_ratio": stem.bleed_analysis.get("analysis_active_ratio"),
                "post_phase_primary_active_ratio": stem.bleed_analysis.get("post_phase_analysis_active_ratio"),
                "primary_loudness_range_delay_shift_ms": stem.bleed_analysis.get("primary_loudness_range_delay_shift_ms"),
                "bleed_dominant": stem.bleed_analysis.get("analysis_bleed_dominant"),
                "measured_lufs": round(before, 3),
                "full_track_lufs": round(_lufs(stem.audio / db_to_amp(gain), stem.sample_rate), 3),
                "gain_db": round(gain, 3),
                "post_lufs": round(_lufs(post_measurement_audio, stem.sample_rate), 3),
                "post_full_track_lufs": round(_lufs(stem.audio, stem.sample_rate), 3),
            }
        )
    return report


def _apply_pair_levels(stems: list[AyaicStem]) -> list[dict[str, Any]]:
    return _validate_summed_ayaic_stems(stems)


def _validate_summed_ayaic_stems(stems: list[AyaicStem]) -> list[dict[str, Any]]:
    report = []
    for spec in _summed_lufs_group_specs():
        label = str(spec["label"])
        target = float(spec["target_lufs"])
        members = _summed_lufs_group_members(stems, spec)
        if len(members) < 2:
            continue
        summed, measurement = _combined_primary_loudness_audio(members)
        sample_rate = members[0].sample_rate
        before = _lufs(summed, sample_rate)
        gain = target - before
        for stem in members:
            stem.audio = stem.audio * db_to_amp(gain)
            stem.track_gain_db += gain
        post_summed, _ = _combined_primary_loudness_audio(members)
        report.append(
            {
                "type": "summed_stem_validation",
                "stem": label,
                "members": [Path(stem.path).name for stem in members],
                "target_lufs": target,
                "loudness_method": _lufs_method(),
                "measurement_scope": measurement["scope"],
                "primary_active_ratio": measurement["active_ratio"],
                "pre_sum_lufs": round(before, 3),
                "shared_gain_db": round(gain, 3),
                "post_sum_lufs": round(_lufs(post_summed, sample_rate), 3),
                "post_full_sum_lufs": round(_lufs(_sum_audio([stem.audio for stem in members]), sample_rate), 3),
            }
        )
    return report


def _summed_lufs_group_specs() -> list[dict[str, Any]]:
    return [
        {
            "label": "Drums Bus",
            "target_lufs": -25.0,
            "kind": "drums_bus",
            "note": "All drum-channel summed LUFS, including overheads, hi-hat, ride, room, and close mics.",
        },
        {
            "label": "Guitar",
            "target_lufs": -25.0,
            "kind": "exact_names",
            "names": ("guitar l", "guitar r", "gtr l", "gtr r"),
        },
        {
            "label": "Playback",
            "target_lufs": -25.0,
            "kind": "exact_names",
            "names": ("playback l", "playback r"),
        },
        {
            "label": "Lead Vocal Bus",
            "target_lufs": -22.0,
            "kind": "lead_vocal_bus",
            "note": "All non-backing lead vocal channels summed together.",
        },
        {
            "label": "Backs Bus",
            "target_lufs": -28.0,
            "kind": "backing_vocal_bus",
            "note": "Backing vocals stay separate from Lead Vocal Bus.",
        },
    ]


def _summed_lufs_group_members(stems: list[AyaicStem], spec: dict[str, Any]) -> list[AyaicStem]:
    kind = str(spec.get("kind") or "")
    if kind == "exact_names":
        names = set(str(name) for name in (spec.get("names") or ()))
        return [stem for stem in stems if _name_key(stem.name) in names]
    if kind == "drums_bus":
        return [stem for stem in stems if _instrument_group(stem.name) == "drums"]
    if kind == "lead_vocal_bus":
        return [
            stem for stem in stems
            if _offline_filter_instrument(stem.name) == "lead_vocal"
            and not _is_backing_vocal_key(_name_key(stem.name))
        ]
    if kind == "backing_vocal_bus":
        return [stem for stem in stems if _is_backing_vocal_key(_name_key(stem.name))]
    return []


def _apply_phase_delay_from_overheads(stems: list[AyaicStem], *, max_delay_ms: float) -> dict[str, Any]:
    snare_report = _apply_snare_bleed_phase_alignment(stems, max_delay_ms=max_delay_ms)
    if snare_report.get("snare_bleed_alignment", {}).get("enabled"):
        return snare_report
    fallback = _apply_global_bleed_phase_alignment(stems, max_delay_ms=max_delay_ms)
    fallback["order"] = ["snare_bleed_phase_alignment_fallback", *fallback.get("order", [])]
    fallback["snare_bleed_alignment"] = snare_report.get("snare_bleed_alignment", snare_report)
    return fallback


def _apply_overhead_pair_phase_alignment(stems: list[AyaicStem], *, max_delay_ms: float) -> dict[str, Any]:
    pair = _overhead_lr(stems)
    if not pair:
        return {"enabled": False, "reason": "overhead_pair_not_found"}

    left_stem, right_stem = pair
    if left_stem.sample_rate != right_stem.sample_rate:
        raise ValueError(
            f"overhead sample rate mismatch {Path(left_stem.path).name}={left_stem.sample_rate} "
            f"{Path(right_stem.path).name}={right_stem.sample_rate}"
        )

    sample_rate = left_stem.sample_rate
    left = _highpass(_mono(left_stem.audio), sample_rate, 40.0)
    right = _highpass(_mono(right_stem.audio), sample_rate, 40.0)
    start, measurement = _representative_delay_measurement(
        left,
        right,
        sample_rate,
        max_delay_ms=max_delay_ms,
        prefer_negative=False,
    )
    if not measurement:
        return {
            "enabled": False,
            "reason": "no_confident_overhead_pair_measurement",
            "left_file": Path(left_stem.path).name,
            "right_file": Path(right_stem.path).name,
        }
    measured_delay_ms = float(measurement["delay_ms"])

    applied_left_ms = 0.0
    applied_right_ms = 0.0
    farthest_stem = left_stem
    adjusted_stem = right_stem
    if measured_delay_ms < -0.3:
        applied_right_ms = min(max_delay_ms, -measured_delay_ms)
        right_stem.delay_ms = max(right_stem.delay_ms, applied_right_ms)
        right_stem.audio = _delay_audio(right_stem.audio, _samples_from_ms(right_stem.delay_ms, sample_rate))
        farthest_stem = left_stem
        adjusted_stem = right_stem
    elif measured_delay_ms > 0.3:
        applied_left_ms = min(max_delay_ms, measured_delay_ms)
        left_stem.delay_ms = max(left_stem.delay_ms, applied_left_ms)
        left_stem.audio = _delay_audio(left_stem.audio, _samples_from_ms(left_stem.delay_ms, sample_rate))
        farthest_stem = right_stem
        adjusted_stem = left_stem

    aligned_left = _highpass(_mono(left_stem.audio), sample_rate, 40.0)
    aligned_right = _highpass(_mono(right_stem.audio), sample_rate, 40.0)
    window_len = int(3.0 * sample_rate)
    corr_end = min(start + window_len, len(aligned_left), len(aligned_right))
    corr_left = aligned_left[start:corr_end]
    corr_right = aligned_right[start:corr_end]
    current_corr = _norm_corr(corr_left, corr_right)
    flipped_corr = _norm_corr(corr_left, -corr_right)
    phase_inverted_file = None
    if abs(flipped_corr) > 0.12 and flipped_corr > current_corr + 0.04:
        adjusted_stem.phase_invert = not adjusted_stem.phase_invert
        adjusted_stem.audio = -adjusted_stem.audio
        phase_inverted_file = Path(adjusted_stem.path).name

    note = {
        "enabled": True,
        "rule": "apply_overhead_pair_phase_alignment",
        "operation_order": ["delay_first", "phase_check_second"],
        "phase_reference": "aligned_overhead_pair",
        "reference_rule": "latest_overhead_is_zero_ms",
        "left_file": Path(left_stem.path).name,
        "right_file": Path(right_stem.path).name,
        "farthest_overhead_file": Path(farthest_stem.path).name,
        "zero_ms_source": Path(farthest_stem.path).name,
        "measured_right_vs_left_delay_ms": round(measured_delay_ms, 3),
        "applied_left_delay_ms": round(left_stem.delay_ms, 3),
        "applied_right_delay_ms": round(right_stem.delay_ms, 3),
        "psr_db": round(float(measurement["psr"]), 2) if np.isfinite(measurement["psr"]) else None,
        "confidence": round(float(measurement["confidence"]), 3) if np.isfinite(measurement["confidence"]) else None,
        "coherence": round(float(measurement["coherence"]), 3) if np.isfinite(measurement["coherence"]) else None,
        "phase_invert_applied": phase_inverted_file is not None,
        "phase_inverted_file": phase_inverted_file,
        "corr_current": round(current_corr, 3),
        "corr_flipped": round(flipped_corr, 3),
    }
    left_stem.notes.append(note)
    right_stem.notes.append(note)
    return note


def _apply_drum_phase_alignment(
    stems: list[AyaicStem],
    *,
    max_delay_ms: float,
) -> list[dict[str, Any]]:
    overheads = [stem for stem in stems if _drum_instrument(stem.name) == "overhead"]
    if len(overheads) < 2:
        return [{"enabled": False, "reason": "overhead_pair_not_found"}]

    sample_rate = overheads[0].sample_rate
    overhead_signals = [
        _highpass(_mono(stem.audio), stem.sample_rate, 40.0)
        for stem in overheads
    ]
    reference = np.mean(np.vstack(overhead_signals), axis=0).astype(np.float32)
    zero_source = min(overheads, key=lambda stem: (stem.delay_ms, stem.channel_id))

    reports: list[dict[str, Any]] = []
    for stem in stems:
        instrument = _drum_instrument(stem.name)
        if instrument not in {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
            continue
        if stem.sample_rate != sample_rate:
            raise ValueError(f"{Path(stem.path).name}: sample rate mismatch {stem.sample_rate} != {sample_rate}")

        target = _highpass(_mono(stem.audio), stem.sample_rate, 40.0)
        start, measurement = _representative_delay_measurement(
            reference,
            target,
            stem.sample_rate,
            max_delay_ms=max_delay_ms,
            prefer_negative=True,
        )
        if not measurement:
            continue

        measured_delay_ms = float(measurement["delay_ms"])
        if measured_delay_ms < -0.3:
            stem.delay_ms = max(stem.delay_ms, min(max_delay_ms, -measured_delay_ms))
            stem.audio = _delay_audio(stem.audio, _samples_from_ms(stem.delay_ms, stem.sample_rate))

        aligned = _delay_mono(target, _samples_from_ms(stem.delay_ms, stem.sample_rate))
        window_len = int(3.0 * stem.sample_rate)
        corr_start = start
        corr_end = min(corr_start + window_len, len(reference), len(aligned))
        corr_ref = reference[corr_start:corr_end]
        corr_target = aligned[corr_start:corr_end]
        current_corr = _norm_corr(corr_ref, -corr_target if stem.phase_invert else corr_target)
        flipped_corr = _norm_corr(corr_ref, corr_target if stem.phase_invert else -corr_target)
        if abs(flipped_corr) > 0.12 and flipped_corr > current_corr + 0.04:
            stem.phase_invert = not stem.phase_invert
            stem.audio = -stem.audio

        note = {
            "channel": stem.channel_id,
            "file": Path(stem.path).name,
            "instrument": instrument,
            "reference": "overhead_pair",
            "operation_order": ["delay_first", "phase_check_second"],
            "phase_reference": "aligned_overhead_group",
            "reference_rule": "close_mic_to_aligned_overhead_group_with_zero_ms_source",
            "zero_ms_source": Path(zero_source.path).name,
            "measured_delay_ms": round(measured_delay_ms, 3),
            "applied_delay_ms": round(stem.delay_ms, 3),
            "psr_db": round(float(measurement["psr"]), 2) if np.isfinite(measurement["psr"]) else None,
            "confidence": round(float(measurement["confidence"]), 3) if np.isfinite(measurement["confidence"]) else None,
            "coherence": round(float(measurement["coherence"]), 3) if np.isfinite(measurement["coherence"]) else None,
            "phase_invert": stem.phase_invert,
            "corr_current": round(current_corr, 3),
            "corr_flipped": round(flipped_corr, 3),
        }
        stem.notes.append(note)
        reports.append(note)

    return reports


def _apply_snare_bleed_phase_alignment(stems: list[AyaicStem], *, max_delay_ms: float) -> dict[str, Any]:
    """Align reliable snare bleed arrivals, leaving the latest mic at 0 ms."""
    if len(stems) < 2:
        return _disabled_snare_bleed_report("not_enough_channels")

    sample_rate = stems[0].sample_rate
    for stem in stems:
        if stem.sample_rate != sample_rate:
            raise ValueError(f"{Path(stem.path).name}: sample rate mismatch {stem.sample_rate} != {sample_rate}")

    reference_stem = _snare_reference_stem(stems)
    if reference_stem is None:
        return _disabled_snare_bleed_report("snare_reference_not_found")

    event_starts = _snare_event_starts(stems, sample_rate)
    if not event_starts:
        return _disabled_snare_bleed_report(
            "snare_events_not_found",
            reference_stem=reference_stem,
        )

    original = {
        stem.channel_id: {
            "audio": ensure_stereo(stem.audio).copy(),
            "delay_ms": float(stem.delay_ms),
            "phase_invert": bool(stem.phase_invert),
        }
        for stem in stems
    }
    reference_signal = _snare_alignment_signal(reference_stem)
    candidates: list[dict[str, Any]] = [
        {
            "channel": reference_stem.channel_id,
            "file": Path(reference_stem.path).name,
            "role": _project_channel_role(reference_stem.name),
            "arrival_ms": 0.0,
            "delay_samples": 0.0,
            "psr": 99.0,
            "confidence": 1.0,
            "coherence": 1.0,
            "delay_stability_ms": 0.0,
            "window_count": len(event_starts),
            "corr_current": 1.0,
            "corr_flipped": -1.0,
            "source": "snare_reference",
        }
    ]
    rejected: list[dict[str, Any]] = []
    for stem in stems:
        if stem.channel_id == reference_stem.channel_id:
            continue
        role = _project_channel_role(stem.name)
        if not _snare_bleed_candidate_allowed(role):
            rejected.append(
                {
                    "channel": stem.channel_id,
                    "file": Path(stem.path).name,
                    "role": role,
                    "reason": "not_a_microphone_snare_bleed_candidate",
                }
            )
            continue
        measurement = _snare_bleed_arrival_measurement(
            reference_signal,
            _snare_alignment_signal(stem),
            sample_rate,
            starts=event_starts,
            max_delay_ms=max_delay_ms,
        )
        reason = _snare_bleed_rejection_reason(measurement, role)
        payload = {
            "channel": stem.channel_id,
            "file": Path(stem.path).name,
            "role": role,
            **measurement,
        }
        if reason:
            payload["reason"] = reason
            rejected.append(_round_snare_bleed_item(payload))
            continue
        candidates.append(payload)

    reliable = [item for item in candidates if str(item.get("source")) == "snare_reference" or item.get("arrival_ms") is not None]
    if len(reliable) < 2:
        return _disabled_snare_bleed_report(
            "not_enough_reliable_snare_bleed_mics",
            reference_stem=reference_stem,
            candidates=candidates,
            rejected=rejected,
            event_starts=event_starts,
        )

    latest = max(reliable, key=lambda item: (float(item["arrival_ms"]), float(item.get("confidence", 0.0))))
    latest_arrival_ms = float(latest["arrival_ms"])
    delays = {
        int(item["channel"]): max(0.0, latest_arrival_ms - float(item["arrival_ms"]))
        for item in reliable
    }
    limited_channels = []
    for channel, delay_ms in list(delays.items()):
        if delay_ms > max_delay_ms:
            delays[channel] = max_delay_ms
            limited_channels.append(channel)

    before_score = _snare_bleed_alignment_score(
        stems,
        channels=sorted(delays),
        zero_channel=int(latest["channel"]),
        starts=event_starts,
        sample_rate=sample_rate,
        zero_arrival_ms=latest_arrival_ms,
    )
    actions = _apply_snare_bleed_delays_and_polarity(
        stems,
        delays,
        zero_channel=int(latest["channel"]),
        zero_arrival_ms=latest_arrival_ms,
        event_starts=event_starts,
        sample_rate=sample_rate,
        max_delay_ms=max_delay_ms,
        limited_channels=set(limited_channels),
    )
    after_score = _snare_bleed_alignment_score(
        stems,
        channels=sorted(delays),
        zero_channel=int(latest["channel"]),
        starts=event_starts,
        sample_rate=sample_rate,
        zero_arrival_ms=latest_arrival_ms,
    )
    rollback = bool(actions and after_score < before_score - 0.03)
    if rollback:
        for stem in stems:
            saved = original[stem.channel_id]
            stem.audio = saved["audio"].copy()
            stem.delay_ms = float(saved["delay_ms"])
            stem.phase_invert = bool(saved["phase_invert"])
        for action in actions:
            action["rolled_back"] = True
            action["rollback_reason"] = "snare_bleed_alignment_score_worse_after_delay"
        after_score = _snare_bleed_alignment_score(
            stems,
            channels=sorted(delays),
            zero_channel=int(latest["channel"]),
            starts=event_starts,
            sample_rate=sample_rate,
            zero_arrival_ms=latest_arrival_ms,
        )

    alignment = {
        "enabled": True,
        "method": "snare_bleed_phase_alignment",
        "reference_file": Path(reference_stem.path).name,
        "reference_channel": reference_stem.channel_id,
        "zero_ms_source": str(latest["file"]),
        "zero_ms_channel": int(latest["channel"]),
        "zero_ms_rule": "latest_reliable_snare_arrival_is_0ms",
        "latest_arrival_ms": round(latest_arrival_ms, 4),
        "event_count": len(event_starts),
        "reliable_mic_count": len(reliable),
        "candidate_mics": [_round_snare_bleed_item(item) for item in reliable],
        "rejected_mics": rejected[:80],
        "limited_channels": sorted(limited_channels),
        "before_alignment_score": round(float(before_score), 4),
        "after_alignment_score": round(float(after_score), 4),
        "alignment_score_delta": round(float(after_score - before_score), 4),
        "rollback": rollback,
        "notes": [
            "Snare transients are measured in every reliable microphone channel.",
            "The microphone with the latest reliable snare arrival stays at 0 ms.",
            "Earlier arrivals receive positive offline fractional delay.",
            "Polarity is checked only after delay alignment.",
        ],
    }
    report = {
        "order": [
            "detect_snare_transients",
            "measure_snare_bleed_arrivals",
            "select_latest_snare_arrival_zero_ms",
            "apply_fractional_delay_and_polarity",
            "validate_snare_bleed_alignment",
        ],
        "overhead_pair_alignment": _snare_bleed_overhead_report(stems, reliable, delays, latest),
        "phase_alignment": actions,
        "global_phase_alignment": {
            "enabled": False,
            "reason": "snare_bleed_phase_alignment_used",
            "fallback_available": True,
        },
        "snare_bleed_alignment": alignment,
        "drum_pan_rule": {"enabled": False, "reason": "panning_not_used_in_phase_delay_stage"},
    }
    for stem in stems:
        if stem.channel_id in delays:
            stem.notes.append(
                {
                    "type": "snare_bleed_phase_alignment",
                    "delay_ms": round(stem.delay_ms, 4),
                    "phase_invert": bool(stem.phase_invert),
                    "zero_ms_source": str(latest["file"]),
                    "rolled_back": rollback,
                }
            )
    return report


def _disabled_snare_bleed_report(
    reason: str,
    *,
    reference_stem: AyaicStem | None = None,
    candidates: list[dict[str, Any]] | None = None,
    rejected: list[dict[str, Any]] | None = None,
    event_starts: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "order": ["snare_bleed_phase_alignment"],
        "overhead_pair_alignment": {"enabled": False, "reason": reason},
        "phase_alignment": [],
        "global_phase_alignment": {"enabled": False, "reason": reason},
        "snare_bleed_alignment": {
            "enabled": False,
            "method": "snare_bleed_phase_alignment",
            "reason": reason,
            "reference_file": Path(reference_stem.path).name if reference_stem else None,
            "reference_channel": reference_stem.channel_id if reference_stem else None,
            "event_count": len(event_starts or []),
            "candidate_mics": [_round_snare_bleed_item(item) for item in candidates or []],
            "rejected_mics": rejected or [],
        },
        "drum_pan_rule": {"enabled": False, "reason": "panning_not_used_in_phase_delay_stage"},
    }


def _snare_reference_stem(stems: list[AyaicStem]) -> AyaicStem | None:
    snare_stems = [stem for stem in stems if _drum_instrument(stem.name) == "snare"]
    if not snare_stems:
        return None
    preferred_keys = {"snare t", "snare top", "snare"}
    return next((stem for stem in snare_stems if _name_key(stem.name) in preferred_keys), snare_stems[0])


def _snare_alignment_signal(stem: AyaicStem) -> np.ndarray:
    # Snare alignment is explicitly about leakage across microphones, so it
    # must use the full rendered stem rather than primary-loudness windows.
    mono = _mono(stem.audio)
    focused = _highpass(mono, stem.sample_rate, 120.0)
    focused = _lowpass(focused, stem.sample_rate, 6500.0)
    return np.asarray(focused, dtype=np.float32)


def _snare_bleed_candidate_allowed(role: str) -> bool:
    return role not in {"playback"}


def _snare_bleed_arrival_measurement(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    *,
    starts: list[int],
    max_delay_ms: float,
) -> dict[str, Any]:
    window_len = max(1024, int(0.170 * sample_rate))
    measurements: list[dict[str, float]] = []
    for start in starts:
        window = min(len(reference) - start, len(target) - start, window_len)
        if window < max(512, window_len // 3):
            continue
        ref_seg = reference[start:start + window]
        tgt_seg = target[start:start + window]
        fft_size = _next_power_of_two(len(ref_seg))
        if fft_size != len(ref_seg):
            ref_seg = np.pad(ref_seg, (0, fft_size - len(ref_seg)))
            tgt_seg = np.pad(tgt_seg, (0, fft_size - len(tgt_seg)))
        measurement = _gcc_phat_measurement(
            ref_seg,
            tgt_seg,
            sample_rate,
            fft_size=fft_size,
            max_delay_ms=max_delay_ms,
        )
        delay_ms = float(measurement["delay_ms"])
        if not np.isfinite(delay_ms) or abs(abs(delay_ms) - max_delay_ms) < 0.05:
            continue
        corr_current, corr_flipped = _corr_for_measured_delay(ref_seg, tgt_seg, delay_ms, sample_rate)
        measurements.append(
            {
                **measurement,
                "arrival_ms": delay_ms,
                "start": float(start),
                "corr_current": float(corr_current),
                "corr_flipped": float(corr_flipped),
                "corr_abs": float(max(abs(corr_current), abs(corr_flipped))),
            }
        )

    if not measurements:
        return {
            "arrival_ms": None,
            "delay_samples": None,
            "psr": 0.0,
            "confidence": 0.0,
            "coherence": 0.0,
            "delay_stability_ms": 999.0,
            "window_count": 0,
            "corr_current": 0.0,
            "corr_flipped": 0.0,
        }
    arrivals = np.array([float(item["arrival_ms"]) for item in measurements], dtype=np.float64)
    median_arrival = float(np.median(arrivals))
    selected = min(measurements, key=lambda item: abs(float(item["arrival_ms"]) - median_arrival))
    stability = float(np.median(np.abs(arrivals - median_arrival))) if len(arrivals) > 1 else 0.0
    psr = float(np.median([float(item["psr"]) for item in measurements]))
    confidence = float(np.median([float(item["confidence"]) for item in measurements]))
    coherence = float(np.median([float(item["coherence"]) for item in measurements]))
    corr_current = float(np.median([float(item["corr_current"]) for item in measurements]))
    corr_flipped = float(np.median([float(item["corr_flipped"]) for item in measurements]))
    corr_abs = max(abs(corr_current), abs(corr_flipped))
    stability_score = max(0.0, min(1.0, 1.0 - stability / max(0.35, max_delay_ms * 0.25)))
    combined_confidence = 0.45 * confidence + 0.30 * min(1.0, corr_abs / 0.45) + 0.25 * stability_score
    return {
        "arrival_ms": median_arrival,
        "delay_samples": median_arrival * sample_rate / 1000.0,
        "psr": psr,
        "confidence": float(max(0.0, min(1.0, combined_confidence))),
        "raw_confidence": confidence,
        "coherence": coherence,
        "corr_abs": corr_abs,
        "delay_stability_ms": stability,
        "window_count": len(measurements),
        "representative_start_sample": int(selected["start"]),
        "corr_current": corr_current,
        "corr_flipped": corr_flipped,
    }


def _snare_bleed_rejection_reason(measurement: dict[str, Any], role: str) -> str | None:
    if measurement.get("arrival_ms") is None:
        return "no_valid_snare_bleed_windows"
    windows = int(measurement.get("window_count", 0))
    confidence = float(measurement.get("confidence", 0.0))
    corr_abs = float(measurement.get("corr_abs", max(abs(float(measurement.get("corr_current", 0.0))), abs(float(measurement.get("corr_flipped", 0.0))))))
    psr = float(measurement.get("psr", 0.0))
    stability = float(measurement.get("delay_stability_ms", 999.0))
    drum_or_ambience = role in {"kick", "snare", "snare_top", "snare_bottom", "tom", "floor_tom", "overhead", "room"}
    if windows == 1 and confidence >= 0.55 and corr_abs >= 0.20 and psr >= 0.7:
        return None
    if windows < (1 if drum_or_ambience else 2):
        return "not_enough_snare_bleed_windows"
    if stability > (1.8 if drum_or_ambience else 1.15):
        return "snare_arrival_unstable_across_hits"
    if psr < (0.55 if drum_or_ambience else 1.0):
        return "weak_snare_gcc_peak"
    if confidence < (0.20 if drum_or_ambience else 0.34):
        return "confidence_below_snare_bleed_threshold"
    if corr_abs < (0.08 if drum_or_ambience else 0.16):
        return "correlation_below_snare_bleed_threshold"
    return None


def _apply_snare_bleed_delays_and_polarity(
    stems: list[AyaicStem],
    delays: dict[int, float],
    *,
    zero_channel: int,
    zero_arrival_ms: float,
    event_starts: list[int],
    sample_rate: int,
    max_delay_ms: float,
    limited_channels: set[int],
) -> list[dict[str, Any]]:
    channel_to_stem = {stem.channel_id: stem for stem in stems}
    actions: list[dict[str, Any]] = []
    for channel, delay_ms in sorted(delays.items()):
        stem = channel_to_stem[channel]
        if delay_ms > 0.05:
            stem.audio = _delay_audio_fractional(stem.audio, delay_ms, sample_rate)
            stem.delay_ms += delay_ms
        actions.append(
            {
                "channel": channel,
                "file": Path(stem.path).name,
                "instrument": _project_channel_role(stem.name),
                "reference": "snare_bleed_latest_arrival",
                "operation_order": ["fractional_delay_first", "polarity_check_second"],
                "phase_reference": "snare_bleed_transient_arrival",
                "reference_rule": "latest_reliable_snare_arrival_is_zero_ms",
                "zero_ms_source_channel": zero_channel,
                "zero_ms_source": Path(channel_to_stem[zero_channel].path).name,
                "applied_delay_ms": round(float(stem.delay_ms), 4),
                "applied_incremental_delay_ms": round(float(delay_ms), 4),
                "delay_limited_by_max_phase_delay": channel in limited_channels,
                "phase_invert": bool(stem.phase_invert),
                "phase_invert_applied": False,
                "rolled_back": False,
            }
        )

    zero_signal = _snare_alignment_signal(channel_to_stem[zero_channel])
    polarity_starts = [
        int(np.clip(start + _samples_from_ms(max(0.0, zero_arrival_ms), sample_rate), 0, max(0, len(zero_signal) - 1)))
        for start in event_starts
    ]
    for action in actions:
        channel = int(action["channel"])
        if channel == zero_channel:
            action["corr_current"] = None
            action["corr_flipped"] = None
            continue
        stem = channel_to_stem[channel]
        target = _snare_alignment_signal(stem)
        polarity = _snare_polarity_decision(
            zero_signal,
            target,
            sample_rate,
            starts=polarity_starts,
            max_lag_ms=min(max_delay_ms, 1.5),
        )
        if polarity["should_invert"]:
            stem.audio = -stem.audio
            stem.phase_invert = not stem.phase_invert
            action["phase_invert"] = bool(stem.phase_invert)
            action["phase_invert_applied"] = True
        action["corr_current"] = round(float(polarity["current_corr"]), 4)
        action["corr_flipped"] = round(float(polarity["flipped_corr"]), 4)
        action["polarity_best_lag_ms"] = round(float(polarity["best_lag_ms"]), 4)
    return [
        action for action in actions
        if action["applied_incremental_delay_ms"] > 0.05 or action.get("phase_invert_applied")
    ]


def _snare_bleed_alignment_score(
    stems: list[AyaicStem],
    *,
    channels: list[int],
    zero_channel: int,
    starts: list[int],
    sample_rate: int,
    zero_arrival_ms: float,
) -> float:
    channel_to_stem = {stem.channel_id: stem for stem in stems}
    if zero_channel not in channel_to_stem:
        return 0.0
    zero_signal = _snare_alignment_signal(channel_to_stem[zero_channel])
    window = max(512, int(0.170 * sample_rate))
    shifted_starts = [
        int(np.clip(start + _samples_from_ms(max(0.0, zero_arrival_ms), sample_rate), 0, max(0, len(zero_signal) - window)))
        for start in starts
    ]
    scores = []
    for channel in channels:
        if channel == zero_channel or channel not in channel_to_stem:
            continue
        target = _snare_alignment_signal(channel_to_stem[channel])
        values = []
        for start in shifted_starts:
            end = min(start + window, len(zero_signal), len(target))
            if end - start < window // 3:
                continue
            values.append(abs(_norm_corr(zero_signal[start:end], target[start:end])))
        if values:
            scores.append(float(np.median(values)))
    return float(np.median(scores)) if scores else 0.0


def _snare_bleed_overhead_report(
    stems: list[AyaicStem],
    reliable: list[dict[str, Any]],
    delays: dict[int, float],
    latest: dict[str, Any],
) -> dict[str, Any]:
    overheads = [stem for stem in stems if _drum_instrument(stem.name) == "overhead"]
    if not overheads:
        return {
            "enabled": False,
            "reason": "overhead_pair_not_found",
            "rule": "snare_bleed_latest_arrival_zero_ms",
            "zero_ms_source": str(latest["file"]),
        }
    reliable_by_channel = {int(item["channel"]): item for item in reliable}
    return {
        "enabled": any(stem.channel_id in reliable_by_channel for stem in overheads),
        "rule": "snare_bleed_latest_arrival_zero_ms",
        "phase_reference": "snare_transient_bleed",
        "reference_rule": "latest_reliable_snare_arrival_is_zero_ms",
        "zero_ms_source": str(latest["file"]),
        "zero_ms_channel": int(latest["channel"]),
        "overheads": [
            {
                "channel": stem.channel_id,
                "file": Path(stem.path).name,
                "reliable": stem.channel_id in reliable_by_channel,
                "arrival_ms": round(float(reliable_by_channel[stem.channel_id]["arrival_ms"]), 4)
                if stem.channel_id in reliable_by_channel else None,
                "applied_delay_ms": round(float(delays.get(stem.channel_id, 0.0)), 4),
            }
            for stem in overheads
        ],
    }


def _round_snare_bleed_item(item: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        elif isinstance(value, np.floating):
            out[key] = round(float(value), 4)
        else:
            out[key] = value
    return out


def _apply_global_bleed_phase_alignment(stems: list[AyaicStem], *, max_delay_ms: float) -> dict[str, Any]:
    """Align all strongly correlated bleed/shared-mic groups, not only drums."""
    if len(stems) < 2:
        return {
            "order": ["global_bleed_phase_alignment"],
            "overhead_pair_alignment": {"enabled": False, "reason": "not_enough_channels"},
            "phase_alignment": [],
            "global_phase_alignment": {"enabled": False, "reason": "not_enough_channels"},
            "drum_pan_rule": {"enabled": False, "reason": "panning_not_used_in_phase_delay_stage"},
        }

    sample_rate = stems[0].sample_rate
    for stem in stems:
        if stem.sample_rate != sample_rate:
            raise ValueError(f"{Path(stem.path).name}: sample rate mismatch {stem.sample_rate} != {sample_rate}")

    original = {
        stem.channel_id: {
            "audio": ensure_stereo(stem.audio).copy(),
            "delay_ms": float(stem.delay_ms),
            "phase_invert": bool(stem.phase_invert),
        }
        for stem in stems
    }
    phase_signals = {stem.channel_id: _phase_alignment_signal(stem) for stem in stems}
    edges, rejected_edges = _global_phase_candidate_edges(
        stems,
        phase_signals,
        sample_rate=sample_rate,
        max_delay_ms=max_delay_ms,
    )
    if not edges:
        return {
            "order": ["build_global_bleed_correlation_graph"],
            "overhead_pair_alignment": _global_overhead_report(stems, [], {}),
            "phase_alignment": [],
            "global_phase_alignment": {
                "enabled": False,
                "method": "global_bleed_phase_alignment",
                "reason": "no_reliable_correlation_edges",
                "candidate_edge_count": 0,
                "rejected_edges": rejected_edges,
            },
            "drum_pan_rule": {"enabled": False, "reason": "panning_not_used_in_phase_delay_stage"},
        }

    components = _global_phase_components(stems, edges)
    solved_offsets = _solve_global_phase_offsets(stems, edges, components, max_delay_ms=max_delay_ms)
    before_score = _global_phase_graph_score(stems, edges)
    actions = _apply_global_phase_offsets_and_polarity(
        stems,
        edges,
        solved_offsets,
        sample_rate=sample_rate,
    )
    after_score = _global_phase_graph_score(stems, edges)
    improvement = after_score - before_score
    rollback = bool(actions and improvement < -0.015)
    if rollback:
        for stem in stems:
            saved = original[stem.channel_id]
            stem.audio = saved["audio"].copy()
            stem.delay_ms = float(saved["delay_ms"])
            stem.phase_invert = bool(saved["phase_invert"])
        after_score = _global_phase_graph_score(stems, edges)
        for action in actions:
            action["rolled_back"] = True
            action["rollback_reason"] = "global_correlation_score_worse_after_alignment"

    overhead_report = _global_overhead_report(stems, edges, solved_offsets)
    report = {
        "order": [
            "build_global_bleed_correlation_graph",
            "solve_global_delay_offsets",
            "apply_fractional_delay_and_polarity",
            "validate_global_phase_alignment",
        ],
        "overhead_pair_alignment": overhead_report,
        "phase_alignment": actions,
        "global_phase_alignment": {
            "enabled": True,
            "method": "global_bleed_phase_alignment",
            "reference_scope": "all_high_correlation_bleed_edges",
            "candidate_edge_count": len(edges),
            "rejected_edge_count": len(rejected_edges),
            "components": components,
            "solved_offsets_ms": {
                str(channel): round(float(offset), 4)
                for channel, offset in sorted(solved_offsets.items())
            },
            "before_graph_score": round(before_score, 4),
            "after_graph_score": round(after_score, 4),
            "graph_score_delta": round(after_score - before_score, 4),
            "rollback": rollback,
            "edges": [_round_global_phase_edge(edge) for edge in edges],
            "rejected_edges": rejected_edges[:80],
            "notes": [
                "Uses all reliable cross-channel bleed/correlation edges, not only drums.",
                "Relative negative offsets are converted into non-negative offline delays per connected component.",
                "Fractional delay is applied offline; live implementations must map to mixer delay resolution.",
                "Room/overhead natural spacing is protected by stricter confidence and lower edge weights.",
            ],
        },
        "drum_pan_rule": {"enabled": False, "reason": "panning_not_used_in_phase_delay_stage"},
    }
    for stem in stems:
        if stem.channel_id in solved_offsets:
            stem.notes.append(
                {
                    "type": "global_bleed_phase_alignment",
                    "delay_ms": round(stem.delay_ms, 4),
                    "phase_invert": bool(stem.phase_invert),
                    "component": _component_index_for_channel(components, stem.channel_id),
                    "rolled_back": rollback,
                }
            )
    return report


def _phase_alignment_signal(stem: AyaicStem) -> np.ndarray:
    mono = _mono(_stem_primary_loudness_audio(stem))
    if len(mono) < int(0.5 * stem.sample_rate):
        mono = _mono(stem.audio)
    return _highpass(mono, stem.sample_rate, 40.0)


def _global_phase_candidate_edges(
    stems: list[AyaicStem],
    phase_signals: dict[int, np.ndarray],
    *,
    sample_rate: int,
    max_delay_ms: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edges: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for left_index, left_stem in enumerate(stems):
        for right_stem in stems[left_index + 1:]:
            relation = _phase_pair_relation(left_stem, right_stem)
            left_role = _project_channel_role(left_stem.name)
            right_role = _project_channel_role(right_stem.name)
            measurement = _global_phase_pair_measurement(
                phase_signals[left_stem.channel_id],
                phase_signals[right_stem.channel_id],
                sample_rate,
                max_delay_ms=max_delay_ms,
            )
            source_stem = left_stem
            target_stem = right_stem
            source_role = left_role
            target_role = right_role
            if (
                relation == "drum_bleed_reference"
                and _is_phase_ambience_role(left_role)
                and _is_phase_close_drum_role(right_role)
            ):
                source_stem = right_stem
                target_stem = left_stem
                source_role = right_role
                target_role = left_role
                measurement = {
                    **measurement,
                    "delay_ms": -float(measurement["delay_ms"]),
                    "delay_samples": -float(measurement["delay_samples"]),
                }
            reason = _global_phase_edge_rejection_reason(measurement, relation)
            payload = {
                "source_channel": source_stem.channel_id,
                "target_channel": target_stem.channel_id,
                "source_file": Path(source_stem.path).name,
                "target_file": Path(target_stem.path).name,
                "source_role": source_role,
                "target_role": target_role,
                "relation": relation,
                **measurement,
            }
            if reason:
                payload["reason"] = reason
                rejected.append(_round_global_phase_edge(payload))
                continue
            weight = _global_phase_edge_weight(measurement, relation)
            edge = {
                **payload,
                "desired_target_minus_source_delay_ms": -float(measurement["delay_ms"]),
                "weight": weight,
            }
            edges.append(edge)
    edges.sort(key=lambda item: (-float(item["weight"]), item["source_channel"], item["target_channel"]))
    return edges, rejected


def _is_phase_close_drum_role(role: str) -> bool:
    return role in {"kick", "snare", "snare_top", "snare_bottom", "tom", "floor_tom"}


def _is_phase_ambience_role(role: str) -> bool:
    return role in {"overhead", "room"}


def _phase_pair_relation(first: AyaicStem, second: AyaicStem) -> str:
    first_key = _name_key(first.name)
    second_key = _name_key(second.name)
    first_role = _project_channel_role(first.name)
    second_role = _project_channel_role(second.name)
    if _stereo_pair_base(first_key) and _stereo_pair_base(first_key) == _stereo_pair_base(second_key):
        return "linked_stereo_pair"
    if (
        _is_phase_close_drum_role(first_role) and _is_phase_ambience_role(second_role)
        or _is_phase_close_drum_role(second_role) and _is_phase_ambience_role(first_role)
    ):
        return "drum_bleed_reference"
    if first.group == second.group and first.group:
        return f"same_group:{first.group}"
    if first_role in {"overhead", "room"} or second_role in {"overhead", "room"}:
        return "ambience_bleed_reference"
    if first_role == second_role:
        return f"same_role:{first_role}"
    return "cross_bleed"


def _stereo_pair_base(key: str) -> str | None:
    for suffix in (" l", " r", " left", " right"):
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def _global_phase_pair_measurement(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    *,
    max_delay_ms: float,
) -> dict[str, Any]:
    starts = _shared_active_segment_starts(reference, target, sample_rate)
    window_len = int(2.0 * sample_rate)
    measurements: list[dict[str, float]] = []
    for start in starts:
        window = min(len(reference) - start, len(target) - start, window_len)
        if window < int(0.35 * sample_rate):
            continue
        ref_seg = reference[start : start + window]
        tgt_seg = target[start : start + window]
        fft_size = _next_power_of_two(len(ref_seg))
        if fft_size != len(ref_seg):
            ref_seg = np.pad(ref_seg, (0, fft_size - len(ref_seg)))
            tgt_seg = np.pad(tgt_seg, (0, fft_size - len(tgt_seg)))
        measurement = _gcc_phat_measurement(
            ref_seg,
            tgt_seg,
            sample_rate,
            fft_size=fft_size,
            max_delay_ms=max_delay_ms,
        )
        delay_ms = float(measurement["delay_ms"])
        if not np.isfinite(delay_ms) or abs(abs(delay_ms) - max_delay_ms) < 0.05:
            continue
        corr_current, corr_flipped = _corr_for_measured_delay(
            ref_seg,
            tgt_seg,
            delay_ms,
            sample_rate,
        )
        measurements.append(
            {
                **measurement,
                "start": float(start),
                "corr_current": float(corr_current),
                "corr_flipped": float(corr_flipped),
                "corr_abs": float(max(abs(corr_current), abs(corr_flipped))),
            }
        )

    if not measurements:
        return {
            "delay_ms": 0.0,
            "delay_samples": 0.0,
            "psr": 0.0,
            "confidence": 0.0,
            "coherence": 0.0,
            "delay_stability_ms": 999.0,
            "window_count": 0,
            "corr_current": 0.0,
            "corr_flipped": 0.0,
            "polarity_hint": "normal",
        }

    delays = np.array([float(item["delay_ms"]) for item in measurements], dtype=np.float64)
    median_delay = float(np.median(delays))
    selected = min(measurements, key=lambda item: abs(float(item["delay_ms"]) - median_delay))
    stability = float(np.median(np.abs(delays - median_delay))) if len(delays) > 1 else 0.0
    psr = float(np.median([float(item["psr"]) for item in measurements]))
    corr_current = float(np.median([float(item["corr_current"]) for item in measurements]))
    corr_flipped = float(np.median([float(item["corr_flipped"]) for item in measurements]))
    corr_abs = max(abs(corr_current), abs(corr_flipped))
    stability_score = max(0.0, min(1.0, 1.0 - stability / max(0.5, max_delay_ms * 0.35)))
    psr_score = max(0.0, min(1.0, psr / 8.0))
    confidence = 0.52 * min(1.0, corr_abs / 0.55) + 0.25 * psr_score + 0.23 * stability_score
    return {
        "delay_ms": median_delay,
        "delay_samples": median_delay * sample_rate / 1000.0,
        "psr": psr,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "coherence": corr_abs,
        "delay_stability_ms": stability,
        "window_count": len(measurements),
        "representative_start_sample": int(selected["start"]),
        "corr_current": corr_current,
        "corr_flipped": corr_flipped,
        "polarity_hint": "invert" if corr_flipped > corr_current + 0.04 and abs(corr_flipped) > 0.12 else "normal",
    }


def _shared_active_segment_starts(first: np.ndarray, second: np.ndarray, sample_rate: int, count: int = 8) -> list[int]:
    n = min(len(first), len(second))
    if n <= int(0.35 * sample_rate):
        return [0]
    window = min(n, max(1024, int(2.0 * sample_rate)))
    hop = max(512, window // 3)
    scored: list[tuple[float, int]] = []
    first_arr = np.asarray(first[:n], dtype=np.float32)
    second_arr = np.asarray(second[:n], dtype=np.float32)
    for start in range(0, max(1, n - window + 1), hop):
        end = start + window
        first_rms = _rms_linear(first_arr[start:end])
        second_rms = _rms_linear(second_arr[start:end])
        shared = min(first_rms, second_rms)
        total = first_rms + second_rms
        if shared <= db_to_amp(-70.0) or total <= db_to_amp(-68.0):
            continue
        scored.append((shared * total, start))
    if not scored:
        return _active_segment_starts(first_arr + second_arr, sample_rate, count=count)
    starts = [start for _, start in sorted(scored, reverse=True)[:count]]
    return sorted(set(starts))


def _corr_for_measured_delay(reference: np.ndarray, target: np.ndarray, delay_ms: float, sample_rate: int) -> tuple[float, float]:
    if delay_ms < 0.0:
        ref = np.asarray(reference, dtype=np.float32)
        tgt = _delay_mono_fractional(target, -delay_ms, sample_rate)
    else:
        ref = _delay_mono_fractional(reference, delay_ms, sample_rate)
        tgt = np.asarray(target, dtype=np.float32)
    current = _norm_corr(ref, tgt)
    flipped = _norm_corr(ref, -tgt)
    return current, flipped


def _global_phase_edge_rejection_reason(measurement: dict[str, Any], relation: str) -> str | None:
    confidence = float(measurement.get("confidence", 0.0))
    coherence = float(measurement.get("coherence", 0.0))
    stability = float(measurement.get("delay_stability_ms", 999.0))
    windows = int(measurement.get("window_count", 0))
    psr = float(measurement.get("psr", 0.0))
    if windows <= 0:
        return "no_valid_windows"
    if relation == "drum_bleed_reference":
        if float(measurement.get("delay_ms", 0.0)) < 0.15:
            return "close_mic_not_earlier_than_bleed_reference"
        if windows == 1 and confidence >= 0.30 and coherence >= 0.25 and psr >= 0.65:
            return None
        if windows < 3:
            return "not_enough_stable_bleed_windows"
        if stability > 1.35:
            return "delay_unstable_across_windows"
        if psr < 0.65:
            return "weak_gcc_peak"
        if confidence < 0.18:
            return "confidence_below_global_phase_threshold"
        if coherence < 0.012:
            return "correlation_below_bleed_threshold"
        return None
    if relation.startswith("same_group:drums") and windows >= 3:
        if stability <= 0.75 and psr >= 1.2 and confidence >= 0.25 and coherence >= 0.04:
            return None
    min_confidence = 0.34
    min_coherence = 0.18
    if relation in {"ambience_bleed_reference", "cross_bleed"}:
        min_confidence = 0.40
        min_coherence = 0.24
    if relation == "linked_stereo_pair":
        min_confidence = 0.30
        min_coherence = 0.16
    if confidence < min_confidence:
        return "confidence_below_global_phase_threshold"
    if coherence < min_coherence:
        return "correlation_below_bleed_threshold"
    if stability > 2.25 and windows > 1:
        return "delay_unstable_across_windows"
    if psr < 0.5 and coherence < 0.35:
        return "weak_gcc_peak"
    return None


def _global_phase_edge_weight(measurement: dict[str, Any], relation: str) -> float:
    weight = float(measurement.get("confidence", 0.0))
    weight *= 0.65 + 0.7 * float(measurement.get("coherence", 0.0))
    if relation == "linked_stereo_pair":
        weight *= 1.2
    elif relation == "drum_bleed_reference":
        weight *= 0.82
    elif relation.startswith("same_group"):
        weight *= 1.1
    elif relation == "ambience_bleed_reference":
        weight *= 0.72
    elif relation == "cross_bleed":
        weight *= 0.85
    return round(float(max(0.01, min(2.0, weight))), 4)


def _global_phase_components(stems: list[AyaicStem], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    channels = {stem.channel_id for stem in stems}
    parent = {channel: channel for channel in channels}

    def find(channel: int) -> int:
        while parent[channel] != channel:
            parent[channel] = parent[parent[channel]]
            channel = parent[channel]
        return channel

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        union(int(edge["source_channel"]), int(edge["target_channel"]))
    grouped: dict[int, list[int]] = {}
    for channel in channels:
        grouped.setdefault(find(channel), []).append(channel)
    channel_to_stem = {stem.channel_id: stem for stem in stems}
    components = []
    for index, members in enumerate(sorted(grouped.values(), key=lambda item: (-len(item), min(item)))):
        if len(members) < 2:
            continue
        anchor = _global_phase_anchor([channel_to_stem[channel] for channel in members])
        components.append(
            {
                "index": index,
                "channels": sorted(members),
                "files": [Path(channel_to_stem[channel].path).name for channel in sorted(members)],
                "anchor_channel": anchor.channel_id,
                "anchor_file": Path(anchor.path).name,
            }
        )
    return components


def _global_phase_anchor(component_stems: list[AyaicStem]) -> AyaicStem:
    role_rank = {
        "overhead": 0,
        "room": 1,
        "lead_vocal": 2,
        "kick": 3,
        "snare": 4,
        "bass": 5,
        "guitar": 6,
        "accordion": 6,
        "keys": 6,
        "playback": 7,
        "back_vocal": 8,
    }
    return sorted(
        component_stems,
        key=lambda stem: (role_rank.get(_project_channel_role(stem.name), 20), stem.channel_id),
    )[0]


def _solve_global_phase_offsets(
    stems: list[AyaicStem],
    edges: list[dict[str, Any]],
    components: list[dict[str, Any]],
    *,
    max_delay_ms: float,
) -> dict[int, float]:
    offsets: dict[int, float] = {}
    component_by_channel = {
        channel: component
        for component in components
        for channel in component["channels"]
    }
    edge_by_component: dict[int, list[dict[str, Any]]] = {}
    for edge in edges:
        component = component_by_channel.get(int(edge["source_channel"]))
        if component and int(edge["target_channel"]) in component["channels"]:
            edge_by_component.setdefault(int(component["index"]), []).append(edge)

    for component in components:
        channels = list(component["channels"])
        anchor = int(component["anchor_channel"])
        variables = [channel for channel in channels if channel != anchor]
        if not variables:
            offsets[anchor] = 0.0
            continue
        variable_index = {channel: idx for idx, channel in enumerate(variables)}
        rows = []
        values = []
        for edge in edge_by_component.get(int(component["index"]), []):
            row = np.zeros(len(variables), dtype=np.float64)
            src = int(edge["source_channel"])
            tgt = int(edge["target_channel"])
            if tgt != anchor:
                row[variable_index[tgt]] += 1.0
            if src != anchor:
                row[variable_index[src]] -= 1.0
            weight = math.sqrt(float(edge.get("weight", 1.0)))
            rows.append(row * weight)
            values.append(float(edge["desired_target_minus_source_delay_ms"]) * weight)
        if not rows:
            for channel in channels:
                offsets[channel] = 0.0
            continue
        solution, *_ = np.linalg.lstsq(np.vstack(rows), np.asarray(values, dtype=np.float64), rcond=None)
        relative = {anchor: 0.0}
        for channel, idx in variable_index.items():
            relative[channel] = float(solution[idx])
        min_offset = min(relative.values())
        shifted = {channel: value - min_offset for channel, value in relative.items()}
        max_offset = max(shifted.values()) if shifted else 0.0
        if max_offset > max_delay_ms:
            scale = max_delay_ms / max_offset
            shifted = {channel: value * scale for channel, value in shifted.items()}
        for channel, value in shifted.items():
            offsets[channel] = round(float(max(0.0, min(max_delay_ms, value))), 5)
    return offsets


def _apply_global_phase_offsets_and_polarity(
    stems: list[AyaicStem],
    edges: list[dict[str, Any]],
    offsets: dict[int, float],
    *,
    sample_rate: int,
) -> list[dict[str, Any]]:
    channel_to_stem = {stem.channel_id: stem for stem in stems}
    actions: list[dict[str, Any]] = []
    for channel, delay_ms in sorted(offsets.items()):
        stem = channel_to_stem[channel]
        if delay_ms > 0.05:
            stem.audio = _delay_audio_fractional(stem.audio, delay_ms, sample_rate)
            stem.delay_ms += delay_ms
        action_edges = [
            edge for edge in edges
            if int(edge["source_channel"]) == channel or int(edge["target_channel"]) == channel
        ]
        action = {
            "channel": channel,
            "file": Path(stem.path).name,
            "instrument": _project_channel_role(stem.name),
            "reference": "global_bleed_correlation_graph",
            "operation_order": ["fractional_delay_first", "polarity_check_second"],
            "phase_reference": "weighted_correlation_component",
            "measured_edge_count": len(action_edges),
            "applied_delay_ms": round(float(stem.delay_ms), 4),
            "applied_incremental_delay_ms": round(float(delay_ms), 4),
            "phase_invert": bool(stem.phase_invert),
            "rolled_back": False,
        }
        actions.append(action)

    delayed_signals = {stem.channel_id: _phase_alignment_signal(stem) for stem in stems}
    for action in actions:
        channel = int(action["channel"])
        neighbor_offsets = [
            float(offsets.get(int(edge["target_channel"]), 0.0))
            if int(edge["source_channel"]) == channel
            else float(offsets.get(int(edge["source_channel"]), 0.0))
            for edge in edges
            if int(edge["source_channel"]) == channel or int(edge["target_channel"]) == channel
        ]
        if (
            float(action["applied_incremental_delay_ms"]) <= 0.05
            and any(offset > 0.05 for offset in neighbor_offsets)
        ):
            action["phase_invert_applied"] = False
            action["corr_current"] = None
            action["corr_flipped"] = None
            continue
        normal_score = 0.0
        flipped_score = 0.0
        total_weight = 0.0
        for edge in edges:
            if int(edge["source_channel"]) != channel and int(edge["target_channel"]) != channel:
                continue
            other_channel = int(edge["target_channel"]) if int(edge["source_channel"]) == channel else int(edge["source_channel"])
            corr = _norm_corr(delayed_signals[other_channel], delayed_signals[channel])
            weight = float(edge.get("weight", 1.0))
            normal_score += weight * corr
            flipped_score += weight * (-corr)
            total_weight += weight
        if total_weight <= 0.0:
            continue
        normal_score /= total_weight
        flipped_score /= total_weight
        stem = channel_to_stem[channel]
        if abs(flipped_score) > 0.12 and flipped_score > normal_score + 0.04:
            stem.audio = -stem.audio
            stem.phase_invert = not stem.phase_invert
            delayed_signals[channel] = -delayed_signals[channel]
            action["phase_invert"] = bool(stem.phase_invert)
            action["phase_invert_applied"] = True
        else:
            action["phase_invert_applied"] = False
        action["corr_current"] = round(float(normal_score), 4)
        action["corr_flipped"] = round(float(flipped_score), 4)
    return [action for action in actions if action["applied_incremental_delay_ms"] > 0.05 or action.get("phase_invert_applied")]


def _global_phase_graph_score(stems: list[AyaicStem], edges: list[dict[str, Any]]) -> float:
    if not edges:
        return 0.0
    signals = {stem.channel_id: _phase_alignment_signal(stem) for stem in stems}
    total = 0.0
    total_weight = 0.0
    for edge in edges:
        source = int(edge["source_channel"])
        target = int(edge["target_channel"])
        weight = float(edge.get("weight", 1.0))
        corr = _norm_corr(signals[source], signals[target])
        total += weight * corr
        total_weight += weight
    return float(total / max(total_weight, 1e-12))


def _round_global_phase_edge(edge: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in edge.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        elif isinstance(value, np.floating):
            out[key] = round(float(value), 4)
        elif isinstance(value, np.integer):
            out[key] = int(value)
        else:
            out[key] = value
    return out


def _global_overhead_report(
    stems: list[AyaicStem],
    edges: list[dict[str, Any]],
    offsets: dict[int, float],
) -> dict[str, Any]:
    pair = _overhead_lr(stems)
    if not pair:
        return {"enabled": False, "reason": "overhead_pair_not_found"}
    left, right = pair
    edge = next(
        (
            item for item in edges
            if {int(item["source_channel"]), int(item["target_channel"])} == {left.channel_id, right.channel_id}
        ),
        None,
    )
    adjusted = left if offsets.get(left.channel_id, 0.0) > offsets.get(right.channel_id, 0.0) else right
    zero = left if offsets.get(left.channel_id, 0.0) <= offsets.get(right.channel_id, 0.0) else right
    return {
        "enabled": edge is not None,
        "rule": "global_bleed_phase_alignment_overhead_edge",
        "operation_order": ["global_graph_solve", "fractional_delay", "polarity_check"],
        "phase_reference": "global_bleed_correlation_graph",
        "reference_rule": "component_minimum_delay_is_zero_ms",
        "left_file": Path(left.path).name,
        "right_file": Path(right.path).name,
        "zero_ms_source": Path(zero.path).name,
        "farthest_overhead_file": Path(zero.path).name,
        "adjusted_file": Path(adjusted.path).name,
        "measured_right_vs_left_delay_ms": round(float(edge["delay_ms"]), 3) if edge else None,
        "applied_left_delay_ms": round(float(left.delay_ms), 4),
        "applied_right_delay_ms": round(float(right.delay_ms), 4),
        "psr_db": round(float(edge["psr"]), 2) if edge and np.isfinite(edge["psr"]) else None,
        "confidence": round(float(edge["confidence"]), 3) if edge else None,
        "coherence": round(float(edge["coherence"]), 3) if edge else None,
        "phase_invert_applied": bool(left.phase_invert or right.phase_invert),
        "phase_inverted_file": Path(left.path).name if left.phase_invert else (Path(right.path).name if right.phase_invert else None),
        "corr_current": round(float(edge["corr_current"]), 3) if edge else None,
        "corr_flipped": round(float(edge["corr_flipped"]), 3) if edge else None,
    }


def _component_index_for_channel(components: list[dict[str, Any]], channel_id: int) -> int | None:
    for component in components:
        if channel_id in component["channels"]:
            return int(component["index"])
    return None


def _apply_overhead_anchored_drum_panning(stems: list[AyaicStem], *, drum_monos: dict[int, np.ndarray]) -> dict[str, Any]:
    pair = _overhead_lr(stems)
    if not pair:
        return {"enabled": False, "reason": "overhead_pair_not_found"}

    left_stem, right_stem = pair
    overhead_l = _highpass(drum_monos[left_stem.channel_id], left_stem.sample_rate, 45.0)
    overhead_r = _highpass(drum_monos[right_stem.channel_id], right_stem.sample_rate, 45.0)
    sample_rate = left_stem.sample_rate

    center_measurements: list[dict[str, Any]] = []
    source_measurements: dict[int, list[dict[str, Any]]] = {}
    for stem in stems:
        instrument = _drum_instrument(stem.name)
        if instrument not in {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
            continue
        source = _highpass(drum_monos[stem.channel_id], stem.sample_rate, 40.0)
        window_ms = 170.0 if instrument in {"kick", "snare"} else 260.0
        count = 18 if instrument in {"kick", "snare"} else 12
        measurements = _source_overhead_measurements(
            source,
            overhead_l,
            overhead_r,
            stem.sample_rate,
            stem.delay_ms,
            window_ms=window_ms,
            count=count,
        )
        source_measurements[stem.channel_id] = measurements
        if instrument in {"kick", "snare"}:
            center_measurements.extend(
                {**measurement, "channel": stem.channel_id, "file": Path(stem.path).name, "instrument": instrument}
                for measurement in measurements
            )

    before_left = float(left_stem.pan)
    before_right = float(right_stem.pan)
    if center_measurements:
        best: tuple[float, float, float, float, float] | None = None
        for width in np.linspace(0.58, 0.88, 31):
            for shift in np.linspace(-0.18, 0.18, 37):
                left_pan = float(np.clip(-width + shift, -0.95, -0.18))
                right_pan = float(np.clip(width + shift, 0.18, 0.95))
                error, signed = _weighted_center_error(center_measurements, left_pan, right_pan)
                width_penalty = max(0.0, 0.74 - width) * 0.7 + max(0.0, width - 0.86) * 0.25
                score = error + width_penalty + abs(float(shift)) * 0.2
                if best is None or score < best[0]:
                    best = (score, left_pan, right_pan, error, signed)
        assert best is not None
        _, new_left, new_right, after_error, after_signed = best
        left_stem.pan = new_left
        right_stem.pan = new_right
    else:
        after_error, after_signed = _weighted_center_error(center_measurements, before_left, before_right)

    before_error, before_signed = _weighted_center_error(center_measurements, before_left, before_right)
    left_note = {
        "rule": "overhead_anchor_first",
        "before_pan": round(before_left, 3),
        "after_pan": round(left_stem.pan, 3),
        "center_sources": ["kick", "snare"],
        "center_error_before_db": round(before_error, 2),
        "center_error_after_db": round(after_error, 2),
        "center_signed_before_db": round(before_signed, 2),
        "center_signed_after_db": round(after_signed, 2),
    }
    right_note = dict(left_note)
    right_note["before_pan"] = round(before_right, 3)
    right_note["after_pan"] = round(right_stem.pan, 3)
    left_stem.pan_notes.append(left_note)
    right_stem.pan_notes.append(right_note)

    placed: list[dict[str, Any]] = []
    for stem in stems:
        instrument = _drum_instrument(stem.name)
        if instrument not in {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
            continue

        before_pan = float(stem.pan)
        if instrument in {"kick", "snare"}:
            stem.pan = 0.0
            note = {
                "rule": "kick_snare_center_after_overheads",
                "before_pan": round(before_pan, 3),
                "after_pan": round(stem.pan, 3),
                "reason": "kick/snare image is the center reference for the overhead pair",
            }
            stem.pan_notes.append(note)
            placed.append({"channel": stem.channel_id, "file": Path(stem.path).name, "instrument": instrument, **note})
            continue

        measurements = source_measurements.get(stem.channel_id, [])
        pan_values = []
        weights = []
        for measurement in measurements:
            diff = _overhead_output_diff_db(
                float(measurement["left_rms"]),
                float(measurement["right_rms"]),
                left_stem.pan,
                right_stem.pan,
            )
            pan_values.append(_pan_from_lr_diff_db(diff))
            weights.append(float(measurement["weight"]))

        if pan_values:
            estimated_pan = _weighted_median(pan_values, weights)
            if instrument in {"rack_tom", "floor_tom"}:
                estimated_pan = float(np.clip(estimated_pan, -0.68, 0.68))
            else:
                estimated_pan = float(np.clip(estimated_pan, -0.82, 0.82))
            stem.pan = estimated_pan
            source_note = {
                "rule": "close_mic_follows_overhead_image",
                "before_pan": round(before_pan, 3),
                "after_pan": round(stem.pan, 3),
                "measurement_count": len(measurements),
                "median_overhead_image_pan": round(estimated_pan, 3),
                "left_right_db_examples": [
                    {
                        "start_sec": measurement["start_sec"],
                        "left_db": measurement["left_db"],
                        "right_db": measurement["right_db"],
                    }
                    for measurement in measurements[:4]
                ],
            }
        else:
            source_note = {
                "rule": "close_mic_follows_overhead_image",
                "before_pan": round(before_pan, 3),
                "after_pan": round(stem.pan, 3),
                "measurement_count": 0,
                "reason": "no confident overhead image measurements, kept existing pan",
            }
        stem.pan_notes.append(source_note)
        placed.append({"channel": stem.channel_id, "file": Path(stem.path).name, "instrument": instrument, **source_note})

    for stem in [left_stem, right_stem, *[candidate for candidate in stems if _drum_instrument(candidate.name) in {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}]]:
        stem.audio = _apply_pan(stem.audio, stem.pan)

    return {
        "enabled": True,
        "rule": "apply_overhead_anchored_drum_panning",
        "overhead_left_channel": left_stem.channel_id,
        "overhead_right_channel": right_stem.channel_id,
        "overhead_left_file": Path(left_stem.path).name,
        "overhead_right_file": Path(right_stem.path).name,
        "overhead_left_pan": round(left_stem.pan, 3),
        "overhead_right_pan": round(right_stem.pan, 3),
        "center_measurements": len(center_measurements),
        "center_error_before_db": round(before_error, 2),
        "center_error_after_db": round(after_error, 2),
        "center_signed_after_db": round(after_signed, 2),
        "placed_close_mics": placed,
        "rule_summary": (
            "Pan overhead pair first so kick/snare leakage is centered; keep kick/snare close mics centered; "
            "place remaining close drum mics where they appear in the panned overhead image."
        ),
    }


def _initial_pan(name: str) -> float:
    key = _name_key(name)
    if key in {"oh l", "overhead l"} or key.endswith(" l"):
        return -0.74
    if key in {"oh r", "overhead r"} or key.endswith(" r"):
        return 0.74
    return 0.0


def _drum_instrument(name: str) -> str | None:
    key = _name_key(name)
    if key in {"oh l", "oh r", "overhead l", "overhead r"} or "overhead" in key:
        return "overhead"
    if "room" in key:
        return "room"
    if "kick" in key:
        return "kick"
    if "snare" in key:
        return "snare"
    if "floor tom" in key or "f tom" in key:
        return "floor_tom"
    if "tom" in key:
        return "rack_tom"
    if "hat" in key:
        return "hi_hat"
    if "ride" in key:
        return "ride"
    if "perc" in key:
        return "percussion"
    return None


def _bleed_analysis_instrument(name: str) -> str | None:
    drum = _drum_instrument(name)
    if drum and drum not in {"overhead", "room"}:
        return drum
    key = _name_key(name)
    if _is_backing_vocal_key(key):
        return "backing_vocal"
    if "vox" in key or "vocal" in key:
        return "lead_vocal"
    return None


def _event_metric_config(instrument: str | None) -> dict[str, float] | None:
    if instrument in {"lead_vocal", "backing_vocal"}:
        return {
            "frame_ms": 240.0,
            "hop_ms": 60.0,
            "pad_ms": 140.0,
            "detect_hpf_hz": 120.0,
            "detect_lpf_hz": 4500.0,
            "percentile": 82.0,
            "peak_offset_db": 18.0,
            "floor_margin_db": 8.0,
            "min_threshold_db": -42.0,
        }
    if instrument in {"rack_tom", "floor_tom"}:
        return {
            "frame_ms": 150.0,
            "hop_ms": 35.0,
            "pad_ms": 90.0,
            "detect_hpf_hz": 60.0,
            "detect_lpf_hz": 1200.0,
            "percentile": 97.5,
            "peak_offset_db": 14.0,
            "floor_margin_db": 10.0,
            "min_threshold_db": -36.0,
        }
    if instrument == "kick":
        return {
            "frame_ms": 140.0,
            "hop_ms": 30.0,
            "pad_ms": 80.0,
            "detect_hpf_hz": 28.0,
            "detect_lpf_hz": 220.0,
            "percentile": 95.0,
            "peak_offset_db": 14.0,
            "floor_margin_db": 6.0,
            "min_threshold_db": -44.0,
        }
    if instrument == "snare":
        return {
            "frame_ms": 135.0,
            "hop_ms": 30.0,
            "pad_ms": 85.0,
            "detect_hpf_hz": 140.0,
            "detect_lpf_hz": 2600.0,
            "percentile": 97.0,
            "peak_offset_db": 13.0,
            "floor_margin_db": 9.0,
            "min_threshold_db": -38.0,
        }
    if instrument == "hi_hat":
        return {
            "frame_ms": 160.0,
            "hop_ms": 40.0,
            "pad_ms": 110.0,
            "detect_hpf_hz": 1800.0,
            "detect_lpf_hz": 14000.0,
            "percentile": 92.0,
            "peak_offset_db": 16.0,
            "floor_margin_db": 6.0,
            "min_threshold_db": -44.0,
        }
    if instrument == "ride":
        return {
            "frame_ms": 170.0,
            "hop_ms": 40.0,
            "pad_ms": 120.0,
            "detect_hpf_hz": 1200.0,
            "detect_lpf_hz": 12000.0,
            "percentile": 94.0,
            "peak_offset_db": 18.0,
            "peak_percentile": 99.9,
            "floor_margin_db": 5.0,
            "min_threshold_db": -50.0,
        }
    if instrument == "percussion":
        return {
            "frame_ms": 160.0,
            "hop_ms": 40.0,
            "pad_ms": 110.0,
            "detect_hpf_hz": 1200.0,
            "detect_lpf_hz": 12000.0,
            "percentile": 92.0,
            "peak_offset_db": 16.0,
            "floor_margin_db": 6.0,
            "min_threshold_db": -44.0,
        }
    return None


def _event_activity_ranges(x: np.ndarray, sample_rate: int, instrument: str | None) -> dict[str, Any] | None:
    config = _event_metric_config(instrument)
    if not config or len(x) == 0:
        return None

    detect = _highpass(x, sample_rate, config["detect_hpf_hz"])
    detect = _lowpass(detect, sample_rate, config["detect_lpf_hz"])
    frame = max(256, int(config["frame_ms"] * sample_rate / 1000.0))
    hop = max(64, int(config["hop_ms"] * sample_rate / 1000.0))
    starts, rms_db = _frame_rms_db(detect, frame, hop)
    if len(rms_db) == 0:
        return {
            "config": config,
            "frame": frame,
            "hop": hop,
            "ranges": [],
            "threshold_db": None,
            "active_samples": 0,
        }

    peak_percentile = float(config.get("peak_percentile", 100.0))
    if peak_percentile >= 100.0:
        detect_peak = float(np.max(np.abs(detect)))
    else:
        detect_peak = float(np.percentile(np.abs(detect), peak_percentile))
    detect_peak_db = amp_to_db(detect_peak)
    noise_floor_db = float(np.percentile(rms_db, 50))
    threshold_db = max(
        float(np.percentile(rms_db, config["percentile"])),
        detect_peak_db - config["peak_offset_db"],
        noise_floor_db + config["floor_margin_db"],
        config["min_threshold_db"],
    )
    active_idx = np.flatnonzero(rms_db >= threshold_db)
    threshold_relaxed = False
    if len(active_idx) == 0:
        relaxed_threshold_db = max(
            float(np.percentile(rms_db, config["percentile"])),
            noise_floor_db + config["floor_margin_db"],
            config["min_threshold_db"],
        )
        if relaxed_threshold_db < threshold_db:
            threshold_db = relaxed_threshold_db
            threshold_relaxed = True
            active_idx = np.flatnonzero(rms_db >= threshold_db)
    if len(active_idx) == 0:
        return {
            "config": config,
            "frame": frame,
            "hop": hop,
            "ranges": [],
            "threshold_db": round(threshold_db, 2),
            "threshold_relaxed": threshold_relaxed,
            "active_samples": 0,
        }

    pad = int(config["pad_ms"] * sample_rate / 1000.0)
    ranges = []
    for idx in active_idx:
        start = max(0, int(starts[idx]) - pad)
        end = min(len(x), int(starts[idx]) + frame + pad)
        if end > start:
            ranges.append((start, end))
    merged = _merge_ranges(ranges, gap=pad // 2)
    active_samples = sum(end - start for start, end in merged)
    return {
        "config": config,
        "frame": frame,
        "hop": hop,
        "ranges": merged,
        "threshold_db": round(threshold_db, 2),
        "threshold_relaxed": threshold_relaxed,
        "active_samples": active_samples,
    }


def _frame_rms_db(x: np.ndarray, frame: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    mono = np.asarray(x, dtype=np.float32)
    if len(mono) < frame:
        mono = np.pad(mono, (0, frame - len(mono)))
    starts = np.arange(0, max(1, len(mono) - frame + 1), hop, dtype=np.int64)
    values = []
    for start in starts:
        block = mono[start:start + frame]
        values.append(amp_to_db(float(np.sqrt(np.mean(np.square(block))) + 1e-12)))
    return starts, np.asarray(values, dtype=np.float32)


def _activity_bleed_metrics(x: np.ndarray, sample_rate: int, activity: dict[str, Any] | None) -> dict[str, Any]:
    del sample_rate
    if not activity or not activity.get("ranges") or len(x) == 0:
        return {
            "analysis_event_rms_db": None,
            "analysis_bleed_rms_db": None,
            "analysis_event_to_bleed_db": None,
            "analysis_bleed_ratio": None,
            "analysis_bleed_dominant": False,
        }

    mask = _ranges_to_mask(len(x), list(activity.get("ranges") or []))
    if not np.any(mask):
        return {
            "analysis_event_rms_db": None,
            "analysis_bleed_rms_db": None,
            "analysis_event_to_bleed_db": None,
            "analysis_bleed_ratio": None,
            "analysis_bleed_dominant": False,
        }

    active = x[mask]
    inactive = x[~mask]
    event_rms_db = _rms_db_for_samples(active)
    bleed_rms_db = _rms_db_for_samples(inactive)
    if event_rms_db is None or bleed_rms_db is None:
        event_to_bleed_db = None
        bleed_ratio = None
        bleed_dominant = False
    else:
        event_to_bleed_db = float(event_rms_db - bleed_rms_db)
        event_power = db_to_amp(event_rms_db) ** 2
        bleed_power = db_to_amp(bleed_rms_db) ** 2
        bleed_ratio = float(bleed_power / max(event_power + bleed_power, 1e-12))
        active_ratio = float(activity.get("active_samples", 0)) / max(1, len(x))
        bleed_dominant = bool(event_to_bleed_db < 8.0 or active_ratio > 0.35 or bleed_ratio > 0.2)

    return {
        "analysis_event_rms_db": round(event_rms_db, 2) if event_rms_db is not None else None,
        "analysis_bleed_rms_db": round(bleed_rms_db, 2) if bleed_rms_db is not None else None,
        "analysis_event_to_bleed_db": round(event_to_bleed_db, 2) if event_to_bleed_db is not None else None,
        "analysis_bleed_ratio": round(bleed_ratio, 4) if bleed_ratio is not None else None,
        "analysis_bleed_dominant": bleed_dominant,
    }


def _ranges_to_mask(length: int, ranges: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(max(0, int(length)), dtype=bool)
    for start, end in ranges:
        mask[max(0, int(start)):min(len(mask), int(end))] = True
    return mask


def _rms_db_for_samples(samples: np.ndarray) -> float | None:
    if len(samples) == 0:
        return None
    return amp_to_db(float(np.sqrt(np.mean(np.square(samples))) + 1e-12))


def _analysis_block_range(x: np.ndarray, sample_rate: int, window_sec: float = 18.0) -> tuple[int, int]:
    window = min(len(x), max(1024, int(window_sec * sample_rate)))
    if len(x) <= window:
        return (0, len(x))
    hop = max(512, window // 3)
    best_start = 0
    best_energy = -1.0
    for start in range(0, len(x) - window, hop):
        energy = float(np.mean(np.square(x[start:start + window])))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return (best_start, best_start + window)


def _merge_ranges(ranges: list[tuple[int, int]], gap: int = 0) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted((max(0, int(start)), max(0, int(end))) for start, end in ranges if end > start)
    if not ordered:
        return []
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _ranges_preview_seconds(ranges: list[tuple[int, int]], sample_rate: int, count: int = 8) -> list[dict[str, float]]:
    if sample_rate <= 0:
        return []
    return [
        {
            "start": round(start / sample_rate, 3),
            "end": round(end / sample_rate, 3),
        }
        for start, end in ranges[:count]
    ]


def _highpass(x: np.ndarray, sample_rate: int, freq: float) -> np.ndarray:
    if freq <= 0:
        return np.asarray(x, dtype=np.float32)
    try:
        from scipy.signal import butter, lfilter

        b, a = butter(2, freq / (sample_rate * 0.5), btype="highpass")
        return lfilter(b, a, np.asarray(x, dtype=np.float32)).astype(np.float32)
    except Exception:
        return np.asarray(x, dtype=np.float32)


def _lowpass(x: np.ndarray, sample_rate: int, freq: float) -> np.ndarray:
    if freq <= 0 or freq >= sample_rate * 0.49:
        return np.asarray(x, dtype=np.float32)
    try:
        from scipy.signal import butter, lfilter

        b, a = butter(2, freq / (sample_rate * 0.5), btype="lowpass")
        return lfilter(b, a, np.asarray(x, dtype=np.float32)).astype(np.float32)
    except Exception:
        return np.asarray(x, dtype=np.float32)


def _peaking_eq(x: np.ndarray, sample_rate: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    if abs(gain_db) < 1e-4 or freq <= 0.0 or freq >= sample_rate * 0.49:
        return np.asarray(x, dtype=np.float32)
    try:
        from scipy.signal import lfilter

        a = db_to_amp(gain_db)
        w0 = 2.0 * math.pi * float(freq) / sample_rate
        alpha = math.sin(w0) / (2.0 * max(float(q), 0.05))
        cos_w0 = math.cos(w0)
        b0 = 1.0 + alpha * a
        b1 = -2.0 * cos_w0
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha / a
        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        aa = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        return lfilter(b, aa, np.asarray(x, dtype=np.float32)).astype(np.float32)
    except Exception:
        return np.asarray(x, dtype=np.float32)


def _active_segment_starts(x: np.ndarray, sample_rate: int, window_sec: float = 3.0, count: int = 8) -> list[int]:
    window = max(1024, int(window_sec * sample_rate))
    if len(x) <= window:
        return [0]
    hop = max(512, window // 2)
    candidates = []
    for start in range(0, len(x) - window, hop):
        energy = float(np.mean(np.square(x[start : start + window])))
        candidates.append((energy, start))
    candidates.sort(reverse=True)
    return [start for _, start in candidates[:count]]


def _snare_event_starts(stems: list[AyaicStem], sample_rate: int) -> list[int] | None:
    snare_stems = [stem for stem in stems if _drum_instrument(stem.name) == "snare"]
    if not snare_stems:
        return None
    preferred = next((stem for stem in snare_stems if _name_key(stem.name) in {"snare t", "snare top"}), snare_stems[0])
    if preferred.sample_rate != sample_rate:
        return None
    snare = _highpass(_mono(preferred.audio), sample_rate, 80.0)
    return _transient_segment_starts(snare, sample_rate, window_ms=170.0, count=24, min_gap_ms=180.0)


def _snare_polarity_decision(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    *,
    starts: list[int] | None,
    max_lag_ms: float,
) -> dict[str, Any]:
    selected_starts = starts or _active_segment_starts(target, sample_rate, window_sec=0.17, count=24)
    window = max(512, int(0.170 * sample_rate))
    max_lag = _samples_from_ms(max_lag_ms, sample_rate)
    best_corr = 0.0
    best_abs = 0.0
    best_lag = 0

    try:
        from scipy.signal import correlate
    except Exception:
        correlate = None

    for start in selected_starts:
        end = min(start + window, len(reference), len(target))
        if end - start < window // 3:
            continue
        ref_seg = np.asarray(reference[start:end], dtype=np.float64)
        tgt_seg = np.asarray(target[start:end], dtype=np.float64)
        ref_seg -= float(np.mean(ref_seg))
        tgt_seg -= float(np.mean(tgt_seg))
        denom = math.sqrt(float(np.dot(ref_seg, ref_seg) * np.dot(tgt_seg, tgt_seg))) + 1e-12
        if denom <= 1e-10:
            continue

        if correlate is not None:
            corr = correlate(tgt_seg, ref_seg, mode="full", method="fft") / denom
        else:
            corr = np.correlate(tgt_seg, ref_seg, mode="full") / denom
        lags = np.arange(-len(ref_seg) + 1, len(tgt_seg), dtype=np.int64)
        mask = np.abs(lags) <= max_lag
        if not np.any(mask):
            continue
        limited = corr[mask]
        limited_lags = lags[mask]
        idx = int(np.argmax(np.abs(limited)))
        signed = float(limited[idx])
        if abs(signed) > best_abs:
            best_abs = abs(signed)
            best_corr = signed
            best_lag = int(limited_lags[idx])

    flipped = -best_corr
    return {
        "current_corr": float(best_corr),
        "flipped_corr": float(flipped),
        "best_lag_ms": 1000.0 * best_lag / float(sample_rate),
        "should_invert": bool(abs(flipped) > 0.12 and flipped > best_corr + 0.04),
    }


def _representative_delay_measurement(
    reference: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    *,
    max_delay_ms: float,
    prefer_negative: bool,
    starts: list[int] | None = None,
) -> tuple[int, dict[str, float]]:
    window_len = int(3.0 * sample_rate)
    measurements: list[tuple[int, dict[str, float]]] = []
    for start in starts or _active_segment_starts(target, sample_rate):
        window = min(len(reference) - start, len(target) - start, window_len)
        if window < int(0.5 * sample_rate):
            continue
        ref_seg = reference[start : start + window]
        tgt_seg = target[start : start + window]
        fft_size = _next_power_of_two(len(ref_seg))
        if fft_size != len(ref_seg):
            ref_seg = np.pad(ref_seg, (0, fft_size - len(ref_seg)))
            tgt_seg = np.pad(tgt_seg, (0, fft_size - len(tgt_seg)))
        measurement = _gcc_phat_measurement(
            ref_seg,
            tgt_seg,
            sample_rate,
            fft_size=fft_size,
            max_delay_ms=max_delay_ms,
        )
        measured_delay_ms = float(measurement["delay_ms"])
        boundary_hit = abs(abs(measured_delay_ms) - max_delay_ms) < 0.05
        if boundary_hit or not np.isfinite(measured_delay_ms):
            continue
        measurements.append((start, measurement))

    if not measurements:
        return 0, {}

    negative_measurements = [(start, measurement) for start, measurement in measurements if float(measurement["delay_ms"]) < -0.3]
    if prefer_negative and negative_measurements:
        median_delay = float(np.median([float(measurement["delay_ms"]) for _, measurement in negative_measurements]))
        return min(
            negative_measurements,
            key=lambda item: abs(float(item[1]["delay_ms"]) - median_delay),
        )
    median_delay = float(np.median([float(measurement["delay_ms"]) for _, measurement in measurements]))
    return min(measurements, key=lambda item: abs(float(item[1]["delay_ms"]) - median_delay))


def _next_power_of_two(value: int) -> int:
    return 1 << max(1, int(value - 1).bit_length())


def _gcc_phat_measurement(
    ref_signal: np.ndarray,
    tgt_signal: np.ndarray,
    sample_rate: int,
    *,
    fft_size: int,
    max_delay_ms: float,
) -> dict[str, float]:
    if len(ref_signal) < fft_size or len(tgt_signal) < fft_size:
        return {
            "delay_ms": 0.0,
            "delay_samples": 0.0,
            "psr": 0.0,
            "confidence": 0.0,
            "coherence": 0.0,
        }

    ref_frame = np.asarray(ref_signal[:fft_size], dtype=np.float64) * np.hanning(fft_size)
    tgt_frame = np.asarray(tgt_signal[:fft_size], dtype=np.float64) * np.hanning(fft_size)
    ref_fft = np.fft.fft(ref_frame)
    tgt_fft = np.fft.fft(tgt_frame)
    cross_spectrum = np.conj(ref_fft) * tgt_fft
    eps = 1e-10
    phat = cross_spectrum / (np.abs(cross_spectrum) + eps)
    gcc = np.fft.fftshift(np.fft.ifft(phat).real)

    max_delay_samples = int(max_delay_ms * sample_rate / 1000.0)
    center = len(gcc) // 2
    search_start = max(0, center - max_delay_samples)
    search_end = min(len(gcc), center + max_delay_samples + 1)
    limited = gcc[search_start:search_end]
    if limited.size == 0:
        return {
            "delay_ms": 0.0,
            "delay_samples": 0.0,
            "psr": 0.0,
            "confidence": 0.0,
            "coherence": 0.0,
        }

    peak_idx = int(np.argmax(np.abs(limited)))
    peak_value = float(limited[peak_idx])
    interpolated_peak_idx = float(peak_idx)
    interpolated_peak_value = peak_value
    if 0 < peak_idx < len(limited) - 1:
        alpha = float(limited[peak_idx - 1])
        beta = float(limited[peak_idx])
        gamma = float(limited[peak_idx + 1])
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > eps:
            p = 0.5 * (alpha - gamma) / denom
            interpolated_peak_idx = peak_idx + p
            interpolated_peak_value = beta - 0.25 * (alpha - gamma) * p

    delay_samples = interpolated_peak_idx - (len(limited) // 2)
    delay_ms = delay_samples * 1000.0 / float(sample_rate)

    vicinity = 5
    excluded = np.abs(limited).copy()
    excluded[max(0, peak_idx - vicinity) : min(len(excluded), peak_idx + vicinity + 1)] = 0.0
    second_peak = float(np.max(excluded)) if excluded.size else 0.0
    psr = 20.0 * math.log10(max(abs(interpolated_peak_value), eps) / (second_peak + eps))

    energy_product = math.sqrt(float(np.sum(ref_frame**2) * np.sum(tgt_frame**2)))
    correlation_peak = float(np.clip(abs(interpolated_peak_value) / (energy_product + eps), 0.0, 1.0))
    snr_db = 10.0 * math.log10(float(np.mean(ref_frame**2)) / (float(np.mean(np.abs(limited) ** 2)) + eps))
    coherence = np.abs(cross_spectrum) ** 2 / ((np.abs(ref_fft) ** 2 + eps) * (np.abs(tgt_fft) ** 2 + eps))
    coherence_value = float(np.mean(coherence[: fft_size // 2]))
    confidence = (
        0.4 * min(1.0, correlation_peak / 0.8)
        + 0.3 * min(1.0, psr / 10.0)
        + 0.2 * min(1.0, max(0.0, snr_db) / 40.0)
        + 0.1 * coherence_value
    )

    return {
        "delay_ms": float(delay_ms),
        "delay_samples": float(delay_samples),
        "psr": float(psr),
        "confidence": float(confidence),
        "coherence": float(coherence_value),
    }


def _samples_from_ms(delay_ms: float, sample_rate: int) -> int:
    return int(round(max(0.0, delay_ms) * sample_rate / 1000.0))


def _delay_audio_fractional(audio: np.ndarray, delay_ms: float, sample_rate: int) -> np.ndarray:
    delay_samples = max(0.0, float(delay_ms)) * float(sample_rate) / 1000.0
    arr = ensure_stereo(audio).astype(np.float32, copy=False)
    if delay_samples <= 1e-4:
        return arr.copy()
    if delay_samples >= arr.shape[0]:
        return np.zeros_like(arr, dtype=np.float32)
    positions = np.arange(arr.shape[0], dtype=np.float64) - delay_samples
    base = np.arange(arr.shape[0], dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.float32)
    for channel_index in range(arr.shape[1]):
        out[:, channel_index] = np.interp(
            positions,
            base,
            arr[:, channel_index],
            left=0.0,
            right=0.0,
        ).astype(np.float32)
    return out


def _delay_mono_fractional(audio: np.ndarray, delay_ms: float, sample_rate: int) -> np.ndarray:
    delay_samples = max(0.0, float(delay_ms)) * float(sample_rate) / 1000.0
    arr = np.asarray(audio, dtype=np.float32)
    if delay_samples <= 1e-4:
        return arr.copy()
    if delay_samples >= len(arr):
        return np.zeros_like(arr, dtype=np.float32)
    positions = np.arange(len(arr), dtype=np.float64) - delay_samples
    base = np.arange(len(arr), dtype=np.float64)
    return np.interp(positions, base, arr, left=0.0, right=0.0).astype(np.float32)


def _delay_mono(audio: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0:
        return np.asarray(audio, dtype=np.float32).copy()
    if samples >= len(audio):
        return np.zeros_like(audio, dtype=np.float32)
    return np.pad(audio[:-samples], (samples, 0)).astype(np.float32)


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    aa = np.asarray(a[:n], dtype=np.float64) - float(np.mean(a[:n]))
    bb = np.asarray(b[:n], dtype=np.float64) - float(np.mean(b[:n]))
    denom = math.sqrt(float(np.dot(aa, aa) * np.dot(bb, bb))) + 1e-12
    return float(np.dot(aa, bb) / denom)


def _overhead_lr(stems: list[AyaicStem]) -> tuple[AyaicStem, AyaicStem] | None:
    overheads = [stem for stem in stems if _drum_instrument(stem.name) == "overhead"]
    if len(overheads) < 2:
        return None
    left = next((stem for stem in overheads if _name_key(stem.name) in {"oh l", "overhead l"} or _name_key(stem.name).endswith(" l")), None)
    right = next((stem for stem in overheads if _name_key(stem.name) in {"oh r", "overhead r"} or _name_key(stem.name).endswith(" r")), None)
    if left and right:
        return left, right
    ordered = sorted(overheads, key=lambda stem: stem.pan)
    return ordered[0], ordered[-1]


def _source_overhead_measurements(
    source: np.ndarray,
    overhead_l: np.ndarray,
    overhead_r: np.ndarray,
    sample_rate: int,
    source_delay_ms: float,
    *,
    window_ms: float,
    count: int,
) -> list[dict[str, Any]]:
    window = max(512, int(window_ms * 0.001 * sample_rate))
    delay_samples = _samples_from_ms(source_delay_ms, sample_rate)
    starts = _transient_segment_starts(source, sample_rate, window_ms=window_ms, count=count)
    measurements: list[dict[str, Any]] = []
    for start in starts:
        oh_start = start + delay_samples
        oh_end = min(oh_start + window, len(overhead_l), len(overhead_r))
        src_end = min(start + window, len(source))
        if oh_start < 0 or oh_end - oh_start < window // 3 or src_end - start < window // 3:
            continue

        src_seg = source[start:src_end]
        l_seg = overhead_l[oh_start:oh_end]
        r_seg = overhead_r[oh_start:oh_end]
        source_rms = _rms_linear(src_seg)
        left_rms = _rms_linear(l_seg)
        right_rms = _rms_linear(r_seg)
        if source_rms < db_to_amp(-60.0) or (left_rms + right_rms) < db_to_amp(-64.0):
            continue
        measurements.append(
            {
                "start_sec": round(start / sample_rate, 3),
                "left_rms": left_rms,
                "right_rms": right_rms,
                "left_db": round(amp_to_db(left_rms), 2),
                "right_db": round(amp_to_db(right_rms), 2),
                "source_db": round(amp_to_db(source_rms), 2),
                "weight": float(np.clip(db_to_amp(amp_to_db(source_rms) + 34.0), 0.15, 4.0)),
            }
        )
    return measurements


def _transient_segment_starts(
    x: np.ndarray,
    sample_rate: int,
    *,
    window_ms: float = 220.0,
    count: int = 16,
    min_gap_ms: float = 260.0,
) -> list[int]:
    window = max(256, int(window_ms * 0.001 * sample_rate))
    if len(x) <= window:
        return [0]

    frame = max(128, int(0.018 * sample_rate))
    hop = max(64, int(0.009 * sample_rate))
    if len(x) <= frame:
        return [0]

    starts = np.arange(0, len(x) - frame, hop, dtype=np.int64)
    envelope = np.asarray(
        [float(np.sqrt(np.mean(np.square(x[start : start + frame]))) + 1e-12) for start in starts],
        dtype=np.float32,
    )
    if len(envelope) < 3:
        return [0]

    floor = float(np.percentile(envelope, 60))
    threshold = max(floor * 1.4, float(np.percentile(envelope, 82)))
    peaks: list[tuple[float, int]] = []
    for idx in range(1, len(envelope) - 1):
        value = float(envelope[idx])
        if value < threshold:
            continue
        if value >= float(envelope[idx - 1]) and value >= float(envelope[idx + 1]):
            peaks.append((value, int(starts[idx] + frame // 2)))

    if not peaks:
        return _active_segment_starts(x, sample_rate, window_sec=window_ms * 0.001, count=count)

    peaks.sort(reverse=True)
    min_gap = int(min_gap_ms * 0.001 * sample_rate)
    selected: list[int] = []
    for _, center in peaks:
        if all(abs(center - existing) >= min_gap for existing in selected):
            selected.append(center)
        if len(selected) >= count:
            break

    selected.sort()
    return [int(np.clip(center - window // 2, 0, len(x) - window)) for center in selected]


def _equal_power_gains(pan: float) -> tuple[float, float]:
    pan = float(np.clip(pan, -1.0, 1.0))
    theta = (pan + 1.0) * math.pi / 4.0
    return math.cos(theta), math.sin(theta)


def _pan_from_lr_diff_db(diff_db: float) -> float:
    ratio = db_to_amp(float(np.clip(diff_db, -48.0, 48.0)))
    theta = math.atan(ratio)
    return float(np.clip((4.0 * theta / math.pi) - 1.0, -1.0, 1.0))


def _weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    order = np.argsort(np.asarray(values, dtype=np.float64))
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cutoff = float(np.sum(sorted_weights)) * 0.5
    cumulative = np.cumsum(sorted_weights)
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(sorted_values[min(idx, len(sorted_values) - 1)])


def _overhead_output_diff_db(left_rms: float, right_rms: float, left_pan: float, right_pan: float) -> float:
    left_to_l, left_to_r = _equal_power_gains(left_pan)
    right_to_l, right_to_r = _equal_power_gains(right_pan)
    out_l = max(1e-12, left_rms * left_to_l + right_rms * right_to_l)
    out_r = max(1e-12, left_rms * left_to_r + right_rms * right_to_r)
    return amp_to_db(out_r) - amp_to_db(out_l)


def _weighted_center_error(measurements: list[dict[str, Any]], left_pan: float, right_pan: float) -> tuple[float, float]:
    if not measurements:
        return 0.0, 0.0
    errors = []
    signed = []
    weights = []
    for measurement in measurements:
        diff = _overhead_output_diff_db(
            float(measurement["left_rms"]),
            float(measurement["right_rms"]),
            left_pan,
            right_pan,
        )
        weight = float(measurement["weight"])
        errors.append(abs(diff))
        signed.append(diff)
        weights.append(weight)
    return (
        float(np.average(np.asarray(errors), weights=np.asarray(weights))),
        float(np.average(np.asarray(signed), weights=np.asarray(weights))),
    )


def _apply_pan(audio: np.ndarray, pan: float) -> np.ndarray:
    mono = _mono(audio)
    left_gain, right_gain = _equal_power_gains(pan)
    return np.column_stack([mono * left_gain, mono * right_gain]).astype(np.float32)


def _apply_group_levels(stems: list[AyaicStem]) -> list[dict[str, Any]]:
    report = []
    for group in AYAIC_BUS_GROUP_ORDER:
        members = [stem for stem in stems if stem.group == group]
        if not members:
            continue
        target = _ayaic_bus_target_lufs(group, members)
        before_audio, measurement = _combined_primary_loudness_audio(members)
        sample_rate = members[0].sample_rate
        before = _lufs(before_audio, sample_rate)
        gain = target - before
        for stem in members:
            stem.audio = stem.audio * db_to_amp(gain)
            stem.group_gain_db += gain
        after_audio, _ = _combined_primary_loudness_audio(members)
        report.append(
            {
                "group": group,
                "target_lufs": target,
                "loudness_method": _lufs_method(),
                "measurement_scope": measurement["scope"],
                "primary_active_ratio": measurement["active_ratio"],
                "target_source": AYAIC_BUS_GROUP_TARGET_SOURCE,
                "pre_group_lufs": round(before, 3),
                "gain_db": round(gain, 3),
                "post_group_lufs": round(_lufs(after_audio, sample_rate), 3),
                "post_full_group_lufs": round(_lufs(_sum_audio([stem.audio for stem in members]), sample_rate), 3),
                "members": [Path(stem.path).name for stem in members],
            }
        )
    return report


def _ayaic_bus_target_lufs(group: str, members: list[AyaicStem]) -> float:
    del members
    if group in AYAIC_BUS_GROUP_TARGETS:
        return AYAIC_BUS_GROUP_TARGETS[group]
    raise KeyError(f"unknown Ayaic bus group: {group}")


def _apply_master_level(audio: np.ndarray, *, sample_rate: int, target_lufs: float, ceiling_dbfs: float | None) -> tuple[dict[str, Any], np.ndarray]:
    pre_lufs = _lufs(audio, sample_rate)
    pre_peak = _peak_dbfs(audio)
    requested_gain = target_lufs - pre_lufs
    ceiling_gain = None if ceiling_dbfs is None else ceiling_dbfs - pre_peak
    gain = requested_gain if ceiling_gain is None else min(requested_gain, ceiling_gain)
    rendered = audio * db_to_amp(gain)
    return (
        {
            "target_lufs": round(target_lufs, 3),
            "loudness_method": _lufs_method(),
            "ceiling_dbfs": round(ceiling_dbfs, 3) if ceiling_dbfs is not None else None,
            "pre_master_lufs": round(pre_lufs, 3),
            "pre_master_peak_dbfs": round(pre_peak, 3),
            "requested_gain_db": round(requested_gain, 3),
            "applied_gain_db": round(gain, 3),
            "limited_by_peak_ceiling": bool(ceiling_gain is not None and gain < requested_gain),
            "final_lufs": round(_lufs(rendered, sample_rate), 3),
            "final_peak_dbfs": round(_peak_dbfs(rendered), 3),
        },
        rendered,
    )


def _track_target_lufs(name: str) -> float:
    key = _name_key(name)
    if key in {"vocal", "lead vocal", "lead vox"}:
        return -22.0
    if key in {"snare t", "snare top"}:
        return -26.0
    if key in {"snare b", "snare bottom"}:
        return -35.0
    if key in {"oh l", "oh r", "overhead l", "overhead r"}:
        return -38.0
    if "room" in key:
        return -45.0
    if key.endswith(" l") or key.endswith(" r"):
        base = key[:-2]
        if base in {"guitar", "gtr", "playback", "back vox", "bgv", "backs"}:
            return -28.0
    if "accordion" in key:
        return -28.0
    if "hat" in key or "ride" in key or "cymbal" in key:
        return -35.0
    if any(token in key for token in ("kick", "bass", "tom", "guitar", "playback")):
        return -25.0
    if _is_backing_vocal_key(key):
        return -25.0
    return -25.0


def _instrument_group(name: str) -> str:
    key = _name_key(name)
    if _is_backing_vocal_key(key):
        return "back_vocals"
    if any(token in key for token in ("vocal", "vox")):
        return "vocals"
    if any(token in key for token in ("oh ", "overhead", "room", "hat", "ride", "cymbal", "kick", "snare", "tom")):
        return "drums"
    if "bass" in key:
        return "bass"
    if "playback" in key:
        return "playback"
    if "accordion" in key:
        return "accordion"
    if "guitar" in key or "gtr" in key:
        return "guitars"
    return "playback"


def _is_close_drum_mic(name: str) -> bool:
    key = _name_key(name)
    return any(token in key for token in ("kick", "snare", "tom")) and not any(token in key for token in ("oh", "overhead"))


def _close_mic_overhead_pan(name: str, close: np.ndarray, oh_l: np.ndarray, oh_r: np.ndarray, sample_rate: int) -> float:
    key = _name_key(name)
    if "kick" in key or "snare" in key:
        return 0.0
    window = max(1024, int(0.080 * sample_rate))
    hop = max(512, int(0.040 * sample_rate))
    if close.size < window:
        return 0.0
    energies = []
    for start in range(0, close.size - window + 1, hop):
        block = close[start : start + window]
        energies.append((float(np.mean(block * block)), start))
    if not energies:
        return 0.0
    energies.sort(reverse=True)
    pans = []
    for _, start in energies[:16]:
        left = _rms_linear(oh_l[start : start + window])
        right = _rms_linear(oh_r[start : start + window])
        if left + right <= 1e-9:
            continue
        pans.append((right - left) / (right + left))
    return float(np.clip(np.median(pans), -0.9, 0.9)) if pans else 0.0


def _pan_weighted_overhead_reference(oh_l: np.ndarray, oh_r: np.ndarray, pan: float) -> np.ndarray:
    amount = float(np.clip(pan, -1.0, 1.0))
    angle = (amount + 1.0) * np.pi / 4.0
    left_weight = float(np.cos(angle))
    right_weight = float(np.sin(angle))
    return ((oh_l * left_weight) + (oh_r * right_weight)) / max(left_weight + right_weight, 1e-12)


def _analysis_blocks(target: np.ndarray, reference: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    window = min(target.size, max(int(2.0 * sample_rate), min(int(12.0 * sample_rate), target.size)))
    if window <= 0:
        return target, reference
    hop = max(1, window // 4)
    best_start = 0
    best_power = -1.0
    for start in range(0, target.size - window + 1, hop):
        block = target[start : start + window]
        power = float(np.mean(block * block))
        if power > best_power:
            best_power = power
            best_start = start
    return target[best_start : best_start + window], reference[best_start : best_start + window]


def _gcc_phat_delay(target: np.ndarray, reference: np.ndarray, sample_rate: int, *, max_delay_ms: float) -> tuple[int, float]:
    n = int(min(target.size, reference.size))
    if n <= 1:
        return 0, 0.0
    target = np.asarray(target[:n], dtype=np.float64)
    reference = np.asarray(reference[:n], dtype=np.float64)
    target -= float(np.mean(target))
    reference -= float(np.mean(reference))
    if _rms_linear(target) <= 1e-10 or _rms_linear(reference) <= 1e-10:
        return 0, 0.0
    fft_size = 1 << int(np.ceil(np.log2(n * 2 - 1)))
    spectrum = np.fft.rfft(target, fft_size) * np.conj(np.fft.rfft(reference, fft_size))
    spectrum /= np.abs(spectrum) + 1e-12
    corr = np.fft.irfft(spectrum, fft_size)
    max_shift = int(round(max_delay_ms * sample_rate / 1000.0))
    corr = np.concatenate((corr[-max_shift:], corr[: max_shift + 1]))
    shift = int(np.argmax(np.abs(corr)) - max_shift)
    peak = float(np.max(np.abs(corr)) + 1e-12)
    sidelobes = np.delete(np.abs(corr), int(np.argmax(np.abs(corr))))
    floor = float(np.median(sidelobes) + 1e-12) if sidelobes.size else 1e-12
    psr_db = 20.0 * float(np.log10(peak / floor))
    return shift, psr_db


def _delay_audio(audio: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0:
        return audio.copy()
    if samples >= audio.shape[0]:
        return np.zeros_like(audio)
    return np.pad(audio[:-samples], ((samples, 0), (0, 0))).astype(np.float32)


def _find_stem(stems: list[AyaicStem], names: tuple[str, ...]) -> AyaicStem | None:
    wanted = set(names)
    return next((stem for stem in stems if _name_key(stem.name) in wanted), None)


def _stem_report(stem: AyaicStem) -> dict[str, Any]:
    primary_audio = _stem_primary_loudness_audio(stem)
    return {
        "channel": stem.channel_id,
        "file": Path(stem.path).name,
        "group": stem.group,
        "target_lufs": round(stem.target_lufs, 3),
        "track_gain_db": round(stem.track_gain_db, 3),
        "phase_invert": stem.phase_invert,
        "delay_ms": round(stem.delay_ms, 3),
        "pan": round(stem.pan, 3),
        "group_gain_db": round(stem.group_gain_db, 3),
        "loudness_method": _lufs_method(),
        "final_lufs_before_master": round(_lufs(stem.audio, stem.sample_rate), 3),
        "final_primary_lufs_before_master": round(_lufs(primary_audio, stem.sample_rate), 3),
        "bleed_analysis": stem.bleed_analysis,
        "phase_notes": stem.notes,
        "pan_notes": stem.pan_notes,
        "corrective_eq_notes": stem.corrective_eq_notes,
        "compression_notes": stem.compression_notes,
        "output_balance_notes": stem.output_balance_notes,
    }


def _sum_stems(stems: list[AyaicStem]) -> np.ndarray:
    return _sum_audio([stem.audio for stem in stems])


def _sum_audio(items: list[np.ndarray]) -> np.ndarray:
    if not items:
        return np.zeros((0, 2), dtype=np.float32)
    max_len = max(item.shape[0] for item in items)
    out = np.zeros((max_len, 2), dtype=np.float32)
    for item in items:
        audio = ensure_stereo(item)
        if audio.shape[0] < max_len:
            audio = np.pad(audio, ((0, max_len - audio.shape[0]), (0, 0)))
        out += audio[:max_len]
    return out


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    sf.write(path, ensure_stereo(audio), int(sample_rate), subtype="FLOAT")


def _write_mp3(wav_path: Path) -> Path | None:
    if shutil.which("ffmpeg") is None:
        return None
    mp3_path = wav_path.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "320k", str(mp3_path)],
        check=True,
    )
    return mp3_path


def _lufs(audio: np.ndarray, sample_rate: int = 48_000) -> float:
    arr = ensure_stereo(audio)
    if arr.size == 0:
        return -120.0
    if arr.shape[0] <= int(0.4 * sample_rate):
        return _rms_db(arr) - 0.691
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(int(sample_rate))
        value = float(meter.integrated_loudness(arr.astype(np.float64, copy=False)))
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return _rms_db(arr) - 0.691


def _lufs_method() -> str:
    return "Integrated LUFS (ITU-R BS.1770 via pyloudnorm; RMS fallback only for sub-400ms signals)"


def _rms_db(audio: np.ndarray) -> float:
    return amp_to_db(_rms_linear(audio))


def _rms_linear(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr) + 1e-12))


def _peak_dbfs(audio: np.ndarray) -> float:
    arr = ensure_stereo(audio)
    if arr.size == 0:
        return -120.0
    return amp_to_db(float(np.max(np.abs(arr))))


def _mono(audio: np.ndarray) -> np.ndarray:
    stereo = ensure_stereo(audio)
    return np.mean(stereo, axis=1).astype(np.float32)


def _corrcoef(first: np.ndarray, second: np.ndarray) -> float:
    n = min(first.size, second.size)
    if n < 2:
        return 0.0
    a = np.asarray(first[:n], dtype=np.float64)
    b = np.asarray(second[:n], dtype=np.float64)
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 0.0
    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def _name_key(name: str) -> str:
    return " ".join(Path(name).stem.lower().replace("_", " ").replace("-", " ").split())


def _is_backing_vocal_key(key: str) -> bool:
    return (
        "back vox" in key
        or "back vocal" in key
        or "bgv" in key
        or key == "backs"
        or key.startswith("backs ")
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run clean Ayaic offline pipeline")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="ayaic_offline_mix")
    parser.add_argument("--master-target-lufs", type=float, default=DEFAULT_MASTER_TARGET_LUFS)
    parser.add_argument("--master-ceiling-dbfs", type=float, default=DEFAULT_MASTER_CEILING_DBFS)
    parser.add_argument("--max-phase-delay-ms", type=float, default=DEFAULT_MAX_PHASE_DELAY_MS)
    parser.add_argument("--no-mp3", action="store_true")
    parser.add_argument("--corrective-eq-method", choices=CORRECTIVE_EQ_METHODS, default=DEFAULT_CORRECTIVE_EQ_METHOD)
    parser.add_argument("--enable-autoeq", action="store_true")
    parser.add_argument("--autoeq-mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--autoeq-report-only", action="store_true")
    args = parser.parse_args(argv)
    result = run_ayaic_offline_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_name=args.output_name,
        master_target_lufs=args.master_target_lufs,
        master_ceiling_dbfs=args.master_ceiling_dbfs,
        max_phase_delay_ms=args.max_phase_delay_ms,
        write_mp3=not args.no_mp3,
        corrective_eq_method=args.corrective_eq_method,
        autoeq_enabled=bool(args.enable_autoeq),
        autoeq_mode=args.autoeq_mode,
        autoeq_report_only=bool(args.autoeq_report_only),
    )
    print(json.dumps({"wav": result.output_wav, "mp3": result.output_mp3, "report": result.report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
