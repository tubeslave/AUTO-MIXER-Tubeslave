"""
Dante Routing Configuration for AUTO-MIXER Tubeslave.

Defines the expected Dante channel layout, signal types, and routing scheme
for the audio analysis program. This configuration tells the user exactly
what signals to send from the mixer (Behringer WING Rack) via Dante.

Channel Layout (64-channel Dante card):
  1-24:  Individual channels, Pre-Fader (post-EQ/Dynamics)
  25-48: Individual channels, Pre-EQ (dry, post-preamp) — for soundcheck analysis
  49-50: Master L/R (post-fader)
  51-52: Drum Bus L/R (post-fader group)
  53-54: Vocal Bus L/R (post-fader group)
  55-56: Instrument Bus L/R (post-fader group)
  57:    Measurement Mic (pre-EQ, dry)
  58:    Ambient Mic (pre-fader, dry)
  59-60: Matrix 1/2 (post-EQ matrix, main PA zone)
  61-64: Reserve
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class TapPoint(Enum):
    """Signal tap point on the mixer channel strip."""
    PRE_EQ = "Pre-EQ (сухой, после преампа)"
    POST_EQ = "Post-EQ / Pre-Dynamics"
    PRE_FADER = "Pre-Fader (после EQ и динамики)"
    POST_FADER = "Post-Fader"


class SignalRole(Enum):
    """Role of the signal in the analysis chain."""
    CHANNEL_ANALYSIS = "channel_analysis"       # Main analysis path
    CHANNEL_DRY = "channel_dry"                 # Dry signal for EQ/compressor soundcheck
    MASTER = "master"                           # Master bus
    DRUM_BUS = "drum_bus"                       # Drum group bus
    VOCAL_BUS = "vocal_bus"                     # Vocal group bus
    INSTRUMENT_BUS = "instrument_bus"           # Instrument group bus
    MEASUREMENT_MIC = "measurement_mic"         # System measurement reference mic
    AMBIENT_MIC = "ambient_mic"                 # Ambient/audience noise mic
    MATRIX = "matrix"                           # Matrix output (PA zone)
    RESERVE = "reserve"                         # Unused/reserve


@dataclass
class DanteChannelRange:
    """Describes a range of Dante channels and their expected signal."""
    start: int                    # First Dante channel (1-based)
    end: int                      # Last Dante channel (1-based, inclusive)
    role: SignalRole              # What this range is for
    tap_point: TapPoint           # Where to tap on the mixer
    label_short: str              # Short label for UI (e.g., "Ch 1-24 Pre-Fader")
    label_full: str               # Full description for tooltips
    wing_routing_hint: str        # How to configure on WING Rack
    stereo: bool = False          # True if L/R pair
    required: bool = True         # False = optional, nice-to-have
    used_by: List[str] = field(default_factory=list)  # Module names that use this


# === ROUTING SCHEME PRESETS ===

ROUTING_64CH: List[DanteChannelRange] = [
    DanteChannelRange(
        start=1, end=24,
        role=SignalRole.CHANNEL_ANALYSIS,
        tap_point=TapPoint.PRE_FADER,
        label_short="Ch 1–24: Pre-Fader",
        label_full="Каналы микшера 1–24, точка отбора Pre-Fader (после EQ и динамической обработки, до фейдера). Основной сигнал для анализа всех модулей.",
        wing_routing_hint="WING: Direct Out → Dante 1-24, Tap = Pre-Fader",
        required=True,
        used_by=["Gain Staging", "Auto Fader", "Auto Gate", "Auto Phase", "Auto Effects", "Auto Panner", "Cross-Adaptive EQ"]
    ),
    DanteChannelRange(
        start=25, end=48,
        role=SignalRole.CHANNEL_DRY,
        tap_point=TapPoint.PRE_EQ,
        label_short="Ch 25–48: Dry (Pre-EQ)",
        label_full="Те же каналы 1–24, но точка отбора Pre-EQ (сухой сигнал после преампа). Для анализа в режиме Soundcheck — Auto EQ и Auto Compressor видят необработанный сигнал.",
        wing_routing_hint="WING: User Signal 1-24 → Ch 1-24 Tap=Input → Dante 25-48",
        required=False,
        used_by=["Auto EQ (Soundcheck)", "Auto Compressor (Soundcheck)"]
    ),
    DanteChannelRange(
        start=49, end=50,
        role=SignalRole.MASTER,
        tap_point=TapPoint.POST_FADER,
        label_short="Ch 49–50: Master L/R",
        label_full="Master шина L/R (Post-Fader). Контроль суммарного уровня микса, референс для всех модулей.",
        wing_routing_hint="WING: Main 1 L/R → Dante 49-50",
        stereo=True,
        required=True,
        used_by=["Cross-Adaptive EQ", "Auto Fader", "System Measurement"]
    ),
    DanteChannelRange(
        start=51, end=52,
        role=SignalRole.DRUM_BUS,
        tap_point=TapPoint.POST_FADER,
        label_short="Ch 51–52: Drum Bus L/R",
        label_full="Группа барабанов (Bus/Group Post-Fader). Cross-adaptive гейтинг, bus-компрессия, баланс групп.",
        wing_routing_hint="WING: Bus (Drums) → Dante 51-52",
        stereo=True,
        required=False,
        used_by=["Auto Gate (Cross-Adaptive)", "Auto Compressor (Bus)", "Auto Fader (Balance)"]
    ),
    DanteChannelRange(
        start=53, end=54,
        role=SignalRole.VOCAL_BUS,
        tap_point=TapPoint.POST_FADER,
        label_short="Ch 53–54: Vocal Bus L/R",
        label_full="Группа вокалов (Bus/Group Post-Fader). Cross-Adaptive EQ для разрешения маскинга вокал/гитары.",
        wing_routing_hint="WING: Bus (Vocals) → Dante 53-54",
        stereo=True,
        required=False,
        used_by=["Cross-Adaptive EQ", "Auto Compressor (Bus)", "Auto Reverb"]
    ),
    DanteChannelRange(
        start=55, end=56,
        role=SignalRole.INSTRUMENT_BUS,
        tap_point=TapPoint.POST_FADER,
        label_short="Ch 55–56: Instrument Bus L/R",
        label_full="Группа инструментов (Bus/Group Post-Fader). Баланс групп «инструменты vs вокал vs барабаны».",
        wing_routing_hint="WING: Bus (Instruments) → Dante 55-56",
        stereo=True,
        required=False,
        used_by=["Auto Fader (Balance)", "Cross-Adaptive EQ"]
    ),
    DanteChannelRange(
        start=57, end=57,
        role=SignalRole.MEASUREMENT_MIC,
        tap_point=TapPoint.PRE_EQ,
        label_short="Ch 57: Measurement Mic",
        label_full="Референсный микрофон (omnidirectional) на позиции FOH. Pre-EQ (сухой). Для измерения АЧХ помещения, RT60.",
        wing_routing_hint="WING: Measurement Mic channel → Dante 57, Tap = Input (no processing)",
        required=False,
        used_by=["System Measurement"]
    ),
    DanteChannelRange(
        start=58, end=58,
        role=SignalRole.AMBIENT_MIC,
        tap_point=TapPoint.PRE_FADER,
        label_short="Ch 58: Ambient Mic",
        label_full="Микрофон зала / аудитории. Pre-Fader. Отслеживание noise floor зала (зал заполняется → шум растёт → пороги гейтов адаптируются).",
        wing_routing_hint="WING: Ambient Mic channel → Dante 58",
        required=False,
        used_by=["Auto Gate (Noise Floor)", "Auto Fader (Noise Threshold)"]
    ),
    DanteChannelRange(
        start=59, end=60,
        role=SignalRole.MATRIX,
        tap_point=TapPoint.POST_FADER,
        label_short="Ch 59–60: Matrix (Main PA)",
        label_full="Matrix выход основной зоны PA (Post-EQ/Fader). Для сравнения с measurement mic и зональной коррекции.",
        wing_routing_hint="WING: Matrix 1/2 → Dante 59-60",
        stereo=True,
        required=False,
        used_by=["System Measurement"]
    ),
    DanteChannelRange(
        start=61, end=64,
        role=SignalRole.RESERVE,
        tap_point=TapPoint.POST_FADER,
        label_short="Ch 61–64: Резерв",
        label_full="Резервные каналы для дополнительных matrix-зон, delay tower, side-fill и т.д.",
        wing_routing_hint="",
        required=False,
        used_by=[]
    ),
]

ROUTING_32CH: List[DanteChannelRange] = [
    DanteChannelRange(
        start=1, end=24,
        role=SignalRole.CHANNEL_ANALYSIS,
        tap_point=TapPoint.PRE_FADER,
        label_short="Ch 1–24: Pre-Fader",
        label_full="Каналы микшера 1–24, точка отбора Pre-Fader (после EQ и динамической обработки, до фейдера). Основной сигнал для всех модулей.",
        wing_routing_hint="WING: Direct Out → Dante 1-24, Tap = Pre-Fader",
        required=True,
        used_by=["Все модули"]
    ),
    DanteChannelRange(
        start=25, end=26,
        role=SignalRole.MASTER,
        tap_point=TapPoint.POST_FADER,
        label_short="Ch 25–26: Master L/R",
        label_full="Master шина L/R (Post-Fader).",
        wing_routing_hint="WING: Main 1 L/R → Dante 25-26",
        stereo=True,
        required=True,
        used_by=["Cross-Adaptive EQ", "Auto Fader", "System Measurement"]
    ),
    DanteChannelRange(
        start=27, end=28,
        role=SignalRole.DRUM_BUS,
        tap_point=TapPoint.POST_FADER,
        label_short="Ch 27–28: Drum Bus L/R",
        label_full="Группа барабанов (Post-Fader).",
        wing_routing_hint="WING: Bus (Drums) → Dante 27-28",
        stereo=True,
        required=False,
        used_by=["Auto Gate", "Auto Compressor"]
    ),
    DanteChannelRange(
        start=29, end=29,
        role=SignalRole.MEASUREMENT_MIC,
        tap_point=TapPoint.PRE_EQ,
        label_short="Ch 29: Measurement Mic",
        label_full="Референсный микрофон FOH (Pre-EQ, сухой).",
        wing_routing_hint="WING: Meas. Mic → Dante 29, Tap = Input",
        required=False,
        used_by=["System Measurement"]
    ),
    DanteChannelRange(
        start=30, end=30,
        role=SignalRole.AMBIENT_MIC,
        tap_point=TapPoint.PRE_FADER,
        label_short="Ch 30: Ambient Mic",
        label_full="Микрофон зала (Pre-Fader). Noise floor tracking.",
        wing_routing_hint="WING: Ambient Mic → Dante 30",
        required=False,
        used_by=["Auto Gate", "Auto Fader"]
    ),
    DanteChannelRange(
        start=31, end=32,
        role=SignalRole.RESERVE,
        tap_point=TapPoint.POST_FADER,
        label_short="Ch 31–32: Резерв",
        label_full="Резерв (Vocal Bus, Matrix и т.д.)",
        wing_routing_hint="",
        required=False,
        used_by=[]
    ),
]


# Module → expected signal info (for display in each module's UI)
MODULE_SIGNAL_INFO: Dict[str, Dict] = {
    "gain_staging": {
        "signal": "Pre-Fader (Dante 1–24)",
        "tap_point": TapPoint.PRE_FADER.value,
        "description": "Измерение LUFS и True Peak до фейдера. Фейдер не влияет на анализ.",
        "icon": "📊",
    },
    "auto_eq": {
        "signal": "Pre-EQ / Dry (Dante 25–48) — Soundcheck\nPre-Fader (Dante 1–24) — Live",
        "tap_point": "Soundcheck: Pre-EQ | Live: Pre-Fader",
        "description": "Soundcheck: анализ сырого спектра для расчёта коррекций.\nLive: мониторинг результата после применения EQ.",
        "icon": "〰",
    },
    "auto_compressor": {
        "signal": "Pre-EQ / Dry (Dante 25–48) — Soundcheck\nPre-Fader (Dante 1–24) — Live",
        "tap_point": "Soundcheck: Pre-EQ | Live: Pre-Fader",
        "description": "Soundcheck: Crest Factor и динамика чистого сигнала.\nLive: мониторинг после компрессии.",
        "icon": "⬇",
    },
    "auto_gate": {
        "signal": "Pre-Fader (Dante 1–24)\n+ Drum Bus (Dante 51–52)\n+ Ambient Mic (Dante 58)",
        "tap_point": TapPoint.PRE_FADER.value,
        "description": "RMS/Peak до фейдера. Drum Bus для cross-adaptive гейтинга. Ambient Mic для noise floor зала.",
        "icon": "🚪",
    },
    "auto_fader": {
        "signal": "Pre-Fader (Dante 1–24)\n+ Master L/R (Dante 49–50)",
        "tap_point": TapPoint.PRE_FADER.value,
        "description": "Pre-Fader уровень для управления фейдерами без обратной связи. Master для контроля суммы.",
        "icon": "🎚",
    },
    "auto_phase": {
        "signal": "Pre-Fader (Dante 1–24)",
        "tap_point": TapPoint.PRE_FADER.value,
        "description": "GCC-PHAT кросс-корреляция между reference и target каналами.",
        "icon": "⟳",
    },
    "auto_panner": {
        "signal": "Pre-Fader (Dante 1–24)",
        "tap_point": TapPoint.PRE_FADER.value,
        "description": "Spectral centroid для определения панорамы по частотному содержанию.",
        "icon": "🎧",
    },
    "auto_reverb": {
        "signal": "Pre-Fader (Dante 1–24)\n+ Vocal Bus (Dante 53–54)",
        "tap_point": TapPoint.PRE_FADER.value,
        "description": "Spectral flux для адаптации decay. Vocal Bus для уровня реверба.",
        "icon": "🌊",
    },
    "auto_effects": {
        "signal": "Pre-Fader (Dante 1–24)",
        "tap_point": TapPoint.PRE_FADER.value,
        "description": "13 аудио-фичей (RMS, Peak, Crest, Spectral) для cross-adaptive автоматизации.",
        "icon": "✨",
    },
    "cross_adaptive_eq": {
        "signal": "Pre-Fader (Dante 1–24)\n+ Master L/R (Dante 49–50)\n+ Bus'ы (Dante 51–56)",
        "tap_point": TapPoint.PRE_FADER.value,
        "description": "Band energy каналов для mirror-EQ. Master и bus'ы как референс маскинга.",
        "icon": "🔀",
    },
    "system_measurement": {
        "signal": "Measurement Mic (Dante 57)\n+ Matrix (Dante 59–60)",
        "tap_point": TapPoint.PRE_EQ.value,
        "description": "Sweep → PA → Measurement Mic. Сравнение с Matrix для зональной коррекции.",
        "icon": "📐",
    },
    "auto_soundcheck": {
        "signal": "Pre-EQ / Dry (Dante 25–48)",
        "tap_point": TapPoint.PRE_EQ.value,
        "description": "Полный саундчек с анализом сырых сигналов. Gain + EQ + Comp + Gate + Pan + Reverb.",
        "icon": "🎯",
    },
}


def get_routing_scheme(total_dante_channels: int = 64) -> List[DanteChannelRange]:
    """Get routing scheme based on available Dante channels."""
    if total_dante_channels >= 64:
        return ROUTING_64CH
    elif total_dante_channels >= 32:
        return ROUTING_32CH
    else:
        # Minimal: just individual channels
        return [
            DanteChannelRange(
                start=1, end=min(total_dante_channels, 24),
                role=SignalRole.CHANNEL_ANALYSIS,
                tap_point=TapPoint.PRE_FADER,
                label_short=f"Ch 1–{min(total_dante_channels, 24)}: Pre-Fader",
                label_full="Каналы микшера Pre-Fader. Минимальная конфигурация.",
                wing_routing_hint="WING: Direct Out → Dante, Tap = Pre-Fader",
                required=True,
                used_by=["Все модули"]
            )
        ]


def get_routing_as_dict(total_dante_channels: int = 64) -> List[dict]:
    """Get routing scheme as JSON-serializable dict for frontend."""
    scheme = get_routing_scheme(total_dante_channels)
    result = []
    for r in scheme:
        result.append({
            "start": r.start,
            "end": r.end,
            "role": r.role.value,
            "tap_point": r.tap_point.value,
            "label_short": r.label_short,
            "label_full": r.label_full,
            "wing_routing_hint": r.wing_routing_hint,
            "stereo": r.stereo,
            "required": r.required,
            "used_by": r.used_by,
        })
    return result


def get_module_signal_info() -> Dict[str, Dict]:
    """Get module signal info as JSON-serializable dict for frontend."""
    return MODULE_SIGNAL_INFO
