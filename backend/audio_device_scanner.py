"""
Audio device scanner — discovers and classifies multichannel audio input devices.

Detects Dante Virtual Soundcard, Waves SoundGrid, USB multichannel interfaces,
CoreAudio/ALSA aggregate devices, and system defaults.

Selects the best device for live sound capture (highest channel count,
professional protocols preferred).

Usage:
    from audio_device_scanner import scan_audio_devices, select_best_device
    devices = scan_audio_devices()
    best = select_best_device(devices)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    HAS_SOUNDDEVICE = False

try:
    import pyaudio
    HAS_PYAUDIO = True
except (ImportError, OSError):
    HAS_PYAUDIO = False


class AudioProtocol(Enum):
    """Audio transport protocol / driver type."""
    SOUNDGRID = "soundgrid"
    DANTE = "dante"
    MADI = "madi"
    AVB = "avb"
    USB = "usb"
    THUNDERBOLT = "thunderbolt"
    AGGREGATE = "aggregate"
    COREAUDIO = "coreaudio"
    ALSA = "alsa"
    ASIO = "asio"
    WASAPI = "wasapi"
    SYSTEM_DEFAULT = "system_default"
    UNKNOWN = "unknown"


# Priority order for protocol selection (lower index = higher priority)
PROTOCOL_PRIORITY = [
    AudioProtocol.SOUNDGRID,
    AudioProtocol.DANTE,
    AudioProtocol.MADI,
    AudioProtocol.AVB,
    AudioProtocol.THUNDERBOLT,
    AudioProtocol.USB,
    AudioProtocol.ASIO,
    AudioProtocol.AGGREGATE,
    AudioProtocol.COREAUDIO,
    AudioProtocol.ALSA,
    AudioProtocol.WASAPI,
    AudioProtocol.SYSTEM_DEFAULT,
    AudioProtocol.UNKNOWN,
]

# Pattern matching for device name → protocol classification
DEVICE_NAME_PATTERNS: Dict[AudioProtocol, list] = {
    AudioProtocol.SOUNDGRID: [
        "soundgrid", "waves sg", "sg driver", "waves audio",
        "waves multirack", "waves ltc",
    ],
    AudioProtocol.DANTE: [
        "dante", "dante virtual soundcard", "dante via",
        "dvs", "audinate", "dante controller",
    ],
    AudioProtocol.MADI: [
        "madi", "rme madi", "digiface", "madiface",
    ],
    AudioProtocol.AVB: [
        "avb", "presonus avb", "motu avb",
    ],
    AudioProtocol.THUNDERBOLT: [
        "thunderbolt", "apollo", "universal audio",
        "ua arrow", "uad",
    ],
    AudioProtocol.USB: [
        "usb", "focusrite", "scarlett", "clarett", "motu",
        "presonus studio", "steinberg ur", "behringer umc",
        "audient", "ssl 2", "id14", "id22", "id44",
        "tascam", "zoom", "x-air", "x32-usb",
        "allen & heath qu", "qu-", "dlive-usb",
    ],
    AudioProtocol.ASIO: [
        "asio", "asio4all",
    ],
    AudioProtocol.AGGREGATE: [
        "aggregate", "multi-output",
    ],
}

# Minimum channel count to consider a device "multichannel" for live sound
MIN_MULTICHANNEL = 8


@dataclass
class AudioDevice:
    """Discovered audio input device."""
    index: int                        # Driver device index
    name: str                         # Device name from driver
    max_input_channels: int           # Number of input channels
    default_samplerate: float         # Default sample rate
    protocol: AudioProtocol           # Detected protocol/driver type
    is_multichannel: bool = False     # ≥ MIN_MULTICHANNEL channels
    is_default: bool = False          # System default device
    driver: str = ""                  # "sounddevice" or "pyaudio"
    host_api: str = ""                # PortAudio host API name
    latency_ms: float = 0.0          # Default low-latency input latency
    score: int = 0                    # Selection score (higher = better)

    def __repr__(self):
        flags = []
        if self.is_multichannel:
            flags.append("multi")
        if self.is_default:
            flags.append("default")
        flag_str = f" [{','.join(flags)}]" if flags else ""
        return (
            f"<AudioDevice [{self.index}] '{self.name}' "
            f"{self.max_input_channels}ch {self.default_samplerate:.0f}Hz "
            f"{self.protocol.value}{flag_str} score={self.score}>"
        )


def _classify_protocol(name: str) -> AudioProtocol:
    """Classify audio protocol from device name."""
    name_lower = name.lower()
    for proto, patterns in DEVICE_NAME_PATTERNS.items():
        for pat in patterns:
            if pat in name_lower:
                return proto
    if "built-in" in name_lower or "macbook" in name_lower:
        return AudioProtocol.COREAUDIO
    if "default" in name_lower:
        return AudioProtocol.SYSTEM_DEFAULT
    return AudioProtocol.UNKNOWN


def _compute_score(device: AudioDevice) -> int:
    """Compute selection score. Higher = better for live sound capture."""
    score = 0

    # Protocol priority (SoundGrid=120, Dante=110, ... Unknown=0)
    try:
        idx = PROTOCOL_PRIORITY.index(device.protocol)
        score += max(0, (len(PROTOCOL_PRIORITY) - idx) * 10)
    except ValueError:
        pass

    # Channel count bonus
    if device.max_input_channels >= 48:
        score += 50
    elif device.max_input_channels >= 32:
        score += 40
    elif device.max_input_channels >= 16:
        score += 30
    elif device.max_input_channels >= 8:
        score += 20
    elif device.max_input_channels >= 2:
        score += 5

    # Professional sample rates
    if device.default_samplerate in (48000, 96000):
        score += 10
    elif device.default_samplerate == 44100:
        score += 5

    # Low latency bonus
    if 0 < device.latency_ms <= 5:
        score += 10
    elif 0 < device.latency_ms <= 10:
        score += 5

    # Penalize system defaults and built-in devices
    if device.protocol in (AudioProtocol.SYSTEM_DEFAULT, AudioProtocol.COREAUDIO):
        if device.max_input_channels <= 2:
            score -= 30

    return score


# ── Scanner using sounddevice ────────────────────────────────────

def _scan_sounddevice() -> List[AudioDevice]:
    """Scan audio devices using sounddevice (PortAudio)."""
    if not HAS_SOUNDDEVICE:
        return []

    devices = []
    try:
        default_input = None
        try:
            default_info = sd.query_devices(kind="input")
            if default_info:
                default_input = default_info.get("name", "")
        except Exception:
            pass

        host_apis = {}
        try:
            for api in sd.query_hostapis():
                host_apis[api.get("index", -1)] = api.get("name", "")
        except Exception:
            pass

        for idx, dev in enumerate(sd.query_devices()):
            max_in = dev.get("max_input_channels", 0)
            if max_in <= 0:
                continue

            name = dev.get("name", f"Device {idx}")
            sr = dev.get("default_samplerate", 48000)
            api_idx = dev.get("hostapi", -1)
            api_name = host_apis.get(api_idx, "")
            latency = dev.get("default_low_input_latency", 0)
            latency_ms = latency * 1000 if latency else 0

            protocol = _classify_protocol(name)
            is_mc = max_in >= MIN_MULTICHANNEL
            is_default = (name == default_input) if default_input else False

            ad = AudioDevice(
                index=idx,
                name=name,
                max_input_channels=max_in,
                default_samplerate=sr,
                protocol=protocol,
                is_multichannel=is_mc,
                is_default=is_default,
                driver="sounddevice",
                host_api=api_name,
                latency_ms=latency_ms,
            )
            ad.score = _compute_score(ad)
            devices.append(ad)

    except Exception as e:
        logger.error(f"sounddevice scan error: {e}")

    return devices


# ── Scanner using PyAudio ────────────────────────────────────────

def _scan_pyaudio() -> List[AudioDevice]:
    """Scan audio devices using PyAudio."""
    if not HAS_PYAUDIO:
        return []

    devices = []
    try:
        pa = pyaudio.PyAudio()

        host_apis = {}
        for i in range(pa.get_host_api_count()):
            try:
                info = pa.get_host_api_info_by_index(i)
                host_apis[i] = info.get("name", "")
            except Exception:
                pass

        default_idx = None
        try:
            default_info = pa.get_default_input_device_info()
            default_idx = default_info.get("index")
        except Exception:
            pass

        for idx in range(pa.get_device_count()):
            try:
                dev = pa.get_device_info_by_index(idx)
            except Exception:
                continue

            max_in = int(dev.get("maxInputChannels", 0))
            if max_in <= 0:
                continue

            name = dev.get("name", f"Device {idx}")
            sr = dev.get("defaultSampleRate", 48000)
            api_idx = int(dev.get("hostApi", -1))
            api_name = host_apis.get(api_idx, "")
            latency = dev.get("defaultLowInputLatency", 0)
            latency_ms = latency * 1000 if latency else 0

            protocol = _classify_protocol(name)
            is_mc = max_in >= MIN_MULTICHANNEL
            is_default = (idx == default_idx)

            ad = AudioDevice(
                index=idx,
                name=name,
                max_input_channels=max_in,
                default_samplerate=sr,
                protocol=protocol,
                is_multichannel=is_mc,
                is_default=is_default,
                driver="pyaudio",
                host_api=api_name,
                latency_ms=latency_ms,
            )
            ad.score = _compute_score(ad)
            devices.append(ad)

        pa.terminate()

    except Exception as e:
        logger.error(f"PyAudio scan error: {e}")

    return devices


# ── Combined scanner ─────────────────────────────────────────────

def scan_audio_devices(
    use_sounddevice: bool = True,
    use_pyaudio: bool = True,
) -> List[AudioDevice]:
    """Scan all available audio input devices using available drivers.

    Returns a deduplicated list sorted by score (best first).
    """
    all_devices: List[AudioDevice] = []

    if use_sounddevice:
        sd_devs = _scan_sounddevice()
        all_devices.extend(sd_devs)
        logger.info(f"sounddevice: found {len(sd_devs)} input device(s)")

    if use_pyaudio and not all_devices:
        pa_devs = _scan_pyaudio()
        all_devices.extend(pa_devs)
        logger.info(f"PyAudio: found {len(pa_devs)} input device(s)")

    # Deduplicate by name (prefer higher-scored entry)
    seen: Dict[str, AudioDevice] = {}
    for dev in all_devices:
        key = dev.name.strip().lower()
        if key not in seen or dev.score > seen[key].score:
            seen[key] = dev

    result = sorted(seen.values(), key=lambda d: d.score, reverse=True)

    if result:
        logger.info(
            f"Audio scan complete: {len(result)} device(s), "
            f"best = '{result[0].name}' ({result[0].protocol.value}, "
            f"{result[0].max_input_channels}ch, score={result[0].score})"
        )
    else:
        logger.warning("No audio input devices found")

    return result


def select_best_device(
    devices: Optional[List[AudioDevice]] = None,
    preferred_protocol: Optional[AudioProtocol] = None,
    preferred_name: Optional[str] = None,
    min_channels: int = 2,
) -> Optional[AudioDevice]:
    """Select the best audio device for live sound capture.

    Priority:
    1. Match preferred_name pattern
    2. Match preferred_protocol
    3. Highest score with at least min_channels
    """
    if devices is None:
        devices = scan_audio_devices()

    if not devices:
        return None

    # 1) Preferred name pattern
    if preferred_name:
        pat = preferred_name.lower()
        for dev in devices:
            if pat in dev.name.lower() and dev.max_input_channels >= min_channels:
                logger.info(f"Selected by name pattern '{preferred_name}': {dev}")
                return dev

    # 2) Preferred protocol
    if preferred_protocol:
        proto_devs = [
            d for d in devices
            if d.protocol == preferred_protocol and d.max_input_channels >= min_channels
        ]
        if proto_devs:
            best = proto_devs[0]  # Already sorted by score
            logger.info(f"Selected by protocol '{preferred_protocol.value}': {best}")
            return best

    # 3) Highest score with enough channels
    for dev in devices:
        if dev.max_input_channels >= min_channels:
            logger.info(f"Selected by score: {dev}")
            return dev

    # 4) Fallback: any device
    logger.info(f"Fallback selection: {devices[0]}")
    return devices[0]


def detect_and_report() -> Dict:
    """Scan devices and return a structured report."""
    devices = scan_audio_devices()
    best = select_best_device(devices)

    device_list = []
    for dev in devices:
        device_list.append({
            "index": dev.index,
            "name": dev.name,
            "channels": dev.max_input_channels,
            "samplerate": dev.default_samplerate,
            "protocol": dev.protocol.value,
            "is_multichannel": dev.is_multichannel,
            "is_default": dev.is_default,
            "driver": dev.driver,
            "host_api": dev.host_api,
            "latency_ms": round(dev.latency_ms, 1),
            "score": dev.score,
        })

    best_info = None
    if best:
        best_info = {
            "index": best.index,
            "name": best.name,
            "channels": best.max_input_channels,
            "protocol": best.protocol.value,
            "score": best.score,
        }

    return {
        "devices": device_list,
        "count": len(device_list),
        "multichannel_count": sum(1 for d in devices if d.is_multichannel),
        "best": best_info,
        "sounddevice_available": HAS_SOUNDDEVICE,
        "pyaudio_available": HAS_PYAUDIO,
    }


# ── CLI ──────────────────────────────────────────────────────────

def main():
    """CLI tool for audio device scanning."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AUTO-MIXER — Audio Device Scanner"
    )
    parser.add_argument("--min-channels", type=int, default=2,
                        help="Minimum input channels to show (default: 2)")
    parser.add_argument("--prefer", default=None,
                        help="Preferred device name pattern")
    parser.add_argument("--protocol", default=None,
                        choices=[p.value for p in AudioProtocol],
                        help="Preferred protocol")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.json:
        import json
        report = detect_and_report()
        print(json.dumps(report, indent=2))
        return

    print("=" * 70)
    print("  AUTO-MIXER — Audio Device Scanner")
    print("=" * 70)
    print(f"\n  sounddevice available: {HAS_SOUNDDEVICE}")
    print(f"  PyAudio available:    {HAS_PYAUDIO}")
    print()

    start = time.time()
    devices = scan_audio_devices()
    elapsed = time.time() - start

    if not devices:
        print("  No audio input devices found!")
        print("\n  Possible fixes:")
        print("  - Install sounddevice: pip install sounddevice")
        print("  - Install PortAudio: brew install portaudio (macOS)")
        print("  - Check that SoundGrid/Dante driver is installed and running")
        return

    print(f"  Found {len(devices)} input device(s) in {elapsed:.2f}s:\n")

    for i, dev in enumerate(devices, 1):
        mc = " [MULTICHANNEL]" if dev.is_multichannel else ""
        df = " [DEFAULT]" if dev.is_default else ""
        print(f"  {i:2d}. [{dev.index:3d}] {dev.name}")
        print(f"       {dev.max_input_channels}ch | {dev.default_samplerate:.0f}Hz | "
              f"{dev.protocol.value}{mc}{df}")
        print(f"       Driver: {dev.driver} | API: {dev.host_api} | "
              f"Latency: {dev.latency_ms:.1f}ms | Score: {dev.score}")
        print()

    # Best selection
    preferred_proto = None
    if args.protocol:
        preferred_proto = AudioProtocol(args.protocol)

    best = select_best_device(
        devices,
        preferred_protocol=preferred_proto,
        preferred_name=args.prefer,
        min_channels=args.min_channels,
    )
    if best:
        print("  " + "-" * 66)
        print(f"  SELECTED: [{best.index}] {best.name}")
        print(f"            {best.max_input_channels}ch | {best.protocol.value} | "
              f"Score: {best.score}")
        print()


if __name__ == "__main__":
    main()
