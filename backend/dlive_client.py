"""Allen & Heath dLive MIDI over TCP/IP client.

The transport codec in this module is intentionally limited to the public
Allen & Heath protocol for dLive firmware v2.0 and later:

https://www.allen-heath.com/content/uploads/2024/06/dLive-MIDI-Over-TCP-Protocol-V2.0.pdf

Read-only queries cover input fader, mute and name. Normal client instances
have no mutation surface. The only encoded mutation retained here is an input
fader write reachable solely through the one-shot attended proof authority in
``dlive_supervised_fader_proof``. Every other mutation family fails closed
without sending bytes until a separate mapping and attended proof exists.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import os
import re
import socket
import ssl
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from mixer_client_base import MixerClientBase

logger = logging.getLogger(__name__)

DLIVE_MIXRACK_TCP_PORT = 51325
DLIVE_MIXRACK_TLS_PORT = 51327
DLIVE_SURFACE_TCP_PORT = 51328
DLIVE_SURFACE_TLS_PORT = 51329

# Backwards-compatible names mean Surface endpoints, as they did historically.
DLIVE_TCP_PORT = DLIVE_SURFACE_TCP_PORT
DLIVE_TLS_PORT = DLIVE_SURFACE_TLS_PORT

DLIVE_ENDPOINT_PORTS = {
    ("mixrack", False): DLIVE_MIXRACK_TCP_PORT,
    ("mixrack", True): DLIVE_MIXRACK_TLS_PORT,
    ("surface", False): DLIVE_SURFACE_TCP_PORT,
    ("surface", True): DLIVE_SURFACE_TLS_PORT,
}

SYSEX_HEADER = bytes((0xF0, 0x00, 0x00, 0x1A, 0x50, 0x10, 0x01, 0x00))
SYSEX_BODY_HEADER = SYSEX_HEADER[1:]
SYSEX_END = 0xF7
DLIVE_MAX_SYSEX_FRAME_BYTES = 64 * 1024

FADER_PARAMETER_ID = 0x17
FADER_MAX = 0x7F
FADER_0DB = 0x6B
FADER_NEG_INF = 0x00

PEQ_BANDS = {
    1: {"type": 0x1A, "freq": 0x1B, "width": 0x1C, "gain": 0x1D},
    2: {"type": 0x1E, "freq": 0x1F, "width": 0x20, "gain": 0x21},
    3: {"type": 0x22, "freq": 0x23, "width": 0x24, "gain": 0x25},
    4: {"type": 0x26, "freq": 0x27, "width": 0x28, "gain": 0x29},
}
HPF_FREQ_PARAMETER_ID = 0x30
HPF_ON_PARAMETER_ID = 0x31

# MIDI status-channel offset, selector base, maximum count.
CHANNEL_SELECTION = {
    "input": (0, 0x00, 128),
    "mono_group": (1, 0x00, 62),
    "stereo_group": (1, 0x40, 31),
    "mono_aux": (2, 0x00, 62),
    "stereo_aux": (2, 0x40, 31),
    "mono_matrix": (3, 0x00, 62),
    "stereo_matrix": (3, 0x40, 31),
    "mono_fx_send": (4, 0x00, 16),
    "stereo_fx_send": (4, 0x10, 16),
    "fx_return": (4, 0x20, 16),
    "main": (4, 0x30, 6),
    "dca": (4, 0x36, 24),
    "mute_group": (4, 0x4E, 8),
    "stereo_ufx_send": (4, 0x56, 8),
    "stereo_ufx_return": (4, 0x5E, 8),
}
CHANNEL_TYPE_OFFSET = {
    name: values[0] for name, values in CHANNEL_SELECTION.items()
}

COLOURS = {
    "off": 0x00,
    "red": 0x01,
    "green": 0x02,
    "yellow": 0x03,
    "blue": 0x04,
    "purple": 0x05,
    "lt_blue": 0x06,
    "white": 0x07,
}

NORMAL_WRITE_FAMILIES = frozenset()
RESTORE_ONLY_PROOF_WRITE_FAMILIES = frozenset({"fader"})
_DLIVE_SUPERVISED_WRITE_SEAL = object()


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field}_must_be_finite_number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}_must_be_finite_number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field}_must_be_finite_number")
    return parsed


def _valid_host(host: str) -> bool:
    if not host or host != host.strip() or len(host) > 253:
        return False
    try:
        parsed = ipaddress.ip_address(host)
        return parsed.version == 4
    except ValueError:
        labels = host.rstrip(".").split(".")
        if len(labels) == 4 and all(label.isdigit() for label in labels):
            return False
        return all(
            label
            and len(label) <= 63
            and re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
            for label in labels
        )


def validate_dlive_connection_config(
    *,
    host: str,
    endpoint: str = "surface",
    tls: bool = False,
    port: Optional[int] = None,
    midi_base_channel: int = 0,
    user_profile: int = 0,
) -> Dict[str, Any]:
    """Validate dLive connection inputs without DNS, sockets or console I/O."""

    endpoint_value = str(endpoint or "").strip().lower()
    if endpoint_value not in {"surface", "mixrack"}:
        raise ValueError("dlive_endpoint_must_be_surface_or_mixrack")
    if not isinstance(tls, bool):
        raise ValueError("dlive_tls_must_be_boolean")
    if not _valid_host(str(host or "")):
        raise ValueError("dlive_host_invalid")
    if isinstance(midi_base_channel, bool) or not isinstance(midi_base_channel, int):
        raise ValueError("dlive_midi_base_channel_must_be_integer")
    if not 0 <= midi_base_channel <= 11:
        raise ValueError("dlive_midi_base_channel_out_of_range_0_11")
    if isinstance(user_profile, bool) or not isinstance(user_profile, int):
        raise ValueError("dlive_user_profile_must_be_integer")
    if not 0 <= user_profile <= 0x1F:
        raise ValueError("dlive_user_profile_out_of_range_0_31")

    expected_port = DLIVE_ENDPOINT_PORTS[(endpoint_value, tls)]
    effective_port = expected_port if port is None else port
    if isinstance(effective_port, bool) or not isinstance(effective_port, int):
        raise ValueError("dlive_port_must_be_integer")
    if effective_port != expected_port:
        raise ValueError(
            f"dlive_port_endpoint_tls_mismatch:expected_{expected_port}"
        )

    return {
        "schema": "automixer.dlive_connection_preflight.v1",
        "status": "valid_no_transport_config",
        "host": str(host),
        "endpoint": endpoint_value,
        "tls": tls,
        "port": effective_port,
        "midi_base_channel": midi_base_channel,
        "user_profile": user_profile,
        "socket_created": False,
        "console_contacted": False,
        "midi_sent": False,
        "live_write": False,
        "live_authority_granted": False,
    }


def db_to_fader_value(db: float) -> int:
    """Convert dB to the official 7-bit dLive fader value."""

    value = _finite_float(db, "fader_db")
    if value <= -54.0:
        return FADER_NEG_INF
    value = min(10.0, value)
    return max(1, min(FADER_MAX, int(((value + 54.0) / 64.0) * 127.0)))


def fader_value_to_db(value: int) -> float:
    """Convert the official 7-bit dLive fader value to dB."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("fader_value_must_be_integer")
    value = max(0, min(FADER_MAX, value))
    if value == FADER_NEG_INF:
        return -100.0
    return (float(value) * 64.0 / 127.0) - 54.0


def db_to_gain_value(db: float) -> int:
    """Convert MixRack socket preamp gain (+5..+60 dB) to 7-bit GV."""

    value = max(5.0, min(60.0, _finite_float(db, "preamp_gain_db")))
    return max(0, min(0x7F, int(((value - 5.0) / 55.0) * 127.0)))


def freq_to_nrpn(freq_hz: float) -> int:
    """Convert PEQ frequency (20 Hz..20 kHz) to official 7-bit value."""

    freq = max(20.0, min(20000.0, _finite_float(freq_hz, "peq_frequency_hz")))
    raw = 127.0 * ((4608.0 * math.log2(freq / 4.0) - 10699.0) / 45922.0)
    return max(0, min(0x7F, int(raw)))


def hpf_freq_to_nrpn(freq_hz: float) -> int:
    """Convert HPF frequency to the official 7-bit value."""

    freq = max(20.0, min(20000.0, _finite_float(freq_hz, "hpf_frequency_hz")))
    raw = 127.0 * ((4608.0 * math.log2(freq / 4.0) - 10699.0) / 41314.0)
    return max(0, min(0x7F, int(raw)))


def db_to_eq_gain(db: float) -> int:
    """Convert PEQ gain (-15..+15 dB) to official 0..126 value."""

    value = max(-15.0, min(15.0, _finite_float(db, "peq_gain_db")))
    return max(0, min(0x7E, int(((value + 15.0) * 126.0) / 30.0)))


_WIDTH_VALUES = (
    1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7,
    2.0 / 3.0, 0.6, 0.55, 0.5, 0.45, 0.4, 1.0 / 3.0, 0.3, 0.25,
    0.2, 1.0 / 6.0, 0.13, 1.0 / 9.0,
)


def q_to_nrpn(q: float) -> int:
    """Approximate Q as the nearest documented dLive width value."""

    q_value = max(0.1, _finite_float(q, "peq_q"))
    width_octaves = 2.0 * math.asinh(1.0 / (2.0 * q_value)) / math.log(2.0)
    return min(range(len(_WIDTH_VALUES)), key=lambda i: abs(_WIDTH_VALUES[i] - width_octaves))


def build_nrpn(
    midi_channel: int,
    channel_selector: int,
    parameter_id: int,
    value: int,
) -> bytes:
    """Build the documented dLive 7-bit NRPN running-status sequence.

    Allen & Heath documents the repeated ``BN`` status bytes in square
    brackets and describes dLive running status. The proof path emits that
    compact seven-byte form. Hardware no-op attempts do not establish that a
    standards-valid full-status form is intrinsically incompatible; receive
    enablement, endpoint state and the actual wire send remain separate gates.
    """

    status = 0xB0 | (int(midi_channel) & 0x0F)
    return bytes(
        (
            status, 0x63, int(channel_selector) & 0x7F,
            0x62, int(parameter_id) & 0x7F,
            0x06, int(value) & 0x7F,
        )
    )


def build_note_on(channel: int, note: int, velocity: int) -> bytes:
    return bytes((0x90 | (int(channel) & 0x0F), note & 0x7F, velocity & 0x7F))


def build_pitchbend(channel: int, socket_selector: int, gain_value: int) -> bytes:
    """Build dLive socket-preamp gain: EN, MP, GV."""

    return bytes(
        (0xE0 | (int(channel) & 0x0F), socket_selector & 0x7F, gain_value & 0x7F)
    )


def build_sysex(payload: bytes) -> bytes:
    return SYSEX_HEADER + bytes(payload) + bytes((SYSEX_END,))


def build_program_change(channel: int, bank: int, program: int) -> bytes:
    """Build official scene bank-select CC0 + Program Change bytes."""

    midi_channel = int(channel) & 0x0F
    return bytes((0xB0 | midi_channel, 0x00, bank & 0x7F, 0xC0 | midi_channel, program & 0x7F))


def build_get_fader(midi_channel: int, selector: int) -> bytes:
    return build_sysex(bytes((midi_channel & 0x0F, 0x05, 0x0B, 0x17, selector & 0x7F)))


def build_get_mute(midi_channel: int, selector: int) -> bytes:
    return build_sysex(bytes((midi_channel & 0x0F, 0x05, 0x09, selector & 0x7F)))


def build_get_name(midi_channel: int, selector: int) -> bytes:
    return build_sysex(bytes((midi_channel & 0x0F, 0x01, selector & 0x7F)))


class DLiveClient(MixerClientBase):
    """Narrow, fail-closed dLive MIDI/TCP client."""

    def __init__(
        self,
        ip: str = "192.168.3.70",
        port: Optional[int] = None,
        tls: bool = False,
        tls_ca_file: Optional[str] = None,
        tls_server_hostname: Optional[str] = None,
        midi_base_channel: int = 0,
        user_profile: int = 0,
        password: str = "",
        endpoint: str = "surface",
        auto_reconnect: bool = False,
    ):
        validated = validate_dlive_connection_config(
            host=ip,
            endpoint=endpoint,
            tls=tls,
            port=port,
            midi_base_channel=midi_base_channel,
            user_profile=user_profile,
        )
        self.ip = validated["host"]
        self.endpoint = validated["endpoint"]
        self.port = validated["port"]
        self.tls = validated["tls"]
        self.tls_ca_file = tls_ca_file
        self.tls_server_hostname = tls_server_hostname
        self.midi_base_channel = validated["midi_base_channel"]
        self.user_profile = validated["user_profile"]
        self.password = str(password or "")
        self.auto_reconnect = bool(auto_reconnect)

        self._sock: Optional[socket.socket] = None
        self._ssl_sock: Optional[ssl.SSLSocket] = None
        self.is_connected = False
        self._lock = threading.Lock()
        self._recv_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._reconnect_interval = 3.0
        self._teardown_failed = False

        self.state: Dict[str, Any] = {}
        self.state_timestamps: Dict[str, float] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self._channel_names: Dict[int, str] = {}
        self._parse_buffer = bytearray()
        self._running_status: Optional[int] = None
        self._discarding_oversized_sysex = False

        try:
            ceiling = float(os.getenv("AUTOMIXER_FADER_CEILING_DB", "") or 0.0)
        except ValueError:
            ceiling = 0.0
        self._fader_ceiling_db = max(-54.0, min(10.0, ceiling))

        logger.info(
            "DLiveClient initialized for %s %s:%s (TLS=%s, base_ch=%s, normal_writes=none)",
            self.endpoint,
            self.ip,
            self.port,
            self.tls,
            self.midi_base_channel,
        )

    def connect(self, timeout: float = 5.0) -> bool:
        if self._teardown_failed:
            logger.error("dLive connection refused after unverified transport teardown")
            return False
        if self.is_connected:
            return True
        validate_dlive_connection_config(
            host=self.ip,
            endpoint=self.endpoint,
            tls=self.tls,
            port=self.port,
            midi_base_channel=self.midi_base_channel,
            user_profile=self.user_profile,
        )
        peer_ip = ipaddress.ip_address(self.ip)
        if not self.tls and not peer_ip.is_loopback:
            logger.error("dLive connection refused: non-loopback cleartext is disabled")
            return False
        tls_context: Optional[ssl.SSLContext] = None
        tls_password = b""
        if self.tls:
            if not self.tls_server_hostname:
                logger.error("dLive TLS connection refused: server identity is required")
                return False
            if not self.tls_ca_file:
                logger.error("dLive TLS connection refused: CA file is required")
                return False
            try:
                tls_password = self.password.encode("utf-8")
            except UnicodeEncodeError:
                logger.error("dLive TLS connection refused: password encoding is invalid")
                return False
            if (
                not tls_password
                or len(tls_password) > 128
                or any(byte in tls_password for byte in (0x00, 0x0A, 0x0D, 0x2C))
            ):
                logger.error("dLive TLS connection refused: password is invalid")
                return False
            try:
                # Load trust and enable hostname verification before any network contact.
                tls_context = ssl.create_default_context(cafile=self.tls_ca_file)
            except Exception as exc:
                logger.error("dLive TLS trust configuration failed: %s", exc)
                return False
        raw: Optional[socket.socket] = None
        try:
            raw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            raw.settimeout(float(timeout))
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            raw.connect((self.ip, self.port))
            connected: socket.socket = raw
            if self.tls:
                assert tls_context is not None
                connected = tls_context.wrap_socket(
                    raw,
                    server_hostname=self.tls_server_hostname,
                )
                self._ssl_sock = connected  # type: ignore[assignment]
                auth = bytes((self.user_profile,)) + b"," + tls_password
                connected.sendall(auth)
                response = connected.recv(256)
                if not response.startswith(b"AuthOK"):
                    raise ConnectionError("dlive_tls_auth_failed")
            self._reset_parser_state()
            self._sock = connected
            self.is_connected = True
            self._stop_event.clear()
            self._recv_thread = threading.Thread(
                target=self._receiver_loop,
                name="dlive-midi-receiver",
                daemon=True,
            )
            self._recv_thread.start()
            logger.info("Connected to dLive %s at %s:%s", self.endpoint, self.ip, self.port)
            return True
        except Exception as exc:
            logger.error("dLive connection failed: %s", exc)
            self.is_connected = False
            failed_transport = self._ssl_sock or raw
            self._sock = None
            self._ssl_sock = None
            if failed_transport is not None:
                try:
                    failed_transport.close()
                except Exception as close_exc:
                    self._teardown_failed = True
                    logger.error("dLive failed transport close error: %s", close_exc)
            return False

    def disconnect(self) -> bool:
        """Stop the socket/receiver and return verified teardown status."""
        self._stop_event.set()
        self.is_connected = False
        sock = self._sock
        self._sock = None
        self._ssl_sock = None
        teardown_ok = not self._teardown_failed
        if sock is not None:
            try:
                sock.close()
            except Exception as exc:
                teardown_ok = False
                self._teardown_failed = True
                logger.error("dLive socket close error: %s", exc)
        thread = self._recv_thread
        if thread and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():
                teardown_ok = False
                self._teardown_failed = True
                logger.error("dLive receiver thread did not stop")
            else:
                self._recv_thread = None
        elif thread is None or thread is not threading.current_thread():
            self._recv_thread = None
        self._reset_parser_state()
        if teardown_ok:
            logger.info("dLive disconnected")
        return teardown_ok

    def _send_raw(self, data: bytes) -> bool:
        payload = bytes(data)
        if not payload:
            return False
        with self._lock:
            if not self.is_connected or self._sock is None:
                return False
            try:
                self._sock.sendall(payload)
                return True
            except Exception as exc:
                logger.error("dLive send error: %s", exc)
                self.is_connected = False
                return False

    def _receiver_loop(self):
        while not self._stop_event.is_set() and self.is_connected:
            try:
                sock = self._sock
                if sock is None:
                    break
                sock.settimeout(1.0)
                data = sock.recv(4096)
                if not data:
                    self.is_connected = False
                    break
                self._parse_incoming(data)
            except socket.timeout:
                continue
            except Exception as exc:
                if self.is_connected:
                    logger.error("dLive receiver error: %s", exc)
                self.is_connected = False
                break
        if self.auto_reconnect and not self._stop_event.wait(self._reconnect_interval):
            self._recv_thread = None
            self.connect()

    @staticmethod
    def _message_data_length(status: int) -> Optional[int]:
        kind = status & 0xF0
        if kind in {0x80, 0x90, 0xA0, 0xB0, 0xE0}:
            return 2
        if kind in {0xC0, 0xD0}:
            return 1
        return None

    def _parse_incoming(self, data: bytes):
        """Parse fragmented MIDI/TCP, including MIDI running status."""

        offset = 0
        while offset < len(data):
            if self._discarding_oversized_sysex:
                terminator = data.find(bytes((SYSEX_END,)), offset)
                if terminator < 0:
                    return
                self._discarding_oversized_sysex = False
                offset = terminator + 1
                continue

            available = DLIVE_MAX_SYSEX_FRAME_BYTES - len(self._parse_buffer)
            if available <= 0:
                self._discard_oversized_sysex()
                continue

            chunk_end = min(len(data), offset + available)
            self._parse_buffer.extend(data[offset:chunk_end])
            offset = chunk_end
            self._drain_parse_buffer()

            if (
                self._parse_buffer
                and self._parse_buffer[0] == 0xF0
                and len(self._parse_buffer) >= DLIVE_MAX_SYSEX_FRAME_BYTES
            ):
                self._discard_oversized_sysex()

    def _discard_oversized_sysex(self) -> None:
        logger.warning(
            "Discarding dLive SysEx frame exceeding %s bytes",
            DLIVE_MAX_SYSEX_FRAME_BYTES,
        )
        self._parse_buffer.clear()
        self._running_status = None
        self._discarding_oversized_sysex = True

    def _reset_parser_state(self) -> None:
        self._parse_buffer.clear()
        self._running_status = None
        self._discarding_oversized_sysex = False

    def _drain_parse_buffer(self) -> None:
        while self._parse_buffer:
            first = self._parse_buffer[0]
            if first == 0xF0:
                try:
                    end = self._parse_buffer.index(SYSEX_END, 1)
                except ValueError:
                    return
                body = bytes(self._parse_buffer[1:end])
                del self._parse_buffer[: end + 1]
                self._running_status = None
                self._handle_sysex(body)
                continue
            if first >= 0xF8:
                del self._parse_buffer[0]
                continue

            if first & 0x80:
                data_length = self._message_data_length(first)
                if data_length is None:
                    del self._parse_buffer[0]
                    self._running_status = None
                    continue
                if len(self._parse_buffer) < data_length + 1:
                    return
                status = first
                message_data = bytes(self._parse_buffer[1 : data_length + 1])
                del self._parse_buffer[: data_length + 1]
                self._running_status = status
            else:
                status = self._running_status
                if status is None:
                    del self._parse_buffer[0]
                    continue
                data_length = self._message_data_length(status)
                if data_length is None or len(self._parse_buffer) < data_length:
                    return
                message_data = bytes(self._parse_buffer[:data_length])
                del self._parse_buffer[:data_length]
            self._dispatch_midi(status, message_data)

    def _dispatch_midi(self, status: int, data: bytes) -> None:
        kind = status & 0xF0
        channel = status & 0x0F
        if kind == 0xB0:
            self._handle_cc(channel, data[0], data[1])
        elif kind == 0x90:
            self._handle_note(channel, data[0], data[1])
        elif kind == 0x80:
            self._handle_note(channel, data[0], 0)
        elif kind == 0xE0:
            self._handle_pitchbend(channel, data[0], data[1])
        elif kind == 0xC0:
            self._set_state_value(f"program/{channel}", data[0])

    def _set_state_value(self, key: str, value: Any) -> None:
        self.state[key] = value
        self.state_timestamps[key] = time.time()

    def _selector(self, channel: int, channel_type: str = "input") -> Tuple[int, int]:
        if channel_type not in CHANNEL_SELECTION:
            raise ValueError("dlive_channel_type_unsupported")
        offset, selector_base, maximum = CHANNEL_SELECTION[channel_type]
        if isinstance(channel, bool) or not isinstance(channel, int) or not 1 <= channel <= maximum:
            raise ValueError(f"dlive_{channel_type}_channel_out_of_range_1_{maximum}")
        return self.midi_base_channel + offset, selector_base + channel - 1

    def _decode_selector(self, midi_channel: int, selector: int) -> Optional[Tuple[str, int]]:
        offset = midi_channel - self.midi_base_channel
        for channel_type, (expected_offset, selector_base, maximum) in CHANNEL_SELECTION.items():
            if expected_offset != offset:
                continue
            if selector_base <= selector < selector_base + maximum:
                return channel_type, selector - selector_base + 1
        return None

    @staticmethod
    def _surface_key(kind: str, channel_type: str, channel: int) -> str:
        if channel_type == "input":
            return f"{kind}/{channel}"
        return f"{kind}/{channel_type}/{channel}"

    def _handle_cc(self, channel: int, cc: int, value: int):
        self._set_state_value(f"cc/{channel}/{cc}", value)
        if cc == 0x63:
            self._set_state_value(f"_nrpn_selector/{channel}", value)
        elif cc == 0x62:
            self._set_state_value(f"_nrpn_parameter/{channel}", value)
        elif cc == 0x06:
            selector = self.state.get(f"_nrpn_selector/{channel}")
            parameter = self.state.get(f"_nrpn_parameter/{channel}")
            if isinstance(selector, int) and isinstance(parameter, int):
                decoded = self._decode_selector(channel, selector)
                if decoded is not None:
                    channel_type, channel_number = decoded
                    raw_key = f"nrpn/{channel_type}/{channel_number}/{parameter}"
                    self._set_state_value(raw_key, value)
                    if parameter == FADER_PARAMETER_ID:
                        key = self._surface_key("fader", channel_type, channel_number)
                        self._set_state_value(key, value)
                        self._notify(key, fader_value_to_db(value))
        self._notify(f"cc/{channel}/{cc}", value)

    def _handle_note(self, channel: int, note: int, velocity: int):
        self._set_state_value(f"note/{channel}/{note}", velocity)
        decoded = self._decode_selector(channel, note)
        if velocity and decoded is not None:
            channel_type, channel_number = decoded
            key = self._surface_key("mute", channel_type, channel_number)
            self._set_state_value(key, velocity >= 0x40)
            self._notify(key, velocity >= 0x40)
        self._notify(f"note/{channel}/{note}", velocity)

    def _handle_pitchbend(self, channel: int, socket_selector: int, gain_value: int):
        self._set_state_value(f"preamp/{socket_selector + 1}/gain_raw", gain_value)
        self._notify(f"preamp/{socket_selector + 1}/gain_raw", gain_value)

    def _handle_sysex(self, body: bytes):
        if not body.startswith(SYSEX_BODY_HEADER):
            return
        payload = body[len(SYSEX_BODY_HEADER) :]
        if len(payload) < 3:
            return
        midi_channel, message_type, selector = payload[:3]
        decoded = self._decode_selector(midi_channel, selector)
        if message_type == 0x02 and decoded is not None:
            channel_type, channel_number = decoded
            name = payload[3:].decode("ascii", errors="replace").rstrip("\x00").strip()
            key = self._surface_key("name", channel_type, channel_number)
            self._set_state_value(key, name)
            if channel_type == "input":
                self._channel_names[channel_number] = name
            self._notify(key, name)
        elif message_type == 0x08 and len(payload) >= 4:
            self._set_state_value(f"preamp/{selector + 1}/pad", payload[3] >= 0x40)
        elif message_type == 0x0B and len(payload) >= 4:
            self._set_state_value(f"preamp/{selector + 1}/phantom", payload[3] >= 0x40)

    def _notify(self, address: str, *args):
        for pattern, callbacks in list(self.callbacks.items()):
            if pattern in {"*", address}:
                for callback in list(callbacks):
                    try:
                        callback(address, *args)
                    except Exception as exc:
                        logger.debug("dLive callback error: %s", exc)

    def send(self, address: str, *args):
        """Reject the generic raw-byte surface; use typed methods only."""

        return False

    def subscribe(self, address: str, callback: Callable):
        self.callbacks.setdefault(address, []).append(callback)

    def get_state(self) -> Dict[str, Any]:
        return dict(self.state)

    def _midi_channel(self, channel_type: str = "input") -> int:
        return self._selector(1, channel_type)[0]

    def set_fader(self, channel: int, value_db: float, channel_type: str = "input"):
        return self._unsupported_write("fader_requires_one_shot_supervised_proof")

    def _supervised_fader_write(
        self,
        channel: int,
        value_db: float,
        *,
        authority: Any,
        phase: str,
        channel_type: str = "input",
    ) -> bool:
        """Send one authority-bound fader proof packet.

        The authority transitions before ``sendall``. A failed or interrupted
        send therefore leaves only the exact rollback phase authorized.
        """

        if channel_type != "input":
            return self._unsupported_write(f"{channel_type}_fader")
        requested = _finite_float(value_db, "fader_db")
        # Apply obeys the configured ceiling. Exact-preimage rollback is
        # recovery authority, not a new target, and must not be stranded by a
        # ceiling that was lowered after the physical preimage was observed.
        effective = (
            min(requested, 0.0)
            if phase == "rollback"
            else min(requested, self._fader_ceiling_db, 0.0)
        )
        consume = getattr(authority, "consume_fader_write", None)
        if (
            getattr(authority, "_dlive_transport_seal", None)
            is not _DLIVE_SUPERVISED_WRITE_SEAL
            or not callable(consume)
            or consume(
                client=self,
                channel=channel,
                value_db=effective,
                phase=phase,
            ) is not True
        ):
            return self._unsupported_write(
                "invalid_or_expired_supervised_fader_authority"
            )
        midi_channel, selector = self._selector(channel, channel_type)
        return self._send_raw(
            build_nrpn(midi_channel, selector, FADER_PARAMETER_ID, db_to_fader_value(effective))
        )

    def set_channel_fader(self, channel: int, value_db: float):
        return self.set_fader(channel, value_db)

    def request_fader(self, channel: int, channel_type: str = "input") -> bool:
        midi_channel, selector = self._selector(channel, channel_type)
        return self._send_raw(build_get_fader(midi_channel, selector))

    def request_mute(self, channel: int, channel_type: str = "input") -> bool:
        midi_channel, selector = self._selector(channel, channel_type)
        return self._send_raw(build_get_mute(midi_channel, selector))

    def request_channel_name(self, channel: int, channel_type: str = "input") -> bool:
        midi_channel, selector = self._selector(channel, channel_type)
        return self._send_raw(build_get_name(midi_channel, selector))

    def _query_input_state(
        self,
        channel: int,
        kind: str,
        *,
        timeout_sec: float = 0.5,
        round_trips: int = 2,
    ) -> Dict[str, Any]:
        self._selector(channel, "input")
        key = f"{kind}/{channel}"
        request = {
            "fader": self.request_fader,
            "mute": self.request_mute,
            "name": self.request_channel_name,
        }.get(kind)
        if request is None:
            return {"status": "unsupported", "key": key, "confirmed_round_trips": 0}
        values: List[Any] = []
        timestamps: List[float] = []
        for _ in range(max(1, int(round_trips))):
            before = self.state_timestamps.get(key, 0.0)
            sent_at = time.time()
            if request(channel) is False:
                return {"status": "send_failed", "key": key, "confirmed_round_trips": len(values)}
            deadline = time.monotonic() + max(0.0, float(timeout_sec))
            while time.monotonic() < deadline:
                timestamp = self.state_timestamps.get(key, 0.0)
                if timestamp > before and timestamp >= sent_at:
                    values.append(self.state.get(key))
                    timestamps.append(timestamp)
                    break
                time.sleep(0.005)
            else:
                return {"status": "unconfirmed", "key": key, "confirmed_round_trips": len(values)}
        if not values or any(value != values[0] for value in values[1:]):
            return {
                "status": "unstable",
                "key": key,
                "values": values,
                "confirmed_round_trips": len(values),
            }
        return {
            "status": "received",
            "key": key,
            "value": values[0],
            "confirmed_round_trips": len(values),
            "response_timestamps": timestamps,
        }

    def query_parameter(self, address: str, timeout_sec: float = 0.5) -> Dict[str, Any]:
        match = re.fullmatch(r"/ch/(\d+)/(fdr|mute|name)", str(address or ""))
        if match is None:
            return {
                "status": "unsupported",
                "address": address,
                "confirmed_round_trips": 0,
            }
        channel = int(match.group(1))
        kind = {"fdr": "fader", "mute": "mute", "name": "name"}[match.group(2)]
        result = self._query_input_state(
            channel,
            kind,
            timeout_sec=timeout_sec,
            round_trips=2,
        )
        value = result.get("value")
        if result.get("status") == "received" and kind == "fader":
            value = fader_value_to_db(int(value))
        return {**result, "address": address, "value": value}

    def set_fader_verified(
        self,
        channel: int,
        value_db: float,
        channel_type: str = "input",
        *,
        timeout_sec: float = 0.5,
        tolerance_db: float = 0.6,
    ) -> Dict[str, Any]:
        if channel_type != "input":
            return {"status": "unsupported", "channel": channel, "channel_type": channel_type}
        requested = _finite_float(value_db, "fader_db")
        effective = min(requested, self._fader_ceiling_db, 10.0)
        target_raw = db_to_fader_value(effective)
        if self.set_fader(channel, effective, channel_type) is False:
            return {"status": "send_failed", "channel": channel, "target_db": requested}
        readback = self._query_input_state(
            channel,
            "fader",
            timeout_sec=timeout_sec,
            round_trips=2,
        )
        if readback.get("status") != "received":
            return {
                "status": "unconfirmed",
                "channel": channel,
                "target_db": requested,
                "effective_db": effective,
                "readback": readback,
            }
        readback_raw = int(readback["value"])
        readback_db = fader_value_to_db(readback_raw)
        matched = readback_raw == target_raw or abs(readback_db - effective) <= tolerance_db
        return {
            "status": "applied" if matched else "mismatch",
            "channel": channel,
            "target_db": requested,
            "effective_db": effective,
            "target_value": target_raw,
            "readback_db": readback_db,
            "readback_value": readback_raw,
            "confirmed_round_trips": readback["confirmed_round_trips"],
        }

    def get_fader(self, channel: int) -> Optional[float]:
        self._selector(channel, "input")
        value = self.state.get(f"fader/{channel}")
        return None if value is None else fader_value_to_db(int(value))

    def set_mute(self, channel: int, muted: bool, channel_type: str = "input"):
        return self._unsupported_write("mute_mapping_not_physically_proven")

    def set_channel_mute(self, channel: int, muted: bool):
        return self.set_mute(channel, muted)

    def get_mute(self, channel: int) -> Optional[bool]:
        self._selector(channel, "input")
        value = self.state.get(f"mute/{channel}")
        return None if value is None else bool(value)

    def set_mute_verified(
        self,
        channel: int,
        muted: bool,
        *,
        timeout_sec: float = 0.5,
    ) -> Dict[str, Any]:
        target = bool(muted)
        if self.set_mute(channel, target) is False:
            return {"status": "send_failed", "channel": channel, "target_muted": target}
        readback = self._query_input_state(
            channel,
            "mute",
            timeout_sec=timeout_sec,
            round_trips=2,
        )
        if readback.get("status") != "received":
            return {
                "status": "unconfirmed",
                "channel": channel,
                "target_muted": target,
                "readback": readback,
            }
        actual = bool(readback["value"])
        return {
            "status": "applied" if actual == target else "mismatch",
            "channel": channel,
            "target_muted": target,
            "readback_muted": actual,
            "confirmed_round_trips": readback["confirmed_round_trips"],
        }

    def get_channel_name(self, channel: int) -> str:
        self._selector(channel, "input")
        return self._channel_names.get(channel) or self.state.get(f"name/{channel}") or f"Ch {channel}"

    def request_channel_names(self, num_channels: int = 48):
        if isinstance(num_channels, bool) or not isinstance(num_channels, int):
            return False
        if not 1 <= num_channels <= 128:
            return False
        return all(self.request_channel_name(channel) for channel in range(1, num_channels + 1))

    @staticmethod
    def _unsupported_write(family: str) -> bool:
        logger.warning("dLive %s write blocked: no promoted official+physical proof", family)
        return False

    def set_gain(self, channel: int, value_db: float):
        return self._unsupported_write("preamp_gain")

    def set_channel_gain(self, channel: int, value_db: float):
        return self.set_gain(channel, value_db)

    def set_phantom(self, channel: int, enabled: bool):
        return self._unsupported_write("phantom_48v")

    def set_pad(self, channel: int, enabled: bool):
        return self._unsupported_write("preamp_pad")

    def set_eq_band(self, channel: int, band: int, freq: float, gain: float, q: float):
        return self._unsupported_write("peq")

    def set_hpf(self, channel: int, freq: float, enabled: bool = True):
        return self._unsupported_write("hpf")

    def recall_scene(self, scene_number: int):
        return self._unsupported_write("scene_recall")

    def set_channel_name(self, channel: int, name: str, channel_type: str = "input"):
        return self._unsupported_write("channel_name")

    def set_channel_colour(self, channel: int, colour: str, channel_type: str = "input"):
        return self._unsupported_write("channel_colour")

    def set_pan(self, channel: int, pan: float, channel_type: str = "input"):
        return self._unsupported_write("pan")

    def set_polarity(self, channel: int, inverted: bool):
        return self._unsupported_write("polarity")

    def set_delay(self, channel: int, delay_ms: float, enabled: bool = True):
        return self._unsupported_write("delay")

    def set_compressor(self, channel: int, **kwargs):
        return self._unsupported_write("compressor")

    def set_compressor_on(self, channel: int, enabled: bool):
        return self._unsupported_write("compressor")

    def set_gate(self, channel: int, *args, **kwargs):
        return self._unsupported_write("gate")

    def set_send_level(self, channel: int, send_bus: int, level_db: float, channel_type: str = "input"):
        return self._unsupported_write("send_level")

    def reset_channel_eq(self, channel: int):
        return self._unsupported_write("peq_reset")

    def reset_channel_hpf(self, channel: int):
        return self._unsupported_write("hpf_reset")

    def reset_channel_dynamics(self, channel: int):
        return self._unsupported_write("dynamics_reset")

    def reset_channel_processing(self, channel: int):
        return self._unsupported_write("channel_processing_reset")

    def get_channel_settings(self, channel: int) -> Dict[str, Any]:
        self._selector(channel, "input")
        return {
            "channel": channel,
            "name": self.get_channel_name(channel),
            "fader_db": self.get_fader(channel),
            "muted": self.get_mute(channel),
            "fader_updated_at": self.state_timestamps.get(f"fader/{channel}"),
            "mute_updated_at": self.state_timestamps.get(f"mute/{channel}"),
            "name_updated_at": self.state_timestamps.get(f"name/{channel}"),
        }

    def find_snap_by_name(self, name: str) -> Optional[int]:
        return None
