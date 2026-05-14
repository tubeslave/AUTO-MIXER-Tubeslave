"""WING meter-backed level source for live input trim."""

from __future__ import annotations

import logging
import math
import re
import socket
import struct
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


DEFAULT_METER_PATHS = (
    "/$meters/ch/{ch}/in",
    "/meters/ch/{ch}/in",
)
DEFAULT_SUBSCRIBE_PATHS = (
    "/$meters/subscribe",
    "/meters/subscribe",
)
METER_ADDRESS_RE = re.compile(r"^/\$?meters/ch/(\d+)/in(?:/.*)?$")


@dataclass
class WingMeterReading:
    """One WING input meter reading normalized to dBFS-like values."""

    channel: int
    rms_db: float = -120.0
    peak_dbfs: float = -120.0
    true_peak_dbtp: float = -120.0
    trusted: bool = False
    confidence: float = 0.0
    age_sec: Optional[float] = None
    status: str = "unavailable"
    source: str = "wing_meter"
    address: str = ""
    raw_value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WingMeterReader:
    """Read WING channel input meters through OSC push frames or direct queries.

    The reader is deliberately conservative: if meter frames are stale or absent,
    readings are returned as untrusted so live trim cannot apply against USB-only
    levels by accident.
    """

    def __init__(self, mixer_client=None, config: Optional[Dict[str, Any]] = None):
        self.mixer_client = mixer_client
        self.config = config or {}
        self.cfg = self._get_cfg(self.config)
        self.enabled = bool(self.cfg.get("wing_meter_enabled", True))
        self.subscribe_interval_ms = int(self.cfg.get("meter_subscribe_interval_ms", 100))
        self.stale_timeout_sec = float(self.cfg.get("meter_stale_timeout_sec", 0.75))
        self.query_timeout_sec = float(self.cfg.get("meter_query_timeout_sec", 0.08))
        self.query_interval_sec = float(self.cfg.get("meter_query_interval_sec", 0.75))
        self.peak_to_rms_margin_db = float(self.cfg.get("meter_peak_to_rms_margin_db", 12.0))
        self.native_enabled = bool(self.cfg.get("native_meter_enabled", True))
        self.native_tcp_port = int(self.cfg.get("native_meter_tcp_port", 2222))
        self.native_report_id = int(self.cfg.get("native_meter_report_id", 0x414D4931))
        self.native_values_per_channel = int(self.cfg.get("native_values_per_channel", 8))
        self.native_renew_interval_sec = float(
            self.cfg.get("native_meter_renew_interval_sec", 2.0)
        )
        self.input_paths = tuple(self.cfg.get("meter_input_paths", DEFAULT_METER_PATHS))
        self.subscribe_paths = tuple(
            self.cfg.get("meter_subscribe_paths", DEFAULT_SUBSCRIBE_PATHS)
        )

        self.channels: set[int] = set()
        self._readings: Dict[int, WingMeterReading] = {}
        self._last_query_at: Dict[int, float] = {}
        self._lock = threading.Lock()
        self._active = False
        self._callback_registered = False
        self._native_channel_order: list[int] = []
        self._native_tcp: Optional[socket.socket] = None
        self._native_udp: Optional[socket.socket] = None
        self._native_thread: Optional[threading.Thread] = None
        self._native_stop = threading.Event()
        self._native_status = "stopped"
        self._native_last_packet_at: Optional[float] = None

    def start(self, channels: Iterable[int]) -> Dict[str, Any]:
        """Subscribe to WING metering and start collecting readings."""
        self.channels = {int(ch) for ch in channels}
        if not self.enabled:
            self._active = False
            return self.get_status()
        if not self.mixer_client or not getattr(self.mixer_client, "is_connected", False):
            self._active = False
            return self.get_status()

        if hasattr(self.mixer_client, "subscribe") and not self._callback_registered:
            self.mixer_client.subscribe("*", self._on_osc_message)
            self._callback_registered = True

        self._active = True
        self._start_native_metering()
        for address in self.subscribe_paths:
            try:
                self.mixer_client.send(address, self.subscribe_interval_ms)
            except Exception as exc:
                logger.debug("Could not subscribe to WING meters at %s: %s", address, exc)
        return self.get_status()

    def stop(self) -> None:
        """Stop using future meter callbacks."""
        self._active = False
        self._stop_native_metering()
        self.channels = set()

    def get_metrics(
        self,
        channels: Iterable[int],
        now: Optional[float] = None,
    ) -> Dict[int, WingMeterReading]:
        """Return current readings for channels, querying stale channels sparingly."""
        current_time = time.time() if now is None else float(now)
        requested_channels = [int(ch) for ch in channels]
        for channel in requested_channels:
            if self._should_query(channel, current_time):
                self._query_channel(channel, current_time)

        readings = {}
        with self._lock:
            for channel in requested_channels:
                reading = self._readings.get(channel)
                if not reading:
                    readings[channel] = WingMeterReading(
                        channel=channel,
                        age_sec=None,
                        status="unavailable",
                        trusted=False,
                    )
                    continue
                reading_age = self._reading_age(reading, current_time)
                trusted = reading.trusted and reading_age <= self.stale_timeout_sec
                status = "ok" if trusted else "stale"
                readings[channel] = WingMeterReading(
                    channel=channel,
                    rms_db=reading.rms_db,
                    peak_dbfs=reading.peak_dbfs,
                    true_peak_dbtp=reading.true_peak_dbtp,
                    trusted=trusted,
                    confidence=reading.confidence if trusted else 0.0,
                    age_sec=reading_age,
                    status=status,
                    source=reading.source,
                    address=reading.address,
                    raw_value=reading.raw_value,
                )
        return readings

    def get_status(self) -> Dict[str, Any]:
        """Return reader status for WebSocket/debug output."""
        now = time.time()
        with self._lock:
            fresh = [
                ch
                for ch, reading in self._readings.items()
                if self._reading_age(reading, now) <= self.stale_timeout_sec
            ]
        return {
            "enabled": self.enabled,
            "active": self._active,
            "trusted": bool(fresh),
            "fresh_channels": fresh,
            "status": "ok" if fresh else "unavailable",
            "native_status": self._native_status,
            "native_last_packet_age_sec": (
                round(now - self._native_last_packet_at, 3)
                if self._native_last_packet_at is not None
                else None
            ),
            "stale_timeout_sec": self.stale_timeout_sec,
            "subscribe_paths": list(self.subscribe_paths),
            "input_paths": list(self.input_paths),
        }

    def _on_osc_message(self, address: str, *args) -> None:
        if not self._active:
            return
        match = METER_ADDRESS_RE.match(address)
        if not match:
            return
        channel = int(match.group(1))
        if self.channels and channel not in self.channels:
            return
        reading = self._reading_from_values(channel, address, args, "wing_meter_push")
        if reading:
            self._store_reading(reading)

    def _start_native_metering(self) -> None:
        if not self.native_enabled or not self.channels:
            self._native_status = "disabled"
            return
        ip = getattr(self.mixer_client, "ip", None)
        if not ip:
            self._native_status = "missing_mixer_ip"
            return
        if self._native_thread and self._native_thread.is_alive():
            return
        self._native_channel_order = sorted(ch for ch in self.channels if 1 <= ch <= 40)
        if not self._native_channel_order:
            self._native_status = "no_channels"
            return
        try:
            self._native_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._native_udp.bind(("0.0.0.0", 0))
            self._native_udp.settimeout(0.25)

            self._native_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._native_tcp.settimeout(3.0)
            self._native_tcp.connect((ip, self.native_tcp_port))
            self._native_stop.clear()
            self._send_native_meter_request()
            self._native_thread = threading.Thread(
                target=self._native_loop,
                name="WingNativeMeterReader",
                daemon=True,
            )
            self._native_thread.start()
            self._native_status = "active"
        except Exception as exc:
            self._native_status = f"error:{exc}"
            logger.warning("Could not start WING native meters: %s", exc)
            self._close_native_sockets()

    def _stop_native_metering(self) -> None:
        self._native_stop.set()
        if self._native_thread and self._native_thread.is_alive():
            self._native_thread.join(timeout=1.0)
        self._native_thread = None
        self._close_native_sockets()
        self._native_status = "stopped"

    def _close_native_sockets(self) -> None:
        for sock_name in ("_native_tcp", "_native_udp"):
            sock = getattr(self, sock_name)
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
                setattr(self, sock_name, None)

    def _native_loop(self) -> None:
        next_renew = time.time() + self.native_renew_interval_sec
        while not self._native_stop.is_set():
            now = time.time()
            if now >= next_renew:
                try:
                    self._send_native_meter_request()
                    next_renew = now + self.native_renew_interval_sec
                except Exception as exc:
                    self._native_status = f"renew_error:{exc}"
                    logger.debug("WING native meter renew failed: %s", exc)
            try:
                if not self._native_udp:
                    break
                packet, _addr = self._native_udp.recvfrom(4096)
                if self._handle_native_packet(packet):
                    self._native_last_packet_at = time.time()
                    self._native_status = "active"
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as exc:
                self._native_status = f"read_error:{exc}"
                logger.debug("WING native meter read failed: %s", exc)
        self._native_status = "stopped"

    def _send_native_meter_request(self) -> None:
        if not self._native_tcp or not self._native_udp:
            return
        udp_port = int(self._native_udp.getsockname()[1])
        body = bytearray()
        body += bytes([0xD3]) + struct.pack(">H", udp_port)
        body += bytes([0xD4]) + struct.pack(">I", self.native_report_id & 0xFFFFFFFF)
        body += bytes([0xDC, 0xA0])
        body += bytes(max(0, min(39, channel - 1)) for channel in self._native_channel_order)
        body += bytes([0xDE])
        self._native_tcp.sendall(bytes([0xDF, 0xD3]) + _escape_native_payload(body))

    def _handle_native_packet(self, packet: bytes) -> int:
        if len(packet) < 4 or not self._native_channel_order:
            return 0
        report_id = struct.unpack(">I", packet[:4])[0]
        if report_id != (self.native_report_id & 0xFFFFFFFF):
            return 0
        value_bytes = packet[4:]
        words = len(value_bytes) // 2
        values_per_channel = self.native_values_per_channel
        expected_words = len(self._native_channel_order) * values_per_channel
        if words < expected_words:
            return 0
        count = 0
        for index, channel in enumerate(self._native_channel_order):
            start = index * values_per_channel * 2
            end = start + values_per_channel * 2
            raw_values = [
                struct.unpack(">h", value_bytes[pos:pos + 2])[0] / 256.0
                for pos in range(start, end, 2)
            ]
            input_values = raw_values[:2] or raw_values[:1]
            peak_dbfs = max(input_values) if input_values else -120.0
            self._store_reading(WingMeterReading(
                channel=channel,
                rms_db=max(-120.0, peak_dbfs - self.peak_to_rms_margin_db),
                peak_dbfs=max(-120.0, peak_dbfs),
                true_peak_dbtp=max(-120.0, peak_dbfs),
                trusted=True,
                confidence=1.0,
                age_sec=0.0,
                status="ok",
                source="wing_meter_native",
                address="native_meter/ch/input",
                raw_value=raw_values,
            ))
            count += 1
        return count

    def _should_query(self, channel: int, now: float) -> bool:
        if not self._active or not self.mixer_client:
            return False
        last_query = self._last_query_at.get(channel, 0.0)
        if now - last_query < self.query_interval_sec:
            return False
        with self._lock:
            reading = self._readings.get(channel)
        if not reading:
            return True
        return self._reading_age(reading, now) > self.stale_timeout_sec

    def _query_channel(self, channel: int, now: float) -> None:
        self._last_query_at[channel] = now
        if not hasattr(self.mixer_client, "query_parameter"):
            return
        for template in self.input_paths:
            address = str(template).format(ch=channel)
            try:
                value = self.mixer_client.query_parameter(
                    address,
                    timeout=self.query_timeout_sec,
                )
            except Exception as exc:
                logger.debug("WING meter query failed for %s: %s", address, exc)
                continue
            if value is None:
                continue
            reading = self._reading_from_values(
                channel,
                address,
                (value,),
                "wing_meter_query",
            )
            if reading:
                self._store_reading(reading)
                return

    def _store_reading(self, reading: WingMeterReading) -> None:
        reading.age_sec = time.time()
        with self._lock:
            self._readings[reading.channel] = reading

    def _reading_from_values(
        self,
        channel: int,
        address: str,
        values: Iterable[Any],
        source: str,
    ) -> Optional[WingMeterReading]:
        numeric = _extract_numeric_values(values)
        if not numeric:
            return None
        db_values = [_meter_value_to_db(value) for value in numeric]
        peak_dbfs = max(db_values)
        rms_db = min(peak_dbfs, peak_dbfs - self.peak_to_rms_margin_db)
        if len(db_values) > 1:
            rms_db = min(peak_dbfs, sum(db_values) / len(db_values))
        return WingMeterReading(
            channel=channel,
            rms_db=max(-120.0, float(rms_db)),
            peak_dbfs=max(-120.0, float(peak_dbfs)),
            true_peak_dbtp=max(-120.0, float(peak_dbfs)),
            trusted=True,
            confidence=1.0,
            age_sec=0.0,
            status="ok",
            source=source,
            address=address,
            raw_value=list(values),
        )

    @staticmethod
    def _reading_age(reading: WingMeterReading, now: float) -> float:
        if reading.age_sec is None:
            return float("inf")
        timestamp = float(reading.age_sec)
        if timestamp <= 0.0:
            return float("inf")
        return max(0.0, now - timestamp)

    @staticmethod
    def _get_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
        automation = config.get("automation", {}) if config else {}
        return dict(
            config.get("live_input_trim")
            or automation.get("live_input_trim")
            or {}
        )


def _extract_numeric_values(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            out.extend(_extract_numeric_values(value))
            continue
        if isinstance(value, str):
            match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
            if match:
                out.append(float(match.group(0)))
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _meter_value_to_db(value: float) -> float:
    value = float(value)
    if value <= -1.0:
        return max(-120.0, value)
    if 0.0 <= value <= 2.0:
        if value <= 0.0:
            return -120.0
        return max(-120.0, 20.0 * math.log10(value))
    return max(-120.0, min(24.0, value))


def _escape_native_payload(data: bytes) -> bytes:
    out = bytearray()
    pending_escape = False
    for value in data:
        if value == 0xDF:
            pending_escape = True
            continue
        if pending_escape:
            if 0xD0 <= value <= 0xDE:
                out.append(0xDE)
            pending_escape = False
        out.append(value)
    if pending_escape:
        out.append(0xDE)
    return bytes(out)
