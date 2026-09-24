"""Read-only native WING Main meter ingress for PATCH_VERIFY.

The WING remote protocol exposes realtime meters on native communication channel
3 rather than as ordinary OSC parameter queries.  A client declares a UDP
receive port over the native TCP connection, requests a meter collection with a
4-byte report id, and receives signed big-endian 16-bit level words at roughly
50 ms cadence for a limited subscription window.

This module implements only the bounded one-shot primitive needed by
``PATCH_VERIFY``: request one Main strip, receive one matching frame, decode its
output L/R meters, and convert them into ``PhysicalMainMeterEvidence``.  It does
not write mixer state, change routing, keep a long-lived meter subscription, or
make musical decisions.

Protocol sources validated during the renovation:
- Behringer / Music Tribe WING Remote Protocols 3.1, Channel 3: Metering;
- the independent ``libwing`` implementation, whose ``request_meter`` path uses
  the same TCP channel selection, UDP return-port declaration and meter tokens.

Physical HIL is still required before this provider may authorize autonomous
Main mutations.  In particular, the exact detector/ballistics relationship
between the console output meter and a sample-domain peak remains a calibration
item for the level-coherence gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
import time
from typing import Callable, Protocol

from .patch_verify import PhysicalMainMeterEvidence


WING_NATIVE_PORT = 2222
METER_CHANNEL_SELECT = bytes((0xDF, 0xD3))
CONTROL_CHANNEL_SELECT = bytes((0xDF, 0xD1))
METER_PORT_TOKEN = 0xD3
METER_REPORT_ID_TOKEN = 0xD4
METER_COLLECTION_START = 0xDC
METER_MAIN_TOKEN = 0xA3
METER_COLLECTION_END = 0xDE
DEFAULT_REPORT_ID = 0x4D41494E  # ASCII "MAIN"; deliberately contains no 0xDF byte.
MAIN_METER_WORDS = 8
LEVEL_DB_PER_LSB = 1.0 / 256.0


class _UdpSocketLike(Protocol):
    def bind(self, address: tuple[str, int]) -> None: ...
    def getsockname(self) -> tuple[str, int]: ...
    def settimeout(self, timeout: float) -> None: ...
    def recvfrom(self, size: int) -> tuple[bytes, tuple[str, int]]: ...
    def close(self) -> None: ...


class _TcpSocketLike(Protocol):
    def settimeout(self, timeout: float) -> None: ...
    def sendall(self, data: bytes) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class WingMainMeterFrame:
    """Decoded legacy Main meter collection (8 level/state words)."""

    report_id: int
    input_left_dbfs: float
    input_right_dbfs: float
    output_left_dbfs: float
    output_right_dbfs: float
    gate_key_db: float
    gate_gain_raw: int
    dyn_key_db: float
    dyn_gain_raw: int

    @property
    def output_level_dbfs(self) -> float:
        """Return the louder post-strip output meter channel in dBFS."""

        return max(self.output_left_dbfs, self.output_right_dbfs)


class WingMainMeterReadError(RuntimeError):
    """Fail-closed error raised when independent Main evidence cannot be proved."""


def _escape_native_value(data: bytes) -> bytes:
    """Escape literal 0xDF bytes inside native-protocol values.

    0xDF is the WING native escape/channel byte.  Literal data bytes equal to
    0xDF are represented as 0xDF 0xDE.  Control tokens passed separately by the
    packet builder are intentionally not escaped.
    """

    out = bytearray()
    for byte in data:
        out.append(byte)
        if byte == 0xDF:
            out.append(0xDE)
    return bytes(out)


def build_main_meter_request(
    *,
    udp_port: int,
    main: int = 1,
    report_id: int = DEFAULT_REPORT_ID,
) -> bytes:
    """Build one native meter subscription request for ``MAIN 1..4``.

    WING meter collection indices are encoded zero-based on the wire, so
    ``MAIN 1`` is token pair ``0xA3 0x00``.
    """

    if isinstance(udp_port, bool) or not isinstance(udp_port, int):
        raise TypeError("udp_port must be an integer")
    if not 1 <= udp_port <= 65535:
        raise ValueError("udp_port must be in 1..65535")
    if isinstance(main, bool) or not isinstance(main, int):
        raise TypeError("main must be an integer")
    if not 1 <= main <= 4:
        raise ValueError("main must be in 1..4")
    if isinstance(report_id, bool) or not isinstance(report_id, int):
        raise TypeError("report_id must be an integer")
    if not 0 <= report_id <= 0xFFFFFFFF:
        raise ValueError("report_id must fit uint32")

    port_bytes = _escape_native_value(struct.pack(">H", udp_port))
    report_bytes = _escape_native_value(struct.pack(">I", report_id))
    return b"".join(
        (
            METER_CHANNEL_SELECT,
            bytes((METER_PORT_TOKEN,)),
            port_bytes,
            bytes((METER_REPORT_ID_TOKEN,)),
            report_bytes,
            bytes(
                (
                    METER_COLLECTION_START,
                    METER_MAIN_TOKEN,
                    main - 1,
                    METER_COLLECTION_END,
                )
            ),
            CONTROL_CHANNEL_SELECT,
        )
    )


def decode_main_meter_datagram(
    datagram: bytes,
    *,
    expected_report_id: int,
) -> WingMainMeterFrame:
    """Decode one exact Main meter datagram.

    The meter UDP payload is not the native TCP byte stream.  It begins with the
    caller's raw 4-byte report id followed by eight signed big-endian 16-bit
    words for the legacy Main collection.
    """

    if not isinstance(datagram, (bytes, bytearray, memoryview)):
        raise TypeError("datagram must be bytes-like")
    if isinstance(expected_report_id, bool) or not isinstance(expected_report_id, int):
        raise TypeError("expected_report_id must be an integer")
    if not 0 <= expected_report_id <= 0xFFFFFFFF:
        raise ValueError("expected_report_id must fit uint32")

    raw = bytes(datagram)
    expected_size = 4 + MAIN_METER_WORDS * 2
    if len(raw) != expected_size:
        raise ValueError(
            f"Main meter datagram must be exactly {expected_size} bytes; got {len(raw)}"
        )

    report_id = struct.unpack(">I", raw[:4])[0]
    if report_id != expected_report_id:
        raise ValueError(
            f"Main meter report id mismatch: expected 0x{expected_report_id:08X}, "
            f"got 0x{report_id:08X}"
        )

    words = struct.unpack(">8h", raw[4:])
    return WingMainMeterFrame(
        report_id=report_id,
        input_left_dbfs=words[0] * LEVEL_DB_PER_LSB,
        input_right_dbfs=words[1] * LEVEL_DB_PER_LSB,
        output_left_dbfs=words[2] * LEVEL_DB_PER_LSB,
        output_right_dbfs=words[3] * LEVEL_DB_PER_LSB,
        gate_key_db=words[4] * LEVEL_DB_PER_LSB,
        gate_gain_raw=words[5],
        dyn_key_db=words[6] * LEVEL_DB_PER_LSB,
        dyn_gain_raw=words[7],
    )


class WingNativeMainMeterProvider:
    """One-shot independent physical Main meter reader.

    Each call opens a short-lived native TCP connection to port 2222, binds a
    private UDP return socket, subscribes to exactly one Main meter collection,
    receives the first matching report and closes both sockets.  That bounded
    lifecycle is deliberate for startup/PATCH_VERIFY and avoids creating a
    second persistent control authority beside the existing OSC transport.
    """

    def __init__(
        self,
        host: str,
        *,
        main: int = 1,
        timeout_s: float = 0.35,
        report_id: int = DEFAULT_REPORT_ID,
        time_fn: Callable[[], float] = time.monotonic,
        udp_socket_factory: Callable[[], _UdpSocketLike] | None = None,
        tcp_connection_factory: Callable[[tuple[str, int], float], _TcpSocketLike] | None = None,
    ) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be a non-empty string")
        if isinstance(main, bool) or not isinstance(main, int) or not 1 <= main <= 4:
            raise ValueError("main must be an integer in 1..4")
        if isinstance(timeout_s, bool) or float(timeout_s) <= 0.0:
            raise ValueError("timeout_s must be > 0")
        if isinstance(report_id, bool) or not isinstance(report_id, int):
            raise TypeError("report_id must be an integer")
        if not 0 <= report_id <= 0xFFFFFFFF:
            raise ValueError("report_id must fit uint32")

        self.host = host.strip()
        self.main = main
        self.timeout_s = float(timeout_s)
        self.report_id = report_id
        self._time_fn = time_fn
        self._udp_socket_factory = udp_socket_factory or self._default_udp_socket
        self._tcp_connection_factory = tcp_connection_factory or self._default_tcp_connection

    @staticmethod
    def _default_udp_socket() -> _UdpSocketLike:
        return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    @staticmethod
    def _default_tcp_connection(address: tuple[str, int], timeout: float) -> _TcpSocketLike:
        return socket.create_connection(address, timeout=timeout)

    def read(self) -> PhysicalMainMeterEvidence:
        udp: _UdpSocketLike | None = None
        tcp: _TcpSocketLike | None = None
        try:
            udp = self._udp_socket_factory()
            udp.bind(("0.0.0.0", 0))
            udp.settimeout(self.timeout_s)
            local = udp.getsockname()
            if len(local) < 2:
                raise WingMainMeterReadError("UDP socket did not expose a local port")
            udp_port = int(local[1])

            tcp = self._tcp_connection_factory((self.host, WING_NATIVE_PORT), self.timeout_s)
            tcp.settimeout(self.timeout_s)
            # Match the validated native connection sequence used by libwing:
            # select the ordinary control channel once, then switch to meter
            # request channel in the request packet itself.
            tcp.sendall(CONTROL_CHANNEL_SELECT)
            tcp.sendall(
                build_main_meter_request(
                    udp_port=udp_port,
                    main=self.main,
                    report_id=self.report_id,
                )
            )

            deadline = self._time_fn() + self.timeout_s
            last_protocol_error: Exception | None = None
            while self._time_fn() <= deadline:
                try:
                    datagram, _source = udp.recvfrom(4096)
                except (TimeoutError, socket.timeout) as exc:
                    raise WingMainMeterReadError("timed out waiting for WING Main meter data") from exc
                try:
                    frame = decode_main_meter_datagram(
                        datagram,
                        expected_report_id=self.report_id,
                    )
                except (TypeError, ValueError) as exc:
                    last_protocol_error = exc
                    continue

                return PhysicalMainMeterEvidence(
                    peak_dbfs=frame.output_level_dbfs,
                    rms_dbfs=None,
                    timestamp_s=float(self._time_fn()),
                    source=f"wing-native-main-meter:MAIN.{self.main}",
                )

            detail = f": {last_protocol_error}" if last_protocol_error is not None else ""
            raise WingMainMeterReadError(f"no matching WING Main meter report before timeout{detail}")
        except WingMainMeterReadError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise WingMainMeterReadError(
                f"failed to read WING MAIN.{self.main} meter: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if tcp is not None:
                try:
                    tcp.close()
                except OSError:
                    pass
            if udp is not None:
                try:
                    udp.close()
                except OSError:
                    pass

    def __call__(self) -> PhysicalMainMeterEvidence:
        return self.read()
