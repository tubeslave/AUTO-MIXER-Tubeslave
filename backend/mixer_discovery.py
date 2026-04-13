"""
Mixer auto-discovery via network scanning.

Supports:
- Behringer WING: UDP broadcast 'WING?' to port 2222
- Allen & Heath dLive: TCP probe to port 51328 / 51329
- Subnet scanning for unknown IPs

Usage:
    from mixer_discovery import discover_mixers, discover_mixer_auto
    results = discover_mixers()  # scan all known protocols
"""

import logging
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

WING_DISCOVERY_PORT = 2222
WING_OSC_PORT = 2223
DLIVE_TCP_PORT = 51328
DLIVE_TLS_PORT = 51329
SCAN_TIMEOUT = 1.5
BROADCAST_TIMEOUT = 2.0
MAX_SUBNET_SCAN_WORKERS = 64


@dataclass
class DiscoveredMixer:
    """A discovered mixer on the network."""
    mixer_type: str         # "wing" or "dlive"
    ip: str
    port: int
    name: str = ""
    firmware: str = ""
    model: str = ""
    tls: bool = False
    discovery_method: str = ""  # "broadcast", "tcp_probe", "subnet_scan"
    response_time_ms: float = 0.0

    def __repr__(self):
        tls_str = " (TLS)" if self.tls else ""
        return (
            f"<{self.mixer_type.upper()} @ {self.ip}:{self.port}{tls_str} "
            f"'{self.name}' [{self.discovery_method}] {self.response_time_ms:.0f}ms>"
        )


def _get_local_ip() -> str:
    """Get the local IP address used for outbound connections."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "192.168.1.1"


def _get_subnet_prefix(ip: str) -> str:
    """Extract subnet prefix from IP (e.g. '192.168.3' from '192.168.3.100')."""
    parts = ip.split(".")
    if len(parts) == 4:
        return ".".join(parts[:3])
    return "192.168.1"


def _get_all_interfaces() -> List[str]:
    """Get IP addresses of all active network interfaces."""
    ips = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    try:
        import netifaces
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            for addr in addrs.get(netifaces.AF_INET, []):
                ip = addr.get("addr", "")
                if ip and not ip.startswith("127."):
                    ips.add(ip)
    except ImportError:
        pass
    if not ips:
        ips.add(_get_local_ip())
    return list(ips)


# ── WING Discovery (UDP broadcast) ──────────────────────────────

def discover_wing_broadcast(
    timeout: float = BROADCAST_TIMEOUT,
) -> List[DiscoveredMixer]:
    """Discover Behringer WING mixers via UDP broadcast on port 2222.

    Sends 'WING?' as a broadcast and collects responses.
    """
    results: List[DiscoveredMixer] = []
    seen_ips: set = set()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(timeout)
        sock.bind(("0.0.0.0", 0))

        logger.info("Sending WING? broadcast on port 2222...")
        sock.sendto(b"WING?", ("255.255.255.255", WING_DISCOVERY_PORT))

        for local_ip in _get_all_interfaces():
            subnet = _get_subnet_prefix(local_ip)
            try:
                sock.sendto(
                    b"WING?",
                    (f"{subnet}.255", WING_DISCOVERY_PORT),
                )
            except Exception:
                pass

        start = time.time()
        while time.time() - start < timeout:
            try:
                data, addr = sock.recvfrom(4096)
                ip = addr[0]
                if ip in seen_ips:
                    continue
                seen_ips.add(ip)

                info = data.decode("utf-8", errors="ignore").strip()
                resp_ms = (time.time() - start) * 1000

                name = ""
                model = ""
                firmware = ""
                parts = info.split("\n")
                for p in parts:
                    p = p.strip()
                    if p and not name:
                        name = p
                    if "wing" in p.lower():
                        model = p
                    if "." in p and any(c.isdigit() for c in p):
                        firmware = p

                mixer = DiscoveredMixer(
                    mixer_type="wing",
                    ip=ip,
                    port=WING_OSC_PORT,
                    name=name or f"WING @ {ip}",
                    model=model,
                    firmware=firmware,
                    discovery_method="broadcast",
                    response_time_ms=resp_ms,
                )
                results.append(mixer)
                logger.info(f"WING discovered: {mixer}")

            except socket.timeout:
                break
            except Exception as e:
                logger.debug(f"WING broadcast recv error: {e}")
                break

        sock.close()
    except Exception as e:
        logger.error(f"WING broadcast discovery error: {e}")

    return results


def _probe_wing_unicast(ip: str, timeout: float = SCAN_TIMEOUT) -> Optional[DiscoveredMixer]:
    """Probe a specific IP for a WING mixer."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        sock.bind(("0.0.0.0", 0))

        start = time.time()
        sock.sendto(b"WING?", (ip, WING_DISCOVERY_PORT))

        data, addr = sock.recvfrom(4096)
        resp_ms = (time.time() - start) * 1000
        sock.close()

        info = data.decode("utf-8", errors="ignore").strip()
        return DiscoveredMixer(
            mixer_type="wing",
            ip=ip,
            port=WING_OSC_PORT,
            name=info.split("\n")[0] if info else f"WING @ {ip}",
            discovery_method="unicast_probe",
            response_time_ms=resp_ms,
        )
    except Exception:
        return None


# ── dLive Discovery (TCP probe) ─────────────────────────────────

def _probe_dlive_tcp(
    ip: str, port: int = DLIVE_TCP_PORT, timeout: float = SCAN_TIMEOUT
) -> Optional[DiscoveredMixer]:
    """Probe a specific IP:port for a dLive mixer via TCP connect."""
    try:
        start = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((ip, port))
        resp_ms = (time.time() - start) * 1000

        if result == 0:
            sock.close()
            tls = port == DLIVE_TLS_PORT
            mixer = DiscoveredMixer(
                mixer_type="dlive",
                ip=ip,
                port=port,
                name=f"dLive @ {ip}",
                tls=tls,
                discovery_method="tcp_probe",
                response_time_ms=resp_ms,
            )
            logger.info(f"dLive discovered: {mixer}")
            return mixer
        sock.close()
    except Exception:
        pass
    return None


def discover_dlive_subnet(
    subnet: Optional[str] = None,
    timeout: float = SCAN_TIMEOUT,
    max_workers: int = MAX_SUBNET_SCAN_WORKERS,
) -> List[DiscoveredMixer]:
    """Scan a /24 subnet for dLive mixers on TCP port 51328 and 51329."""
    results: List[DiscoveredMixer] = []

    if subnet is None:
        subnets = set()
        for ip in _get_all_interfaces():
            subnets.add(_get_subnet_prefix(ip))
        if not subnets:
            subnets.add("192.168.1")
    else:
        subnets = {subnet}

    targets = []
    for sub in subnets:
        for host in range(1, 255):
            ip = f"{sub}.{host}"
            targets.append((ip, DLIVE_TCP_PORT))
            targets.append((ip, DLIVE_TLS_PORT))

    logger.info(f"Scanning {len(targets)} targets for dLive (subnets: {subnets})...")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_probe_dlive_tcp, ip, port, timeout): (ip, port)
            for ip, port in targets
        }
        for future in as_completed(futures):
            mixer = future.result()
            if mixer:
                results.append(mixer)

    return results


def discover_dlive_known_ips(
    ips: Optional[List[str]] = None,
    timeout: float = SCAN_TIMEOUT,
) -> List[DiscoveredMixer]:
    """Try connecting to known/common dLive IPs."""
    if ips is None:
        base_ips = set()
        for local_ip in _get_all_interfaces():
            subnet = _get_subnet_prefix(local_ip)
            base_ips.add(f"{subnet}.1")
            base_ips.add(f"{subnet}.70")
            base_ips.add(f"{subnet}.100")
        base_ips.update([
            "192.168.1.70",
            "192.168.3.70",
            "192.168.0.70",
            "10.0.0.70",
        ])
        ips = list(base_ips)

    results: List[DiscoveredMixer] = []
    with ThreadPoolExecutor(max_workers=min(len(ips) * 2, 32)) as pool:
        futures = []
        for ip in ips:
            futures.append(pool.submit(_probe_dlive_tcp, ip, DLIVE_TCP_PORT, timeout))
            futures.append(pool.submit(_probe_dlive_tcp, ip, DLIVE_TLS_PORT, timeout))
        for future in as_completed(futures):
            mixer = future.result()
            if mixer:
                results.append(mixer)

    return results


# ── Combined discovery ───────────────────────────────────────────

def discover_mixers(
    scan_subnet: bool = False,
    subnet: Optional[str] = None,
    timeout: float = SCAN_TIMEOUT,
) -> List[DiscoveredMixer]:
    """Discover all mixers on the network.

    Steps:
    1. WING broadcast (fast — ~2s)
    2. dLive TCP probe to known IPs (fast — ~1.5s)
    3. Optionally: full subnet scan for dLive (slower — ~5-10s)

    Returns list of DiscoveredMixer sorted by response time.
    """
    all_results: List[DiscoveredMixer] = []
    seen = set()

    wing_thread = threading.Thread(
        target=lambda: all_results.extend(discover_wing_broadcast(timeout)),
        daemon=True,
    )
    dlive_thread = threading.Thread(
        target=lambda: all_results.extend(discover_dlive_known_ips(timeout=timeout)),
        daemon=True,
    )

    wing_thread.start()
    dlive_thread.start()
    wing_thread.join(timeout=timeout + 1)
    dlive_thread.join(timeout=timeout + 1)

    if scan_subnet:
        subnet_results = discover_dlive_subnet(subnet=subnet, timeout=timeout)
        all_results.extend(subnet_results)

    unique = []
    for m in all_results:
        key = (m.mixer_type, m.ip, m.port)
        if key not in seen:
            seen.add(key)
            unique.append(m)

    unique.sort(key=lambda m: m.response_time_ms)
    return unique


def discover_mixer_auto(
    preferred_type: Optional[str] = None,
    preferred_ip: Optional[str] = None,
    scan_subnet: bool = True,
    timeout: float = SCAN_TIMEOUT,
) -> Optional[DiscoveredMixer]:
    """Auto-discover and return the best mixer.

    Priority:
    1. Preferred IP if specified and responding
    2. Preferred type if specified
    3. First responding mixer

    Args:
        preferred_type: "wing" or "dlive" — prefer this type
        preferred_ip: Try this IP first
        scan_subnet: Do full subnet scan if quick probes fail
        timeout: Per-probe timeout in seconds

    Returns:
        DiscoveredMixer or None if nothing found.
    """
    logger.info(
        f"Auto-discovering mixer (preferred_type={preferred_type}, "
        f"preferred_ip={preferred_ip})..."
    )

    if preferred_ip:
        for port in [DLIVE_TCP_PORT, DLIVE_TLS_PORT]:
            mixer = _probe_dlive_tcp(preferred_ip, port, timeout)
            if mixer:
                logger.info(f"Found preferred dLive at {preferred_ip}:{port}")
                return mixer
        wing = _probe_wing_unicast(preferred_ip, timeout)
        if wing:
            logger.info(f"Found preferred WING at {preferred_ip}")
            return wing

    results = discover_mixers(scan_subnet=scan_subnet, timeout=timeout)

    if not results:
        logger.warning("No mixers discovered on the network")
        return None

    logger.info(f"Discovered {len(results)} mixer(s): "
                + ", ".join(str(m) for m in results))

    if preferred_type:
        typed = [m for m in results if m.mixer_type == preferred_type]
        if typed:
            return typed[0]

    return results[0]


# ── CLI ──────────────────────────────────────────────────────────

def main():
    """CLI tool for mixer discovery."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AUTO-MIXER — Mixer Network Discovery"
    )
    parser.add_argument("--subnet", default=None,
                        help="Subnet to scan (e.g. 192.168.3)")
    parser.add_argument("--full-scan", action="store_true",
                        help="Full /24 subnet scan for dLive")
    parser.add_argument("--timeout", type=float, default=2.0,
                        help="Probe timeout in seconds")
    parser.add_argument("--prefer", default=None, choices=["wing", "dlive"],
                        help="Prefer this mixer type")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("  AUTO-MIXER — Mixer Network Discovery")
    print("=" * 60)

    local_ips = _get_all_interfaces()
    print(f"\nLocal interfaces: {', '.join(local_ips)}")
    print(f"Scanning...\n")

    start = time.time()
    results = discover_mixers(
        scan_subnet=args.full_scan,
        subnet=args.subnet,
        timeout=args.timeout,
    )
    elapsed = time.time() - start

    if results:
        print(f"Found {len(results)} mixer(s) in {elapsed:.1f}s:\n")
        for i, m in enumerate(results, 1):
            tls_str = " [TLS]" if m.tls else ""
            print(f"  {i}. {m.mixer_type.upper()} @ {m.ip}:{m.port}{tls_str}")
            print(f"     Name: {m.name}")
            print(f"     Method: {m.discovery_method}")
            print(f"     Response: {m.response_time_ms:.0f}ms")
            print()
    else:
        print(f"No mixers found ({elapsed:.1f}s)")
        print("\nTips:")
        print("  - Check that the mixer is on the same network")
        print("  - Try --full-scan for a full subnet scan")
        print("  - For dLive, ensure TCP MIDI is enabled (port 51328)")
        print("  - For WING, check that UDP port 2222 is reachable")


if __name__ == "__main__":
    main()
