"""
Mixing Station Client - connects to Mixing Station app for mixer control

Mixing Station provides a unified API that works with multiple mixers:
- Behringer Wing, X32
- Midas M32
- Allen & Heath SQ, dLive
- and more

API Documentation: https://mixingstation.app/ms-docs/integrations/apis/
"""

from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_message import OscMessage
import socket
import threading
import logging
import time
import requests
from typing import Callable, Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MixingStationClient:
    """
    Client for Mixing Station app API
    
    Mixing Station provides REST and OSC APIs that work with any supported mixer.
    This client uses OSC for real-time communication.
    """
    
    # Default ports to try for Mixing Station
    DEFAULT_OSC_PORT = 8000
    DEFAULT_REST_PORT = 8080
    
    def __init__(self, host: str = "127.0.0.1", osc_port: int = 8000, rest_port: int = 8080):
        """
        Initialize Mixing Station client
        
        Args:
            host: Mixing Station host (usually localhost)
            osc_port: OSC port configured in Mixing Station
            rest_port: REST API port configured in Mixing Station
        """
        self.host = host
        self.osc_port = osc_port
        self.rest_port = rest_port
        
        self.sock = None
        self.receiver_thread = None
        self._stop_receiver = False
        self._osc_peer = None
        self._expected_osc_sources = frozenset()
        
        self.state = {}
        self.state_timestamps = {}
        self._state_events = {}
        self.callbacks = {}
        
        self.is_connected = False
        self.mixer_info = {}
        
        logger.info(f"MixingStationClient initialized for {host}:{osc_port} (REST: {rest_port})")

    def _resolve_osc_peer(self):
        """Resolve the configured UDP peer once and bind replies to its IP/port."""
        addresses = {
            (sockaddr[0], int(sockaddr[1]))
            for _family, _type, _proto, _canonname, sockaddr in socket.getaddrinfo(
                self.host,
                self.osc_port,
                family=socket.AF_INET,
                type=socket.SOCK_DGRAM,
            )
        }
        if not addresses:
            raise OSError("Mixing Station OSC peer did not resolve")
        self._expected_osc_sources = frozenset(addresses)
        self._osc_peer = sorted(addresses)[0]

    def _is_expected_osc_source(self, addr) -> bool:
        try:
            source = (str(addr[0]), int(addr[1]))
        except (IndexError, TypeError, ValueError):
            return False
        return source in self._expected_osc_sources

    def _osc_target(self):
        return self._osc_peer or (self.host, self.osc_port)

    def _process_received_datagram(self, data: bytes, addr) -> bool:
        """Accept state only from the peer resolved for this connection."""
        if not self._is_expected_osc_source(addr):
            logger.debug("Discarded OSC datagram from unexpected source %r", addr)
            return False
        try:
            osc_msg = OscMessage(data)
            self._handle_message(osc_msg.address, *osc_msg.params)
        except Exception as exc:
            logger.debug("Error parsing OSC message from expected peer: %s", exc)
            return False
        return True

    def connect(self, timeout: float = 5.0) -> bool:
        """
        Connect to Mixing Station
        
        Returns:
            True if connected successfully, False otherwise
        """
        logger.warning(
            "Mixing Station connection blocked: authenticated local IPC is required"
        )
        return False

        try:
            # Step 1: Try REST API to verify Mixing Station is running
            logger.info(f"Checking Mixing Station REST API at {self.host}:{self.rest_port}...")
            
            try:
                response = requests.get(
                    f"http://{self.host}:{self.rest_port}/api/status",
                    timeout=timeout,
                    allow_redirects=False,
                )
                if response.status_code == 200:
                    self.mixer_info = response.json()
                    logger.info(f"Mixing Station found: {self.mixer_info}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"REST API not available: {e}")
                # Continue anyway - OSC might still work
            
            # Step 2: Set up OSC socket
            self._resolve_osc_peer()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(timeout)
            self.sock.bind(('0.0.0.0', 0))
            
            local_port = self.sock.getsockname()[1]
            logger.info(f"Bound to local port {local_port}")
            
            # Step 3: Subscribe to updates (send /hi/v)
            # Note: Mixing Station may not respond to subscription, but will send updates
            builder = OscMessageBuilder(address='/hi/v')
            msg = builder.build()
            self.sock.sendto(msg.dgram, self._osc_target())
            logger.info("Sent subscription request /hi/v")
            
            # Step 4: Verify connection by querying a parameter
            builder = OscMessageBuilder(address='/con/v/ch.0.mix.lvl')
            msg = builder.build()
            self.sock.sendto(msg.dgram, self._osc_target())
            
            expected_address = '/con/v/ch.0.mix.lvl'
            deadline = time.monotonic() + timeout
            verified_message = None
            while time.monotonic() < deadline:
                self.sock.settimeout(max(0.01, deadline - time.monotonic()))
                try:
                    data, addr = self.sock.recvfrom(4096)
                except socket.timeout:
                    break
                if not self._is_expected_osc_source(addr):
                    logger.debug("Ignored connection response from %r", addr)
                    continue
                try:
                    candidate = OscMessage(data)
                except Exception as exc:
                    logger.debug("Ignored malformed connection response: %s", exc)
                    continue
                if candidate.address != expected_address or not candidate.params:
                    logger.debug(
                        "Ignored unrelated connection response %s",
                        candidate.address,
                    )
                    continue
                verified_message = candidate
                break

            if verified_message is None:
                logger.error("Mixing Station OSC not responding")
                logger.error("Please ensure:")
                logger.error("  1. Mixing Station is running")
                logger.error("  2. OSC is enabled in Settings -> API")
                logger.error(f"  3. OSC port is set to {self.osc_port}")
                self.sock.close()
                return False

            logger.info(
                "Mixing Station verified: %s = %s",
                verified_message.address,
                verified_message.params,
            )
            self._handle_message(
                verified_message.address,
                *verified_message.params,
            )
            
            self.is_connected = True
            
            # Start receiver thread
            self._stop_receiver = False
            self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
            self.receiver_thread.start()
            
            # Start subscription renewal
            self._start_subscription_renewal()
            
            # Scan initial state
            self._scan_mixer_state()
            
            logger.info(f"Connected to Mixing Station at {self.host}:{self.osc_port}")
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from Mixing Station"""
        self._stop_receiver = True
        self.is_connected = False
        if self.sock:
            self.sock.close()
        logger.info("Disconnected from Mixing Station")

    def _receiver_loop(self):
        """Background thread to receive OSC messages"""
        self.sock.settimeout(0.5)
        while not self._stop_receiver and self.is_connected:
            try:
                data, addr = self.sock.recvfrom(4096)
                self._process_received_datagram(data, addr)
            except socket.timeout:
                pass
            except Exception as e:
                if self.is_connected:
                    logger.debug(f"Receiver error: {e}")

    def _handle_message(self, address: str, *args):
        """Handle incoming OSC message from Mixing Station"""
        logger.debug(f"Received: {address} {args}")
        
        # Convert Mixing Station address to our state format
        # /con/v/ch.0.mix.lvl -> /ch/1/fdr
        state_key = self._ms_to_state_address(address)
        
        if args:
            self.state[state_key] = args[0]
            self.state_timestamps[state_key] = time.time()
            event = self._state_events.get(state_key)
            if event is not None:
                event.set()
        
        # Call registered callbacks for both the raw Mixing Station address and
        # the canonical state key used by the rest of the backend.
        for callback_address in dict.fromkeys((address, state_key)):
            if callback_address in self.callbacks:
                for callback in self.callbacks[callback_address]:
                    try:
                        callback(callback_address, *args)
                    except Exception as e:
                        logger.error(f"Callback error for {callback_address}: {e}")
        if "*" in self.callbacks:
            for callback in self.callbacks["*"]:
                try:
                    callback(state_key, *args)
                except Exception as e:
                    logger.error(f"Global callback error: {e}")

    def _ms_to_state_address(self, ms_address: str) -> str:
        """Convert Mixing Station address to Wing-style state address"""
        # /con/v/ch.0.mix.lvl -> /ch/1/fdr
        # /con/v/ch.0.mix.on -> /ch/1/mute
        if ms_address.startswith('/con/'):
            path = ms_address.split('/', 3)[-1]  # Get the data path part
            parts = path.split('.')
            
            if len(parts) >= 3 and parts[0] == 'ch':
                ch_num = int(parts[1]) + 1  # Mixing Station is 0-indexed
                
                if parts[2] == 'mix':
                    if len(parts) > 3:
                        if parts[3] == 'lvl':
                            return f'/ch/{ch_num}/fdr'
                        elif parts[3] == 'on':
                            return f'/ch/{ch_num}/mute'
                        elif parts[3] == 'pan':
                            return f'/ch/{ch_num}/pan'

                if parts[2] == 'preamp' and len(parts) > 3:
                    if parts[3] in {'gain', 'trim'}:
                        return f'/ch/{ch_num}/in/set/trim'
                elif parts[2] in {'gain', 'trim'}:
                    return f'/ch/{ch_num}/in/set/trim'
            
            elif len(parts) >= 2 and parts[0] == 'main':
                main_num = int(parts[1]) + 1
                if len(parts) > 2 and parts[2] == 'lvl':
                    return f'/main/{main_num}/fdr'
            
            elif len(parts) >= 2 and parts[0] == 'dca':
                dca_num = int(parts[1]) + 1
                if len(parts) > 2 and parts[2] == 'lvl':
                    return f'/dca/{dca_num}/fdr'
        
        return ms_address

    @staticmethod
    def _canonical_state_address(state_address: str) -> str:
        """Normalize equivalent public state keys to the canonical Wing-style key."""
        if not isinstance(state_address, str) or not state_address.startswith('/ch/'):
            return state_address
        parts = state_address.strip('/').split('/')
        if len(parts) < 3 or parts[0] != 'ch':
            return state_address
        try:
            ch_num = int(parts[1])
        except (TypeError, ValueError):
            return state_address
        if parts[2:] == ['in', 'set', 'trim']:
            return f'/ch/{ch_num}/in/set/trim'
        if parts[2:] in (['gain'], ['trim'], ['preamp', 'gain'], ['preamp', 'trim']):
            return f'/ch/{ch_num}/in/set/trim'
        return state_address

    def _state_to_ms_address(self, state_address: str) -> str:
        """Convert Wing-style state address to Mixing Station address"""
        state_address = self._canonical_state_address(state_address)
        # /ch/1/fdr -> ch.0.mix.lvl
        if state_address.startswith('/ch/'):
            parts = state_address.split('/')
            if len(parts) >= 4:
                ch_num = int(parts[2]) - 1  # Convert to 0-indexed
                param = parts[3]
                
                if param == 'fdr':
                    return f'ch.{ch_num}.mix.lvl'
                elif param == 'mute':
                    return f'ch.{ch_num}.mix.on'
                elif param == 'pan':
                    return f'ch.{ch_num}.mix.pan'
                elif parts[3:] == ['in', 'set', 'trim'] or param in {'gain', 'trim'}:
                    return f'ch.{ch_num}.preamp.gain'
        
        elif state_address.startswith('/main/'):
            parts = state_address.split('/')
            if len(parts) >= 4:
                main_num = int(parts[2]) - 1
                if parts[3] == 'fdr':
                    return f'main.{main_num}.lvl'
        
        elif state_address.startswith('/dca/'):
            parts = state_address.split('/')
            if len(parts) >= 4:
                dca_num = int(parts[2]) - 1
                if parts[3] == 'fdr':
                    return f'dca.{dca_num}.lvl'
        
        return state_address

    def send(self, address: str, *values):
        """
        Send OSC message to Mixing Station
        
        Args:
            address: Wing-style OSC address (e.g., '/ch/1/fdr')
            values: Optional values to send. If empty, this is a query.
        """
        if values:
            logger.warning(
                "Mixing Station mutation blocked: authenticated local IPC is required"
            )
            return False
        if not self.is_connected:
            logger.warning("Not connected to Mixing Station")
            return False
            
        try:
            # Convert to Mixing Station format
            ms_path = self._state_to_ms_address(address)
            
            if values:
                # Set value: /con/v/{path} f {value}
                osc_address = f'/con/v/{ms_path}'
                builder = OscMessageBuilder(address=osc_address)
                for v in values:
                    builder.add_arg(float(v))
            else:
                # Query: /con/v/{path}
                osc_address = f'/con/v/{ms_path}'
                builder = OscMessageBuilder(address=osc_address)
            
            msg = builder.build()
            self.sock.sendto(msg.dgram, self._osc_target())
            logger.debug(f"Sent: {osc_address} {values}")
            return True
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False

    def subscribe(self, address_pattern: str, callback: Callable):
        """Subscribe to address changes"""
        if address_pattern not in self.callbacks:
            self.callbacks[address_pattern] = []
        self.callbacks[address_pattern].append(callback)

    def _start_subscription_renewal(self):
        """Periodically send /hi/v to maintain subscription"""
        def renewal_loop():
            while self.is_connected and not self._stop_receiver:
                time.sleep(4)  # Mixing Station requires /hi every 5 seconds
                if self.is_connected:
                    builder = OscMessageBuilder(address='/hi/v')
                    msg = builder.build()
                    try:
                        self.sock.sendto(msg.dgram, self._osc_target())
                        logger.debug("Renewed Mixing Station subscription")
                    except:
                        pass
        
        thread = threading.Thread(target=renewal_loop, daemon=True)
        thread.start()

    def _scan_mixer_state(self):
        """Scan initial mixer state"""
        logger.info("Scanning mixer state via Mixing Station...")
        
        # Scan first 8 channels
        for ch in range(8):
            self.send(f'/ch/{ch+1}/fdr')      # Fader
            self.send(f'/ch/{ch+1}/mute')     # Mute
            self.send(f'/ch/{ch+1}/in/set/trim')  # Input trim/gain
            time.sleep(0.02)
        
        # Scan mains
        for m in range(4):
            self.send(f'/main/{m+1}/fdr')
        
        time.sleep(0.3)
        
        logger.info(f"Initial state scan complete, received {len(self.state)} parameters")

    # ========== Channel Methods (same interface as WingClient) ==========
    
    def get_channel_fader(self, channel: int) -> Optional[float]:
        """Get channel fader value"""
        address = f"/ch/{channel}/fdr"
        return self.state.get(address)

    def set_channel_fader(self, channel: int, value: float):
        """Set channel fader value"""
        return self.send(f"/ch/{channel}/fdr", value)

    def get_channel_mute(self, channel: int) -> Optional[int]:
        """Get channel mute state"""
        address = f"/ch/{channel}/mute"
        return self.state.get(address)

    def set_channel_mute(self, channel: int, value: int):
        """Set channel mute (0=unmuted, 1=muted)"""
        self.send(f"/ch/{channel}/mute", value)

    def set_channel_gain(self, channel: int, value: float):
        """Set channel input trim/gain using the canonical supervised-write key."""
        return self.send(f"/ch/{channel}/in/set/trim", value)

    def get_channel_gain(self, channel: int) -> Optional[float]:
        """Get channel input trim/gain from the canonical cache key."""
        address = f"/ch/{channel}/in/set/trim"
        return self.state.get(address)

    def get_main_fader(self, main: int = 1) -> Optional[float]:
        """Get main fader value"""
        address = f"/main/{main}/fdr"
        return self.state.get(address)

    def set_main_fader(self, main: int, value: float):
        """Set main fader value"""
        self.send(f"/main/{main}/fdr", value)

    def get_dca_fader(self, dca: int) -> Optional[float]:
        """Get DCA fader value"""
        address = f"/dca/{dca}/fdr"
        return self.state.get(address)

    def set_dca_fader(self, dca: int, value: float):
        """Set DCA fader value"""
        self.send(f"/dca/{dca}/fdr", value)

    def set_eq_band(self, channel: int, band: int, freq: float = None, gain: float = None, q: float = None):
        """Set EQ band parameters"""
        ch_idx = channel - 1
        if freq is not None:
            self._send_ms_raw(f'/con/v/ch.{ch_idx}.eq.{band}.f', freq)
        if gain is not None:
            self._send_ms_raw(f'/con/v/ch.{ch_idx}.eq.{band}.g', gain)
        if q is not None:
            self._send_ms_raw(f'/con/v/ch.{ch_idx}.eq.{band}.q', q)

    def set_compressor(self, channel: int, threshold: float = None, ratio: float = None, 
                       attack: float = None, release: float = None):
        """Set compressor parameters"""
        ch_idx = channel - 1
        if threshold is not None:
            self._send_ms_raw(f'/con/v/ch.{ch_idx}.dyn.thr', threshold)
        if ratio is not None:
            self._send_ms_raw(f'/con/v/ch.{ch_idx}.dyn.ratio', ratio)

    def _send_ms_raw(self, osc_address: str, value: float):
        """Send raw Mixing Station OSC message"""
        if not self.is_connected:
            return False
        try:
            builder = OscMessageBuilder(address=osc_address)
            builder.add_arg(float(value))
            msg = builder.build()
            self.sock.sendto(msg.dgram, self._osc_target())
            return True
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get copy of current mixer state"""
        return self.state.copy()

    def query(self, address: str) -> Optional[Any]:
        """Query a parameter and wait for response"""
        address = self._canonical_state_address(address)
        event = self._state_events.setdefault(address, threading.Event())
        event.clear()
        issued_at = time.time()
        if not self.send(address):
            return None
        if not event.wait(timeout=0.1):
            return None
        if self.state_timestamps.get(address, 0.0) < issued_at:
            return None
        return self.state.get(address)

    def query_parameter(self, address: str) -> Dict[str, Any]:
        """Query one parameter and return the server's readback-compatible shape."""
        canonical_address = self._canonical_state_address(address)
        value = self.query(canonical_address)
        return {
            "status": "received" if value is not None else "readback_unavailable",
            "address": canonical_address,
            "value": value,
            "source": "mixing_station_query",
        }


# Discovery function to find Mixing Station
def discover_mixing_station(timeout: float = 2.0) -> Optional[Dict[str, int]]:
    """
    Try to discover Mixing Station on common ports
    
    Returns:
        Dict with 'osc_port' and 'rest_port' if found, None otherwise
    """
    common_ports = [8000, 8080, 9000, 10000]
    
    for port in common_ports:
        try:
            # Try REST API
            response = requests.get(f"http://127.0.0.1:{port}/api/status", timeout=timeout)
            if response.status_code == 200:
                logger.info(f"Found Mixing Station REST API on port {port}")
                return {'rest_port': port, 'osc_port': port}
        except:
            pass
        
        # Try OSC
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(timeout)
            sock.bind(('0.0.0.0', 0))
            
            builder = OscMessageBuilder(address='/hi/v')
            msg = builder.build()
            sock.sendto(msg.dgram, ('127.0.0.1', port))
            
            data, addr = sock.recvfrom(4096)
            sock.close()

            if (str(addr[0]), int(addr[1])) != ('127.0.0.1', port):
                continue
            OscMessage(data)
            
            logger.info(f"Found Mixing Station OSC on port {port}")
            return {'osc_port': port, 'rest_port': port}
        except:
            pass
    
    return None
