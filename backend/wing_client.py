from pythonosc import udp_client, dispatcher, osc_server
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_message import OscMessage
from pythonosc.osc_bundle import OscBundle
import socket
import struct
import threading
import logging
import time
from typing import Callable, Dict, Any, Optional

from mixer_client_base import MixerClientBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WingClient(MixerClientBase):
    """OSC client for Behringer Wing mixer - uses correct Wing OSC address format"""

    def __init__(
        self,
        ip: str = "192.168.1.102",
        port: int = 2223,
        safety_limits: Optional[dict] = None,
    ):
        """
        Initialize Wing client.

        Args:
            ip: Wing mixer IP address
            port: Wing OSC port (default 2223)
            safety_limits: Optional dict with max_fader, max_gain (dB). Applied before hardware clamp.
        """
        self.ip = ip
        self.port = int(port) if port is not None else 2223
        self._safety_limits = safety_limits or {}

        # Per-channel meter values from Wing /meters blobs (peak dBFS, -90..0)
        self.channel_peak_db: Dict[int, float] = {}

        self.sock = None
        self.receiver_thread = None
        self._stop_receiver = False

        self.state = {}
        self.callbacks = {}

        self.is_connected = False

        # OSC throttle: track last send time per address to limit rate
        self._osc_throttle_enabled = True
        self._osc_throttle_hz = 10.0  # Default: 10 Hz = 100ms minimum interval
        self._osc_last_send_time: Dict[str, float] = {}  # address -> last send time

        logger.info(f"WingClient initialized for {ip}:{port}")

    def connect(self, timeout: float = 5.0) -> bool:
        """
        Connect to Wing mixer and validate connection

        Per Wing Remote Protocols documentation:
        1. Send 'WING?' to port 2222 to initiate connection
        2. Then use OSC on port 2223

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            # Create UDP socket for bidirectional communication
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(timeout)
            self.sock.bind(("0.0.0.0", 0))  # Bind to any available port

            local_port = self.sock.getsockname()[1]
            logger.info(f"Bound to local port {local_port}")

            # Step 1: Initiate connection by sending 'WING?' to port 2222
            # Per documentation: "Initiating a communication with WING starts with
            # sending the 5 bytes [UDP] datagram 'WING?' to port 2222"
            logger.info(f"Initiating connection to Wing at {self.ip}...")
            self.sock.sendto(b"WING?", (self.ip, 2222))

            try:
                data, addr = self.sock.recvfrom(4096)
                wing_info = data.decode("utf-8", errors="ignore")
                logger.info(f"Wing found: {wing_info}")
            except socket.timeout:
                logger.error(
                    f"Connection timeout: Wing at {self.ip} did not respond to handshake"
                )
                self.sock.close()
                return False

            # Step 2: Verify OSC is working by querying a parameter
            builder = OscMessageBuilder(address="/ch/1/fdr")
            msg = builder.build()
            self.sock.sendto(msg.dgram, (self.ip, self.port))

            try:
                data, addr = self.sock.recvfrom(4096)
                osc_msg = OscMessage(data)
                logger.info(f"OSC verified, ch1 fader: {osc_msg.params}")
            except socket.timeout:
                logger.error(f"OSC timeout: Wing not responding on port {self.port}")
                self.sock.close()
                return False

            self.is_connected = True

            # Start receiver thread
            self._stop_receiver = False
            self.receiver_thread = threading.Thread(
                target=self._receiver_loop, daemon=True
            )
            self.receiver_thread.start()

            # Subscribe to updates
            self._subscribe_to_updates()

            # Scan initial state
            self._scan_console_state()

            logger.info(f"Connected to Wing at {self.ip}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """Disconnect from Wing"""
        self._stop_receiver = True
        self.is_connected = False
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.debug(f"Error closing socket: {e}")
            self.sock = None
        logger.info("Disconnected from Wing")

    def _receiver_loop(self):
        """Background thread to receive OSC messages from Wing"""
        if not self.sock:
            return
        try:
            self.sock.settimeout(0.5)
        except Exception:
            return

        while not self._stop_receiver and self.is_connected:
            if not self.sock:
                break
            try:
                data, addr = self.sock.recvfrom(65535)
                try:
                    if data[:8] == b"#bundle\x00":
                        # OSC Bundle — iterate over contained messages
                        try:
                            bundle = OscBundle(data)
                            for item in bundle:
                                try:
                                    self._handle_message(item.address, *item.params)
                                except Exception as be:
                                    logger.debug(f"Bundle item parse error: {be}")
                        except Exception as be:
                            logger.debug(f"Bundle parse error: {be}")
                    else:
                        osc_msg = OscMessage(data)
                        self._handle_message(osc_msg.address, *osc_msg.params)
                except Exception as e:
                    logger.debug(f"Error parsing OSC message: {e}")
            except socket.timeout:
                pass
            except OSError as e:
                # Socket was closed
                if e.errno in (9, 57):  # Bad file descriptor or Socket not connected
                    break
                if self.is_connected:
                    logger.debug(f"Receiver OS error: {e}")
            except Exception as e:
                if self.is_connected:
                    logger.debug(f"Receiver error: {e}")

    def _handle_message(self, address: str, *args):
        """Handle incoming OSC message from Wing"""
        # Decode Wing meter bundles → per-channel peak dBFS
        if address.startswith("/meters/"):
            self._decode_meters(address, args)
            return

        # Log responses for name-related addresses (including $name)
        if "/name" in address or "/$name" in address:
            logger.info(f"Name response: {address} = {args}")
        else:
            logger.debug(f"Received: {address} {args}")

        # Wing returns [display_string, normalized_value, actual_value]
        # For string values like names, use the first argument
        if ("/name" in address or "/$name" in address) and args:
            # $name returns the actual name string
            self.state[address] = args[0] if args else ""
        elif len(args) >= 3:
            self.state[address] = args[2]  # Actual value
        elif len(args) == 1:
            self.state[address] = args[0]
        else:
            self.state[address] = args

        # Call registered callbacks
        if address in self.callbacks:
            for callback in self.callbacks[address]:
                try:
                    callback(address, *args)
                except Exception as e:
                    logger.error(f"Callback error for {address}: {e}")
        if "*" in self.callbacks:
            for callback in self.callbacks["*"]:
                try:
                    callback(address, *args)
                except Exception as e:
                    logger.error(f"Global callback error: {e}")

    def _decode_meters(self, address: str, args: tuple) -> None:
        """Decode Wing /meters blob into per-channel peak dBFS values.

        Wing sends /meters/1 as a blob (bytes) containing float32 values.
        Page 1 = channels 1-40 pre-fader input level (dBFS, typically -90..0).
        Values are big-endian IEEE 754 floats.
        """
        if not args:
            return
        blob = args[0]
        if not isinstance(blob, (bytes, bytearray)):
            # Some pythonosc versions unwrap blobs as lists of ints
            try:
                blob = bytes(blob)
            except Exception:
                return
        n_floats = len(blob) // 4
        if n_floats == 0:
            return
        try:
            values = struct.unpack(f">{n_floats}f", blob[:n_floats * 4])
        except struct.error:
            return

        if address == "/meters/1":
            # First 40 values → channels 1-40
            for i, val in enumerate(values[:40]):
                ch = i + 1
                # Clamp to valid dBFS range
                db = max(-90.0, min(0.0, float(val)))
                self.channel_peak_db[ch] = db
                self.state[f"/ch/{ch}/meter_peak"] = db
            logger.debug(f"Meters decoded: {len(values)} values, ch1={values[0]:.1f}dBFS")

    def set_osc_throttle(self, enabled: bool = True, hz: float = 10.0):
        """
        Configure OSC throttle to limit command rate.

        Args:
            enabled: Enable/disable throttle
            hz: Maximum frequency in Hz (default 10 Hz = 100ms minimum interval)
        """
        self._osc_throttle_enabled = enabled
        self._osc_throttle_hz = hz
        logger.info(f"OSC throttle: {'enabled' if enabled else 'disabled'}, {hz} Hz")

    def send(self, address: str, *values):
        """
        Send OSC message to Wing

        Args:
            address: OSC address (e.g., '/ch/1/fdr')
            values: Optional values to send. If empty, this is a query.
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False

        if not self.sock:
            logger.warning("Socket not available")
            self.is_connected = False
            return False

        # Throttle check: only for commands with values (not queries)
        if self._osc_throttle_enabled and values:
            current_time = time.time()
            min_interval = 1.0 / self._osc_throttle_hz

            if address in self._osc_last_send_time:
                time_since_last = current_time - self._osc_last_send_time[address]
                if time_since_last < min_interval:
                    # Skip this command - too soon
                    logger.debug(
                        f"OSC throttle: skipping {address} (last sent {time_since_last * 1000:.1f}ms ago, min {min_interval * 1000:.1f}ms)"
                    )
                    return False

            self._osc_last_send_time[address] = current_time

        try:
            builder = OscMessageBuilder(address=address)
            for v in values:
                builder.add_arg(v)
            msg = builder.build()
            self.sock.sendto(msg.dgram, (self.ip, self.port))
            logger.debug(f"Sent: {address} {values}")
            # Update local state cache when setting values (not queries)
            if values:
                if len(values) == 1:
                    self.state[address] = values[0]
                else:
                    self.state[address] = values
            return True
        except OSError as e:
            # Socket error - mark as disconnected
            if e.errno in (
                9,
                57,
                32,
            ):  # Bad file descriptor, Socket not connected, Broken pipe
                logger.warning(f"Socket error, connection lost: {e}")
                self.is_connected = False
            else:
                logger.error(f"Send OSError: {e}")
            return False
        except Exception as e:
            logger.error(f"Send failed: {e}")
            return False

    def subscribe(self, address_pattern: str, callback: Callable):
        """Subscribe to OSC address changes"""
        if address_pattern not in self.callbacks:
            self.callbacks[address_pattern] = []
        self.callbacks[address_pattern].append(callback)

    def _subscribe_to_updates(self):
        """Subscribe to Wing updates and meter bundles."""
        self.send("/xremote")
        logger.info("Subscribed to Wing updates via /xremote")

        # Start periodic renewal (also re-requests meters each cycle)
        self._start_xremote_renewal()

    def _start_xremote_renewal(self):
        """Periodically send /xremote and request meter data."""

        def renewal_loop():
            while self.is_connected and not self._stop_receiver:
                time.sleep(8)
                if self.is_connected:
                    self.send("/xremote")
                    # Request channel input meters (page 1 = ch 1-40 pre-fader)
                    self.send("/meters/1")
                    logger.debug("Renewed /xremote subscription + meter request")

        thread = threading.Thread(target=renewal_loop, daemon=True)
        thread.start()

        # Initial meter request
        self.send("/meters/1")

    def _scan_console_state(self):
        """Scan initial Wing state"""
        logger.info("Scanning Wing console state...")

        # Scan first 8 channels (basic scan)
        for ch in range(1, 9):
            self.send(f"/ch/{ch}/fdr")  # Fader
            self.send(f"/ch/{ch}/mute")  # Mute
            self.send(f"/ch/{ch}/name")  # Name
            time.sleep(0.02)

        # Scan mains
        for m in range(1, 5):
            self.send(f"/main/{m}/fdr")

        time.sleep(0.3)

        logger.info(
            f"Initial state scan complete, received {len(self.state)} parameters"
        )

        # Now scan all channels with inserts in background
        threading.Thread(
            target=self._scan_all_channels_with_inserts, daemon=True
        ).start()

    def _scan_all_channels_with_inserts(self):
        """Scan all channels (1-40) with their inserts, FX modules, and names"""
        logger.info("Starting full channel scan with inserts...")

        # First, scan all channel names (important for voice control)
        logger.info("Scanning channel names...")
        for ch in range(1, 41):
            self.send(f"/ch/{ch}/name")
            self.send(f"/ch/{ch}/fdr")
            self.send(f"/ch/{ch}/mute")
            time.sleep(0.02)

        time.sleep(0.3)  # Wait for name responses

        # Log found channel names
        found_names = []
        for ch in range(1, 41):
            name = self.state.get(f"/ch/{ch}/name")
            if name:
                found_names.append(f"Ch{ch}: {name}")
        if found_names:
            logger.info(f"Found channel names: {', '.join(found_names)}")

        # Scan all channels for inserts
        for ch in range(1, 41):
            self._scan_channel_inserts(ch)
            time.sleep(0.05)  # Small delay between channels

        # Log summary
        channels_with_inserts = self.get_all_channels_with_inserts()
        logger.info(f"Full channel scan complete. Total parameters: {len(self.state)}")
        logger.info(f"Found {len(channels_with_inserts)} channels with inserts:")
        for ch_num, ch_info in channels_with_inserts.items():
            inserts_info = []
            if ch_info["inserts"]["pre_insert"]:
                fx = ch_info["inserts"]["pre_insert"]["fx_module"]
                inserts_info.append(f"Pre: {fx['slot']} ({fx.get('model', 'N/A')})")
            if ch_info["inserts"]["post_insert"]:
                fx = ch_info["inserts"]["post_insert"]["fx_module"]
                inserts_info.append(f"Post: {fx['slot']} ({fx.get('model', 'N/A')})")
            logger.info(f"  Channel {ch_num}: {', '.join(inserts_info)}")

    def _scan_channel_inserts(self, channel: int):
        """Scan a single channel's insert information and FX modules"""
        ch = channel

        # Query insert information
        self.send(f"/ch/{ch}/preins/on")
        self.send(f"/ch/{ch}/preins/ins")
        self.send(f"/ch/{ch}/preins/$stat")
        self.send(f"/ch/{ch}/postins/on")
        self.send(f"/ch/{ch}/postins/mode")
        self.send(f"/ch/{ch}/postins/ins")
        self.send(f"/ch/{ch}/postins/$stat")
        time.sleep(0.1)  # Wait for responses

        # Check if there are inserts and scan FX modules
        preins_slot = self.state.get(f"/ch/{ch}/preins/ins")
        postins_slot = self.state.get(f"/ch/{ch}/postins/ins")

        fx_slots = []
        if preins_slot and preins_slot != "NONE":
            fx_slots.append(preins_slot)
        if postins_slot and postins_slot != "NONE":
            fx_slots.append(postins_slot)

        # Scan FX modules if present
        for fx_slot in fx_slots:
            self._scan_fx_module(fx_slot)

    def _scan_fx_module(self, fx_slot: str):
        """Scan an FX module to get its model and parameters"""
        fx_num = fx_slot.replace("FX", "") if fx_slot.startswith("FX") else fx_slot

        # Query FX model (most important)
        self.send(f"/fx/{fx_num}/mdl")
        self.send(f"/fx/{fx_num}/on")
        self.send(f"/fx/{fx_num}/mix")
        self.send(f"/fx/{fx_num}/")  # Root node
        time.sleep(0.05)

        # Query numbered parameters (WING uses numbered parameters for FX)
        # Query first 32 parameters (most FX modules have parameters 1-32)
        for i in range(1, 33):
            self.send(f"/fx/{fx_num}/{i}")
            if i % 8 == 0:  # Small delay every 8 parameters
                time.sleep(0.02)

        time.sleep(0.1)  # Final wait for responses

        fx_model = self.state.get(f"/fx/{fx_num}/mdl")
        if fx_model:
            logger.debug(f"Scanned {fx_slot}: {fx_model}")

    # ========== Channel Methods ==========

    def get_channel_fader(self, channel: int) -> Optional[float]:
        """Get channel fader value"""
        address = f"/ch/{channel}/fdr"
        return self.state.get(address)

    def set_channel_fader(self, channel: int, value: float):
        """
        Set channel fader value.

        According to Wing Remote Protocols v3.0.5 documentation:
        /ch/{ch}/fdr accepts dB values in range -144..10

        Safety limits (if set) are applied before hardware clamp.
        """
        address = f"/ch/{channel}/fdr"
        db_value = float(value)
        # Apply safety limits first (defense in depth)
        max_fader = self._safety_limits.get("max_fader")
        if max_fader is not None and db_value > max_fader:
            logger.debug(
                "set_channel_fader: safety clamp %.2f -> %s dB", db_value, max_fader
            )
            db_value = max_fader
        # Hardware range -144..10
        if db_value < -144.0:
            db_value = -144.0
        elif db_value > 10.0:
            db_value = 10.0

        logger.debug("Setting fader /ch/%d/fdr = %.2f dB", channel, db_value)
        self.send(address, db_value)

    def get_channel_mute(self, channel: int) -> Optional[int]:
        """Get channel mute state"""
        address = f"/ch/{channel}/mute"
        return self.state.get(address)

    def set_channel_mute(self, channel: int, value: int):
        """Set channel mute (0=unmuted, 1=muted)"""
        address = f"/ch/{channel}/mute"
        self.send(address, value)

    def get_channel_gain(self, channel: int) -> Optional[float]:
        """Get channel input trim/gain"""
        address = f"/ch/{channel}/in/set/trim"
        return self.state.get(address)

    def set_channel_gain(self, channel: int, value: float):
        """Set channel input trim/gain (dB, range: -18..18). Safety limits applied first."""
        address = f"/ch/{channel}/in/set/trim"
        trim_value = float(value)
        # Apply safety limits first (defense in depth)
        max_gain = self._safety_limits.get("max_gain")
        if max_gain is not None and trim_value > max_gain:
            logger.info(
                f"set_channel_gain: safety clamp {trim_value:.2f} -> {max_gain} dB"
            )
            trim_value = max_gain
        # Hardware range -18..18
        trim_value = max(-18.0, min(18.0, trim_value))
        logger.info(
            f"Setting TRIM for channel {channel} to {trim_value} dB (address: {address})"
        )
        result = self.send(address, trim_value)
        if result:
            logger.info(f"TRIM command sent successfully for channel {channel}")
        else:
            logger.error(f"Failed to send TRIM command for channel {channel}")
        return result

    def set_channel_balance(self, channel: int, value: float):
        """Set channel input balance (dB, range: -9..9)"""
        self.send(f"/ch/{channel}/in/set/bal", value)

    def set_channel_phase_invert(self, channel: int, value: int):
        """Set channel phase invert (0=normal, 1=inverted)"""
        self.send(f"/ch/{channel}/in/set/inv", value)

    def set_channel_delay(self, channel: int, value: float, mode: str = "MS"):
        """
        Set channel input delay

        Args:
            channel: Channel number (1-40)
            value: Delay value (depends on mode)
            mode: Delay mode - "M" (meters 0..150), "FT" (feet 0.5..500),
                  "MS" (milliseconds 0.5..500), "SMP" (samples 16..500)
        """
        self.send(f"/ch/{channel}/in/set/dlymode", mode)
        self.send(f"/ch/{channel}/in/set/dly", value)
        self.send(f"/ch/{channel}/in/set/dlyon", 1)

    def get_channel_pan(self, channel: int) -> Optional[float]:
        """Get channel pan"""
        address = f"/ch/{channel}/pan"
        return self.state.get(address)

    def set_channel_pan(self, channel: int, value: float):
        """Set channel pan (-100=left, 0=center, 100=right)"""
        address = f"/ch/{channel}/pan"
        self.send(address, value)

    # ========== EQ Methods ==========

    def set_eq_on(self, channel: int, on: int):
        """Set EQ on/off (0=off, 1=on)"""
        self.send(f"/ch/{channel}/eq/on", on)

    def get_eq_on(self, channel: int) -> Optional[int]:
        """Get EQ on/off state"""
        return self.state.get(f"/ch/{channel}/eq/on")

    def set_eq_band_gain(self, channel: int, band: str, gain: float):
        """
        Set EQ band gain (dB, range: -15..15)

        Args:
            channel: Channel number (1-40)
            band: Band name - "lg" (low shelf), "hg" (high shelf), "1g", "2g", "3g", "4g" (parametric bands)
            gain: Gain in dB (-15 to 15)
        """
        self.send(f"/ch/{channel}/eq/{band}", gain)

    def get_eq_band_gain(self, channel: int, band: str) -> Optional[float]:
        """Get EQ band gain"""
        return self.state.get(f"/ch/{channel}/eq/{band}")

    def set_eq_band_frequency(self, channel: int, band: str, frequency: float):
        """
        Set EQ band frequency (Hz)

        Args:
            channel: Channel number (1-40)
            band: Band name - "lf" (low shelf), "hf" (high shelf), "1f", "2f", "3f", "4f"
            frequency: Frequency in Hz (20-20000)
        """
        self.send(f"/ch/{channel}/eq/{band}", frequency)

    def get_eq_band_frequency(self, channel: int, band: str) -> Optional[float]:
        """Get EQ band frequency"""
        return self.state.get(f"/ch/{channel}/eq/{band}")

    def set_eq_band_q(self, channel: int, band: str, q: float):
        """
        Set EQ band Q (0.44-10)

        Args:
            channel: Channel number (1-40)
            band: Band name - "lq" (low shelf), "hq" (high shelf), "1q", "2q", "3q", "4q"
            q: Q value (0.44-10)
        """
        self.send(f"/ch/{channel}/eq/{band}", q)

    # ========== Compressor (Dynamics) Methods ==========

    def set_compressor_on(self, channel: int, on: int):
        """Set compressor on/off (0=off, 1=on)"""
        self.send(f"/ch/{channel}/dyn/on", on)

    def get_compressor_on(self, channel: int) -> Optional[int]:
        """Get compressor on/off state"""
        return self.state.get(f"/ch/{channel}/dyn/on")

    def set_compressor_threshold(self, channel: int, threshold: float):
        """Set compressor threshold (dB, range: -60..0)"""
        self.send(f"/ch/{channel}/dyn/thr", threshold)

    def get_compressor_threshold(self, channel: int) -> Optional[float]:
        """Get compressor threshold"""
        return self.state.get(f"/ch/{channel}/dyn/thr")

    def set_compressor_ratio(self, channel: int, ratio: str):
        """
        Set compressor ratio

        Args:
            channel: Channel number (1-40)
            ratio: Ratio string - "1.1", "1.2", "1.3", "1.5", "1.7", "2.0", "2.5", "3.0",
                   "3.5", "4.0", "5.0", "6.0", "8.0", "10", "20", "50", "100"
        """
        self.send(f"/ch/{channel}/dyn/ratio", ratio)

    def get_compressor_ratio(self, channel: int) -> Optional[str]:
        """Get compressor ratio"""
        return self.state.get(f"/ch/{channel}/dyn/ratio")

    def set_compressor_gain(self, channel: int, gain: float):
        """Set compressor make-up gain (dB, range: -6..12)"""
        self.send(f"/ch/{channel}/dyn/gain", gain)

    def get_compressor_gain(self, channel: int) -> Optional[float]:
        """Get compressor make-up gain"""
        return self.state.get(f"/ch/{channel}/dyn/gain")

    def get_compressor_gr(self, channel: int) -> Optional[float]:
        """
        Get current gain reduction (GR) from compressor meter if supported by mixer.
        Wing may expose this as read-only. Returns None if not available.
        """
        # Try common read-only meter addresses (WING protocol may use $gr or gr)
        v = self.state.get(f"/ch/{channel}/dyn/gr")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        v = self.state.get(f"/ch/{channel}/dyn/$gr")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        return None

    def set_compressor_attack(self, channel: int, attack: float):
        """Set compressor attack (ms, range: 0..120)"""
        self.send(f"/ch/{channel}/dyn/att", attack)

    def set_compressor_release(self, channel: int, release: float):
        """Set compressor release (ms, range: 4..4000)"""
        self.send(f"/ch/{channel}/dyn/rel", release)

    def set_compressor_knee(self, channel: int, knee: float):
        """Set compressor knee (0-5)"""
        self.send(f"/ch/{channel}/dyn/knee", knee)

    def set_compressor_mix(self, channel: int, mix: float):
        """Set compressor mix (%, range: 0..100)"""
        self.send(f"/ch/{channel}/dyn/mix", mix)

    def set_channel_width(self, channel: int, value: float):
        """Set channel width (%) -150..150"""
        self.send(f"/ch/{channel}/wid", value)

    def set_channel_name(self, channel: int, name: str):
        """Set channel name (max 16 chars)"""
        self.send(f"/ch/{channel}/name", name)

    def get_channel_name(self, channel: int, retries: int = 3) -> Optional[str]:
        """Get channel name with retries
        Uses /ch/X/$name (read-only) which reflects linked source or current strip value
        """
        # Try $name first (read-only, reflects linked source)
        address_dollar = f"/ch/{channel}/$name"
        address_regular = f"/ch/{channel}/name"

        for attempt in range(retries):
            # Query $name (read-only address)
            self.send(address_dollar)
            time.sleep(0.15)

            name = self.state.get(address_dollar)
            if name and name != "":
                logger.debug(
                    f"Channel {channel} $name: '{name}' (attempt {attempt + 1})"
                )
                return name

        # Fallback to regular name
        self.send(address_regular)
        time.sleep(0.15)
        name = self.state.get(address_regular)
        if name:
            return name

        logger.debug(f"Channel {channel} name not found after {retries} attempts")
        return None

    def get_all_channel_names(self, num_channels: int = 40) -> Dict[int, str]:
        """
        Get names of all channels (batch query for speed)
        Uses /ch/X/$name which reflects linked source or current strip value

        Returns:
            Dict mapping channel number to name, e.g. {1: "Kick", 2: "Snare", ...}
        """
        # First, send all $name queries at once (read-only, reflects linked source)
        for ch in range(1, num_channels + 1):
            self.send(f"/ch/{ch}/$name")
            time.sleep(0.02)  # Small delay between sends

        # Wait for all responses
        time.sleep(0.5)

        # Now collect all names from state
        names = {}
        for ch in range(1, num_channels + 1):
            # Try $name first
            name = self.state.get(f"/ch/{ch}/$name")
            if not name or name == "":
                # Fallback to regular name
                name = self.state.get(f"/ch/{ch}/name")

            if name and name != "":
                names[ch] = name
                logger.debug(f"Channel {ch} name: '{name}'")

        logger.info(f"Retrieved {len(names)} channel names")
        return names

    def set_channel_color(self, channel: int, color: int):
        """Set channel color (1-12)"""
        self.send(f"/ch/{channel}/col", color)

    # ========== Filter Methods ==========

    def set_low_cut(
        self, channel: int, enabled: int, frequency: float = None, slope: str = None
    ):
        """
        Set low cut filter

        Args:
            channel: Channel number (1-40)
            enabled: 0=off, 1=on
            frequency: Frequency in Hz (20..2000)
            slope: Slope "6", "12", "18", or "24"
        """
        self.send(f"/ch/{channel}/flt/lc", enabled)
        if frequency is not None:
            self.send(f"/ch/{channel}/flt/lcf", frequency)
        if slope is not None:
            self.send(f"/ch/{channel}/flt/lcs", slope)

    def set_high_cut(
        self, channel: int, enabled: int, frequency: float = None, slope: str = None
    ):
        """
        Set high cut filter

        Args:
            channel: Channel number (1-40)
            enabled: 0=off, 1=on
            frequency: Frequency in Hz (50..20000)
            slope: Slope "6" or "12"
        """
        self.send(f"/ch/{channel}/flt/hc", enabled)
        if frequency is not None:
            self.send(f"/ch/{channel}/flt/hcf", frequency)
        if slope is not None:
            self.send(f"/ch/{channel}/flt/hcs", slope)

    # ========== EQ Methods ==========
    # (set_eq_on already defined above)

    def set_eq_model(self, channel: int, model: str):
        """Set EQ model: STD, SOUL, E88, E84, F110, PULSAR, MACH4"""
        self.send(f"/ch/{channel}/eq/mdl", model)

    def set_eq_mix(self, channel: int, mix: float):
        """Set EQ mix (%) 0..125"""
        self.send(f"/ch/{channel}/eq/mix", mix)

    def set_eq_low_shelf(
        self,
        channel: int,
        gain: float = None,
        freq: float = None,
        q: float = None,
        eq_type: str = None,
    ):
        """Set EQ low shelf parameters"""
        if any(v is not None for v in (gain, freq, q, eq_type)):
            logger.info(
                "WING EQ OSC: ch=%s low_shelf dB=%s Hz=%s Q=%s type=%s",
                channel,
                gain,
                freq,
                q,
                eq_type,
            )
        if gain is not None:
            self.send(f"/ch/{channel}/eq/lg", gain)  # -15..15 dB
        if freq is not None:
            self.send(f"/ch/{channel}/eq/lf", freq)  # 20..2000 Hz
        if q is not None:
            self.send(f"/ch/{channel}/eq/lq", q)  # 0.44..10
        if eq_type is not None:
            self.send(f"/ch/{channel}/eq/leq", eq_type)  # PEQ, SHV

    def set_eq_band(
        self,
        channel: int,
        band: int,
        freq: float = None,
        gain: float = None,
        q: float = None,
    ):
        """
        Set EQ band parameters (bands 1-4)

        Args:
            channel: Channel number (1-40)
            band: EQ band (1-4)
            freq: Frequency in Hz (20..20000)
            gain: Gain in dB (-15..15)
            q: Q factor (0.44..10)
        """
        if band < 1 or band > 4:
            raise ValueError("Band must be 1-4")
        if any(v is not None for v in (freq, gain, q)):
            logger.info(
                "WING EQ OSC: ch=%s band=%s dB=%s Hz=%s Q=%s",
                channel,
                band,
                gain,
                freq,
                q,
            )
        if freq is not None:
            self.send(f"/ch/{channel}/eq/{band}f", freq)
        if gain is not None:
            self.send(f"/ch/{channel}/eq/{band}g", gain)
        if q is not None:
            self.send(f"/ch/{channel}/eq/{band}q", q)

    def set_eq_high_shelf(
        self,
        channel: int,
        gain: float = None,
        freq: float = None,
        q: float = None,
        eq_type: str = None,
    ):
        """Set EQ high shelf parameters"""
        if any(v is not None for v in (gain, freq, q, eq_type)):
            logger.info(
                "WING EQ OSC: ch=%s high_shelf dB=%s Hz=%s Q=%s type=%s",
                channel,
                gain,
                freq,
                q,
                eq_type,
            )
        if gain is not None:
            self.send(f"/ch/{channel}/eq/hg", gain)  # -15..15 dB
        if freq is not None:
            self.send(f"/ch/{channel}/eq/hf", freq)  # 50..20000 Hz
        if q is not None:
            self.send(f"/ch/{channel}/eq/hq", q)  # 0.44..10
        if eq_type is not None:
            self.send(f"/ch/{channel}/eq/heq", eq_type)  # SHV, PEQ

    # ========== Dynamics Methods ==========
    # (set_compressor_on already defined above)

    def set_compressor(
        self,
        channel: int,
        threshold: float = None,
        ratio: str = None,
        knee: int = None,
        attack: float = None,
        hold: float = None,
        release: float = None,
        gain: float = None,
        mix: float = None,
        det: str = None,
        env: str = None,
        auto: int = None,
    ):
        """
        Set compressor/dynamics parameters

        Args:
            channel: Channel number (1-40)
            threshold: Threshold in dB (-60..0)
            ratio: Ratio string: "1.1", "1.2", "1.3", "1.5", "1.7", "2.0", "2.5",
                   "3.0", "3.5", "4.0", "5.0", "6.0", "8.0", "10", "20", "50", "100"
            knee: Knee (0..5)
            attack: Attack time in ms (0..120)
            hold: Hold time in ms (1..200)
            release: Release time in ms (4..4000)
            gain: Make-up gain in dB (-6..12)
            mix: Mix percentage (0..100)
            det: Detection mode: "PEAK" or "RMS"
            env: Envelope: "LIN" or "LOG"
            auto: Auto switch (0/1)
        """
        base = f"/ch/{channel}/dyn"
        # Small delay between OSC commands to ensure mixer processes them
        if threshold is not None:
            self.send(f"{base}/thr", threshold)
            time.sleep(0.01)
        if ratio is not None:
            self.send(f"{base}/ratio", ratio)
            time.sleep(0.01)
        if knee is not None:
            self.send(f"{base}/knee", knee)
            time.sleep(0.01)
        if attack is not None:
            self.send(f"{base}/att", attack)
            time.sleep(0.01)
        if hold is not None:
            self.send(f"{base}/hld", hold)
            time.sleep(0.01)
        if release is not None:
            self.send(f"{base}/rel", release)
            time.sleep(0.01)
        if gain is not None:
            self.send(f"{base}/gain", gain)
            time.sleep(0.01)
        if mix is not None:
            self.send(f"{base}/mix", mix)
            time.sleep(0.01)
        if det is not None:
            self.send(f"{base}/det", det)
            time.sleep(0.01)
        if env is not None:
            self.send(f"{base}/env", env)
            time.sleep(0.01)
        if auto is not None:
            self.send(f"{base}/auto", auto)
            time.sleep(0.01)

    def set_compressor_model(self, channel: int, model: str):
        """Set compressor model: COMP, EXP, B160, B560, D241, ECL33, 9000C, SBUS,
        RED3, 76LA, LA, F670, BLISS, NSTR, WAVE, RIDE, 2250, L100, CMB"""
        self.send(f"/ch/{channel}/dyn/mdl", model)

    def set_gate_on(self, channel: int, on: int):
        """Enable/disable gate"""
        self.send(f"/ch/{channel}/gate/on", on)

    def set_gate(
        self,
        channel: int,
        threshold: float = None,
        range_db: float = None,
        attack: float = None,
        hold: float = None,
        release: float = None,
        accent: float = None,
        ratio: str = None,
    ):
        """
        Set gate parameters

        Args:
            channel: Channel number (1-40)
            threshold: Threshold in dB (-80..0)
            range_db: Range in dB (3..60)
            attack: Attack time in ms (0..120)
            hold: Hold time in ms (0..200)
            release: Release time in ms (4..4000)
            accent: Accent (0..100)
            ratio: Ratio string: "1:1.5", "1:2", "1:3", "1:4", "GATE"
        """
        base = f"/ch/{channel}/gate"
        if threshold is not None:
            self.send(f"{base}/thr", threshold)
        if range_db is not None:
            self.send(f"{base}/range", range_db)
        if attack is not None:
            self.send(f"{base}/att", attack)
        if hold is not None:
            self.send(f"{base}/hld", hold)
        if release is not None:
            self.send(f"{base}/rel", release)
        if accent is not None:
            self.send(f"{base}/acc", accent)
        if ratio is not None:
            self.send(f"{base}/ratio", ratio)

    def set_gate_model(self, channel: int, model: str):
        """Set gate model: GATE, DUCK, E88, 9000G, D241, DS902, WAVE, DEQ, WARM, 76LA, LA, RIDE, PSE, CMB"""
        self.send(f"/ch/{channel}/gate/mdl", model)

    # ========== Main/DCA Methods ==========

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

    def set_dca_mute(self, dca: int, value: int):
        """Set DCA mute (0=unmuted, 1=muted)"""
        self.send(f"/dca/{dca}/mute", value)

    # ========== Bus Methods ==========

    def set_bus_fader(self, bus: int, value: float):
        """Set bus fader value (dB, -144..10)"""
        self.send(f"/bus/{bus}/fdr", value)

    def set_bus_mute(self, bus: int, value: int):
        """Set bus mute (0=unmuted, 1=muted)"""
        self.send(f"/bus/{bus}/mute", value)

    def set_bus_pan(self, bus: int, value: float):
        """Set bus pan (-100..100)"""
        self.send(f"/bus/{bus}/pan", value)

    def set_bus_eq_on(self, bus: int, on: int):
        """Enable/disable bus EQ (0=off, 1=on)"""
        self.send(f"/bus/{bus}/eq/on", on)

    def set_bus_eq_band(
        self,
        bus: int,
        band: int,
        freq: float = None,
        gain: float = None,
        q: float = None,
    ):
        """Set bus EQ band (1-6) parameters"""
        if band < 1 or band > 6:
            raise ValueError("Bus EQ band must be 1-6")
        if freq is not None:
            self.send(f"/bus/{bus}/eq/{band}f", freq)
        if gain is not None:
            self.send(f"/bus/{bus}/eq/{band}g", gain)
        if q is not None:
            self.send(f"/bus/{bus}/eq/{band}q", q)

    # ========== Main Methods ==========

    def set_main_mute(self, main: int, value: int):
        """Set main mute (0=unmuted, 1=muted)"""
        self.send(f"/main/{main}/mute", value)

    def set_main_pan(self, main: int, value: float):
        """Set main pan (-100..100)"""
        self.send(f"/main/{main}/pan", value)

    def set_main_eq_on(self, main: int, on: int):
        """Enable/disable main EQ (0=off, 1=on)"""
        self.send(f"/main/{main}/eq/on", on)

    # ========== Send Methods ==========

    # ========== FX Module Methods ==========

    def set_fx_parameter(self, fx_slot: str, parameter: int, value):
        """
        Set an FX module parameter

        Args:
            fx_slot: FX slot (e.g., 'FX1', 'FX13')
            parameter: Parameter number (1-32)
            value: Parameter value
        """
        fx_num = fx_slot.replace("FX", "") if fx_slot.startswith("FX") else fx_slot
        self.send(f"/fx/{fx_num}/{parameter}", value)

    def set_fx_model(self, fx_slot: str, model: str):
        """
        Set FX module model

        Args:
            fx_slot: FX slot (e.g., 'FX1', 'FX13')
            model: FX model name (e.g., 'P-BASS', 'REVERB', 'PCORR')
        """
        fx_num = fx_slot.replace("FX", "") if fx_slot.startswith("FX") else fx_slot
        self.send(f"/fx/{fx_num}/mdl", model)

    def set_fx_on(self, fx_slot: str, on: int):
        """Enable/disable FX module (0=off, 1=on)"""
        fx_num = fx_slot.replace("FX", "") if fx_slot.startswith("FX") else fx_slot
        self.send(f"/fx/{fx_num}/on", on)

    def set_fx_mix(self, fx_slot: str, mix: float):
        """Set FX mix percentage (0-100)"""
        fx_num = fx_slot.replace("FX", "") if fx_slot.startswith("FX") else fx_slot
        self.send(f"/fx/{fx_num}/mix", mix)

    def get_fx_parameter(self, fx_slot: str, parameter: int):
        """Get an FX module parameter value"""
        fx_num = fx_slot.replace("FX", "") if fx_slot.startswith("FX") else fx_slot
        return self.state.get(f"/fx/{fx_num}/{parameter}")

    def set_channel_send(
        self,
        channel: int,
        send: int,
        level: float = None,
        on: int = None,
        mode: str = None,
        pan: float = None,
    ):
        """
        Set channel send parameters

        Args:
            channel: Channel number (1-40)
            send: Send number (1-16)
            level: Send level in dB (-144..10)
            on: Send on/off (0/1)
            mode: Send mode: "PRE", "POST", "GRP"
            pan: Send pan (-100..100)
        """
        base = f"/ch/{channel}/send/{send}"
        if level is not None:
            self.send(f"{base}/lvl", level)
        if on is not None:
            self.send(f"{base}/on", on)
        if mode is not None:
            self.send(f"{base}/mode", mode)
        if pan is not None:
            self.send(f"{base}/pan", pan)

    def set_channel_main_send(
        self,
        channel: int,
        main: int,
        level: float = None,
        on: int = None,
        pre: int = None,
    ):
        """
        Set channel send to main

        Args:
            channel: Channel number (1-40)
            main: Main number (1-4)
            level: Level in dB (-144..10)
            on: On/off (0/1)
            pre: Pre fader (0/1)
        """
        if level is not None:
            self.send(f"/ch/{channel}/main/{main}/lvl", level)
        if on is not None:
            self.send(f"/ch/{channel}/main/{main}/on", on)
        if pre is not None:
            self.send(f"/ch/{channel}/main/pre", pre)

    # ========== Utility Methods ==========

    def get_state(self) -> Dict[str, Any]:
        """Get copy of current mixer state"""
        return self.state.copy()

    def get_channel_inserts(self, channel: int) -> Dict[str, Any]:
        """
        Get insert information for a channel

        Returns:
            Dict with 'pre_insert' and 'post_insert' info including FX module details
        """
        ch = channel
        result = {"pre_insert": None, "post_insert": None}

        preins_on = self.state.get(f"/ch/{ch}/preins/on")
        preins_slot = self.state.get(f"/ch/{ch}/preins/ins")
        preins_stat = self.state.get(f"/ch/{ch}/preins/$stat")

        if preins_on == 1 and preins_slot and preins_slot != "NONE":
            fx_info = self._get_fx_module_info(preins_slot)
            result["pre_insert"] = {
                "on": True,
                "slot": preins_slot,
                "status": preins_stat,
                "fx_module": fx_info,
            }

        postins_on = self.state.get(f"/ch/{ch}/postins/on")
        postins_slot = self.state.get(f"/ch/{ch}/postins/ins")
        postins_mode = self.state.get(f"/ch/{ch}/postins/mode")
        postins_stat = self.state.get(f"/ch/{ch}/postins/$stat")
        postins_w = self.state.get(f"/ch/{ch}/postins/w")

        if postins_on == 1 and postins_slot and postins_slot != "NONE":
            fx_info = self._get_fx_module_info(postins_slot)
            result["post_insert"] = {
                "on": True,
                "slot": postins_slot,
                "mode": postins_mode,
                "status": postins_stat,
                "weight": postins_w,
                "fx_module": fx_info,
            }

        return result

    def _get_fx_module_info(self, fx_slot: str) -> Dict[str, Any]:
        """Get FX module information including model and parameters"""
        fx_num = fx_slot.replace("FX", "") if fx_slot.startswith("FX") else fx_slot

        fx_model = self.state.get(f"/fx/{fx_num}/mdl")
        fx_on = self.state.get(f"/fx/{fx_num}/on")
        fx_mix = self.state.get(f"/fx/{fx_num}/mix")
        fx_node = self.state.get(f"/fx/{fx_num}/")

        # Get all FX parameters
        fx_params = {}
        for key, value in self.state.items():
            if key.startswith(f"/fx/{fx_num}/") and key not in [
                f"/fx/{fx_num}/mdl",
                f"/fx/{fx_num}/on",
                f"/fx/{fx_num}/mix",
                f"/fx/{fx_num}/",
                f"/fx/{fx_num}/name",
                f"/fx/{fx_num}/type",
                f"/fx/{fx_num}/node",
            ]:
                param_name = key.replace(f"/fx/{fx_num}/", "")
                if param_name and param_name.isdigit():
                    fx_params[int(param_name)] = value

        return {
            "slot": fx_slot,
            "model": fx_model,
            "on": fx_on,
            "mix": fx_mix,
            "node_type": fx_node,
            "parameters": fx_params,
        }

    def get_all_channels_with_inserts(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about all channels (1-40) with their inserts

        Returns:
            Dict mapping channel number to channel info including inserts
        """
        channels = {}
        for ch in range(1, 41):
            inserts = self.get_channel_inserts(ch)
            if inserts["pre_insert"] or inserts["post_insert"]:
                channels[ch] = {"channel": ch, "inserts": inserts}
        return channels

    def query(self, address: str) -> Optional[Any]:
        """Query a parameter and wait for response"""
        self.send(address)
        time.sleep(0.1)
        return self.state.get(address)

    # ========== Routing Methods ==========

    def route_output(
        self,
        output_group: str,
        output_number: int,
        source_group: str,
        source_channel: int,
    ):
        """
        Маршрутизация выхода на источник

        Args:
            output_group: OUTPUT GROUP (куда посылать) - например, "MOD" для DANTE модуля, "CRD" для карт
            output_number: Номер выхода в OUTPUT GROUP (1..64)
            source_group: SOURCE GROUP (откуда брать сигнал) - например, "CRD" для Card, "PLAY" для USB Player
            source_channel: Номер канала источника из SOURCE GROUP (1..64)

        Примеры:
            # Маршрутизация DANTE выхода 1 на WLIVE PLAY канал 1:
            route_output("MOD", 1, "CRD", 1)

            # Маршрутизация CRD выхода 5 на канал пульта 10:
            route_output("CRD", 5, "CH", 10)
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False

        try:
            # Устанавливаем SOURCE GROUP
            result1 = self.send(
                f"/io/out/{output_group}/{output_number}/grp", source_group
            )
            time.sleep(0.01)

            # Устанавливаем номер канала источника
            result2 = self.send(
                f"/io/out/{output_group}/{output_number}/in", source_channel
            )
            time.sleep(0.01)

            if result1 and result2:
                logger.debug(
                    f"Routed {output_group}/{output_number} <- {source_group}/{source_channel}"
                )
                return True
            else:
                logger.warning(f"Failed to route {output_group}/{output_number}")
                return False
        except Exception as e:
            logger.error(f"Error routing {output_group}/{output_number}: {e}")
            return False

    def route_multiple_outputs(
        self,
        output_group: str,
        start_output: int,
        num_outputs: int,
        source_group: str,
        start_source_channel: int,
    ):
        """
        Маршрутизация нескольких выходов последовательно

        Args:
            output_group: OUTPUT GROUP (куда посылать) - "MOD", "CRD", и т.д.
            start_output: Начальный номер выхода
            num_outputs: Количество выходов для маршрутизации
            source_group: SOURCE GROUP (откуда брать сигнал) - "CRD", "PLAY", "CH", и т.д.
            start_source_channel: Начальный номер канала источника

        Пример:
            # Маршрутизация 24 DANTE выходов (MOD 1-24) на WLIVE PLAY каналы (CRD 1-24):
            route_multiple_outputs("MOD", 1, 24, "CRD", 1)
        """
        success_count = 0
        for i in range(num_outputs):
            output_num = start_output + i
            source_ch = start_source_channel + i
            if self.route_output(output_group, output_num, source_group, source_ch):
                success_count += 1
            time.sleep(0.01)
        return success_count

    def get_output_routing(
        self, output_group: str, output_number: int
    ) -> Optional[Dict[str, Any]]:
        """
        Получить текущую маршрутизацию выхода

        Args:
            output_group: OUTPUT GROUP - "MOD", "CRD", и т.д.
            output_number: Номер выхода

        Returns:
            Dict с 'source_group' и 'source_channel', или None если не найдено
        """
        if not self.is_connected:
            return None

        try:
            self.send(f"/io/out/{output_group}/{output_number}/grp")
            self.send(f"/io/out/{output_group}/{output_number}/in")
            time.sleep(0.1)

            source_group = self.state.get(f"/io/out/{output_group}/{output_number}/grp")
            source_channel = self.state.get(
                f"/io/out/{output_group}/{output_number}/in"
            )

            if source_group is not None:
                return {
                    "output_group": output_group,
                    "output_number": output_number,
                    "source_group": source_group,
                    "source_channel": source_channel,
                }
        except Exception as e:
            logger.error(
                f"Error getting routing for {output_group}/{output_number}: {e}"
            )

        return None

    # ========== Snapshot/Scene Methods ==========

    def get_current_show(self) -> Optional[str]:
        """
        Получить имя текущего активного Show

        Returns:
            Имя Show или None
        """
        if not self.is_connected:
            return None

        self.send("/$ctl/lib/$actshow")
        time.sleep(0.1)
        return self.state.get("/$ctl/lib/$actshow")

    def get_snap_list(self) -> Dict[int, str]:
        """
        Получить список snapshots с их индексами

        Возвращает словарь {индекс: имя_snapshot}
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return {}

        snapshots = {}

        # Получаем текущий активный snapshot
        self.send("/$ctl/lib/$active")
        self.send("/$ctl/lib/$actidx")
        time.sleep(0.2)

        current_active = self.state.get("/$ctl/lib/$active")
        current_idx = self.state.get("/$ctl/lib/$actidx")

        if current_active and current_idx is not None:
            snapshots[current_idx] = current_active

        return snapshots

    def find_snap_by_name(self, snap_name: str, max_index: int = 200) -> Optional[int]:
        """
        Найти индекс snapshot по имени БЕЗ загрузки (неразрушающий поиск).

        C-04 FIX: The original implementation sent a "GO" (load) command for
        each index to read back the name, which destructively changed the
        mixer state on every iteration during a live performance.  This
        rewrite queries snapshot names via the read-only OSC address
        ``/$ctl/lib/$name`` (set index first, then query name without GO)
        so the active scene is never altered during the search.

        Args:
            snap_name: Имя snapshot для поиска
            max_index: Максимальный индекс для проверки

        Returns:
            Индекс snapshot или None если не найден
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return None

        search_name_upper = snap_name.upper().strip()

        # Query names without loading — set index then read $name (not $active)
        for idx in range(1, max_index + 1):
            try:
                # Point the library cursor at this index (read-only operation)
                self.send("/$ctl/lib/$actionidx", idx)
                time.sleep(0.05)

                # Request the snapshot name at the current index.
                # $name is a read-only query; it does NOT load the snapshot.
                self.send("/$ctl/lib/$name")
                time.sleep(0.05)
                snap_name_at_idx = self.state.get("/$ctl/lib/$name")

                if snap_name_at_idx:
                    name_upper = str(snap_name_at_idx).upper()
                    name_clean = (
                        name_upper.replace("I:/", "").replace(".SNAP", "").strip()
                    )

                    if (
                        search_name_upper == name_upper
                        or search_name_upper == name_clean
                        or search_name_upper in name_upper
                    ):
                        logger.debug(
                            f"Found snapshot '{snap_name_at_idx}' at index {idx}"
                        )
                        return idx

            except Exception as e:
                logger.debug(f"Error checking snapshot index {idx}: {e}")

        logger.warning(f"Snapshot '{snap_name}' not found in first {max_index} slots")
        return None

    def load_snap(self, snap_name: str, max_index: int = 200) -> bool:
        """
        Загрузить snapshot/scene по имени

        Процесс:
        1. Находит индекс snapshot по имени через перебор
        2. Устанавливает индекс через /$ctl/lib/$actionidx
        3. Загружает через /$ctl/lib/$action = "GO"

        Args:
            snap_name: Имя snapshot для загрузки
            max_index: Максимальный индекс для поиска (по умолчанию 200)

        Примеры:
            # Загрузить snapshot "HULI REPA AC"
            load_snap("HULI REPA AC")
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False

        try:
            # Находим индекс snapshot по имени
            snap_index = self.find_snap_by_name(snap_name, max_index)

            if snap_index is None:
                logger.warning(f"Snapshot '{snap_name}' not found")
                return False

            # Загружаем snapshot по найденному индексу
            return self.load_snap_by_index(snap_index)

        except Exception as e:
            logger.error(f"Error loading snapshot {snap_name}: {e}")
            return False

    def load_snap_by_index(self, snap_index: int) -> bool:
        """
        Загрузить snapshot по индексу

        Args:
            snap_index: Индекс snapshot для загрузки

        Returns:
            True если команды отправлены успешно
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False

        try:
            # Шаг 1: Устанавливаем индекс snapshot
            result1 = self.send("/$ctl/lib/$actionidx", snap_index)
            time.sleep(0.1)

            if not result1:
                logger.warning(f"Failed to set snapshot index: {snap_index}")
                return False

            # Шаг 2: Отправляем команду GO для загрузки
            result2 = self.send("/$ctl/lib/$action", "GO")
            time.sleep(0.2)

            if result1 and result2:
                logger.debug(f"Loaded snapshot at index: {snap_index}")
                return True
            else:
                logger.warning(f"Failed to load snapshot at index: {snap_index}")
                return False
        except Exception as e:
            logger.error(f"Error loading snapshot at index {snap_index}: {e}")
            return False

    def save_snap(self, snap_name: str) -> bool:
        """
        Сохранить текущее состояние как snapshot/scene

        Args:
            snap_name: Имя snapshot для сохранения

        Примеры:
            # Сохранить текущее состояние как "HULI REPA AC"
            save_snap("HULI REPA AC")
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False

        try:
            # Возможные адреса для сохранения snapshot
            # Пробуем /snap/store с именем
            result = self.send("/snap/store", snap_name)
            time.sleep(0.1)

            if result:
                logger.debug(f"Saved snapshot: {snap_name}")
                return True
            else:
                logger.warning(f"Failed to save snapshot: {snap_name}")
                return False
        except Exception as e:
            logger.error(f"Error saving snapshot {snap_name}: {e}")
            return False

    # ========== Channel Input Routing Methods ==========

    def set_channel_input(self, channel: int, source_group: str, source_channel: int):
        """
        Назначить входной источник для канала (Channel Main)

        Args:
            channel: Номер канала пульта (1-40)
            source_group: SOURCE GROUP (откуда брать сигнал) - "MOD", "CRD", "CH", "AUX", "BUS", "MAIN", "MTX", "SEND", "MON", "USR", "OSC"
            source_channel: Номер канала источника из SOURCE GROUP (1-64)

        Примеры:
            # Канал 1 получает сигнал с MOD канала 1:
            set_channel_input(1, "MOD", 1)

            # Канал 10 получает сигнал с CRD канала 5:
            set_channel_input(10, "CRD", 5)
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False

        try:
            # Устанавливаем SOURCE GROUP для Main input
            result1 = self.send(f"/ch/{channel}/in/conn/grp", source_group)
            time.sleep(0.01)

            # Устанавливаем номер канала источника
            result2 = self.send(f"/ch/{channel}/in/conn/in", source_channel)
            time.sleep(0.01)

            if result1 and result2:
                logger.debug(
                    f"Channel {channel} Main input: {source_group}/{source_channel}"
                )
                return True
            else:
                logger.warning(f"Failed to set channel {channel} Main input")
                return False
        except Exception as e:
            logger.error(f"Error setting channel {channel} Main input: {e}")
            return False

    def set_channel_alt_input(
        self, channel: int, source_group: str, source_channel: int
    ):
        """
        Назначить альтернативный входной источник для канала (Channel ALT)

        Args:
            channel: Номер канала пульта (1-40)
            source_group: SOURCE GROUP (откуда брать сигнал) - "MOD", "CRD", "CH", "AUX", "BUS", "MAIN", "MTX", "SEND", "MON", "USR", "OSC"
            source_channel: Номер канала источника из SOURCE GROUP (1-64)

        Примеры:
            # Канал 1 ALT получает сигнал с CRD канала 1:
            set_channel_alt_input(1, "CRD", 1)
        """
        if not self.is_connected:
            logger.warning("Not connected to Wing")
            return False

        try:
            # Устанавливаем SOURCE GROUP для ALT input
            result1 = self.send(f"/ch/{channel}/in/conn/altgrp", source_group)
            time.sleep(0.01)

            # Устанавливаем номер канала источника
            result2 = self.send(f"/ch/{channel}/in/conn/altin", source_channel)
            time.sleep(0.01)

            if result1 and result2:
                logger.debug(
                    f"Channel {channel} ALT input: {source_group}/{source_channel}"
                )
                return True
            else:
                logger.warning(f"Failed to set channel {channel} ALT input")
                return False
        except Exception as e:
            logger.error(f"Error setting channel {channel} ALT input: {e}")
            return False

    def get_channel_input_routing(self, channel: int) -> Optional[Dict[str, Any]]:
        """
        Получить текущую маршрутизацию входов канала

        Args:
            channel: Номер канала пульта (1-40)

        Returns:
            Dict с 'main_group', 'main_channel', 'alt_group', 'alt_channel', или None если не найдено
        """
        if not self.is_connected:
            return None

        try:
            self.send(f"/ch/{channel}/in/conn/grp")
            self.send(f"/ch/{channel}/in/conn/in")
            self.send(f"/ch/{channel}/in/conn/altgrp")
            self.send(f"/ch/{channel}/in/conn/altin")
            time.sleep(0.1)

            main_group = self.state.get(f"/ch/{channel}/in/conn/grp")
            main_channel = self.state.get(f"/ch/{channel}/in/conn/in")
            alt_group = self.state.get(f"/ch/{channel}/in/conn/altgrp")
            alt_channel = self.state.get(f"/ch/{channel}/in/conn/altin")

            if main_group is not None:
                return {
                    "channel": channel,
                    "main_group": main_group,
                    "main_channel": main_channel,
                    "alt_group": alt_group,
                    "alt_channel": alt_channel,
                }
        except Exception as e:
            logger.error(f"Error getting channel {channel} input routing: {e}")

        return None

    # ── MixingAgent convenience API ─────────────────────────────
    def set_channel_fader_db(self, channel: int, value_db: float):
        """Set absolute channel fader in dB (alias for set_channel_fader)."""
        self.set_channel_fader(channel, value_db)

    def adjust_channel_fader(self, channel: int, delta_db: float):
        """Relative fader move in dB.

        If current fader value is unknown, requests it from Wing before adjusting.
        Falls back to 0 dB (unity) when state is unavailable to avoid large jumps.
        """
        cur = self.get_channel_fader(channel)
        if cur is None:
            # Try requesting current value from Wing
            self.send(f"/ch/{channel}/fdr")
            time.sleep(0.05)
            cur = self.get_channel_fader(channel)
        if cur is None:
            # Safe fallback: assume unity gain (0 dB) rather than -30 dB
            cur = 0.0
            logger.warning(f"ch{channel} fader unknown, assuming 0 dB before adjusting {delta_db:+.2f}dB")
        self.set_channel_fader(channel, float(cur) + float(delta_db))

    def set_channel_hpf(self, channel: int, frequency_hz: float, slope: str = "12"):
        """Enable high-pass (low cut) at frequency_hz."""
        self.set_low_cut(channel, 1, float(frequency_hz), slope)

    def apply_eq_profile(self, channel: int, profile: dict) -> None:
        """Apply a full EQ profile dict to a channel.

        Profile format::
            {
                "low_shelf":  {"gain": float, "freq": float, "type": "SHV"|"PEQ"},
                "bands":      [{"band": 1-4, "freq": float, "gain": float, "q": float}, ...],
                "high_shelf": {"gain": float, "freq": float, "type": "SHV"|"PEQ"},
            }
        """
        if not profile:
            return
        self.set_eq_on(channel, 1)
        ls = profile.get("low_shelf")
        if ls:
            self.set_eq_low_shelf(
                channel,
                gain=float(ls["gain"]) if "gain" in ls else None,
                freq=float(ls["freq"]) if "freq" in ls else None,
                eq_type=ls.get("type", "SHV"),
            )
        for bd in profile.get("bands", []):
            band_n = int(bd.get("band", 1))
            if 1 <= band_n <= 4:
                self.set_eq_band(
                    channel, band_n,
                    freq=float(bd["freq"]) if "freq" in bd else None,
                    gain=float(bd["gain"]) if "gain" in bd else None,
                    q=float(bd.get("q", 1.5)),
                )
        hs = profile.get("high_shelf")
        if hs:
            self.set_eq_high_shelf(
                channel,
                gain=float(hs["gain"]) if "gain" in hs else None,
                freq=float(hs["freq"]) if "freq" in hs else None,
                eq_type=hs.get("type", "SHV"),
            )

    def set_channel_compressor_full(
        self,
        channel: int,
        threshold: float = -18.0,
        ratio: float = 3.0,
        attack: float = 10.0,
        release: float = 100.0,
        knee: int = 2,
        makeup: float = 0.0,
        det: str = "RMS",
    ) -> None:
        """Enable compressor with full parameter set including makeup gain and detector."""
        self.set_compressor_on(channel, 1)
        ratio_s = self._compressor_ratio_to_string(ratio)
        self.set_compressor(
            channel,
            threshold=float(threshold),
            ratio=ratio_s,
            attack=float(attack),
            release=float(release),
            knee=int(knee),
            gain=float(makeup),
            det=det,
        )

    _COMP_RATIO_STRINGS = (
        "1.1",
        "1.2",
        "1.3",
        "1.5",
        "1.7",
        "2.0",
        "2.5",
        "3.0",
        "3.5",
        "4.0",
        "5.0",
        "6.0",
        "8.0",
        "10",
        "20",
        "50",
        "100",
    )

    @classmethod
    def _compressor_ratio_to_string(cls, ratio: float) -> str:
        try:
            r = float(ratio)
        except (TypeError, ValueError):
            return "3.0"
        best = cls._COMP_RATIO_STRINGS[0]
        best_err = 1e9
        for s in cls._COMP_RATIO_STRINGS:
            try:
                v = float(s)
            except ValueError:
                continue
            err = abs(v - r)
            if err < best_err:
                best_err = err
                best = s
        return best

    def set_channel_compressor(
        self,
        channel: int,
        threshold: float = -18.0,
        ratio: float = 3.0,
        attack: float = 10.0,
        release: float = 100.0,
        knee: Optional[float] = None,
    ):
        """Enable dynamics and set common compressor parameters (WING OSC)."""
        self.set_compressor_on(channel, 1)
        ratio_s = self._compressor_ratio_to_string(ratio)
        self.set_compressor(
            channel,
            threshold=float(threshold),
            ratio=ratio_s,
            attack=float(attack),
            release=float(release),
            knee=int(knee) if knee is not None else None,
        )

    def set_channel_deesser(
        self,
        channel: int,
        frequency: float = 6500.0,
        threshold_db: float = -20.0,
        ratio: float = 4.0,
    ):
        """De-esser — not exposed as a single OSC group on all WING builds; no-op."""
        logger.debug(
            "set_channel_deesser: no standard OSC mapping for ch=%s (freq=%s)",
            channel,
            frequency,
        )

    # ── MixerClientBase ABC bridge methods ──────────────────────
    def set_fader(self, channel: int, value_db: float):
        self.set_channel_fader(channel, value_db)

    def get_fader(self, channel: int) -> float:
        address = f"/ch/{channel}/fdr"
        val = self.state.get(address)
        return float(val) if val is not None else -144.0

    def set_mute(self, channel: int, muted: bool):
        self.set_channel_mute(channel, 1 if muted else 0)

    def get_mute(self, channel: int) -> bool:
        val = self.get_channel_mute(channel)
        return bool(val) if val is not None else False

    def set_gain(self, channel: int, value_db: float):
        self.set_channel_gain(channel, value_db)

    def recall_scene(self, scene_number: int):
        self.load_snap_by_index(scene_number)
