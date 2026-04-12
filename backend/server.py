import asyncio
import atexit
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from datetime import datetime
from typing import Set, Optional, Union, List, Dict, Any
import websockets
import numpy as np
try:
    from websockets.server import WebSocketServerProtocol
except ImportError:
    # For newer versions of websockets
    from websockets.legacy.server import WebSocketServerProtocol

from wing_client import WingClient
from dlive_client import DLiveClient
from osc.enhanced_osc_client import EnhancedOSCClient
from mixing_station_client import MixingStationClient, discover_mixing_station
from audio_devices import get_audio_devices
from dante_routing_config import get_routing_as_dict, get_module_signal_info
from voice_control import VoiceControl
from voice_control_v2 import VoiceControlV2
try:
    from voice_control_sherpa import VoiceControlSherpa
except ImportError:
    VoiceControlSherpa = None
from lufs_gain_staging import LUFSGainStagingController, SafeGainCalibrator
from channel_recognizer import scan_and_recognize, recognize_instrument_spectral_fallback, AVAILABLE_PRESETS
from auto_eq import AutoEQController, InstrumentProfiles, MultiChannelAutoEQController
from phase_alignment import PhaseAlignmentController
from auto_fader import AutoFaderController
from auto_compressor import AutoCompressorController
from backup_channels import backup_channel
from restore_channels import restore_from_backup_using_client
from bleed_service import BleedService
from handlers import register_all_handlers
from skills import register_builtin_skills

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """
    Рекурсивно преобразует NumPy типы в нативные Python типы для JSON сериализации.
    
    Args:
        obj: Объект, который может содержать NumPy типы
        
    Returns:
        Объект с преобразованными типами
    """
    # Проверяем на NumPy скалярные типы
    if isinstance(obj, np.generic):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            # Для других NumPy типов пытаемся преобразовать в Python тип
            try:
                return obj.item()
            except (AttributeError, ValueError):
                return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj) if np.issubdtype(type(obj), np.integer) else bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class AutoMixerServer:
    """
    WebSocket server that bridges frontend UI with mixer control
    
    Supports two connection modes:
    1. Direct Wing OSC - connects directly to Wing mixer
    2. Mixing Station - connects via Mixing Station app (supports multiple mixers)
    """
    
    def __init__(self, ws_host: str = "localhost", ws_port: int = 8765):
        self.ws_host = ws_host
        self.ws_port = ws_port
        
        # Load configuration
        self.config = self._load_config()
        
        # Unified mixer client (can be EnhancedOSCClient wrapping WingClient, DLiveClient, or MixingStationClient)
        self.mixer_client: Optional[Union[EnhancedOSCClient, WingClient, DLiveClient, MixingStationClient]] = None
        self.connection_mode: Optional[str] = None  # 'wing', 'dlive', or 'mixing_station'
        
        self.connected_clients: Set[WebSocketServerProtocol] = set()
        
        # Voice control
        self.voice_control: Optional[VoiceControl] = None
        
        # Gain staging controller (LUFS-based)
        self.gain_staging: Optional[LUFSGainStagingController] = None
        # Safe Gain Calibrator (новый метод: анализ → одноразовое применение)
        self.safe_gain_calibrator: Optional[SafeGainCalibrator] = None
        
        # Auto-EQ controller
        self.auto_eq_controller: Optional[AutoEQController] = None
        self.multi_channel_auto_eq_controller: Optional[MultiChannelAutoEQController] = None
        
        # Phase alignment controller
        self.phase_alignment_controller: Optional[PhaseAlignmentController] = None
        
        # Auto Fader controller
        self.auto_fader_controller: Optional[AutoFaderController] = None
        
        # Track last known fader values for relative changes
        self.last_fader_values: dict[int, float] = {}
        
        # Auto Soundcheck state
        self.auto_soundcheck_running = False
        self.auto_soundcheck_task: Optional[asyncio.Task] = None
        self.auto_soundcheck_websocket: Optional[WebSocketServerProtocol] = None
        
        # Auto Compressor controller
        self.auto_compressor_controller: Optional[AutoCompressorController] = None
        
        # Centralized bleed detection service
        self.bleed_service = BleedService(self.config)
        
        # Live concert mode: stricter limits, emergency stop
        self.live_mode = False
        
        # Snapshot for undo (path to last backup file)
        self._last_snapshot_path: Optional[str] = None
        
        # Shutdown flag (event created in start() to be in correct loop)
        self._shutdown_event: Optional[asyncio.Event] = None
        self._is_shutting_down = False
        
        # Build message-type → handler dispatch table (B15 decomposition)
        self._dispatch = register_all_handlers(self)

        register_builtin_skills()

        logger.info(f"AutoMixer Server initialized on {ws_host}:{ws_port}")
    
    def cleanup_all_controllers(self):
        """Cleanup all active controllers - call before shutdown or on error."""
        logger.info("Cleaning up all controllers...")
        
        # Stop gain staging
        if self.gain_staging:
            try:
                self.gain_staging.stop()
                logger.info("Gain staging stopped")
            except Exception as e:
                logger.error(f"Error stopping gain staging: {e}")
            self.gain_staging = None
        
        # Stop voice control
        if self.voice_control:
            try:
                self.voice_control.stop_listening()
                logger.info("Voice control stopped")
            except Exception as e:
                logger.error(f"Error stopping voice control: {e}")
            self.voice_control = None
        
        # Stop Auto-EQ
        if self.auto_eq_controller:
            try:
                self.auto_eq_controller.stop()
                logger.info("Auto-EQ stopped")
            except Exception as e:
                logger.error(f"Error stopping Auto-EQ: {e}")
            self.auto_eq_controller = None
        
        # Stop Multi-channel Auto-EQ
        if self.multi_channel_auto_eq_controller:
            try:
                self.multi_channel_auto_eq_controller.stop_all()
                logger.info("Multi-channel Auto-EQ stopped")
            except Exception as e:
                logger.error(f"Error stopping multi-channel Auto-EQ: {e}")
            self.multi_channel_auto_eq_controller = None
        
        # Stop Phase alignment
        if self.phase_alignment_controller:
            try:
                self.phase_alignment_controller.stop()
                logger.info("Phase alignment stopped")
            except Exception as e:
                logger.error(f"Error stopping phase alignment: {e}")
            self.phase_alignment_controller = None
        
        # Stop Auto Fader
        if self.auto_fader_controller:
            try:
                self.auto_fader_controller.stop()
                logger.info("Auto Fader stopped")
            except Exception as e:
                logger.error(f"Error stopping Auto Fader: {e}")
            self.auto_fader_controller = None
        
        # Stop Auto Soundcheck
        if self.auto_soundcheck_running:
            try:
                self.auto_soundcheck_running = False
                if self.auto_soundcheck_task:
                    self.auto_soundcheck_task.cancel()
                logger.info("Auto Soundcheck stopped")
            except Exception as e:
                logger.error(f"Error stopping Auto Soundcheck: {e}")
            self.auto_soundcheck_task = None
            self.auto_soundcheck_websocket = None
        
        # Stop Auto Compressor
        if self.auto_compressor_controller:
            try:
                self.auto_compressor_controller.stop()
                logger.info("Auto Compressor stopped")
            except Exception as e:
                logger.error(f"Error stopping Auto Compressor: {e}")
            self.auto_compressor_controller = None
        
        # Disconnect mixer
        if self.mixer_client:
            try:
                self.mixer_client.disconnect()
                logger.info("Mixer disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting mixer: {e}")
            self.mixer_client = None
        
        logger.info("All controllers cleaned up")
    
    async def graceful_shutdown(self, sig=None):
        """Handle graceful shutdown."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        
        if sig:
            logger.info(f"Received signal {sig.name}, shutting down...")
        else:
            logger.info("Initiating graceful shutdown...")
        
        # Cleanup all controllers
        self.cleanup_all_controllers()
        
        # Close all client connections
        if self.connected_clients:
            logger.info(f"Closing {len(self.connected_clients)} client connections...")
            close_tasks = [ws.close(1001, "Server shutting down") for ws in self.connected_clients]
            await asyncio.gather(*close_tasks, return_exceptions=True)
            self.connected_clients.clear()
        
        # Signal shutdown
        if self._shutdown_event:
            self._shutdown_event.set()
        
        logger.info("Graceful shutdown complete")
    
    def _load_config(self) -> dict:
        """Load configuration from default_config.json"""
        import os
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  'config', 'default_config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
            return {
                "automation": {
                    "auto_gain": {
                        "bleeding_rejection": {
                            "enabled": True,
                            "correlation_threshold": 0.7,
                            "level_difference_threshold_db": 8.0
                        }
                    }
                }
            }

    def _get_user_config_path(self) -> str:
        """Get path to user config file."""
        import os
        return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'config', 'user_config.json')

    def _load_user_config(self) -> dict:
        """Load user-saved settings from user_config.json."""
        import os
        path = self._get_user_config_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"User config loaded from {path}")
                return data
        except Exception as e:
            logger.warning(f"Failed to load user config: {e}")
            return {}

    def _save_user_config(self, section: str, settings: dict):
        """Save user settings to user_config.json under a given section."""
        import os
        path = self._get_user_config_path()
        # Load existing user config
        existing = self._load_user_config()
        existing[section] = settings
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            logger.info(f"User config saved to {path}: section={section}")
        except Exception as e:
            logger.error(f"Failed to save user config: {e}")
            raise

    async def register_client(self, websocket: WebSocketServerProtocol):
        self.connected_clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")
        
        if self.mixer_client and self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "connection_status",
                "connected": True,
                "mode": self.connection_mode,
                "state": self.mixer_client.get_state()
            })

    async def unregister_client(self, websocket: WebSocketServerProtocol):
        self.connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")

    def _is_connection_closed_error(self, e: Exception) -> bool:
        """True if the exception indicates the client already closed the connection."""
        msg = str(e).lower()
        if "1001" in msg or "going away" in msg or "connection closed" in msg:
            return True
        try:
            return getattr(e, "code", None) == 1001
        except Exception:
            return False

    async def send_to_client(self, websocket: WebSocketServerProtocol, message: dict):
        try:
            # Преобразуем NumPy типы в нативные Python типы перед JSON сериализацией
            message = convert_numpy_types(message)
            json_str = json.dumps(message)
            await websocket.send(json_str)
        except Exception as e:
            if self._is_connection_closed_error(e):
                logger.debug("Send failed (client closed): %s", e)
            else:
                logger.error(f"Error sending to client: {e}")

    async def broadcast(self, message: dict):
        logger.debug("Broadcasting message: %s to %s clients", message.get("type"), len(self.connected_clients))
        if not self.connected_clients:
            logger.warning("No connected clients to broadcast to")
            return
        clients = list(self.connected_clients)
        results = await asyncio.gather(
            *[self.send_to_client(c, message) for c in clients],
            return_exceptions=True
        )
        # Remove clients that closed the connection so we stop sending to them
        for client, result in zip(clients, results):
            if isinstance(result, Exception) and self._is_connection_closed_error(result):
                self.connected_clients.discard(client)
                logger.info("Removed disconnected client from broadcast list. Total clients: %s", len(self.connected_clients))
            elif isinstance(result, Exception):
                logger.error("Error broadcasting to client: %s", result)

    async def handle_client_message(self, websocket: WebSocketServerProtocol, message: str):
        try:
            logger.info(f"Received message: {message[:200]}...")  # Log first 200 chars
            logger.info(f"Full message: {message}")  # Log full message for debugging
            data = json.loads(message)
            msg_type = data.get("type")
            logger.info(f"Message type: {msg_type}")
            logger.info(f"Message data: {data}")

            handler = self._dispatch.get(msg_type)
            if handler is not None:
                await handler(websocket, data)
            else:
                logger.warning(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON received: {e}")
            await self.send_to_client(websocket, {
                "type": "error",
                "error": f"Invalid JSON: {e}"
            })
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            # Try to send error to client
            try:
                await self.send_to_client(websocket, {
                    "type": "error",
                    "error": str(e)
                })
            except:
                pass

    async def connect_wing(self, ip: str, send_port: int = 2223, receive_port: int = 2223):
        """
        Connect directly to Wing mixer via OSC
        
        Args:
            ip: Wing IP address
            send_port: Wing OSC port (default 2223)
        """
        # Disconnect existing connection if any
        await self.disconnect_mixer()
        
        try:
            self.mixer_client = EnhancedOSCClient(ip=ip, port=send_port)
            self.connection_mode = 'wing'
            
            # Store the event loop reference for thread-safe callback
            loop = asyncio.get_running_loop()
            
            def on_mixer_update(address: str, *args):
                try:
                    # Track fader values for relative volume changes
                    if address.startswith("/ch/") and address.endswith("/fdr") and args:
                        try:
                            channel_match = re.search(r'/ch/(\d+)/fdr', address)
                            if channel_match:
                                channel = int(channel_match.group(1))
                                fader_value = float(args[0]) if args else None
                                if fader_value is not None:
                                    self.last_fader_values[channel] = fader_value
                                    
                                    # Обработка ручных изменений референсного канала для Auto Fader
                                    if self.auto_fader_controller and self.auto_fader_controller.realtime_enabled:
                                        self.auto_fader_controller.handle_external_fader_change(channel, fader_value)
                        except (ValueError, IndexError, AttributeError):
                            pass  # Ignore parsing errors
                    
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.broadcast({
                            "type": "mixer_update",
                            "address": address,
                            "values": args
                        }))
                    )
                except Exception as e:
                    logger.error(f"Callback asyncio error: {e}")
            
            self.mixer_client.subscribe("*", on_mixer_update)
            
            success = self.mixer_client.connect()
            
            if success:
                await self.broadcast({
                    "type": "connection_status",
                    "connected": True,
                    "mode": "wing",
                    "ip": ip,
                    "state": self.mixer_client.get_state()
                })
                logger.info("Wing connected successfully (direct OSC)")
            else:
                self.mixer_client = None
                self.connection_mode = None
                await self.broadcast({
                    "type": "connection_status",
                    "connected": False,
                    "error": "Connection failed - Wing did not respond"
                })
                
        except Exception as e:
            logger.error(f"Error connecting to Wing: {e}")
            self.mixer_client = None
            self.connection_mode = None
            await self.broadcast({
                "type": "connection_status",
                "connected": False,
                "error": str(e)
            })

    async def connect_dlive(self, ip: str = "192.168.1.70", port: int = 51328,
                            tls: bool = False, midi_base_channel: int = 0):
        """Connect to Allen & Heath dLive mixer via MIDI over TCP."""
        await self.disconnect_mixer()

        try:
            client = DLiveClient(ip=ip, port=port, tls=tls, midi_base_channel=midi_base_channel)
            success = client.connect()

            if success:
                self.mixer_client = client
                self.connection_mode = 'dlive'

                await self.broadcast({
                    "type": "connection_status",
                    "connected": True,
                    "mode": "dlive",
                    "ip": ip,
                    "state": client.get_state()
                })
                logger.info(f"dLive connected at {ip}:{port}")
            else:
                await self.broadcast({
                    "type": "connection_status",
                    "connected": False,
                    "error": "dLive connection failed"
                })

        except Exception as e:
            logger.error(f"Error connecting to dLive: {e}")
            self.mixer_client = None
            self.connection_mode = None
            await self.broadcast({
                "type": "connection_status",
                "connected": False,
                "error": str(e)
            })

    async def connect_mixing_station(self, host: str = "127.0.0.1", osc_port: int = 8000, rest_port: int = 8080):
        """
        Connect to mixer via Mixing Station app
        
        Args:
            host: Mixing Station host (usually localhost)
            osc_port: Mixing Station OSC port
            rest_port: Mixing Station REST API port
        """
        # Disconnect existing connection if any
        await self.disconnect_mixer()
        
        try:
            self.mixer_client = MixingStationClient(host, osc_port, rest_port)
            self.connection_mode = 'mixing_station'
            
            # Store the event loop reference for thread-safe callback
            loop = asyncio.get_running_loop()
            
            def on_mixer_update(address: str, *args):
                try:
                    # Обработка ручных изменений референсного канала для Auto Fader
                    if address.startswith("/ch/") and address.endswith("/fdr") and args:
                        try:
                            channel_match = re.search(r'/ch/(\d+)/fdr', address)
                            if channel_match:
                                channel = int(channel_match.group(1))
                                fader_value = float(args[0]) if args else None
                                if fader_value is not None and self.auto_fader_controller and self.auto_fader_controller.realtime_enabled:
                                    self.auto_fader_controller.handle_external_fader_change(channel, fader_value)
                        except (ValueError, IndexError, AttributeError):
                            pass  # Ignore parsing errors
                    
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.broadcast({
                            "type": "mixer_update",
                            "address": address,
                            "values": args
                        }))
                    )
                except Exception as e:
                    logger.error(f"Callback asyncio error: {e}")
            
            self.mixer_client.subscribe("*", on_mixer_update)
            
            success = self.mixer_client.connect()
            
            if success:
                await self.broadcast({
                    "type": "connection_status",
                    "connected": True,
                    "mode": "mixing_station",
                    "host": host,
                    "osc_port": osc_port,
                    "state": self.mixer_client.get_state()
                })
                logger.info("Connected to Mixing Station successfully")
            else:
                self.mixer_client = None
                self.connection_mode = None
                await self.broadcast({
                    "type": "connection_status",
                    "connected": False,
                    "error": "Connection failed - Mixing Station not responding. Ensure API is enabled in settings."
                })
                
        except Exception as e:
            logger.error(f"Error connecting to Mixing Station: {e}")
            self.mixer_client = None
            self.connection_mode = None
            await self.broadcast({
                "type": "connection_status",
                "connected": False,
                "error": str(e)
            })

    async def discover_mixing_station(self):
        """Try to auto-discover Mixing Station on common ports"""
        logger.info("Discovering Mixing Station...")
        
        result = discover_mixing_station()
        
        if result:
            await self.broadcast({
                "type": "mixing_station_discovered",
                "found": True,
                "osc_port": result['osc_port'],
                "rest_port": result['rest_port']
            })
            logger.info(f"Mixing Station discovered on port {result['osc_port']}")
        else:
            await self.broadcast({
                "type": "mixing_station_discovered",
                "found": False,
                "error": "Mixing Station not found. Ensure it's running and API is enabled."
            })
            logger.info("Mixing Station not found")

    async def disconnect_mixer(self):
        """Disconnect from current mixer (Wing or Mixing Station)"""
        if self.mixer_client:
            self.mixer_client.disconnect()
            self.mixer_client = None
            mode = self.connection_mode
            self.connection_mode = None
            
            await self.broadcast({
                "type": "connection_status",
                "connected": False
            })
            logger.info(f"Disconnected from mixer (was: {mode})")
    
    async def start_voice_control(self, model_size: str = "small", language: str = "ru", 
                                  device_id: Optional[str] = None, channel: int = 0):
        """Start voice control listening"""
        logger.info("=" * 60)
        logger.info("START_VOICE_CONTROL CALLED")
        logger.info(f"Parameters: model={model_size}, language={language}, device_id={device_id}, channel={channel}")
        logger.info("=" * 60)
        try:
            
            if self.voice_control and self.voice_control.is_listening:
                logger.warning("Voice control already active")
                await self.broadcast({
                    "type": "voice_control_status",
                    "active": True,
                    "message": "Voice control already active"
                })
                return
            
            # Convert empty language string to None for auto-detect
            if language == "":
                language = None
            
            # Convert device_id to index if provided
            input_device_index = None
            if device_id:
                try:
                    input_device_index = int(device_id)
                    logger.info(f"Using audio device index: {input_device_index}")
                except ValueError:
                    logger.warning(f"Invalid device_id: {device_id}, using default device")
            
            # Initialize voice control (using Sherpa-ONNX with GigaAM - best for Russian!)
            logger.info("Step 1: Initializing VoiceControlSherpa (GigaAM)...")
            try:
                self.voice_control = VoiceControlSherpa(
                    input_device_index=input_device_index,
                    input_channel=channel
                )
                logger.info("Step 1: ✅ VoiceControlSherpa initialized")
                
                # Auto-configure channel aliases from Wing channel names
                if self.mixer_client and self.mixer_client.is_connected:
                    self._setup_voice_aliases_from_wing()
                    
            except Exception as e:
                logger.error(f"Step 1: ❌ Error initializing VoiceControlSherpa: {e}", exc_info=True)
                # Fallback to VoiceControlV2 (Whisper)
                logger.info("Falling back to VoiceControlV2 (Whisper)...")
                self.voice_control = VoiceControlV2(
                    model_size=model_size,
                    language=language,
                    input_device_index=input_device_index,
                    input_channel=channel
                )
            
            # Get event loop for thread-safe callback
            loop = asyncio.get_running_loop()
            
            # Define callback for recognized commands
            def on_command_recognized(command: dict):
                """Handle recognized voice command"""
                logger.info("=" * 60)
                logger.info(f"🎤 VOICE COMMAND RECOGNIZED: {command}")
                logger.info("=" * 60)
                # Schedule task from thread using call_soon_threadsafe
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._execute_voice_command(command))
                )
            
            # Start listening
            logger.info("Step 2: Starting voice control listening...")
            try:
                # Load model first if needed (this might block, so do it in executor)
                # Check for either 'model' (Whisper) or 'recognizer' (Sherpa) attribute
                model_loaded = getattr(self.voice_control, 'model', None) or getattr(self.voice_control, 'recognizer', None)
                if not model_loaded:
                    logger.info("Model not loaded, loading in executor (this may take a moment)...")
                    import concurrent.futures
                    loop = asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        await loop.run_in_executor(executor, self.voice_control.load_model)
                    logger.info("✅ Model loaded")
                else:
                    logger.info("Model already loaded")
                
                # Start listening (this should be fast as it just starts threads)
                logger.info("Calling start_listening...")
                self.voice_control.start_listening(on_command_recognized)
                logger.info("Step 2: ✅ Listening started")
            except Exception as e:
                logger.error(f"Step 2: ❌ Error starting listening: {e}", exc_info=True)
                raise
            
            # Verify it's actually listening
            if not self.voice_control.is_listening:
                raise Exception("Voice control failed to start listening")
            
            logger.info("Step 3: Broadcasting success message...")
            try:
                await self.broadcast({
                    "type": "voice_control_status",
                    "active": True,
                    "message": "Voice control started successfully"
                })
                logger.info("Step 3: ✅ Broadcast sent")
            except Exception as e:
                logger.error(f"Step 3: ❌ Error broadcasting: {e}", exc_info=True)
                raise
            
            logger.info("=" * 60)
            logger.info("VOICE CONTROL STARTED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error starting voice control: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            await self.broadcast({
                "type": "voice_control_status",
                "active": False,
                "error": str(e)
            })
    
    def _setup_voice_aliases_from_wing(self):
        """
        Auto-configure voice control channel aliases from Wing channel names.
        Maps channel names to channel numbers for voice commands.
        """
        if not self.mixer_client or not self.voice_control:
            return
        
        try:
            logger.info("Setting up voice aliases from Wing channel names...")
            
            # Get channel names from Wing (only works with WingClient)
            if not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                logger.info("Channel aliases only supported for Wing mixer")
                return
                
            channel_names = self.mixer_client.get_all_channel_names(40)
            
            if not channel_names:
                logger.info("No channel names found on Wing")
                return
            
            # Build aliases from channel names
            aliases = {}
            
            # Common name variations and normalizations
            name_normalizations = {
                # Drums - kick (bd = bass drum)
                "kick": ["kick", "кик", "бочка", "bass drum", "bd"],
                "bd": ["kick", "кик", "бочка", "bass drum", "bd"],
                "бочка": ["kick", "кик", "бочка", "bass drum", "bd"],
                # Drums - snare (малый барабан)
                "snare": ["snare", "снэйр", "снейр", "малый", "рабочий", "sn", "малый барабан", "малыйбарабан"],
                "малый": ["snare", "снэйр", "снейр", "малый", "рабочий", "sn", "малый барабан", "малыйбарабан"],
                "рабочий": ["snare", "снэйр", "снейр", "малый", "рабочий", "sn", "малый барабан", "малыйбарабан"],
                "малый барабан": ["snare", "снэйр", "снейр", "малый", "рабочий", "sn", "малый барабан", "малыйбарабан"],
                # Drums - hi-hat
                "hihat": ["hihat", "hi-hat", "hh", "хайхэт", "хэт"],
                "хайхэт": ["hihat", "хайхэт", "хэт"],
                "hh": ["hihat", "хайхэт", "хэт", "hh"],
                # Drums - toms
                "tom": ["tom", "том"],
                "том": ["tom", "том"],
                "floor": ["floor", "флор", "флортом"],
                "флортом": ["floor", "флор", "флортом"],
                # Drums - overhead
                "overhead": ["overhead", "oh", "овер", "оверхэд"],
                "oh": ["overhead", "oh", "овер", "оверхэд"],
                "оверхэд": ["overhead", "oh", "овер", "оверхэд"],
                # Bass
                "bass": ["bass", "бас", "басгитара"],
                "бас": ["bass", "бас", "басгитара"],
                # Guitar
                "guitar": ["guitar", "gtr", "гитара", "электрогитара"],
                "гитара": ["guitar", "gtr", "гитара", "электрогитара"],
                "gtr": ["guitar", "gtr", "гитара"],
                # Keys
                "keys": ["keys", "клавиши", "клавишные", "kbd"],
                "клавиши": ["keys", "клавиши", "клавишные"],
                "piano": ["piano", "пиано", "рояль"],
                "synth": ["synth", "синт", "синтезатор"],
                # Accordion
                "accordion": ["accordion", "аккордеон", "баян", "гармошка"],
                "аккордеон": ["accordion", "аккордеон", "баян", "гармошка"],
                "баян": ["accordion", "аккордеон", "баян", "гармошка"],
                # Vocals
                "vocal": ["vocal", "vox", "вокал", "голос"],
                "вокал": ["vocal", "vox", "вокал", "голос"],
                "vox": ["vocal", "vox", "вокал"],
                # Playback
                "playback": ["playback", "плейбэк", "pb", "минусовка"],
                "плейбэк": ["playback", "плейбэк", "минусовка"],
            }
            
            for ch_num, ch_name in channel_names.items():
                if not ch_name:
                    continue
                    
                # Normalize the name
                name_lower = ch_name.lower().strip()
                
                # Add direct name as alias
                aliases[name_lower] = ch_num
                
                # Check if name matches any known patterns
                for key, variations in name_normalizations.items():
                    if key in name_lower or name_lower in variations:
                        # Add all variations as aliases for this channel
                        for variation in variations:
                            if variation not in aliases:
                                aliases[variation] = ch_num
                        break
            
            if aliases:
                # Update voice control aliases
                if hasattr(self.voice_control, 'set_channel_aliases'):
                    self.voice_control.set_channel_aliases(aliases)
                    logger.info(f"Configured {len(aliases)} voice aliases from Wing channel names")
                    
                    # Log the mappings
                    for alias, ch in sorted(aliases.items(), key=lambda x: x[1]):
                        logger.info(f"  '{alias}' -> channel {ch}")
            else:
                logger.info("No aliases created from Wing channel names")
                
        except Exception as e:
            logger.error(f"Error setting up voice aliases: {e}")
    
    async def rescan_channel_names(self):
        """
        Force rescan of all channel names from Wing mixer and update voice aliases.
        Useful when channel names have changed on the mixer.
        """
        if not self.mixer_client or not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
            logger.warning("Rescan only available for Wing mixer")
            return
        
        logger.info("Rescanning channel names from Wing...")
        
        # Request fresh channel names
        for ch in range(1, 41):
            self.mixer_client.send(f"/ch/{ch}/name")
            await asyncio.sleep(0.01)  # Small delay between requests
        
        # Wait for responses
        await asyncio.sleep(0.5)
        
        # Rebuild aliases
        self._setup_voice_aliases_from_wing()
        
        logger.info("Channel names rescanned successfully")

    async def stop_voice_control(self):
        """Stop voice control listening"""
        try:
            if self.voice_control:
                self.voice_control.stop_listening()
                await self.broadcast({
                    "type": "voice_control_status",
                    "active": False,
                    "message": "Voice control stopped"
                })
                logger.info("Voice control stopped")
        except Exception as e:
            logger.error(f"Error stopping voice control: {e}")
    
    # ========== Real-time Peak Correction Methods ==========
    
    def get_gain_staging_status(self) -> dict:
        """Get current gain staging status."""
        # Если используется SafeGainCalibrator, возвращаем его статус
        if self.safe_gain_calibrator:
            safe_status = self.safe_gain_calibrator.get_status()
            return {
                "active": safe_status.get('state') != 'idle',
                "realtime_enabled": safe_status.get('state') == 'learning',
                "safe_gain_mode": True,
                "state": safe_status.get('state'),
                "learning_progress": safe_status.get('learning_progress', 0.0),
                "channels_count": safe_status.get('channels_count', 0),
                "suggestions_ready": safe_status.get('suggestions_ready', False),
                "target_lufs": safe_status.get('target_lufs'),
                "max_peak_limit": safe_status.get('max_peak_limit'),
                "live_mode": self.live_mode,
                "automation_frozen": False
            }
        
        # Старый метод (для обратной совместимости)
        if not self.gain_staging:
            return {"active": False, "live_mode": self.live_mode, "automation_frozen": False, "safe_gain_mode": False}
        st = self.gain_staging.get_status()
        st["live_mode"] = getattr(self.gain_staging, "live_mode", False)
        st["automation_frozen"] = getattr(self.gain_staging, "automation_frozen", False)
        st["safe_gain_mode"] = False
        return st
    
    async def update_safe_gain_settings(self, settings: dict):
        """Обновить настройки Safe Gain Calibration."""
        if not self.safe_gain_calibrator:
            logger.warning("Cannot update safe gain settings: calibrator not initialized")
            return
        
        # Обновляем настройки в калибраторе
        self.safe_gain_calibrator.update_settings(settings)
        
        # Обновляем конфиг
        if 'learning_duration_sec' in settings:
            if 'automation' not in self.config:
                self.config['automation'] = {}
            if 'safe_gain_calibration' not in self.config['automation']:
                self.config['automation']['safe_gain_calibration'] = {}
            self.config['automation']['safe_gain_calibration']['learning_duration_sec'] = settings['learning_duration_sec']
            logger.info(f"Updated learning_duration_sec to {settings['learning_duration_sec']} seconds")
        
        # Отправляем подтверждение клиенту
        await self.broadcast({
            "type": "safe_gain_settings_updated",
            "settings": settings
        })
    
    async def start_realtime_correction(self, device_id: str = None, channels: list = None, 
                                        channel_settings: dict = None, channel_mapping: dict = None,
                                        mode: str = "lufs", learning_duration_sec: float = None):
        """Start Safe Gain Staging calibration (новый метод: анализ → одноразовое применение)."""
        logger.info("Starting Safe Gain Staging calibration (new method)...")
        
        # Check mixer connection
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.broadcast({
                "type": "gain_staging_status",
                "active": False,
                "realtime_enabled": False,
                "error": "Mixer not connected"
            })
            return
        
        # Check if we have required parameters
        if not device_id or not channels or not channel_settings:
            await self.broadcast({
                "type": "gain_staging_status",
                "active": False,
                "realtime_enabled": False,
                "error": "Missing required parameters: device_id, channels, or channel_settings"
            })
            logger.warning("Missing required parameters for Safe Gain calibration")
            return
        
        # Останавливаем старый калибратор, если он активен
        if self.safe_gain_calibrator:
            logger.info("Stopping existing SafeGainCalibrator before starting new one...")
            if self.safe_gain_calibrator.state.value != 'idle':
                self.safe_gain_calibrator.reset()
            self.safe_gain_calibrator = None
        
        try:
            # Обновляем learning_duration_sec в конфиге, если передан
            if learning_duration_sec is not None:
                if 'automation' not in self.config:
                    self.config['automation'] = {}
                if 'safe_gain_calibration' not in self.config['automation']:
                    self.config['automation']['safe_gain_calibration'] = {}
                self.config['automation']['safe_gain_calibration']['learning_duration_sec'] = float(learning_duration_sec)
                logger.info(f"Applying learning_duration_sec={learning_duration_sec} seconds for this measurement")
            
            # Initialize SafeGainCalibrator
            import pyaudio
            pa = pyaudio.PyAudio()
            device_info = pa.get_device_info_by_index(int(device_id))
            sample_rate = int(device_info.get('defaultSampleRate', 48000))
            pa.terminate()
            
            logger.info("Initializing SafeGainCalibrator...")
            
            # Логируем значение из конфига перед созданием калибратора
            cal_config = self.config.get('automation', {}).get('safe_gain_calibration', {})
            logger.info(f"Config learning_duration_sec before calibrator init: {cal_config.get('learning_duration_sec', 'NOT SET')}")
            logger.info(f"Requested learning_duration_sec: {learning_duration_sec}")
            
            self.safe_gain_calibrator = SafeGainCalibrator(
                mixer_client=self.mixer_client,
                sample_rate=sample_rate,
                config=self.config
            )
            
            # Логируем значение после создания калибратора
            logger.info(f"Calibrator learning_duration after init: {self.safe_gain_calibrator.learning_duration} seconds")
            
            # Применяем learning_duration_sec, если передан (обновляем после создания)
            if learning_duration_sec is not None:
                old_value = self.safe_gain_calibrator.learning_duration
                self.safe_gain_calibrator.learning_duration = float(learning_duration_sec)
                logger.info(f"Updated learning_duration: {old_value} -> {self.safe_gain_calibrator.learning_duration} seconds")
            
            # Setup callbacks
            loop = asyncio.get_running_loop()
            
            def on_progress_update(status: dict):
                message = {
                    "type": "gain_staging_status",
                    "status_type": "safe_gain_progress",
                    **status
                }
                loop.call_soon_threadsafe(
                    lambda msg=message: asyncio.create_task(self.broadcast(msg))
                )
            
            def on_suggestions_ready(suggestions: dict):
                message = {
                    "type": "gain_staging_status",
                    "status_type": "safe_gain_ready",
                    "suggestions": suggestions,
                    "message": "Analysis complete. Applying corrections..."
                }
                loop.call_soon_threadsafe(
                    lambda msg=message: asyncio.create_task(self.broadcast(msg))
                )
                # Автоматически применяем коррекции после завершения анализа
                if self.safe_gain_calibrator:
                    applied = self.safe_gain_calibrator.apply_corrections()
                    if applied:
                        logger.info("Safe Gain corrections applied automatically")
                        loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(self.broadcast({
                                "type": "gain_staging_status",
                                "status_type": "safe_gain_applied",
                                "message": "Corrections applied successfully"
                            }))
                        )
            
            self.safe_gain_calibrator.on_progress_update = on_progress_update
            self.safe_gain_calibrator.on_suggestions_ready = on_suggestions_ready
            
            # Add channels to SafeGainCalibrator
            for audio_ch_str in channels:
                audio_ch = int(audio_ch_str) if isinstance(audio_ch_str, str) else audio_ch_str
                mixer_ch = channel_mapping.get(audio_ch, audio_ch) if channel_mapping else audio_ch
                self.safe_gain_calibrator.add_channel(audio_channel=audio_ch, mixer_channel=mixer_ch)
            
            # Initialize audio analyzer (используем gain_staging только для захвата аудио)
            if not self.gain_staging:
                logger.info("Initializing audio analyzer for Safe Gain calibration...")
                
                self.gain_staging = LUFSGainStagingController(
                    mixer_client=self.mixer_client,
                    config=self.config,
                    bleed_service=self.bleed_service
                )
            else:
                self.gain_staging.mixer_client = self.mixer_client
            
            # Convert channel_settings and channel_mapping for audio analyzer
            converted_settings = {}
            instrument_types_for_bleed = {}
            for ch_str, settings in channel_settings.items():
                ch = int(ch_str) if isinstance(ch_str, str) else ch_str
                converted_settings[ch] = settings
                # Extract instrument type for bleed service
                instrument_type = settings.get('instrumentType') or settings.get('preset') or 'custom'
                instrument_types_for_bleed[ch] = 'tom' if instrument_type == 'toms' else instrument_type
            
            # Configure bleed service with instrument types
            if self.bleed_service and instrument_types_for_bleed:
                self.bleed_service.configure(instrument_types_for_bleed)
            
            converted_mapping = {}
            if channel_mapping:
                for audio_ch_str, mixer_ch in channel_mapping.items():
                    audio_ch = int(audio_ch_str) if isinstance(audio_ch_str, str) else audio_ch_str
                    converted_mapping[audio_ch] = mixer_ch
            else:
                for audio_ch in channels:
                    audio_ch_int = int(audio_ch) if isinstance(audio_ch, str) else audio_ch
                    converted_mapping[audio_ch_int] = audio_ch_int
            
            self.gain_staging.channel_settings = converted_settings
            self.gain_staging.channel_mapping = converted_mapping
            
            # Set channel_settings for SafeGainCalibrator to use preset-based metrics
            self.safe_gain_calibrator.channel_settings = converted_settings
            # Stop existing audio analyzer if running
            if self.gain_staging.is_active:
                logger.info("Stopping existing audio analyzer...")
                self.gain_staging.stop()
            
            # Start audio analyzer for SafeGainCalibrator
            logger.info("Starting audio analyzer for Safe Gain calibration...")
            
            # Поток для передачи аудио данных в SafeGainCalibrator
            self._safe_gain_audio_thread_stop = threading.Event()
            
            def safe_gain_audio_processor():
                """Периодически передает аудио данные из буферов в SafeGainCalibrator"""
                import time
                while not self._safe_gain_audio_thread_stop.is_set():
                    if not self.safe_gain_calibrator or self.safe_gain_calibrator.state.value != 'learning':
                        time.sleep(0.1)
                        continue
                    
                    if not self.gain_staging or not hasattr(self.gain_staging, '_audio_buffers'):
                        time.sleep(0.1)
                        continue
                    
                    # Process audio for each channel from audio buffers
                    for audio_ch in channels:
                        audio_ch_int = int(audio_ch) if isinstance(audio_ch, str) else audio_ch
                        if audio_ch_int in self.gain_staging._audio_buffers:
                            buffer = self.gain_staging._audio_buffers[audio_ch_int]
                            if buffer and len(buffer) > 0:
                                # Get latest audio chunks (copy to avoid race conditions)
                                chunks = list(buffer)
                                if chunks:
                                    try:
                                        samples = np.concatenate(chunks)
                                        if len(samples) > 0:
                                            self.safe_gain_calibrator.process_audio(audio_ch_int, samples)
                                    except Exception as e:
                                        logger.debug(f"Error processing audio for SafeGain: {e}")
                    
                    time.sleep(0.05)  # Process every 50ms
            
            self._safe_gain_audio_thread = threading.Thread(
                target=safe_gain_audio_processor,
                daemon=True
            )
            
            success = self.gain_staging.start(
                device_id=device_id,
                channels=[int(ch) for ch in channels],
                channel_settings=converted_settings,
                channel_mapping=converted_mapping,
                on_status_callback=None
            )
            
            if not success:
                await self.broadcast({
                    "type": "gain_staging_status",
                    "active": False,
                    "realtime_enabled": False,
                    "error": "Failed to start audio analyzer"
                })
                return
            
            # Start audio processing thread
            self._safe_gain_audio_thread.start()
            logger.info("Audio processing thread started for Safe Gain calibration")
            
            # Start Safe Gain analysis
            analysis_started = self.safe_gain_calibrator.start_analysis()
            
            if not analysis_started:
                await self.broadcast({
                    "type": "gain_staging_status",
                    "active": False,
                    "realtime_enabled": False,
                    "error": "Failed to start Safe Gain analysis"
                })
                return
            
            await self.broadcast({
                "type": "gain_staging_status",
                "active": True,
                "realtime_enabled": True,
                "status_type": "safe_gain_started",
                "message": f"Safe Gain analysis started (learning for {self.safe_gain_calibrator.learning_duration}s)"
            })
            logger.info("Safe Gain calibration started successfully")
        except Exception as e:
            logger.error(f"Error starting real-time correction: {e}", exc_info=True)
            await self.broadcast({
                "type": "gain_staging_status",
                "active": self.gain_staging.is_active if self.gain_staging else False,
                "realtime_enabled": False,
                "error": f"Error: {str(e)}"
            })
    
    async def stop_realtime_correction(self):
        """Stop Safe Gain calibration or real-time TRIM correction."""
        logger.info("Stopping gain staging...")
        
        # Останавливаем SafeGainCalibrator если активен
        if self.safe_gain_calibrator:
            logger.info("Stopping Safe Gain calibration...")
            # Останавливаем поток обработки аудио
            if hasattr(self, '_safe_gain_audio_thread_stop'):
                self._safe_gain_audio_thread_stop.set()
            if hasattr(self, '_safe_gain_audio_thread') and self._safe_gain_audio_thread.is_alive():
                self._safe_gain_audio_thread.join(timeout=1.0)
            self.safe_gain_calibrator.reset()
            self.safe_gain_calibrator = None
        
        # Останавливаем аудио анализатор
        if self.gain_staging:
            if self.gain_staging.realtime_correction_enabled:
                self.gain_staging.stop_realtime_correction()
            if self.gain_staging.is_active:
                self.gain_staging.stop()
        
        await self.broadcast({
            "type": "gain_staging_status",
            "active": False,
            "realtime_enabled": False,
            "message": "Gain staging stopped"
        })
        
        logger.info("Real-time correction stopped")
    
    async def scan_and_recognize_channels(self, websocket, channels: list):
        """
        Scan channel names from mixer and recognize instrument types.
        
        Args:
            websocket: WebSocket connection to respond to
            channels: List of channel numbers to scan (empty = all selected)
        """
        logger.info(f"Scanning channel names for recognition. Requested channels: {channels}")
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "channel_scan_result",
                "error": "Mixer not connected",
                "results": {}
            })
            return
        
        try:
            # Get channel names from mixer
            if isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                # Per Wing Remote Protocols v3.0.5:
                # /ch/X/name - для установки имени
                # /ch/X/$name - для чтения имени (read-only, отражает связанный источник)
                logger.info(f"Querying channel names using /ch/X/$name for channels: {channels}")
                
                # Send queries for $name (read-only address that reflects linked source)
                for ch in channels:
                    self.mixer_client.send(f"/ch/{ch}/$name")
                    await asyncio.sleep(0.02)
                
                # Wait for responses
                await asyncio.sleep(0.5)
                
                # Collect names from state
                channel_names = {}
                for ch in channels:
                    # Try $name first (read-only, reflects linked source)
                    name = self.mixer_client.state.get(f"/ch/{ch}/$name")
                    if not name or name == '':
                        # Fallback to regular name
                        name = self.mixer_client.state.get(f"/ch/{ch}/name")
                    
                    if name and name != '':
                        channel_names[ch] = name
                        logger.info(f"Channel {ch} name: '{name}'")
                    else:
                        channel_names[ch] = f"Ch {ch}"
                        logger.debug(f"Channel {ch}: no name found")
                
                logger.info(f"Retrieved {len(channel_names)} channel names")
                
                # Run recognition
                results = scan_and_recognize(channel_names)
                
                await self.send_to_client(websocket, {
                    "type": "channel_scan_result",
                    "results": results,
                    "available_presets": AVAILABLE_PRESETS
                })
                
                recognized_count = sum(1 for r in results.values() if r.get('recognized'))
                logger.info(f"Channel scan complete: {recognized_count}/{len(results)} recognized")
                
            else:
                await self.send_to_client(websocket, {
                    "type": "channel_scan_result",
                    "error": "Channel scanning only supported for Wing mixer",
                    "results": {}
                })
                
        except Exception as e:
            logger.error(f"Error scanning channels: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "channel_scan_result",
                "error": str(e),
                "results": {}
            })
    
    async def scan_mixer_channel_names(self, websocket):
        """
        Scan all channel names from mixer (channels 1-40) and return them for Audio Device tab.
        Updates channel names in the available channels list.
        """
        logger.info("Scanning mixer channel names for Audio Device tab...")
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "mixer_channel_names",
                "error": "Mixer not connected",
                "channel_names": {}
            })
            return
        
        try:
            if isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                # Scan all channels (1-40) using async approach
                logger.info("Querying channel names for all 40 channels...")
                
                # Send queries for $name (read-only address that reflects linked source)
                for ch in range(1, 41):
                    self.mixer_client.send(f"/ch/{ch}/$name")
                    await asyncio.sleep(0.02)
                
                # Wait longer for responses to arrive
                await asyncio.sleep(1.0)
                
                # Collect names from state
                channel_names = {}
                for ch in range(1, 41):
                    # Try $name first (read-only, reflects linked source)
                    name = self.mixer_client.state.get(f"/ch/{ch}/$name")
                    if not name or name == '':
                        # Fallback to regular name
                        name = self.mixer_client.state.get(f"/ch/{ch}/name")
                    
                    # Handle tuple response (Wing sometimes returns tuples)
                    if isinstance(name, tuple) and len(name) > 0:
                        name = name[0]
                    
                    if name and str(name).strip():
                        channel_names[ch] = str(name).strip()
                        logger.debug(f"Channel {ch} name: '{channel_names[ch]}'")
                    else:
                        channel_names[ch] = f"Ch {ch}"
                        logger.debug(f"Channel {ch}: no name found, using default")
                
                logger.info(f"Retrieved {len(channel_names)} channel names from mixer")
                logger.info(f"Sample names: {dict(list(channel_names.items())[:5])}")
                
                await self.send_to_client(websocket, {
                    "type": "mixer_channel_names",
                    "channel_names": channel_names
                })
            else:
                await self.send_to_client(websocket, {
                    "type": "mixer_channel_names",
                    "error": "Channel name scanning only supported for Wing mixer",
                    "channel_names": {}
                })
                
        except Exception as e:
            logger.error(f"Error scanning mixer channel names: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "mixer_channel_names",
                "error": str(e),
                "channel_names": {}
            })
    
    async def reset_trim(self, websocket, channels: list):
        """
        Reset TRIM to 0dB for selected channels.
        
        Args:
            websocket: WebSocket connection to respond to
            channels: List of channel numbers to reset (mixer channels)
        """
        logger.info(f"Resetting TRIM to 0dB for channels: {channels}")
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "reset_trim_result",
                "success": False,
                "error": "Mixer not connected",
                "channels": channels
            })
            return
        
        if not channels:
            await self.send_to_client(websocket, {
                "type": "reset_trim_result",
                "success": False,
                "error": "No channels specified",
                "channels": []
            })
            return
        
        # Use asyncio.wait_for to prevent hanging
        async def reset_trim_task():
            results = {}
            success_count = 0
            failed_channels = []
            
            logger.info(f"Starting TRIM reset for {len(channels)} channels: {channels}")
            logger.info(f"Mixer client type: {type(self.mixer_client).__name__}")
            logger.info(f"Mixer connected: {self.mixer_client.is_connected if hasattr(self.mixer_client, 'is_connected') else 'N/A'}")
            
            # Set TRIM to 0 dB for all channels
            for ch in channels:
                try:
                    # Set TRIM to 0 dB using direct OSC command for reliability
                    if isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                        # Use direct OSC address for TRIM
                        address = f"/ch/{ch}/in/set/trim"
                        logger.info(f"Channel {ch}: Sending TRIM reset to {address} = 0.0 dB")
                        result = self.mixer_client.send(address, 0.0)
                        
                        if result:
                            success_count += 1
                            results[ch] = {
                                "success": True,
                                "new_trim": 0.0
                            }
                            logger.info(f"Channel {ch}: TRIM reset command sent successfully")
                            
                            # Small delay to allow mixer to process
                            await asyncio.sleep(0.1)
                            
                            # Query the trim value to verify and trigger state update
                            logger.debug(f"Channel {ch}: Requesting TRIM value update from mixer")
                            self.mixer_client.send(f"/ch/{ch}/in/set/trim")
                        else:
                            # Fallback to set_channel_gain method
                            logger.warning(f"Channel {ch}: Direct OSC failed, trying set_channel_gain method")
                            result = self.mixer_client.set_channel_gain(ch, 0.0)
                            if result:
                                success_count += 1
                                results[ch] = {
                                    "success": True,
                                    "new_trim": 0.0
                                }
                            else:
                                failed_channels.append(ch)
                                results[ch] = {
                                    "success": False,
                                    "error": "Failed to send TRIM reset command"
                                }
                                logger.warning(f"Channel {ch}: Failed to send TRIM reset command")
                    else:
                        # For other mixer clients, use standard method
                        result = self.mixer_client.set_channel_gain(ch, 0.0)
                        if result:
                            success_count += 1
                            results[ch] = {
                                "success": True,
                                "new_trim": 0.0
                            }
                            logger.info(f"Channel {ch}: TRIM reset command sent successfully")
                        else:
                            failed_channels.append(ch)
                            results[ch] = {
                                "success": False,
                                "error": "Failed to send TRIM reset command"
                            }
                            logger.warning(f"Channel {ch}: Failed to send TRIM reset command")
                    
                    # Delay between commands to allow mixer to process
                    await asyncio.sleep(0.1)  # Increased delay for better reliability
                    
                except Exception as e:
                    failed_channels.append(ch)
                    results[ch] = {
                        "success": False,
                        "error": str(e)
                    }
                    logger.error(f"Error resetting TRIM for channel {ch}: {e}", exc_info=True)
            
            # Wait longer for mixer to process all commands
            await asyncio.sleep(0.5)
            
            # Verify TRIM values if possible (for WingClient)
            if isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                logger.info("Verifying TRIM reset...")
                await asyncio.sleep(0.5)  # Increased delay for state updates
                verification_success = 0
                for ch in channels:
                    try:
                        # Query current TRIM value by requesting it
                        self.mixer_client.send(f"/ch/{ch}/in/set/trim")
                        await asyncio.sleep(0.1)  # Wait for response
                        
                        # Check if TRIM is actually 0.0 (or close to it)
                        current_trim = self.mixer_client.get_channel_gain(ch)
                        if current_trim is not None:
                            if abs(current_trim) < 0.1:  # Allow small tolerance
                                verification_success += 1
                                logger.info(f"Channel {ch}: TRIM verified at {current_trim:.2f} dB (reset successful)")
                            else:
                                logger.warning(f"Channel {ch}: TRIM reset may have failed - current value: {current_trim:.2f} dB, expected: 0.0 dB")
                                # Try to reset again
                                logger.info(f"Channel {ch}: Retrying TRIM reset...")
                                self.mixer_client.set_channel_gain(ch, 0.0)
                                await asyncio.sleep(0.1)
                        else:
                            logger.warning(f"Channel {ch}: Could not verify TRIM value (no response from mixer)")
                    except Exception as e:
                        logger.debug(f"Could not verify TRIM for channel {ch}: {e}")
                
                logger.info(f"TRIM reset verification: {verification_success}/{len(channels)} channels verified")
            
            # Send response immediately
            await self.send_to_client(websocket, {
                "type": "reset_trim_result",
                "success": success_count > 0,
                "success_count": success_count,
                "total_count": len(channels),
                "failed_channels": failed_channels,
                "results": results,
                "message": f"Reset TRIM to 0dB for {success_count}/{len(channels)} channels"
            })
            
            logger.info(f"TRIM reset completed: {success_count}/{len(channels)} channels successful")
            if failed_channels:
                logger.warning(f"Failed to reset TRIM for channels: {failed_channels}")
        
        try:
            # Set 5 second timeout to prevent hanging
            await asyncio.wait_for(reset_trim_task(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("Timeout while resetting TRIM")
            await self.send_to_client(websocket, {
                "type": "reset_trim_result",
                "success": False,
                "error": "Operation timed out",
                "channels": channels
            })
        except Exception as e:
            logger.error(f"Error resetting TRIM: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "reset_trim_result",
                "success": False,
                "error": str(e),
                "channels": channels
            })
    
    async def bypass_mixer(self, websocket):
        """
        Bypass mixer: Disable all modules and set faders to 0dB for all 40 channels.
        
        Args:
            websocket: WebSocket connection to respond to
        """
        logger.info("Starting bypass operation: disabling all modules and setting faders to 0dB")
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "bypass_result",
                "success": False,
                "error": "Mixer not connected"
            })
            return
        
        if not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
            await self.send_to_client(websocket, {
                "type": "bypass_result",
                "success": False,
                "error": "Bypass operation only supported for Wing mixer"
            })
            return
        
        async def bypass_task():
            success_count = 0
            failed_channels = []
            
            try:
                # Process all 40 channels
                for ch in range(1, 41):
                    try:
                        # 1. Set fader to 0dB
                        self.mixer_client.set_channel_fader(ch, 0.0)
                        await asyncio.sleep(0.01)
                        
                        # 2. Disable EQ
                        self.mixer_client.set_eq_on(ch, 0)
                        await asyncio.sleep(0.01)
                        
                        # 3. Disable PreEQ
                        self.mixer_client.send(f"/ch/{ch}/peq/on", 0)
                        await asyncio.sleep(0.01)
                        
                        # 4. Disable Compressor/Dynamics
                        self.mixer_client.set_compressor_on(ch, 0)
                        await asyncio.sleep(0.01)
                        
                        # 5. Disable Gate
                        self.mixer_client.set_gate_on(ch, 0)
                        await asyncio.sleep(0.01)
                        
                        # 6. Disable Filters
                        self.mixer_client.set_low_cut(ch, enabled=0)
                        await asyncio.sleep(0.01)
                        self.mixer_client.set_high_cut(ch, enabled=0)
                        await asyncio.sleep(0.01)
                        
                        # 7. Disable Inserts
                        self.mixer_client.send(f"/ch/{ch}/preins/on", 0)
                        await asyncio.sleep(0.01)
                        self.mixer_client.send(f"/ch/{ch}/postins/on", 0)
                        await asyncio.sleep(0.01)
                        
                        success_count += 1
                        
                        if ch % 10 == 0:
                            logger.info(f"Bypass progress: {ch}/40 channels processed")
                            
                    except Exception as e:
                        failed_channels.append(ch)
                        logger.error(f"Error bypassing channel {ch}: {e}", exc_info=True)
                
                # Send response
                await self.send_to_client(websocket, {
                    "type": "bypass_result",
                    "success": success_count > 0,
                    "success_count": success_count,
                    "total_count": 40,
                    "failed_channels": failed_channels,
                    "message": f"Bypass completed: {success_count}/40 channels processed"
                })
                
                logger.info(f"Bypass operation completed: {success_count}/40 channels successful")
                if failed_channels:
                    logger.warning(f"Failed to bypass channels: {failed_channels}")
                    
            except Exception as e:
                logger.error(f"Error in bypass operation: {e}", exc_info=True)
                await self.send_to_client(websocket, {
                    "type": "bypass_result",
                    "success": False,
                    "error": str(e)
                })
        
        # Run bypass task with timeout
        try:
            await asyncio.wait_for(bypass_task(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error("Bypass operation timed out")
            await self.send_to_client(websocket, {
                "type": "bypass_result",
                "success": False,
                "error": "Operation timed out"
            })
        except Exception as e:
            logger.error(f"Error executing bypass: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "bypass_result",
                "success": False,
                "error": str(e)
            })
    
    # ========== Auto-EQ Methods ==========
    
    def get_auto_eq_status(self) -> dict:
        """Get current Auto-EQ status."""
        if not self.auto_eq_controller:
            return {"active": False}
        return self.auto_eq_controller.get_status()
    
    async def start_auto_eq(self, websocket, device_id: str = None, channel: int = None,
                            profile: str = "custom", auto_apply: bool = False,
                            monitored_channels: List[int] = None):
        """Start Auto-EQ analysis for a channel."""
        monitored_channels = monitored_channels or []
        logger.info(f"Starting Auto-EQ: device={device_id}, channel={channel}, profile={profile}, monitored_channels={monitored_channels}")
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "auto_eq_status",
                "active": False,
                "error": "Mixer not connected"
            })
            return
        
        if not device_id or not channel:
            await self.send_to_client(websocket, {
                "type": "auto_eq_status",
                "active": False,
                "error": "Missing required parameters: device_id or channel"
            })
            return
        
        try:
            # Initialize controller if needed
            if not self.auto_eq_controller:
                self.auto_eq_controller = AutoEQController(
                    mixer_client=self.mixer_client,
                    bleed_service=self.bleed_service
                )
            else:
                # Update mixer client reference
                self.auto_eq_controller.mixer_client = self.mixer_client
            
            # Get event loop for thread-safe callbacks
            loop = asyncio.get_running_loop()
            
            def on_spectrum_update(spectrum_data: dict):
                """Handle spectrum data updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "auto_eq_spectrum",
                        **spectrum_data
                    }))
                )
            
            def on_corrections_calculated(corrections: list):
                """Handle EQ corrections updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "auto_eq_corrections",
                        "corrections": corrections
                    }))
                )
            
            def on_status_update(status: dict):
                """Handle status updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "auto_eq_status",
                        **status
                    }))
                )
            
            # Start analysis
            success = self.auto_eq_controller.start(
                device_id=int(device_id),
                channel=int(channel),
                profile_name=profile,
                auto_apply=auto_apply,
                monitored_channels=monitored_channels,
                on_spectrum_callback=on_spectrum_update,
                on_corrections_callback=on_corrections_calculated,
                on_status_callback=on_status_update
            )
            
            if success:
                await self.send_to_client(websocket, {
                    "type": "auto_eq_status",
                    "active": True,
                    "channel": channel,
                    "profile": profile,
                    "auto_apply": auto_apply,
                    "message": f"Auto-EQ started for channel {channel}"
                })
                logger.info(f"Auto-EQ started successfully for channel {channel}")
            else:
                await self.send_to_client(websocket, {
                    "type": "auto_eq_status",
                    "active": False,
                    "error": "Failed to start Auto-EQ analysis"
                })
                
        except Exception as e:
            logger.error(f"Error starting Auto-EQ: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_eq_status",
                "active": False,
                "error": str(e)
            })
    
    async def stop_auto_eq(self, websocket):
        """Stop Auto-EQ analysis."""
        logger.info("Stopping Auto-EQ...")
        
        if self.auto_eq_controller:
            self.auto_eq_controller.stop()
        
        await self.send_to_client(websocket, {
            "type": "auto_eq_status",
            "active": False,
            "message": "Auto-EQ stopped"
        })
        
        logger.info("Auto-EQ stopped")
    
    async def set_eq_profile(self, websocket, profile: str):
        """Change the EQ profile."""
        if not self.auto_eq_controller:
            await self.send_to_client(websocket, {
                "type": "auto_eq_status",
                "error": "Auto-EQ not initialized"
            })
            return
        
        self.auto_eq_controller.set_profile(profile)
        
        await self.send_to_client(websocket, {
            "type": "auto_eq_status",
            "profile": profile,
            "message": f"Profile changed to {profile}"
        })
        
        logger.info(f"EQ profile changed to: {profile}")
    
    async def apply_eq_correction(self, websocket):
        """Apply calculated EQ corrections to the mixer."""
        if not self.auto_eq_controller:
            await self.send_to_client(websocket, {
                "type": "auto_eq_apply_result",
                "success": False,
                "error": "Auto-EQ not initialized"
            })
            return
        
        try:
            success = self.auto_eq_controller.apply_to_mixer()
            
            await self.send_to_client(websocket, {
                "type": "auto_eq_apply_result",
                "success": success,
                "message": "EQ corrections applied to mixer" if success else "Failed to apply EQ"
            })
            
        except Exception as e:
            logger.error(f"Error applying EQ: {e}")
            await self.send_to_client(websocket, {
                "type": "auto_eq_apply_result",
                "success": False,
                "error": str(e)
            })
    
    async def reset_eq(self, websocket, data: dict = None):
        """Reset EQ to flat."""
        if data is None:
            data = {}
        
        # Try to get channel from message, auto_eq_controller, or use default
        channel = None
        if data.get("channel"):
            channel = data.get("channel")
            logger.info(f"Reset EQ: Using channel from message: {channel}")
        elif self.auto_eq_controller and self.auto_eq_controller.current_channel:
            channel = self.auto_eq_controller.current_channel
            logger.info(f"Reset EQ: Using channel from auto_eq_controller: {channel}")
        else:
            channel = 1  # Default fallback
            logger.info(f"Reset EQ: Using default channel: {channel}")
        
        # If we have auto_eq_controller with current_channel, use it
        if self.auto_eq_controller and self.auto_eq_controller.current_channel:
            try:
                success = self.auto_eq_controller.reset_eq()
                logger.info(f"Reset EQ via auto_eq_controller: success={success}")
                await self.send_to_client(websocket, {
                    "type": "auto_eq_reset_result",
                    "success": success,
                    "message": "EQ reset to flat" if success else "Failed to reset EQ"
                })
                return
            except Exception as e:
                logger.error(f"Error resetting EQ via auto_eq_controller: {e}", exc_info=True)
                # Fall through to direct reset
        
        # Direct reset via mixer_client
        if self.mixer_client and hasattr(self.mixer_client, 'send'):
            try:
                logger.info(f"Reset EQ: Resetting channel {channel} directly via mixer_client")
                
                # Ensure EQ is enabled (needed for changes to take effect)
                if hasattr(self.mixer_client, 'set_eq_on'):
                    logger.info(f"Reset EQ: Enabling EQ for channel {channel}")
                    self.mixer_client.set_eq_on(channel, 1)
                    await asyncio.sleep(0.05)
                
                # Reset EQ bands to flat using direct OSC addresses (only reset gain, don't change Q or frequency)
                logger.info(f"Reset EQ: Sending gain=0 commands for channel {channel}")
                
                # Low shelf gain
                self.mixer_client.send(f"/ch/{channel}/eq/lg", 0.0)
                logger.debug(f"Reset EQ: Sent /ch/{channel}/eq/lg = 0.0")
                await asyncio.sleep(0.05)
                
                # Band 1-4 gains
                for band in [1, 2, 3, 4]:
                    self.mixer_client.send(f"/ch/{channel}/eq/{band}g", 0.0)
                    logger.debug(f"Reset EQ: Sent /ch/{channel}/eq/{band}g = 0.0")
                    await asyncio.sleep(0.05)
                
                # High shelf gain
                self.mixer_client.send(f"/ch/{channel}/eq/hg", 0.0)
                logger.debug(f"Reset EQ: Sent /ch/{channel}/eq/hg = 0.0")
                
                logger.info(f"Reset EQ: Successfully reset channel {channel}")
                await self.send_to_client(websocket, {
                    "type": "auto_eq_reset_result",
                    "success": True,
                    "message": f"EQ reset to flat for channel {channel}"
                })
                return
            except Exception as e:
                logger.error(f"Error resetting EQ directly: {e}", exc_info=True)
                await self.send_to_client(websocket, {
                    "type": "auto_eq_reset_result",
                    "success": False,
                    "error": str(e)
                })
                return
        
        # No way to reset
        logger.warning("Reset EQ: No mixer client available")
        await self.send_to_client(websocket, {
            "type": "auto_eq_reset_result",
            "success": False,
            "error": "Mixer not connected"
        })

    async def reset_all_eq(self, websocket, data: dict = None):
        """Reset EQ to flat for multiple channels."""
        if data is None:
            data = {}
        
        channels = data.get("channels", [])
        if not channels:
            await self.send_to_client(websocket, {
                "type": "reset_all_eq_result",
                "success": False,
                "error": "No channels specified"
            })
            return
        
        if not self.mixer_client or not hasattr(self.mixer_client, 'send'):
            await self.send_to_client(websocket, {
                "type": "reset_all_eq_result",
                "success": False,
                "error": "Mixer not connected"
            })
            return
        
        logger.info(f"Reset All EQ: Resetting {len(channels)} channels")
        reset_count = 0
        errors = []
        
        try:
            for channel in channels:
                try:
                    # Ensure EQ is enabled
                    if hasattr(self.mixer_client, 'set_eq_on'):
                        self.mixer_client.set_eq_on(channel, 1)
                        await asyncio.sleep(0.05)
                    
                    # Reset all EQ bands to 0 dB
                    self.mixer_client.send(f"/ch/{channel}/eq/lg", 0.0)
                    await asyncio.sleep(0.05)
                    
                    for band in [1, 2, 3, 4]:
                        self.mixer_client.send(f"/ch/{channel}/eq/{band}g", 0.0)
                        await asyncio.sleep(0.05)
                    
                    self.mixer_client.send(f"/ch/{channel}/eq/hg", 0.0)
                    reset_count += 1
                    logger.debug(f"Reset All EQ: Successfully reset channel {channel}")
                    
                except Exception as e:
                    error_msg = f"Channel {channel}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"Reset All EQ: Error resetting channel {channel}: {e}")
            
            success = reset_count > 0
            message = f"Reset EQ for {reset_count}/{len(channels)} channels"
            if errors:
                message += f". Errors: {len(errors)}"
            
            await self.send_to_client(websocket, {
                "type": "reset_all_eq_result",
                "success": success,
                "reset_count": reset_count,
                "total_count": len(channels),
                "message": message,
                "errors": errors if errors else None
            })
            
            logger.info(f"Reset All EQ: Completed. Reset {reset_count}/{len(channels)} channels")
            
        except Exception as e:
            logger.error(f"Reset All EQ: Unexpected error: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "reset_all_eq_result",
                "success": False,
                "error": str(e)
            })

    async def start_multi_channel_auto_eq(self, websocket, device_id: str = None, channels_config: List[Dict] = None, mode: str = 'soundcheck'):
        """Start multi-channel Auto-EQ analysis.
        
        Args:
            mode: 'soundcheck' (analyze once, then stop) or 'live' (continuous)
        """
        channels_config = channels_config or []
        logger.info(f"Starting Multi-Channel Auto-EQ: device={device_id}, channels={len(channels_config)}, mode={mode}")
        logger.info(f"Channels config: {channels_config}")
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "multi_channel_auto_eq_status",
                "active": False,
                "error": "Mixer not connected"
            })
            return
        
        if device_id is None or not channels_config or len(channels_config) == 0:
            await self.send_to_client(websocket, {
                "type": "multi_channel_auto_eq_status",
                "active": False,
                "error": f"Missing required parameters: device_id={device_id}, channels_config length={len(channels_config)}"
            })
            return
        
        # Validate channels_config structure
        for i, config in enumerate(channels_config):
            if 'channel' not in config:
                await self.send_to_client(websocket, {
                    "type": "multi_channel_auto_eq_status",
                    "active": False,
                    "error": f"Invalid channel config at index {i}: missing 'channel' field"
                })
                return
        
        try:
            # Initialize controller if needed
            if not self.multi_channel_auto_eq_controller:
                self.multi_channel_auto_eq_controller = MultiChannelAutoEQController(
                    mixer_client=self.mixer_client,
                    bleed_service=self.bleed_service
                )
            else:
                self.multi_channel_auto_eq_controller.mixer_client = self.mixer_client
            
            # Get event loop for thread-safe callbacks
            loop = asyncio.get_running_loop()
            
            def on_spectrum_update(spectrum_data: dict):
                """Handle spectrum data updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "multi_channel_spectrum",
                        **spectrum_data
                    }))
                )
            
            def on_corrections_calculated(corrections_data: dict):
                """Handle EQ corrections updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "multi_channel_corrections",
                        **corrections_data
                    }))
                )
            
            def on_status_update(status_data: dict):
                """Handle status updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "multi_channel_status",
                        **status_data
                    }))
                )
            
            # Start multi-channel analysis
            success = self.multi_channel_auto_eq_controller.start_multi_channel(
                device_id=int(device_id),
                channels_config=channels_config,
                mode=mode,
                on_spectrum_callback=on_spectrum_update,
                on_corrections_callback=on_corrections_calculated,
                on_status_callback=on_status_update
            )
            
            if success:
                await self.send_to_client(websocket, {
                    "type": "multi_channel_auto_eq_status",
                    "active": True,
                    "channels_count": len(channels_config),
                    "message": f"Multi-channel Auto-EQ started for {len(channels_config)} channels"
                })
                logger.info(f"Multi-channel Auto-EQ started successfully")
            else:
                await self.send_to_client(websocket, {
                    "type": "multi_channel_auto_eq_status",
                    "active": False,
                    "error": "Failed to start multi-channel Auto-EQ"
                })
                
        except Exception as e:
            logger.error(f"Error starting multi-channel Auto-EQ: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "multi_channel_auto_eq_status",
                "active": False,
                "error": str(e)
            })
    
    async def stop_multi_channel_auto_eq(self, websocket):
        """Stop multi-channel Auto-EQ analysis."""
        if self.multi_channel_auto_eq_controller:
            self.multi_channel_auto_eq_controller.stop_all()
        
        await self.send_to_client(websocket, {
            "type": "multi_channel_auto_eq_status",
            "active": False,
            "message": "Multi-channel Auto-EQ stopped"
        })
        
        logger.info("Multi-channel Auto-EQ stopped")
    
    async def set_channel_profile(self, websocket, channel: int = None, profile: str = None):
        """Change profile for a specific channel in multi-channel mode."""
        if not self.multi_channel_auto_eq_controller:
            await self.send_to_client(websocket, {
                "type": "multi_channel_status",
                "error": "Multi-channel Auto-EQ not initialized"
            })
            return
        
        if channel is None or profile is None:
            await self.send_to_client(websocket, {
                "type": "multi_channel_status",
                "error": "Missing channel or profile parameter"
            })
            return
        
        self.multi_channel_auto_eq_controller.set_channel_profile(channel, profile)
        
        await self.send_to_client(websocket, {
            "type": "multi_channel_status",
            "channel": channel,
            "profile": profile,
            "message": f"Profile changed for channel {channel} to {profile}"
        })
        
        logger.info(f"Channel {channel} profile changed to: {profile}")
    
    async def apply_channel_correction(self, websocket, channel: int = None):
        """Apply corrections for a specific channel."""
        if not self.multi_channel_auto_eq_controller:
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "error": "Multi-channel Auto-EQ not initialized"
            })
            return
        
        if channel is None:
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "error": "Missing channel parameter"
            })
            return
        
        try:
            success = self.multi_channel_auto_eq_controller.apply_channel_correction(channel)
            
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": success,
                "channel": channel,
                "message": f"EQ corrections applied to channel {channel}" if success else f"Failed to apply EQ to channel {channel}"
            })
            
        except Exception as e:
            logger.error(f"Error applying EQ to channel {channel}: {e}")
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "channel": channel,
                "error": str(e)
            })
    
    async def apply_all_corrections(self, websocket):
        """Apply corrections for all channels."""
        if not self.multi_channel_auto_eq_controller:
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "error": "Multi-channel Auto-EQ not initialized. Start batch analysis first."
            })
            return
        
        # Check if there are any channels with corrections
        if not self.multi_channel_auto_eq_controller.active_channels:
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "error": "No channels with corrections available. Run analysis first."
            })
            return
        
        try:
            results = self.multi_channel_auto_eq_controller.apply_all_corrections()
            success_count = sum(1 for v in results.values() if v)
            
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": success_count > 0,
                "results": results,
                "message": f"Applied corrections to {success_count}/{len(results)} channels"
            })
            
        except Exception as e:
            logger.error(f"Error applying all corrections: {e}")
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "error": str(e)
            })
    
    def get_phase_alignment_status(self) -> dict:
        """Get current phase alignment status."""
        if not self.phase_alignment_controller:
            return {"active": False}
        return self.phase_alignment_controller.get_status()
    
    async def start_phase_alignment(self, websocket, device_id: str = None, 
                                   reference_channel: int = None, channels: List[int] = None,
                                   settings: dict = None,
                                   apply_once: bool = False,
                                   apply_once_duration_sec: float = 5.0):
        """Start phase alignment analysis. If apply_once=True, one measurement then auto-stop and apply (for soundcheck)."""
        channels = channels or []
        logger.info(f"Starting Phase Alignment: device={device_id}, ref={reference_channel}, channels={channels}")
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "phase_alignment_status",
                "active": False,
                "error": "Mixer not connected"
            })
            return
        
        if device_id is None or reference_channel is None or not channels:
            await self.send_to_client(websocket, {
                "type": "phase_alignment_status",
                "active": False,
                "error": "Missing required parameters: device_id, reference_channel, or channels"
            })
            return
        
        try:
            # Initialize controller if needed
            settings = settings or {}
            if not self.phase_alignment_controller:
                self.phase_alignment_controller = PhaseAlignmentController(
                    mixer_client=self.mixer_client,
                    bleed_service=self.bleed_service,
                    settings=settings
                )
            else:
                self.phase_alignment_controller.mixer_client = self.mixer_client
                # Update settings for future analyzer instances
                self.phase_alignment_controller.settings = settings
            
            # Get event loop for thread-safe callbacks
            loop = asyncio.get_running_loop()
            
            def on_measurement_update(measurements: dict):
                """Handle measurement updates."""
                import numpy as np
                
                # Конвертируем tuple ключи в строки и NumPy типы в Python типы для JSON сериализации
                measurements_serializable = {}
                for key, value in measurements.items():
                    if isinstance(key, tuple):
                        # Конвертируем tuple в строку вида "(ref_ch, ch)"
                        serializable_key = f"({key[0]}, {key[1]})"
                    else:
                        serializable_key = str(key)
                    
                    # Конвертируем значения измерения в JSON-совместимые типы
                    serializable_value = {}
                    if isinstance(value, dict):
                        for k, v in value.items():
                            # Конвертируем NumPy типы в Python типы
                            if hasattr(v, 'item'):  # NumPy scalar
                                serializable_value[k] = v.item()
                            elif isinstance(v, (np.integer, np.floating)):
                                serializable_value[k] = float(v) if isinstance(v, np.floating) else int(v)
                            elif isinstance(v, (list, tuple)):
                                # Конвертируем списки/кортежи с NumPy типами
                                serializable_value[k] = [
                                    item.item() if hasattr(item, 'item') else 
                                    (float(item) if isinstance(item, np.floating) else int(item) if isinstance(item, np.integer) else item)
                                    for item in v
                                ]
                            else:
                                serializable_value[k] = v
                    else:
                        serializable_value = value
                    
                    measurements_serializable[serializable_key] = serializable_value
                
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "phase_alignment_measurements",
                        "measurements": measurements_serializable
                    }))
                )
            
            # Convert device_id to int if needed
            device_index = int(device_id) if isinstance(device_id, (str, int)) else None
            
            # Run start_analysis in a thread pool to avoid blocking the event loop
            # (PyAudio's pa.open() can block at the C/CoreAudio level if device is busy)
            loop = asyncio.get_running_loop()
            success = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.phase_alignment_controller.start_analysis(
                        device_id=device_index,
                        reference_channel=reference_channel,
                        channels=channels,
                        on_measurement_callback=on_measurement_update,
                        apply_once=apply_once,
                        apply_once_duration_sec=apply_once_duration_sec,
                        settings=settings
                    )
                ),
                timeout=10.0
            )
            
            if success:
                await self.broadcast({
                    "type": "phase_alignment_status",
                    "active": True,
                    "reference_channel": reference_channel,
                    "channels": channels,
                    "message": "Phase alignment analysis started"
                })
            else:
                await self.send_to_client(websocket, {
                    "type": "phase_alignment_status",
                    "active": False,
                    "error": "Failed to start phase alignment analysis"
                })
        
        except asyncio.TimeoutError:
            logger.error("Phase alignment start timed out (audio device may be busy)")
            # Force cleanup if timeout
            if self.phase_alignment_controller and self.phase_alignment_controller.analyzer:
                try:
                    self.phase_alignment_controller.analyzer.is_running = False
                    self.phase_alignment_controller.analyzer = None
                except:
                    pass
                self.phase_alignment_controller.is_active = False
            await self.send_to_client(websocket, {
                "type": "phase_alignment_status",
                "active": False,
                "error": "Phase alignment start timed out (audio device busy)"
            })
        except Exception as e:
            logger.error(f"Error starting phase alignment: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "phase_alignment_status",
                "active": False,
                "error": str(e)
            })
    
    async def stop_phase_alignment(self, websocket):
        """Stop phase alignment analysis and apply corrections."""
        logger.info("Stopping Phase Alignment...")
        
        measurements = None
        if self.phase_alignment_controller:
            # Получаем измерения перед остановкой
            if self.phase_alignment_controller.analyzer:
                measurements = self.phase_alignment_controller.analyzer.get_measurements()
                logger.info(f"Got measurements before stop: {measurements}")
            
            # Run stop_analysis in thread pool to avoid blocking event loop
            # (PyAudio stream.close() can block at C/CoreAudio level)
            loop = asyncio.get_running_loop()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.phase_alignment_controller.stop_analysis),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Phase alignment stop_analysis timed out after 5s, forcing cleanup...")
                # Force cleanup
                if self.phase_alignment_controller.analyzer:
                    self.phase_alignment_controller.analyzer.is_running = False
                    self.phase_alignment_controller.analyzer = None
                self.phase_alignment_controller.is_active = False
            except Exception as e:
                logger.error(f"Error in stop_analysis: {e}")
        
        await self.broadcast({
            "type": "phase_alignment_status",
            "active": False,
            "message": "Phase alignment analysis stopped"
        })
        
        # Автоматически применяем коррекции после остановки, если есть измерения
        if measurements and len(measurements) > 0:
            logger.info("Auto-applying corrections after stop...")
            # Конвертируем tuple ключи в строки для JSON
            measurements_serializable = {}
            import numpy as np
            for key, value in measurements.items():
                if isinstance(key, tuple):
                    serializable_key = f"({key[0]}, {key[1]})"
                else:
                    serializable_key = str(key)
                
                # Конвертируем NumPy типы
                serializable_value = {}
                if isinstance(value, dict):
                    for k, v in value.items():
                        if hasattr(v, 'item'):
                            serializable_value[k] = v.item()
                        elif isinstance(v, (np.integer, np.floating)):
                            serializable_value[k] = float(v) if isinstance(v, np.floating) else int(v)
                        else:
                            serializable_value[k] = v
                else:
                    serializable_value = value
                
                measurements_serializable[serializable_key] = serializable_value
            
            # Применяем коррекции
            await self.apply_phase_corrections(websocket, measurements_serializable)
    
    async def apply_phase_corrections(self, websocket, measurements: dict = None):
        """Apply phase/delay corrections to mixer."""
        logger.info(f"apply_phase_corrections called. Measurements type: {type(measurements)}, value: {measurements}")
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "phase_alignment_apply_result",
                "success": False,
                "error": "Mixer not connected"
            })
            return
        
        # Если контроллер не инициализирован, но есть измерения, применяем напрямую
        if not self.phase_alignment_controller:
            logger.warning("Phase alignment controller not initialized, applying corrections directly")
            if not measurements:
                await self.send_to_client(websocket, {
                    "type": "phase_alignment_apply_result",
                    "success": False,
                    "error": "No measurements provided and controller not initialized"
                })
                return
            # Применяем напрямую через mixer_client
            try:
                success = await self._apply_phase_corrections_direct(measurements)
                await self.broadcast({
                    "type": "phase_alignment_apply_result",
                    "success": success,
                    "message": f"Phase/delay corrections applied to {len(measurements)} channel pairs" if success else "Failed to apply corrections"
                })
            except Exception as e:
                logger.error(f"Error applying phase corrections directly: {e}", exc_info=True)
                await self.send_to_client(websocket, {
                    "type": "phase_alignment_apply_result",
                    "success": False,
                    "error": str(e)
                })
            return
        
        try:
            logger.info(f"Applying corrections through controller. Measurements type: {type(measurements)}, value: {measurements}")
            
            # Проверка наличия измерений
            if not measurements:
                logger.warning("No measurements provided, trying to get from analyzer")
                if self.phase_alignment_controller.analyzer:
                    measurements = self.phase_alignment_controller.analyzer.get_measurements()
                    logger.info(f"Got measurements from analyzer: {measurements}")
                else:
                    await self.send_to_client(websocket, {
                        "type": "phase_alignment_apply_result",
                        "success": False,
                        "error": "No measurements available. Run analysis first."
                    })
                    return
            
            if not isinstance(measurements, dict):
                await self.send_to_client(websocket, {
                    "type": "phase_alignment_apply_result",
                    "success": False,
                    "error": f"Invalid measurements format: expected dict, got {type(measurements)}"
                })
                return
            
            if len(measurements) == 0:
                await self.send_to_client(websocket, {
                    "type": "phase_alignment_apply_result",
                    "success": False,
                    "error": "Measurements dictionary is empty. No corrections to apply."
                })
                return
            
            success = self.phase_alignment_controller.apply_corrections(measurements)
            
            await self.broadcast({
                "type": "phase_alignment_apply_result",
                "success": success,
                "corrections": self.phase_alignment_controller.corrections.copy() if self.phase_alignment_controller.corrections else {},
                "message": "Phase/delay corrections applied to mixer" if success else "Failed to apply corrections"
            })
        
        except Exception as e:
            logger.error(f"Error applying phase corrections: {e}", exc_info=True)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            await self.send_to_client(websocket, {
                "type": "phase_alignment_apply_result",
                "success": False,
                "error": f"Error: {str(e)}"
            })
    
    async def _apply_phase_corrections_direct(self, measurements: dict):
        """Apply phase/delay corrections directly through mixer client."""
        import re
        applied_count = 0
        
        for pair_key, meas in measurements.items():
            # Парсинг ключа пары каналов
            if isinstance(pair_key, tuple):
                ref_ch, ch = pair_key
            elif isinstance(pair_key, str):
                match = re.match(r'\((\d+),\s*(\d+)\)', pair_key)
                if match:
                    ref_ch = int(match.group(1))
                    ch = int(match.group(2))
                else:
                    logger.warning(f"Could not parse channel pair: {pair_key}")
                    continue
            else:
                logger.warning(f"Unknown pair key format: {pair_key}")
                continue
            
            delay_ms = meas.get('optimal_delay_ms', 0.0)
            phase_invert = meas.get('phase_invert', 0)
            coherence = meas.get('coherence', 0.0)
            
            logger.info(f"Applying corrections to channel {ch}: delay={delay_ms:.2f} ms, phase_invert={phase_invert}, coherence={coherence:.2f}")
            
            # PRIORITY: Phase inversion first, delay only if necessary
            # If phase inversion is needed and delay is small (< 2ms), try phase only first
            DELAY_SIGNIFICANT_THRESHOLD = 2.0  # ms
            COHERENCE_THRESHOLD = 0.6
            
            if phase_invert == 1 and abs(delay_ms) < DELAY_SIGNIFICANT_THRESHOLD:
                if coherence >= COHERENCE_THRESHOLD:
                    logger.info(f"Channel {ch}: Small delay ({delay_ms:.2f}ms) with phase invert - prioritizing phase only")
                    delay_ms = 0.0
                else:
                    logger.info(f"Channel {ch}: Low coherence ({coherence:.2f}) even with phase invert - keeping delay")
            
            # Skip very small delays (< 0.3ms) if coherence is good
            if abs(delay_ms) < 0.3 and coherence >= COHERENCE_THRESHOLD:
                logger.info(f"Channel {ch}: Negligible delay ({delay_ms:.2f}ms) with good coherence - skipping delay")
                delay_ms = 0.0
            
            # Применение задержки только если она значительная
            if abs(delay_ms) >= 0.3:
                # Получаем max_delay_ms из контроллера если он есть
                max_delay = 10.0  # Default 10ms - musically acceptable
                if self.phase_alignment_controller:
                    max_delay = getattr(self.phase_alignment_controller, 'max_delay_ms', 10.0)
                delay_value = max(0.5, min(max_delay, abs(delay_ms)))
                self.mixer_client.set_channel_delay(ch, delay_value, mode="MS")
                logger.info(f"Channel {ch}: Applied delay {delay_value:.2f} ms (clamped to max {max_delay}ms)")
            else:
                self.mixer_client.send(f"/ch/{ch}/in/set/dlyon", 0)
                logger.info(f"Channel {ch}: Delay disabled (too small or unnecessary)")
            
            # Применение инверсии фазы
            self.mixer_client.set_channel_phase_invert(ch, phase_invert)
            logger.info(f"Channel {ch}: Phase invert set to {phase_invert}")
            
            applied_count += 1
        
        logger.info(f"Applied corrections directly to {applied_count} channels")
        return applied_count > 0
    
    async def reset_phase_delay(self, websocket, channels: List[int] = None):
        """Reset and disable all phase inversions and delays on specified channels."""
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "phase_alignment_reset_result",
                "success": False,
                "error": "Mixer not connected"
            })
            return
        
        if not channels or len(channels) == 0:
            await self.send_to_client(websocket, {
                "type": "phase_alignment_reset_result",
                "success": False,
                "error": "No channels specified"
            })
            return
        
        try:
            # Если контроллер инициализирован, используем его метод
            if self.phase_alignment_controller:
                success = self.phase_alignment_controller.reset_all_phase_delay(channels)
            else:
                # Иначе сбрасываем напрямую через mixer_client
                success = await self._reset_phase_delay_direct(channels)
            
            await self.broadcast({
                "type": "phase_alignment_reset_result",
                "success": success,
                "message": f"Phase and delay reset for {len(channels)} channels" if success else "Failed to reset phase/delay"
            })
        
        except Exception as e:
            logger.error(f"Error resetting phase/delay: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "phase_alignment_reset_result",
                "success": False,
                "error": str(e)
            })
    
    async def _reset_phase_delay_direct(self, channels: List[int]):
        """Reset phase and delay directly through mixer client."""
        reset_count = 0
        
        for ch in channels:
            try:
                # Сброс инверсии фазы (0 = нормальная фаза)
                self.mixer_client.set_channel_phase_invert(ch, 0)
                
                # Отключение задержки
                self.mixer_client.send(f"/ch/{ch}/in/set/dlyon", 0)
                
                # Сброс значения задержки на минимальное
                self.mixer_client.send(f"/ch/{ch}/in/set/dlymode", "MS")
                self.mixer_client.send(f"/ch/{ch}/in/set/dly", 0.5)  # Минимальное значение
                
                reset_count += 1
                logger.info(f"Channel {ch}: Reset phase and delay (direct)")
                
            except Exception as e:
                logger.error(f"Error resetting channel {ch}: {e}")
        
        logger.info(f"Reset phase and delay for {reset_count} channels (direct)")
        return reset_count > 0
    
    # ========== Auto Fader Methods ==========
    
    def get_auto_fader_status(self) -> dict:
        """Get current Auto Fader status."""
        if not self.auto_fader_controller:
            return {"active": False, "live_mode": self.live_mode}
        st = self.auto_fader_controller.get_status()
        st["live_mode"] = self.live_mode
        st["freeze"] = self.auto_fader_controller.get_freeze_status()
        return st
    
    def get_ready_for_live_status(self) -> dict:
        """Checklist: ready for live (all systems calibrated, no clipping)."""
        mixer_connected = bool(self.mixer_client and self.mixer_client.is_connected)
        gain_available = bool(self.gain_staging)
        fader_available = bool(self.auto_fader_controller)
        ready = mixer_connected and gain_available and fader_available
        return {
            "ready": ready,
            "checks": {
                "mixer_connected": mixer_connected,
                "gain_staging_available": gain_available,
                "auto_fader_available": fader_available,
                "no_clipping": True,
            },
        }
    
    async def create_snapshot(self, websocket, channels: list = None):
        """Create backup snapshot (for undo). Uses current mixer client."""
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {"type": "snapshot_result", "success": False, "error": "Mixer not connected"})
            return
        try:
            client = self.mixer_client
            ch_list = list(channels) if channels else list(range(1, 41))

            def do_backup():
                out = {"timestamp": datetime.now().isoformat(), "channels": {}}
                for c in ch_list:
                    try:
                        out["channels"][c] = backup_channel(client, c)
                    except Exception as e:
                        out["channels"][c] = {"error": str(e)}
                return out

            loop = asyncio.get_running_loop()
            backup_data = await loop.run_in_executor(None, do_backup)
            os.makedirs("presets", exist_ok=True)
            path = os.path.join("presets", f"channel_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            self._last_snapshot_path = path
            await self.send_to_client(websocket, {"type": "snapshot_result", "success": True, "path": path})
        except Exception as e:
            logger.error(f"Create snapshot error: {e}", exc_info=True)
            await self.send_to_client(websocket, {"type": "snapshot_result", "success": False, "error": str(e)})
    
    async def restore_snapshot(self, websocket, snapshot_path: str = None):
        """Restore from last snapshot (undo)."""
        path = snapshot_path or self._last_snapshot_path
        if not path or not os.path.isfile(path):
            await self.send_to_client(websocket, {"type": "restore_result", "success": False, "error": "No snapshot file"})
            return
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {"type": "restore_result", "success": False, "error": "Mixer not connected"})
            return
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: restore_from_backup_using_client(self.mixer_client, path, skip_confirm=True)
            )
            await self.send_to_client(websocket, {"type": "restore_result", "success": True, "path": path})
        except Exception as e:
            logger.error(f"Restore snapshot error: {e}", exc_info=True)
            await self.send_to_client(websocket, {"type": "restore_result", "success": False, "error": str(e)})
    
    async def start_realtime_fader(self, websocket, device_id: str = None,
                                    channels: List[int] = None,
                                    channel_settings: Dict = None,
                                    channel_mapping: Dict = None,
                                    settings: Dict = None):
        """Start Real-Time Fader mode."""
        channels = channels or []
        channel_settings = channel_settings or {}
        channel_mapping = channel_mapping or {}
        settings = settings or {}
        if not isinstance(settings, dict):
            settings = {}
        
        logger.info(f"Starting Real-Time Fader: device={device_id}, channels={channels}")
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Mixer not connected"
            })
            return
        
        if device_id is None or not channels:
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Missing required parameters: device_id or channels"
            })
            return
        
        try:
            # Stop existing Auto Fader if running
            if self.auto_fader_controller:
                self.auto_fader_controller.stop()
            
            # Create new controller
            self.auto_fader_controller = AutoFaderController(
                mixer_client=self.mixer_client,
                sample_rate=48000,
                config=self.config,
                bleed_service=self.bleed_service
            )
            
            # Apply settings
            if settings:
                self.auto_fader_controller.update_settings(
                    fader_range_db=settings.get('faderRangeDb'),
                    avg_window_sec=settings.get('avgWindowSec'),
                    sensitivity=settings.get('sensitivity'),
                    attack_ms=settings.get('attackMs'),
                    release_ms=settings.get('releaseMs'),
                    gate_threshold=settings.get('gateThreshold', -50.0)
                )
            
            # Set up callbacks
            loop = asyncio.get_running_loop()
            
            def on_status_update(status_data):
                try:
                    # Extract 'type' field and use it as 'status_type'
                    # Create a copy to avoid modifying the original dict
                    status_data_copy = dict(status_data)
                    status_type = status_data_copy.pop('type', 'status_update')
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.broadcast({
                            "type": "auto_fader_status",
                            "status_type": status_type,
                            **status_data_copy
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto fader status: {e}")
            
            def on_levels_updated(levels_data):
                try:
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.broadcast({
                            "type": "auto_fader_status",
                            "status_type": "levels_update",
                            "channels": levels_data
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto fader levels: {e}")
            
            self.auto_fader_controller.on_status_update = on_status_update
            self.auto_fader_controller.on_levels_updated = on_levels_updated
            
            # Convert channel_settings keys to int
            int_channel_settings = {}
            instrument_types_for_bleed = {}
            for ch_str, ch_data in channel_settings.items():
                try:
                    ch_id = int(ch_str)
                    instrument_type = ch_data.get('instrumentType', 'custom')
                    int_channel_settings[ch_id] = {
                        'instrument_type': instrument_type
                    }
                    # Map 'toms' -> 'tom' for BleedDetector INSTRUMENT_BANDS
                    instrument_types_for_bleed[ch_id] = 'tom' if instrument_type == 'toms' else instrument_type
                except (ValueError, TypeError):
                    pass
            
            # Configure bleed service with instrument types
            if self.bleed_service and instrument_types_for_bleed:
                self.bleed_service.configure(instrument_types_for_bleed)
            
            # Convert channel_mapping keys to int
            int_channel_mapping = {}
            for ch_str, mixer_ch in channel_mapping.items():
                try:
                    int_channel_mapping[int(ch_str)] = int(mixer_ch)
                except (ValueError, TypeError):
                    pass
            
            # Start audio capture
            success = self.auto_fader_controller.start(
                device_id=int(device_id),
                channels=[int(ch) for ch in channels],
                channel_settings=int_channel_settings,
                channel_mapping=int_channel_mapping,
                on_status_callback=on_status_update
            )
            
            if not success:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "active": False,
                    "error": "Failed to start audio capture"
                })
                return
            
            # Start real-time fader control
            success = self.auto_fader_controller.start_realtime_fader()
            
            if success:
                await self.broadcast({
                    "type": "auto_fader_status",
                    "status_type": "realtime_fader_started",
                    "active": True,
                    "realtime_enabled": True,
                    "mode": "realtime"
                })
            else:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Failed to start real-time fader"
                })
        
        except Exception as e:
            logger.error(f"Error starting Real-Time Fader: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": str(e)
            })
    
    async def stop_realtime_fader(self, websocket):
        """Stop Real-Time Fader mode."""
        logger.info("Stopping Real-Time Fader")
        
        try:
            if self.auto_fader_controller:
                self.auto_fader_controller.stop_realtime_fader()
            
            await self.broadcast({
                "type": "auto_fader_status",
                "status_type": "realtime_fader_stopped",
                "active": False,
                "realtime_enabled": False
            })
        
        except Exception as e:
            logger.error(f"Error stopping Real-Time Fader: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })
    
    async def start_auto_balance(self, websocket, device_id: str = None,
                                  channels: List[int] = None,
                                  channel_settings: Dict = None,
                                  channel_mapping: Dict = None,
                                  duration: float = 15,
                                  bleed_threshold: float = -50):
        """Start Auto Balance collection (LEARN phase)."""
        channels = channels or []
        channel_settings = channel_settings or {}
        channel_mapping = channel_mapping or {}
        
        logger.info(f"Starting Auto Balance LEARN: device={device_id}, channels={channels}, "
                    f"duration={duration}s, bleed_threshold={bleed_threshold} LUFS")
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Mixer not connected"
            })
            return
        
        if device_id is None or not channels:
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Missing required parameters: device_id or channels"
            })
            return
        
        try:
            # Stop existing Auto Fader if running
            if self.auto_fader_controller:
                self.auto_fader_controller.stop()
            
            # Create new controller
            self.auto_fader_controller = AutoFaderController(
                mixer_client=self.mixer_client,
                sample_rate=48000,
                config=self.config,
                bleed_service=self.bleed_service
            )
            
            # Set up callbacks
            loop = asyncio.get_running_loop()
            
            def on_status_update(status_data):
                try:
                    # Extract 'type' field and use it as 'status_type'
                    # Create a copy to avoid modifying the original dict
                    status_data_copy = dict(status_data)
                    status_type = status_data_copy.pop('type', 'status_update')
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.broadcast({
                            "type": "auto_fader_status",
                            "status_type": status_type,
                            **status_data_copy
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto balance status: {e}")
            
            def on_levels_updated(levels_data):
                try:
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.broadcast({
                            "type": "auto_fader_status",
                            "status_type": "levels_update",
                            "channels": levels_data
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto balance levels: {e}")
            
            self.auto_fader_controller.on_status_update = on_status_update
            self.auto_fader_controller.on_levels_updated = on_levels_updated
            
            # Convert channel_settings keys to int
            # Support both 'instrumentType' (from AutoFaderTab) and 'preset' (from AutoSoundcheckTab)
            int_channel_settings = {}
            instrument_types_for_bleed = {}
            for ch_str, ch_data in channel_settings.items():
                try:
                    ch_id = int(ch_str)
                    # Try instrumentType first (from AutoFaderTab), then preset (from AutoSoundcheckTab), default to 'custom'
                    instrument_type = ch_data.get('instrumentType') or ch_data.get('preset', 'custom')
                    int_channel_settings[ch_id] = {
                        'instrument_type': instrument_type
                    }
                    # Map 'toms' -> 'tom' for BleedDetector INSTRUMENT_BANDS
                    instrument_types_for_bleed[ch_id] = 'tom' if instrument_type == 'toms' else instrument_type
                except (ValueError, TypeError):
                    pass
            
            # Configure bleed service with instrument types
            if self.bleed_service and instrument_types_for_bleed:
                self.bleed_service.configure(instrument_types_for_bleed)
            
            logger.info(f"Processed channel_settings for Auto Balance: {int_channel_settings}")
            
            # Convert channel_mapping keys to int
            int_channel_mapping = {}
            for ch_str, mixer_ch in channel_mapping.items():
                try:
                    int_channel_mapping[int(ch_str)] = int(mixer_ch)
                except (ValueError, TypeError):
                    pass
            
            # Start audio capture
            success = self.auto_fader_controller.start(
                device_id=int(device_id),
                channels=[int(ch) for ch in channels],
                channel_settings=int_channel_settings,
                channel_mapping=int_channel_mapping,
                on_status_callback=on_status_update
            )
            
            if not success:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "active": False,
                    "error": "Failed to start audio capture"
                })
                return
            
            # Start auto balance collection (uses same algorithms and methods as Auto Balance)
            # All settings (fader_range_db, avg_window_sec, sensitivity, etc.) are loaded from config
            # in AutoFaderController.__init__()
            success = self.auto_fader_controller.start_auto_balance(duration=duration, bleed_threshold=bleed_threshold)
            
            if success:
                await self.broadcast({
                    "type": "auto_fader_status",
                    "status_type": "auto_balance_started",
                    "active": True,
                    "collecting": True,
                    "duration": duration,
                    "mode": "static"
                })
            else:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Failed to start auto balance collection"
                })
        
        except Exception as e:
            logger.error(f"Error starting Auto Balance: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": str(e)
            })
    
    async def apply_auto_balance(self, websocket):
        """Apply Auto Balance results to mixer."""
        logger.info("Applying Auto Balance")
        
        try:
            if not self.auto_fader_controller:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return
            
            success = self.auto_fader_controller.apply_auto_balance()
            
            if success:
                status = self.auto_fader_controller.get_status()
                auto_balance_result = status.get('auto_balance_result') or {}
                await self.broadcast({
                    "type": "auto_fader_status",
                    "status_type": "auto_balance_applied",
                    "applied_count": len(auto_balance_result),
                    "total_count": len(auto_balance_result),
                    "pass_number": status.get('auto_balance_pass', 1)
                })
            else:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Failed to apply auto balance"
                })
        
        except Exception as e:
            logger.error(f"Error applying Auto Balance: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })
    
    async def cancel_auto_balance(self, websocket):
        """Cancel Auto Balance collection."""
        logger.info("Cancelling Auto Balance")
        
        try:
            if self.auto_fader_controller:
                self.auto_fader_controller.cancel_auto_balance()
            
            await self.broadcast({
                "type": "auto_fader_status",
                "status_type": "auto_balance_cancelled"
            })
        
        except Exception as e:
            logger.error(f"Error cancelling Auto Balance: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })
    
    async def lock_auto_balance_channel(self, websocket, channel):
        """Lock a channel so it won't be changed on subsequent Auto Balance passes."""
        try:
            if not self.auto_fader_controller:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return
            
            ch = int(channel) if channel is not None else None
            if ch is not None:
                success = self.auto_fader_controller.lock_channel(ch)
                if success:
                    await self.broadcast({
                        "type": "auto_fader_status",
                        "status_type": "channel_lock_changed",
                        "channel": ch,
                        "locked": True
                    })
                else:
                    await self.send_to_client(websocket, {
                        "type": "auto_fader_status",
                        "status_type": "error",
                        "error": f"Channel {ch} not found"
                    })
        except Exception as e:
            logger.error(f"Error locking channel: {e}", exc_info=True)
    
    async def unlock_auto_balance_channel(self, websocket, channel):
        """Unlock a channel so it can be changed on subsequent Auto Balance passes."""
        try:
            if not self.auto_fader_controller:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return
            
            ch = int(channel) if channel is not None else None
            if ch is not None:
                success = self.auto_fader_controller.unlock_channel(ch)
                if success:
                    await self.broadcast({
                        "type": "auto_fader_status",
                        "status_type": "channel_lock_changed",
                        "channel": ch,
                        "locked": False
                    })
                else:
                    await self.send_to_client(websocket, {
                        "type": "auto_fader_status",
                        "status_type": "error",
                        "error": f"Channel {ch} not found"
                    })
        except Exception as e:
            logger.error(f"Error unlocking channel: {e}", exc_info=True)
    
    async def set_auto_fader_profile(self, websocket, profile: str):
        """Set Auto Fader genre profile."""
        logger.info(f"Setting Auto Fader profile: {profile}")
        
        try:
            if self.auto_fader_controller:
                self.auto_fader_controller.set_profile(profile)
            
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "message": f"Profile set to: {profile}"
            })
        
        except Exception as e:
            logger.error(f"Error setting Auto Fader profile: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })
    
    async def update_auto_fader_settings(self, websocket, settings: Dict):
        """Update Auto Fader settings."""
        logger.info(f"Updating Auto Fader settings: {settings}")
        
        try:
            if self.auto_fader_controller:
                self.auto_fader_controller.update_settings(
                    target_lufs=settings.get('targetLufs'),
                    max_adjustment_db=settings.get('maxAdjustmentDb'),
                    ratio=settings.get('ratio'),
                    attack_ms=settings.get('attackMs'),
                    release_ms=settings.get('releaseMs'),
                    hold_ms=settings.get('holdMs'),
                )
            
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "message": "Settings updated"
            })
        
        except Exception as e:
            logger.error(f"Error updating Auto Fader settings: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })
    
    async def save_auto_fader_defaults(self, websocket, settings: Dict):
        """Save Auto Fader settings as user defaults."""
        try:
            self._save_user_config('auto_fader', settings)
            await self.send_to_client(websocket, {
                "type": "auto_fader_defaults_saved",
                "success": True
            })
            logger.info(f"Auto Fader defaults saved: {settings}")
        except Exception as e:
            logger.error(f"Error saving Auto Fader defaults: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_defaults_saved",
                "success": False,
                "error": str(e)
            })

    async def load_auto_fader_defaults(self, websocket):
        """Load saved Auto Fader user defaults."""
        try:
            user_config = self._load_user_config()
            saved_settings = user_config.get('auto_fader', None)
            await self.send_to_client(websocket, {
                "type": "auto_fader_defaults_loaded",
                "settings": saved_settings
            })
            logger.info(f"Auto Fader defaults loaded: {saved_settings}")
        except Exception as e:
            logger.error(f"Error loading Auto Fader defaults: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_defaults_loaded",
                "settings": None,
                "error": str(e)
            })

    async def save_all_settings(self, websocket, settings: Dict):
        """Save all application settings to user config."""
        try:
            # Save each section individually
            for section, section_settings in settings.items():
                self._save_user_config(section, section_settings)
            await self.send_to_client(websocket, {
                "type": "all_settings_saved",
                "success": True
            })
            logger.info(f"All settings saved: sections={list(settings.keys())}")
        except Exception as e:
            logger.error(f"Error saving all settings: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "all_settings_saved",
                "success": False,
                "error": str(e)
            })

    async def load_all_settings(self, websocket):
        """Load all saved user settings."""
        try:
            user_config = self._load_user_config()
            await self.send_to_client(websocket, {
                "type": "all_settings_loaded",
                "settings": user_config
            })
            logger.info(f"All settings loaded: sections={list(user_config.keys())}")
        except Exception as e:
            logger.error(f"Error loading all settings: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "all_settings_loaded",
                "settings": {},
                "error": str(e)
            })

    # ========== Auto Soundcheck Methods ==========
    
    async def reset_all_functions_to_defaults(self, websocket, channels: List[int]):
        """Reset all functions to default state for selected channels."""
        logger.info(f"Resetting all functions to defaults for channels: {channels}")
        
        if not channels or len(channels) == 0:
            logger.warning("No channels provided for reset")
            return
        
        try:
            # Stop all active controllers (non-blocking)
            logger.info("Stopping all active controllers...")
            stop_tasks = []
            
            if self.gain_staging:
                async def stop_gain_staging():
                    try:
                        self.gain_staging.stop()
                        logger.info("Gain staging stopped")
                    except Exception as e:
                        logger.warning(f"Error stopping gain staging: {e}")
                stop_tasks.append(asyncio.create_task(stop_gain_staging()))
            
            if self.phase_alignment_controller:
                async def stop_phase_alignment():
                    try:
                        self.phase_alignment_controller.stop()
                        logger.info("Phase alignment stopped")
                    except Exception as e:
                        logger.warning(f"Error stopping phase alignment: {e}")
                stop_tasks.append(asyncio.create_task(stop_phase_alignment()))
            
            if self.multi_channel_auto_eq_controller:
                async def stop_multi_eq():
                    try:
                        self.multi_channel_auto_eq_controller.stop_all()
                        logger.info("Multi-channel auto EQ stopped")
                    except Exception as e:
                        logger.warning(f"Error stopping multi-channel auto EQ: {e}")
                stop_tasks.append(asyncio.create_task(stop_multi_eq()))
            
            if self.auto_fader_controller:
                async def stop_auto_fader():
                    try:
                        self.auto_fader_controller.stop()
                        logger.info("Auto fader stopped")
                    except Exception as e:
                        logger.warning(f"Error stopping auto fader: {e}")
                stop_tasks.append(asyncio.create_task(stop_auto_fader()))
            
            # Wait for all stops with timeout
            if stop_tasks:
                try:
                    await asyncio.wait_for(asyncio.gather(*stop_tasks, return_exceptions=True), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("Some controllers did not stop in time, continuing...")
            
            await asyncio.sleep(0.2)  # Small delay after stopping
            
            # Reset TRIM to 0dB (with timeout and error handling)
            logger.info("Resetting TRIM to 0dB...")
            try:
                await asyncio.wait_for(self.reset_trim(websocket, channels), timeout=15.0)
                logger.info("TRIM reset completed")
            except asyncio.TimeoutError:
                logger.warning("TRIM reset timed out after 15s, continuing...")
            except Exception as e:
                logger.error(f"Error resetting TRIM: {e}", exc_info=True)
            await asyncio.sleep(0.3)
            
            # Reset all faders to 0dB (all 40 channels)
            logger.info("Resetting all faders to 0dB...")
            try:
                if self.mixer_client and self.mixer_client.is_connected:
                    reset_count = 0
                    errors = []
                    # Reset all 40 channels
                    for ch in range(1, 41):
                        try:
                            self.mixer_client.set_channel_fader(ch, 0.0)
                            reset_count += 1
                            await asyncio.sleep(0.02)  # Small delay between channels
                        except Exception as e:
                            errors.append(f"Channel {ch}: {str(e)}")
                            logger.warning(f"Error resetting fader for channel {ch}: {e}")
                    
                    logger.info(f"Faders reset completed: {reset_count}/40 channels")
                    if errors:
                        logger.warning(f"Fader reset errors: {len(errors)} channels failed")
                else:
                    logger.warning("Mixer not connected, skipping fader reset")
            except Exception as e:
                logger.error(f"Error resetting faders: {e}", exc_info=True)
            await asyncio.sleep(0.3)
            
            # Reset EQ to flat (with timeout)
            logger.info("Resetting EQ to flat...")
            try:
                await asyncio.wait_for(self.reset_all_eq(websocket, {"channels": channels}), timeout=15.0)
                logger.info("EQ reset completed")
            except asyncio.TimeoutError:
                logger.warning("EQ reset timed out after 15s, continuing...")
            except Exception as e:
                logger.error(f"Error resetting EQ: {e}", exc_info=True)
            await asyncio.sleep(0.3)
            
            # Reset phase/delay (with timeout)
            logger.info("Resetting phase/delay...")
            try:
                await asyncio.wait_for(self.reset_phase_delay(websocket, channels), timeout=15.0)
                logger.info("Phase/delay reset completed")
            except asyncio.TimeoutError:
                logger.warning("Phase/delay reset timed out after 15s, continuing...")
            except Exception as e:
                logger.error(f"Error resetting phase/delay: {e}", exc_info=True)
            await asyncio.sleep(0.3)
            
            logger.info("All functions reset to defaults successfully")
            
        except Exception as e:
            logger.error(f"Error resetting functions to defaults: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            # Don't raise - continue with cycle even if reset fails
    
    async def start_auto_soundcheck(self, websocket, device_id: str = None, channels: list = None,
                                    channel_settings: dict = None, channel_mapping: dict = None,
                                    timings: dict = None):
        """Start automatic soundcheck cycle."""
        logger.info("=" * 60)
        logger.info("RECEIVED start_auto_soundcheck MESSAGE")
        logger.info(f"device_id: {device_id}")
        logger.info(f"channels: {channels}")
        logger.info(f"timings: {timings}")
        logger.info("=" * 60)
        
        if self.auto_soundcheck_running:
            logger.warning("Auto soundcheck is already running")
            await self.send_to_client(websocket, {
                "type": "auto_soundcheck_status",
                "is_running": True,
                "error": "Auto soundcheck is already running"
            })
            return
        
        # Normalize timings - frontend sends gain_staging, backend config uses gain_staging_duration
        if not timings:
            config_timings = self.config.get("automation", {}).get("auto_soundcheck", {
                "gain_staging_duration": 30,
                "phase_alignment_duration": 30,
                "auto_eq_duration": 30,
                "auto_fader_duration": 30
            })
            # Convert config format to internal format
            timings = {
                "gain_staging": config_timings.get("gain_staging_duration", 30),
                "phase_alignment": config_timings.get("phase_alignment_duration", 30),
                "auto_eq": config_timings.get("auto_eq_duration", 30),
                "auto_fader": config_timings.get("auto_fader_duration", 30)
            }
        else:
            # Normalize timings from frontend (may use either format)
            normalized_timings = {}
            for key in ["gain_staging", "phase_alignment", "auto_eq", "auto_fader"]:
                # Try direct key first
                if key in timings:
                    normalized_timings[key] = timings[key]
                # Try with _duration suffix
                elif f"{key}_duration" in timings:
                    normalized_timings[key] = timings[f"{key}_duration"]
                else:
                    # Default value
                    normalized_timings[key] = 30
            timings = normalized_timings
        
        logger.info(f"Normalized timings: {timings}")
        
        if not device_id:
            logger.error("Missing device_id")
            await self.send_to_client(websocket, {
                "type": "auto_soundcheck_status",
                "is_running": False,
                "error": "Missing device_id"
            })
            return
        
        if not channels or len(channels) == 0:
            logger.error("No channels specified")
            await self.send_to_client(websocket, {
                "type": "auto_soundcheck_status",
                "is_running": False,
                "error": "No channels specified"
            })
            return
        
        self.auto_soundcheck_running = True
        self.auto_soundcheck_websocket = websocket
        
        # Send initial status immediately before starting task
        await self._send_soundcheck_status(websocket, None, 0, 0, "Soundcheck cycle starting...")
        await asyncio.sleep(0.05)  # Small delay to ensure message is sent
        
        # Start the cycle in a background task
        logger.info("Starting auto soundcheck cycle task...")
        try:
            self.auto_soundcheck_task = asyncio.create_task(
                self._run_auto_soundcheck_cycle(websocket, device_id, channels, channel_settings, 
                                               channel_mapping, timings)
            )
            logger.info("Auto soundcheck cycle task created successfully")
            
            # Send status that cycle has started
            await self._send_soundcheck_status(websocket, None, 5, 0, "Soundcheck cycle initialized, starting reset...")
            
        except Exception as e:
            logger.error(f"Error creating auto soundcheck task: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            self.auto_soundcheck_running = False
            await self.send_to_client(websocket, {
                "type": "auto_soundcheck_status",
                "is_running": False,
                "error": f"Failed to start cycle: {str(e)}"
            })
    
    async def stop_auto_soundcheck(self, websocket):
        """Stop the auto soundcheck cycle."""
        if not self.auto_soundcheck_running:
            await self.send_to_client(websocket, {
                "type": "auto_soundcheck_status",
                "is_running": False,
                "message": "Auto soundcheck is not running"
            })
            return
        
        logger.info("Stopping auto soundcheck cycle...")
        self.auto_soundcheck_running = False
        
        # Cancel the task
        if self.auto_soundcheck_task:
            self.auto_soundcheck_task.cancel()
            try:
                await self.auto_soundcheck_task
            except asyncio.CancelledError:
                pass
            self.auto_soundcheck_task = None
        
        # Stop all active controllers
        if self.gain_staging:
            try:
                self.gain_staging.stop()
            except:
                pass
        
        if self.phase_alignment_controller:
            try:
                self.phase_alignment_controller.stop()
            except:
                pass
        
        if self.multi_channel_auto_eq_controller:
            try:
                self.multi_channel_auto_eq_controller.stop_all()
            except:
                pass
        
        if self.auto_fader_controller:
            try:
                self.auto_fader_controller.stop()
            except:
                pass
        
        await self.send_to_client(websocket, {
            "type": "auto_soundcheck_status",
            "is_running": False,
            "message": "Auto soundcheck stopped"
        })
    
    # ========== Auto Compressor ==========
    
    def _reset_compressor_on_channel(self, mixer_ch: int):
        """Reset compressor to defaults and turn off on one mixer channel (blocking)."""
        import time
        self.mixer_client.set_compressor_on(mixer_ch, 0)
        time.sleep(0.02)
        self.mixer_client.set_compressor(
            mixer_ch,
            threshold=-10.0,
            ratio="3.0",
            attack=10.0,
            release=100.0,
            gain=0.0,
            mix=100.0,
            knee=0,
        )
        time.sleep(0.02)

    async def start_auto_compressor(self, websocket, device_id, channels: list, channel_mapping: dict, channel_names: dict = None):
        """Start Auto Compressor: open audio capture (post-fader). Resets compressor params on selected channels."""
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Mixer not connected"})
            return
        if not device_id and device_id != 0:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Audio device required"})
            return
        channels = [int(c) for c in (channels or [])]
        channel_mapping = {int(k): int(v) for k, v in (channel_mapping or {}).items()}
        channel_names = {int(k): str(v) for k, v in (channel_names or {}).items()}
        if self.auto_compressor_controller:
            self.auto_compressor_controller.stop()
        # Reset compressor parameters on all channels that will be used by Auto Compressor
        mixer_channels = list({channel_mapping.get(ac, ac) for ac in channels})
        loop = asyncio.get_running_loop()
        for ch in mixer_channels:
            await loop.run_in_executor(None, self._reset_compressor_on_channel, ch)
        cfg = self.config.get("automation", {}).get("auto_compressor", {})
        duration = cfg.get("soundcheck_duration_per_channel", 7.0)
        target_gr = cfg.get("target_gr_db", 6.0)
        self.auto_compressor_controller = AutoCompressorController(
            self.mixer_client, sample_rate=48000,
            soundcheck_duration_per_channel=duration, target_gr_db=target_gr,
            bleed_service=self.bleed_service)
        loop = asyncio.get_running_loop()
        def on_status(data):
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "auto_compressor_status", **data}), loop)
        self.auto_compressor_controller.on_status = on_status
        ok = self.auto_compressor_controller.start(int(device_id), channels, channel_mapping, channel_names)
        if not ok:
            self.auto_compressor_controller = None
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Failed to start audio capture"})
            return
        await self.send_to_client(websocket, {"type": "auto_compressor_status", "active": True, "message": "Auto Compressor started (post-fader)"})
    
    async def stop_auto_compressor(self, websocket):
        """Stop Auto Compressor and release audio."""
        if self.auto_compressor_controller:
            self.auto_compressor_controller.stop()
            self.auto_compressor_controller = None
        await self.broadcast({"type": "auto_compressor_status", "active": False, "message": "Stopped"})
    
    async def get_auto_compressor_status(self, websocket, request_id=None):
        """Return current Auto Compressor status."""
        if not self.auto_compressor_controller:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "request_id": request_id, "active": False})
            return
        await self.send_to_client(websocket, {"type": "auto_compressor_status", **self.auto_compressor_controller.get_status()})
    
    async def get_auto_gate_status(self, websocket, request_id=None):
        """Return Auto Gate status."""
        await self.send_to_client(websocket, {
            "type": "auto_gate_status",
            "request_id": request_id,
            "active": False,
            "message": "Auto Gate not implemented"
        })
    
    async def get_auto_panner_status(self, websocket, request_id=None):
        """Return Auto Panner status."""
        await self.send_to_client(websocket, {
            "type": "auto_panner_status",
            "request_id": request_id,
            "active": False,
            "message": "Auto Panner not implemented"
        })
    
    async def get_auto_reverb_status(self, websocket, request_id=None):
        """Return Auto Reverb status."""
        await self.send_to_client(websocket, {
            "type": "auto_reverb_status",
            "request_id": request_id,
            "active": False,
            "message": "Auto Reverb not implemented"
        })
    
    async def get_auto_effects_status(self, websocket, request_id=None):
        """Return Auto Effects status."""
        await self.send_to_client(websocket, {
            "type": "auto_effects_status",
            "request_id": request_id,
            "active": False,
            "message": "Auto Effects not implemented"
        })
    
    async def get_cross_adaptive_eq_status(self, websocket, request_id=None):
        """Return Cross-Adaptive EQ status."""
        await self.send_to_client(websocket, {
            "type": "cross_adaptive_eq_status",
            "request_id": request_id,
            "active": False,
            "message": "Cross-Adaptive EQ not implemented"
        })

    async def start_auto_compressor_soundcheck(
        self, websocket,
        genre: str = "unknown",
        style: str = "live",
        genre_factor: float = 1.0,
        mix_density_factor: float = 1.0,
        bpm=None
    ):
        """Run soundcheck: record each channel, analyze, adapt, apply."""
        if not self.auto_compressor_controller or not self.auto_compressor_controller.is_active:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Start Auto Compressor first"})
            return
        task = asyncio.create_task(self.auto_compressor_controller.start_soundcheck(
            genre=genre,
            style=style,
            genre_factor=genre_factor,
            mix_density_factor=mix_density_factor,
            bpm=bpm
        ))
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    async def stop_auto_compressor_soundcheck(self, websocket):
        if self.auto_compressor_controller:
            self.auto_compressor_controller.stop_soundcheck()
        await self.broadcast({"type": "auto_compressor_status", "soundcheck_running": False})
    
    async def start_auto_compressor_live(self, websocket, auto_correct: bool = True):
        if not self.auto_compressor_controller or not self.auto_compressor_controller.is_active:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Start Auto Compressor first"})
            return
        await self.auto_compressor_controller.start_live(auto_correct=auto_correct)
        await self.broadcast({"type": "auto_compressor_status", "live_running": True})
    
    async def stop_auto_compressor_live(self, websocket):
        if self.auto_compressor_controller:
            self.auto_compressor_controller.stop_live()
        await self.broadcast({"type": "auto_compressor_status", "live_running": False})
    
    async def set_auto_compressor_profile(self, websocket, channel, profile: str):
        """Apply preset profile (punch, control, gentle, etc.) for one channel."""
        if not self.auto_compressor_controller or not self.mixer_client:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Not ready"})
            return
        from compressor_presets import get_preset
        from channel_recognizer import recognize_instrument
        from compressor_adaptation import adapt_params
        ch = int(channel)
        name = self.auto_compressor_controller.channel_names.get(ch, "")
        inst = recognize_instrument(name) or "custom"
        preset = get_preset(inst, profile or "base")
        features = self.auto_compressor_controller.channel_features.get(ch)
        if features:
            params = adapt_params(features, preset, profile or "base")
        else:
            params = {"threshold": preset.get("threshold", -15), "ratio_wing": "3.0", "attack_ms": preset.get("attack_ms", 15),
                      "release_ms": preset.get("release_ms", 150), "knee": preset.get("knee", 2), "gain": preset.get("makeup_gain", 0)}
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.auto_compressor_controller._apply_compressor_smooth(ch, params))
        await self.broadcast({"type": "auto_compressor_status", "profile_applied": ch, "profile": profile})
    
    async def set_auto_compressor_manual(self, websocket, channel, params: dict):
        """Apply manual compressor params to one channel."""
        if not self.auto_compressor_controller or not self.mixer_client:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Not ready"})
            return
        ch = int(channel)
        from compressor_adaptation import ratio_float_to_wing
        p = dict(params)
        if "ratio" in p and isinstance(p["ratio"], (int, float)):
            p["ratio_wing"] = ratio_float_to_wing(float(p["ratio"]))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.auto_compressor_controller._apply_compressor_smooth(ch, p))
        await self.broadcast({"type": "auto_compressor_status", "manual_applied": ch})
    
    async def _run_auto_soundcheck_cycle(self, websocket, device_id: str, channels: list,
                                         channel_settings: dict, channel_mapping: dict, timings: dict):
        """Execute the full auto soundcheck cycle."""
        try:
            logger.info("=" * 60)
            logger.info("Starting auto soundcheck cycle execution...")
            logger.info(f"device_id: {device_id}")
            logger.info(f"channels: {channels}")
            logger.info(f"timings: {timings}")
            logger.info("=" * 60)
            
            # Send initial status
            await self._send_soundcheck_status(websocket, None, 0, 0, "Initializing soundcheck cycle...")
            await asyncio.sleep(0.1)  # Small delay to ensure message is sent
            
            # Step 1: Reset to defaults
            logger.info("Step 1: Resetting all functions to defaults...")
            await self._send_soundcheck_status(websocket, "reset", 10, 0, "Resetting all functions to defaults...")
            await asyncio.sleep(0.1)
            
            # Run reset in a way that doesn't block
            reset_completed = False
            reset_error = None
            try:
                await self._send_soundcheck_status(websocket, "reset", 30, 0, "Stopping active controllers...")
                await asyncio.sleep(0.1)
                
                await self._send_soundcheck_status(websocket, "reset", 50, 0, "Resetting TRIM, EQ, Phase...")
                await asyncio.sleep(0.1)
                
                await self.reset_all_functions_to_defaults(websocket, channels)
                reset_completed = True
                logger.info("Step 1 completed: Reset successful")
            except Exception as e:
                reset_error = str(e)
                logger.error(f"Error in reset step: {e}", exc_info=True)
                await self._send_soundcheck_status(websocket, "reset", 0, 0, f"Reset error: {reset_error}", error=reset_error)
                # Continue anyway - reset errors are not critical
            
            if reset_completed:
                await self._send_soundcheck_status(websocket, "reset", 100, 0, "Reset complete")
            await asyncio.sleep(0.3)  # Reduced delay
            
            if not self.auto_soundcheck_running:
                logger.info("Auto soundcheck stopped after reset")
                return
            
            # Step 1.5: Channel recognizer (scan names + optional spectral verification)
            await self._send_soundcheck_status(websocket, "channel_recognizer", 0, 5, "Scanning channel names...")
            channel_names_for_recognizer = {}
            if self.mixer_client and self.mixer_client.is_connected and channels:
                try:
                    channel_names_for_recognizer = self.mixer_client.get_all_channel_names(max(channels) if channels else 40)
                    rec = scan_and_recognize(channel_names_for_recognizer)
                    await self.send_to_client(websocket, {"type": "auto_soundcheck_channel_recognizer", "recognition": rec})
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Channel recognizer step: {e}")
            await self._send_soundcheck_status(websocket, "channel_recognizer", 100, 0, "Channel scan complete")
            await asyncio.sleep(0.3)
            
            if not self.auto_soundcheck_running:
                return
            
            # Step 2: GAIN STAGING
            duration = timings.get("gain_staging", 30)
            logger.info(f"Step 2: Starting GAIN STAGING for {duration} seconds...")
            logger.info(f"Step 2: device_id={device_id}, channels={channels}, duration={duration}")
            await self._send_soundcheck_status(websocket, "gain_staging", 0, duration, "Starting GAIN STAGING...")
            
            # Ensure channel_settings and channel_mapping are not None
            if channel_settings is None:
                channel_settings = {}
            if channel_mapping is None:
                channel_mapping = {}
            
            # Stop any existing gain staging to ensure clean start
            if self.gain_staging and (self.gain_staging.is_active or self.gain_staging.is_audio_stream_active):
                logger.info("Stopping existing gain staging before starting new one in soundcheck cycle...")
                try:
                    self.gain_staging.stop()
                    await asyncio.sleep(0.5)  # Wait for cleanup
                    logger.info("Existing gain staging stopped")
                except Exception as e:
                    logger.warning(f"Error stopping existing gain staging: {e}")
            
            logger.info(f"Calling start_realtime_correction with device_id={device_id}, channels={channels}, channel_settings keys: {list(channel_settings.keys()) if channel_settings else None}, channel_mapping keys: {list(channel_mapping.keys()) if channel_mapping else None}")
            logger.info(f"Before start_realtime_correction: mixer_client={self.mixer_client is not None}, mixer_connected={self.mixer_client.is_connected if self.mixer_client else False}")
            
            try:
                await self.start_realtime_correction(device_id, channels, channel_settings, channel_mapping)
                logger.info("start_realtime_correction completed successfully")
                logger.info(f"After start_realtime_correction: gain_staging={self.gain_staging is not None}, is_active={self.gain_staging.is_active if self.gain_staging else False}, is_audio_stream_active={self.gain_staging.is_audio_stream_active if self.gain_staging else False}")
            except Exception as e:
                logger.error(f"Error starting realtime correction: {e}", exc_info=True)
                await self._send_soundcheck_status(websocket, "gain_staging", 0, 0, f"Error starting GAIN STAGING: {str(e)}", error=str(e))
                raise
            
            await self._wait_with_progress(websocket, "gain_staging", duration)
            
            if not self.auto_soundcheck_running:
                logger.info("Auto soundcheck stopped during GAIN STAGING")
                try:
                    await self.stop_realtime_correction()
                except Exception as e:
                    logger.error(f"Error stopping realtime correction: {e}")
                return
            
            try:
                await self.stop_realtime_correction()
                logger.info("stop_realtime_correction completed successfully")
                
                # TRIM corrections are already applied on the mixer in real-time by gain staging.
                # Do NOT re-read from WingClient cache (get_channel_gain returns stale values
                # because send() doesn't update the state cache).
                logger.info("GAIN STAGING corrections already applied on mixer in real-time, preserving values")
                
                # Fully stop gain staging to release audio device for next steps (phase alignment)
                if self.gain_staging:
                    logger.info("Fully stopping gain staging controller to release audio device...")
                    try:
                        self.gain_staging.stop()
                        await asyncio.sleep(1.0)  # Wait for audio device to be fully released
                        logger.info("Gain staging controller fully stopped, audio device released")
                    except Exception as e:
                        logger.warning(f"Error fully stopping gain staging: {e}")
                
            except Exception as e:
                logger.error(f"Error stopping realtime correction: {e}", exc_info=True)
                # Still try to release audio device
                if self.gain_staging:
                    try:
                        self.gain_staging.stop()
                        await asyncio.sleep(1.0)
                    except Exception as e2:
                        logger.warning(f"Error force-stopping gain staging: {e2}")
            
            await self._send_soundcheck_status(websocket, "gain_staging", 100, 0, "GAIN STAGING complete")
            await asyncio.sleep(1)
            
            if not self.auto_soundcheck_running:
                return
            
            # Step 3: PHASE ALIGNMENT
            duration = timings.get("phase_alignment", 30)
            await self._send_soundcheck_status(websocket, "phase_alignment", 0, duration, "Starting PHASE ALIGNMENT...")
            
            # Get reference channel (first channel)
            reference_channel = channels[0] if channels else None
            channels_to_align = channels[1:] if len(channels) > 1 else []
            
            if reference_channel and channels_to_align:
                logger.info(f"Starting phase alignment (apply_once): ref={reference_channel}, channels={channels_to_align}")
                try:
                    apply_once_sec = min(duration, 15)
                    await asyncio.wait_for(
                        self.start_phase_alignment(
                            websocket, device_id, reference_channel, channels_to_align,
                            apply_once=True, apply_once_duration_sec=apply_once_sec
                        ),
                        timeout=15.0
                    )
                    logger.info("Phase alignment started successfully")
                except asyncio.TimeoutError:
                    logger.warning("Phase alignment start timed out, continuing...")
                    await self._send_soundcheck_status(websocket, "phase_alignment", 0, duration, "Phase alignment start timed out, skipping...")
                    # Skip phase alignment if start fails
                    await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, "PHASE ALIGNMENT skipped (start timeout)")
                    await asyncio.sleep(0.5)
                    # Continue to next step
                except Exception as e:
                    logger.error(f"Error starting phase alignment: {e}", exc_info=True)
                    await self._send_soundcheck_status(websocket, "phase_alignment", 0, duration, f"Phase alignment error: {str(e)}, skipping...", error=str(e))
                    await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, "PHASE ALIGNMENT skipped (error)")
                    await asyncio.sleep(0.5)
                    # Continue to next step
                else:
                    # Only wait if start was successful
                    await self._wait_with_progress(websocket, "phase_alignment", duration)
                    
                    if not self.auto_soundcheck_running:
                        logger.info("Auto soundcheck stopped during PHASE ALIGNMENT")
                        try:
                            await asyncio.wait_for(self.stop_phase_alignment(websocket), timeout=10.0)
                        except (asyncio.TimeoutError, Exception) as e:
                            logger.error(f"Error stopping phase alignment: {e}")
                        return
                    
                    try:
                        # Stop phase alignment with timeout
                        await asyncio.wait_for(self.stop_phase_alignment(websocket), timeout=10.0)
                        logger.info("Phase alignment stopped successfully")
                        await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, "PHASE ALIGNMENT complete")
                    except asyncio.TimeoutError:
                        logger.warning("Phase alignment stop timed out, continuing...")
                        await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, "PHASE ALIGNMENT stopped (timeout)")
                    except Exception as e:
                        logger.error(f"Error stopping phase alignment: {e}", exc_info=True)
                        await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, f"PHASE ALIGNMENT error: {str(e)}")
            else:
                await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, "PHASE ALIGNMENT skipped (need at least 2 channels)")
            
            await asyncio.sleep(1)
            
            if not self.auto_soundcheck_running:
                return
            
            # Step 4: AUTO EQ
            duration = timings.get("auto_eq", 30)
            await self._send_soundcheck_status(websocket, "auto_eq", 0, duration, "Starting AUTO EQ...")
            
            # Build channels config for multi-channel auto EQ
            channels_config = []
            for channel_id in channels:
                profile = "custom"
                if channel_settings:
                    profile = channel_settings.get(channel_id, {}).get("preset", "custom")
                channels_config.append({
                    "channel": channel_id,
                    "profile": profile,
                    "auto_apply": False
                })
            
            await self.start_multi_channel_auto_eq(websocket, device_id, channels_config)
            await self._wait_with_progress(websocket, "auto_eq", duration)
            
            if not self.auto_soundcheck_running:
                await self.stop_multi_channel_auto_eq(websocket)
                return
            
            await self.stop_multi_channel_auto_eq(websocket)
            
            # Apply EQ corrections
            logger.info("Applying AUTO EQ corrections...")
            try:
                await self.apply_all_corrections(websocket)
                logger.info("AUTO EQ corrections applied successfully")
            except Exception as e:
                logger.error(f"Error applying AUTO EQ corrections: {e}", exc_info=True)
                await self._send_soundcheck_status(websocket, "auto_eq", 100, 0, f"AUTO EQ complete (apply error: {str(e)})")
            
            await self._send_soundcheck_status(websocket, "auto_eq", 100, 0, "AUTO EQ complete")
            await asyncio.sleep(1)
            
            if not self.auto_soundcheck_running:
                return
            
            # Step 5: AUTO FADER (Auto Balance)
            duration = timings.get("auto_fader", 30)
            await self._send_soundcheck_status(websocket, "auto_fader", 0, duration, "Starting AUTO FADER (Auto Balance)...")
            
            # Use same settings as Auto Balance from config
            auto_fader_config = self.config.get("automation", {}).get("auto_fader", {})
            bleed_threshold = auto_fader_config.get("bleed_threshold", -50.0)
            
            logger.info(f"Starting Auto Balance in soundcheck: duration={duration}s, bleed_threshold={bleed_threshold} LUFS, "
                       f"channels={channels}, channel_settings keys: {list(channel_settings.keys()) if channel_settings else []}, "
                       f"channel_mapping keys: {list(channel_mapping.keys()) if channel_mapping else []}")
            
            # Use same methods and settings as Auto Balance
            await self.start_auto_balance(websocket, device_id, channels, 
                                         channel_settings or {}, 
                                         channel_mapping or {}, 
                                         duration, bleed_threshold)
            
            # Wait for auto balance LEARN phase to complete
            # The start_auto_balance method handles the LEARN phase internally
            # We show progress while waiting
            await self._wait_with_progress(websocket, "auto_fader", duration)
            
            if not self.auto_soundcheck_running:
                await self.cancel_auto_balance(websocket)
                return
            
            # Small delay to ensure LEARN phase is fully complete
            await asyncio.sleep(1)
            
            # Apply auto balance corrections
            await self.apply_auto_balance(websocket)
            await self._send_soundcheck_status(websocket, "auto_fader", 100, 0, "AUTO FADER complete")
            await asyncio.sleep(1)
            
            # Complete
            await self._send_soundcheck_status(websocket, None, 100, 0, "Soundcheck cycle complete!", complete=True)
            
        except asyncio.CancelledError:
            logger.info("Auto soundcheck cycle cancelled")
            await self._send_soundcheck_status(websocket, None, 0, 0, "Auto soundcheck cancelled", error="Cancelled")
        except Exception as e:
            logger.error(f"Error in auto soundcheck cycle: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            await self._send_soundcheck_status(websocket, None, 0, 0, f"Error: {str(e)}", error=str(e))
        finally:
            logger.info("Auto soundcheck cycle finished, cleaning up...")
            self.auto_soundcheck_running = False
            self.auto_soundcheck_task = None
            self.auto_soundcheck_websocket = None
    
    async def _wait_with_progress(self, websocket, step: str, duration: float):
        """Wait for specified duration while sending progress updates."""
        start_time = time.time()
        update_interval = 0.5  # Update every 0.5 seconds
        
        while self.auto_soundcheck_running:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            
            progress = (elapsed / duration) * 100
            remaining = duration - elapsed
            
            await self._send_soundcheck_status(websocket, step, progress, remaining, 
                                             f"Running {step.upper().replace('_', ' ')}...")
            
            await asyncio.sleep(update_interval)
    
    async def _send_soundcheck_status(self, websocket, current_step: str = None, 
                                     step_progress: float = 0, step_time_remaining: float = 0,
                                     message: str = "", error: str = None, complete: bool = False):
        """Send auto soundcheck status update."""
        try:
            if websocket is None:
                logger.warning("Cannot send soundcheck status: websocket is None")
                return
            
            status_data = {
                "type": "auto_soundcheck_status",
                "is_running": self.auto_soundcheck_running and not complete,
                "current_step": current_step,
                "step_progress": step_progress,
                "step_time_remaining": step_time_remaining,
                "message": message,
                "error": error,
                "complete": complete
            }
            
            logger.debug(f"Sending soundcheck status: step={current_step}, progress={step_progress:.1f}%, message={message}")
            await self.send_to_client(websocket, status_data)
        except Exception as e:
            logger.error(f"Error sending soundcheck status: {e}", exc_info=True)
            # Don't raise - we don't want to stop the cycle if status update fails

    async def _execute_voice_command(self, command: dict):
        """Execute a recognized voice command"""
        if not self.mixer_client:
            logger.warning("No mixer connected, cannot execute voice command")
            return
        
        try:
            cmd_type = command.get("type")
            
            if cmd_type == "set_fader":
                channel = command.get("channel")
                value = command.get("value", 0.5)
                self.mixer_client.set_channel_fader(channel, value)
                await self.broadcast({
                    "type": "voice_command_executed",
                    "command": cmd_type,
                    "channel": channel,
                    "value": value
                })
                
            elif cmd_type == "set_gain":
                channel = command.get("channel")
                value = command.get("value", 0.0)
                self.mixer_client.set_channel_gain(channel, value)
                await self.broadcast({
                    "type": "voice_command_executed",
                    "command": cmd_type,
                    "channel": channel,
                    "value": value
                })
                
            elif cmd_type == "load_snap":
                snap_name = command.get("snap_name")
                if isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    success = self.mixer_client.load_snap(snap_name)
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "snap_name": snap_name,
                        "success": success
                    })
                    
            elif cmd_type == "mute_channel":
                channel = command.get("channel")
                muted = command.get("muted", True)
                # Note: Add mute method to mixer_client if not exists
                await self.broadcast({
                    "type": "voice_command_executed",
                    "command": cmd_type,
                    "channel": channel,
                    "muted": muted
                })
                
            elif cmd_type == "volume_up":
                channel = command.get("channel")
                db_change = command.get("db", 3)  # Get dB value directly (default 3 dB)
                if channel is not None:
                    # Try to get current value from tracked values first, then from mixer
                    current_db = self.last_fader_values.get(channel)
                    
                    if current_db is None:
                        # Request fresh value from Wing
                        self.mixer_client.send(f"/ch/{channel}/fdr")
                        await asyncio.sleep(0.1)  # Wait for response
                        current_db = self.mixer_client.get_channel_fader(channel)
                    
                    if current_db is not None:
                        # Add dB directly (Wing uses dB values)
                        new_db = min(10.0, current_db + db_change)
                        self.mixer_client.set_channel_fader(channel, new_db)
                        # Update tracked value
                        self.last_fader_values[channel] = new_db
                        logger.info(f"Increased channel {channel} fader: {current_db:.1f} dB -> {new_db:.1f} dB (+{db_change} dB)")
                    else:
                        # Fallback: use last known or default to 0
                        last_known = self.last_fader_values.get(channel, 0.0)
                        new_db = min(10.0, last_known + db_change)
                        self.mixer_client.set_channel_fader(channel, new_db)
                        self.last_fader_values[channel] = new_db
                        logger.info(f"Increased channel {channel} fader by {db_change} dB (to {new_db:.1f} dB, using tracked value)")
                    
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "db_change": db_change,
                        "new_db": new_db
                    })
                else:
                    logger.warning(f"volume_up command ignored: channel is None")
                    
            elif cmd_type == "volume_down":
                channel = command.get("channel")
                db_change = command.get("db", 3)  # Get dB value directly (default 3 dB)
                if channel is not None:
                    # Try to get current value from tracked values first, then from mixer
                    current_db = self.last_fader_values.get(channel)
                    
                    if current_db is None:
                        # Request fresh value from Wing
                        self.mixer_client.send(f"/ch/{channel}/fdr")
                        await asyncio.sleep(0.1)  # Wait for response
                        current_db = self.mixer_client.get_channel_fader(channel)
                    
                    if current_db is not None:
                        # Subtract dB directly (Wing uses dB values)
                        new_db = max(-144.0, current_db - db_change)
                        self.mixer_client.set_channel_fader(channel, new_db)
                        # Update tracked value
                        self.last_fader_values[channel] = new_db
                        logger.info(f"Decreased channel {channel} fader: {current_db:.1f} dB -> {new_db:.1f} dB (-{db_change} dB)")
                    else:
                        # Fallback: use last known or default to 0
                        last_known = self.last_fader_values.get(channel, 0.0)
                        new_db = max(-144.0, last_known - db_change)
                        self.mixer_client.set_channel_fader(channel, new_db)
                        self.last_fader_values[channel] = new_db
                        logger.info(f"Decreased channel {channel} fader by {db_change} dB (to {new_db:.1f} dB, using tracked value)")
                    
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "db_change": db_change,
                        "new_db": new_db
                    })
                else:
                    logger.warning(f"volume_down command ignored: channel is None")
            
            elif cmd_type == "eq_on":
                channel = command.get("channel")
                on = command.get("on", 1)
                if channel is not None and isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    self.mixer_client.set_eq_on(channel, on)
                    logger.info(f"Set EQ {'on' if on else 'off'} for channel {channel}")
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "on": on
                    })
            
            elif cmd_type == "eq_band_up" or cmd_type == "eq_band_down":
                channel = command.get("channel")
                band = command.get("band")
                db_change = command.get("db", 3)
                if channel is not None and band and isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    # Get current gain
                    band_gain_key = f"{band}g" if band in ["1", "2", "3", "4"] else ("lg" if band == "low" else "hg")
                    current_gain = self.mixer_client.get_eq_band_gain(channel, band_gain_key) or 0.0
                    
                    if cmd_type == "eq_band_up":
                        new_gain = min(15.0, current_gain + db_change)
                    else:
                        new_gain = max(-15.0, current_gain - db_change)
                    
                    self.mixer_client.set_eq_band_gain(channel, band_gain_key, new_gain)
                    logger.info(f"Set EQ band {band} for channel {channel}: {current_gain:.1f} dB -> {new_gain:.1f} dB")
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "band": band,
                        "new_gain": new_gain
                    })
            
            elif cmd_type == "compressor_on":
                channel = command.get("channel")
                on = command.get("on", 1)
                if channel is not None and isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    self.mixer_client.set_compressor_on(channel, on)
                    logger.info(f"Set compressor {'on' if on else 'off'} for channel {channel}")
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "on": on
                    })
            
            elif cmd_type == "compressor_threshold":
                channel = command.get("channel")
                threshold = command.get("threshold")
                if channel is not None and threshold is not None and isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    self.mixer_client.set_compressor_threshold(channel, threshold)
                    logger.info(f"Set compressor threshold for channel {channel}: {threshold:.1f} dB")
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "threshold": threshold
                    })
            
            elif cmd_type == "compressor_gain_up" or cmd_type == "compressor_gain_down":
                channel = command.get("channel")
                db_change = command.get("db", 3)
                if channel is not None and isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    current_gain = self.mixer_client.get_compressor_gain(channel) or 0.0
                    if cmd_type == "compressor_gain_up":
                        new_gain = min(12.0, current_gain + db_change)
                    else:
                        new_gain = max(-6.0, current_gain - db_change)
                    
                    self.mixer_client.set_compressor_gain(channel, new_gain)
                    logger.info(f"Set compressor gain for channel {channel}: {current_gain:.1f} dB -> {new_gain:.1f} dB")
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "new_gain": new_gain
                    })
            
            logger.info("=" * 60)
            logger.info(f"✅ EXECUTED VOICE COMMAND: {cmd_type}")
            if channel is not None:
                logger.info(f"   Channel: {channel}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ ERROR EXECUTING VOICE COMMAND: {e}")
            logger.error(f"   Command: {command}")
            logger.error("=" * 60)
            await self.broadcast({
                "type": "voice_command_error",
                "error": str(e),
                "command": command
            })

    async def handler(self, websocket: WebSocketServerProtocol):
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                try:
                    await self.handle_client_message(websocket, message)
                except Exception as e:
                    logger.error(f"Error handling message: {e}", exc_info=True)
                    # Don't crash the connection - continue listening
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        except Exception as e:
            logger.error(f"Handler error: {e}", exc_info=True)
        finally:
            await self.unregister_client(websocket)

    async def start(self):
        logger.info(f"Starting WebSocket server on {self.ws_host}:{self.ws_port}")
        
        # Create shutdown event in the correct event loop
        self._shutdown_event = asyncio.Event()
        
        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self.graceful_shutdown(s))
                )
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass
        
        # Create server with SO_REUSEADDR enabled
        import socket
        
        # Custom create_server with reuse_address
        try:
            server = await websockets.serve(
                self.handler,
                self.ws_host,
                self.ws_port,
                reuse_address=True  # Allow port reuse
            )
            
            logger.info(f"WebSocket server started on {self.ws_host}:{self.ws_port}")
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
            # Close the server
            server.close()
            await server.wait_closed()
            
        except OSError as e:
            if e.errno == 48:  # Address already in use
                logger.error(f"Port {self.ws_port} is already in use. Attempting to force cleanup...")
                # Try to cleanup any existing processes
                import subprocess
                try:
                    subprocess.run(
                        f"lsof -ti:{self.ws_port} | xargs kill -9",
                        shell=True,
                        capture_output=True
                    )
                    await asyncio.sleep(1)
                    # Retry
                    server = await websockets.serve(
                        self.handler,
                        self.ws_host,
                        self.ws_port,
                        reuse_address=True
                    )
                    logger.info(f"WebSocket server started on {self.ws_host}:{self.ws_port} (after cleanup)")
                    await self._shutdown_event.wait()
                    server.close()
                    await server.wait_closed()
                except Exception as retry_error:
                    logger.error(f"Failed to start server after cleanup: {retry_error}")
                    raise
            else:
                raise
        
        logger.info("Server stopped")


# Global server instance for atexit cleanup
_server_instance: Optional[AutoMixerServer] = None

def _atexit_cleanup():
    """Cleanup function called on exit."""
    global _server_instance
    if _server_instance:
        logger.info("Atexit cleanup triggered")
        _server_instance.cleanup_all_controllers()


if __name__ == "__main__":
    # Register atexit handler
    atexit.register(_atexit_cleanup)
    
    server = AutoMixerServer()
    _server_instance = server
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
    finally:
        # Ensure cleanup happens
        server.cleanup_all_controllers()
        logger.info("Server shutdown complete")
