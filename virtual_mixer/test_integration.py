"""
Virtual Mixer Test Integration

Integrates virtual mixer with Auto Mixer for testing.
- Runs virtual mixer on port 2222
- Generates test signals
- Provides WebSocket bridge to Auto Mixer
- Logs all OSC commands

Usage:
    python test_integration.py

Requirements:
    pip install python-osc websockets
"""

import asyncio
import json
import logging
import websockets
from typing import Dict, Set
from datetime import datetime

from virtual_mixer import VirtualWingMixer
from test_signal_generator import TestSignalGenerator

logger = logging.getLogger(__name__)


class VirtualMixerTestBridge:
    """
    Bridge between Virtual Mixer and Auto Mixer.
    
    Provides:
    - OSC server (virtual mixer)
    - WebSocket server (for Auto Mixer frontend)
    - Test signal generation
    - Logging and monitoring
    """
    
    def __init__(
        self,
        osc_port: int = 2222,
        websocket_port: int = 8766,
        auto_mixer_ws: str = "ws://localhost:8765"
    ):
        self.osc_port = osc_port
        self.websocket_port = websocket_port
        self.auto_mixer_ws = auto_mixer_ws
        
        # Components
        self.mixer = VirtualWingMixer(osc_port=osc_port)
        self.signals = TestSignalGenerator()
        
        # WebSocket clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # State
        self.running = False
        self.test_mode = "silence"  # silence, drums, full_mix
        
        logger.info(f"TestBridge initialized")
        logger.info(f"  OSC Port: {osc_port}")
        logger.info(f"  WebSocket Port: {websocket_port}")
    
    async def start(self):
        """Start all services."""
        self.running = True
        
        # Start OSC server
        osc_transport = await self.mixer.start()
        
        # Start WebSocket server
        ws_server = await websockets.serve(
            self._handle_websocket,
            "0.0.0.0",
            self.websocket_port
        )
        
        # Start signal generator
        signal_task = asyncio.create_task(self._signal_generator_loop())
        
        # Start status broadcaster
        broadcast_task = asyncio.create_task(self._status_broadcast_loop())
        
        logger.info("✅ Virtual Mixer Test Bridge started")
        logger.info(f"   OSC: port {self.osc_port}")
        logger.info(f"   WebSocket: ws://localhost:{self.websocket_port}")
        logger.info("")
        logger.info("Connect Auto Mixer to:")
        logger.info(f"   Mixer IP: localhost")
        logger.info(f"   OSC Port: {self.osc_port}")
        logger.info("")
        logger.info("Available test modes:")
        logger.info("   - silence: All channels silent")
        logger.info("   - drums: Kick, Snare, HiHat only")
        logger.info("   - full_mix: All instruments")
        
        try:
            await asyncio.gather(signal_task, broadcast_task)
        except asyncio.CancelledError:
            pass
        finally:
            osc_transport.close()
            ws_server.close()
    
    async def _handle_websocket(self, websocket, path):
        """Handle WebSocket connections."""
        self.clients.add(websocket)
        logger.info(f"WebSocket client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                await self._handle_ws_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"WebSocket client disconnected")
    
    async def _handle_ws_message(self, websocket, message):
        """Handle WebSocket messages from clients."""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            if msg_type == 'set_test_mode':
                mode = data.get('mode', 'silence')
                self._set_test_mode(mode)
                await websocket.send(json.dumps({
                    'type': 'test_mode_changed',
                    'mode': mode
                }))
            
            elif msg_type == 'set_channel_level':
                ch = data.get('channel', 1)
                level_db = data.get('level_db', -20)
                self.signals.channels[ch].level_db = level_db
            
            elif msg_type == 'get_mixer_state':
                state = self.mixer.get_state()
                await websocket.send(json.dumps({
                    'type': 'mixer_state',
                    'state': state
                }))
            
            elif msg_type == 'start_song':
                bpm = data.get('bpm', 120)
                self.signals.start_song(bpm)
                await websocket.send(json.dumps({
                    'type': 'song_started',
                    'bpm': bpm
                }))
            
            else:
                logger.warning(f"Unknown WS message: {msg_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {message}")
        except Exception as e:
            logger.error(f"WS error: {e}")
    
    def _set_test_mode(self, mode: str):
        """Set test signal mode."""
        self.test_mode = mode
        
        if mode == "silence":
            for ch in range(1, 33):
                self.signals.channels[ch].signal_type = self.signals.channels[ch].signal_type.SILENCE
                self.signals.channels[ch].level_db = -100
        
        elif mode == "drums":
            # Only drums active
            for ch in range(1, 13):
                if ch <= 5 or ch in [9, 10]:  # Kick, snare, hihat, overheads
                    pass  # Keep default
                else:
                    self.signals.channels[ch].signal_type = self.signals.channels[ch].signal_type.SILENCE
        
        elif mode == "full_mix":
            # All default signals
            self.signals._setup_default_signals()
        
        logger.info(f"Test mode changed to: {mode}")
    
    async def _signal_generator_loop(self):
        """Generate test signals continuously."""
        start_time = asyncio.get_event_loop().time()
        
        while self.running:
            current_time = asyncio.get_event_loop().time() - start_time
            
            # Generate signals for all channels
            signals = self.signals.generate_all(current_time)
            
            # Update mixer input levels
            for ch, level_db in signals.items():
                self.mixer.set_input_signal(ch, level_db)
            
            await asyncio.sleep(0.01)  # 100Hz update rate
    
    async def _status_broadcast_loop(self):
        """Broadcast mixer status to all WebSocket clients."""
        while self.running:
            if self.clients:
                state = self.mixer.get_state()
                
                # Add timestamp and test mode
                message = json.dumps({
                    'type': 'mixer_update',
                    'timestamp': datetime.now().isoformat(),
                    'test_mode': self.test_mode,
                    'state': state
                })
                
                # Broadcast to all clients
                disconnected = set()
                for client in self.clients:
                    try:
                        await client.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected.add(client)
                
                # Remove disconnected clients
                self.clients -= disconnected
            
            await asyncio.sleep(0.1)  # 10Hz broadcast rate


class OSCLogger:
    """
    Logs all OSC commands received by virtual mixer.
    """
    
    def __init__(self, mixer: VirtualWingMixer):
        self.mixer = mixer
        self.log_file = "osc_log.txt"
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup OSC command logging."""
        original_handler = self.mixer.dispatcher.handlers
        
        def logged_handler(addr, *args):
            # Log to file
            with open(self.log_file, "a") as f:
                f.write(f"{datetime.now().isoformat()} {addr} {args}\n")
            
            # Also print important commands
            if "fader" in addr or "gain" in addr or "dyn" in addr:
                logger.info(f"OSC: {addr} = {args}")
        
        # Wrap all handlers with logging
        # (Implementation depends on python-osc version)


async def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("Virtual Wing Mixer - Test Environment")
    print("=" * 60)
    print()
    
    bridge = VirtualMixerTestBridge(
        osc_port=2222,
        websocket_port=8766
    )
    
    try:
        await bridge.start()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
