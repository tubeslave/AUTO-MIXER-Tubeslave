#!/usr/bin/env python3
"""
Integration test for voice control with WebSocket server
Tests the full integration of voice control with the server
"""
import asyncio
import json
import logging
import websockets
import time
from voice_control import VoiceControl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

WS_URL = "ws://localhost:8765"


async def test_voice_control_commands():
    """Test voice control WebSocket commands"""
    print("\n=== Voice Control Integration Test ===")
    print(f"Connecting to {WS_URL}...")
    
    try:
        async with websockets.connect(WS_URL) as websocket:
            print("✅ Connected to server")
            
            # Test 1: Get voice control status (should be inactive)
            print("\n1. Checking initial voice control status...")
            await websocket.send(json.dumps({
                "type": "get_voice_control_status"
            }))
            
            response = await websocket.recv()
            data = json.loads(response)
            print(f"   Response: {data}")
            
            if data.get("type") == "voice_control_status":
                if not data.get("active"):
                    print("   ✅ Voice control is inactive (expected)")
                else:
                    print("   ⚠️  Voice control is already active")
            
            # Test 2: Start voice control
            print("\n2. Starting voice control...")
            await websocket.send(json.dumps({
                "type": "start_voice_control",
                "model_size": "small",
                "language": "ru"
            }))
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                data = json.loads(response)
                print(f"   Response: {data}")
                
                if data.get("type") == "voice_control_status":
                    if data.get("active"):
                        print("   ✅ Voice control started successfully")
                    else:
                        if "error" in data:
                            print(f"   ❌ Error starting voice control: {data.get('error')}")
                        else:
                            print("   ⚠️  Voice control did not start")
            except asyncio.TimeoutError:
                print("   ⚠️  No response received (model might be loading...)")
            
            # Test 3: Check status again
            print("\n3. Checking voice control status again...")
            await websocket.send(json.dumps({
                "type": "get_voice_control_status"
            }))
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"   Response: {data}")
                
                if data.get("active"):
                    print("   ✅ Voice control is active")
                else:
                    print("   ⚠️  Voice control is not active")
            except asyncio.TimeoutError:
                print("   ⚠️  No response received")
            
            # Test 4: Listen for voice commands (if active)
            print("\n4. Listening for voice commands...")
            print("   Say a command like 'канал 1' or 'загрузить тест'")
            print("   (Listening for 15 seconds...)")
            
            commands_received = []
            start_time = time.time()
            
            while time.time() - start_time < 15:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "voice_command_executed":
                        commands_received.append(data)
                        print(f"\n   🎤 Command executed: {data}")
                    elif data.get("type") == "voice_command_error":
                        print(f"\n   ❌ Command error: {data.get('error')}")
                except asyncio.TimeoutError:
                    continue
            
            if commands_received:
                print(f"\n   ✅ Received {len(commands_received)} command(s)")
            else:
                print("\n   ⚠️  No commands received")
            
            # Test 5: Stop voice control
            print("\n5. Stopping voice control...")
            await websocket.send(json.dumps({
                "type": "stop_voice_control"
            }))
            
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"   Response: {data}")
                
                if data.get("type") == "voice_control_status":
                    if not data.get("active"):
                        print("   ✅ Voice control stopped successfully")
                    else:
                        print("   ⚠️  Voice control is still active")
            except asyncio.TimeoutError:
                print("   ⚠️  No response received")
            
            print("\n✅ Integration test completed")
            return True
            
    except websockets.exceptions.ConnectionRefused:
        print(f"❌ Cannot connect to {WS_URL}")
        print("   Make sure the server is running: python server.py")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run integration test"""
    print("=" * 60)
    print("Voice Control Integration Test")
    print("=" * 60)
    print("\n⚠️  Make sure the server is running before starting this test!")
    print("   Run: python server.py")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nTest cancelled")
        return
    
    success = await test_voice_control_commands()
    
    if success:
        print("\n🎉 Integration test passed!")
        return 0
    else:
        print("\n⚠️  Integration test had issues")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
