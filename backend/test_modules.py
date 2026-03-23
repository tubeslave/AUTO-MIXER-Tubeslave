import asyncio
import websockets
import json
import sys

async def test_module():
    uri = "ws://localhost:8765"
    try:
        async with websockets.connect(uri) as websocket:
            # Получаем статус
            await websocket.send(json.dumps({"type": "get_gain_staging_status"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"Gain Staging Status: {response}")
            
            # Проверяем статус Auto EQ
            await websocket.send(json.dumps({"type": "get_auto_eq_status"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"Auto EQ Status: {response}")
            
            # Проверяем статус Auto Fader
            await websocket.send(json.dumps({"type": "get_auto_fader_status"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"Auto Fader Status: {response}")
            
            # Проверяем статус Auto Compressor
            await websocket.send(json.dumps({"type": "get_auto_compressor_status"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"Auto Compressor Status: {response}")
            
            # Проверяем статус Auto Gate
            await websocket.send(json.dumps({"type": "get_auto_gate_status"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"Auto Gate Status: {response}")
            
            # Проверяем статус Auto Panner
            await websocket.send(json.dumps({"type": "get_auto_panner_status"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"Auto Panner Status: {response}")
            
            # Проверяем статус Auto Reverb
            await websocket.send(json.dumps({"type": "get_auto_reverb_status"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"Auto Reverb Status: {response}")
            
            # Проверяем статус Auto Effects
            await websocket.send(json.dumps({"type": "get_auto_effects_status"}))
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"Auto Effects Status: {response}")
            
    except Exception as e:
        print(f"Error: {e}")
        return False
    return True

if __name__ == "__main__":
    result = asyncio.run(test_module())
    sys.exit(0 if result else 1)
