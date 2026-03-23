#!/usr/bin/env python3
"""
Тестовый скрипт для проверки real-time gain staging коррекции
"""

import asyncio
import json
import websockets
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_gain_staging():
    """Тест gain staging через WebSocket"""
    
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to WebSocket server")
            
            # 1. Проверка статуса gain staging
            logger.info("1. Checking gain staging status...")
            await websocket.send(json.dumps({"type": "get_gain_staging_status"}))
            response = await websocket.recv()
            status = json.loads(response)
            logger.info(f"Gain staging status: {status}")
            
            # 2. Запуск real-time коррекции
            logger.info("2. Starting real-time correction...")
            
            # Тестовые данные - каналы 1-5 с разными пресетами
            test_message = {
                "type": "start_realtime_correction",
                "device_id": "1",
                "channels": [1, 2, 3, 4, 5],
                "channel_settings": {
                    "1": {"preset": "kick"},
                    "2": {"preset": "snare"},
                    "3": {"preset": "tom"},
                    "4": {"preset": "hihat"},
                    "5": {"preset": "leadVocal"}
                },
                "channel_mapping": {
                    "1": 1,
                    "2": 2,
                    "3": 3,
                    "4": 4,
                    "5": 5
                }
            }
            
            await websocket.send(json.dumps(test_message))
            
            # Получение ответов
            logger.info("3. Waiting for responses...")
            for i in range(10):
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    if data.get("type") == "gain_staging_status":
                        logger.info(f"Response {i+1}: {json.dumps(data, indent=2)}")
                        if data.get("realtime_enabled"):
                            logger.info("✓ Real-time correction is ENABLED")
                        if data.get("error"):
                            logger.error(f"✗ Error: {data.get('error')}")
                except asyncio.TimeoutError:
                    logger.info(f"No response {i+1}, continuing...")
                    break
            
            # 4. Мониторинг статуса в течение 10 секунд
            logger.info("4. Monitoring for 10 seconds...")
            start_time = time.time()
            trim_updates = []
            
            while time.time() - start_time < 10:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "gain_staging_status":
                        if "channels" in data:
                            # Обновления уровней
                            for ch, ch_data in data["channels"].items():
                                peak = ch_data.get("measured_peak", -60.0)
                                signal = ch_data.get("signal_present", False)
                                if signal or peak > -50:
                                    logger.info(f"  Channel {ch}: peak={peak:.1f} dB, signal={signal}")
                    
                    # Проверка на сообщения о TRIM
                    if "TRIM" in str(data) or "trim" in str(data).lower():
                        logger.info(f"  TRIM update: {data}")
                        trim_updates.append(data)
                        
                except asyncio.TimeoutError:
                    continue
            
            logger.info(f"5. Test completed. Found {len(trim_updates)} TRIM updates")
            
            # 5. Остановка коррекции
            logger.info("6. Stopping real-time correction...")
            await websocket.send(json.dumps({"type": "stop_realtime_correction"}))
            
            response = await websocket.recv()
            logger.info(f"Stop response: {json.loads(response)}")
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_gain_staging())
