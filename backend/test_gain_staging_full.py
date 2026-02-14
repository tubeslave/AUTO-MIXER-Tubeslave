#!/usr/bin/env python3
"""
Полный тест gain staging: подключение к микшеру и запуск коррекции
"""

import asyncio
import json
import websockets
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_full_gain_staging():
    """Полный тест gain staging"""
    
    uri = "ws://localhost:8765"
    mixer_ip = "192.168.1.102"  # Из логов видно этот IP
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("=== Подключение к WebSocket серверу ===")
            
            # 1. Подключение к микшеру
            logger.info("1. Подключение к микшеру...")
            connect_msg = {
                "type": "connect_wing",
                "ip": mixer_ip,
                "send_port": 2222,
                "receive_port": 2223
            }
            await websocket.send(json.dumps(connect_msg))
            
            # Ждем подтверждения подключения
            for i in range(5):
                response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(response)
                if data.get("type") == "connection_status" and data.get("connected"):
                    logger.info("✓ Микшер подключен!")
                    break
                logger.info(f"  Ответ {i+1}: {data.get('type')}")
            
            await asyncio.sleep(2)  # Даем время на инициализацию
            
            # 2. Запуск real-time коррекции
            logger.info("2. Запуск real-time коррекции...")
            
            test_message = {
                "type": "start_realtime_correction",
                "device_id": "1",
                "channels": [1, 2, 3],
                "channel_settings": {
                    "1": {"preset": "kick"},
                    "2": {"preset": "snare"},
                    "3": {"preset": "leadVocal"}
                },
                "channel_mapping": {
                    "1": 1,
                    "2": 2,
                    "3": 3
                }
            }
            
            await websocket.send(json.dumps(test_message))
            
            # Получение ответов о запуске
            logger.info("3. Ожидание подтверждения запуска...")
            started = False
            for i in range(10):
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    if data.get("type") == "gain_staging_status":
                        if data.get("realtime_enabled"):
                            logger.info("✓ Real-time коррекция ЗАПУЩЕНА!")
                            started = True
                        if data.get("error"):
                            logger.error(f"✗ Ошибка: {data.get('error')}")
                            return
                except asyncio.TimeoutError:
                    break
            
            if not started:
                logger.error("✗ Не удалось запустить коррекцию")
                return
            
            # 4. Мониторинг в течение 15 секунд
            logger.info("4. Мониторинг коррекции в течение 15 секунд...")
            logger.info("   Смотрите логи бэкенда для деталей:")
            logger.info("   tail -f /tmp/backend.log | grep -E '(Channel|TRIM|correction)'")
            
            start_time = time.time()
            status_updates = 0
            trim_updates_found = 0
            
            while time.time() - start_time < 15:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(response)
                    
                    if data.get("type") == "gain_staging_status":
                        status_updates += 1
                        if "channels" in data:
                            for ch, ch_data in data["channels"].items():
                                peak = ch_data.get("measured_peak", -60.0)
                                signal = ch_data.get("signal_present", False)
                                if signal or peak > -50:
                                    logger.info(f"  Канал {ch}: peak={peak:.1f} dB, signal={signal}")
                    
                    if "TRIM" in str(data) or "trim" in str(data).lower():
                        trim_updates_found += 1
                        logger.info(f"  → Обновление TRIM: {data}")
                        
                except asyncio.TimeoutError:
                    continue
            
            logger.info(f"5. Мониторинг завершен:")
            logger.info(f"   - Получено обновлений статуса: {status_updates}")
            logger.info(f"   - Найдено обновлений TRIM: {trim_updates_found}")
            
            # 5. Остановка
            logger.info("6. Остановка коррекции...")
            await websocket.send(json.dumps({"type": "stop_realtime_correction"}))
            await asyncio.sleep(1)
            
            logger.info("=== Тест завершен ===")
            
    except Exception as e:
        logger.error(f"Ошибка во время теста: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_full_gain_staging())
