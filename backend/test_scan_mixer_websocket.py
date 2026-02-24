#!/usr/bin/env python3
"""
Тестовый скрипт для проверки функции scan_mixer_channel_names через WebSocket
Симулирует frontend клиент
"""
import asyncio
import json
import websockets
import sys
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_scan_mixer_via_websocket():
    """Тест функции scan_mixer_channel_names через WebSocket"""
    
    ws_url = "ws://localhost:8765"
    mixer_ip = sys.argv[1] if len(sys.argv) >= 2 else "192.168.1.102"
    mixer_port = 2223
    
    logger.info("="*60)
    logger.info("Тест функции scan_mixer_channel_names через WebSocket")
    logger.info("="*60)
    
    try:
        logger.info(f"Подключение к WebSocket серверу: {ws_url}")
        async with websockets.connect(ws_url) as websocket:
            logger.info("✅ Подключено к WebSocket серверу")
            
            # Шаг 1: Подключение к микшеру
            logger.info(f"\nШаг 1: Подключение к микшеру {mixer_ip}:{mixer_port}")
            connect_message = {
                "type": "connect_wing",
                "ip": mixer_ip,
                "send_port": mixer_port,
                "receive_port": mixer_port
            }
            await websocket.send(json.dumps(connect_message))
            logger.info(f"Отправлено: {json.dumps(connect_message)}")
            
            # Ждем подтверждения подключения
            logger.info("Ожидание подтверждения подключения...")
            response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            data = json.loads(response)
            logger.info(f"Получено: {data.get('type')}")
            
            if data.get('type') == 'connection_status' and data.get('connected'):
                logger.info("✅ Подключено к микшеру")
            else:
                logger.error(f"❌ Не удалось подключиться: {data}")
                return False
            
            # Ждем еще немного для полной инициализации
            await asyncio.sleep(2)
            
            # Шаг 2: Отправка запроса на сканирование имен каналов
            logger.info(f"\nШаг 2: Отправка запроса scan_mixer_channel_names")
            scan_message = {
                "type": "scan_mixer_channel_names"
            }
            await websocket.send(json.dumps(scan_message))
            logger.info(f"Отправлено: {json.dumps(scan_message)}")
            
            # Шаг 3: Ожидание ответа
            logger.info("Ожидание ответа от сервера...")
            logger.info("(Будут пропущены сообщения mixer_update)")
            start_time = time.time()
            timeout = 20.0
            
            response_received = False
            mixer_update_count = 0
            response_data = None
            
            while time.time() - start_time < timeout:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    msg_type = data.get('type')
                    
                    if msg_type == 'mixer_update':
                        mixer_update_count += 1
                        if mixer_update_count % 50 == 0:
                            logger.debug(f"Получено {mixer_update_count} сообщений mixer_update...")
                        continue
                    
                    logger.info(f"Получено сообщение типа: {msg_type}")
                    
                    if msg_type == 'mixer_channel_names':
                        response_received = True
                        response_data = data
                        logger.info(f"(Пропущено {mixer_update_count} сообщений mixer_update)")
                        logger.info("\n" + "="*60)
                        logger.info("✅ ОТВЕТ ПОЛУЧЕН")
                        logger.info("="*60)
                        break
                    
                    else:
                        logger.debug(f"Пропущено сообщение типа: {msg_type}")
                        
                except asyncio.TimeoutError:
                    logger.debug("Таймаут ожидания ответа, продолжаем...")
                    continue
                except Exception as e:
                    logger.error(f"Ошибка при получении сообщения: {e}")
                    continue
            
            logger.info(f"DEBUG: response_received={response_received}, response_data={response_data is not None}")
            if response_received and response_data:
                logger.info(f"DEBUG: response_data type: {type(response_data)}")
                logger.info(f"DEBUG: response_data keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'N/A'}")
                if isinstance(response_data, dict) and response_data.get('error'):
                    logger.error(f"❌ Ошибка: {response_data.get('error')}")
                    return False
                
                channel_names = response_data.get('channel_names', {})
                if not channel_names:
                    logger.error("❌ channel_names пуст")
                    return False
                
                logger.info(f"Количество каналов в ответе: {len(channel_names)}")
                logger.info(f"\nПримеры имен каналов:")
                count = 0
                for ch_num, name in sorted(channel_names.items()):
                    if name != f"Ch {ch_num}":
                        logger.info(f"  Канал {ch_num}: '{name}'")
                        count += 1
                        if count >= 10:
                            break
                
                # Проверка формата данных
                logger.info(f"\nПроверка формата данных:")
                logger.info(f"  Тип channel_names: {type(channel_names)}")
                logger.info(f"  Ключи - это числа: {all(isinstance(k, (int, str)) for k in channel_names.keys())}")
                logger.info(f"  Значения - это строки: {all(isinstance(v, str) for v in channel_names.values())}")
                
                # Проверка JSON сериализации (как будет отправлено на frontend)
                test_json = json.dumps({
                    "type": "mixer_channel_names",
                    "channel_names": channel_names
                }, ensure_ascii=False)
                logger.info(f"  JSON размер: {len(test_json)} байт")
                logger.info(f"  JSON валиден: ✅")
                
                logger.info("\n" + "="*60)
                logger.info("✅ ТЕСТ ПРОЙДЕН УСПЕШНО")
                logger.info("="*60)
                return True
            
            if not response_received:
                logger.error("\n" + "="*60)
                logger.error("❌ ТЕСТ ПРОВАЛЕН: Ответ не получен в течение 20 секунд")
                logger.error("="*60)
                return False
                
    except websockets.exceptions.ConnectionRefused:
        logger.error(f"❌ Не удалось подключиться к WebSocket серверу: {ws_url}")
        logger.error("Убедитесь, что backend сервер запущен (python server.py)")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = asyncio.run(test_scan_mixer_via_websocket())
    sys.exit(0 if success else 1)
