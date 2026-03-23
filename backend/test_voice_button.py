#!/usr/bin/env python3
"""
Тест для проверки работы голосового управления через WebSocket
Имитирует нажатие кнопки из frontend
"""
import asyncio
import json
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WS_URL = "ws://localhost:8765"


async def test_start_voice_control():
    """Тест запуска голосового управления"""
    print("=" * 60)
    print("Тест запуска голосового управления")
    print("=" * 60)
    
    try:
        print(f"\n1. Подключение к {WS_URL}...")
        async with websockets.connect(WS_URL) as websocket:
            print("   ✅ Подключено")
            
            # Пропускаем первое сообщение (connection_status при подключении)
            try:
                first_msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                first_data = json.loads(first_msg)
                print(f"   Первое сообщение (пропускаем): {first_data.get('type')}")
            except asyncio.TimeoutError:
                pass
            
            # Запрос списка аудио устройств
            print("\n2. Запрос списка аудио устройств...")
            await websocket.send(json.dumps({
                "type": "get_audio_devices"
            }))
            
            # Ждем ответ с audio_devices
            devices_data = None
            for _ in range(3):  # Максимум 3 попытки
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                print(f"   Получено: {data.get('type')}")
                if data.get("type") == "audio_devices":
                    devices_data = data
                    break
            
            if devices_data:
                devices = data.get("devices", [])
                print(f"   Найдено устройств: {len(devices)}")
                if devices:
                    first_device = devices[0]
                    device_id = first_device.get("id")
                    print(f"   Первое устройство: {first_device.get('name')} (ID: {device_id})")
                    
                    # Тест запуска голосового управления
                    print("\n3. Запуск голосового управления...")
                    await websocket.send(json.dumps({
                        "type": "start_voice_control",
                        "model_size": "tiny",  # Используем tiny для быстрого теста
                        "language": "ru",
                        "device_id": device_id,
                        "channel": 0
                    }))
                    
                    print("   Запрос отправлен, ожидание ответа...")
                    
                    # Ждем ответ
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        data = json.loads(response)
                        print(f"\n   ✅ Получен ответ: {data.get('type')}")
                        print(f"   Данные: {json.dumps(data, indent=2, ensure_ascii=False)}")
                        
                        if data.get("type") == "voice_control_status":
                            if data.get("active"):
                                print("\n   ✅ Голосовое управление успешно запущено!")
                                
                                # Остановка через 5 секунд
                                print("\n4. Остановка через 5 секунд...")
                                await asyncio.sleep(5)
                                
                                await websocket.send(json.dumps({
                                    "type": "stop_voice_control"
                                }))
                                
                                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                                data = json.loads(response)
                                print(f"   ✅ Остановлено: {data.get('message')}")
                            else:
                                if data.get("error"):
                                    print(f"\n   ❌ Ошибка: {data.get('error')}")
                                else:
                                    print("\n   ⚠️  Голосовое управление не запустилось")
                    except asyncio.TimeoutError:
                        print("\n   ⚠️  Таймаут ожидания ответа (возможно, модель загружается...)")
                else:
                    print("   ⚠️  Нет доступных аудио устройств")
            else:
                print(f"   ⚠️  Неожиданный тип ответа: {data.get('type')}")
            
            print("\n" + "=" * 60)
            print("Тест завершен")
            print("=" * 60)
            
    except (OSError, ConnectionRefusedError, Exception) as e:
        if "Connect call failed" in str(e) or "Connection refused" in str(e):
            print(f"\n❌ Не удалось подключиться к {WS_URL}")
            print("   Убедитесь, что сервер запущен: python server.py")
        else:
            print(f"\n❌ Ошибка подключения: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_start_voice_control())
    exit(0 if success else 1)
