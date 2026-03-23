#!/usr/bin/env python3
"""
Тестовый скрипт для установки trim каналов 2-40 на -6db
"""
from wing_client import WingClient
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def on_trim_update(address, *values):
    """Callback для получения обновлений trim"""
    logger.info(f"📥 Получено обновление: {address} = {values}")


def main():
    logger.info("=== Тест установки Trim каналов 2-40 ===")
    logger.info("Устанавливаем trim каналов 2-40 на -6db\n")
    
    # Параметры подключения
    ip = "192.168.1.102"  # IP адрес Wing микшера
    port = 2223  # OSC порт
    
    # Если передан IP как аргумент командной строки
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    logger.info(f"Подключение к Wing микшеру: {ip}:{port}")
    
    client = WingClient(ip, port)
    
    # Подписываемся на обновления trim
    client.subscribe("*", lambda addr, *vals: logger.debug(f"OSC: {addr} = {vals}"))
    
    if client.connect():
        logger.info("✓ Подключено успешно!\n")
        
        # Устанавливаем trim на -6db для каналов 2-40
        trim_value = -6.0  # в децибелах
        start_channel = 2
        end_channel = 40
        
        logger.info(f"Установка trim на {trim_value} dB для каналов {start_channel}-{end_channel}")
        logger.info("=" * 60)
        
        success_count = 0
        failed_channels = []
        
        for channel in range(start_channel, end_channel + 1):
            # Используем формат с нулём (02, 03, ...)
            trim_address = f"/ch/{channel:02d}/in/set/trim"
            
            print(f"Канал {channel:2d}: Отправка команды на {trim_address} = {trim_value} dB ... ", end="")
            
            success = client.send(trim_address, trim_value)
            
            if success:
                print("✓")
                success_count += 1
            else:
                print("✗")
                failed_channels.append(channel)
            
            # Небольшая задержка между командами, чтобы не перегружать микшер
            time.sleep(0.05)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Установка завершена:")
        logger.info(f"  Успешно: {success_count} каналов")
        if failed_channels:
            logger.warning(f"  Ошибки: {len(failed_channels)} каналов - {failed_channels}")
        
        # Проверяем установленные значения
        logger.info("\n=== Проверка установленных значений ===")
        logger.info("Запрос значений trim для каналов 2-40...")
        
        # Запрашиваем значения для всех каналов
        for channel in range(start_channel, end_channel + 1):
            trim_address = f"/ch/{channel:02d}/in/set/trim"
            client.send(trim_address)
            time.sleep(0.02)
        
        # Ждем ответы
        time.sleep(2.0)
        
        # Показываем результаты
        logger.info("\nРезультаты проверки:")
        verified_count = 0
        for channel in range(start_channel, end_channel + 1):
            trim_address = f"/ch/{channel:02d}/in/set/trim"
            value = client.state.get(trim_address)
            if value is not None:
                if abs(value - trim_value) < 0.1:  # Проверка с небольшой погрешностью
                    logger.info(f"  Канал {channel:2d}: ✓ {value:6.1f} dB")
                    verified_count += 1
                else:
                    logger.warning(f"  Канал {channel:2d}: ⚠ {value:6.1f} dB (ожидалось {trim_value} dB)")
            else:
                logger.warning(f"  Канал {channel:2d}: ✗ значение не получено")
        
        logger.info(f"\nПроверено: {verified_count} каналов имеют значение {trim_value} dB")
        
        client.disconnect()
        logger.info("\n✓ Отключено")
    else:
        logger.error("✗ Не удалось подключиться к микшеру!")
        logger.error("Проверьте:")
        logger.error("  1. Микшер включен и подключен к сети")
        logger.error("  2. IP адрес правильный")
        logger.error("  3. OSC включен на микшере")
        logger.error("  4. Файрвол разрешает UDP порты 2222/2223")
        sys.exit(1)


if __name__ == "__main__":
    main()
