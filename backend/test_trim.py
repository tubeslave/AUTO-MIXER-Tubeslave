#!/usr/bin/env python3
"""
Тестовый скрипт для установки trim первого канала на -5db
"""
from wing_client import WingClient
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Включаем debug для детального логирования OSC сообщений
logging.getLogger('wing_client').setLevel(logging.DEBUG)


def on_trim_update(address, *values):
    """Callback для получения обновлений trim"""
    logger.info(f"📥 Получено обновление: {address} = {values}")


def main():
    logger.info("=== Тест установки Trim первого канала ===")
    logger.info("Устанавливаем trim канала 1 на -5db\n")
    
    # Параметры подключения (можно изменить)
    ip = "192.168.1.102"  # IP адрес Wing микшера
    port = 2223  # OSC порт
    
    # Если передан IP как аргумент командной строки
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    logger.info(f"Подключение к Wing микшеру: {ip}:{port}")
    
    client = WingClient(ip, port)
    
    # Подписываемся на все обновления для отладки
    client.subscribe("*", on_trim_update)
    
    if client.connect():
        logger.info("✓ Подключено успешно!")
        
        # Сначала запрашиваем текущее значение с разных адресов
        logger.info("\n=== Запрос текущего значения trim ===")
        addresses_to_try = [
            "/ch/01/preamp/trim",      # Из документации
            "/ch/1/preamp/trim",      # Без нуля
            "/ch/01/in/set/trim",      # Из wing_client.py
            "/ch/1/in/set/trim",       # Без нуля
        ]
        
        for addr in addresses_to_try:
            logger.info(f"Запрос: {addr}")
            client.send(addr)
            time.sleep(0.3)
        
        time.sleep(1.0)
        
        logger.info("\n=== Проверка полученных значений ===")
        for addr in addresses_to_try:
            value = client.state.get(addr)
            if value is not None:
                logger.info(f"✓ {addr} = {value}")
        
        # Показываем все параметры с trim/preamp
        logger.info("\n=== Все параметры с 'trim' или 'preamp' ===")
        trim_params = {k: v for k, v in client.state.items() if 'trim' in k.lower() or 'preamp' in k.lower() or 'in' in k.lower()}
        for k, v in sorted(trim_params.items()):
            logger.info(f"  {k} = {v}")
        
        # Устанавливаем trim на -5db
        logger.info("\n=== Установка trim на -5db ===")
        trim_value = -5.0  # в децибелах
        
        # Используем правильный адрес, который работает для чтения
        addresses_to_set = [
            "/ch/01/in/set/trim",      # Правильный адрес (с нулём)
            "/ch/1/in/set/trim",       # Без нуля
        ]
        
        for trim_address in addresses_to_set:
            logger.info(f"\nОтправка команды: {trim_address} = {trim_value} dB")
            success = client.send(trim_address, trim_value)
            
            if success:
                logger.info(f"✓ Команда отправлена на {trim_address}")
            else:
                logger.error(f"✗ Ошибка отправки на {trim_address}")
            
            time.sleep(0.5)
            
            # Сразу запрашиваем значение для проверки
            logger.info(f"Запрос значения после установки...")
            client.send(trim_address)
            time.sleep(0.5)
        
        # Ждем ответа
        logger.info("\nОжидание ответа от микшера (3 секунды)...")
        time.sleep(3.0)
        
        # Проверяем обновленные значения
        logger.info("\n=== Проверка обновленных значений ===")
        for addr in addresses_to_try:
            value = client.state.get(addr)
            if value is not None:
                logger.info(f"  {addr} = {value}")
        
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
