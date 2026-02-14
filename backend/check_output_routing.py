#!/usr/bin/env python3
"""
Скрипт для проверки текущей маршрутизации выходов каналов
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_channel_output_routing(client: WingClient, channel: int):
    """Проверяет все возможные адреса маршрутизации выходов канала"""
    logger.info(f"\n=== Проверка маршрутизации канала {channel} ===")
    
    # Список возможных адресов для проверки
    addresses_to_check = [
        f"/ch/{channel}/out/conn/grp",
        f"/ch/{channel}/out/conn/out",
        f"/ch/{channel}/out/grp",
        f"/ch/{channel}/out/out",
        f"/ch/{channel}/out/conn/type",  # Тип выхода (user signal vs card)
        f"/ch/{channel}/out/type",
        f"/ch/{channel}/out/mode",
        f"/ch/{channel}/out/conn/mode",
    ]
    
    logger.info("Запрос текущих значений...")
    for addr in addresses_to_check:
        client.send(addr)
        time.sleep(0.05)
    
    time.sleep(0.3)  # Даем время на получение ответов
    
    logger.info("\nТекущие значения:")
    for addr in addresses_to_check:
        value = client.state.get(addr)
        if value is not None:
            logger.info(f"  {addr:40s} = {value}")
        else:
            logger.debug(f"  {addr:40s} = (нет ответа)")


def main():
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = "192.168.1.102"
    
    if len(sys.argv) > 2:
        channel = int(sys.argv[2])
    else:
        channel = 1
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться!")
        return 1
    
    try:
        check_channel_output_routing(client, channel)
        
        # Проверяем еще несколько каналов для сравнения
        if channel == 1:
            logger.info("\n" + "="*60)
            logger.info("Проверка канала 22 (который уже настроен на Dante):")
            check_channel_output_routing(client, 22)
        
    finally:
        client.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
