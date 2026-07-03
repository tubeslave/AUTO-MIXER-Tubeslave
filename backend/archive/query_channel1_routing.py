#!/usr/bin/env python3
"""
Запрос текущего состояния маршрутизации канала 1
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def query_channel_routing(client: WingClient, channel: int = 1):
    """Запрашивает все возможные адреса маршрутизации канала"""
    
    logger.info(f"Запрос маршрутизации канала {channel}...")
    
    # Расширенный список адресов для проверки
    addresses = [
        # Основные адреса
        f"/ch/{channel}",
        f"/ch/{channel}/",
        
        # Output адреса
        f"/ch/{channel}/out",
        f"/ch/{channel}/out/",
        f"/ch/{channel}/out/conn",
        f"/ch/{channel}/out/conn/",
        f"/ch/{channel}/out/conn/grp",
        f"/ch/{channel}/out/conn/out",
        f"/ch/{channel}/out/conn/type",
        f"/ch/{channel}/out/conn/mode",
        f"/ch/{channel}/out/grp",
        f"/ch/{channel}/out/out",
        f"/ch/{channel}/out/type",
        f"/ch/{channel}/out/mode",
        
        # Routing адреса
        f"/ch/{channel}/routing",
        f"/ch/{channel}/routing/",
        f"/ch/{channel}/routing/out",
        
        # Mix адреса
        f"/ch/{channel}/mix",
        f"/ch/{channel}/mix/",
        f"/ch/{channel}/mix/out",
        
        # Signal адреса
        f"/ch/{channel}/signal",
        f"/ch/{channel}/signal/",
        f"/ch/{channel}/signal/out",
        
        # Config адреса
        f"/ch/{channel}/config",
        f"/ch/{channel}/config/out",
    ]
    
    logger.info("Отправка запросов...")
    for addr in addresses:
        client.send(addr)
        time.sleep(0.02)
    
    time.sleep(0.5)
    
    logger.info("\nНайденные адреса и их значения:")
    logger.info("=" * 70)
    
    found_any = False
    for addr in addresses:
        value = client.state.get(addr)
        if value is not None:
            logger.info(f"{addr:50s} = {value}")
            found_any = True
    
    if not found_any:
        logger.warning("Адреса маршрутизации не найдены в ответах.")
        logger.info("\nПопробуем запросить node данные...")
        
        # Пробуем запросить node данные
        node_addresses = [
            f"/ch/{channel}/out/conn",
            f"/ch/{channel}/out",
            f"/ch/{channel}/routing",
        ]
        
        for addr in node_addresses:
            client.send(addr)
            time.sleep(0.1)
        
        time.sleep(0.5)
        
        logger.info("\nNode данные:")
        for addr in node_addresses:
            value = client.state.get(addr)
            if value is not None:
                logger.info(f"{addr:50s} = {value}")
    
    # Выводим все адреса канала, которые содержат что-то связанное с выходом
    logger.info("\nВсе адреса канала в state:")
    logger.info("=" * 70)
    
    channel_addresses = [addr for addr in sorted(client.state.keys()) 
                        if addr.startswith(f"/ch/{channel}/")]
    
    for addr in channel_addresses:
        value = client.state[addr]
        logger.info(f"{addr:50s} = {value}")


def main():
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = "192.168.1.102"
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться!")
        return 1
    
    try:
        query_channel_routing(client, 1)
    finally:
        client.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
