#!/usr/bin/env python3
"""
Запрос node данных канала для поиска адресов маршрутизации
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_channel_node(client: WingClient, channel: int = 1):
    """Запрашивает node данные канала"""
    
    logger.info(f"Запрос node данных канала {channel}...")
    
    # Запрашиваем корневой node канала
    node_address = f"/ch/{channel}/"
    logger.info(f"Запрос: {node_address}")
    client.send(node_address)
    time.sleep(0.2)
    
    # Проверяем ответ
    node_value = client.state.get(node_address)
    if node_value:
        logger.info(f"\nNode данные канала {channel}:")
        logger.info("=" * 70)
        logger.info(f"{node_value}")
        
        # Если это список параметров, пробуем запросить каждый
        if isinstance(node_value, (list, tuple)):
            logger.info(f"\nНайдено {len(node_value)} параметров в node")
            logger.info("Пробую запросить каждый параметр...")
            
            for param in node_value:
                if isinstance(param, str):
                    param_addr = f"/ch/{channel}/{param}"
                    logger.info(f"  Запрос: {param_addr}")
                    client.send(param_addr)
                    time.sleep(0.05)
            
            time.sleep(0.3)
            
            # Выводим все найденные адреса
            logger.info("\nНайденные адреса канала:")
            logger.info("=" * 70)
            for addr in sorted(client.state.keys()):
                if addr.startswith(f"/ch/{channel}/"):
                    value = client.state[addr]
                    logger.info(f"{addr:60s} = {value}")
    
    # Также пробуем запросить возможные адреса маршрутизации напрямую
    logger.info("\n" + "=" * 70)
    logger.info("Прямой запрос возможных адресов маршрутизации:")
    logger.info("=" * 70)
    
    routing_addresses = [
        f"/ch/{channel}/out/conn",
        f"/ch/{channel}/out",
        f"/ch/{channel}/config",
        f"/ch/{channel}/routing",
    ]
    
    for addr in routing_addresses:
        logger.info(f"Запрос: {addr}")
        client.send(addr)
        time.sleep(0.1)
    
    time.sleep(0.5)
    
    logger.info("\nРезультаты:")
    for addr in routing_addresses:
        value = client.state.get(addr)
        if value is not None:
            logger.info(f"  {addr:60s} = {value}")
            # Если это node, пробуем запросить его содержимое
            if isinstance(value, (list, tuple)):
                logger.info(f"    (node с {len(value)} параметрами)")
                for param in value[:10]:  # Первые 10 параметров
                    if isinstance(param, str):
                        sub_addr = f"{addr}/{param}"
                        client.send(sub_addr)
                        time.sleep(0.02)
                time.sleep(0.2)
                
                # Проверяем подпараметры
                for sub_addr in sorted(client.state.keys()):
                    if sub_addr.startswith(addr + "/"):
                        logger.info(f"      {sub_addr:58s} = {client.state[sub_addr]}")


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
        get_channel_node(client, 1)
    finally:
        client.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
