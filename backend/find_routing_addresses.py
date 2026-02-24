#!/usr/bin/env python3
"""
Скрипт для поиска правильных OSC адресов маршрутизации выходов на карту
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def query_all_possible_addresses(client: WingClient, channel: int = 22):
    """Запрашивает все возможные адреса маршрутизации для канала"""
    
    logger.info(f"Поиск правильных OSC адресов для маршрутизации канала {channel}")
    logger.info("=" * 60)
    
    # Список возможных адресов для проверки
    addresses_to_query = [
        # Output connection адреса
        f"/ch/{channel}/out",
        f"/ch/{channel}/out/",
        f"/ch/{channel}/out/conn",
        f"/ch/{channel}/out/conn/",
        f"/ch/{channel}/out/conn/grp",
        f"/ch/{channel}/out/conn/out",
        f"/ch/{channel}/out/conn/type",
        f"/ch/{channel}/out/conn/mode",
        
        # Альтернативные адреса
        f"/ch/{channel}/out/grp",
        f"/ch/{channel}/out/out",
        f"/ch/{channel}/out/type",
        f"/ch/{channel}/out/mode",
        
        # Routing адреса
        f"/ch/{channel}/routing",
        f"/ch/{channel}/routing/",
        f"/ch/{channel}/routing/out",
        f"/ch/{channel}/routing/out/grp",
        f"/ch/{channel}/routing/out/out",
        
        # Mix адреса (может быть связано с routing)
        f"/ch/{channel}/mix/out",
        f"/ch/{channel}/mix/out/grp",
        f"/ch/{channel}/mix/out/out",
        
        # Card адреса
        f"/card/1",
        f"/card/1/",
        f"/card/WLIVE",
        f"/card/WLIVE PLAY",
        
        # Routing matrix
        f"/routing/ch/{channel}/out",
        f"/routing/ch/{channel}/out/grp",
        f"/routing/ch/{channel}/out/out",
    ]
    
    logger.info("Запрос всех возможных адресов...")
    for addr in addresses_to_query:
        client.send(addr)
        time.sleep(0.03)
    
    time.sleep(0.5)  # Даем время на получение ответов
    
    logger.info("\nАдреса, которые вернули ответы:")
    logger.info("-" * 60)
    
    found_any = False
    for addr in addresses_to_query:
        value = client.state.get(addr)
        if value is not None:
            logger.info(f"  {addr:50s} = {value}")
            found_any = True
    
    if not found_any:
        logger.warning("  (нет ответов от пульта)")
        logger.info("\nВозможно, эти адреса не существуют или требуют другого формата запроса.")
        logger.info("Попробуйте проверить документацию Wing Remote Protocols для правильных адресов.")
    
    # Также проверим все адреса в state, которые содержат "out" или "card"
    logger.info("\nВсе адреса в state, содержащие 'out' или 'card':")
    logger.info("-" * 60)
    
    found_in_state = False
    for addr in sorted(client.state.keys()):
        if 'out' in addr.lower() or 'card' in addr.lower() or 'routing' in addr.lower():
            logger.info(f"  {addr:50s} = {client.state[addr]}")
            found_in_state = True
    
    if not found_in_state:
        logger.info("  (нет таких адресов в текущем state)")


def main():
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = "192.168.1.102"
    
    if len(sys.argv) > 2:
        channel = int(sys.argv[2])
    else:
        channel = 22  # Канал, который уже настроен на Dante
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться!")
        return 1
    
    try:
        query_all_possible_addresses(client, channel)
    finally:
        client.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
