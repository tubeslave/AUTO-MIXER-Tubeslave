#!/usr/bin/env python3
"""
Поиск правильных OSC адресов для ROUTING/OUTPUTS
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_routing_addresses(client: WingClient, channel: int = 1):
    """Ищет адреса маршрутизации выходов"""
    
    logger.info(f"Поиск адресов ROUTING/OUTPUTS для канала {channel}")
    logger.info("=" * 70)
    
    # Варианты адресов на основе структуры ROUTING/OUTPUTS/SOURCE GROUP и OUTPUTS GROUP
    addresses_to_try = [
        # Вариант 1: /routing/outputs/...
        f"/routing/outputs/ch/{channel}/srcgrp",
        f"/routing/outputs/ch/{channel}/src/grp",
        f"/routing/outputs/ch/{channel}/source/grp",
        f"/routing/outputs/ch/{channel}/sourcegrp",
        f"/routing/outputs/ch/{channel}/outgrp",
        f"/routing/outputs/ch/{channel}/out/grp",
        f"/routing/outputs/ch/{channel}/output/grp",
        f"/routing/outputs/ch/{channel}/outputgrp",
        
        # Вариант 2: /routing/ch/{ch}/outputs/...
        f"/routing/ch/{channel}/outputs/srcgrp",
        f"/routing/ch/{channel}/outputs/src/grp",
        f"/routing/ch/{channel}/outputs/source/grp",
        f"/routing/ch/{channel}/outputs/outgrp",
        f"/routing/ch/{channel}/outputs/out/grp",
        f"/routing/ch/{channel}/outputs/output/grp",
        
        # Вариант 3: /outputs/...
        f"/outputs/ch/{channel}/srcgrp",
        f"/outputs/ch/{channel}/src/grp",
        f"/outputs/ch/{channel}/source/grp",
        f"/outputs/ch/{channel}/outgrp",
        f"/outputs/ch/{channel}/out/grp",
        f"/outputs/ch/{channel}/output/grp",
        
        # Вариант 4: /ch/{ch}/routing/outputs/...
        f"/ch/{channel}/routing/outputs/srcgrp",
        f"/ch/{channel}/routing/outputs/src/grp",
        f"/ch/{channel}/routing/outputs/source/grp",
        f"/ch/{channel}/routing/outputs/outgrp",
        f"/ch/{channel}/routing/outputs/out/grp",
        f"/ch/{channel}/routing/outputs/output/grp",
        
        # Вариант 5: С индексом выхода
        f"/routing/outputs/ch/{channel}/srcgrp/1",
        f"/routing/outputs/ch/{channel}/outgrp/1",
        f"/routing/outputs/ch/{channel}/src/grp/1",
        f"/routing/outputs/ch/{channel}/out/grp/1",
        
        # Вариант 6: Короткие варианты
        f"/routing/out/ch/{channel}/srcgrp",
        f"/routing/out/ch/{channel}/outgrp",
        
        # Вариант 7: Без указания канала (глобальные настройки)
        f"/routing/outputs/srcgrp",
        f"/routing/outputs/outgrp",
    ]
    
    logger.info("Запрос всех возможных адресов...")
    for addr in addresses_to_try:
        client.send(addr)
        time.sleep(0.02)
    
    time.sleep(0.5)
    
    logger.info("\nАдреса, которые вернули ответы:")
    logger.info("-" * 70)
    
    found_any = False
    for addr in addresses_to_try:
        value = client.state.get(addr)
        if value is not None:
            logger.info(f"  ✓ {addr:60s} = {value}")
            found_any = True
    
    if not found_any:
        logger.warning("  (нет ответов от пульта)")
        logger.info("\nПопробуем запросить node данные для /routing/...")
        
        # Пробуем запросить node данные
        node_addresses = [
            "/routing",
            "/routing/",
            "/routing/outputs",
            "/routing/outputs/",
            f"/routing/ch/{channel}",
            f"/routing/ch/{channel}/",
        ]
        
        for addr in node_addresses:
            logger.info(f"Запрос node: {addr}")
            client.send(addr)
            time.sleep(0.1)
        
        time.sleep(0.5)
        
        logger.info("\nNode данные:")
        for addr in node_addresses:
            value = client.state.get(addr)
            if value is not None:
                logger.info(f"  {addr:60s} = {value}")
                # Если это список параметров, пробуем запросить каждый
                if isinstance(value, (list, tuple)):
                    logger.info(f"    (node с {len(value)} параметрами)")
                    for param in value[:20]:  # Первые 20 параметров
                        if isinstance(param, str):
                            sub_addr = f"{addr}/{param}" if addr.endswith("/") else f"{addr}/{param}"
                            client.send(sub_addr)
                            time.sleep(0.02)
                    time.sleep(0.3)
                    
                    # Проверяем подпараметры
                    for sub_addr in sorted(client.state.keys()):
                        if sub_addr.startswith(addr):
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
        find_routing_addresses(client, 1)
    finally:
        client.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
