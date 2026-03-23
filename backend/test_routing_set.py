#!/usr/bin/env python3
"""
Тестовый скрипт для установки маршрутизации через различные возможные адреса
Пользователь должен проверить на пульте, какой вариант сработал
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_routing_set(client: WingClient, channel: int = 1, dante_output: int = 1):
    """Пробует установить маршрутизацию через различные адреса"""
    
    logger.info(f"Тестирование установки маршрутизации канала {channel} на Dante выход {dante_output}")
    logger.info("=" * 70)
    logger.info("Проверяйте на пульте после каждого метода!")
    logger.info("=" * 70)
    
    # Варианты адресов для установки маршрутизации
    test_methods = [
        {
            "name": "Метод 1: /routing/outputs/ch/{ch}/srcgrp и /routing/outputs/ch/{ch}/outgrp",
            "addresses": [
                (f"/routing/outputs/ch/{channel}/srcgrp", "WLIVE PLAY"),
                (f"/routing/outputs/ch/{channel}/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 2: /routing/outputs/ch/{ch}/src/grp и /routing/outputs/ch/{ch}/out/grp",
            "addresses": [
                (f"/routing/outputs/ch/{channel}/src/grp", "WLIVE PLAY"),
                (f"/routing/outputs/ch/{channel}/out/grp", dante_output),
            ]
        },
        {
            "name": "Метод 3: /routing/ch/{ch}/outputs/srcgrp и /routing/ch/{ch}/outputs/outgrp",
            "addresses": [
                (f"/routing/ch/{channel}/outputs/srcgrp", "WLIVE PLAY"),
                (f"/routing/ch/{channel}/outputs/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 4: /outputs/ch/{ch}/srcgrp и /outputs/ch/{ch}/outgrp",
            "addresses": [
                (f"/outputs/ch/{channel}/srcgrp", "WLIVE PLAY"),
                (f"/outputs/ch/{channel}/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 5: /ch/{ch}/routing/outputs/srcgrp и /ch/{ch}/routing/outputs/outgrp",
            "addresses": [
                (f"/ch/{channel}/routing/outputs/srcgrp", "WLIVE PLAY"),
                (f"/ch/{channel}/routing/outputs/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 6: С числовыми индексами (WLIVE может быть индексом)",
            "addresses": [
                (f"/routing/outputs/ch/{channel}/srcgrp", 0),  # Возможно, WLIVE PLAY это индекс 0
                (f"/routing/outputs/ch/{channel}/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 7: Альтернативные названия - sourcegrp и outputgrp",
            "addresses": [
                (f"/routing/outputs/ch/{channel}/sourcegrp", "WLIVE PLAY"),
                (f"/routing/outputs/ch/{channel}/outputgrp", dante_output),
            ]
        },
        {
            "name": "Метод 8: Без указания канала (глобальная настройка)",
            "addresses": [
                (f"/routing/outputs/srcgrp", "WLIVE PLAY"),
                (f"/routing/outputs/outgrp", dante_output),
            ]
        },
    ]
    
    for i, method in enumerate(test_methods, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"{method['name']}")
        logger.info(f"{'='*70}")
        
        input(f"Нажмите Enter, чтобы попробовать метод {i}...")
        
        for addr, value in method['addresses']:
            try:
                logger.info(f"  Отправка: {addr} = {value}")
                result = client.send(addr, value)
                time.sleep(0.1)
                if result:
                    logger.info(f"    ✓ Успешно отправлено")
                else:
                    logger.warning(f"    ✗ Ошибка отправки")
            except Exception as e:
                logger.error(f"    ✗ Ошибка: {e}")
        
        logger.info(f"\nПроверьте на пульте - изменилась ли маршрутизация канала {channel}?")
        response = input("Сработал этот метод? (y/n): ")
        
        if response.lower() == 'y':
            logger.info(f"\n{'='*70}")
            logger.info(f"✓ НАЙДЕН ПРАВИЛЬНЫЙ МЕТОД: {method['name']}")
            logger.info(f"{'='*70}")
            for addr, value in method['addresses']:
                logger.info(f"  {addr} = {value}")
            return True
    
    logger.warning("\nНи один из методов не сработал.")
    return False


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
        test_routing_set(client, channel=1, dante_output=1)
    finally:
        client.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
