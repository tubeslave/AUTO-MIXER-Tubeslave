#!/usr/bin/env python3
"""
Тестовый скрипт для проверки правильных OSC адресов маршрутизации на карту
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_routing_addresses(client: WingClient, channel: int = 1):
    """Тестирует различные варианты OSC адресов для маршрутизации на карту"""
    
    logger.info(f"Тестирование маршрутизации канала {channel} на карту WLIVE PLAY")
    logger.info("=" * 60)
    
    # Варианты адресов для тестирования
    test_cases = [
        # Формат 1: Прямая установка группы и выхода
        {
            "name": "Метод 1: /ch/{ch}/out/conn/grp и /ch/{ch}/out/conn/out",
            "addresses": [
                (f"/ch/{channel}/out/conn/grp", "WLIVE PLAY"),
                (f"/ch/{channel}/out/conn/out", 1),
            ]
        },
        # Формат 2: Альтернативные адреса
        {
            "name": "Метод 2: /ch/{ch}/out/grp и /ch/{ch}/out/out",
            "addresses": [
                (f"/ch/{channel}/out/grp", "WLIVE PLAY"),
                (f"/ch/{channel}/out/out", 1),
            ]
        },
        # Формат 3: С указанием типа карты
        {
            "name": "Метод 3: С типом 'card'",
            "addresses": [
                (f"/ch/{channel}/out/conn/type", "card"),
                (f"/ch/{channel}/out/conn/grp", "WLIVE PLAY"),
                (f"/ch/{channel}/out/conn/out", 1),
            ]
        },
        # Формат 4: Через node данные
        {
            "name": "Метод 4: Через node /ch/{ch}/out/conn",
            "addresses": [
                (f"/ch/{channel}/out/conn", {"grp": "WLIVE PLAY", "out": 1}),
            ]
        },
        # Формат 5: Различные варианты имени карты
        {
            "name": "Метод 5: card(WLIVE PLAY)",
            "addresses": [
                (f"/ch/{channel}/out/conn/grp", "card(WLIVE PLAY)"),
                (f"/ch/{channel}/out/conn/out", 1),
            ]
        },
        {
            "name": "Метод 6: CARD(WLIVE PLAY)",
            "addresses": [
                (f"/ch/{channel}/out/conn/grp", "CARD(WLIVE PLAY)"),
                (f"/ch/{channel}/out/conn/out", 1),
            ]
        },
        # Формат 7: Через индекс карты (может быть числовым)
        {
            "name": "Метод 7: Индекс карты как число",
            "addresses": [
                (f"/ch/{channel}/out/conn/grp", 0),  # Возможно, карта имеет индекс
                (f"/ch/{channel}/out/conn/out", 1),
            ]
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{test_case['name']}:")
        logger.info("-" * 60)
        
        for addr, value in test_case['addresses']:
            try:
                if isinstance(value, dict):
                    # Для node данных нужно отправить JSON
                    logger.info(f"  Отправка: {addr} (node data)")
                    # Пока пропускаем node данные, так как нужен специальный формат
                else:
                    logger.info(f"  Отправка: {addr} = {value}")
                    result = client.send(addr, value)
                    time.sleep(0.05)
                    
                    if result:
                        logger.info(f"    ✓ Успешно отправлено")
                    else:
                        logger.warning(f"    ✗ Ошибка отправки")
            except Exception as e:
                logger.error(f"    ✗ Ошибка: {e}")
        
        time.sleep(0.2)
        
        # Проверяем результат
        logger.info("  Проверка результата...")
        time.sleep(0.3)
        
        # Запрашиваем текущие значения
        check_addrs = [
            f"/ch/{channel}/out/conn/grp",
            f"/ch/{channel}/out/conn/out",
            f"/ch/{channel}/out/grp",
            f"/ch/{channel}/out/out",
        ]
        
        for check_addr in check_addrs:
            client.send(check_addr)
            time.sleep(0.05)
        
        time.sleep(0.2)
        
        found_values = False
        for check_addr in check_addrs:
            value = client.state.get(check_addr)
            if value is not None:
                logger.info(f"    {check_addr:40s} = {value}")
                found_values = True
        
        if not found_values:
            logger.warning("    (нет ответа от пульта)")
        
        logger.info("")


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
        test_routing_addresses(client, channel)
        logger.info("\n" + "=" * 60)
        logger.info("Тестирование завершено. Проверьте на пульте, какой метод сработал.")
    finally:
        client.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
