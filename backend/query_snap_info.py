#!/usr/bin/env python3
"""
Запрос информации о текущем snapshot и списка доступных snapshots
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def query_snap_info(client: WingClient):
    """
    Запрос информации о snapshots через различные адреса
    """
    logger.info("=" * 70)
    logger.info("ЗАПРОС ИНФОРМАЦИИ О SNAPSHOTS")
    logger.info("=" * 70)
    
    # Адреса для запроса информации о snapshots
    query_addresses = [
        # Текущий snapshot
        "/snap/name",
        "/snap/current",
        "/snap/$name",
        "/snap/$current",
        "/scene/name",
        "/scene/current",
        "/scene/$name",
        "/scene/$current",
        
        # Список snapshots
        "/snap/list",
        "/snap/$list",
        "/scene/list",
        "/scene/$list",
        
        # Информация о snapshot
        "/snap",
        "/scene",
        "/show",
        
        # Возможно через node data
        "/snap ,s *",
        "/scene ,s *",
    ]
    
    logger.info("Запрашиваю информацию о snapshots...\n")
    
    for address in query_addresses:
        logger.info(f"Запрос: {address}")
        try:
            client.send(address)
            time.sleep(0.2)
            
            # Проверяем, что вернулось в state
            value = client.state.get(address)
            if value is not None:
                logger.info(f"  ✓ Получено: {value}")
            else:
                logger.info(f"  - Нет ответа")
        except Exception as e:
            logger.error(f"  ✗ Ошибка: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Все запросы выполнены. Проверьте ответы выше.")
    logger.info("=" * 70)


def main():
    """Основная функция"""
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.102"
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        return 1
    
    try:
        query_snap_info(client)
        time.sleep(1)
        logger.info("\nТекущее состояние state:")
        for key, value in sorted(client.state.items()):
            if 'snap' in key.lower() or 'scene' in key.lower() or 'show' in key.lower():
                logger.info(f"  {key} = {value}")
    except KeyboardInterrupt:
        logger.info("\n\nПрервано пользователем")
        return 1
    except Exception as e:
        logger.error(f"\nОшибка: {e}", exc_info=True)
        return 1
    finally:
        client.disconnect()
        logger.info("Отключено от пульта")


if __name__ == "__main__":
    sys.exit(main())
