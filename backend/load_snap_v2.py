#!/usr/bin/env python3
"""
Загрузка snapshot/scene по имени используя правильные адреса из документации
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_snap_by_name(client: WingClient, snap_name: str):
    """
    Загрузка snapshot по имени используя адреса из документации
    
    Согласно документации (стр. 79):
    - /$ctl/lib/$actionidx - Scene index user selection
    - /$ctl/lib/$action - Show control actions: GO, PREV, NEXT, etc.
    - /$ctl/lib/$active - Name of the active scene [RO]
    """
    logger.info("=" * 70)
    logger.info(f"ЗАГРУЗКА SNAPSHOT: {snap_name}")
    logger.info("=" * 70)
    
    # Сначала запрашиваем текущую активную сцену
    logger.info("\n1. Запрашиваю текущую активную сцену...")
    client.send("/$ctl/lib/$active")
    time.sleep(0.2)
    current_active = client.state.get("/$ctl/lib/$active")
    logger.info(f"   Текущая активная сцена: {current_active}")
    
    # Пробуем различные методы загрузки
    methods = [
        # Метод 1: Использовать имя как индекс (если это число)
        {
            "name": "Метод 1: /$ctl/lib/$actionidx с именем как строкой",
            "steps": [
                ("/$ctl/lib/$actionidx", snap_name),
                ("/$ctl/lib/$action", "GO"),
            ]
        },
        # Метод 2: Попробовать через /snap/load
        {
            "name": "Метод 2: /snap/load с именем",
            "steps": [
                ("/snap/load", snap_name),
            ]
        },
        # Метод 3: Попробовать через /scene/load
        {
            "name": "Метод 3: /scene/load с именем",
            "steps": [
                ("/scene/load", snap_name),
            ]
        },
        # Метод 4: Попробовать через /snap/recall
        {
            "name": "Метод 4: /snap/recall с именем",
            "steps": [
                ("/snap/recall", snap_name),
            ]
        },
        # Метод 5: Попробовать через /scene/recall
        {
            "name": "Метод 5: /scene/recall с именем",
            "steps": [
                ("/scene/recall", snap_name),
            ]
        },
        # Метод 6: Попробовать через полный путь с правильным форматом
        {
            "name": "Метод 6: /snap/load с полным путем (I:/NAME.snap)",
            "steps": [
                ("/snap/load", f"I:/{snap_name}.snap"),
            ]
        },
        # Метод 7: Попробовать через /$ctl/lib/$actionidx с полным путем
        {
            "name": "Метод 7: /$ctl/lib/$actionidx с полным путем (I:/NAME.snap) + GO",
            "steps": [
                ("/$ctl/lib/$actionidx", f"I:/{snap_name}.snap"),
                ("/$ctl/lib/$action", "GO"),
            ]
        },
        # Метод 8: Попробовать через /snap/recall с полным путем
        {
            "name": "Метод 8: /snap/recall с полным путем (I:/NAME.snap)",
            "steps": [
                ("/snap/recall", f"I:/{snap_name}.snap"),
            ]
        },
    ]
    
    for method in methods:
        logger.info(f"\n{method['name']}:")
        success = True
        for address, value in method['steps']:
            logger.info(f"  Отправка: {address} = {value}")
            result = client.send(address, value)
            time.sleep(0.2)
            if not result:
                logger.warning(f"    ✗ Ошибка отправки")
                success = False
                break
            else:
                logger.info(f"    ✓ Команда отправлена")
        
        if success:
            # Проверяем, изменилась ли активная сцена
            time.sleep(0.5)
            client.send("/$ctl/lib/$active")
            time.sleep(0.2)
            new_active = client.state.get("/$ctl/lib/$active")
            logger.info(f"  Новая активная сцена: {new_active}")
            
            if new_active and new_active != current_active:
                logger.info(f"  ✓ Сцена изменилась! Возможно, загрузка успешна.")
                logger.info(f"  Проверьте на пульте, загрузилась ли сцена '{snap_name}'")
                return True
            else:
                logger.info(f"  → Сцена не изменилась. Пробуем следующий метод...")
    
    logger.warning("\n" + "=" * 70)
    logger.warning("✗ Не удалось загрузить snapshot ни одним из методов")
    logger.warning("Проверьте на пульте, возможно нужен другой формат имени")
    logger.warning("=" * 70)
    return False


def main():
    """Основная функция"""
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python3 load_snap_v2.py <IP> <SNAP_NAME>")
        print("")
        print("Пример:")
        print("  python3 load_snap_v2.py 192.168.1.102 'HULI REPA AC'")
        return 1
    
    ip = sys.argv[1]
    snap_name = sys.argv[2]
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        return 1
    
    try:
        success = load_snap_by_name(client, snap_name)
        return 0 if success else 1
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
