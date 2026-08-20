#!/usr/bin/env python3
"""
Список всех snapshots с их индексами и именами
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_READ_ONLY_SNAPSHOT_INDEX = 500


def list_all_snapshots(client: WingClient, max_index: int = 500):
    """
    Получить список snapshots без загрузки или изменения активной сцены.
    """
    if type(max_index) is not int or not 1 <= max_index <= MAX_READ_ONLY_SNAPSHOT_INDEX:
        raise ValueError(
            f"max_index must be an integer in range 1..{MAX_READ_ONLY_SNAPSHOT_INDEX}"
        )
    logger.info("=" * 70)
    logger.info(f"ЧТЕНИЕ СПИСКА SNAPSHOTS (максимум {max_index} индексов)")
    logger.info("=" * 70)

    # Read current scene and library cursor. The cursor is restored in finally.
    client.send("/$ctl/lib/$active")
    client.send("/$ctl/lib/$actidx")
    client.send("/$ctl/lib/$actionidx")
    time.sleep(0.2)
    original_active = client.state.get("/$ctl/lib/$active")
    original_idx = client.state.get("/$ctl/lib/$actidx")
    original_cursor = client.state.get("/$ctl/lib/$actionidx")

    logger.info(f"\nТекущий активный snapshot: [{original_idx}] {original_active}")
    logger.info("\nЧтение имён snapshots без команд LOAD/GO...\n")

    snapshots = {}
    try:
        for idx in range(1, max_index + 1):
            try:
                client.send("/$ctl/lib/$actionidx", idx)
                time.sleep(0.05)

                # Clear stale cached data, then query the selected library name.
                client.state.pop("/$ctl/lib/$name", None)
                client.send("/$ctl/lib/$name")
                time.sleep(0.05)
                name = client.state.get("/$ctl/lib/$name")
                if isinstance(name, str) and name.strip():
                    snapshots[idx] = name
                    logger.info(f"  [{idx:4d}] {name}")
            except Exception as exc:
                logger.debug(f"Ошибка при чтении индекса {idx}: {exc}")

            if idx % 50 == 0:
                logger.info(
                    f"  ... проверено {idx} индексов, найдено {len(snapshots)} snapshots ..."
                )
    finally:
        if type(original_cursor) is int:
            client.send("/$ctl/lib/$actionidx", original_cursor)
            time.sleep(0.05)

    logger.info("\n" + "=" * 70)
    logger.info(f"НАЙДЕНО SNAPSHOTS: {len(snapshots)}")
    logger.info("=" * 70)
    
    if snapshots:
        logger.info("\nСписок snapshots:")
        for idx, name in sorted(snapshots.items()):
            logger.info(f"  [{idx:4d}] {name}")
    else:
        logger.warning("Snapshots не найдены")
        logger.warning("Возможно, нужно увеличить MAX_INDEX или проверить активный Show")

    return snapshots


def main():
    """Основная функция"""
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.102"
    max_index = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        return 1
    
    try:
        snapshots = list_all_snapshots(client, max_index)
        return 0
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
