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


def list_all_snapshots(client: WingClient, max_index: int = 500):
    """
    Получить список всех snapshots
    """
    logger.info("=" * 70)
    logger.info(f"ПОЛУЧЕНИЕ СПИСКА ВСЕХ SNAPSHOTS (максимум {max_index} индексов)")
    logger.info("=" * 70)
    
    # Получаем текущий активный snapshot
    client.send("/$ctl/lib/$active")
    client.send("/$ctl/lib/$actidx")
    time.sleep(0.2)
    original_active = client.state.get("/$ctl/lib/$active")
    original_idx = client.state.get("/$ctl/lib/$actidx")
    
    logger.info(f"\nТекущий активный snapshot: [{original_idx}] {original_active}")
    logger.info(f"\nСканирование snapshots...")
    logger.info("ВНИМАНИЕ: Будет временно загружаться каждый snapshot для проверки имени\n")
    
    snapshots = {}
    seen_names = set()
    
    if original_active and original_idx is not None:
        snapshots[original_idx] = original_active
        seen_names.add(original_active)
        logger.info(f"  [{original_idx:4d}] {original_active}")
    
    for idx in range(1, max_index + 1):
        try:
            # Устанавливаем индекс
            client.send("/$ctl/lib/$actionidx", idx)
            time.sleep(0.1)
            
            # Отправляем GO для загрузки snapshot
            client.send("/$ctl/lib/$action", "GO")
            time.sleep(0.2)
            
            # Запрашиваем имя активной сцены
            client.send("/$ctl/lib/$active")
            time.sleep(0.1)
            
            active_name = client.state.get("/$ctl/lib/$active")
            
            # Если это новый snapshot
            if active_name and active_name not in seen_names:
                snapshots[idx] = active_name
                seen_names.add(active_name)
                logger.info(f"  [{idx:4d}] {active_name}")
            
        except Exception as e:
            logger.debug(f"Ошибка при проверке индекса {idx}: {e}")
        
        # Прогресс каждые 50 индексов
        if idx % 50 == 0:
            logger.info(f"  ... проверено {idx} индексов, найдено {len(snapshots)} snapshots ...")
    
    # Возвращаемся к оригинальному snapshot
    if original_idx is not None:
        logger.info(f"\nВозврат к оригинальному snapshot (индекс {original_idx})...")
        client.send("/$ctl/lib/$actionidx", original_idx)
        time.sleep(0.1)
        client.send("/$ctl/lib/$action", "GO")
        time.sleep(0.2)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"НАЙДЕНО SNAPSHOTS: {len(snapshots)}")
    logger.info("=" * 70)
    
    if snapshots:
        logger.info("\nСписок snapshots:")
        for idx, name in sorted(snapshots.items()):
            logger.info(f"  [{idx:4d}] {name}")
    else:
        logger.warning("Snapshots не найдены")
        logger.warning("Возможно, нужно увеличить MAX_INDEX или snapshots находятся в Show")
    
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
