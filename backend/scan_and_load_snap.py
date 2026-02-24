#!/usr/bin/env python3
"""
Сканирование snapshots по номерам и загрузка по имени
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def scan_snapshots(client: WingClient, max_snapshots: int = 1000):
    """
    Сканирование snapshots по номерам
    
    Пробует получить информацию о snapshots через различные методы
    """
    logger.info("=" * 70)
    logger.info("СКАНИРОВАНИЕ SNAPSHOTS")
    logger.info("=" * 70)
    
    snapshots = {}
    
    # Метод 1: Пробуем получить список через node data
    logger.info("\nМетод 1: Запрос node data для /$ctl/lib")
    try:
        # Пробуем получить node data для библиотеки
        client.send("/$ctl/lib ,s *")
        time.sleep(0.5)
        # Проверяем state на наличие данных
        for key in client.state.keys():
            if 'lib' in key.lower() or 'snap' in key.lower() or 'scene' in key.lower():
                logger.info(f"  Найдено: {key} = {client.state.get(key)}")
    except Exception as e:
        logger.error(f"  Ошибка: {e}")
    
    # Метод 2: Пробуем перебрать индексы snapshots
    logger.info(f"\nМетод 2: Перебор snapshots по индексам (1-{max_snapshots})")
    logger.info("Это может занять некоторое время...")
    
    # Сначала запрашиваем текущий активный snapshot
    client.send("/$ctl/lib/$active")
    time.sleep(0.2)
    current_active = client.state.get("/$ctl/lib/$active")
    logger.info(f"Текущий активный snapshot: {current_active}")
    
    # Пробуем получить информацию о snapshots через перебор индексов
    # Согласно документации, можно использовать /$ctl/lib/$actionidx для выбора
    # и затем запросить /$ctl/lib/$active для получения имени
    
    found_count = 0
    for idx in range(1, min(max_snapshots + 1, 101)):  # Ограничим до 100 для начала
        try:
            # Устанавливаем индекс
            client.send("/$ctl/lib/$actionidx", idx)
            time.sleep(0.1)
            
            # Запрашиваем имя активной сцены
            client.send("/$ctl/lib/$active")
            time.sleep(0.1)
            
            active_name = client.state.get("/$ctl/lib/$active")
            
            if active_name and active_name != current_active:
                # Если имя изменилось, значит это другой snapshot
                snapshots[idx] = active_name
                logger.info(f"  [{idx:4d}] {active_name}")
                found_count += 1
                
                # Возвращаемся к текущему активному
                if current_active:
                    # Пробуем найти индекс текущего активного
                    pass
        except Exception as e:
            logger.debug(f"  Ошибка при проверке индекса {idx}: {e}")
        
        if idx % 10 == 0:
            logger.info(f"  Проверено {idx} индексов, найдено {found_count} snapshots...")
    
    logger.info(f"\nНайдено snapshots: {found_count}")
    return snapshots


def find_snap_by_name(client: WingClient, snap_name: str, max_snapshots: int = 1000):
    """
    Поиск snapshot по имени и возврат его индекса
    """
    logger.info(f"\nПоиск snapshot по имени: '{snap_name}'")
    
    # Нормализуем имя для поиска
    search_name = snap_name.upper().strip()
    
    # Пробуем разные форматы имени
    name_variants = [
        snap_name,
        f"I:/{snap_name}.snap",
        f"I:/{snap_name}",
        f"{snap_name}.snap",
    ]
    
    snapshots = scan_snapshots(client, max_snapshots)
    
    # Ищем совпадение
    for idx, name in snapshots.items():
        name_upper = name.upper() if name else ""
        for variant in name_variants:
            if variant.upper() in name_upper or name_upper in variant.upper():
                logger.info(f"✓ Найден snapshot: [{idx}] {name}")
                return idx, name
    
    logger.warning(f"✗ Snapshot '{snap_name}' не найден")
    logger.info("\nДоступные snapshots:")
    for idx, name in sorted(snapshots.items()):
        logger.info(f"  [{idx:4d}] {name}")
    
    return None, None


def load_snap_by_index(client: WingClient, snap_index: int):
    """
    Загрузка snapshot по индексу
    
    Согласно документации:
    1. Установить /$ctl/lib/$actionidx = индекс
    2. Отправить /$ctl/lib/$action = "GO" для загрузки
    """
    logger.info(f"\nЗагрузка snapshot по индексу: {snap_index}")
    
    try:
        # Шаг 1: Устанавливаем индекс snapshot
        result1 = client.send("/$ctl/lib/$actionidx", snap_index)
        time.sleep(0.1)
        
        if not result1:
            logger.error("  ✗ Ошибка установки индекса")
            return False
        
        logger.info(f"  ✓ Индекс {snap_index} установлен")
        
        # Шаг 2: Отправляем команду GO для загрузки
        result2 = client.send("/$ctl/lib/$action", "GO")
        time.sleep(0.2)
        
        if not result2:
            logger.error("  ✗ Ошибка отправки команды GO")
            return False
        
        logger.info(f"  ✓ Команда GO отправлена")
        
        # Проверяем, изменилась ли активная сцена
        client.send("/$ctl/lib/$active")
        time.sleep(0.2)
        new_active = client.state.get("/$ctl/lib/$active")
        logger.info(f"  Новая активная сцена: {new_active}")
        
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Ошибка при загрузке: {e}")
        return False


def main():
    """Основная функция"""
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python3 scan_and_load_snap.py <IP> <SNAP_NAME> [MAX_SNAPSHOTS]")
        print("")
        print("Пример:")
        print("  python3 scan_and_load_snap.py 192.168.1.102 'HULI REPA AC'")
        print("  python3 scan_and_load_snap.py 192.168.1.102 'HULI REPA AC' 500")
        return 1
    
    ip = sys.argv[1]
    snap_name = sys.argv[2]
    max_snapshots = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        return 1
    
    try:
        # Сохраняем текущий активный snapshot
        client.send("/$ctl/lib/$active")
        time.sleep(0.2)
        original_active = client.state.get("/$ctl/lib/$active")
        logger.info(f"Текущий активный snapshot: {original_active}")
        
        # Ищем snapshot по имени
        snap_index, snap_full_name = find_snap_by_name(client, snap_name, max_snapshots)
        
        if snap_index is None:
            logger.error("\n✗ Snapshot не найден")
            return 1
        
        # Загружаем snapshot по найденному индексу
        logger.info("\n" + "=" * 70)
        success = load_snap_by_index(client, snap_index)
        
        if success:
            logger.info("\n" + "=" * 70)
            logger.info("✓ Snapshot загружен")
            logger.info("Проверьте на пульте, применилась ли сцена")
            logger.info("=" * 70)
            return 0
        else:
            logger.error("\n" + "=" * 70)
            logger.error("✗ Ошибка загрузки snapshot")
            logger.error("=" * 70)
            return 1
            
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
