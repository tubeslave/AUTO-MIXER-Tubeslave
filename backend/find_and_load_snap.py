#!/usr/bin/env python3
"""
Поиск snapshot по имени через перебор индексов и загрузка
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def scan_snapshots_by_index(client: WingClient, max_index: int = 200):
    """
    Сканирование snapshots по индексам
    
    Для каждого индекса:
    1. Устанавливаем /$ctl/lib/$actionidx = индекс
    2. Отправляем GO для временной загрузки (чтобы увидеть имя)
    3. Запрашиваем /$ctl/lib/$active для получения имени
    4. Возвращаемся к оригинальному snapshot
    """
    logger.info("=" * 70)
    logger.info("СКАНИРОВАНИЕ SNAPSHOTS ПО ИНДЕКСАМ")
    logger.info("=" * 70)
    
    # Получаем текущий активный snapshot и его индекс
    client.send("/$ctl/lib/$active")
    client.send("/$ctl/lib/$actidx")
    time.sleep(0.2)
    original_active = client.state.get("/$ctl/lib/$active")
    original_idx = client.state.get("/$ctl/lib/$actidx")
    
    logger.info(f"Текущий активный snapshot: {original_active}")
    logger.info(f"Текущий индекс: {original_idx}")
    
    snapshots = {}
    if original_active and original_idx is not None:
        snapshots[original_idx] = original_active
        logger.info(f"  [{original_idx:4d}] {original_active}")
    
    logger.info(f"\nСканирование индексов 1-{max_index}...")
    logger.info("ВНИМАНИЕ: Будет временно загружаться каждый snapshot для проверки имени")
    logger.info("Это может занять некоторое время...\n")
    
    found_count = 0
    seen_names = set()
    if original_active:
        seen_names.add(original_active)
    
    for idx in range(1, max_index + 1):
        try:
            # Устанавливаем индекс
            client.send("/$ctl/lib/$actionidx", idx)
            time.sleep(0.1)
            
            # Отправляем GO для загрузки snapshot (чтобы увидеть его имя)
            client.send("/$ctl/lib/$action", "GO")
            time.sleep(0.2)
            
            # Запрашиваем имя активной сцены
            client.send("/$ctl/lib/$active")
            time.sleep(0.1)
            
            active_name = client.state.get("/$ctl/lib/$active")
            
            # Если это новый snapshot (еще не видели)
            if active_name and active_name not in seen_names:
                snapshots[idx] = active_name
                seen_names.add(active_name)
                logger.info(f"  [{idx:4d}] {active_name}")
                found_count += 1
            
        except Exception as e:
            logger.debug(f"  Ошибка при проверке индекса {idx}: {e}")
        
        # Прогресс каждые 10 индексов
        if idx % 10 == 0:
            logger.info(f"  Проверено {idx} индексов, найдено {found_count} уникальных snapshots...")
    
    # Возвращаемся к оригинальному snapshot
    if original_idx is not None:
        logger.info(f"\nВозврат к оригинальному snapshot (индекс {original_idx})...")
        client.send("/$ctl/lib/$actionidx", original_idx)
        time.sleep(0.1)
        client.send("/$ctl/lib/$action", "GO")
        time.sleep(0.2)
    
    logger.info(f"\nСканирование завершено. Найдено {len(snapshots)} snapshots")
    return snapshots


def find_snap_index_by_name(snapshots: dict, search_name: str):
    """
    Поиск индекса snapshot по имени
    """
    search_name_upper = search_name.upper().strip()
    
    # Пробуем разные варианты поиска
    for idx, name in snapshots.items():
        if not name:
            continue
            
        name_upper = name.upper()
        
        # Точное совпадение
        if search_name_upper == name_upper:
            return idx, name
        
        # Совпадение без префикса и расширения
        name_clean = name_upper.replace("I:/", "").replace(".SNAP", "").strip()
        if search_name_upper == name_clean:
            return idx, name
        
        # Частичное совпадение
        if search_name_upper in name_upper or name_upper in search_name_upper:
            return idx, name
    
    return None, None


def load_snap_by_index(client: WingClient, snap_index: int):
    """
    Загрузка snapshot по индексу
    
    Согласно документации и инструкции пользователя:
    1. Выбрать snapshot по номеру: /$ctl/lib/$actionidx = индекс
    2. Загрузить кнопкой LOAD: возможно /$ctl/lib/$action = "LOAD" или "GO"
    """
    logger.info(f"\nЗагрузка snapshot по индексу: {snap_index}")
    
    try:
        # Шаг 1: Выбираем snapshot по номеру
        logger.info(f"  1. Установка индекса: /$ctl/lib/$actionidx = {snap_index}")
        result1 = client.send("/$ctl/lib/$actionidx", snap_index)
        time.sleep(0.2)
        
        if not result1:
            logger.error("     ✗ Ошибка установки индекса")
            return False
        
        logger.info(f"     ✓ Индекс установлен")
        
        # Проверяем, что индекс установлен
        client.send("/$ctl/lib/$actionidx")
        time.sleep(0.1)
        set_idx = client.state.get("/$ctl/lib/$actionidx")
        logger.info(f"     Проверка: установленный индекс = {set_idx}")
        
        # Шаг 2: Загружаем snapshot
        # Пробуем разные варианты команды загрузки
        load_commands = ["LOAD", "GO"]
        
        for cmd in load_commands:
            logger.info(f"  2. Отправка команды загрузки: /$ctl/lib/$action = '{cmd}'")
            result2 = client.send("/$ctl/lib/$action", cmd)
            time.sleep(0.3)
            
            if result2:
                logger.info(f"     ✓ Команда '{cmd}' отправлена")
                
                # Проверяем, изменилась ли активная сцена
                client.send("/$ctl/lib/$active")
                time.sleep(0.2)
                new_active = client.state.get("/$ctl/lib/$active")
                
                client.send("/$ctl/lib/$actidx")
                time.sleep(0.1)
                new_idx = client.state.get("/$ctl/lib/$actidx")
                
                logger.info(f"     Новая активная сцена: {new_active}")
                logger.info(f"     Новый индекс: {new_idx}")
                
                # Если сцена изменилась, значит загрузка успешна
                if new_active and new_idx == snap_index:
                    logger.info(f"     ✓ Snapshot загружен успешно!")
                    return True
        
        logger.warning("     ⚠ Команды отправлены, но не подтверждено изменение")
        return True  # Возвращаем True, так как команды отправлены
        
    except Exception as e:
        logger.error(f"  ✗ Ошибка при загрузке: {e}")
        return False


def main():
    """Основная функция"""
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python3 find_and_load_snap.py <IP> <SNAP_NAME> [MAX_INDEX]")
        print("")
        print("Пример:")
        print("  python3 find_and_load_snap.py 192.168.1.102 'HULI REPA AC'")
        print("  python3 find_and_load_snap.py 192.168.1.102 'HULI REPA AC' 500")
        return 1
    
    ip = sys.argv[1]
    snap_name = sys.argv[2]
    max_index = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    
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
        logger.info(f"\nТекущий активный snapshot: {original_active}")
        
        # Сканируем snapshots
        snapshots = scan_snapshots_by_index(client, max_index)
        
        if not snapshots:
            logger.error("\n✗ Не найдено ни одного snapshot")
            logger.error("Возможно, нужно увеличить MAX_INDEX или проверить подключение")
            return 1
        
        logger.info(f"\nНайдено {len(snapshots)} snapshots")
        
        # Ищем snapshot по имени
        snap_index, snap_full_name = find_snap_index_by_name(snapshots, snap_name)
        
        if snap_index is None:
            logger.error(f"\n✗ Snapshot '{snap_name}' не найден")
            logger.info("\nДоступные snapshots:")
            for idx, name in sorted(snapshots.items()):
                logger.info(f"  [{idx:4d}] {name}")
            return 1
        
        logger.info(f"\n✓ Найден snapshot: [{snap_index}] {snap_full_name}")
        
        # Загружаем snapshot
        logger.info("\n" + "=" * 70)
        success = load_snap_by_index(client, snap_index)
        
        if success:
            logger.info("\n" + "=" * 70)
            logger.info("✓ Команды загрузки snapshot отправлены")
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
