#!/usr/bin/env python3
"""
Финальная версия загрузки snapshot по имени или индексу
"""

import sys
import time
import logging
from wing_client import WingClient
from maintenance_cli_guard import (
    extract_maintenance_cli_args,
    validate_maintenance_cli_request,
    format_block_message,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Основная функция"""
    try:
        args, maintenance_ctx = extract_maintenance_cli_args(sys.argv[1:])
    except ValueError as exc:
        print(str(exc))
        print("Использование: python3 load_snap_final.py <IP> <SNAP_NAME_OR_INDEX> [MAX_INDEX]")
        print("  --confirm-maintenance")
        print("  --rollback-snapshot PATH")
        print("  --dry-run")
        return 1

    if len(args) < 2:
        print("Использование:")
        print("  python3 load_snap_final.py <IP> <SNAP_NAME_OR_INDEX> [MAX_INDEX]")
        print("")
        print("Примеры:")
        print("  # Загрузка по имени:")
        print("  python3 load_snap_final.py 192.168.1.102 'HULI REPA AC'")
        print("")
        print("  # Загрузка по индексу:")
        print("  python3 load_snap_final.py 192.168.1.102 5")
        print("")
        print("  # С указанием максимального индекса для поиска:")
        print("  python3 load_snap_final.py 192.168.1.102 'HULI REPA AC' 500")
        return 1
    
    ip = args[0]
    snap_param = args[1]
    max_index = int(args[2]) if len(args) > 2 else 500

    approved, reason = validate_maintenance_cli_request(
        operation="load_snapshot",
        confirm_maintenance=maintenance_ctx["confirm_maintenance"],
        rollback_snapshot_path=maintenance_ctx["rollback_snapshot_path"],
        dry_run=maintenance_ctx["dry_run"],
    )
    if not approved:
        logger.error(format_block_message(reason))
        return 1

    if maintenance_ctx["dry_run"]:
        logger.info("Dry run: would load snapshot '%s' on %s", snap_param, ip)
        return 0
    
    # Проверяем, является ли параметр числом (индексом)
    try:
        snap_index = int(snap_param)
        use_index = True
        snap_name = None
    except ValueError:
        use_index = False
        snap_name = snap_param
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        return 1
    
    try:
        # Получаем информацию о текущем состоянии
        client.send("/$ctl/lib/$active")
        client.send("/$ctl/lib/$actidx")
        client.send("/$ctl/lib/$actshow")
        time.sleep(0.2)
        
        current_active = client.state.get("/$ctl/lib/$active")
        current_idx = client.state.get("/$ctl/lib/$actidx")
        current_show = client.state.get("/$ctl/lib/$actshow")
        
        logger.info("=" * 70)
        logger.info("ТЕКУЩЕЕ СОСТОЯНИЕ")
        logger.info("=" * 70)
        logger.info(f"Активный snapshot: [{current_idx}] {current_active}")
        logger.info(f"Активный Show: {current_show if current_show else '(нет)'}")
        logger.info("=" * 70)
        
        if use_index:
            # Загрузка по индексу
            logger.info(f"\nЗагрузка snapshot по индексу: {snap_index}")
            success = client.load_snap_by_index(snap_index)
        else:
            # Загрузка по имени
            logger.info(f"\nПоиск и загрузка snapshot: '{snap_name}'")
            logger.info(f"Максимальный индекс для поиска: {max_index}")
            logger.info("Это может занять некоторое время...\n")
            
            success = client.load_snap(snap_name, max_index)
        
        if success:
            # Проверяем результат
            time.sleep(0.3)
            client.send("/$ctl/lib/$active")
            client.send("/$ctl/lib/$actidx")
            time.sleep(0.2)
            
            new_active = client.state.get("/$ctl/lib/$active")
            new_idx = client.state.get("/$ctl/lib/$actidx")
            
            logger.info("\n" + "=" * 70)
            logger.info("РЕЗУЛЬТАТ")
            logger.info("=" * 70)
            logger.info(f"Новый активный snapshot: [{new_idx}] {new_active}")
            
            if new_active != current_active:
                logger.info("✓ Snapshot изменился - загрузка успешна!")
            else:
                logger.warning("⚠ Snapshot не изменился - возможно, загрузка не сработала")
                logger.warning("Проверьте на пульте вручную")
            
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
