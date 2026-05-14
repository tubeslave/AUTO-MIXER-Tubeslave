#!/usr/bin/env python3
"""
Загрузка snapshot/scene по имени
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


def load_snap(client: WingClient, snap_name: str, max_index: int = 200):
    """
    Загрузка snapshot по имени
    
    Процесс:
    1. Сканирует snapshots по индексам для поиска по имени
    2. Находит индекс snapshot
    3. Загружает snapshot по найденному индексу
    """
    logger.info("=" * 70)
    logger.info(f"ЗАГРУЗКА SNAPSHOT: {snap_name}")
    logger.info("=" * 70)
    
    # Используем метод find_snap_by_name из WingClient
    logger.info(f"\nПоиск snapshot по имени (максимум {max_index} индексов)...")
    snap_index = client.find_snap_by_name(snap_name, max_index)
    
    if snap_index is None:
        logger.error("\n" + "=" * 70)
        logger.error("✗ Snapshot не найден")
        logger.error("Проверьте, что snapshot существует и имя указано правильно")
        logger.error("Попробуйте увеличить MAX_INDEX или проверить имя snapshot на пульте")
        logger.error("=" * 70)
        return False
    
    logger.info(f"✓ Найден snapshot с индексом: {snap_index}")
    
    # Загружаем snapshot по индексу
    logger.info(f"\nЗагрузка snapshot по индексу {snap_index}...")
    success = client.load_snap_by_index(snap_index)
    
    if success:
        logger.info("\n" + "=" * 70)
        logger.info("✓ Snapshot загружен")
        logger.info("Проверьте на пульте, применилась ли сцена")
        logger.info("=" * 70)
    else:
        logger.error("\n" + "=" * 70)
        logger.error("✗ Ошибка загрузки snapshot")
        logger.error("=" * 70)
    
    return success


def main():
    """Основная функция"""
    try:
        args, maintenance_ctx = extract_maintenance_cli_args(sys.argv[1:])
    except ValueError as exc:
        print(str(exc))
        print("Использование: python3 load_snap.py <IP> <SNAP_NAME> [MAX_INDEX]")
        print("  --confirm-maintenance")
        print("  --rollback-snapshot PATH")
        print("  --dry-run")
        return 1

    if len(args) < 2:
        print("Использование:")
        print("  python3 load_snap.py <IP> <SNAP_NAME>")
        print("")
        print("Параметры:")
        print("  IP        - IP адрес пульта (по умолчанию: 192.168.1.102)")
        print("  SNAP_NAME - Имя snapshot для загрузки")
        print("")
        print("Пример:")
        print("  python3 load_snap.py 192.168.1.102 'HULI REPA AC'")
        return 1

    ip = args[0] if len(args) > 0 else "192.168.1.102"
    snap_name = args[1] if len(args) > 1 else None
    max_index = int(args[2]) if len(args) > 2 else 200
    
    if not snap_name:
        print("Ошибка: необходимо указать имя snapshot")
        print("Использование: python3 load_snap.py <IP> <SNAP_NAME> [MAX_INDEX]")
        return 1

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
        logger.info(
            "Dry run: would load snapshot '%s' on %s (max_index=%s)",
            snap_name,
            ip,
            max_index,
        )
        return 0
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        return 1
    
    try:
        success = load_snap(client, snap_name, max_index)
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
