#!/usr/bin/env python3
"""
Тестирование различных адресов для загрузки snapshots/scenes
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_snap_addresses(client: WingClient, snap_name: str):
    """
    Тестирование различных адресов для загрузки snapshot
    """
    logger.info(f"Тестирование загрузки snapshot: {snap_name}")
    logger.info("=" * 70)
    
    # Возможные адреса для snapshots
    test_addresses = [
        # Варианты с /snap
        (f"/snap/{snap_name}/load", None),
        (f"/snap/{snap_name}/recall", None),
        (f"/snap/{snap_name}/on", 1),
        (f"/snap/load", snap_name),
        (f"/snap/recall", snap_name),
        (f"/snap/name", snap_name),
        
        # Варианты с /scene
        (f"/scene/{snap_name}/load", None),
        (f"/scene/{snap_name}/recall", None),
        (f"/scene/load", snap_name),
        (f"/scene/recall", snap_name),
        
        # Варианты с /snapshot
        (f"/snapshot/{snap_name}/load", None),
        (f"/snapshot/{snap_name}/recall", None),
        (f"/snapshot/load", snap_name),
        (f"/snapshot/recall", snap_name),
        
        # Варианты с /show
        (f"/show/{snap_name}/load", None),
        (f"/show/{snap_name}/recall", None),
        (f"/show/load", snap_name),
        (f"/show/recall", snap_name),
        
        # Варианты с /preset
        (f"/preset/{snap_name}/load", None),
        (f"/preset/{snap_name}/recall", None),
        (f"/preset/load", snap_name),
        (f"/preset/recall", snap_name),
        
        # Варианты с /store
        (f"/store/{snap_name}/load", None),
        (f"/store/{snap_name}/recall", None),
        (f"/store/load", snap_name),
        (f"/store/recall", snap_name),
        
        # Варианты с /snap/recall
        (f"/snap/recall", snap_name),
        (f"/snap/recall/name", snap_name),
        
        # Варианты с индексом (если snap_name это число)
        (f"/snap/recall", 1),  # Если это индекс
    ]
    
    # Также пробуем с URL-encoded именем
    snap_name_encoded = snap_name.replace(" ", "%20")
    test_addresses.extend([
        (f"/snap/{snap_name_encoded}/load", None),
        (f"/snap/{snap_name_encoded}/recall", None),
    ])
    
    logger.info(f"Будет протестировано {len(test_addresses)} адресов")
    logger.info("Проверьте на пульте, изменилась ли сцена после каждой команды")
    logger.info("=" * 70)
    
    for i, (address, value) in enumerate(test_addresses, 1):
        logger.info(f"\n[{i}/{len(test_addresses)}] Тестирую: {address}")
        if value is not None:
            logger.info(f"  Значение: {value}")
            result = client.send(address, value)
        else:
            logger.info(f"  Без значения")
            result = client.send(address)
        
        if result:
            logger.info(f"  ✓ Команда отправлена успешно")
        else:
            logger.warning(f"  ✗ Ошибка отправки команды")
        
        time.sleep(2)  # Даем время на применение
        logger.info(f"  → Проверьте на пульте, применилась ли сцена '{snap_name}'")
        
        if i < len(test_addresses):
            input("  Нажмите Enter для продолжения или Ctrl+C для остановки...")


def main():
    """Основная функция"""
    if len(sys.argv) < 3:
        print("Использование:")
        print("  python3 test_snap_recall.py <IP> <SNAP_NAME>")
        print("")
        print("Пример:")
        print("  python3 test_snap_recall.py 192.168.1.102 'HULI REPA AC'")
        return 1
    
    ip = sys.argv[1]
    snap_name = sys.argv[2]
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        return 1
    
    try:
        test_snap_addresses(client, snap_name)
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
