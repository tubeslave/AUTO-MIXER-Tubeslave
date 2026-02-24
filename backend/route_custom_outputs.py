#!/usr/bin/env python3
"""
Скрипт для маршрутизации на произвольные выходы с произвольными источниками
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def route_custom_outputs(client: WingClient, output_group: str, output_source_pairs: list):
    """
    Маршрутизация на произвольные выходы с произвольными источниками
    
    Args:
        client: WingClient instance
        output_group: OUTPUT GROUP (куда посылать) - "A", "B", "C", "MOD", "CRD", и т.д.
        output_source_pairs: Список кортежей [(output_number, source_group, source_channel), ...]
    
    Пример:
        # Маршрутизация AES50 выходов 1,3,5,10-20 на MOD каналы 1-14:
        route_custom_outputs(client, "A", [
            (1, "MOD", 1),
            (3, "MOD", 2),
            (5, "MOD", 3),
            (10, "MOD", 4),
            (11, "MOD", 5),
            (12, "MOD", 6),
            (13, "MOD", 7),
            (14, "MOD", 8),
            (15, "MOD", 9),
            (16, "MOD", 10),
            (17, "MOD", 11),
            (18, "MOD", 12),
            (19, "MOD", 13),
            (20, "MOD", 14),
        ])
    """
    logger.info("=" * 70)
    logger.info("МАРШРУТИЗАЦИЯ НА ПРОИЗВОЛЬНЫЕ ВЫХОДЫ")
    logger.info("=" * 70)
    logger.info(f"OUTPUT GROUP: {output_group}")
    logger.info(f"Количество выходов: {len(output_source_pairs)}")
    logger.info("=" * 70)
    
    success_count = 0
    for output_num, source_group, source_channel in output_source_pairs:
        if client.route_output(output_group, output_num, source_group, source_channel):
            logger.info(f"  ✓ {output_group} выход {output_num:2d} <- {source_group} канал {source_channel:2d}")
            success_count += 1
        else:
            logger.warning(f"  ✗ Ошибка маршрутизации {output_group} выход {output_num}")
        time.sleep(0.01)
    
    logger.info("=" * 70)
    logger.info(f"Маршрутизация завершена: {success_count}/{len(output_source_pairs)} выходов")
    if success_count == len(output_source_pairs):
        logger.info("✓ Все выходы успешно маршрутизированы!")
    else:
        logger.warning(f"⚠ Маршрутизировано только {success_count} из {len(output_source_pairs)} выходов")
    
    return success_count == len(output_source_pairs)


def main():
    """Основная функция"""
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python3 route_custom_outputs.py <IP>")
        print("")
        print("Скрипт маршрутизирует:")
        print("  AES50 выходы: 1, 3, 5, 10-20")
        print("  На MOD каналы: 1-14")
        print("")
        print("Пример:")
        print("  python3 route_custom_outputs.py 192.168.1.102")
        return 1
    
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.102"
    
    # Определяем пары: (AES50 выход, SOURCE GROUP, MOD канал)
    # Выходы: 1, 3, 5, 10-20 (14 выходов)
    # MOD каналы: 1-14
    output_source_pairs = [
        (1, "MOD", 1),
        (3, "MOD", 2),
        (5, "MOD", 3),
        (10, "MOD", 4),
        (11, "MOD", 5),
        (12, "MOD", 6),
        (13, "MOD", 7),
        (14, "MOD", 8),
        (15, "MOD", 9),
        (16, "MOD", 10),
        (17, "MOD", 11),
        (18, "MOD", 12),
        (19, "MOD", 13),
        (20, "MOD", 14),
    ]
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        return 1
    
    try:
        success = route_custom_outputs(
            client=client,
            output_group="A",  # AES50 A
            output_source_pairs=output_source_pairs
        )
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
