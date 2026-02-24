#!/usr/bin/env python3
"""
Универсальный скрипт для маршрутизации каналов на выходы
Поддерживает любые комбинации OUTPUT GROUP и SOURCE GROUP
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def route_channels(client: WingClient, output_group: str, start_output: int, num_outputs: int,
                   source_group: str, start_source_channel: int):
    """
    Универсальная маршрутизация каналов на выходы
    
    Args:
        client: WingClient instance
        output_group: OUTPUT GROUP (куда посылать) - "MOD", "CRD", "LCL", "AUX", "A", "B", "C", "SC", "USB", "AES", "REC"
        start_output: Начальный номер выхода в OUTPUT GROUP
        num_outputs: Количество выходов для маршрутизации
        source_group: SOURCE GROUP (откуда брать сигнал) - "CRD", "PLAY", "CH", "AUX", "BUS", "MAIN", "MTX", "SEND", "MON", "USR", "OSC"
        start_source_channel: Начальный номер канала источника из SOURCE GROUP
    
    Примеры:
        # Маршрутизация 24 DANTE выходов (MOD 1-24) на WLIVE PLAY каналы (CRD 1-24):
        route_channels(client, "MOD", 1, 24, "CRD", 1)
        
        # Маршрутизация 8 локальных выходов (LCL 1-8) на каналы пульта (CH 1-8):
        route_channels(client, "LCL", 1, 8, "CH", 1)
        
        # Маршрутизация USB выходов (USB 1-4) на BUS шины (BUS 1-4):
        route_channels(client, "USB", 1, 4, "BUS", 1)
    """
    logger.info("=" * 70)
    logger.info("МАРШРУТИЗАЦИЯ КАНАЛОВ")
    logger.info("=" * 70)
    logger.info(f"OUTPUT GROUP: {output_group}")
    logger.info(f"Выходы: {start_output}-{start_output + num_outputs - 1}")
    logger.info(f"SOURCE GROUP: {source_group}")
    logger.info(f"Каналы источника: {start_source_channel}-{start_source_channel + num_outputs - 1}")
    logger.info("=" * 70)
    
    success_count = client.route_multiple_outputs(
        output_group, start_output, num_outputs, source_group, start_source_channel
    )
    
    logger.info(f"\nМаршрутизация завершена: {success_count}/{num_outputs} выходов")
    if success_count == num_outputs:
        logger.info("✓ Все выходы успешно маршрутизированы!")
    else:
        logger.warning(f"⚠ Маршрутизировано только {success_count} из {num_outputs} выходов")
    
    return success_count == num_outputs


def main():
    """Основная функция"""
    if len(sys.argv) < 6:
        print("Использование:")
        print("  python3 route_channels.py <IP> <OUTPUT_GROUP> <START_OUTPUT> <NUM_OUTPUTS> <SOURCE_GROUP> <START_SOURCE_CHANNEL>")
        print("")
        print("Параметры:")
        print("  IP                  - IP адрес пульта (по умолчанию: 192.168.1.102)")
        print("  OUTPUT_GROUP        - OUTPUT GROUP (куда): MOD, CRD, LCL, AUX, A, B, C, SC, USB, AES, REC")
        print("  START_OUTPUT        - Начальный номер выхода")
        print("  NUM_OUTPUTS         - Количество выходов")
        print("  SOURCE_GROUP        - SOURCE GROUP (откуда): CRD, PLAY, CH, AUX, BUS, MAIN, MTX, SEND, MON, USR, OSC")
        print("  START_SOURCE_CHANNEL - Начальный номер канала источника")
        print("")
        print("Примеры:")
        print("  # DANTE выходы (MOD 1-24) <- WLIVE PLAY каналы (CRD 1-24):")
        print("  python3 route_channels.py 192.168.1.102 MOD 1 24 CRD 1")
        print("")
        print("  # Локальные выходы (LCL 1-8) <- каналы пульта (CH 1-8):")
        print("  python3 route_channels.py 192.168.1.102 LCL 1 8 CH 1")
        print("")
        print("  # USB выходы (USB 1-4) <- BUS шины (BUS 1-4):")
        print("  python3 route_channels.py 192.168.1.102 USB 1 4 BUS 1")
        return 1
    
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.102"
    output_group = sys.argv[2].upper()
    start_output = int(sys.argv[3])
    num_outputs = int(sys.argv[4])
    source_group = sys.argv[5].upper()
    start_source_channel = int(sys.argv[6])
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        return 1
    
    try:
        success = route_channels(
            client=client,
            output_group=output_group,
            start_output=start_output,
            num_outputs=num_outputs,
            source_group=source_group,
            start_source_channel=start_source_channel
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
