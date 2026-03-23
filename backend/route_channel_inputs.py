#!/usr/bin/env python3
"""
Скрипт для маршрутизации входов каналов пульта
Назначает Channel Main и Channel ALT входы для каналов 1-40
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def route_channel_inputs(client: WingClient, start_channel: int = 1, num_channels: int = 40,
                          main_source_group: str = "MOD", alt_source_group: str = "CRD"):
    """
    Маршрутизация входов каналов пульта
    
    Args:
        client: WingClient instance
        start_channel: Начальный канал (по умолчанию 1)
        num_channels: Количество каналов (по умолчанию 40)
        main_source_group: SOURCE GROUP для Channel Main (по умолчанию "MOD")
        alt_source_group: SOURCE GROUP для Channel ALT (по умолчанию "CRD")
    
    Для каждого канала:
        - Channel Main получает сигнал из main_source_group с соответствующим номером канала
        - Channel ALT получает сигнал из alt_source_group с соответствующим номером канала
    
    Пример:
        # Каналы 1-40:
        #   Channel Main <- MOD 1-40
        #   Channel ALT <- CRD 1-40
        route_channel_inputs(client, 1, 40, "MOD", "CRD")
    """
    logger.info("=" * 70)
    logger.info("МАРШРУТИЗАЦИЯ ВХОДОВ КАНАЛОВ ПУЛЬТА")
    logger.info("=" * 70)
    logger.info(f"Каналы: {start_channel}-{start_channel + num_channels - 1}")
    logger.info(f"Channel Main: {main_source_group} каналы {start_channel}-{start_channel + num_channels - 1}")
    logger.info(f"Channel ALT: {alt_source_group} каналы {start_channel}-{start_channel + num_channels - 1}")
    logger.info("=" * 70)
    
    main_success = 0
    alt_success = 0
    
    for i in range(num_channels):
        channel = start_channel + i
        source_channel = start_channel + i
        
        # Маршрутизация Channel Main
        if client.set_channel_input(channel, main_source_group, source_channel):
            logger.info(f"  ✓ Канал {channel:2d} Main <- {main_source_group} канал {source_channel:2d}")
            main_success += 1
        else:
            logger.warning(f"  ✗ Ошибка маршрутизации канала {channel} Main")
        
        time.sleep(0.01)
        
        # Маршрутизация Channel ALT
        if client.set_channel_alt_input(channel, alt_source_group, source_channel):
            logger.info(f"  ✓ Канал {channel:2d} ALT  <- {alt_source_group} канал {source_channel:2d}")
            alt_success += 1
        else:
            logger.warning(f"  ✗ Ошибка маршрутизации канала {channel} ALT")
        
        time.sleep(0.01)
    
    logger.info("=" * 70)
    logger.info(f"Маршрутизация завершена:")
    logger.info(f"  Channel Main: {main_success}/{num_channels} каналов")
    logger.info(f"  Channel ALT:  {alt_success}/{num_channels} каналов")
    
    if main_success == num_channels and alt_success == num_channels:
        logger.info("✓ Все каналы успешно маршрутизированы!")
        return True
    else:
        logger.warning(f"⚠ Маршрутизировано не все каналы")
        return False


def main():
    """Основная функция"""
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python3 route_channel_inputs.py <IP> [START_CHANNEL] [NUM_CHANNELS] [MAIN_SOURCE] [ALT_SOURCE]")
        print("")
        print("Параметры:")
        print("  IP              - IP адрес пульта (по умолчанию: 192.168.1.102)")
        print("  START_CHANNEL   - Начальный канал (по умолчанию: 1)")
        print("  NUM_CHANNELS    - Количество каналов (по умолчанию: 40)")
        print("  MAIN_SOURCE     - SOURCE GROUP для Channel Main (по умолчанию: MOD)")
        print("  ALT_SOURCE      - SOURCE GROUP для Channel ALT (по умолчанию: CRD)")
        print("")
        print("Примеры:")
        print("  # Каналы 1-40: Main <- MOD 1-40, ALT <- CRD 1-40")
        print("  python3 route_channel_inputs.py 192.168.1.102")
        print("")
        print("  # Каналы 1-20: Main <- MOD 1-20, ALT <- CRD 1-20")
        print("  python3 route_channel_inputs.py 192.168.1.102 1 20")
        print("")
        print("  # Каналы 1-40: Main <- CRD 1-40, ALT <- MOD 1-40")
        print("  python3 route_channel_inputs.py 192.168.1.102 1 40 CRD MOD")
        return 1
    
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.102"
    start_channel = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    num_channels = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    main_source = sys.argv[4].upper() if len(sys.argv) > 4 else "MOD"
    alt_source = sys.argv[5].upper() if len(sys.argv) > 5 else "CRD"
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        return 1
    
    try:
        success = route_channel_inputs(
            client=client,
            start_channel=start_channel,
            num_channels=num_channels,
            main_source_group=main_source,
            alt_source_group=alt_source
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
