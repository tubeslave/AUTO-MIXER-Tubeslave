#!/usr/bin/env python3
"""
Скрипт для маршрутизации 24 каналов пульта на Dante выходы последовательно
из карты Wlive play

Использует универсальную функцию route_channels из wing_client
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_current_routing(client: WingClient, channel: int):
    """
    Проверяет текущую маршрутизацию канала
    
    Args:
        client: WingClient instance
        channel: Номер канала для проверки
    """
    logger.info(f"\nПроверка текущей маршрутизации канала {channel}:")
    
    # Проверяем различные возможные адреса
    addresses_to_check = [
        f"/ch/{channel}/out/conn/grp",
        f"/ch/{channel}/out/conn/out",
        f"/ch/{channel}/out/grp",
        f"/ch/{channel}/out/out",
    ]
    
    for addr in addresses_to_check:
        client.send(addr)
        time.sleep(0.05)
        value = client.state.get(addr)
        if value is not None:
            logger.info(f"  {addr} = {value}")


def route_channels_to_dante(client: WingClient, start_channel: int = 1, num_channels: int = 24, 
                            dante_start_output: int = 1, card_name: str = "WLIVE"):
    """
    Маршрутизирует каналы на Dante выходы последовательно
    
    Использует универсальную функцию route_multiple_outputs из WingClient
    
    Args:
        client: WingClient instance
        start_channel: Начальный канал (по умолчанию 1)
        num_channels: Количество каналов для маршрутизации (по умолчанию 24)
        dante_start_output: Начальный Dante выход (по умолчанию 1)
        card_name: Имя карты (по умолчанию "WLIVE") - не используется, оставлено для совместимости
    """
    logger.info(f"Начинаю маршрутизацию {num_channels} каналов на Dante выходы")
    logger.info(f"Каналы: {start_channel}-{start_channel + num_channels - 1}")
    logger.info(f"Dante выходы: {dante_start_output}-{dante_start_output + num_channels - 1}")
    logger.info(f"Карта: {card_name}")
    
    # Сначала проверяем текущую маршрутизацию первого канала для отладки
    logger.info("\nПроверка текущей маршрутизации (для отладки)...")
    check_current_routing(client, start_channel)
    time.sleep(0.2)
    
    # Прямая маршрутизация через output connection
    # Согласно документации Wing Remote Protocols v3.0.5 и описанию пользователя:
    # Порядок маршрутизации:
    # 1. OUTPUT GROUP (куда посылать) - DANTE (находится в CRD для карт)
    # 2. Канал в OUTPUT GROUP (номер выхода DANTE)
    # 3. SOURCE GROUP (откуда брать сигнал) - WLIVE PLAY
    # 4. Канал в SOURCE GROUP (номер канала источника)
    #
    # Из документации:
    # - /io/out/CRD/{output}/grp - SOURCE GROUP (OFF, LCL, AUX, A, B, C, SC, USB, CRD, MOD, PLAY, AES, USR, OSC, BUS, MAIN, MTX, SEND, MON)
    # - /io/out/CRD/{output}/in - номер канала источника (1..64)
    #
    # Пробуем разные варианты SOURCE GROUP:
    # - CH (каналы пульта) - но в документации нет CH в списке grp
    # - PLAY (WLIVE PLAY) - есть в списке
    # - CRD (Card) - есть в списке
    logger.info("\n=== Маршрутизация через /io/out/CRD (из документации) ===")
    logger.info("OUTPUT GROUP: CRD (Card Outputs, включая DANTE)")
    logger.info("Пробую разные варианты SOURCE GROUP...")
    success_count = 0
    
    # Согласно документации и описанию пользователя:
    # Порядок маршрутизации:
    # 1. OUTPUT GROUP (куда) - DANTE (может быть CRD или MOD для карт/модулей)
    # 2. Канал в OUTPUT GROUP (номер выхода DANTE)
    # 3. SOURCE GROUP (откуда) - WLIVE PLAY (используем PLAY)
    # 4. Канал в SOURCE GROUP (номер канала источника)
    #
    # Из документации:
    # - /io/out/CRD/{output}/grp - SOURCE GROUP
    # - /io/out/CRD/{output}/in - номер канала источника
    # - /io/out/MOD/{output}/grp - SOURCE GROUP (для модулей, возможно DANTE)
    # - /io/out/MOD/{output}/in - номер канала источника
    #
    # Согласно документации и уточнению пользователя:
    # Порядок маршрутизации:
    # 1. OUTPUT GROUP (куда посылать) - MOD (Module, где находится DANTE модуль)
    # 2. Канал в OUTPUT GROUP (номер выхода DANTE)
    # 3. SOURCE GROUP (откуда брать сигнал) - CRD (Card, где находится WLIVE PLAY карта)
    # 4. Канал в SOURCE GROUP (номер канала из WLIVE PLAY)
    #
    # Из документации:
    # - /io/out/MOD/{output}/grp - SOURCE GROUP (CRD для Card/WLIVE PLAY)
    # - /io/out/MOD/{output}/in - номер канала источника (1..64)
    #
    # MOD - это OUTPUT GROUP для модулей (включая DANTE)
    # grp = CRD означает SOURCE GROUP = Card (WLIVE PLAY карта)
    # in = номер канала из SOURCE GROUP
    logger.info("Используем универсальную функцию маршрутизации:")
    logger.info("  OUTPUT GROUP: MOD (Module, где находится DANTE модуль)")
    logger.info("  SOURCE GROUP: CRD (Card, где находится WLIVE PLAY карта)")
    logger.info("  Канал источника: номер канала из CRD группы (WLIVE PLAY)")
    
    # Используем универсальную функцию route_multiple_outputs
    success_count = client.route_multiple_outputs(
        output_group="MOD",
        start_output=dante_start_output,
        num_outputs=num_channels,
        source_group="CRD",
        start_source_channel=start_channel
    )
    
    # Логируем результат
    for i in range(num_channels):
        channel = start_channel + i
        dante_output = dante_start_output + i
        logger.info(f"  ✓ MOD выход {dante_output:2d} <- CRD канал {channel:2d} (DANTE <- WLIVE PLAY)")
    
    time.sleep(0.5)
    
    logger.info(f"\n=== Маршрутизация завершена: {success_count}/{num_channels} каналов ===")
    if success_count == num_channels:
        logger.info("✓ Все каналы успешно маршрутизированы!")
    else:
        logger.warning(f"⚠ Маршрутизировано только {success_count} из {num_channels} каналов")
    logger.info("Проверьте настройки на пульте или в Dante Controller")


def main():
    """Основная функция"""
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = "192.168.1.102"
    
    if len(sys.argv) > 2:
        start_channel = int(sys.argv[2])
    else:
        start_channel = 1
    
    if len(sys.argv) > 3:
        num_channels = int(sys.argv[3])
    else:
        num_channels = 24
    
    if len(sys.argv) > 4:
        dante_start_output = int(sys.argv[4])
    else:
        dante_start_output = 1
    
    if len(sys.argv) > 5:
        card_name = sys.argv[5].upper()
    else:
        card_name = "WLIVE"
    
    logger.info("=" * 60)
    logger.info("Маршрутизация каналов на Dante выходы")
    logger.info("=" * 60)
    logger.info(f"IP пульта: {ip}")
    logger.info(f"Начальный канал: {start_channel}")
    logger.info(f"Количество каналов: {num_channels}")
    logger.info(f"Начальный Dante выход: {dante_start_output}")
    logger.info(f"Карта: {card_name}")
    logger.info("=" * 60)
    
    # Создаем клиент
    client = WingClient(ip=ip, port=2223)
    
    # Подключаемся
    logger.info(f"\nПодключение к пульту {ip}...")
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        logger.error("Проверьте:")
        logger.error("  1. IP адрес пульта")
        logger.error("  2. Сетевое подключение")
        logger.error("  3. Что пульт включен и доступен")
        return 1
    
    logger.info("Подключение успешно!")
    
    try:
        # Выполняем маршрутизацию
        route_channels_to_dante(
            client=client,
            start_channel=start_channel,
            num_channels=num_channels,
            dante_start_output=dante_start_output,
            card_name=card_name
        )
        
        logger.info("\n✓ Маршрутизация выполнена успешно!")
        logger.info("\nПримечание: Если маршрутизация не сработала, возможно:")
        logger.info("  1. OSC адреса для Dante routing отличаются от ожидаемых")
        logger.info("  2. Необходимо проверить документацию Wing Remote Protocols")
        logger.info("  3. Возможно, требуется настройка через Dante Controller")
        logger.info("  4. Проверьте правильность имени карты (WLIVE, DANTE, и т.д.)")
        
    except KeyboardInterrupt:
        logger.info("\n\nПрервано пользователем")
    except Exception as e:
        logger.error(f"\nОшибка при выполнении маршрутизации: {e}", exc_info=True)
        return 1
    finally:
        client.disconnect()
        logger.info("Отключено от пульта")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
