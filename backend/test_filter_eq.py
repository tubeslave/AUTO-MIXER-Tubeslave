#!/usr/bin/env python3
"""
Тестовый скрипт для проверки hi cut filter и настройки эквалайзера
"""
from wing_client import WingClient
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def on_update(address, *values):
    """Callback для получения обновлений"""
    logger.info(f"📥 Получено обновление: {address} = {values}")


def main():
    logger.info("=== Тест Hi Cut Filter и EQ ===")
    logger.info("Канал 1: Включить hi cut filter, установить EQ на 500 Гц, -3 dB, Q=1.0\n")
    
    # Параметры подключения
    ip = "192.168.1.102"
    port = 2223
    
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    logger.info(f"Подключение к Wing микшеру: {ip}:{port}")
    
    client = WingClient(ip, port)
    client.subscribe("*", on_update)
    
    if client.connect():
        logger.info("✓ Подключено успешно!\n")
        
        channel = 1
        
        # ========== 1. Hi Cut Filter (Low-Pass Filter) ==========
        logger.info("=== 1. Включение Hi Cut Filter ===")
        
        # Пробуем разные возможные адреса для hi cut / low pass filter
        # Сначала проверяем адреса по аналогии с trim (который работает через /in/set/)
        hi_cut_addresses = [
            "/ch/01/in/set/lpf",           # Low-pass filter (по аналогии с trim)
            "/ch/1/in/set/lpf",
            "/ch/01/in/set/hicut",         # Hi cut
            "/ch/1/in/set/hicut",
            "/ch/01/in/set/lcut",          # Low cut
            "/ch/1/in/set/lcut",
            "/ch/01/preamp/lpf",           # Low-pass filter в преампе
            "/ch/1/preamp/lpf",
            "/ch/01/preamp/lcut",          # Low cut
            "/ch/1/preamp/lcut",
            "/ch/01/preamp/hicut",         # Hi cut
            "/ch/1/preamp/hicut",
        ]
        
        logger.info("Запрос текущих значений hi cut filter...")
        for addr in hi_cut_addresses:
            client.send(addr)
            time.sleep(0.1)
        
        time.sleep(1.0)
        
        logger.info("Проверка полученных значений:")
        found_addresses = []
        for addr in hi_cut_addresses:
            value = client.state.get(addr)
            if value is not None:
                logger.info(f"  ✓ {addr} = {value}")
                found_addresses.append(addr)
        
        if not found_addresses:
            logger.warning("  Не найдено активных адресов для hi cut filter")
            logger.info("  Попробуем включить через установку частоты...")
            # Пробуем установить частоту напрямую
            for addr in hi_cut_addresses:
                logger.info(f"  Попытка установки частоты на {addr} = 10000 Hz")
                client.send(addr, 10000.0)  # 10 kHz как пример
                time.sleep(0.1)
        
        # ========== 2. EQ настройка ==========
        logger.info("\n=== 2. Настройка эквалайзера ===")
        logger.info("Включение EQ и установка параметров:")
        logger.info("  Полоса: 1 (первая полоса)")
        logger.info("  Частота: 500 Гц")
        logger.info("  Gain: -3 dB (вырез)")
        logger.info("  Q: 1.0")
        
        # Включаем EQ - пробуем оба формата
        eq_on_addresses = ["/ch/01/eq/on", "/ch/1/eq/on"]
        logger.info(f"\nВключение EQ...")
        for addr in eq_on_addresses:
            logger.info(f"  {addr} = 1")
            client.send(addr, 1)
            time.sleep(0.1)
        time.sleep(0.2)
        
        # Устанавливаем параметры первой полосы EQ
        eq_band = 1
        freq = 500.0  # Гц
        gain = -3.0   # dB (отрицательное значение = вырез)
        q = 1.0
        
        # Пробуем разные форматы адресов (включая /in/set/ по аналогии с trim)
        eq_addresses = {
            "type": [
                f"/ch/01/in/set/eq/{eq_band}/type",  # По аналогии с trim
                f"/ch/1/in/set/eq/{eq_band}/type",
                f"/ch/01/eq/{eq_band}/type",
                f"/ch/1/eq/{eq_band}/type",
            ],
            "freq": [
                f"/ch/01/in/set/eq/{eq_band}/f",
                f"/ch/1/in/set/eq/{eq_band}/f",
                f"/ch/01/eq/{eq_band}/f",
                f"/ch/1/eq/{eq_band}/f",
            ],
            "gain": [
                f"/ch/01/in/set/eq/{eq_band}/g",
                f"/ch/1/in/set/eq/{eq_band}/g",
                f"/ch/01/eq/{eq_band}/g",
                f"/ch/1/eq/{eq_band}/g",
            ],
            "q": [
                f"/ch/01/in/set/eq/{eq_band}/q",
                f"/ch/1/in/set/eq/{eq_band}/q",
                f"/ch/01/eq/{eq_band}/q",
                f"/ch/1/eq/{eq_band}/q",
            ],
        }
        
        # Сначала запрашиваем текущие значения
        logger.info("\nЗапрос текущих значений EQ полосы 1:")
        for param_name, addrs in eq_addresses.items():
            for addr in addrs:
                logger.info(f"  Запрос {param_name}: {addr}")
                client.send(addr)
                time.sleep(0.05)
        
        time.sleep(1.0)
        
        logger.info("\nТекущие значения EQ полосы 1:")
        for param_name, addrs in eq_addresses.items():
            found = False
            for addr in addrs:
                value = client.state.get(addr)
                if value is not None:
                    logger.info(f"  {param_name} ({addr}): {value}")
                    found = True
            if not found:
                logger.warning(f"  {param_name}: значение не получено")
        
        # Устанавливаем параметры
        logger.info("\nУстановка параметров EQ:")
        
        # Устанавливаем частоту, gain и Q (пробуем оба формата адресов)
        for addr in eq_addresses['freq']:
            logger.info(f"  Частота: {addr} = {freq} Hz")
            client.send(addr, freq)
            time.sleep(0.1)
        
        for addr in eq_addresses['gain']:
            logger.info(f"  Gain: {addr} = {gain} dB")
            client.send(addr, gain)
            time.sleep(0.1)
        
        for addr in eq_addresses['q']:
            logger.info(f"  Q: {addr} = {q}")
            client.send(addr, q)
            time.sleep(0.1)
        
        # Проверяем установленные значения
        logger.info("\nПроверка установленных значений EQ:")
        time.sleep(1.5)
        
        for param_name, addrs in eq_addresses.items():
            found = False
            for addr in addrs:
                value = client.state.get(addr)
                if value is not None:
                    logger.info(f"  {param_name} ({addr}): {value}")
                    found = True
            if not found:
                logger.warning(f"  {param_name}: значение не получено")
        
        # Показываем все параметры EQ для первого канала
        logger.info("\n=== Все параметры EQ канала 1 ===")
        eq_params = {k: v for k, v in client.state.items() if '/ch/01/eq/' in k or '/ch/1/eq/' in k}
        for k, v in sorted(eq_params.items()):
            logger.info(f"  {k} = {v}")
        
        logger.info("\n✓ Тест завершен")
        logger.info("Ожидание 2 секунды для наблюдения обновлений...")
        time.sleep(2)
        
        client.disconnect()
        logger.info("✓ Отключено")
    else:
        logger.error("✗ Не удалось подключиться к микшеру!")
        sys.exit(1)


if __name__ == "__main__":
    main()
