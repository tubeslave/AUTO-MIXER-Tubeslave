#!/usr/bin/env python3
"""
Финальный тест для hi cut filter и EQ на канале 1
"""
from wing_client import WingClient
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Тест Hi Cut Filter и EQ на канале 1 ===")
    logger.info("Задачи:")
    logger.info("  1. Включить hi cut filter")
    logger.info("  2. Настроить EQ: 500 Гц, -3 dB, Q=1.0\n")
    
    ip = "192.168.1.102"
    port = 2223
    
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    
    logger.info(f"Подключение к {ip}:{port}")
    client = WingClient(ip, port)
    
    if client.connect():
        logger.info("✓ Подключено!\n")
        
        channel = 1
        
        # ========== 1. Hi Cut Filter (Low-Pass Filter) ==========
        logger.info("=== 1. Hi Cut Filter ===")
        logger.info("Попытка включить hi cut filter через установку частоты...")
        
        # Пробуем разные адреса для hi cut / low pass filter
        # Устанавливаем частоту среза (например, 10 kHz)
        hi_cut_freq = 10000.0
        
        hi_cut_addresses = [
            "/ch/1/in/set/lpf",      # Low-pass filter (по аналогии с trim)
            "/ch/01/in/set/lpf",
            "/ch/1/preamp/lpf",      # В преампе
            "/ch/01/preamp/lpf",
        ]
        
        for addr in hi_cut_addresses:
            logger.info(f"  {addr} = {hi_cut_freq} Hz")
            client.send(addr, hi_cut_freq)
            time.sleep(0.2)
        
        logger.info("  Команды отправлены. Проверьте физический микшер.\n")
        
        # ========== 2. EQ настройка ==========
        logger.info("=== 2. Настройка эквалайзера ===")
        
        # Включаем EQ
        logger.info("Включение EQ...")
        eq_on_addresses = ["/ch/1/eq/on", "/ch/01/eq/on"]
        for addr in eq_on_addresses:
            logger.info(f"  {addr} = 1")
            client.send(addr, 1)
            time.sleep(0.2)
        
        # Параметры EQ полосы 1
        freq = 500.0   # Гц
        gain = -3.0    # dB (отрицательное = вырез)
        q = 1.0
        
        logger.info(f"\nУстановка параметров EQ полосы 1:")
        logger.info(f"  Частота: {freq} Hz")
        logger.info(f"  Gain: {gain} dB")
        logger.info(f"  Q: {q}")
        
        # Адреса EQ (пробуем формат без нуля, как работает trim)
        eq_params = {
            'freq': ["/ch/1/eq/1/f", "/ch/01/eq/1/f"],
            'gain': ["/ch/1/eq/1/g", "/ch/01/eq/1/g"],
            'q': ["/ch/1/eq/1/q", "/ch/01/eq/1/q"],
        }
        
        # Устанавливаем частоту
        for addr in eq_params['freq']:
            logger.info(f"  {addr} = {freq}")
            client.send(addr, freq)
            time.sleep(0.2)
        
        # Устанавливаем gain
        for addr in eq_params['gain']:
            logger.info(f"  {addr} = {gain}")
            client.send(addr, gain)
            time.sleep(0.2)
        
        # Устанавливаем Q
        for addr in eq_params['q']:
            logger.info(f"  {addr} = {q}")
            client.send(addr, q)
            time.sleep(0.2)
        
        logger.info("\n✓ Все команды отправлены!")
        logger.info("\nПРИМЕЧАНИЕ: Микшер может не отвечать на запросы этих параметров,")
        logger.info("но команды могут работать. Пожалуйста, проверьте физический микшер:")
        logger.info("  - Hi cut filter должен быть включен")
        logger.info("  - EQ полоса 1: 500 Гц, -3 dB, Q=1.0")
        
        time.sleep(2)
        client.disconnect()
        logger.info("\n✓ Отключено")
    else:
        logger.error("✗ Не удалось подключиться!")
        sys.exit(1)


if __name__ == "__main__":
    main()
