#!/usr/bin/env python3
"""
Тест с правильными адресами из документации WING Remote Protocols v3.0.5
"""
from wing_client import WingClient
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Включаем debug для отслеживания всех OSC сообщений
logging.getLogger('wing_client').setLevel(logging.DEBUG)

received_addresses = []

def on_update(address, *values):
    """Callback для получения обновлений"""
    global received_addresses
    received_addresses.append((address, values))
    logger.info(f"📥 Получено: {address} = {values}")


def main():
    logger.info("=== Тест Hi Cut Filter и EQ (правильные адреса) ===")
    logger.info("Канал 1:")
    logger.info("  1. Включить hi cut filter")
    logger.info("  2. Настроить EQ: 500 Гц, -3 dB, Q=1.0\n")
    
    ip = "192.168.1.102"
    port = 2223
    
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    
    logger.info(f"Подключение к {ip}:{port}")
    client = WingClient(ip, port)
    client.subscribe("*", on_update)
    
    if client.connect():
        logger.info("✓ Подключено!\n")
        
        channel = 1
        
        # ========== 1. Hi Cut Filter ==========
        logger.info("=== 1. Hi Cut Filter ===")
        logger.info("Согласно документации:")
        logger.info("  /ch/1/flt/hc - включение hi cut (0/1)")
        logger.info("  /ch/1/flt/hcf - частота hi cut (50..20000 Hz)")
        logger.info("  /ch/1/flt/hcs - крутизна среза (6, 12)")
        
        # Включаем hi cut filter
        logger.info("\nВключение hi cut filter:")
        logger.info("  /ch/1/flt/hc = 1")
        client.send("/ch/1/flt/hc", 1)
        time.sleep(0.2)
        
        # Устанавливаем частоту среза (например, 10 kHz)
        hi_cut_freq = 10000.0
        logger.info(f"  /ch/1/flt/hcf = {hi_cut_freq} Hz")
        client.send("/ch/1/flt/hcf", hi_cut_freq)
        time.sleep(0.2)
        
        # Устанавливаем крутизну среза (12 dB/octave)
        logger.info("  /ch/1/flt/hcs = 12")
        client.send("/ch/1/flt/hcs", "12")
        time.sleep(0.2)
        
        # ========== 2. EQ ==========
        logger.info("\n=== 2. Эквалайзер ===")
        logger.info("Согласно документации:")
        logger.info("  /ch/1/eq/on - включение EQ (0/1)")
        logger.info("  /ch/1/eq/1f - частота полосы 1 (20..20000 Hz)")
        logger.info("  /ch/1/eq/1g - gain полосы 1 (-15..15 dB)")
        logger.info("  /ch/1/eq/1q - Q полосы 1 (0.44..10)")
        
        # Включаем EQ
        logger.info("\nВключение EQ:")
        logger.info("  /ch/1/eq/on = 1")
        client.send("/ch/1/eq/on", 1)
        time.sleep(0.2)
        
        # Устанавливаем параметры EQ полосы 1
        freq = 500.0   # Гц
        gain = -3.0    # dB (отрицательное = вырез)
        q = 1.0
        
        logger.info(f"\nУстановка параметров EQ полосы 1:")
        logger.info(f"  /ch/1/eq/1f = {freq} Hz")
        client.send("/ch/1/eq/1f", freq)
        time.sleep(0.2)
        
        logger.info(f"  /ch/1/eq/1g = {gain} dB")
        client.send("/ch/1/eq/1g", gain)
        time.sleep(0.2)
        
        logger.info(f"  /ch/1/eq/1q = {q}")
        client.send("/ch/1/eq/1q", q)
        time.sleep(0.2)
        
        # Запрашиваем установленные значения
        logger.info("\nЗапрос установленных значений...")
        client.send("/ch/1/flt/hc")
        client.send("/ch/1/flt/hcf")
        client.send("/ch/1/eq/on")
        client.send("/ch/1/eq/1f")
        client.send("/ch/1/eq/1g")
        client.send("/ch/1/eq/1q")
        
        # Ждем ответы
        logger.info("Ожидание ответов от микшера (3 секунды)...")
        time.sleep(3.0)
        
        # Показываем полученные значения
        logger.info("\n=== Проверка установленных значений ===")
        
        hi_cut_on = client.state.get("/ch/1/flt/hc")
        hi_cut_freq = client.state.get("/ch/1/flt/hcf")
        eq_on = client.state.get("/ch/1/eq/on")
        eq_freq = client.state.get("/ch/1/eq/1f")
        eq_gain = client.state.get("/ch/1/eq/1g")
        eq_q = client.state.get("/ch/1/eq/1q")
        
        logger.info("Hi Cut Filter:")
        if hi_cut_on is not None:
            logger.info(f"  Включен: {hi_cut_on}")
        else:
            logger.warning("  Включен: значение не получено")
        
        if hi_cut_freq is not None:
            logger.info(f"  Частота: {hi_cut_freq} Hz")
        else:
            logger.warning("  Частота: значение не получено")
        
        logger.info("EQ:")
        if eq_on is not None:
            logger.info(f"  Включен: {eq_on}")
        else:
            logger.warning("  Включен: значение не получено")
        
        if eq_freq is not None:
            logger.info(f"  Частота полосы 1: {eq_freq} Hz")
        else:
            logger.warning("  Частота: значение не получено")
        
        if eq_gain is not None:
            logger.info(f"  Gain полосы 1: {eq_gain} dB")
        else:
            logger.warning("  Gain: значение не получено")
        
        if eq_q is not None:
            logger.info(f"  Q полосы 1: {eq_q}")
        else:
            logger.warning("  Q: значение не получено")
        
        logger.info("\n✓ Все команды отправлены с правильными адресами из документации!")
        logger.info("Проверьте физический микшер:")
        logger.info("  - Hi cut filter должен быть включен")
        logger.info("  - EQ полоса 1: 500 Гц, -3 dB, Q=1.0")
        
        time.sleep(1)
        client.disconnect()
        logger.info("\n✓ Отключено")
    else:
        logger.error("✗ Не удалось подключиться!")
        sys.exit(1)


if __name__ == "__main__":
    main()
