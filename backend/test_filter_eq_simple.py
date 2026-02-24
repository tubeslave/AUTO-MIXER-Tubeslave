#!/usr/bin/env python3
"""
Упрощенный тест для hi cut filter и EQ
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
    logger.info("=== Тест Hi Cut Filter и EQ (упрощенный) ===")
    logger.info("Канал 1: Hi cut filter + EQ 500 Гц, -3 dB, Q=1.0\n")
    
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
        
        # ========== Hi Cut Filter ==========
        logger.info("=== 1. Hi Cut Filter ===")
        # Пробуем установить через разные адреса
        hi_cut_addrs = [
            "/ch/01/in/set/lpf",
            "/ch/1/in/set/lpf",
        ]
        
        # Устанавливаем частоту hi cut (например, 10 kHz)
        hi_cut_freq = 10000.0
        for addr in hi_cut_addrs:
            logger.info(f"Установка {addr} = {hi_cut_freq} Hz")
            client.send(addr, hi_cut_freq)
            time.sleep(0.2)
        
        # ========== EQ ==========
        logger.info("\n=== 2. Эквалайзер ===")
        
        # Включаем EQ
        logger.info("Включение EQ...")
        client.send("/ch/01/eq/on", 1)
        client.send("/ch/1/eq/on", 1)
        time.sleep(0.3)
        
        # Устанавливаем параметры EQ полосы 1
        freq = 500.0
        gain = -3.0
        q = 1.0
        
        logger.info(f"Установка EQ полосы 1:")
        logger.info(f"  Частота: {freq} Hz")
        logger.info(f"  Gain: {gain} dB")
        logger.info(f"  Q: {q}")
        
        # Пробуем адреса без нуля (как работает trim)
        eq_addrs = {
            'f': ["/ch/1/eq/1/f", "/ch/01/eq/1/f"],
            'g': ["/ch/1/eq/1/g", "/ch/01/eq/1/g"],
            'q': ["/ch/1/eq/1/q", "/ch/01/eq/1/q"],
        }
        
        for param, addrs in eq_addrs.items():
            for addr in addrs:
                value = freq if param == 'f' else (gain if param == 'g' else q)
                logger.info(f"  {addr} = {value}")
                client.send(addr, value)
                time.sleep(0.2)
        
        # Ждем ответы
        logger.info("\nОжидание ответов от микшера (5 секунд)...")
        time.sleep(5.0)
        
        # Показываем все полученные адреса, связанные с EQ и фильтрами
        logger.info("\n=== Полученные адреса ===")
        filter_eq_addresses = [
            addr for addr, vals in received_addresses
            if 'eq' in addr.lower() or 'filter' in addr.lower() or 'lpf' in addr.lower() or 'hpf' in addr.lower() or 'cut' in addr.lower()
        ]
        
        if filter_eq_addresses:
            logger.info("Найдены адреса, связанные с EQ и фильтрами:")
            for addr in set(filter_eq_addresses):
                vals = next((v for a, v in received_addresses if a == addr), None)
                logger.info(f"  {addr} = {vals}")
        else:
            logger.info("Адреса EQ/фильтров не получены в ответах")
        
        # Показываем все адреса канала 1
        logger.info("\n=== Все адреса канала 1 ===")
        ch1_addresses = {addr: vals for addr, vals in received_addresses if '/ch/1' in addr or '/ch/01' in addr}
        for addr, vals in sorted(ch1_addresses.items()):
            logger.info(f"  {addr} = {vals}")
        
        logger.info("\n✓ Команды отправлены. Проверьте физический микшер.")
        logger.info("Если значения не изменились, возможно адреса другие.")
        
        time.sleep(1)
        client.disconnect()
        logger.info("✓ Отключено")
    else:
        logger.error("✗ Не удалось подключиться!")
        sys.exit(1)


if __name__ == "__main__":
    main()
