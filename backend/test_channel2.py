#!/usr/bin/env python3
"""
Тест установки параметров второго канала
"""
from wing_client import WingClient
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Включаем debug для отслеживания OSC сообщений
logging.getLogger('wing_client').setLevel(logging.DEBUG)

received_updates = []

def on_update(address, *values):
    """Callback для получения обновлений"""
    received_updates.append((address, values))
    logger.info(f"📥 Получено: {address} = {values}")


def main():
    logger.info("=== Тест параметров канала 2 ===")
    logger.info("Установка:")
    logger.info("  Trim: -3 dB")
    logger.info("  Панорама: -33")
    logger.info("  Фейдер: -24 dB\n")
    
    ip = "192.168.1.102"
    port = 2223
    
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    
    logger.info(f"Подключение к {ip}:{port}")
    client = WingClient(ip, port)
    client.subscribe("*", on_update)
    
    if client.connect():
        logger.info("✓ Подключено!\n")
        
        channel = 2
        
        # Запрашиваем текущие значения
        logger.info("=== Запрос текущих значений ===")
        client.send(f"/ch/{channel}/in/set/trim")
        client.send(f"/ch/{channel}/pan")
        client.send(f"/ch/{channel}/fdr")
        time.sleep(1.0)
        
        logger.info("Текущие значения:")
        trim_current = client.state.get(f"/ch/{channel}/in/set/trim")
        pan_current = client.state.get(f"/ch/{channel}/pan")
        fdr_current = client.state.get(f"/ch/{channel}/fdr")
        
        logger.info(f"  Trim: {trim_current} dB" if trim_current is not None else "  Trim: не получено")
        logger.info(f"  Панорама: {pan_current}" if pan_current is not None else "  Панорама: не получено")
        logger.info(f"  Фейдер: {fdr_current} dB" if fdr_current is not None else "  Фейдер: не получено")
        
        # Устанавливаем новые значения
        logger.info("\n=== Установка новых значений ===")
        
        trim_value = -3.0  # dB
        pan_value = -33.0  # -100..100
        fdr_value = -24.0  # dB
        
        logger.info(f"1. Trim: /ch/{channel}/in/set/trim = {trim_value} dB")
        client.set_channel_gain(channel, trim_value)
        time.sleep(0.3)
        
        logger.info(f"2. Панорама: /ch/{channel}/pan = {pan_value}")
        client.set_channel_pan(channel, pan_value)
        time.sleep(0.3)
        
        logger.info(f"3. Фейдер: /ch/{channel}/fdr = {fdr_value} dB")
        client.set_channel_fader(channel, fdr_value)
        time.sleep(0.3)
        
        # Проверяем установленные значения
        logger.info("\n=== Проверка установленных значений ===")
        logger.info("Запрос значений...")
        client.send(f"/ch/{channel}/in/set/trim")
        client.send(f"/ch/{channel}/pan")
        client.send(f"/ch/{channel}/fdr")
        
        time.sleep(1.5)
        
        trim_new = client.state.get(f"/ch/{channel}/in/set/trim")
        pan_new = client.state.get(f"/ch/{channel}/pan")
        fdr_new = client.state.get(f"/ch/{channel}/fdr")
        
        logger.info("\nУстановленные значения:")
        if trim_new is not None:
            logger.info(f"  Trim: {trim_new} dB {'✓' if abs(trim_new - trim_value) < 0.1 else '⚠'}")
        else:
            logger.warning("  Trim: значение не получено")
        
        if pan_new is not None:
            logger.info(f"  Панорама: {pan_new} {'✓' if abs(pan_new - pan_value) < 1.0 else '⚠'}")
        else:
            logger.warning("  Панорама: значение не получено")
        
        if fdr_new is not None:
            logger.info(f"  Фейдер: {fdr_new} dB {'✓' if abs(fdr_new - fdr_value) < 0.5 else '⚠'}")
        else:
            logger.warning("  Фейдер: значение не получено")
        
        logger.info("\n✓ Команды отправлены. Проверьте физический микшер:")
        logger.info(f"  Канал {channel}: Trim={trim_value} dB, Pan={pan_value}, Fader={fdr_value} dB")
        
        time.sleep(1)
        client.disconnect()
        logger.info("\n✓ Отключено")
    else:
        logger.error("✗ Не удалось подключиться!")
        sys.exit(1)


if __name__ == "__main__":
    main()
