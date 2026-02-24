#!/usr/bin/env python3
"""
Тестовый скрипт для проверки сброса TRIM
"""
import logging
import sys
import time
from wing_client import WingClient

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_trim_reset():
    """Тест сброса TRIM для каналов"""
    
    # IP адрес микшера (измените на ваш)
    wing_ip = "192.168.1.100"
    
    logger.info("=" * 60)
    logger.info("TEST: TRIM Reset")
    logger.info("=" * 60)
    
    # Подключение к микшеру
    client = WingClient(ip=wing_ip, port=2223)
    
    if not client.connect():
        logger.error("Failed to connect to Wing mixer")
        return False
    
    logger.info("✅ Connected to Wing mixer")
    
    # Тестовые каналы
    test_channels = [1, 2, 3]
    
    logger.info(f"\nTesting TRIM reset for channels: {test_channels}")
    
    # Шаг 1: Получить текущие значения TRIM
    logger.info("\n--- Step 1: Reading current TRIM values ---")
    current_trims = {}
    for ch in test_channels:
        trim = client.get_channel_gain(ch)
        current_trims[ch] = trim
        logger.info(f"Channel {ch}: Current TRIM = {trim} dB")
    
    # Шаг 2: Установить TRIM в 0 dB
    logger.info("\n--- Step 2: Resetting TRIM to 0 dB ---")
    for ch in test_channels:
        logger.info(f"Setting channel {ch} TRIM to 0.0 dB...")
        result = client.set_channel_gain(ch, 0.0)
        if result:
            logger.info(f"✅ Channel {ch}: Command sent successfully")
        else:
            logger.error(f"❌ Channel {ch}: Failed to send command")
        time.sleep(0.1)  # Небольшая задержка между командами
    
    # Шаг 3: Подождать применения
    logger.info("\n--- Step 3: Waiting for TRIM values to be applied ---")
    time.sleep(0.5)
    
    # Шаг 4: Проверить значения после сброса
    logger.info("\n--- Step 4: Verifying TRIM values after reset ---")
    success_count = 0
    for ch in test_channels:
        trim = client.get_channel_gain(ch)
        logger.info(f"Channel {ch}: TRIM after reset = {trim} dB")
        if trim is not None and abs(trim - 0.0) < 0.1:  # Допуск 0.1 dB
            logger.info(f"✅ Channel {ch}: TRIM successfully reset to 0 dB")
            success_count += 1
        else:
            logger.warning(f"⚠️  Channel {ch}: TRIM is {trim} dB (expected 0.0 dB)")
    
    # Результаты
    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS: {success_count}/{len(test_channels)} channels reset successfully")
    logger.info("=" * 60)
    
    # Восстановить исходные значения (опционально)
    try:
        restore = input("\nRestore original TRIM values? (y/n): ")
        if restore.lower() == 'y':
            logger.info("\n--- Restoring original TRIM values ---")
            for ch in test_channels:
                if ch in current_trims and current_trims[ch] is not None:
                    client.set_channel_gain(ch, current_trims[ch])
                    logger.info(f"Channel {ch}: Restored to {current_trims[ch]} dB")
                    time.sleep(0.1)
    except (EOFError, KeyboardInterrupt):
        logger.info("\nSkipping restore (non-interactive mode)")
    
    client.disconnect()
    logger.info("\n✅ Test completed")
    
    return success_count == len(test_channels)

if __name__ == "__main__":
    try:
        success = test_trim_reset()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nTest failed with error: {e}", exc_info=True)
        sys.exit(1)
