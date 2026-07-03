#!/usr/bin/env python3
"""
Скрипт для выключения всех модулей и установки фейдеров на 0дБ на всех 40 каналах

Что делает скрипт:
- Выключает все модули (EQ, PreEQ, Compressor, Gate, Filters, Inserts)
- Устанавливает все фейдеры на 0дБ
"""
from wing_client import WingClient
import time
import logging
import sys
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def disable_channel_modules_and_set_fader(client: WingClient, channel: int):
    """Выключить все модули и установить фейдер на 0дБ для одного канала"""
    ch = channel
    
    logger.debug(f"Обработка канала {ch}...")
    
    # 1. Установить фейдер на 0дБ
    client.set_channel_fader(ch, 0.0)
    time.sleep(0.01)
    
    # 2. Выключить EQ
    client.set_eq_on(ch, 0)
    time.sleep(0.01)
    
    # 3. Выключить PreEQ
    client.send(f"/ch/{ch}/peq/on", 0)
    time.sleep(0.01)
    
    # 4. Выключить Compressor/Dynamics
    client.set_compressor_on(ch, 0)
    time.sleep(0.01)
    
    # 5. Выключить Gate
    client.set_gate_on(ch, 0)
    time.sleep(0.01)
    
    # 6. Выключить Filters (Low cut и High cut)
    client.set_low_cut(ch, enabled=0)
    time.sleep(0.01)
    client.set_high_cut(ch, enabled=0)
    time.sleep(0.01)
    
    # 7. Выключить Inserts (Pre и Post)
    client.send(f"/ch/{ch}/preins/on", 0)
    time.sleep(0.01)
    client.send(f"/ch/{ch}/postins/on", 0)
    time.sleep(0.01)


def disable_all_modules_and_set_faders(auto_confirm=False):
    """Выключить все модули и установить фейдеры на 0дБ для всех 40 каналов"""
    
    # Попытка загрузить IP из конфига
    default_ip = "192.168.1.102"
    default_port = 2223
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  'config', 'default_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'wing' in config and 'default_ip' in config['wing']:
                    default_ip = config['wing']['default_ip']
                if 'wing' in config and 'receive_port' in config['wing']:
                    default_port = config['wing']['receive_port']
    except Exception as e:
        logger.debug(f"Не удалось загрузить конфиг: {e}")
    
    # Подключение к Wing - разрешить аргументы командной строки
    # Если auto_confirm=True, используем значения по умолчанию без запроса
    if len(sys.argv) >= 2 and not sys.argv[1].startswith('--'):
        ip = sys.argv[1]
    elif auto_confirm:
        ip = default_ip
        logger.info(f"Используется IP адрес из конфига: {ip}")
    else:
        ip = input(f"Введите IP адрес Wing [{default_ip}]: ").strip() or default_ip
    
    if len(sys.argv) >= 3 and not sys.argv[2].startswith('--'):
        port = int(sys.argv[2])
    elif auto_confirm:
        port = default_port
        logger.info(f"Используется порт из конфига: {port}")
    else:
        port = int(input(f"Введите OSC порт [{default_port}]: ").strip() or default_port)
    
    logger.info(f"Подключение к Wing по адресу {ip}:{port}...")
    client = WingClient(ip, port)
    
    if not client.connect():
        logger.error("Не удалось подключиться к Wing")
        return
    
    logger.info("Подключено! Ожидание начального сканирования...")
    time.sleep(3)  # Ожидание начального сканирования
    
    # Подтверждение перед выполнением (если не auto_confirm)
    if not auto_confirm:
        print("\n" + "="*60)
        print("ВНИМАНИЕ: Это выключит все модули и установит фейдеры на 0дБ на ВСЕХ 40 каналах!")
        print("  - Все модули ВЫКЛЮЧЕНЫ (EQ, PreEQ, Gate, Dynamics, Filters, Inserts)")
        print("  - Все фейдеры = 0дБ")
        print("="*60)
        confirm = input("\nВведите 'ДА' для продолжения: ").strip()
        
        if confirm != 'ДА':
            logger.info("Операция отменена")
            client.disconnect()
            return
    
    logger.info("\nНачало обработки всех 40 каналов...")
    logger.info("Это может занять некоторое время...\n")
    
    start_time = time.time()
    
    # Обработка всех каналов
    for ch in range(1, 41):
        try:
            disable_channel_modules_and_set_fader(client, ch)
            if ch % 10 == 0:
                logger.info(f"Прогресс: {ch}/40 каналов обработано")
        except Exception as e:
            logger.error(f"Ошибка при обработке канала {ch}: {e}")
    
    elapsed = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Обработка завершена! Обработано 40 каналов за {elapsed:.1f} секунд")
    logger.info(f"{'='*60}\n")
    
    client.disconnect()
    logger.info("Отключено")


if __name__ == "__main__":
    import sys
    # Разрешить автоматическое подтверждение с флагом --yes
    auto_confirm = '--yes' in sys.argv or '-y' in sys.argv
    disable_all_modules_and_set_faders(auto_confirm=auto_confirm)
