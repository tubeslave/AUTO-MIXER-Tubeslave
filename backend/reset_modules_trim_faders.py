#!/usr/bin/env python3
"""
Скрипт для полного сброса всех модулей, их параметров, trim и фейдеров на всех каналах:

Модули и их параметры:
- EQ: выключен, все полосы (low shelf, bands 1-4, high shelf) gain=0, mix=100%
- PreEQ: выключен, все полосы (1-3) gain=0
- Compressor/Dynamics: выключен, threshold=-10дБ, ratio=3.0, attack=10мс, release=100мс, gain=0, mix=100%, knee=0
- Gate: выключен, threshold=-40дБ, range=10дБ, attack=5мс, release=100мс, accent=50
- Filters: Low cut и High cut выключены, model=TILT, tilt=0
- Inserts: Pre и Post выключены, все параметры FX модулей сброшены

Базовые параметры каналов:
- Trim = 0дБ
- Fader = 0дБ
- Pan = 0 (центр)
"""
from wing_client import WingClient
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def reset_fx_module(client: WingClient, fx_slot: str):
    """Сбросить параметры FX модуля к значениям по умолчанию"""
    fx_num = fx_slot.replace('FX', '') if fx_slot.startswith('FX') else fx_slot
    
    # Выключить FX модуль
    client.set_fx_on(fx_slot, 0)
    time.sleep(0.01)
    
    # Сбросить mix на 100%
    client.set_fx_mix(fx_slot, 100.0)
    time.sleep(0.01)
    
    # Сбросить все пронумерованные параметры на 0 (по умолчанию)
    # Большинство FX модулей используют параметры 1-32
    for param_num in range(1, 33):
        try:
            client.set_fx_parameter(fx_slot, param_num, 0.0)
            if param_num % 8 == 0:  # Небольшая задержка каждые 8 параметров
                time.sleep(0.01)
        except Exception as e:
            logger.debug(f"Не удалось сбросить параметр {param_num} FX{fx_num}: {e}")
    
    time.sleep(0.05)


def reset_channel_modules_trim_fader(client: WingClient, channel: int):
    """Сбросить модули, их параметры, trim и фейдер для одного канала"""
    ch = channel
    
    logger.info(f"Сброс канала {ch}...")
    
    # 1. Сбросить trim на 0дБ
    client.set_channel_gain(ch, 0.0)
    time.sleep(0.01)
    
    # 2. Сбросить фейдер на 0дБ
    client.set_channel_fader(ch, 0.0)
    time.sleep(0.01)
    
    # 3. Сбросить панораму на центр (0)
    client.set_channel_pan(ch, 0.0)
    time.sleep(0.01)
    
    # 4. Выключить и сбросить параметры EQ
    client.set_eq_on(ch, 0)
    time.sleep(0.01)
    
    # Сбросить все полосы EQ: gain=0
    # Low shelf
    client.set_eq_low_shelf(ch, gain=0.0)
    time.sleep(0.01)
    
    # Bands 1-4
    for band in range(1, 5):
        client.set_eq_band(ch, band, gain=0.0)
        time.sleep(0.01)
    
    # High shelf
    client.set_eq_high_shelf(ch, gain=0.0)
    time.sleep(0.01)
    
    # Сбросить EQ mix на 100%
    client.set_eq_mix(ch, 100.0)
    time.sleep(0.01)
    
    # 5. Выключить и сбросить параметры PreEQ
    client.send(f"/ch/{ch}/peq/on", 0)
    time.sleep(0.01)
    
    # Сбросить все полосы PreEQ: gain=0
    for band in range(1, 4):
        client.send(f"/ch/{ch}/peq/{band}g", 0.0)  # Gain = 0
        time.sleep(0.01)
    
    # 6. Выключить и сбросить параметры Compressor/Dynamics
    client.set_compressor_on(ch, 0)
    time.sleep(0.01)
    
    # Сбросить параметры компрессора к значениям по умолчанию
    client.set_compressor(
        ch,
        threshold=-10.0,  # Дефолтный threshold
        ratio="3.0",      # Дефолтный ratio
        attack=10.0,      # Дефолтный attack
        release=100.0,    # Дефолтный release
        gain=0.0,         # Без make-up gain
        mix=100.0,        # 100% mix
        knee=0            # Дефолтный knee
    )
    time.sleep(0.01)
    
    # 7. Выключить и сбросить параметры Gate
    client.set_gate_on(ch, 0)
    time.sleep(0.01)
    
    # Сбросить параметры gate к значениям по умолчанию
    client.set_gate(
        ch,
        threshold=-40.0,   # Дефолтный threshold
        range_db=10.0,     # Дефолтный range
        attack=5.0,        # Дефолтный attack
        release=100.0,     # Дефолтный release
        accent=50.0        # Дефолтный accent
    )
    time.sleep(0.01)
    
    # 8. Выключить Filters (Low cut и High cut)
    client.set_low_cut(ch, enabled=0)
    time.sleep(0.01)
    client.set_high_cut(ch, enabled=0)
    time.sleep(0.01)
    
    # Сбросить filter model на TILT (по умолчанию)
    client.send(f"/ch/{ch}/flt/mdl", "TILT")
    time.sleep(0.01)
    
    # Выключить tool filter
    client.send(f"/ch/{ch}/flt/tf", 0)
    time.sleep(0.01)
    
    # Сбросить filter tilt на 0
    client.send(f"/ch/{ch}/flt/tilt", 0.0)
    time.sleep(0.01)
    
    # 9. Выключить и сбросить параметры Inserts (Pre и Post)
    # Проверяем, есть ли активные инсерты, и сбрасываем их параметры
    preins_slot = client.state.get(f'/ch/{ch}/preins/ins')
    postins_slot = client.state.get(f'/ch/{ch}/postins/ins')
    
    # Выключить pre-insert
    client.send(f"/ch/{ch}/preins/on", 0)
    time.sleep(0.01)
    
    # Если есть FX модуль в pre-insert, сбросить его параметры
    if preins_slot and preins_slot != 'NONE':
        reset_fx_module(client, preins_slot)
    
    # Выключить post-insert
    client.send(f"/ch/{ch}/postins/on", 0)
    time.sleep(0.01)
    
    # Если есть FX модуль в post-insert, сбросить его параметры
    if postins_slot and postins_slot != 'NONE':
        reset_fx_module(client, postins_slot)
    
    logger.debug(f"Канал {ch} сброшен: модули выключены и параметры сброшены, trim=0дБ, фейдер=0дБ, pan=0")


def reset_all_channels(auto_confirm=False):
    """Сбросить все 40 каналов"""
    
    # Попытка загрузить IP из конфига
    default_ip = "192.168.1.102"
    default_port = 2223
    try:
        import os
        import json
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
        print("ВНИМАНИЕ: Это сбросит настройки на ВСЕХ 40 каналах!")
        print("  - Все модули ВЫКЛЮЧЕНЫ (EQ, PreEQ, Gate, Dynamics, Filters, Inserts)")
        print("  - Все параметры модулей сброшены к значениям по умолчанию:")
        print("    * EQ: все полосы gain=0, mix=100%")
        print("    * PreEQ: все полосы gain=0")
        print("    * Compressor: threshold=-10дБ, ratio=3.0, attack=10мс, release=100мс, gain=0, mix=100%")
        print("    * Gate: threshold=-40дБ, range=10дБ, attack=5мс, release=100мс, accent=50")
        print("    * Filters: выключены, model=TILT, tilt=0")
        print("    * Inserts: выключены, FX параметры сброшены")
        print("  - Все trim = 0дБ")
        print("  - Все фейдеры = 0дБ")
        print("  - Все панорамы = 0 (центр)")
        print("="*60)
        confirm = input("\nВведите 'ДА' для продолжения: ").strip()
        
        if confirm != 'ДА':
            logger.info("Операция отменена")
            client.disconnect()
            return
    
    logger.info("\nНачало сброса всех 40 каналов...")
    logger.info("Это может занять некоторое время...\n")
    
    start_time = time.time()
    
    # Сброс всех каналов
    for ch in range(1, 41):
        try:
            reset_channel_modules_trim_fader(client, ch)
            if ch % 10 == 0:
                logger.info(f"Прогресс: {ch}/40 каналов обработано")
        except Exception as e:
            logger.error(f"Ошибка при сбросе канала {ch}: {e}")
    
    elapsed = time.time() - start_time
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Сброс завершен! Обработано 40 каналов за {elapsed:.1f} секунд")
    logger.info(f"{'='*60}\n")
    
    client.disconnect()
    logger.info("Отключено")


if __name__ == "__main__":
    import sys
    # Разрешить автоматическое подтверждение с флагом --yes
    auto_confirm = '--yes' in sys.argv or '-y' in sys.argv
    reset_all_channels(auto_confirm=auto_confirm)
