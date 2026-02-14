#!/usr/bin/env python3
"""
Программа для мониторинга OSC сообщений от пульта Wing
Слушает все сообщения и выводит их, чтобы найти правильные адреса маршрутизации
"""

import sys
import time
import logging
from datetime import datetime
from wing_client import WingClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Хранилище для отслеживания изменений
seen_addresses = set()
recent_messages = []


def on_osc_message(address: str, *args):
    """
    Callback для обработки входящих OSC сообщений
    """
    global seen_addresses, recent_messages
    
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    
    # Форматируем аргументы
    if len(args) == 0:
        args_str = "(нет аргументов)"
    elif len(args) == 1:
        args_str = str(args[0])
    else:
        args_str = str(args)
    
    # Проверяем, новый ли это адрес или изменилось значение
    is_new_address = address not in seen_addresses
    seen_addresses.add(address)
    
    # Сохраняем последние сообщения
    message = {
        'timestamp': timestamp,
        'address': address,
        'args': args,
        'is_new': is_new_address
    }
    recent_messages.append(message)
    if len(recent_messages) > 100:  # Храним последние 100 сообщений
        recent_messages.pop(0)
    
    # Выделяем важные адреса (связанные с routing, out, card)
    is_routing_related = any(keyword in address.lower() for keyword in [
        'out', 'routing', 'card', 'conn', 'grp', 'signal', 'dante', 'wlive', 
        'srcgrp', 'outgrp', 'source', 'outputs', 'config'
    ])
    
    # Выводим сообщение
    if is_routing_related:
        # Важные адреса выделяем
        marker = "🔍" if is_new_address else "  "
        logger.info(f"{marker} [{timestamp}] {address:60s} = {args_str}")
    elif is_new_address:
        # Новые адреса тоже показываем
        logger.info(f"✨ [{timestamp}] {address:60s} = {args_str}")
    else:
        # Остальные адреса показываем только если они изменились
        # (для уменьшения шума)
        pass


def print_summary(client: WingClient):
    """Выводит сводку по найденным адресам"""
    logger.info("\n" + "=" * 80)
    logger.info("СВОДКА ПО НАЙДЕННЫМ АДРЕСАМ")
    logger.info("=" * 80)
    
    routing_addresses = [addr for addr in seen_addresses if any(
        keyword in addr.lower() for keyword in ['out', 'routing', 'card', 'conn', 'grp', 'signal']
    )]
    
    if routing_addresses:
        logger.info(f"\nНайдено {len(routing_addresses)} адресов, связанных с маршрутизацией:")
        for addr in sorted(routing_addresses):
            value = client.state.get(addr)
            logger.info(f"  {addr:60s} = {value}")
    else:
        logger.info("\nАдреса маршрутизации пока не найдены.")
    
    logger.info(f"\nВсего уникальных адресов получено: {len(seen_addresses)}")
    logger.info("=" * 80)


def main():
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = "192.168.1.102"
    
    logger.info("=" * 80)
    logger.info("МОНИТОРИНГ OSC СООБЩЕНИЙ ОТ ПУЛЬТА WING")
    logger.info("=" * 80)
    logger.info(f"IP пульта: {ip}")
    logger.info(f"Порт: 2223")
    logger.info("")
    logger.info("📡 Подключение к пульту...")
    logger.info("")
    
    # Создаем клиент
    client = WingClient(ip=ip, port=2223)
    
    # Подключаемся
    if not client.connect():
        logger.error("❌ Не удалось подключиться к пульту!")
        logger.error("Проверьте:")
        logger.error("  1. IP адрес пульта")
        logger.error("  2. Сетевое подключение")
        logger.error("  3. Что пульт включен и доступен")
        return 1
    
    logger.info("✅ Подключение успешно!")
    logger.info("")
    logger.info("=" * 80)
    logger.info("НАЧАЛО МОНИТОРИНГА")
    logger.info("=" * 80)
    logger.info("")
    logger.info("🔍 Слушаю все OSC сообщения от пульта...")
    logger.info("")
    logger.info("💡 ИНСТРУКЦИЯ:")
    logger.info("   1. Подождите несколько секунд для стабилизации")
    logger.info("   2. На пульте переключите маршрутизацию канала 1 на Dante выход")
    logger.info("   3. Я увижу какие OSC адреса изменились")
    logger.info("")
    logger.info("⏸  Нажмите Ctrl+C для остановки")
    logger.info("")
    logger.info("-" * 80)
    logger.info("")
    
    # Подписываемся на все обновления
    client.subscribe("*", on_osc_message)
    
    # Также запрашиваем текущее состояние канала 1 для сравнения
    logger.info("📋 Запрашиваю текущее состояние канала 1...")
    channel = 1
    
    # Запрашиваем различные возможные адреса маршрутизации
    test_addresses = [
        f"/ch/{channel}/out",
        f"/ch/{channel}/out/conn",
        f"/ch/{channel}/out/conn/grp",
        f"/ch/{channel}/out/conn/out",
        f"/ch/{channel}/out/conn/type",
        f"/ch/{channel}/out/grp",
        f"/ch/{channel}/out/out",
        f"/ch/{channel}/routing",
        f"/ch/{channel}/mix/out",
    ]
    
    for addr in test_addresses:
        client.send(addr)
        time.sleep(0.05)
    
    time.sleep(0.5)
    logger.info("✅ Запросы отправлены. Жду изменений...")
    logger.info("")
    
    try:
        # Мониторим в течение длительного времени
        # Пользователь должен переключить маршрутизацию на пульте
        start_time = time.time()
        last_summary_time = start_time
        
        while True:
            time.sleep(0.1)
            
            # Каждые 5 секунд выводим краткую сводку
            current_time = time.time()
            if current_time - last_summary_time > 5:
                routing_count = len([m for m in recent_messages[-20:] if any(
                    keyword in m['address'].lower() 
                    for keyword in ['out', 'routing', 'card', 'conn', 'grp', 'signal']
                )])
                if routing_count > 0:
                    logger.info(f"📊 Получено {routing_count} сообщений о маршрутизации за последние 5 секунд")
                last_summary_time = current_time
            
    except KeyboardInterrupt:
        logger.info("")
        logger.info("")
        logger.info("⏹  Остановка мониторинга...")
        logger.info("")
        
        # Выводим сводку
        print_summary(client)
        
        # Выводим последние важные сообщения
        logger.info("")
        logger.info("=" * 80)
        logger.info("ПОСЛЕДНИЕ СООБЩЕНИЯ О МАРШРУТИЗАЦИИ")
        logger.info("=" * 80)
        
        routing_messages = [m for m in recent_messages if any(
            keyword in m['address'].lower() 
            for keyword in ['out', 'routing', 'card', 'conn', 'grp', 'signal', 'dante', 'wlive']
        )]
        
        if routing_messages:
            for msg in routing_messages[-20:]:  # Последние 20 сообщений
                args_str = str(msg['args']) if msg['args'] else "(нет аргументов)"
                logger.info(f"[{msg['timestamp']}] {msg['address']:60s} = {args_str}")
        else:
            logger.info("Сообщений о маршрутизации не найдено.")
        
        logger.info("")
        logger.info("=" * 80)
    
    finally:
        client.disconnect()
        logger.info("Отключено от пульта")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
