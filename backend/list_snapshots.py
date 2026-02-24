#!/usr/bin/env python3
"""
Получение списка snapshots через node data запросы
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_snapshots_list(client: WingClient):
    """
    Получение списка snapshots через различные методы
    """
    logger.info("=" * 70)
    logger.info("ПОЛУЧЕНИЕ СПИСКА SNAPSHOTS")
    logger.info("=" * 70)
    
    # Метод 1: Node data dump для /$ctl/lib
    logger.info("\nМетод 1: Node data dump для /$ctl/lib")
    try:
        # Отправляем запрос на получение всех данных узла
        client.send("/$ctl/lib ,s *")
        time.sleep(1.0)  # Даем больше времени на ответ
        
        # Проверяем все полученные данные
        logger.info("Полученные данные:")
        lib_keys = [k for k in client.state.keys() if '/$ctl/lib' in k]
        for key in sorted(lib_keys):
            value = client.state.get(key)
            logger.info(f"  {key} = {value}")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
    
    # Метод 2: Пробуем получить список через файловую систему
    # Возможно, есть команда для получения списка файлов snapshots
    logger.info("\nМетод 2: Поиск команд для получения списка файлов")
    file_list_commands = [
        "/snap/list",
        "/snap/$list",
        "/scene/list",
        "/scene/$list",
        "/$ctl/lib/list",
        "/$ctl/lib/$list",
        "/$ctl/lib/snap/list",
        "/$ctl/lib/scene/list",
    ]
    
    for cmd in file_list_commands:
        logger.info(f"  Пробую: {cmd}")
        client.send(cmd)
        time.sleep(0.3)
        value = client.state.get(cmd)
        if value:
            logger.info(f"    ✓ Получено: {value}")
    
    # Метод 3: Пробуем использовать команды навигации для получения имен
    logger.info("\nМетод 3: Навигация по snapshots через PREV/NEXT")
    try:
        # Запоминаем текущий активный
        client.send("/$ctl/lib/$active")
        time.sleep(0.2)
        start_active = client.state.get("/$ctl/lib/$active")
        logger.info(f"Начальный активный: {start_active}")
        
        snapshots = {}
        if start_active:
            snapshots[0] = start_active  # Текущий активный
        
        # Пробуем перейти к следующему и предыдущему
        for direction in ["NEXT", "PREV"]:
            logger.info(f"\nНавигация: {direction}")
            for i in range(10):  # Пробуем до 10 раз
                client.send("/$ctl/lib/$action", direction)
                time.sleep(0.3)
                
                client.send("/$ctl/lib/$active")
                time.sleep(0.2)
                active = client.state.get("/$ctl/lib/$active")
                
                if active and active != start_active:
                    # Нашли новый snapshot
                    # Нужно определить его индекс
                    logger.info(f"  Найден snapshot: {active}")
                    # Но мы не знаем индекс, только имя
                
                # Возвращаемся обратно
                if direction == "NEXT":
                    client.send("/$ctl/lib/$action", "PREV")
                else:
                    client.send("/$ctl/lib/$action", "NEXT")
                time.sleep(0.2)
        
        # Возвращаемся к начальному
        # Пробуем найти его через GO
        if start_active:
            # Может быть нужно использовать другой метод для возврата
            pass
            
    except Exception as e:
        logger.error(f"Ошибка при навигации: {e}")
    
    return {}


def main():
    """Основная функция"""
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.102"
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться к пульту!")
        return 1
    
    try:
        snapshots = get_snapshots_list(client)
        
        logger.info("\n" + "=" * 70)
        if snapshots:
            logger.info("Найденные snapshots:")
            for idx, name in sorted(snapshots.items()):
                logger.info(f"  [{idx:4d}] {name}")
        else:
            logger.info("Не удалось получить список snapshots автоматически")
            logger.info("Проверьте вывод выше для ручного анализа")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.info("\n\nПрервано пользователем")
        return 1
    except Exception as e:
        logger.error(f"\nОшибка: {e}", exc_info=True)
        return 1
    finally:
        client.disconnect()
        logger.info("Отключено от пульта")


if __name__ == "__main__":
    sys.exit(main())
