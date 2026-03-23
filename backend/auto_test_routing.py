#!/usr/bin/env python3
"""
Автоматический тест установки маршрутизации через различные адреса
"""

import sys
import time
import logging
from wing_client import WingClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def auto_test_routing(client: WingClient, channel: int = 1, dante_output: int = 1):
    """Автоматически пробует установить маршрутизацию через различные адреса"""
    
    logger.info(f"Автоматическое тестирование маршрутизации канала {channel} на Dante выход {dante_output}")
    logger.info("=" * 70)
    logger.info("Проверяйте на пульте после каждого метода (пауза 5 секунд)")
    logger.info("=" * 70)
    
    # Варианты адресов для установки маршрутизации
    test_methods = [
        {
            "name": "Метод 1: /routing/outputs/ch/{ch}/srcgrp и /routing/outputs/ch/{ch}/outgrp",
            "addresses": [
                (f"/routing/outputs/ch/{channel}/srcgrp", "WLIVE PLAY"),
                (f"/routing/outputs/ch/{channel}/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 2: /routing/outputs/ch/{ch}/src/grp и /routing/outputs/ch/{ch}/out/grp",
            "addresses": [
                (f"/routing/outputs/ch/{channel}/src/grp", "WLIVE PLAY"),
                (f"/routing/outputs/ch/{channel}/out/grp", dante_output),
            ]
        },
        {
            "name": "Метод 3: /routing/ch/{ch}/outputs/srcgrp и /routing/ch/{ch}/outputs/outgrp",
            "addresses": [
                (f"/routing/ch/{channel}/outputs/srcgrp", "WLIVE PLAY"),
                (f"/routing/ch/{channel}/outputs/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 4: /outputs/ch/{ch}/srcgrp и /outputs/ch/{ch}/outgrp",
            "addresses": [
                (f"/outputs/ch/{channel}/srcgrp", "WLIVE PLAY"),
                (f"/outputs/ch/{channel}/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 5: /ch/{ch}/routing/outputs/srcgrp и /ch/{ch}/routing/outputs/outgrp",
            "addresses": [
                (f"/ch/{channel}/routing/outputs/srcgrp", "WLIVE PLAY"),
                (f"/ch/{channel}/routing/outputs/outgrp", dante_output),
            ]
        },
        {
            "name": "Метод 6: Альтернативные названия - sourcegrp и outputgrp",
            "addresses": [
                (f"/routing/outputs/ch/{channel}/sourcegrp", "WLIVE PLAY"),
                (f"/routing/outputs/ch/{channel}/outputgrp", dante_output),
            ]
        },
    ]
    
    for i, method in enumerate(test_methods, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"МЕТОД {i}: {method['name']}")
        logger.info(f"{'='*70}")
        
        for addr, value in method['addresses']:
            try:
                logger.info(f"  Отправка: {addr} = {value}")
                result = client.send(addr, value)
                time.sleep(0.1)
                if result:
                    logger.info(f"    ✓ Успешно отправлено")
                else:
                    logger.warning(f"    ✗ Ошибка отправки")
            except Exception as e:
                logger.error(f"    ✗ Ошибка: {e}")
        
        logger.info(f"\n⏸  Пауза 5 секунд - проверьте на пульте, изменилась ли маршрутизация канала {channel}")
        logger.info(f"   Если сработало, нажмите Ctrl+C и сообщите номер метода: {i}")
        time.sleep(5)
    
    logger.info("\n" + "="*70)
    logger.info("Все методы протестированы.")
    logger.info("Если ни один не сработал, возможно нужны другие адреса.")
    logger.info("="*70)


def main():
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = "192.168.1.102"
    
    logger.info(f"Подключение к пульту {ip}...")
    client = WingClient(ip=ip, port=2223)
    
    if not client.connect():
        logger.error("Не удалось подключиться!")
        return 1
    
    try:
        auto_test_routing(client, channel=1, dante_output=1)
    except KeyboardInterrupt:
        logger.info("\n\nПрервано пользователем")
    finally:
        client.disconnect()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
