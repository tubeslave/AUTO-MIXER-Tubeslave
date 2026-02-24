"""
Скрипт для проверки готовности к тестированию Gain Staging с Dante

Проверяет:
- Установку зависимостей
- Доступность backend сервера
- Наличие Dante устройств
- Структуру директорий
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Проверка установленных зависимостей"""
    logger.info("=== Проверка зависимостей ===")
    
    dependencies = {
        'websockets': 'websockets>=11.0',
        'pyaudio': 'pyaudio>=0.2.13',
        'pyloudnorm': 'pyloudnorm>=0.1.1',
        'numpy': 'numpy>=1.24.0',
        'asyncio': 'asyncio (built-in)'
    }
    
    missing = []
    for module, requirement in dependencies.items():
        try:
            __import__(module)
            logger.info(f"  ✓ {module}")
        except ImportError:
            logger.error(f"  ✗ {module} - требуется: {requirement}")
            missing.append(requirement)
    
    if missing:
        logger.error(f"\nУстановите недостающие зависимости:")
        logger.error(f"  pip install {' '.join(missing)}")
        return False
    
    logger.info("✓ Все зависимости установлены\n")
    return True


def check_audio_devices():
    """Проверка доступности аудиоустройств"""
    logger.info("=== Проверка аудиоустройств ===")
    
    try:
        from audio_devices import get_audio_devices
        
        devices = get_audio_devices()
        
        if not devices:
            logger.warning("  ⚠ Не найдено аудиоустройств")
            return False
        
        logger.info(f"  Найдено устройств: {len(devices)}")
        
        dante_found = False
        for device in devices:
            name = device.get('name', '')
            max_channels = device.get('max_channels', 0)
            is_dante = 'dante' in name.lower() or 'audinate' in name.lower()
            
            if is_dante:
                logger.info(f"  ✓ Dante устройство: {name} ({max_channels} каналов)")
                dante_found = True
            else:
                logger.info(f"    {name} ({max_channels} каналов)")
        
        if not dante_found:
            logger.warning("  ⚠ Dante устройство не найдено")
            logger.warning("  Убедитесь, что Dante Virtual Soundcard установлен и запущен")
            return False
        
        logger.info("✓ Dante устройство найдено\n")
        return True
        
    except Exception as e:
        logger.error(f"  ✗ Ошибка проверки устройств: {e}")
        return False


def check_directories():
    """Проверка структуры директорий"""
    logger.info("=== Проверка директорий ===")
    
    import os
    from pathlib import Path
    
    required_dirs = [
        'test_results',
        'config'
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.info(f"  Создание директории: {dir_name}")
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"  ✓ {dir_name}")
    
    # Проверка конфига
    config_file = Path('config/default_config.json')
    if config_file.exists():
        logger.info(f"  ✓ config/default_config.json")
    else:
        logger.warning(f"  ⚠ config/default_config.json не найден")
    
    logger.info("✓ Структура директорий готова\n")
    return True


def check_backend_connection():
    """Проверка доступности backend сервера"""
    logger.info("=== Проверка backend сервера ===")
    
    import asyncio
    import websockets
    
    async def check():
        try:
            async with websockets.connect("ws://localhost:8765", timeout=2.0) as ws:
                logger.info("  ✓ Backend сервер доступен на ws://localhost:8765")
                return True
        except Exception as e:
            logger.warning(f"  ⚠ Backend сервер недоступен: {e}")
            logger.warning("  Запустите backend: python server.py")
            return False
    
    try:
        return asyncio.run(check())
    except Exception as e:
        logger.warning(f"  ⚠ Ошибка проверки: {e}")
        return False


def main():
    """Главная функция проверки"""
    logger.info("="*60)
    logger.info("ПРОВЕРКА ГОТОВНОСТИ К ТЕСТИРОВАНИЮ")
    logger.info("="*60 + "\n")
    
    checks = [
        ("Зависимости", check_dependencies),
        ("Аудиоустройства", check_audio_devices),
        ("Директории", check_directories),
        ("Backend сервер", check_backend_connection)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Ошибка при проверке {name}: {e}")
            results.append((name, False))
    
    logger.info("="*60)
    logger.info("РЕЗУЛЬТАТЫ ПРОВЕРКИ")
    logger.info("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✓ ПРОЙДЕНА" if result else "✗ НЕ ПРОЙДЕНА"
        logger.info(f"{name}: {status}")
        if not result:
            all_passed = False
    
    logger.info("="*60)
    
    if all_passed:
        logger.info("\n✓ Все проверки пройдены! Готово к тестированию.")
        logger.info("\nЗапуск теста:")
        logger.info("  python test_gain_staging_dante.py --duration 10.0")
        return 0
    else:
        logger.warning("\n⚠ Некоторые проверки не пройдены.")
        logger.warning("Исправьте проблемы перед запуском тестов.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
