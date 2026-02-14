"""
Тестовый скрипт для Gain Staging с Dante аудиоинтерфейсом

Согласно плану тестирования:
- Подключается к backend через WebSocket
- Инициализирует gain staging с Dante устройством
- Запускает анализ для всех каналов
- Собирает данные измерений (RMS, Peak, Crest Factor, LUFS)
- Сохраняет результаты в JSON файл
- Логирует все операции
"""

import sys
import time
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
import websockets
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_gain_staging_dante.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DanteGainStagingTest:
    """Класс для тестирования Gain Staging с Dante через WebSocket"""
    
    def __init__(self, backend_url: str = "ws://localhost:8765", 
                 wing_ip: Optional[str] = None,
                 results_dir: str = "test_results"):
        """
        Инициализация теста
        
        Args:
            backend_url: URL WebSocket сервера backend
            wing_ip: IP адрес пульта Wing (опционально)
            results_dir: Директория для сохранения результатов
        """
        self.backend_url = backend_url
        self.wing_ip = wing_ip
        self.results_dir = results_dir
        self.websocket = None
        self.test_data = {
            'start_time': None,
            'end_time': None,
            'dante_device': None,
            'channels': [],
            'channel_settings': {},
            'measurements': [],
            'adjustments': {},
            'errors': [],
            'warnings': []
        }
        
        # Создать директорию для результатов
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
    
    async def connect_to_backend(self) -> bool:
        """Подключение к backend WebSocket серверу"""
        logger.info(f"=== Подключение к backend: {self.backend_url} ===")
        
        try:
            self.websocket = await websockets.connect(self.backend_url)
            logger.info("✓ Подключено к backend")
            return True
        except Exception as e:
            logger.error(f"✗ Ошибка подключения к backend: {e}")
            logger.error("Убедитесь, что backend сервер запущен: python server.py")
            return False
    
    async def send_message(self, message: dict) -> None:
        """Отправка сообщения через WebSocket"""
        if not self.websocket:
            raise RuntimeError("WebSocket не подключен")
        
        await self.websocket.send(json.dumps(message))
        logger.debug(f"Отправлено: {message.get('type')}")
    
    async def receive_message(self, timeout: float = 5.0) -> Optional[dict]:
        """Получение сообщения через WebSocket"""
        if not self.websocket:
            raise RuntimeError("WebSocket не подключен")
        
        try:
            message = await asyncio.wait_for(self.websocket.recv(), timeout=timeout)
            data = json.loads(message)
            logger.debug(f"Получено: {data.get('type')}")
            return data
        except asyncio.TimeoutError:
            logger.warning(f"Таймаут ожидания сообщения ({timeout}s)")
            return None
        except Exception as e:
            logger.error(f"Ошибка получения сообщения: {e}")
            return None
    
    async def get_audio_devices(self) -> List[Dict]:
        """Получение списка аудиоустройств через WebSocket"""
        logger.info("=== Получение списка аудиоустройств ===")
        
        await self.send_message({"type": "get_audio_devices"})
        
        # Ждем ответ
        response = await self.receive_message(timeout=10.0)
        
        if response and response.get("type") == "audio_devices":
            devices = response.get("devices", [])
            logger.info(f"Найдено устройств: {len(devices)}")
            
            for i, device in enumerate(devices):
                name = device.get('name', 'Unknown')
                max_channels = device.get('max_channels', 0)
                is_dante = 'dante' in name.lower() or 'audinate' in name.lower()
                logger.info(f"  [{i}] {name} - {max_channels} каналов {'[DANTE]' if is_dante else ''}")
            
            return devices
        else:
            logger.error("Не удалось получить список устройств")
            return []
    
    def find_dante_device(self, devices: List[Dict]) -> Optional[Dict]:
        """Поиск Dante устройства"""
        logger.info("=== Поиск Dante устройства ===")
        
        dante_devices = []
        for device in devices:
            name = device.get('name', '')
            if 'dante' in name.lower() or 'audinate' in name.lower():
                dante_devices.append(device)
        
        if not dante_devices:
            logger.warning("Dante устройства не найдены автоматически")
            logger.info("Доступные устройства:")
            for i, device in enumerate(devices):
                logger.info(f"  [{i}] {device.get('name')} - {device.get('max_channels')} каналов")
            return None
        
        if len(dante_devices) == 1:
            device = dante_devices[0]
            logger.info(f"✓ Найдено Dante устройство: {device.get('name')}")
            return device
        else:
            logger.info(f"Найдено {len(dante_devices)} Dante устройств:")
            for i, device in enumerate(dante_devices):
                logger.info(f"  [{i}] {device.get('name')} - {device.get('max_channels')} каналов")
            # Используем первое по умолчанию
            device = dante_devices[0]
            logger.info(f"Используется: {device.get('name')}")
            return device
    
    def select_all_channels(self, device: Dict) -> List[int]:
        """Автоматический выбор всех каналов Dante устройства"""
        max_channels = device.get('max_channels', 0)
        
        if max_channels == 0:
            logger.error("Устройство не имеет входных каналов")
            return []
        
        channels = list(range(1, max_channels + 1))
        logger.info(f"=== Выбраны все каналы: {channels} (всего {len(channels)}) ===")
        
        return channels
    
    def create_channel_settings(self, channels: List[int]) -> Dict[int, Dict]:
        """Создание настроек для каналов с дефолтными значениями"""
        logger.info("=== Создание настроек каналов ===")
        
        # Загрузка конфигурации
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'config', 'default_config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            auto_gain_config = config.get('automation', {}).get('auto_gain', {})
            target_level = auto_gain_config.get('target_level', -18.0)
        except Exception as e:
            logger.warning(f"Не удалось загрузить конфиг: {e}, используются значения по умолчанию")
            target_level = -18.0
        
        channel_settings = {}
        for ch in channels:
            channel_settings[ch] = {
                'rms': target_level,
                'peak': -6.0,
                'crestFactor': 12.0
            }
        
        logger.info(f"Настройки для {len(channels)} каналов:")
        logger.info(f"  Target RMS: {target_level} dB")
        logger.info(f"  Target Peak: -6.0 dB")
        logger.info(f"  Target Crest Factor: 12.0 dB")
        
        return channel_settings
    
    async def connect_to_wing(self) -> bool:
        """Подключение к пульту Wing"""
        if not self.wing_ip:
            logger.info("=== Подключение к Wing пропущено (IP не указан) ===")
            return False
        
        logger.info(f"=== Подключение к Wing ({self.wing_ip}) ===")
        
        await self.send_message({
            "type": "connect_wing",
            "ip": self.wing_ip,
            "send_port": 2222,
            "receive_port": 2223
        })
        
        # Ждем ответ о статусе подключения
        response = await self.receive_message(timeout=10.0)
        
        if response and response.get("type") == "connection_status":
            connected = response.get("connected", False)
            if connected:
                logger.info("✓ Подключено к Wing")
                return True
            else:
                logger.error("✗ Не удалось подключиться к Wing")
                return False
        else:
            logger.warning("Не получен ответ о статусе подключения")
            return False
    
    async def start_gain_staging(self, device: Dict, channels: List[int], 
                               channel_settings: Dict[int, Dict]) -> bool:
        """Запуск анализа Gain Staging"""
        logger.info("=== Запуск анализа Gain Staging ===")
        
        device_id = device.get('id')
        device_name = device.get('name')
        
        logger.info(f"Устройство: {device_name} (ID: {device_id})")
        logger.info(f"Каналы: {channels}")
        
        # Создание channel_mapping (1:1 mapping)
        channel_mapping = {str(ch): ch for ch in channels}
        
        # Конвертация channel_settings в формат для WebSocket
        ws_channel_settings = {}
        for ch, settings in channel_settings.items():
            ws_channel_settings[str(ch)] = settings
        
        # Сохранение в test_data
        self.test_data['dante_device'] = {
            'id': device_id,
            'name': device_name,
            'max_channels': device.get('max_channels')
        }
        self.test_data['channels'] = channels
        self.test_data['channel_settings'] = channel_settings
        
        # Отправка команды запуска
        await self.send_message({
            "type": "start_gain_staging",
            "device_id": device_id,
            "channels": channels,
            "channel_settings": ws_channel_settings,
            "channel_mapping": channel_mapping
        })
        
        # Ждем подтверждение
        response = await self.receive_message(timeout=10.0)
        
        if response and response.get("type") == "gain_staging_status":
            active = response.get("active", False)
            if active:
                logger.info("✓ Анализ Gain Staging запущен")
                self.test_data['start_time'] = datetime.now().isoformat()
                return True
            else:
                error = response.get("error", "Unknown error")
                logger.error(f"✗ Не удалось запустить анализ: {error}")
                self.test_data['errors'].append({
                    'time': datetime.now().isoformat(),
                    'error': f"Failed to start: {error}"
                })
                return False
        else:
            logger.error("Не получен ответ о статусе запуска")
            return False
    
    async def collect_measurements(self, duration: float = 10.0) -> None:
        """Сбор данных измерений"""
        logger.info(f"=== Сбор данных измерений ({duration} секунд) ===")
        logger.info("Подавайте тестовый сигнал на каналы Dante...")
        
        start_time = time.time()
        last_status_time = 0
        
        while time.time() - start_time < duration:
            try:
                # Получаем сообщения от backend
                response = await self.receive_message(timeout=1.0)
                
                if response:
                    if response.get("type") == "gain_staging_status":
                        status_type = response.get("status_type", "unknown")
                        
                        if status_type == "levels_update":
                            channels_data = response.get("channels", {})
                            
                            # Сохраняем измерения
                            measurement = {
                                'timestamp': datetime.now().isoformat(),
                                'elapsed_time': round(time.time() - start_time, 2),
                                'channels': {}
                            }
                            
                            for ch_str, data in channels_data.items():
                                ch = int(ch_str)
                                measurement['channels'][ch] = {
                                    'measured_rms': data.get('measured_rms'),
                                    'measured_peak': data.get('measured_peak'),
                                    'measured_crest': data.get('measured_crest'),
                                    'measured_lufs': data.get('measured_lufs'),
                                    'adjustment': data.get('adjustment'),
                                    'signal_present': data.get('signal_present'),
                                    'stabilized': data.get('stabilized'),
                                    'bleeding_detected': data.get('bleeding_detected'),
                                    'bleeding_dominant_channel': data.get('bleeding_dominant_channel'),
                                    'valid_samples': data.get('valid_samples'),
                                    'total_samples': data.get('total_samples')
                                }
                            
                            self.test_data['measurements'].append(measurement)
                            
                            # Логируем каждые 5 секунд
                            elapsed = time.time() - start_time
                            if elapsed - last_status_time >= 5.0:
                                logger.info(f"Прошло {int(elapsed)} секунд...")
                                self._log_current_status(channels_data)
                                last_status_time = elapsed
                        
                        elif status_type == "analysis_complete":
                            logger.info("=== Анализ завершен ===")
                            channels_data = response.get("channels", {})
                            self._log_final_status(channels_data)
                            break
                
                # Проверка времени
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    break
                    
            except Exception as e:
                logger.error(f"Ошибка при сборе данных: {e}")
                self.test_data['errors'].append({
                    'time': datetime.now().isoformat(),
                    'error': str(e)
                })
        
        logger.info(f"✓ Сбор данных завершен ({duration} секунд)")
    
    def _log_current_status(self, channels_data: Dict) -> None:
        """Логирование текущего статуса"""
        logger.info("--- Текущие уровни ---")
        for ch_str, data in channels_data.items():
            ch = int(ch_str)
            rms = data.get('measured_rms', 0)
            peak = data.get('measured_peak', 0)
            signal = '✓' if data.get('signal_present') else '✗'
            stabilized = '✓' if data.get('stabilized') else '✗'
            logger.info(f"Канал {ch}: RMS={rms:.1f} dB, Peak={peak:.1f} dB, "
                       f"Signal={signal}, Stabilized={stabilized}")
    
    def _log_final_status(self, channels_data: Dict) -> None:
        """Логирование финального статуса"""
        logger.info("--- Финальные результаты ---")
        for ch_str, data in channels_data.items():
            ch = int(ch_str)
            rms = data.get('measured_rms', 0)
            peak = data.get('measured_peak', 0)
            crest = data.get('measured_crest', 0)
            adjustment = data.get('adjustment', 0)
            lufs = data.get('measured_lufs')
            
            logger.info(f"Канал {ch}:")
            logger.info(f"  RMS: {rms:.1f} dB, Peak: {peak:.1f} dB, Crest: {crest:.1f} dB")
            if lufs:
                logger.info(f"  LUFS: {lufs:.2f} LUFS")
            logger.info(f"  Корректировка: {adjustment:+.1f} dB")
    
    async def stop_gain_staging(self) -> None:
        """Остановка анализа Gain Staging"""
        logger.info("=== Остановка анализа Gain Staging ===")
        
        await self.send_message({"type": "stop_gain_staging"})
        
        # Ждем подтверждение
        response = await self.receive_message(timeout=10.0)
        
        if response and response.get("type") == "gain_staging_status":
            logger.info("✓ Анализ остановлен")
            self.test_data['end_time'] = datetime.now().isoformat()
        else:
            logger.warning("Не получен ответ о статусе остановки")
    
    async def get_final_status(self) -> Dict:
        """Получение финального статуса"""
        logger.info("=== Получение финального статуса ===")
        
        await self.send_message({"type": "get_gain_staging_status"})
        
        response = await self.receive_message(timeout=10.0)
        
        if response and response.get("type") == "gain_staging_status":
            status = response.get("status", {})
            
            # Сохраняем финальные данные
            measured_levels = status.get('measured_levels', {})
            calculated_adjustments = status.get('calculated_adjustments', {})
            
            self.test_data['final_measured_levels'] = measured_levels
            self.test_data['adjustments'] = calculated_adjustments
            
            return status
        else:
            logger.warning("Не получен финальный статус")
            return {}
    
    async def apply_adjustments(self) -> Dict:
        """Применение корректировок к пульту"""
        logger.info("=== Применение корректировок ===")
        
        await self.send_message({"type": "apply_gain_adjustments"})
        
        # Ждем результат
        response = await self.receive_message(timeout=10.0)
        
        if response:
            if response.get("type") == "gain_staging_applied":
                results = response.get("results", {})
                logger.info(f"✓ Корректировки применены для {len(results)} каналов")
                
                for mixer_ch, result in results.items():
                    if result.get('applied'):
                        logger.info(f"  Канал {mixer_ch}: TRIM {result.get('previous_trim', 0):.1f} -> "
                                  f"{result.get('new_trim', 0):.1f} dB")
                    else:
                        logger.warning(f"  Канал {mixer_ch}: {result.get('reason', 'Unknown error')}")
                
                self.test_data['applied_results'] = results
                return results
            else:
                logger.warning("Неожиданный тип ответа")
        
        return {}
    
    def save_results(self) -> str:
        """Сохранение результатов в JSON файл"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gain_staging_dante_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Подготовка данных для сохранения
        results_data = {
            'test_info': {
                'start_time': self.test_data['start_time'],
                'end_time': self.test_data['end_time'],
                'dante_device': self.test_data['dante_device'],
                'channels_tested': self.test_data['channels'],
                'channel_settings': self.test_data['channel_settings']
            },
            'measurements': self.test_data['measurements'],
            'final_results': {
                'measured_levels': self.test_data.get('final_measured_levels', {}),
                'adjustments': self.test_data.get('adjustments', {}),
                'applied_results': self.test_data.get('applied_results', {})
            },
            'errors': self.test_data['errors'],
            'warnings': self.test_data['warnings']
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Результаты сохранены: {filepath}")
        return filepath
    
    async def cleanup(self) -> None:
        """Очистка ресурсов"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass


async def main():
    """Главная функция теста"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Тест Gain Staging с Dante аудиоинтерфейсом через WebSocket'
    )
    parser.add_argument('--backend-url', type=str, default='ws://localhost:8765',
                       help='URL WebSocket сервера backend')
    parser.add_argument('--wing-ip', type=str, help='IP адрес пульта Wing (опционально)')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Длительность анализа в секундах')
    parser.add_argument('--auto-apply', action='store_true',
                       help='Автоматически применить корректировки')
    parser.add_argument('--results-dir', type=str, default='test_results',
                       help='Директория для сохранения результатов')
    
    args = parser.parse_args()
    
    test = DanteGainStagingTest(
        backend_url=args.backend_url,
        wing_ip=args.wing_ip,
        results_dir=args.results_dir
    )
    
    try:
        # 1. Подключение к backend
        if not await test.connect_to_backend():
            return 1
        
        # 2. Получение списка устройств
        devices = await test.get_audio_devices()
        if not devices:
            logger.error("Не найдено аудиоустройств")
            return 1
        
        # 3. Поиск Dante устройства
        dante_device = test.find_dante_device(devices)
        if not dante_device:
            logger.error("Dante устройство не найдено")
            return 1
        
        # 4. Выбор всех каналов
        channels = test.select_all_channels(dante_device)
        if not channels:
            logger.error("Не удалось выбрать каналы")
            return 1
        
        # 5. Создание настроек каналов
        channel_settings = test.create_channel_settings(channels)
        
        # 6. Подключение к Wing (если указан IP)
        if args.wing_ip:
            await test.connect_to_wing()
        
        # 7. Запуск анализа
        if not await test.start_gain_staging(dante_device, channels, channel_settings):
            logger.error("Не удалось запустить анализ")
            return 1
        
        # 8. Сбор данных
        await test.collect_measurements(duration=args.duration)
        
        # 9. Остановка анализа
        await test.stop_gain_staging()
        
        # 10. Получение финального статуса
        await test.get_final_status()
        
        # 11. Применение корректировок (если подключен Wing и указан флаг)
        if args.wing_ip and args.auto_apply:
            await test.apply_adjustments()
        
        # 12. Сохранение результатов
        results_file = test.save_results()
        
        logger.info("\n" + "="*60)
        logger.info("ТЕСТ ЗАВЕРШЕН УСПЕШНО")
        logger.info("="*60)
        logger.info(f"Результаты сохранены: {results_file}")
        logger.info(f"Измерений собрано: {len(test.test_data['measurements'])}")
        logger.info(f"Ошибок: {len(test.test_data['errors'])}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n\nТест прерван пользователем")
        await test.cleanup()
        return 1
    except Exception as e:
        logger.error(f"\n✗ Ошибка во время теста: {e}", exc_info=True)
        test.test_data['errors'].append({
            'time': datetime.now().isoformat(),
            'error': str(e)
        })
        test.save_results()
        await test.cleanup()
        return 1
    finally:
        await test.cleanup()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
