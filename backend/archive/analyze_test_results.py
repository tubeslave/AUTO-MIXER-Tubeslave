"""
Скрипт для анализа результатов тестирования Gain Staging

Анализирует сохраненные JSON файлы с результатами тестов и выводит статистику.
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
import statistics

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_results(filepath: str) -> Dict:
    """Загрузка результатов из JSON файла"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки файла {filepath}: {e}")
        return {}


def analyze_measurements(measurements: List[Dict]) -> Dict[str, Any]:
    """Анализ измерений"""
    if not measurements:
        return {}
    
    analysis = {
        'total_measurements': len(measurements),
        'duration_seconds': 0,
        'channels': {}
    }
    
    if measurements:
        first_time = measurements[0].get('elapsed_time', 0)
        last_time = measurements[-1].get('elapsed_time', 0)
        analysis['duration_seconds'] = round(last_time - first_time, 2)
    
    # Анализ по каналам
    channel_data = {}
    for measurement in measurements:
        channels = measurement.get('channels', {})
        for ch_str, data in channels.items():
            ch = int(ch_str) if isinstance(ch_str, str) else ch_str
            
            if ch not in channel_data:
                channel_data[ch] = {
                    'rms_values': [],
                    'peak_values': [],
                    'crest_values': [],
                    'lufs_values': [],
                    'adjustment_values': [],
                    'signal_present_count': 0,
                    'stabilized_count': 0,
                    'bleeding_detected_count': 0,
                    'total_samples': 0
                }
            
            if data.get('measured_rms') is not None:
                channel_data[ch]['rms_values'].append(data['measured_rms'])
            if data.get('measured_peak') is not None:
                channel_data[ch]['peak_values'].append(data['measured_peak'])
            if data.get('measured_crest') is not None:
                channel_data[ch]['crest_values'].append(data['measured_crest'])
            if data.get('measured_lufs') is not None:
                channel_data[ch]['lufs_values'].append(data['measured_lufs'])
            if data.get('adjustment') is not None:
                channel_data[ch]['adjustment_values'].append(data['adjustment'])
            
            if data.get('signal_present'):
                channel_data[ch]['signal_present_count'] += 1
            if data.get('stabilized'):
                channel_data[ch]['stabilized_count'] += 1
            if data.get('bleeding_detected'):
                channel_data[ch]['bleeding_detected_count'] += 1
            
            total_samples = data.get('total_samples', 0)
            if total_samples > channel_data[ch]['total_samples']:
                channel_data[ch]['total_samples'] = total_samples
    
    # Расчет статистики для каждого канала
    for ch, data in channel_data.items():
        stats = {
            'rms': {},
            'peak': {},
            'crest': {},
            'lufs': {},
            'adjustment': {},
            'signal_present_ratio': data['signal_present_count'] / len(measurements) if measurements else 0,
            'stabilized_ratio': data['stabilized_count'] / len(measurements) if measurements else 0,
            'bleeding_detected_ratio': data['bleeding_detected_count'] / len(measurements) if measurements else 0,
            'max_samples': data['total_samples']
        }
        
        if data['rms_values']:
            stats['rms'] = {
                'mean': round(statistics.mean(data['rms_values']), 2),
                'median': round(statistics.median(data['rms_values']), 2),
                'min': round(min(data['rms_values']), 2),
                'max': round(max(data['rms_values']), 2),
                'stdev': round(statistics.stdev(data['rms_values']), 2) if len(data['rms_values']) > 1 else 0
            }
        
        if data['peak_values']:
            stats['peak'] = {
                'mean': round(statistics.mean(data['peak_values']), 2),
                'median': round(statistics.median(data['peak_values']), 2),
                'min': round(min(data['peak_values']), 2),
                'max': round(max(data['peak_values']), 2)
            }
        
        if data['crest_values']:
            stats['crest'] = {
                'mean': round(statistics.mean(data['crest_values']), 2),
                'median': round(statistics.median(data['crest_values']), 2)
            }
        
        if data['lufs_values']:
            stats['lufs'] = {
                'mean': round(statistics.mean(data['lufs_values']), 2),
                'median': round(statistics.median(data['lufs_values']), 2),
                'min': round(min(data['lufs_values']), 2),
                'max': round(max(data['lufs_values']), 2)
            }
        
        if data['adjustment_values']:
            stats['adjustment'] = {
                'mean': round(statistics.mean(data['adjustment_values']), 2),
                'median': round(statistics.median(data['adjustment_values']), 2),
                'min': round(min(data['adjustment_values']), 2),
                'max': round(max(data['adjustment_values']), 2)
            }
        
        analysis['channels'][ch] = stats
    
    return analysis


def print_analysis(results: Dict) -> None:
    """Вывод анализа результатов"""
    test_info = results.get('test_info', {})
    measurements = results.get('measurements', [])
    final_results = results.get('final_results', {})
    errors = results.get('errors', [])
    
    print("\n" + "="*70)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ GAIN STAGING")
    print("="*70)
    
    # Информация о тесте
    print("\n--- Информация о тесте ---")
    print(f"Начало: {test_info.get('start_time', 'N/A')}")
    print(f"Конец: {test_info.get('end_time', 'N/A')}")
    
    dante_device = test_info.get('dante_device', {})
    print(f"Устройство: {dante_device.get('name', 'N/A')} (ID: {dante_device.get('id', 'N/A')})")
    print(f"Каналов протестировано: {len(test_info.get('channels_tested', []))}")
    
    # Анализ измерений
    if measurements:
        analysis = analyze_measurements(measurements)
        
        print(f"\n--- Статистика измерений ---")
        print(f"Всего измерений: {analysis.get('total_measurements', 0)}")
        print(f"Длительность: {analysis.get('duration_seconds', 0)} секунд")
        
        print(f"\n--- Анализ по каналам ---")
        for ch in sorted(analysis.get('channels', {}).keys()):
            ch_stats = analysis['channels'][ch]
            print(f"\nКанал {ch}:")
            
            if ch_stats.get('rms'):
                rms = ch_stats['rms']
                print(f"  RMS: среднее={rms['mean']:.1f} dB, "
                     f"мин={rms['min']:.1f} dB, макс={rms['max']:.1f} dB, "
                     f"σ={rms['stdev']:.2f} dB")
            
            if ch_stats.get('peak'):
                peak = ch_stats['peak']
                print(f"  Peak: среднее={peak['mean']:.1f} dB, "
                     f"мин={peak['min']:.1f} dB, макс={peak['max']:.1f} dB")
            
            if ch_stats.get('crest'):
                crest = ch_stats['crest']
                print(f"  Crest Factor: среднее={crest['mean']:.1f} dB")
            
            if ch_stats.get('lufs'):
                lufs = ch_stats['lufs']
                print(f"  LUFS: среднее={lufs['mean']:.2f} LUFS, "
                     f"мин={lufs['min']:.2f} LUFS, макс={lufs['max']:.2f} LUFS")
            
            if ch_stats.get('adjustment'):
                adj = ch_stats['adjustment']
                print(f"  Корректировка: среднее={adj['mean']:+.1f} dB, "
                     f"мин={adj['min']:+.1f} dB, макс={adj['max']:+.1f} dB")
            
            print(f"  Signal present: {ch_stats['signal_present_ratio']*100:.1f}%")
            print(f"  Stabilized: {ch_stats['stabilized_ratio']*100:.1f}%")
            if ch_stats['bleeding_detected_ratio'] > 0:
                print(f"  ⚠ Bleeding detected: {ch_stats['bleeding_detected_ratio']*100:.1f}%")
            print(f"  Макс. samples: {ch_stats['max_samples']}")
    
    # Финальные результаты
    if final_results:
        print(f"\n--- Финальные результаты ---")
        
        measured_levels = final_results.get('measured_levels', {})
        adjustments = final_results.get('adjustments', {})
        applied_results = final_results.get('applied_results', {})
        
        if measured_levels:
            print(f"\nИзмеренные уровни (финальные):")
            for ch_str, data in measured_levels.items():
                ch = int(ch_str) if isinstance(ch_str, str) else ch_str
                rms = data.get('rms', 0)
                peak = data.get('peak', 0)
                crest = data.get('crest_factor', 0)
                lufs = data.get('integrated_lufs')
                
                print(f"  Канал {ch}: RMS={rms:.1f} dB, Peak={peak:.1f} dB, Crest={crest:.1f} dB", end='')
                if lufs:
                    print(f", LUFS={lufs:.2f} LUFS")
                else:
                    print()
        
        if adjustments:
            print(f"\nРассчитанные корректировки:")
            for mixer_ch, adjustment in adjustments.items():
                print(f"  Канал {mixer_ch}: {adjustment:+.1f} dB")
        
        if applied_results:
            print(f"\nПримененные корректировки:")
            applied_count = sum(1 for r in applied_results.values() if r.get('applied'))
            print(f"  Успешно применено: {applied_count}/{len(applied_results)}")
            for mixer_ch, result in applied_results.items():
                if result.get('applied'):
                    print(f"  Канал {mixer_ch}: TRIM {result.get('previous_trim', 0):.1f} -> "
                         f"{result.get('new_trim', 0):.1f} dB")
                else:
                    print(f"  Канал {mixer_ch}: ✗ {result.get('reason', 'Unknown error')}")
    
    # Ошибки
    if errors:
        print(f"\n--- Ошибки ({len(errors)}) ---")
        for error in errors:
            print(f"  [{error.get('time', 'N/A')}] {error.get('error', 'Unknown error')}")
    else:
        print(f"\n--- Ошибки: нет ---")
    
    print("\n" + "="*70)


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ результатов тестирования Gain Staging')
    parser.add_argument('file', type=str, help='Путь к JSON файлу с результатами')
    
    args = parser.parse_args()
    
    if not Path(args.file).exists():
        logger.error(f"Файл не найден: {args.file}")
        return 1
    
    results = load_results(args.file)
    
    if not results:
        logger.error("Не удалось загрузить результаты")
        return 1
    
    print_analysis(results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
