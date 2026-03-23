"""
Пример использования SafeGainCalibrator для умной калибровки gain

Демонстрация нового метода "Safe Static Gain":
1. Запуск анализа (learning phase)
2. Получение рекомендаций
3. Применение коррекций

Этот метод решает проблемы текущего realtime подхода:
- Фильтрует bleed (шум/фон) через noise gate
- Учитывает Crest Factor для разных типов инструментов
- Использует двойной лимитер (LUFS + TruePeak)
- Применяет коррекцию один раз, не создавая модуляции
"""

from lufs_gain_staging import SafeGainCalibrator
import numpy as np
import time

calibrator = SafeGainCalibrator(
    mixer_client=None,
    sample_rate=48000,
    config={
        'automation': {
            'safe_gain_calibration': {
                'target_lufs': -18.0,
                'max_peak_limit': -3.0,
                'noise_gate_threshold': -40.0,
                'min_signal_presence': 0.05,
                'learning_duration_sec': 15.0
            }
        }
    }
)

calibrator.add_channel(audio_channel=1, mixer_channel=1)
calibrator.add_channel(audio_channel=2, mixer_channel=2)
calibrator.add_channel(audio_channel=3, mixer_channel=3)

print("Starting analysis (learning phase)...")
calibrator.start_analysis()

kick_samples = np.random.randn(2048) * 0.3
kick_samples[::100] += 0.8

snare_samples = np.random.randn(2048) * 0.2
snare_samples[::150] += 0.7

vocal_samples = np.random.randn(2048) * 0.1

for i in range(int(calibrator.learning_duration * 100)):
    calibrator.process_audio(1, kick_samples)
    calibrator.process_audio(2, snare_samples) 
    calibrator.process_audio(3, vocal_samples)
    
    time.sleep(0.01)
    
    if i % 50 == 0:
        status = calibrator.get_status()
        print(f"Progress: {status['learning_progress']*100:.0f}%")

suggestions = calibrator.get_suggestions()

print("\n=== Gain Calibration Results ===")
for ch_id, suggestion in suggestions.items():
    print(f"\nChannel {ch_id}:")
    print(f"  Peak: {suggestion['peak_db']} dBTP")
    print(f"  LUFS: {suggestion['lufs']} dB")
    print(f"  Crest Factor: {suggestion['crest_factor_db']} dB")
    print(f"  Signal Presence: {suggestion['signal_presence']}%")
    print(f"  Suggested Gain: {suggestion['suggested_gain_db']:+.1f} dB")
    print(f"  Limited by: {suggestion['limited_by']}")

print("\nExample complete. In real usage, call calibrator.apply_corrections() to apply.")
