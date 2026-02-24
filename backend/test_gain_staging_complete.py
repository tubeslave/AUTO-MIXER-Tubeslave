#!/usr/bin/env python3
"""
Полный набор тестов для Gain Staging

Запуск:
    cd /Users/dmitrijvolkov/AUTO\ MIXER\ Tubeslave/backend
    python -m pytest test_gain_staging_complete.py -v
    
Или напрямую:
    python test_gain_staging_complete.py
"""

import unittest
import numpy as np
import time
import threading
from collections import deque
import sys
import os

# Добавляем путь к бэкенду
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gain_staging_fixes import (
    SignalStats, AGCEnvelope, OSCRateLimiter, 
    ThreadSafeAudioBuffer, AnalysisState
)


class TestSignalStats(unittest.TestCase):
    """Тесты для SignalStats."""
    
    def test_initial_state(self):
        """Проверка начального состояния."""
        stats = SignalStats(channel_id=1)
        self.assertEqual(stats.channel_id, 1)
        self.assertEqual(stats.max_true_peak_db, -100.0)
        self.assertEqual(stats.integrated_lufs, -100.0)
        self.assertEqual(stats.suggested_gain_db, 0.0)
    
    def test_buffer_size_limit(self):
        """Проверка ограничения размера буфера."""
        stats = SignalStats(channel_id=1, _max_buffer_size=100)
        
        # Добавляем больше элементов чем размер буфера
        for i in range(200):
            stats.update_sample(-10.0, -20.0, -25.0)
        
        # Буфер должен содержать только последние 100
        self.assertEqual(len(stats.rms_values), 100)
        self.assertEqual(len(stats.peak_values), 100)
    
    def test_division_by_zero_protection(self):
        """Проверка защиты от деления на ноль."""
        stats = SignalStats(channel_id=1)
        
        # Не должно быть ошибки при min_signal_presence = 0
        stats.calculate_safe_gain(min_signal_presence=0)
        
        # Не должно быть ошибки при min_signal_presence < 0
        stats.calculate_safe_gain(min_signal_presence=-1)
    
    def test_crest_factor_calculation(self):
        """Проверка расчета crest factor."""
        stats = SignalStats(channel_id=1)
        
        # Добавляем тестовые данные
        for _ in range(100):
            stats.update_sample(-6.0, -20.0, -25.0)  # Peak -6, LUFS -20
        
        stats.calculate_integrated_lufs()
        stats.calculate_crest_factor()
        
        # Crest factor должен быть примерно 14 dB (-6 - (-20))
        self.assertGreater(stats.crest_factor_db, 10.0)
        self.assertLess(stats.crest_factor_db, 20.0)
    
    def test_safe_gain_calculation(self):
        """Проверка расчета safe gain."""
        stats = SignalStats(channel_id=1)
        
        # Добавляем данные с пиком -6 dB и LUFS -20
        for _ in range(100):
            stats.update_sample(-6.0, -20.0, -25.0)
        
        stats.calculate_safe_gain(
            target_lufs=-18.0,
            max_peak_limit=-3.0,
            min_signal_presence=0.05
        )
        
        # Gain должен быть положительным (нужно усилить с -20 до -18)
        self.assertGreater(stats.suggested_gain_db, 0)
        
        # Но не должен превысить лимит по пику (-6 + gain <= -3)
        self.assertLessEqual(stats.suggested_gain_db, 3.0)
    
    def test_silent_channel_detection(self):
        """Проверка обнаружения тихого канала."""
        stats = SignalStats(channel_id=1)
        
        # Добавляем очень тихие данные
        for _ in range(100):
            stats.update_sample(-80.0, -80.0, -80.0)
        
        stats.calculate_safe_gain(min_signal_presence=0.1)
        
        # Для тихого канала gain должен быть 0
        self.assertEqual(stats.suggested_gain_db, 0.0)
        self.assertEqual(stats.gain_limited_by, "silent_channel")
    
    def test_get_report_types(self):
        """Проверка типов в отчете."""
        stats = SignalStats(channel_id=1)
        stats.update_sample(-10.0, -20.0, -25.0)
        
        report = stats.get_report()
        
        # Все значения должны быть Python types (не numpy)
        self.assertIsInstance(report['channel'], int)
        self.assertIsInstance(report['peak_db'], float)
        self.assertIsInstance(report['lufs'], float)
        self.assertIsInstance(report['signal_presence'], float)
        self.assertIsInstance(report['limited_by'], str)


class TestAGCEnvelope(unittest.TestCase):
    """Тесты для AGCEnvelope."""
    
    def test_initial_state(self):
        """Проверка начального состояния."""
        env = AGCEnvelope()
        self.assertEqual(env.get_current_gain(), 0.0)
        self.assertFalse(env._is_holding)
    
    def test_attack_behavior(self):
        """Проверка поведения attack (снижение gain)."""
        env = AGCEnvelope(attack_ms=10.0, release_ms=100.0, hold_ms=0.0)
        env.reset(initial_gain=0.0)
        
        # Целевой gain меньше текущего - должен сработать attack
        result = env.process(-10.0)
        
        # Gain должен уменьшиться
        self.assertLess(result, 0.0)
        self.assertGreater(result, -10.0)  # Но не сразу достигнуть цели
    
    def test_release_behavior(self):
        """Проверка поведения release (увеличение gain)."""
        env = AGCEnvelope(attack_ms=10.0, release_ms=100.0, hold_ms=0.0)
        env.reset(initial_gain=-10.0)
        
        # Целевой gain больше текущего - должен сработать release
        result = env.process(0.0)
        
        # Gain должен увеличиться
        self.assertGreater(result, -10.0)
        self.assertLess(result, 0.0)  # Но не сразу достигнуть цели
    
    def test_hold_behavior(self):
        """Проверка поведения hold."""
        env = AGCEnvelope(
            attack_ms=10.0, 
            release_ms=100.0, 
            hold_ms=50.0,
            update_interval_ms=10.0
        )
        env.reset(initial_gain=0.0)
        
        # Attack
        env.process(-10.0)
        gain_after_attack = env.get_current_gain()
        
        # Hold должен активироваться
        self.assertTrue(env._is_holding)
        self.assertEqual(env._hold_counter, 5)  # 50ms / 10ms
        
        # Во время hold gain не должен меняться
        for _ in range(3):
            env.process(-10.0)
        
        self.assertEqual(env.get_current_gain(), gain_after_attack)
    
    def test_attack_release_sequence(self):
        """Проверка полной последовательности attack -> hold -> release."""
        env = AGCEnvelope(
            attack_ms=10.0,
            release_ms=100.0,
            hold_ms=30.0,
            update_interval_ms=10.0
        )
        env.reset(initial_gain=0.0)
        
        gains = []
        
        # Attack phase - gain должен снизиться
        for _ in range(5):
            gains.append(env.process(-10.0))
        
        # Проверяем что gain снизился после attack
        self.assertLess(gains[-1], gains[0])
        
        # Hold phase (3 итерации) - gain не должен меняться
        hold_start = env.get_current_gain()
        for _ in range(3):
            gains.append(env.process(-10.0))
        self.assertEqual(env.get_current_gain(), hold_start)
        
        # Release phase - gain должен вернуться к 0
        for _ in range(20):
            gains.append(env.process(0.0))
        
        # Проверяем что gain восстановился ближе к 0
        self.assertGreater(gains[-1], gains[5])  # После release больше чем после attack
    
    def test_time_constants_validation(self):
        """Проверка валидации временных констант."""
        env = AGCEnvelope()
        
        # Не должно быть ошибок с нулевыми или отрицательными значениями
        env.set_times(attack_ms=0, release_ms=0, hold_ms=-10)
        
        # Значения должны быть скорректированы
        self.assertGreater(env.attack_ms, 0)
        self.assertGreater(env.release_ms, 0)
        self.assertEqual(env.hold_ms, 0)


class TestOSCRateLimiter(unittest.TestCase):
    """Тесты для OSC Rate Limiter."""
    
    def test_deadband_filtering(self):
        """Проверка deadband фильтрации."""
        limiter = OSCRateLimiter(deadband_db=1.0, normal_rate_hz=1000.0)
        
        # Первое сообщение должно пройти
        self.assertTrue(limiter.should_send(1, 0.0))
        
        # Ждем немного чтобы сбросить rate limiting
        time.sleep(0.01)
        
        # Изменение меньше deadband - не должно пройти
        self.assertFalse(limiter.should_send(1, 0.5))
        
        # Ждем еще
        time.sleep(0.01)
        
        # Изменение больше deadband - должно пройти
        self.assertTrue(limiter.should_send(1, 2.0))
    
    def test_rate_limiting(self):
        """Проверка rate limiting."""
        limiter = OSCRateLimiter(normal_rate_hz=10.0)  # 10 msg/s = 100ms interval
        
        # Первое сообщение
        self.assertTrue(limiter.should_send(1, 0.0))
        
        # Следующее сразу - не должно пройти
        self.assertFalse(limiter.should_send(1, 5.0))
        
        # Ждем 150ms
        time.sleep(0.15)
        
        # Теперь должно пройти
        self.assertTrue(limiter.should_send(1, 5.0))
    
    def test_emergency_rate(self):
        """Проверка emergency rate."""
        limiter = OSCRateLimiter(
            normal_rate_hz=10.0,
            emergency_rate_hz=1000.0  # Высокий emergency rate
        )
        
        # Обычное сообщение
        self.assertTrue(limiter.should_send(1, 0.0, is_emergency=False))
        
        # Ждем немного
        time.sleep(0.01)
        
        # Emergency - должно пройти из-за высокого emergency rate
        self.assertTrue(limiter.should_send(1, -5.0, is_emergency=True))
    
    def test_per_channel_tracking(self):
        """Проверка отслеживания по каналам."""
        limiter = OSCRateLimiter(deadband_db=1.0)
        
        # Канал 1
        self.assertTrue(limiter.should_send(1, 0.0))
        
        # Канал 2 (другой канал) - должен пройти независимо
        self.assertTrue(limiter.should_send(2, 0.0))
        
        # Снова канал 1 с маленьким изменением
        self.assertFalse(limiter.should_send(1, 0.5))


class TestThreadSafeAudioBuffer(unittest.TestCase):
    """Тесты для ThreadSafeAudioBuffer."""
    
    def test_basic_operations(self):
        """Проверка базовых операций."""
        buf = ThreadSafeAudioBuffer(max_chunks=5)
        
        # Добавляем данные
        data = np.array([1.0, 2.0, 3.0])
        buf.append(data)
        
        # Получаем данные
        result = buf.get_and_clear()
        
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, data)
        
        # Буфер должен быть пустым
        self.assertIsNone(buf.get_and_clear())
    
    def test_max_chunks_limit(self):
        """Проверка ограничения количества chunks."""
        buf = ThreadSafeAudioBuffer(max_chunks=3)
        
        # Добавляем больше чем max_chunks
        for i in range(5):
            buf.append(np.array([float(i)]))
        
        # Должно остаться только 3
        result = buf.get_and_clear()
        self.assertEqual(len(result), 3)
        
        # Должны быть последние 3 значения
        np.testing.assert_array_equal(result, np.array([2.0, 3.0, 4.0]))
    
    def test_thread_safety(self):
        """Проверка thread-safety."""
        buf = ThreadSafeAudioBuffer(max_chunks=100)
        errors = []
        
        def writer():
            try:
                for i in range(100):
                    buf.append(np.array([float(i)]))
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        def reader():
            try:
                for _ in range(50):
                    buf.get_and_clear()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)
        
        # Запускаем потоки
        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # Не должно быть ошибок
        self.assertEqual(len(errors), 0)


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты."""
    
    def test_full_gain_staging_pipeline(self):
        """Проверка полного пайплайна gain staging."""
        # 1. Собираем статистику
        stats = SignalStats(channel_id=1)
        
        # Генерируем тестовый сигнал
        np.random.seed(42)
        for _ in range(1000):
            peak = -10.0 + np.random.randn() * 2
            lufs = -22.0 + np.random.randn() * 3
            stats.update_sample(peak, lufs, lufs - 5)
        
        # 2. Рассчитываем gain
        stats.calculate_safe_gain(
            target_lufs=-18.0,
            max_peak_limit=-6.0,
            min_signal_presence=0.05
        )
        
        # 3. Применяем envelope
        env = AGCEnvelope(
            attack_ms=50.0,
            release_ms=500.0,
            hold_ms=200.0,
            update_interval_ms=100.0
        )
        
        target_gain = stats.suggested_gain_db
        applied_gains = []
        
        for _ in range(20):
            applied_gains.append(env.process(target_gain))
        
        # Проверяем что gain применился плавно
        self.assertNotEqual(applied_gains[0], applied_gains[-1])
        
        # 4. Проверяем rate limiting
        limiter = OSCRateLimiter(deadband_db=0.5, normal_rate_hz=10.0)
        
        send_count = 0
        for gain in applied_gains:
            if limiter.should_send(1, gain):
                send_count += 1
        
        # Не все сообщения должны были пройти из-за deadband
        self.assertLess(send_count, len(applied_gains))
    
    def test_emergency_scenario(self):
        """Проверка сценария emergency (клиппинг)."""
        stats = SignalStats(channel_id=1)
        
        # Симулируем клиппинг (пик близко к 0 dB)
        for _ in range(100):
            stats.update_sample(-0.5, -10.0, -15.0)
        
        stats.calculate_safe_gain(
            target_lufs=-18.0,
            max_peak_limit=-3.0,
            min_signal_presence=0.05
        )
        
        # Gain должен быть отрицательным (нужно снизить)
        self.assertLess(stats.suggested_gain_db, 0)
        
        # Ограничен по пику (-0.5 + gain <= -3)
        self.assertGreaterEqual(stats.suggested_gain_db, -2.5)


class TestPerformance(unittest.TestCase):
    """Тесты производительности."""
    
    def test_envelope_performance(self):
        """Проверка производительности envelope."""
        env = AGCEnvelope()
        
        start = time.time()
        
        for _ in range(10000):
            env.process(-5.0)
        
        elapsed = time.time() - start
        
        # Должно выполняться быстрее 1 секунды
        self.assertLess(elapsed, 1.0)
    
    def test_rate_limiter_performance(self):
        """Проверка производительности rate limiter."""
        limiter = OSCRateLimiter()
        
        start = time.time()
        
        for i in range(10000):
            limiter.should_send(i % 48, float(i % 20))
        
        elapsed = time.time() - start
        
        # Должно выполняться быстрее 1 секунды
        self.assertLess(elapsed, 1.0)


def run_tests():
    """Запуск всех тестов."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавляем все тесты
    suite.addTests(loader.loadTestsFromTestCase(TestSignalStats))
    suite.addTests(loader.loadTestsFromTestCase(TestAGCEnvelope))
    suite.addTests(loader.loadTestsFromTestCase(TestOSCRateLimiter))
    suite.addTests(loader.loadTestsFromTestCase(TestThreadSafeAudioBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Запускаем
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
