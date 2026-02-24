"""
Phase and Delay Alignment Module

Анализ, вычисление и коррекция фаз и задержек на каналах микшера.
Использует различные методы измерения:
- GCC-PHAT (Generalized Cross-Correlation)
- Phase Angle Difference
- Impulse Response Analysis
- Magnitude Squared Coherence
- Group Delay Calculation
- Hilbert Transform
- LMS Adaptive Filtering
"""

import numpy as np
import threading
import logging
import time
from typing import Dict, List, Callable, Optional, Tuple
from collections import deque
from scipy import signal
from scipy.signal import hilbert, coherence, find_peaks

logger = logging.getLogger(__name__)


class PhaseAlignmentAnalyzer:
    """
    Анализатор фаз и задержек между каналами.
    
    Использует различные методы для измерения временных задержек и фазовых сдвигов
    между парами каналов.
    """
    
    def __init__(self,
                 device_index: int = None,
                 sample_rate: int = 48000,
                 chunk_size: int = 4096,
                 analysis_window: float = 2.0,
                 max_delay_ms: float = 10.0):
        """
        Инициализация анализатора фаз и задержек.
        
        Args:
            device_index: Индекс аудио устройства PyAudio
            sample_rate: Частота дискретизации в Hz
            chunk_size: Размер буфера аудио данных
            analysis_window: Временное окно для анализа (секунды)
            max_delay_ms: Максимальная задержка для поиска (миллисекунды)
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.analysis_window = analysis_window
        self.max_delay_ms = max_delay_ms
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000.0)
        
        # Буферы аудио данных по каналам {channel_id: deque of chunks}
        self.audio_buffers: Dict[int, deque] = {}
        
        # Результаты измерений {channel_pair: {delay_ms, phase_diff, coherence, ...}}
        self.measurements: Dict[Tuple[int, int], Dict[str, float]] = {}
        
        # PyAudio stream
        self.stream = None
        self.pa = None
        self.is_running = False
        
        # Threading
        self._stop_event = threading.Event()
        self._analysis_thread = None
        
        # Callbacks
        self.on_measurement_updated: Optional[Callable[[Dict], None]] = None
        
    def start(self, 
              reference_channel: int,
              channels: List[int],
              on_measurement_callback: Callable = None):
        """
        Запуск анализа фаз и задержек.
        
        Args:
            reference_channel: Референсный канал (базовый для сравнения)
            channels: Список каналов для анализа относительно reference_channel
            on_measurement_callback: Callback функция для обновления результатов
        """
        if self.is_running:
            logger.warning("Phase alignment analyzer already running")
            return False
        
        self.on_measurement_updated = on_measurement_callback
        
        # Сохраняем reference channel для использования в измерениях
        self.reference_channel = reference_channel
        
        # Инициализация буферов
        all_channels = [reference_channel] + channels
        chunks_per_window = int((self.sample_rate * self.analysis_window) / self.chunk_size)
        
        for ch in all_channels:
            self.audio_buffers[ch] = deque(maxlen=chunks_per_window)
        
        logger.info(f"=== PHASE ALIGNMENT: Starting analysis ===")
        logger.info(f"Reference channel: {reference_channel}")
        logger.info(f"Channels to align: {channels}")
        logger.info(f"Device index: {self.device_index}")
        logger.info(f"Max delay: {self.max_delay_ms} ms")
        
        try:
            import pyaudio
            self.pyaudio = pyaudio
            
            self.pa = pyaudio.PyAudio()
            
            if self.device_index is not None:
                device_info = self.pa.get_device_info_by_index(int(self.device_index))
                max_channels = int(device_info.get('maxInputChannels', 2))
                device_sample_rate = int(device_info.get('defaultSampleRate', 48000))
                logger.info(f"Selected device {self.device_index}: {device_info.get('name')}")
            else:
                device_info = self.pa.get_default_input_device_info()
                max_channels = int(device_info.get('maxInputChannels', 2))
                device_sample_rate = int(device_info.get('defaultSampleRate', 48000))
                logger.info(f"Using default device: {device_info.get('name')}")
            
            self.sample_rate = device_sample_rate
            # Для phase alignment нужно захватывать все каналы одновременно
            # Используем максимальное количество каналов из доступных
            required_channels = max(all_channels) if all_channels else 2
            num_channels = min(required_channels, max_channels)
            
            # Сохранение для callback
            self._num_channels = num_channels
            self._channel_ids = sorted(all_channels)
            
            logger.info(f"Opening stream: {num_channels} channels, {self.sample_rate} Hz")
            
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=num_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=int(self.device_index) if self.device_index is not None else None,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            self.is_running = True
            self._stop_event.clear()
            
            # Запуск потока анализа
            self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self._analysis_thread.start()
            
            logger.info("Phase alignment analyzer started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start phase alignment analyzer: {e}", exc_info=True)
            self.is_running = False
            return False
    
    def stop(self):
        """Остановка анализатора."""
        if not self.is_running:
            return
        
        logger.info("Stopping phase alignment analyzer...")
        self._stop_event.set()
        self.is_running = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        
        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass
        
        if self._analysis_thread:
            self._analysis_thread.join(timeout=2.0)
        
        logger.info("Phase alignment analyzer stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback для получения аудио данных."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        try:
            # Конвертация в numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Разделение на каналы
            # PyAudio возвращает interleaved данные: [ch0_sample0, ch1_sample0, ch0_sample1, ch1_sample1, ...]
            if hasattr(self, '_num_channels') and hasattr(self, '_channel_ids'):
                num_channels = self._num_channels
                channel_ids = self._channel_ids
                
                if num_channels > 0 and len(channel_ids) > 0:
                    # Reshape для разделения каналов
                    samples_per_channel = len(audio_data) // num_channels
                    if samples_per_channel > 0 and len(audio_data) % num_channels == 0:
                        audio_reshaped = audio_data.reshape(samples_per_channel, num_channels)
                        
                        # Map channels to buffer indices
                        for i, ch in enumerate(channel_ids):
                            if i < num_channels and ch in self.audio_buffers:
                                channel_data = audio_reshaped[:, i]
                                self.audio_buffers[ch].append(channel_data)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
        
        return (None, self.pyaudio.paContinue)
    
    def _analysis_loop(self):
        """Основной цикл анализа."""
        while not self._stop_event.is_set() and self.is_running:
            try:
                # Проверка наличия достаточного количества данных
                if not self.audio_buffers:
                    time.sleep(0.1)
                    continue
                
                # Получение данных из буферов
                channel_data = {}
                min_length = float('inf')
                
                for ch, buffer in self.audio_buffers.items():
                    if len(buffer) == 0:
                        continue
                    data = np.concatenate(list(buffer))
                    channel_data[ch] = data
                    min_length = min(min_length, len(data))
                
                if min_length < self.chunk_size:
                    time.sleep(0.1)
                    continue
                
                # Обрезка до одинаковой длины
                for ch in channel_data:
                    channel_data[ch] = channel_data[ch][:int(min_length)]
                
                # Выполнение измерений
                self._perform_measurements(channel_data)
                
                # Вызов callback
                if self.on_measurement_updated:
                    self.on_measurement_updated(self.measurements.copy())
                
                time.sleep(0.1)  # Обновление каждые 100ms
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}", exc_info=True)
                time.sleep(0.1)
    
    def _perform_measurements(self, channel_data: Dict[int, np.ndarray]):
        """Выполнение всех измерений между каналами."""
        if len(channel_data) < 2:
            return
        
        # Используем сохраненный reference channel, а не минимальный
        if not hasattr(self, 'reference_channel') or self.reference_channel not in channel_data:
            # Fallback: используем минимальный канал если reference не задан
            reference_ch = min(channel_data.keys())
            logger.warning(f"Reference channel not set, using minimum channel: {reference_ch}")
        else:
            reference_ch = self.reference_channel
        
        ref_signal = channel_data[reference_ch]
        
        # Измерения для каждого канала относительно reference
        # Включаем ВСЕ каналы, не только те что после reference
        for ch, signal_data in channel_data.items():
            if ch == reference_ch:
                continue
            
            pair = (reference_ch, ch)
            
            # Выполнение различных методов измерения
            measurements = {}
            
            # 1. GCC-PHAT
            delay_gcc, gcc_peak_value = self._gcc_phat(ref_signal, signal_data)
            measurements['delay_gcc_ms'] = delay_gcc * 1000.0 / self.sample_rate
            measurements['gcc_peak_value'] = float(gcc_peak_value)
            measurements['coherence_gcc'] = self._estimate_coherence(ref_signal, signal_data)
            
            # 2. Phase Angle Difference
            phase_diff = self._phase_angle_difference(ref_signal, signal_data)
            measurements['phase_diff_deg'] = np.degrees(phase_diff)
            
            # 3. Impulse Response Analysis
            delay_ir = self._impulse_response_delay(ref_signal, signal_data)
            measurements['delay_ir_ms'] = delay_ir * 1000.0 / self.sample_rate
            
            # 4. Magnitude Squared Coherence
            coherence_val, freq_range = self._magnitude_coherence(ref_signal, signal_data)
            measurements['coherence'] = coherence_val
            measurements['coherence_freq_range'] = freq_range
            
            # 5. Group Delay
            group_delay = self._group_delay(ref_signal, signal_data)
            measurements['group_delay_ms'] = group_delay * 1000.0 / self.sample_rate
            
            # 6. Hilbert Transform (instantaneous phase)
            inst_phase_diff = self._hilbert_phase_diff(ref_signal, signal_data)
            measurements['inst_phase_diff_deg'] = np.degrees(inst_phase_diff)
            
            # 7. LMS Adaptive Filtering
            delay_lms = self._lms_delay(ref_signal, signal_data)
            measurements['delay_lms_ms'] = delay_lms * 1000.0 / self.sample_rate
            
            # GCC-PHAT priority with IR fallback when coherence is low.
            if measurements.get('coherence', 0.0) > 0.5:
                measurements['optimal_delay_ms'] = measurements['delay_gcc_ms']
            else:
                measurements['optimal_delay_ms'] = measurements['delay_ir_ms']
            
            # Определение необходимости инверсии полярности по знаку GCC пика
            measurements['phase_invert'] = 1 if gcc_peak_value < 0 else 0
            
            self.measurements[pair] = measurements
    
    def _gcc_phat(self, ref_signal: np.ndarray, signal: np.ndarray) -> Tuple[int, float]:
        """
        Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
        
        Finds the MINIMUM delay that achieves good correlation (>= 70% of max).
        This prioritizes minimal latency while maintaining phase alignment.

        IMP Eq. 8.1: ψ[k] = X1*[k] · X2[k]
        IMP Eq. 8.2: ψP[k] = X1*[k] · X2[k] / |X1*[k] · X2[k]|
        IMP Eq. 8.3: τ = argmin_n { |n| : F^{-1}{ψP[k]}[n] >= 0.7 * max }

        Positive τ means signal X2 is delayed relative to reference X1.
        """
        # FFT of reference (X1) and signal (X2)
        X1 = np.fft.fft(ref_signal)
        X2 = np.fft.fft(signal)

        # Cross-power spectrum (IMP Eq. 8.1): ψ[k] = X1*[k] · X2[k]
        psi = np.conj(X1) * X2

        # PHAT weighting (IMP Eq. 8.2): ψP[k] = ψ[k] / |ψ[k]|
        magnitude = np.abs(psi)
        psi_phat = psi / (magnitude + 1e-10)

        # Inverse FFT (IMP Eq. 8.3): τ = argmax F^{-1}{ψP}
        gcc = np.real(np.fft.ifft(psi_phat))

        # Search for peak within allowed delay range
        search_range = min(self.max_delay_samples, len(gcc) // 2)
        search_start = len(gcc) // 2 - search_range
        search_end = len(gcc) // 2 + search_range

        search_region = gcc[search_start:search_end]
        abs_region = np.abs(search_region)
        max_peak = np.max(abs_region)
        
        # Find minimum delay with good correlation (>= 70% of max peak)
        # This prioritizes minimal latency while maintaining phase alignment
        threshold = 0.7 * max_peak
        good_peaks_mask = abs_region >= threshold
        
        if np.any(good_peaks_mask):
            # Get indices of all good peaks
            good_indices = np.where(good_peaks_mask)[0]
            # Calculate delays for these peaks
            delays = good_indices + search_start - len(gcc) // 2
            # Find the index with minimum absolute delay
            min_delay_idx_in_subset = np.argmin(np.abs(delays))
            peak_idx = good_indices[min_delay_idx_in_subset]
        else:
            # Fallback: use maximum peak if no peaks meet threshold
            peak_idx = np.argmax(abs_region)
        
        delay_samples = peak_idx + search_start - len(gcc) // 2
        peak_value = float(search_region[peak_idx])

        return delay_samples, peak_value

    def _estimate_coherence(self, ref_signal: np.ndarray, signal: np.ndarray) -> float:
        """Standard magnitude-squared coherence (Welch method)."""
        _, coh = coherence(
            ref_signal,
            signal,
            fs=self.sample_rate,
            nperseg=min(len(ref_signal), 2048),
        )
        return float(np.mean(coh))
    
    def _phase_angle_difference(self, ref_signal: np.ndarray, signal: np.ndarray) -> float:
        """
        Phase Angle Difference.
        
        Расчет разности фаз через аргумент комплексного спектра кросс-корреляции.
        """
        # FFT
        ref_fft = np.fft.fft(ref_signal)
        sig_fft = np.fft.fft(signal)
        
        # Cross-power spectrum
        cross_power = ref_fft * np.conj(sig_fft)
        
        # Средний фазовый угол
        phase_angles = np.angle(cross_power)
        mean_phase = np.angle(np.mean(np.exp(1j * phase_angles)))
        
        return mean_phase
    
    def _impulse_response_delay(self, ref_signal: np.ndarray, signal: np.ndarray) -> int:
        """
        Impulse Response Analysis.
        
        Сравнение времени прихода первых пиков (транзиентов) в двух сигналах.
        """
        # Cross-correlation
        correlation = np.correlate(ref_signal, signal, mode='full')
        
        # Поиск первого значимого пика
        threshold = np.max(np.abs(correlation)) * 0.3
        peaks, _ = find_peaks(np.abs(correlation), height=threshold)
        
        if len(peaks) > 0:
            # Первый пик
            peak_idx = peaks[0]
            delay_samples = peak_idx - len(ref_signal) + 1
        else:
            # Fallback: максимальный пик
            peak_idx = np.argmax(np.abs(correlation))
            delay_samples = peak_idx - len(ref_signal) + 1
        
        # Ограничение диапазона
        delay_samples = np.clip(delay_samples, -self.max_delay_samples, self.max_delay_samples)
        
        return delay_samples
    
    def _magnitude_coherence(self, ref_signal: np.ndarray, signal: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        """
        Magnitude Squared Coherence.
        
        Определение частотных областей, где сигналы наиболее связаны
        и фазовые измерения достоверны.
        """
        # Вычисление coherence
        freq, coh = coherence(ref_signal, signal, fs=self.sample_rate, nperseg=min(len(ref_signal), 2048))
        
        # Средняя coherence
        mean_coherence = np.mean(coh)
        
        # Частотный диапазон с высокой coherence (>0.7)
        high_coh_mask = coh > 0.7
        if np.any(high_coh_mask):
            high_coh_freqs = freq[high_coh_mask]
            freq_range = (np.min(high_coh_freqs), np.max(high_coh_freqs))
        else:
            freq_range = (freq[0], freq[-1])
        
        return mean_coherence, freq_range
    
    def _group_delay(self, ref_signal: np.ndarray, signal: np.ndarray) -> float:
        """
        Group Delay Calculation.
        
        Измерение задержки как производной фазового сдвига по частоте
        для выявления частотно-зависимых смещений.
        """
        # FFT
        ref_fft = np.fft.fft(ref_signal)
        sig_fft = np.fft.fft(signal)
        
        # Phase difference
        phase_diff = np.angle(sig_fft) - np.angle(ref_fft)
        
        # Unwrap phase
        phase_diff = np.unwrap(phase_diff)
        
        # Frequency array
        freqs = np.fft.fftfreq(len(ref_signal), 1.0 / self.sample_rate)
        
        # Group delay = -d(phase)/d(frequency)
        # Используем центральные частоты (игнорируем DC и Nyquist)
        valid_range = slice(1, len(phase_diff) // 2)
        phase_valid = phase_diff[valid_range]
        freq_valid = freqs[valid_range]
        
        if len(phase_valid) > 1:
            # Численная производная
            phase_diff_grad = np.gradient(phase_valid)
            freq_diff = np.gradient(freq_valid)
            group_delay = -phase_diff_grad / (freq_diff + 1e-10)
            
            # Средний group delay
            mean_group_delay = np.mean(group_delay)
        else:
            mean_group_delay = 0.0
        
        return mean_group_delay
    
    def _hilbert_phase_diff(self, ref_signal: np.ndarray, signal: np.ndarray) -> float:
        """
        Hilbert Transform.
        
        Получение мгновенной фазы и огибающей для анализа фазовых соотношений.
        """
        # Analytic signal через Hilbert transform
        ref_analytic = hilbert(ref_signal)
        sig_analytic = hilbert(signal)
        
        # Instantaneous phase
        ref_phase = np.angle(ref_analytic)
        sig_phase = np.angle(sig_analytic)
        
        # Phase difference
        phase_diff = sig_phase - ref_phase
        
        # Unwrap
        phase_diff = np.unwrap(phase_diff)
        
        # Средняя разность фаз
        mean_phase_diff = np.mean(phase_diff)
        
        return mean_phase_diff
    
    def _lms_delay(self, ref_signal: np.ndarray, signal: np.ndarray) -> float:
        """
        LMS (Least Mean Squares) Adaptive Filtering.
        
        Использование адаптивного фильтра для автоматического поиска задержки,
        минимизирующей разность между сигналами.
        """
        # Упрощенная версия LMS для оценки задержки
        # Используем cross-correlation как приближение
        
        # Нормализация
        ref_norm = ref_signal / (np.linalg.norm(ref_signal) + 1e-10)
        sig_norm = signal / (np.linalg.norm(signal) + 1e-10)
        
        # Cross-correlation
        correlation = np.correlate(ref_norm, sig_norm, mode='full')
        
        # Поиск задержки с минимальной ошибкой
        search_range = min(self.max_delay_samples, len(correlation) // 2)
        search_start = len(correlation) // 2 - search_range
        search_end = len(correlation) // 2 + search_range
        
        search_region = correlation[search_start:search_end]
        peak_idx = np.argmax(np.abs(search_region))
        delay_samples = peak_idx + search_start - len(correlation) // 2
        
        return delay_samples
    
    def get_measurements(self) -> Dict:
        """Получение текущих результатов измерений."""
        return self.measurements.copy()


class PhaseAlignmentController:
    """
    Контроллер для управления анализом и коррекцией фаз и задержек.
    """
    
    def __init__(self, mixer_client=None, bleed_service=None, settings: dict = None):
        """
        Инициализация контроллера.
        
        Args:
            mixer_client: Клиент для отправки OSC команд микшеру
            bleed_service: Centralized bleed detection service
            settings: Настройки анализа (max_delay_ms, fft_size, num_coherence_freqs)
        """
        self.mixer_client = mixer_client
        self.bleed_service = bleed_service
        self.settings = settings or {}
        self.max_delay_ms = self.settings.get('maxDelayMs', 10.0)  # Default 10ms - musically acceptable
        self.analyzer: Optional[PhaseAlignmentAnalyzer] = None
        self.is_active = False
        self.reference_channel = None
        self.channels_to_align = []
        self.all_channels = []  # Все каналы включая reference
        self.corrections: Dict[int, Dict[str, float]] = {}  # {channel: {delay_ms, phase_invert}}
        self._apply_once_timer: Optional[threading.Timer] = None
        
    def start_analysis(self,
                      device_id: int,
                      reference_channel: int,
                      channels: List[int],
                      on_measurement_callback: Callable = None,
                      apply_once: bool = False,
                      apply_once_duration_sec: float = 5.0,
                      settings: dict = None):
        """
        Запуск анализа фаз и задержек.
        
        Args:
            device_id: Индекс аудио устройства
            reference_channel: Референсный канал
            channels: Список каналов для выравнивания
            on_measurement_callback: Callback для обновления результатов
            apply_once: Если True — один замер на саундчеке, затем автостоп и применение (без непрерывной коррекции)
            apply_once_duration_sec: Длительность замера в режиме apply_once (сек)
        """
        if self.is_active:
            logger.warning("Phase alignment already active")
            return False
        
        if self._apply_once_timer:
            self._apply_once_timer.cancel()
            self._apply_once_timer = None
        
        self.reference_channel = reference_channel
        self.channels_to_align = channels
        # Сохраняем все каналы включая reference для правильной обработки
        self.all_channels = [reference_channel] + channels
        
        # Use settings from parameter or stored settings
        settings = settings or self.settings or {}
        # Update max_delay_ms from settings
        self.max_delay_ms = settings.get('maxDelayMs', self.max_delay_ms)
        self.analyzer = PhaseAlignmentAnalyzer(
            device_index=device_id,
            max_delay_ms=self.max_delay_ms,
            chunk_size=settings.get('fftSize', 4096)
        )
        
        def measurement_callback(measurements):
            if on_measurement_callback:
                on_measurement_callback(measurements)
        
        success = self.analyzer.start(
            reference_channel=reference_channel,
            channels=channels,
            on_measurement_callback=measurement_callback
        )
        
        if success:
            self.is_active = True
            logger.info(f"Phase alignment analysis started: ref={reference_channel}, channels={channels}")
            if apply_once:
                def _after_duration():
                    if not self.analyzer or not self.is_active:
                        return
                    measurements = self.analyzer.get_measurements()
                    self.stop_analysis()
                    if measurements and self.mixer_client:
                        self.apply_corrections(measurements)
                        logger.info("Phase alignment apply_once: corrections applied")
                    self._apply_once_timer = None
                self._apply_once_timer = threading.Timer(apply_once_duration_sec, _after_duration)
                self._apply_once_timer.daemon = True
                self._apply_once_timer.start()
                logger.info(f"Phase alignment apply_once: will stop and apply after {apply_once_duration_sec}s")
        
        return success
    
    def stop_analysis(self):
        """Остановка анализа."""
        if not self.is_active:
            return
        if self._apply_once_timer:
            self._apply_once_timer.cancel()
            self._apply_once_timer = None
        if self.analyzer:
            self.analyzer.stop()
            self.analyzer = None
        self.is_active = False
        logger.info("Phase alignment analysis stopped")
    
    def apply_corrections(self, measurements: Dict = None):
        """
        Применение коррекций к микшеру через OSC.
        
        Args:
            measurements: Результаты измерений (если None, используются последние)
        """
        if not self.mixer_client:
            logger.error("No mixer client available")
            return False
        
        if measurements is None:
            if self.analyzer:
                measurements = self.analyzer.get_measurements()
                logger.info(f"Got measurements from analyzer: {measurements}")
            else:
                logger.error("No measurements available and no analyzer")
                return False
        
        if not isinstance(measurements, dict):
            logger.error(f"Invalid measurements type: {type(measurements)}, expected dict")
            return False
        
        if len(measurements) == 0:
            logger.warning("Measurements dictionary is empty")
            return False
        
        logger.info("Applying phase/delay corrections to mixer...")
        logger.info(f"Measurements received: {measurements}")
        logger.info(f"Measurements type: {type(measurements)}, length: {len(measurements)}")
        logger.info(f"Channels to align: {self.channels_to_align}")
        
        applied_count = 0
        
        # Обработка измерений
        for pair_key, meas in measurements.items():
            try:
                # Парсинг ключа пары каналов
                # Формат может быть: "(ref_ch, ch)" или tuple
                if isinstance(pair_key, tuple):
                    if len(pair_key) != 2:
                        logger.warning(f"Invalid tuple length: {pair_key}")
                        continue
                    ref_ch, ch = pair_key
                elif isinstance(pair_key, str):
                    # Парсинг строки вида "(1, 2)"
                    import re
                    match = re.match(r'\((\d+),\s*(\d+)\)', pair_key)
                    if match:
                        ref_ch = int(match.group(1))
                        ch = int(match.group(2))
                    else:
                        logger.warning(f"Could not parse channel pair: {pair_key}")
                        continue
                else:
                    logger.warning(f"Unknown pair key format: {pair_key} (type: {type(pair_key)})")
                    continue
                
                # Проверка типа данных измерения
                if not isinstance(meas, dict):
                    logger.warning(f"Invalid measurement data type for {pair_key}: {type(meas)}")
                    continue
                
            except Exception as e:
                logger.error(f"Error parsing pair_key {pair_key}: {e}")
                continue
            
            # Проверка, что канал должен быть обработан
            # Обрабатываем все каналы из измерений, которые есть в all_channels или channels_to_align
            # Это включает каналы перед reference channel
            should_process = False
            
            # Проверяем, что канал в списке всех каналов (включая reference)
            if hasattr(self, 'all_channels') and self.all_channels:
                if ch in self.all_channels:
                    should_process = True
            # Или проверяем channels_to_align
            elif self.channels_to_align:
                if ch in self.channels_to_align:
                    should_process = True
                # Также обрабатываем каналы перед reference
                elif self.reference_channel and ch < self.reference_channel:
                    should_process = True
                    logger.debug(f"Processing channel {ch} (before reference channel {self.reference_channel})")
            else:
                # Если списки не заданы, обрабатываем все каналы из измерений
                should_process = True
            
            if not should_process:
                logger.debug(f"Skipping channel {ch} (not in processing list)")
                continue
            
            # Check for bleed: skip or reduce confidence if high bleed detected
            # Bleed can distort cross-correlation measurements
            if self.bleed_service and self.bleed_service.enabled:
                bleed_info = self.bleed_service.get_bleed_info(ch)
                if bleed_info and bleed_info.bleed_ratio > 0.5:
                    coherence = meas.get('coherence', 0.0)
                    # If coherence is already low (<0.5) and bleed is high, skip correction
                    if coherence < 0.5:
                        logger.warning(f"Skipping phase correction for channel {ch}: high bleed (ratio={bleed_info.bleed_ratio:.2f}) "
                                     f"and low coherence ({coherence:.2f}) - measurement unreliable")
                        continue
                    else:
                        # Reduce confidence: only apply if delay is significant and coherence is high
                        if abs(meas.get('optimal_delay_ms', 0.0)) < 1.0:
                            logger.debug(f"Skipping small delay correction for channel {ch} due to bleed (ratio={bleed_info.bleed_ratio:.2f})")
                            continue
            
            delay_ms = meas.get('optimal_delay_ms', 0.0)
            phase_invert = meas.get('phase_invert', 0)
            coherence = meas.get('coherence', 0.0)
            
            logger.info(f"Processing channel {ch}: delay={delay_ms:.2f} ms, phase_invert={phase_invert}, coherence={coherence:.2f}")
            
            # PRIORITY: Phase inversion first, delay only if necessary
            # If phase inversion is needed and delay is small (< 2ms), try phase only first
            # Delay is applied only if it's significant (> 2ms) or coherence is low even with phase
            DELAY_SIGNIFICANT_THRESHOLD = 2.0  # ms - below this, phase inversion may be sufficient
            COHERENCE_THRESHOLD = 0.6  # minimum coherence for acceptable alignment
            
            # Determine if delay is truly needed
            # If delay is small and phase inversion is needed, prioritize phase inversion only
            if phase_invert == 1 and abs(delay_ms) < DELAY_SIGNIFICANT_THRESHOLD:
                if coherence >= COHERENCE_THRESHOLD:
                    # Good coherence with phase inversion - delay may not be needed
                    logger.info(f"Channel {ch}: Small delay ({delay_ms:.2f}ms) with phase invert - prioritizing phase only")
                    delay_ms = 0.0  # Skip delay, phase inversion should suffice
                else:
                    logger.info(f"Channel {ch}: Low coherence ({coherence:.2f}) even with phase invert - keeping delay")
            
            # If delay is very small (< 0.3ms), it's likely not musically significant
            # Skip it unless coherence is very low
            if abs(delay_ms) < 0.3 and coherence >= COHERENCE_THRESHOLD:
                logger.info(f"Channel {ch}: Negligible delay ({delay_ms:.2f}ms) with good coherence - skipping delay")
                delay_ms = 0.0
            
            # Сохранение коррекций
            self.corrections[ch] = {
                'delay_ms': delay_ms,
                'phase_invert': phase_invert
            }
            
            # Применение задержки только если она значительная
            if abs(delay_ms) >= 0.3:  # Минимальный порог 0.3ms (musically significant)
                # Конвертация в миллисекунды (режим MS)
                # Убеждаемся, что задержка положительная (абсолютное значение)
                # set_channel_delay автоматически включает задержку (dlyon = 1)
                delay_value = abs(delay_ms)
                # Убеждаемся, что значение в допустимом диапазоне (0.5 - max_delay_ms)
                max_delay = getattr(self, 'max_delay_ms', 10.0)
                delay_value = max(0.5, min(max_delay, delay_value))
                
                logger.info(f"Setting delay for channel {ch}: {delay_value:.2f} ms")
                self.mixer_client.set_channel_delay(ch, delay_value, mode="MS")
                logger.info(f"Channel {ch}: Applied delay {delay_value:.2f} ms (enabled)")
            else:
                # Отключение задержки если она очень мала
                logger.info(f"Disabling delay for channel {ch} (too small: {delay_ms:.2f} ms)")
                self.mixer_client.send(f"/ch/{ch}/in/set/dlyon", 0)
                logger.info(f"Channel {ch}: Delay disabled (too small: {delay_ms:.2f} ms)")
            
            # Применение инверсии фазы (1 = инвертировано, 0 = нормально)
            logger.info(f"Setting phase invert for channel {ch}: {phase_invert}")
            self.mixer_client.set_channel_phase_invert(ch, phase_invert)
            if phase_invert:
                logger.info(f"Channel {ch}: Applied phase inversion")
            else:
                logger.info(f"Channel {ch}: Phase normal (not inverted)")
            
            applied_count += 1
        
        logger.info(f"Applied corrections to {applied_count} channels")
        return applied_count > 0
    
    def reset_all_phase_delay(self, channels: List[int] = None):
        """
        Сброс и выключение всех фаз и задержек на указанных каналах.
        
        Args:
            channels: Список каналов для сброса (если None, сбрасываются все каналы с коррекциями)
        """
        if not self.mixer_client:
            logger.error("No mixer client available")
            return False
        
        if channels is None:
            # Используем все каналы с коррекциями или все каналы для выравнивания
            channels = list(self.corrections.keys()) if self.corrections else self.channels_to_align
        
        if not channels:
            logger.warning("No channels specified for reset")
            return False
        
        logger.info(f"Resetting phase and delay for {len(channels)} channels...")
        
        reset_count = 0
        
        for ch in channels:
            try:
                # Сброс инверсии фазы (0 = нормальная фаза)
                self.mixer_client.set_channel_phase_invert(ch, 0)
                
                # Отключение задержки
                self.mixer_client.send(f"/ch/{ch}/in/set/dlyon", 0)
                
                # Сброс значения задержки на минимальное
                self.mixer_client.send(f"/ch/{ch}/in/set/dlymode", "MS")
                self.mixer_client.send(f"/ch/{ch}/in/set/dly", 0.5)  # Минимальное значение
                
                # Удаление из коррекций
                if ch in self.corrections:
                    del self.corrections[ch]
                
                reset_count += 1
                logger.info(f"Channel {ch}: Reset phase and delay")
                
            except Exception as e:
                logger.error(f"Error resetting channel {ch}: {e}")
        
        logger.info(f"Reset phase and delay for {reset_count} channels")
        return True
    
    def get_status(self) -> Dict:
        """Получение статуса контроллера."""
        return {
            'active': self.is_active,
            'reference_channel': self.reference_channel,
            'channels': self.channels_to_align,
            'corrections': self.corrections.copy()
        }
