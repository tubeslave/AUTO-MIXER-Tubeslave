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
from typing import Any, Dict, List, Callable, Optional, Tuple
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
        N = len(gcc)

        # GCC output layout: indices [0..N/2] = positive lags,
        # indices [N/2+1..N-1] = negative lags (wrapped).
        # Use fftshift to get a contiguous array centered at lag 0.
        gcc_shifted = np.fft.fftshift(gcc)
        center = N // 2

        search_range = min(self.max_delay_samples, N // 2)
        search_start = center - search_range
        search_end = center + search_range + 1

        search_region = gcc_shifted[search_start:search_end]
        abs_region = np.abs(search_region)
        max_peak = np.max(abs_region)

        # Find minimum delay with good correlation (>= 70% of max peak)
        threshold = 0.7 * max_peak
        good_peaks_mask = abs_region >= threshold

        if np.any(good_peaks_mask):
            good_indices = np.where(good_peaks_mask)[0]
            delays = good_indices + search_start - center
            min_delay_idx_in_subset = np.argmin(np.abs(delays))
            peak_idx = good_indices[min_delay_idx_in_subset]
        else:
            peak_idx = np.argmax(abs_region)

        delay_samples = peak_idx + search_start - center
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
        self.live_write_allowed = False
        self.write_policy = "proposal_only_quarantined"
        self.last_write_attempted = False
        self.last_write_blocked = False
        self.last_write_count = 0
        self.last_blocked_calls: List[Dict[str, Any]] = []
        self.last_apply_result: Dict[str, Any] = {}
        
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
                        result = self.apply_corrections(measurements)
                        logger.info("Phase alignment apply_once: corrections blocked as evidence: %s", result)
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
    
    def apply_corrections(self, measurements: Dict = None) -> Dict[str, Any]:
        """
        Build blocked phase/delay correction evidence without mutating a mixer.

        Args:
            measurements: Результаты измерений (если None, используются последние)
        """
        self._reset_write_telemetry()

        if measurements is None:
            if self.analyzer:
                measurements = self.analyzer.get_measurements()
                logger.info(f"Got measurements from analyzer: {measurements}")
            else:
                logger.error("No measurements available and no analyzer")
                return self._blocked_phase_result("no_phase_measurements", [])

        if not isinstance(measurements, dict):
            logger.error(f"Invalid measurements type: {type(measurements)}, expected dict")
            return self._blocked_phase_result("invalid_phase_measurements", [])

        if len(measurements) == 0:
            logger.warning("Measurements dictionary is empty")
            return self._blocked_phase_result("empty_phase_measurements", [])

        logger.info("Building blocked phase/delay correction evidence...")
        logger.info(f"Measurements received: {measurements}")
        logger.info(f"Measurements type: {type(measurements)}, length: {len(measurements)}")
        logger.info(f"Channels to align: {self.channels_to_align}")

        blocked_calls: List[Dict[str, Any]] = []

        # Обработка измерений
        for pair_key, meas in measurements.items():
            parsed_pair = self._parse_measurement_pair(pair_key)
            if parsed_pair is None:
                continue
            ref_ch, ch = parsed_pair

            if not isinstance(meas, dict):
                logger.warning(f"Invalid measurement data type for {pair_key}: {type(meas)}")
                continue

            if not self._should_process_channel(ch):
                logger.debug(f"Skipping channel {ch} (not in processing list)")
                continue

            if self._should_skip_for_bleed(ch, meas):
                continue

            delay_ms = float(meas.get('optimal_delay_ms', 0.0) or 0.0)
            phase_invert = int(meas.get('phase_invert', 0) or 0)
            coherence = float(meas.get('coherence', 0.0) or 0.0)

            logger.info(f"Processing channel {ch}: delay={delay_ms:.2f} ms, phase_invert={phase_invert}, coherence={coherence:.2f}")

            delay_ms = self._phase_delay_candidate(delay_ms, phase_invert, coherence, ch)

            # Сохранение коррекций as local evidence only; no console writes happen here.
            self.corrections[ch] = {
                'delay_ms': delay_ms,
                'phase_invert': phase_invert
            }

            blocked_calls.extend(
                self._build_phase_apply_calls(
                    reference_channel=ref_ch,
                    channel=ch,
                    delay_ms=delay_ms,
                    phase_invert=phase_invert,
                    coherence=coherence,
                )
            )

        reason = "phase_delay_direct_apply_quarantined" if blocked_calls else "no_phase_corrections"
        result = self._blocked_phase_result(reason, blocked_calls)
        logger.info("Phase/delay direct apply blocked; %s would-be calls kept as evidence", len(blocked_calls))
        return result

    def _reset_write_telemetry(self):
        self.last_write_attempted = False
        self.last_write_blocked = False
        self.last_write_count = 0
        self.last_blocked_calls = []
        self.last_apply_result = {}

    def _parse_measurement_pair(self, pair_key) -> Optional[Tuple[int, int]]:
        try:
            if isinstance(pair_key, tuple):
                if len(pair_key) != 2:
                    logger.warning(f"Invalid tuple length: {pair_key}")
                    return None
                return int(pair_key[0]), int(pair_key[1])
            if isinstance(pair_key, str):
                import re
                match = re.match(r'\((\d+),\s*(\d+)\)', pair_key)
                if match:
                    return int(match.group(1)), int(match.group(2))
                logger.warning(f"Could not parse channel pair: {pair_key}")
                return None
            logger.warning(f"Unknown pair key format: {pair_key} (type: {type(pair_key)})")
            return None
        except Exception as exc:
            logger.error(f"Error parsing pair_key {pair_key}: {exc}")
            return None

    def _should_process_channel(self, channel: int) -> bool:
        # Обрабатываем все каналы из измерений, которые есть в all_channels или channels_to_align.
        if hasattr(self, 'all_channels') and self.all_channels:
            return channel in self.all_channels
        if self.channels_to_align:
            if channel in self.channels_to_align:
                return True
            if self.reference_channel and channel < self.reference_channel:
                logger.debug(f"Processing channel {channel} (before reference channel {self.reference_channel})")
                return True
            return False
        return True

    def _should_skip_for_bleed(self, channel: int, measurement: Dict[str, Any]) -> bool:
        if not self.bleed_service or not self.bleed_service.enabled:
            return False

        bleed_info = self.bleed_service.get_bleed_info(channel)
        if not bleed_info or bleed_info.bleed_ratio <= 0.5:
            return False

        coherence = float(measurement.get('coherence', 0.0) or 0.0)
        if coherence < 0.5:
            logger.warning(
                "Skipping phase correction for channel %s: high bleed (ratio=%.2f) and low coherence (%.2f)",
                channel,
                bleed_info.bleed_ratio,
                coherence,
            )
            return True
        if abs(float(measurement.get('optimal_delay_ms', 0.0) or 0.0)) < 1.0:
            logger.debug(
                "Skipping small delay correction for channel %s due to bleed (ratio=%.2f)",
                channel,
                bleed_info.bleed_ratio,
            )
            return True
        return False

    def _phase_delay_candidate(
        self,
        delay_ms: float,
        phase_invert: int,
        coherence: float,
        channel: int,
    ) -> float:
        delay_significant_threshold = 2.0
        coherence_threshold = 0.6

        if phase_invert == 1 and abs(delay_ms) < delay_significant_threshold:
            if coherence >= coherence_threshold:
                logger.info(
                    "Channel %s: Small delay (%.2fms) with phase invert - prioritizing phase only",
                    channel,
                    delay_ms,
                )
                delay_ms = 0.0
            else:
                logger.info(
                    "Channel %s: Low coherence (%.2f) even with phase invert - keeping delay",
                    channel,
                    coherence,
                )

        if abs(delay_ms) < 0.3 and coherence >= coherence_threshold:
            logger.info(
                "Channel %s: Negligible delay (%.2fms) with good coherence - skipping delay",
                channel,
                delay_ms,
            )
            delay_ms = 0.0

        return delay_ms

    def _build_phase_apply_calls(
        self,
        *,
        reference_channel: int,
        channel: int,
        delay_ms: float,
        phase_invert: int,
        coherence: float,
    ) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        if abs(delay_ms) >= 0.3:
            max_delay = getattr(self, 'max_delay_ms', 10.0)
            delay_value = max(0.5, min(max_delay, abs(delay_ms)))
            calls.append(
                {
                    "cmd": "set_channel_delay",
                    "channel": channel,
                    "delay_ms": delay_value,
                    "mode": "MS",
                    "reference_channel": reference_channel,
                    "measured_delay_ms": delay_ms,
                    "coherence": coherence,
                }
            )
        else:
            calls.append(
                {
                    "cmd": "send",
                    "path": f"/ch/{channel}/in/set/dlyon",
                    "value": 0,
                    "semantic": "disable_channel_delay",
                    "channel": channel,
                    "reference_channel": reference_channel,
                    "measured_delay_ms": delay_ms,
                    "coherence": coherence,
                }
            )

        calls.append(
            {
                "cmd": "set_channel_phase_invert",
                "channel": channel,
                "phase_invert": phase_invert,
                "reference_channel": reference_channel,
                "coherence": coherence,
            }
        )
        return calls

    def _build_phase_reset_calls(self, channels: List[int]) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for channel in channels:
            calls.extend(
                [
                    {
                        "cmd": "set_channel_phase_invert",
                        "channel": channel,
                        "phase_invert": 0,
                        "semantic": "reset_phase_invert",
                    },
                    {
                        "cmd": "send",
                        "path": f"/ch/{channel}/in/set/dlyon",
                        "value": 0,
                        "semantic": "disable_channel_delay",
                        "channel": channel,
                    },
                    {
                        "cmd": "send",
                        "path": f"/ch/{channel}/in/set/dlymode",
                        "value": "MS",
                        "semantic": "reset_delay_mode",
                        "channel": channel,
                    },
                    {
                        "cmd": "send",
                        "path": f"/ch/{channel}/in/set/dly",
                        "value": 0.5,
                        "semantic": "reset_delay_value",
                        "channel": channel,
                    },
                ]
            )
        return calls

    def _blocked_phase_result(self, reason: str, blocked_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.last_write_attempted = bool(blocked_calls)
        self.last_write_blocked = bool(blocked_calls)
        self.last_write_count = 0
        self.last_blocked_calls = list(blocked_calls)
        result = {
            "applied": False,
            "blocked": bool(blocked_calls),
            "calls": [],
            "blocked_calls": list(blocked_calls),
            "blocked_call_count": len(blocked_calls),
            "reason": reason,
            "write_policy": self.write_policy,
            "live_write_allowed": self.live_write_allowed,
            "can_apply_live": False,
            "mixer_client_present": self.mixer_client is not None,
            "mutates_mixer": False,
        }
        self.last_apply_result = result
        if blocked_calls:
            logger.warning(
                "Phase/delay live writes are quarantined; %s commands kept as evidence",
                len(blocked_calls),
            )
        return result
    
    def reset_all_phase_delay(self, channels: List[int] = None) -> Dict[str, Any]:
        """
        Build blocked reset evidence for phase and delay values.
        
        Args:
            channels: Список каналов для сброса (если None, сбрасываются все каналы с коррекциями)
        """
        self._reset_write_telemetry()

        if channels is None:
            # Используем все каналы с коррекциями или все каналы для выравнивания
            channels = list(self.corrections.keys()) if self.corrections else self.channels_to_align
        
        if not channels:
            logger.warning("No channels specified for reset")
            return self._blocked_phase_result("no_phase_reset_channels", [])

        logger.info(f"Building blocked phase/delay reset evidence for {len(channels)} channels...")
        blocked_calls = self._build_phase_reset_calls(channels)
        result = self._blocked_phase_result("phase_delay_reset_quarantined", blocked_calls)
        logger.info("Phase/delay reset blocked; %s would-be calls kept as evidence", len(blocked_calls))
        return result
    
    def get_status(self) -> Dict:
        """Получение статуса контроллера."""
        return {
            'active': self.is_active,
            'reference_channel': self.reference_channel,
            'channels': self.channels_to_align,
            'corrections': self.corrections.copy(),
            'live_write_allowed': self.live_write_allowed,
            'write_policy': self.write_policy,
            'mixer_client_present': self.mixer_client is not None,
            'last_write_attempted': self.last_write_attempted,
            'last_write_blocked': self.last_write_blocked,
            'last_write_count': self.last_write_count,
            'last_blocked_call_count': len(self.last_blocked_calls),
            'last_blocked_calls': list(self.last_blocked_calls),
            'last_apply_result': dict(self.last_apply_result),
        }
