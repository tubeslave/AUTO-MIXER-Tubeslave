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

    def __init__(
        self,
        device_index: int = None,
        sample_rate: int = 48000,
        chunk_size: int = 4096,
        analysis_window: float = 2.0,
        max_delay_ms: float = 10.0,
    ):
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
        self.on_audio_data_updated: Optional[
            Callable[[Dict[int, np.ndarray]], None]
        ] = None

    def start(
        self,
        reference_channel: int,
        channels: List[int],
        on_measurement_callback: Callable = None,
    ):
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

        all_channels = [reference_channel] + channels

        logger.info(f"=== PHASE ALIGNMENT: Starting analysis ===")
        logger.info(f"Reference channel: {reference_channel}")
        logger.info(f"Channels to align: {channels}")
        logger.info(f"Device index: {self.device_index}")
        logger.info(
            f"Analysis window: {self.analysis_window}s (accumulate full window)"
        )
        logger.info(f"Max delay: {self.max_delay_ms} ms")

        try:
            import pyaudio

            self.pyaudio = pyaudio

            self.pa = pyaudio.PyAudio()

            if self.device_index is not None:
                device_info = self.pa.get_device_info_by_index(int(self.device_index))
                max_channels = int(device_info.get("maxInputChannels", 2))
                device_sample_rate = int(device_info.get("defaultSampleRate", 48000))
                logger.info(
                    f"Selected device {self.device_index}: {device_info.get('name')}"
                )
            else:
                device_info = self.pa.get_default_input_device_info()
                max_channels = int(device_info.get("maxInputChannels", 2))
                device_sample_rate = int(device_info.get("defaultSampleRate", 48000))
                logger.info(f"Using default device: {device_info.get('name')}")

            self.sample_rate = device_sample_rate

            # Инициализация буферов после получения sample_rate устройства
            chunks_per_window = int(
                (self.sample_rate * self.analysis_window) / self.chunk_size
            )
            for ch in all_channels:
                self.audio_buffers[ch] = deque(maxlen=chunks_per_window)
            # Для phase alignment нужно захватывать все каналы одновременно
            # Используем максимальное количество каналов из доступных
            required_channels = max(all_channels) if all_channels else 2
            num_channels = min(required_channels, max_channels)

            # Сохранение для callback
            self._num_channels = num_channels
            self._channel_ids = sorted(all_channels)

            logger.info(
                f"Opening stream: {num_channels} channels, {self.sample_rate} Hz"
            )

            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=num_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=int(self.device_index)
                if self.device_index is not None
                else None,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
            )

            self.stream.start_stream()
            self.is_running = True
            self._stop_event.clear()

            # Запуск потока анализа
            self._analysis_thread = threading.Thread(
                target=self._analysis_loop, daemon=True
            )
            self._analysis_thread.start()

            logger.info("Phase alignment analyzer started")
            return True

        except Exception as e:
            logger.error(
                f"Failed to start phase alignment analyzer: {e}", exc_info=True
            )
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
            except Exception:
                pass

        if self.pa:
            try:
                self.pa.terminate()
            except Exception:
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
            if hasattr(self, "_num_channels") and hasattr(self, "_channel_ids"):
                num_channels = self._num_channels
                channel_ids = self._channel_ids

                if num_channels > 0 and len(channel_ids) > 0:
                    # Reshape для разделения каналов
                    samples_per_channel = len(audio_data) // num_channels
                    if samples_per_channel > 0 and len(audio_data) % num_channels == 0:
                        audio_reshaped = audio_data.reshape(
                            samples_per_channel, num_channels
                        )

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
                min_length = float("inf")

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
                    channel_data[ch] = channel_data[ch][: int(min_length)]

                # Выполнение измерений
                self._perform_measurements(channel_data)

                if self.on_audio_data_updated:
                    try:
                        self.on_audio_data_updated(channel_data.copy())
                    except Exception as e:
                        logger.debug(f"Error in on_audio_data_updated callback: {e}")

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
        if (
            not hasattr(self, "reference_channel")
            or self.reference_channel not in channel_data
        ):
            # Fallback: используем минимальный канал если reference не задан
            reference_ch = min(channel_data.keys())
            logger.warning(
                f"Reference channel not set, using minimum channel: {reference_ch}"
            )
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
            measurements["delay_gcc_ms"] = delay_gcc * 1000.0 / self.sample_rate
            measurements["gcc_peak_value"] = float(gcc_peak_value)
            coherence_gcc = self._estimate_coherence(ref_signal, signal_data)
            measurements["coherence_gcc"] = (
                coherence_gcc if np.isfinite(coherence_gcc) else 0.0
            )

            # 2. Phase Angle Difference
            phase_diff = self._phase_angle_difference(ref_signal, signal_data)
            measurements["phase_diff_deg"] = np.degrees(phase_diff)

            # 3. Impulse Response Analysis
            delay_ir = self._impulse_response_delay(ref_signal, signal_data)
            measurements["delay_ir_ms"] = delay_ir * 1000.0 / self.sample_rate

            # 4. Magnitude Squared Coherence
            coherence_val, freq_range = self._magnitude_coherence(
                ref_signal, signal_data
            )
            measurements["coherence"] = (
                coherence_val if np.isfinite(coherence_val) else 0.0
            )
            measurements["coherence_freq_range"] = freq_range
            measurements["spectral_overlap_mid_high"] = self._spectral_overlap_mid_high(
                ref_signal,
                signal_data,
            )

            # 5. Group Delay
            group_delay = self._group_delay(ref_signal, signal_data)
            measurements["group_delay_ms"] = group_delay * 1000.0 / self.sample_rate

            # 6. Hilbert Transform (instantaneous phase)
            inst_phase_diff = self._hilbert_phase_diff(ref_signal, signal_data)
            measurements["inst_phase_diff_deg"] = np.degrees(inst_phase_diff)

            # 7. LMS Adaptive Filtering
            delay_lms = self._lms_delay(ref_signal, signal_data)
            measurements["delay_lms_ms"] = delay_lms * 1000.0 / self.sample_rate

            # Phase invert: negative GCC peak = anti-correlation → invert phase
            phase_invert = 1 if gcc_peak_value < 0 else 0
            measurements["phase_invert"] = phase_invert

            # For UI display during analysis
            measurements["delay_ms"] = float(measurements["delay_gcc_ms"])
            measurements["invert_phase"] = bool(phase_invert)

            self.measurements[pair] = measurements

    def _gcc_phat(
        self, ref_signal: np.ndarray, signal: np.ndarray
    ) -> Tuple[int, float]:
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
        mean_coh = np.mean(coh)
        if not np.isfinite(mean_coh):
            return 0.0
        return float(mean_coh)

    def _phase_angle_difference(
        self, ref_signal: np.ndarray, signal: np.ndarray
    ) -> float:
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

    def _impulse_response_delay(
        self, ref_signal: np.ndarray, signal: np.ndarray
    ) -> int:
        """
        Impulse Response Analysis.

        Сравнение времени прихода первых пиков (транзиентов) в двух сигналах.
        """
        # Cross-correlation
        correlation = np.correlate(ref_signal, signal, mode="full")

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
        delay_samples = np.clip(
            delay_samples, -self.max_delay_samples, self.max_delay_samples
        )

        return delay_samples

    def _magnitude_coherence(
        self, ref_signal: np.ndarray, signal: np.ndarray
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Magnitude Squared Coherence.

        Определение частотных областей, где сигналы наиболее связаны
        и фазовые измерения достоверны.
        """
        # Вычисление coherence
        freq, coh = coherence(
            ref_signal, signal, fs=self.sample_rate, nperseg=min(len(ref_signal), 2048)
        )

        # Средняя coherence
        mean_coherence = np.mean(coh)
        if not np.isfinite(mean_coherence):
            mean_coherence = 0.0

        # Частотный диапазон с высокой coherence (>0.7)
        high_coh_mask = coh > 0.7
        if np.any(high_coh_mask):
            high_coh_freqs = freq[high_coh_mask]
            freq_range = (np.min(high_coh_freqs), np.max(high_coh_freqs))
        else:
            freq_range = (freq[0], freq[-1])

        return float(mean_coherence), freq_range

    def _spectral_overlap_mid_high(
        self, ref_signal: np.ndarray, signal: np.ndarray
    ) -> float:
        """
        Spectral overlap for mid/high bands, useful for snare-like reference checks.
        Returns 0..1 where 1 means high overlap.
        """
        n_fft = min(len(ref_signal), len(signal), 4096)
        if n_fft < 256:
            return 0.0

        ref_win = ref_signal[:n_fft] * np.hanning(n_fft)
        sig_win = signal[:n_fft] * np.hanning(n_fft)
        ref_spec = np.abs(np.fft.rfft(ref_win))
        sig_spec = np.abs(np.fft.rfft(sig_win))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        # Snare-related energy is usually concentrated in mid/high bands.
        band_mask = (freqs >= 800.0) & (freqs <= 8000.0)
        if not np.any(band_mask):
            return 0.0

        ref_band = ref_spec[band_mask]
        sig_band = sig_spec[band_mask]
        ref_norm = np.linalg.norm(ref_band)
        sig_norm = np.linalg.norm(sig_band)
        if ref_norm <= 1e-10 or sig_norm <= 1e-10:
            return 0.0

        overlap = float(np.dot(ref_band, sig_band) / (ref_norm * sig_norm))
        if not np.isfinite(overlap):
            return 0.0
        return float(np.clip(overlap, 0.0, 1.0))

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
        correlation = np.correlate(ref_norm, sig_norm, mode="full")

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
            settings: Настройки анализа
        """
        self.mixer_client = mixer_client
        self.bleed_service = bleed_service
        self.settings = settings or {}
        self.max_delay_ms = 10.0
        self.analysis_window_sec = 10.0
        # Поиск референса в других каналах только по когерентности.
        self.reference_coherence_min = 0.10
        self.reference_gcc_peak_min = 0.15
        self.reference_min_hits = 5
        self.reference_spectral_overlap_min = 0.20
        self.reference_exclude_presets: set[str] = (
            set()
        )  # Все каналы участвуют в поиске
        self._update_runtime_settings(self.settings)

        self.analyzer: Optional[PhaseAlignmentAnalyzer] = None
        self.is_active = False
        self.reference_channel = None
        self.channels_to_align: List[int] = []
        self.all_channels: List[int] = []
        self.corrections: Dict[int, Dict[str, float]] = {}
        self.last_apply_detail: Dict[int, dict] = {}

        self._apply_once_timer: Optional[threading.Timer] = None
        self._finalize_timer: Optional[threading.Timer] = None
        self._analysis_complete_callback: Optional[
            Callable[[Dict, Dict[int, dict]], None]
        ] = None

        self._candidate_channels: List[int] = []
        self._latest_measurements: Dict = {}
        self._reference_hits: Dict[int, int] = {}
        self._last_coherence: Dict[int, float] = {}
        self._last_gcc_peak: Dict[int, float] = {}
        self._last_spectral_overlap: Dict[int, float] = {}
        self._measurement_frames: int = 0
        self._analysis_started_at: Optional[float] = None
        self._first_measurement_at: Optional[float] = None
        self._last_measurement_at: Optional[float] = None
        self.channel_presets: Dict[int, str] = {}
        self._excluded_by_preset: set[int] = set()

        # Каналы, найденные на предыдущем анализе и ожидающие Apply.
        self._locked_participants: set[int] = set()
        self._last_start_fail_reason: Optional[str] = (
            None  # all_excluded_by_preset | all_locked
        )
        self._pending_participants: set[int] = set()
        self._analysis_detail: Dict[int, dict] = {}

    def _update_runtime_settings(self, settings: Optional[dict]):
        settings = settings or {}
        self.max_delay_ms = float(settings.get("maxDelayMs", self.max_delay_ms))
        self.analysis_window_sec = float(
            settings.get("analysisWindowSec", self.analysis_window_sec)
        )
        self.reference_coherence_min = float(
            settings.get("referenceCoherenceMin", self.reference_coherence_min)
        )
        self.reference_gcc_peak_min = float(
            settings.get("referenceGccPeakMin", self.reference_gcc_peak_min)
        )
        self.reference_min_hits = int(
            settings.get("referenceMinHits", self.reference_min_hits)
        )
        self.reference_spectral_overlap_min = float(
            settings.get(
                "referenceSpectralOverlapMin",
                self.reference_spectral_overlap_min,
            )
        )
        exclude_presets = settings.get("referenceExcludePresets")
        if isinstance(exclude_presets, list) and len(exclude_presets) > 0:
            self.reference_exclude_presets = {
                str(preset).strip().lower()
                for preset in exclude_presets
                if str(preset).strip()
            }
        else:
            self.reference_exclude_presets = set()

    def _is_reference_detected(self, measurement: dict) -> bool:
        """Reference is present if coherence passes threshold."""
        if not isinstance(measurement, dict):
            return False
        coherence_value = self._to_float(measurement.get("coherence", 0.0))
        return coherence_value >= self.reference_coherence_min

    def _process_measurement_frame(self, measurements: Dict):
        """Track per-channel reference hits during 10s analysis window."""
        import re

        if not measurements or not self._candidate_channels:
            return

        now = time.monotonic()
        if self._first_measurement_at is None:
            self._first_measurement_at = now
        self._last_measurement_at = now
        self._measurement_frames += 1
        candidate_set = set(self._candidate_channels)
        for pair_key, measurement in measurements.items():
            if isinstance(pair_key, tuple) and len(pair_key) == 2:
                ref_ch, ch = pair_key
            elif isinstance(pair_key, str):
                match = re.match(r"\((\d+),\s*(\d+)\)", pair_key)
                if not match:
                    continue
                ref_ch, ch = int(match.group(1)), int(match.group(2))
            else:
                continue

            if ref_ch != self.reference_channel or ch not in candidate_set:
                continue

            preset_name = str(self.channel_presets.get(ch, "")).strip().lower()
            if preset_name and preset_name in self.reference_exclude_presets:
                continue

            coherence_value = self._to_float(measurement.get("coherence", 0.0))
            gcc_peak_abs = abs(self._to_float(measurement.get("gcc_peak_value", 0.0)))
            spectral_overlap = self._to_float(
                measurement.get("spectral_overlap_mid_high", 0.0)
            )
            self._last_coherence[ch] = coherence_value
            self._last_gcc_peak[ch] = gcc_peak_abs
            self._last_spectral_overlap[ch] = spectral_overlap

            if self._is_reference_detected(measurement):
                self._reference_hits[ch] = self._reference_hits.get(ch, 0) + 1

    def _filter_measurements_for_channels(
        self, measurements: Dict, channels: set[int]
    ) -> Dict:
        import re

        if not channels:
            return {}

        filtered: Dict = {}
        for pair_key, meas in (measurements or {}).items():
            if isinstance(pair_key, tuple) and len(pair_key) == 2:
                ref_ch, ch = pair_key
            elif isinstance(pair_key, str):
                match = re.match(r"\((\d+),\s*(\d+)\)", pair_key)
                if not match:
                    continue
                ref_ch, ch = int(match.group(1)), int(match.group(2))
            else:
                continue

            if ref_ch != self.reference_channel:
                continue
            if ch in channels:
                filtered[pair_key] = meas
        return filtered

    def _build_analysis_detail(self, participating: set[int]) -> Dict[int, dict]:
        detail: Dict[int, dict] = {}

        if self.reference_channel is not None:
            detail[int(self.reference_channel)] = {
                "detected": True,
                "eligible_for_alignment": len(participating) > 0,
                "ignored_reason": None,
                "status": "ref",
                "coherence": 1.0,
                "gcc_peak_abs": 1.0,
                "reference_hits": 0,
            }

        for ch in self._candidate_channels:
            hits = int(self._reference_hits.get(ch, 0))
            coherence_value = float(self._last_coherence.get(ch, 0.0))
            gcc_peak_abs = float(self._last_gcc_peak.get(ch, 0.0))
            spectral_overlap = float(self._last_spectral_overlap.get(ch, 0.0))
            preset_name = str(self.channel_presets.get(ch, "")).strip().lower()
            is_excluded = (
                preset_name in self.reference_exclude_presets if preset_name else False
            )
            is_participant = ch in participating
            ignored_reason = None
            if not is_participant:
                if is_excluded:
                    ignored_reason = "excluded_preset"
                else:
                    ignored_reason = "not_detected"
            detail[int(ch)] = {
                "detected": is_participant,
                "eligible_for_alignment": is_participant,
                "ignored_reason": ignored_reason,
                "status": "participant" if is_participant else "ignored",
                "coherence": coherence_value,
                "gcc_peak_abs": gcc_peak_abs,
                "spectral_overlap_mid_high": spectral_overlap,
                "reference_hits": hits,
                "preset": preset_name or None,
            }

        for ch in sorted(self._excluded_by_preset):
            if ch == self.reference_channel or ch in detail:
                continue
            preset_name = str(self.channel_presets.get(ch, "")).strip().lower()
            detail[int(ch)] = {
                "detected": False,
                "eligible_for_alignment": False,
                "ignored_reason": "excluded_preset",
                "status": "ignored",
                "coherence": 0.0,
                "gcc_peak_abs": 0.0,
                "spectral_overlap_mid_high": 0.0,
                "reference_hits": 0,
                "preset": preset_name or None,
            }

        for ch in sorted(self._locked_participants):
            if ch == self.reference_channel:
                continue
            if ch not in detail:
                detail[int(ch)] = {
                    "detected": True,
                    "eligible_for_alignment": True,
                    "ignored_reason": None,
                    "status": "participant",
                    "coherence": 0.0,
                    "gcc_peak_abs": 0.0,
                    "reference_hits": 0,
                }
        return detail

    def _finalize_analysis_window(self):
        """
        Завершение 10-секундного окна:
        1) фиксируем каналы с найденным референсным сигналом,
        2) фильтруем измерения только для них,
        3) останавливаем анализ и отдаём статус в UI.
        """
        if not self.is_active:
            return

        try:
            raw_measurements = self.analyzer.get_measurements() if self.analyzer else {}

            participating = {
                ch
                for ch in self._candidate_channels
                if float(self._last_coherence.get(ch, 0.0))
                >= self.reference_coherence_min
            }

            self._pending_participants = set(participating)
            self._locked_participants.update(participating)

            filtered = self._filter_measurements_for_channels(
                raw_measurements, participating
            )
            self._latest_measurements = filtered
            self.channels_to_align = sorted(participating)
            self._analysis_detail = self._build_analysis_detail(participating)
            self.last_apply_detail = self._analysis_detail.copy()

            logger.info(
                "Phase analyze finalized: participants=%s, candidates=%s, frames=%s, coherence_min=%.2f",
                sorted(participating),
                self._candidate_channels,
                self._measurement_frames,
                self.reference_coherence_min,
            )
            if self._measurement_frames < 20:
                logger.warning(
                    "Phase analyze collected too few frames: %s for %.1fs window",
                    self._measurement_frames,
                    self.analysis_window_sec,
                )
            if (
                self._first_measurement_at is not None
                and self._last_measurement_at is not None
                and self._measurement_frames > 1
            ):
                span = self._last_measurement_at - self._first_measurement_at
                avg_interval = span / max(self._measurement_frames - 1, 1)
                logger.info(
                    "Phase analyze timing: first_data_after=%.3fs avg_interval=%.3fs total_span=%.3fs",
                    (self._first_measurement_at - self._analysis_started_at)
                    if self._analysis_started_at is not None
                    else -1.0,
                    avg_interval,
                    span,
                )
        except Exception as e:
            logger.error("Error finalizing phase analysis window: %s", e, exc_info=True)
            self._latest_measurements = {}
            self._analysis_detail = self._build_analysis_detail(set())
            self.last_apply_detail = self._analysis_detail.copy()
        finally:
            self._finalize_timer = None
            self.stop_analysis()
            if self._analysis_complete_callback:
                try:
                    self._analysis_complete_callback(
                        self._latest_measurements.copy(),
                        self._analysis_detail.copy(),
                    )
                except Exception as e:
                    logger.error("on_analysis_complete callback failed: %s", e)

    def start_analysis(
        self,
        device_id: int,
        reference_channel: int,
        channels: List[int],
        channel_presets: Optional[Dict[int, str]] = None,
        on_measurement_callback: Callable = None,
        apply_once: bool = False,
        apply_once_duration_sec: float = 5.0,
        settings: dict = None,
        on_analysis_complete_callback: Callable = None,
    ):
        """
        Запуск анализа фаз и задержек.

        В обычном режиме (apply_once=False):
        - сначала идет 10-секундный поиск референсного сигнала в других каналах,
        - затем фиксируются каналы "участвует",
        - и выдаются измерения только для этих каналов.
        """
        if self.is_active:
            logger.warning("Phase alignment already active")
            return False

        if self._apply_once_timer:
            self._apply_once_timer.cancel()
            self._apply_once_timer = None
        if self._finalize_timer:
            self._finalize_timer.cancel()
            self._finalize_timer = None

        settings = settings or self.settings or {}
        self.settings = settings
        self._update_runtime_settings(settings)
        self._analysis_complete_callback = on_analysis_complete_callback

        self.reference_channel = reference_channel
        self.channel_presets = {
            int(ch): str(preset)
            for ch, preset in (channel_presets or {}).items()
            if str(ch).isdigit()
        }
        requested_channels = [ch for ch in channels if ch != reference_channel]
        self._excluded_by_preset = set()
        filtered_channels = list(requested_channels)
        if self.reference_exclude_presets:
            filtered_channels = []
            for ch in requested_channels:
                preset_name = str(self.channel_presets.get(ch, "")).strip().lower()
                if preset_name and preset_name in self.reference_exclude_presets:
                    self._excluded_by_preset.add(ch)
                    logger.info(
                        "Skipping channel %s from reference search due to preset '%s'",
                        ch,
                        preset_name,
                    )
                    continue
                filtered_channels.append(ch)
        self._candidate_channels = [
            ch for ch in filtered_channels if ch not in self._locked_participants
        ]
        self.all_channels = [reference_channel] + requested_channels
        self.channels_to_align = self._candidate_channels.copy()

        self._latest_measurements = {}
        self._reference_hits = {}
        self._last_coherence = {}
        self._last_gcc_peak = {}
        self._last_spectral_overlap = {}
        self._measurement_frames = 0
        self._analysis_started_at = time.monotonic()
        self._first_measurement_at = None
        self._last_measurement_at = None
        self._analysis_detail = self._build_analysis_detail(set())
        self.last_apply_detail = self._analysis_detail.copy()

        if not self._candidate_channels:
            if self._excluded_by_preset and not filtered_channels:
                self._last_start_fail_reason = "all_excluded_by_preset"
                logger.info(
                    "No channels for phase analyze: all excluded by preset (vocal/bass/playback/etc): %s",
                    sorted(self._excluded_by_preset),
                )
            else:
                self._last_start_fail_reason = "all_locked"
                logger.info(
                    "No channels left for phase analyze (all pending Apply): requested=%s locked=%s",
                    requested_channels,
                    sorted(self._locked_participants),
                )
            return False

        self.analyzer = PhaseAlignmentAnalyzer(
            device_index=device_id,
            max_delay_ms=self.max_delay_ms,
            chunk_size=settings.get("fftSize", 4096),
        )

        def measurement_callback(measurements):
            self._latest_measurements = (
                measurements.copy() if isinstance(measurements, dict) else {}
            )
            if isinstance(measurements, dict):
                self._process_measurement_frame(measurements)
            if on_measurement_callback:
                on_measurement_callback(measurements)

        success = self.analyzer.start(
            reference_channel=reference_channel,
            channels=self._candidate_channels,
            on_measurement_callback=measurement_callback,
        )

        if success:
            self._last_start_fail_reason = None
            self.is_active = True
            logger.info(
                "Phase alignment started: ref=%s candidates=%s apply_once=%s",
                reference_channel,
                self._candidate_channels,
                apply_once,
            )

            if apply_once:
                # Режим саундчека сохраняем без изменений.
                def _after_duration():
                    if not self.analyzer or not self.is_active:
                        return
                    measurements = self.analyzer.get_measurements()
                    self.stop_analysis()
                    if measurements and self.mixer_client:
                        self.apply_corrections(measurements)
                        logger.info("Phase alignment apply_once: corrections applied")
                    self._apply_once_timer = None

                self._apply_once_timer = threading.Timer(
                    apply_once_duration_sec, _after_duration
                )
                self._apply_once_timer.daemon = True
                self._apply_once_timer.start()
                logger.info(
                    "Phase alignment apply_once: will stop and apply after %.1fs",
                    apply_once_duration_sec,
                )
            else:
                self._finalize_timer = threading.Timer(
                    self.analysis_window_sec,
                    self._finalize_analysis_window,
                )
                self._finalize_timer.daemon = True
                self._finalize_timer.start()
                logger.info(
                    "Phase alignment reference search started for %.1fs "
                    "(coherence>=%.2f)",
                    self.analysis_window_sec,
                    self.reference_coherence_min,
                )

        return success

    def stop_analysis(self):
        """Остановка анализа."""
        if not self.is_active:
            return
        if self._apply_once_timer:
            self._apply_once_timer.cancel()
            self._apply_once_timer = None
        if self._finalize_timer:
            self._finalize_timer.cancel()
            self._finalize_timer = None
        if self.analyzer:
            self.analyzer.stop()
            self.analyzer = None
        self.is_active = False
        logger.info("Phase alignment analysis stopped")

    def _to_float(self, v) -> float:
        """Convert numpy scalar to native Python float for JSON."""
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, (np.integer, np.floating)):
            v = float(v)
        elif v is None:
            return 0.0
        else:
            v = float(v)
        return float(v) if np.isfinite(v) else 0.0

    def apply_corrections(self, measurements: Dict = None):
        """
        Применение коррекций delay/phase по новой логике:
        - Участвуют только каналы с delay_gcc_ms <= max_delay_ms (когерентность не учитывается)
        - Нормализация к самому позднему каналу: reference получает max_delay, остальные max_delay - delay_gcc
        - Исключённые каналы не корректируются
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

        if not isinstance(measurements, dict) or len(measurements) == 0:
            logger.warning("Measurements empty or invalid")
            return False

        import re

        ref_ch = self.reference_channel
        max_delay_ms_setting = getattr(self, "max_delay_ms", 10.0)

        # 0. Только каналы со статусом «Участвует» — применяем к ним
        participating = self._pending_participants
        if not participating and self._latest_measurements:
            participating = set()
            for pk in self._latest_measurements:
                if isinstance(pk, tuple) and len(pk) == 2:
                    _, ch = pk
                    participating.add(ch)
                elif isinstance(pk, str):
                    m = re.match(r"\((\d+),\s*(\d+)\)", pk)
                    if m:
                        participating.add(int(m.group(2)))
        if not participating:
            logger.warning(
                "No participating channels (run analysis and wait for completion)"
            )
            return False
        logger.info(
            "Applying only to participating channels: %s", sorted(participating)
        )

        # 1. Parse measurements into per-channel data (только participating)
        channel_data: Dict[int, dict] = {}
        for pair_key, meas in measurements.items():
            if isinstance(pair_key, tuple) and len(pair_key) == 2:
                r, ch = pair_key
            elif isinstance(pair_key, str):
                m = re.match(r"\((\d+),\s*(\d+)\)", pair_key)
                if m:
                    r, ch = int(m.group(1)), int(m.group(2))
                else:
                    continue
            else:
                continue
            if r != ref_ch or not isinstance(meas, dict):
                continue
            if participating and ch not in participating:
                continue

            delay_gcc = self._to_float(meas.get("delay_gcc_ms", 0))
            effective_delay = abs(delay_gcc)
            eligible = effective_delay <= max_delay_ms_setting

            channel_data[ch] = {
                "delay_gcc_ms": delay_gcc,
                "effective_delay_ms": effective_delay,
                "phase_invert": int(meas.get("phase_invert", 0)),
                "coherence": self._to_float(meas.get("coherence", 0)),
                "eligible_for_alignment": eligible,
                "ignored_reason": None if eligible else "delay_above_10ms",
                "detected": True,  # coherence no longer used for eligibility; kept for UI
            }

        # 2. Eligible channels and max_delay_ms
        eligible_channels = [
            ch for ch, d in channel_data.items() if d["eligible_for_alignment"]
        ]
        max_delay_ms = max(
            (channel_data[ch]["effective_delay_ms"] for ch in eligible_channels),
            default=0.0,
        )

        # 3. Compute applied_delay_ms for each channel
        for ch, d in channel_data.items():
            if d["eligible_for_alignment"]:
                d["applied_delay_ms"] = max(0.0, max_delay_ms - d["effective_delay_ms"])
                d["phase_applied"] = True
            else:
                d["applied_delay_ms"] = 0.0
                d["phase_applied"] = False

        # 4. Reference channel: delay = max_delay_ms (or 0 if no eligible)
        ref_applied_delay = max_delay_ms if eligible_channels else 0.0

        logger.info("Applying phase/delay corrections (normalize-to-latest logic)...")
        logger.info(
            f"Eligible channels: {eligible_channels}, max_delay_ms={max_delay_ms:.2f}"
        )

        self.corrections.clear()
        applied_count = 0

        # 5. Apply to reference channel
        if ref_ch and ref_applied_delay >= 0.3:
            self.corrections[ref_ch] = {
                "delay_ms": ref_applied_delay,
                "phase_invert": 0,
                "eligible_for_alignment": True,
                "applied_delay_ms": ref_applied_delay,
            }
            delay_val = max(0.5, min(max_delay_ms_setting, ref_applied_delay))
            logger.info(f"Reference ch {ref_ch}: delay={delay_val:.2f} ms")
            self.mixer_client.set_channel_delay(ref_ch, delay_val, mode="MS")
            self.mixer_client.set_channel_phase_invert(ref_ch, 0)
            applied_count += 1
        elif ref_ch:
            logger.info(f"Reference ch {ref_ch}: delay=0 ms (no eligible channels)")
            is_ableton = type(self.mixer_client).__name__ == "AbletonClient"
            if is_ableton:
                self.mixer_client.set_channel_delay(ref_ch, 0.0, mode="MS")
            else:
                self.mixer_client.send(f"/ch/{ref_ch}/in/set/dlyon", 0)
                self.mixer_client.send(f"/ch/{ref_ch}/in/set/dlymode", "MS")
                self.mixer_client.send(f"/ch/{ref_ch}/in/set/dly", 0.5)
            self.mixer_client.set_channel_phase_invert(ref_ch, 0)

        # 6. Apply to non-reference channels
        for ch, d in channel_data.items():
            if not d["eligible_for_alignment"]:
                logger.info(f"Channel {ch}: ignored ({d['ignored_reason']})")
                continue

            applied_delay = d["applied_delay_ms"]
            phase_invert = d["phase_invert"]

            self.corrections[ch] = {
                "delay_ms": applied_delay,
                "phase_invert": phase_invert,
                "eligible_for_alignment": True,
                "applied_delay_ms": applied_delay,
                "ignored_reason": None,
            }

            if applied_delay >= 0.3:
                delay_val = max(0.5, min(max_delay_ms_setting, applied_delay))
                logger.info(
                    f"Channel {ch}: delay={delay_val:.2f} ms, phase_invert={phase_invert}"
                )
                self.mixer_client.set_channel_delay(ch, delay_val, mode="MS")
            else:
                logger.info(
                    f"Channel {ch}: delay=0 (latest), phase_invert={phase_invert}"
                )
                is_ableton = type(self.mixer_client).__name__ == "AbletonClient"
                if is_ableton:
                    self.mixer_client.set_channel_delay(ch, 0.0, mode="MS")
                else:
                    self.mixer_client.send(f"/ch/{ch}/in/set/dlyon", 0)

            self.mixer_client.set_channel_phase_invert(ch, phase_invert)
            applied_count += 1

        # Store full result for UI (all channels including excluded)
        self.last_apply_detail.clear()
        if ref_ch:
            self.last_apply_detail[ref_ch] = {
                "detected": True,
                "eligible_for_alignment": len(eligible_channels) > 0,
                "ignored_reason": None,
                "applied_delay_ms": ref_applied_delay,
                "phase_applied": False,
                "delay_gcc_ms": 0.0,
            }
        for ch, d in channel_data.items():
            self.last_apply_detail[ch] = {
                "detected": d.get("detected", True),
                "eligible_for_alignment": d["eligible_for_alignment"],
                "ignored_reason": d["ignored_reason"],
                "applied_delay_ms": d["applied_delay_ms"],
                "phase_applied": d["phase_applied"],
                "delay_gcc_ms": d["delay_gcc_ms"],
            }

        logger.info(f"Applied corrections to {applied_count} channels")

        # После нажатия Apply разрешаем новый полный анализ по всем каналам.
        self._pending_participants.clear()
        self._locked_participants.clear()
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
            channels = (
                list(self.corrections.keys())
                if self.corrections
                else self.channels_to_align
            )

        if not channels:
            logger.warning("No channels specified for reset")
            return False

        logger.info(f"Resetting phase and delay for {len(channels)} channels...")

        reset_count = 0

        for ch in channels:
            try:
                self.mixer_client.set_channel_phase_invert(ch, 0)
                # Delay reset: Ableton uses set_channel_delay(0), Wing uses dlyon/dly
                is_ableton = type(self.mixer_client).__name__ == "AbletonClient"
                if is_ableton:
                    self.mixer_client.set_channel_delay(ch, 0.0, mode="MS")
                else:
                    self.mixer_client.send(f"/ch/{ch}/in/set/dlyon", 0)
                    self.mixer_client.send(f"/ch/{ch}/in/set/dlymode", "MS")
                    self.mixer_client.send(f"/ch/{ch}/in/set/dly", 0.5)
                if ch in self.corrections:
                    del self.corrections[ch]
                reset_count += 1
                logger.info(f"Channel {ch}: Reset phase and delay")
            except Exception as e:
                logger.error(f"Error resetting channel {ch}: {e}")

        logger.info(f"Reset phase and delay for {reset_count} channels")
        self._pending_participants.clear()
        self._locked_participants.clear()
        self._analysis_detail.clear()
        return True

    def _serialize_measurements_for_json(self, measurements: Dict) -> Dict:
        """Сериализация measurements для JSON (tuple keys, numpy types)."""
        if not measurements:
            return {}
        out = {}
        for key, value in measurements.items():
            if isinstance(key, tuple) and len(key) == 2:
                sk = f"({key[0]}, {key[1]})"
            else:
                sk = str(key)
            if isinstance(value, dict):
                sv = {}
                for k, v in value.items():
                    if hasattr(v, "item"):
                        sv[k] = v.item()
                    elif isinstance(v, (np.integer, np.floating)):
                        sv[k] = float(v) if isinstance(v, np.floating) else int(v)
                    elif isinstance(v, (list, tuple)):
                        sv[k] = [
                            x.item()
                            if hasattr(x, "item")
                            else (
                                float(x)
                                if isinstance(x, np.floating)
                                else int(x)
                                if isinstance(x, np.integer)
                                else x
                            )
                            for x in v
                        ]
                    else:
                        sv[k] = v
                out[sk] = sv
            else:
                out[sk] = value
        return out

    def get_status(self) -> Dict:
        """Получение статуса контроллера."""
        return {
            "active": self.is_active,
            "reference_channel": self.reference_channel,
            "channels": self.channels_to_align,
            "corrections": self.corrections.copy(),
            "detail": {int(k): v for k, v in self.last_apply_detail.items()},
            "measurements": self._serialize_measurements_for_json(
                self._latest_measurements
            ),
            "locked_participants": sorted(self._locked_participants),
            "analysis_window_sec": self.analysis_window_sec,
            "reference_coherence_min": self.reference_coherence_min,
            "reference_gcc_peak_min": self.reference_gcc_peak_min,
            "reference_min_hits": self.reference_min_hits,
            "reference_spectral_overlap_min": self.reference_spectral_overlap_min,
        }
