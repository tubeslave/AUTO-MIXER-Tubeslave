"""
ML-based channel/instrument classifier using MFCC + spectral features.

Uses a RandomForest classifier trained on audio features to identify
instrument types from live audio channels. Falls back to channel name
matching when ML confidence is low.
"""

import numpy as np
import os
import re
import logging

logger = logging.getLogger(__name__)

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import joblib

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Instrument classes
INSTRUMENT_CLASSES = [
    "kick",
    "snare",
    "hihat",
    "toms",
    "overheads",
    "bass_guitar",
    "electric_guitar",
    "acoustic_guitar",
    "keys",
    "vocals",
    "brass",
    "strings",
    "percussion",
]

# Channel name patterns for fallback classification
_NAME_PATTERNS = {
    "kick": re.compile(r"\b(kick|kik|bd|bass\s*drum)\b", re.IGNORECASE),
    "snare": re.compile(r"\b(snare|snr|sd)\b", re.IGNORECASE),
    "hihat": re.compile(r"\b(hi[\s\-]?hat|hh|hat)\b", re.IGNORECASE),
    "toms": re.compile(r"\b(tom|floor|rack)\b", re.IGNORECASE),
    "overheads": re.compile(r"\b(oh|overhead|over\s*head|cymbal|ride|crash)\b", re.IGNORECASE),
    "bass_guitar": re.compile(r"\b(bass|bg|b\.?\s*gtr|bass\s*gtr|bass\s*guitar|di\s*bass)\b", re.IGNORECASE),
    "electric_guitar": re.compile(
        r"\b(e[\s\-]?gtr|elec.*gtr|electric.*guitar|gtr|guitar|lead\s*gtr|rhythm\s*gtr)\b",
        re.IGNORECASE,
    ),
    "acoustic_guitar": re.compile(
        r"\b(a[\s\-]?gtr|acou.*gtr|acoustic.*guitar|ac.*guitar|nylon)\b", re.IGNORECASE
    ),
    "keys": re.compile(
        r"\b(keys|key|piano|pno|organ|synth|rhodes|wurli|clav)\b", re.IGNORECASE
    ),
    "vocals": re.compile(
        r"\b(vox|vocal|voc|voice|sing|lead\s*voc|bgv|bv|back.*voc|choir|harmony)\b",
        re.IGNORECASE,
    ),
    "brass": re.compile(
        r"\b(brass|trumpet|trpt|trombone|tbn|sax|horn|flugelhorn)\b", re.IGNORECASE
    ),
    "strings": re.compile(
        r"\b(string|violin|viola|cello|fiddle|str)\b", re.IGNORECASE
    ),
    "percussion": re.compile(
        r"\b(perc|percussion|conga|bongo|tambourine|shaker|cabasa|triangle|timbale|djembe|cajon)\b",
        re.IGNORECASE,
    ),
}


def extract_features(audio, sr=48000):
    """
    Extract audio features for classification.

    Features extracted:
    - 13 MFCCs (mean + std = 26 features)
    - Spectral centroid (mean + std = 2)
    - Spectral rolloff (mean + std = 2)
    - Spectral flux / onset strength (mean + std = 2)
    - Zero crossing rate (mean + std = 2)
    - RMS energy (mean + std = 2)
    Total: 36 features

    Args:
        audio: 1D numpy array of audio samples
        sr: sample rate

    Returns:
        feature_vector: 1D numpy array of shape (36,)
    """
    # Ensure mono float
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    features = []

    if HAS_LIBROSA:
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))  # 13
        features.extend(np.std(mfccs, axis=1))  # 13

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features.append(np.mean(centroid))
        features.append(np.std(centroid))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))

        # Onset strength (spectral flux proxy)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features.append(np.mean(onset_env))
        features.append(np.std(onset_env))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features.append(np.mean(rms))
        features.append(np.std(rms))
    else:
        # Numpy-only fallback feature extraction
        n_fft = 2048
        hop = 512
        n_frames = max(1, (len(audio) - n_fft) // hop + 1)

        # Compute short-time spectrogram
        spectrogram = np.zeros((n_fft // 2 + 1, n_frames))
        window = np.hanning(n_fft)
        for i in range(n_frames):
            start = i * hop
            frame = audio[start: start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            windowed = frame * window
            spectrum = np.abs(np.fft.rfft(windowed))
            spectrogram[:, i] = spectrum

        # Pseudo-MFCCs using DCT of log-mel approximation
        n_mels = 40
        n_mfcc = 13
        mel_freqs = librosa_mel_frequencies(n_mels + 2, fmin=0, fmax=sr / 2)
        mel_filters = _create_mel_filterbank(sr, n_fft, n_mels, mel_freqs)
        mel_spec = np.dot(mel_filters, spectrogram)
        log_mel = np.log(mel_spec + 1e-8)
        # DCT-II approximation
        mfccs = np.zeros((n_mfcc, n_frames))
        for k in range(n_mfcc):
            for n_idx in range(n_mels):
                mfccs[k] += log_mel[n_idx] * np.cos(
                    np.pi * k * (2 * n_idx + 1) / (2 * n_mels)
                )
        features.extend(np.mean(mfccs, axis=1))  # 13
        features.extend(np.std(mfccs, axis=1))  # 13

        # Spectral centroid
        freq_bins = np.arange(spectrogram.shape[0]) * sr / n_fft
        total_energy = np.sum(spectrogram, axis=0) + 1e-8
        centroid = np.sum(spectrogram * freq_bins[:, np.newaxis], axis=0) / total_energy
        features.append(np.mean(centroid))
        features.append(np.std(centroid))

        # Spectral rolloff (85th percentile)
        cumsum = np.cumsum(spectrogram, axis=0)
        threshold = 0.85 * total_energy
        rolloff = np.zeros(n_frames)
        for i in range(n_frames):
            idx = np.searchsorted(cumsum[:, i], threshold[i])
            rolloff[i] = idx * sr / n_fft
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))

        # Spectral flux
        flux = np.zeros(n_frames)
        for i in range(1, n_frames):
            flux[i] = np.sum((spectrogram[:, i] - spectrogram[:, i - 1]) ** 2)
        flux = np.sqrt(flux)
        features.append(np.mean(flux))
        features.append(np.std(flux))

        # Zero crossing rate
        zcr_vals = []
        for i in range(n_frames):
            start = i * hop
            end = min(start + n_fft, len(audio))
            frame = audio[start:end]
            if len(frame) > 1:
                zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * len(frame))
            else:
                zcr = 0.0
            zcr_vals.append(zcr)
        features.append(np.mean(zcr_vals))
        features.append(np.std(zcr_vals))

        # RMS energy
        rms_vals = []
        for i in range(n_frames):
            start = i * hop
            end = min(start + n_fft, len(audio))
            frame = audio[start:end]
            rms_vals.append(np.sqrt(np.mean(frame ** 2) + 1e-8))
        features.append(np.mean(rms_vals))
        features.append(np.std(rms_vals))

    return np.array(features, dtype=np.float32)


def librosa_mel_frequencies(n_mels, fmin=0.0, fmax=11025.0):
    """Compute mel frequency points (utility for numpy fallback)."""

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels)
    return mel_to_hz(mels)


def _create_mel_filterbank(sr, n_fft, n_mels, mel_freqs):
    """Create triangular mel filterbank matrix."""
    fft_freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        f_left = mel_freqs[i]
        f_center = mel_freqs[i + 1] if i + 1 < len(mel_freqs) else mel_freqs[i]
        f_right = mel_freqs[i + 2] if i + 2 < len(mel_freqs) else mel_freqs[i + 1] if i + 1 < len(mel_freqs) else mel_freqs[i]

        if f_center == f_left:
            f_center = f_left + 1.0
        if f_right == f_center:
            f_right = f_center + 1.0

        # Rising slope
        mask_rise = (fft_freqs >= f_left) & (fft_freqs <= f_center)
        filterbank[i, mask_rise] = (fft_freqs[mask_rise] - f_left) / (f_center - f_left + 1e-8)

        # Falling slope
        mask_fall = (fft_freqs > f_center) & (fft_freqs <= f_right)
        filterbank[i, mask_fall] = (f_right - fft_freqs[mask_fall]) / (f_right - f_center + 1e-8)

    return filterbank


def classify_from_name(channel_name):
    """
    Classify instrument type from channel name string.

    Args:
        channel_name: string name of the mixer channel

    Returns:
        (class_name, confidence) or (None, 0.0) if no match
    """
    if not channel_name:
        return None, 0.0

    # Check electric_guitar BEFORE bass_guitar to avoid 'guitar' matching bass patterns
    # But check bass first since it's more specific
    priority_order = [
        "kick", "snare", "hihat", "toms", "overheads",
        "bass_guitar", "acoustic_guitar", "electric_guitar",
        "keys", "vocals", "brass", "strings", "percussion",
    ]
    for cls in priority_order:
        pattern = _NAME_PATTERNS[cls]
        if pattern.search(channel_name):
            # Higher confidence for exact/common matches
            return cls, 0.85

    return None, 0.0


class ChannelClassifier:
    """
    ML-based channel/instrument classifier.

    Uses a RandomForest trained on MFCC + spectral features to
    classify audio channels into one of 13 instrument types.
    Falls back to channel name pattern matching when ML is unavailable
    or confidence is low.
    """

    def __init__(self, model_path=None):
        """
        Args:
            model_path: optional path to pre-trained model (.joblib)
        """
        self.model = None
        self.scaler = None
        self.classes = INSTRUMENT_CLASSES

        if model_path and os.path.exists(model_path):
            self.load(model_path)
        elif HAS_SKLEARN:
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            self.scaler = StandardScaler()
            self._is_fitted = False
        else:
            self._is_fitted = False

    def train(self, X, y):
        """
        Train the classifier on feature matrix and labels.

        Args:
            X: (n_samples, n_features) feature matrix
            y: (n_samples,) integer class labels or string labels

        Returns:
            accuracy: training accuracy (float)
        """
        if not HAS_SKLEARN:
            raise RuntimeError("sklearn required for training. Install scikit-learn.")

        X = np.asarray(X, dtype=np.float32)

        # Convert string labels to indices if needed
        if isinstance(y[0], str):
            class_to_idx = {c: i for i, c in enumerate(self.classes)}
            y = np.array([class_to_idx.get(label, 0) for label in y])
        else:
            y = np.asarray(y, dtype=np.int64)

        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X)

        # Train
        self.model.fit(X_scaled, y)
        self._is_fitted = True

        # Return training accuracy
        predictions = self.model.predict(X_scaled)
        accuracy = np.mean(predictions == y)
        logger.info(f"ChannelClassifier trained: accuracy={accuracy:.4f}")
        return accuracy

    def classify(self, audio, sr=48000):
        """
        Classify an audio segment into an instrument type.

        Args:
            audio: 1D numpy array of audio samples
            sr: sample rate

        Returns:
            (class_name, confidence): tuple of string class and float confidence
        """
        if not HAS_SKLEARN or self.model is None or not getattr(self, "_is_fitted", False):
            # Return a heuristic classification based on spectral content
            return self._heuristic_classify(audio, sr)

        features = extract_features(audio, sr)
        features = features.reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        # Get class probabilities
        proba = self.model.predict_proba(features_scaled)[0]
        class_idx = np.argmax(proba)
        confidence = float(proba[class_idx])
        class_name = self.classes[class_idx]

        return class_name, confidence

    def classify_with_fallback(self, audio, sr=48000, channel_name=None):
        """
        Classify using ML first, fall back to name-based recognition.

        If ML confidence is below threshold (0.5), use channel name
        pattern matching. If both are available, prefer the higher
        confidence result.

        Args:
            audio: 1D numpy array
            sr: sample rate
            channel_name: optional channel name string

        Returns:
            (class_name, confidence): best classification result
        """
        ml_class, ml_conf = self.classify(audio, sr)
        name_class, name_conf = classify_from_name(channel_name)

        # If ML is confident enough, use it
        if ml_conf >= 0.5:
            return ml_class, ml_conf

        # If name matching found something, use it
        if name_class is not None:
            return name_class, name_conf

        # Fall back to ML result even with low confidence
        return ml_class, ml_conf

    def _heuristic_classify(self, audio, sr):
        """
        Simple heuristic classification based on spectral content.
        Used when sklearn is not available.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < 256:
            return "vocals", 0.3  # default guess

        # Compute spectrum
        n_fft = min(4096, len(audio))
        spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        total_energy = np.sum(spectrum ** 2) + 1e-8

        # Energy in frequency bands
        sub_bass = np.sum(spectrum[(freqs >= 20) & (freqs < 100)] ** 2) / total_energy
        bass = np.sum(spectrum[(freqs >= 100) & (freqs < 300)] ** 2) / total_energy
        low_mid = np.sum(spectrum[(freqs >= 300) & (freqs < 1000)] ** 2) / total_energy
        mid = np.sum(spectrum[(freqs >= 1000) & (freqs < 4000)] ** 2) / total_energy
        high = np.sum(spectrum[(freqs >= 4000) & (freqs < 10000)] ** 2) / total_energy
        air = np.sum(spectrum[(freqs >= 10000)] ** 2) / total_energy

        # Zero crossing rate (transient detection)
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0

        # Crest factor (peak-to-RMS ratio, high for drums)
        rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
        crest = np.max(np.abs(audio)) / (rms + 1e-8)

        # Decision tree
        if sub_bass > 0.5 and crest > 5.0:
            return "kick", 0.6
        elif sub_bass > 0.3 and bass > 0.2:
            return "bass_guitar", 0.5
        elif high > 0.3 and crest > 8.0:
            return "hihat", 0.5
        elif mid > 0.15 and crest > 6.0 and high > 0.1:
            return "snare", 0.5
        elif bass > 0.2 and crest > 4.0:
            return "toms", 0.4
        elif air > 0.1 and high > 0.2:
            return "overheads", 0.4
        elif low_mid > 0.25 and mid > 0.2 and zcr > 0.1:
            return "vocals", 0.5
        elif bass > 0.15 and low_mid > 0.2 and mid > 0.15:
            return "electric_guitar", 0.4
        elif mid > 0.2 and high > 0.15:
            return "keys", 0.4
        elif low_mid > 0.2 and mid > 0.15 and high > 0.1:
            return "brass", 0.35
        elif mid > 0.25 and zcr < 0.05:
            return "strings", 0.35
        elif low_mid > 0.2:
            return "acoustic_guitar", 0.3
        else:
            return "vocals", 0.3

    def save(self, path):
        """Save the trained model to disk."""
        if not HAS_JOBLIB:
            raise RuntimeError("joblib required for saving. Install joblib.")
        if self.model is None:
            raise RuntimeError("No model to save.")

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "classes": self.classes,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(data, path)
        logger.info(f"ChannelClassifier saved to {path}")

    def load(self, path):
        """Load a trained model from disk."""
        if not HAS_JOBLIB:
            raise RuntimeError("joblib required for loading. Install joblib.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.classes = data.get("classes", INSTRUMENT_CLASSES)
        self._is_fitted = True
        logger.info(f"ChannelClassifier loaded from {path}")
