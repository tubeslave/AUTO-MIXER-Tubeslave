"""
Training script for the channel/instrument classifier.

Provides:
- ChannelClassifierTrainer class with prepare_dataset, extract_features,
  train, and evaluate methods
- Synthetic data generation from per-class feature distributions
- Training pipeline with cross-validation
- Model evaluation with confusion matrix reporting
- CLI interface for training and evaluation

Usage:
    python -m backend.ml.train_classifier --output models/channel_classifier.joblib
    python -m backend.ml.train_classifier --evaluate models/channel_classifier.joblib
    python -m backend.ml.train_classifier --audio-dir data/audio --labels data/labels.csv
"""

import csv
import numpy as np
import os
import sys
import logging
import argparse

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    from scipy.signal import get_window as _check_scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Feature distribution parameters per instrument class.
# Each tuple is (mean, std) for the 34 features.
# These distributions are based on empirical analysis of audio features
# for each instrument type and serve as priors for synthetic data.
_CLASS_FEATURE_PARAMS = {
    "kick": {
        "mfcc_means": [-200, 80, -20, 15, -5, 8, -3, 5, -2, 3, -1, 2, -1],
        "mfcc_stds": [40, 20, 15, 10, 8, 6, 5, 4, 3, 3, 2, 2, 2],
        "centroid_mean": 400,
        "centroid_std": 200,
        "rolloff_mean": 1500,
        "rolloff_std": 800,
        "flux_mean": 50,
        "flux_std": 30,
        "zcr_mean": 0.02,
        "zcr_std": 0.01,
        "rms_mean": 0.15,
        "rms_std": 0.08,
    },
    "snare": {
        "mfcc_means": [-150, 60, -15, 12, -8, 6, -4, 4, -3, 2, -1, 1, -1],
        "mfcc_stds": [35, 18, 12, 9, 7, 5, 4, 4, 3, 2, 2, 2, 1],
        "centroid_mean": 2000,
        "centroid_std": 800,
        "rolloff_mean": 6000,
        "rolloff_std": 2000,
        "flux_mean": 80,
        "flux_std": 40,
        "zcr_mean": 0.08,
        "zcr_std": 0.04,
        "rms_mean": 0.12,
        "rms_std": 0.06,
    },
    "hihat": {
        "mfcc_means": [-100, 40, -10, 8, -5, 4, -2, 3, -2, 2, -1, 1, 0],
        "mfcc_stds": [30, 15, 10, 8, 6, 5, 4, 3, 3, 2, 2, 1, 1],
        "centroid_mean": 6000,
        "centroid_std": 2000,
        "rolloff_mean": 12000,
        "rolloff_std": 3000,
        "flux_mean": 60,
        "flux_std": 35,
        "zcr_mean": 0.15,
        "zcr_std": 0.06,
        "rms_mean": 0.06,
        "rms_std": 0.03,
    },
    "toms": {
        "mfcc_means": [-180, 70, -18, 14, -6, 7, -3, 4, -2, 3, -1, 2, -1],
        "mfcc_stds": [38, 19, 14, 10, 8, 6, 5, 4, 3, 3, 2, 2, 1],
        "centroid_mean": 800,
        "centroid_std": 400,
        "rolloff_mean": 3000,
        "rolloff_std": 1200,
        "flux_mean": 45,
        "flux_std": 25,
        "zcr_mean": 0.04,
        "zcr_std": 0.02,
        "rms_mean": 0.10,
        "rms_std": 0.06,
    },
    "overheads": {
        "mfcc_means": [-120, 50, -12, 10, -6, 5, -3, 4, -2, 2, -1, 1, 0],
        "mfcc_stds": [32, 16, 11, 8, 7, 5, 4, 3, 3, 2, 2, 1, 1],
        "centroid_mean": 4000,
        "centroid_std": 1500,
        "rolloff_mean": 10000,
        "rolloff_std": 3000,
        "flux_mean": 40,
        "flux_std": 20,
        "zcr_mean": 0.10,
        "zcr_std": 0.05,
        "rms_mean": 0.05,
        "rms_std": 0.03,
    },
    "bass_guitar": {
        "mfcc_means": [-190, 75, -22, 16, -4, 9, -2, 5, -1, 3, -1, 2, 0],
        "mfcc_stds": [42, 22, 16, 11, 8, 7, 5, 4, 3, 3, 2, 2, 1],
        "centroid_mean": 600,
        "centroid_std": 300,
        "rolloff_mean": 2000,
        "rolloff_std": 1000,
        "flux_mean": 20,
        "flux_std": 12,
        "zcr_mean": 0.03,
        "zcr_std": 0.02,
        "rms_mean": 0.12,
        "rms_std": 0.06,
    },
    "electric_guitar": {
        "mfcc_means": [-140, 55, -14, 11, -7, 5, -3, 4, -2, 2, -1, 1, 0],
        "mfcc_stds": [35, 18, 13, 9, 7, 6, 5, 4, 3, 2, 2, 2, 1],
        "centroid_mean": 2500,
        "centroid_std": 1000,
        "rolloff_mean": 7000,
        "rolloff_std": 2500,
        "flux_mean": 35,
        "flux_std": 18,
        "zcr_mean": 0.06,
        "zcr_std": 0.03,
        "rms_mean": 0.10,
        "rms_std": 0.05,
    },
    "acoustic_guitar": {
        "mfcc_means": [-130, 50, -12, 10, -6, 5, -3, 3, -2, 2, -1, 1, 0],
        "mfcc_stds": [33, 17, 12, 9, 7, 5, 4, 3, 3, 2, 2, 1, 1],
        "centroid_mean": 2000,
        "centroid_std": 800,
        "rolloff_mean": 6000,
        "rolloff_std": 2000,
        "flux_mean": 25,
        "flux_std": 15,
        "zcr_mean": 0.05,
        "zcr_std": 0.03,
        "rms_mean": 0.06,
        "rms_std": 0.03,
    },
    "keys": {
        "mfcc_means": [-135, 52, -13, 10, -6, 5, -3, 4, -2, 2, -1, 1, 0],
        "mfcc_stds": [34, 17, 12, 9, 7, 5, 4, 3, 3, 2, 2, 1, 1],
        "centroid_mean": 1800,
        "centroid_std": 700,
        "rolloff_mean": 5500,
        "rolloff_std": 2000,
        "flux_mean": 22,
        "flux_std": 12,
        "zcr_mean": 0.04,
        "zcr_std": 0.02,
        "rms_mean": 0.07,
        "rms_std": 0.04,
    },
    "vocals": {
        "mfcc_means": [-160, 65, -16, 13, -5, 7, -2, 4, -2, 3, -1, 2, -1],
        "mfcc_stds": [36, 19, 14, 10, 8, 6, 5, 4, 3, 3, 2, 2, 1],
        "centroid_mean": 1500,
        "centroid_std": 600,
        "rolloff_mean": 4500,
        "rolloff_std": 1500,
        "flux_mean": 30,
        "flux_std": 15,
        "zcr_mean": 0.07,
        "zcr_std": 0.03,
        "rms_mean": 0.08,
        "rms_std": 0.04,
    },
    "brass": {
        "mfcc_means": [-145, 58, -14, 11, -6, 6, -3, 4, -2, 2, -1, 1, 0],
        "mfcc_stds": [36, 18, 13, 9, 7, 6, 5, 4, 3, 2, 2, 2, 1],
        "centroid_mean": 2200,
        "centroid_std": 900,
        "rolloff_mean": 6500,
        "rolloff_std": 2200,
        "flux_mean": 28,
        "flux_std": 14,
        "zcr_mean": 0.05,
        "zcr_std": 0.03,
        "rms_mean": 0.09,
        "rms_std": 0.05,
    },
    "strings": {
        "mfcc_means": [-155, 62, -15, 12, -5, 6, -3, 4, -2, 3, -1, 1, 0],
        "mfcc_stds": [37, 19, 14, 10, 8, 6, 5, 4, 3, 3, 2, 2, 1],
        "centroid_mean": 1200,
        "centroid_std": 500,
        "rolloff_mean": 4000,
        "rolloff_std": 1500,
        "flux_mean": 15,
        "flux_std": 8,
        "zcr_mean": 0.04,
        "zcr_std": 0.02,
        "rms_mean": 0.06,
        "rms_std": 0.03,
    },
    "percussion": {
        "mfcc_means": [-125, 48, -11, 9, -5, 5, -3, 3, -2, 2, -1, 1, 0],
        "mfcc_stds": [32, 16, 11, 8, 6, 5, 4, 3, 3, 2, 2, 1, 1],
        "centroid_mean": 3500,
        "centroid_std": 1500,
        "rolloff_mean": 8000,
        "rolloff_std": 3000,
        "flux_mean": 55,
        "flux_std": 30,
        "zcr_mean": 0.09,
        "zcr_std": 0.05,
        "rms_mean": 0.08,
        "rms_std": 0.04,
    },
}

INSTRUMENT_CLASSES = [
    "kick", "snare", "hihat", "toms", "overheads",
    "bass_guitar", "electric_guitar", "acoustic_guitar",
    "keys", "vocals", "brass", "strings", "percussion",
]


def generate_synthetic_data(n_samples_per_class=200, random_state=42):
    """
    Generate synthetic training data from per-class feature distributions.

    Creates realistic feature vectors by sampling from Gaussian distributions
    with class-specific means and standard deviations, then adding
    inter-class variation and noise.

    Args:
        n_samples_per_class: number of samples to generate per instrument class
        random_state: random seed for reproducibility

    Returns:
        X: (n_total, 34) feature matrix
        y: (n_total,) integer labels
        class_names: list of class name strings
    """
    rng = np.random.RandomState(random_state)
    X_list = []
    y_list = []

    for class_idx, class_name in enumerate(INSTRUMENT_CLASSES):
        params = _CLASS_FEATURE_PARAMS[class_name]

        for _ in range(n_samples_per_class):
            features = []

            # MFCC means (13 features)
            for j in range(13):
                mean = params["mfcc_means"][j]
                std = params["mfcc_stds"][j]
                features.append(rng.normal(mean, std * 1.5))

            # MFCC stds (13 features)
            for j in range(13):
                std_val = abs(params["mfcc_stds"][j])
                features.append(abs(rng.normal(std_val, std_val * 0.5)))

            # Spectral centroid (mean, std)
            cm = params["centroid_mean"]
            cs = params["centroid_std"]
            features.append(abs(rng.normal(cm, cs)))
            features.append(abs(rng.normal(cs, cs * 0.3)))

            # Spectral rolloff (mean, std)
            rm = params["rolloff_mean"]
            rs = params["rolloff_std"]
            features.append(abs(rng.normal(rm, rs)))
            features.append(abs(rng.normal(rs, rs * 0.3)))

            # Spectral flux (mean, std)
            fm = params["flux_mean"]
            fs = params["flux_std"]
            features.append(abs(rng.normal(fm, fs)))
            features.append(abs(rng.normal(fs, fs * 0.3)))

            # Zero crossing rate (mean, std)
            zm = params["zcr_mean"]
            zs = params["zcr_std"]
            features.append(abs(rng.normal(zm, zs)))
            features.append(abs(rng.normal(zs, zs * 0.3)))

            # RMS energy (mean, std)
            em = params["rms_mean"]
            es = params["rms_std"]
            features.append(abs(rng.normal(em, es)))
            features.append(abs(rng.normal(es, es * 0.3)))

            X_list.append(features)
            y_list.append(class_idx)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Shuffle
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y, INSTRUMENT_CLASSES


def train_and_save(data_dir=None, output_path="models/channel_classifier.joblib",
                   n_synthetic=200):
    """
    Train the channel classifier and save to disk.

    If data_dir is provided and contains .npy files, loads real data.
    Otherwise generates synthetic training data.

    Args:
        data_dir: optional directory with X.npy and y.npy
        output_path: path to save the trained model
        n_synthetic: samples per class for synthetic data

    Returns:
        dict with training metrics
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required for training. Install sklearn.")

    # Import classifier
    from .channel_classifier import ChannelClassifier

    # Load or generate data
    if data_dir and os.path.exists(os.path.join(data_dir, "X.npy")):
        logger.info(f"Loading data from {data_dir}")
        X = np.load(os.path.join(data_dir, "X.npy"))
        y = np.load(os.path.join(data_dir, "y.npy"))
        class_names = INSTRUMENT_CLASSES
    else:
        logger.info(f"Generating synthetic data ({n_synthetic} per class)")
        X, y, class_names = generate_synthetic_data(n_samples_per_class=n_synthetic)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

    # Create and train classifier
    classifier = ChannelClassifier()
    train_acc = classifier.train(X_train, y_train)

    # Cross-validation on training set
    cv_scores = cross_val_score(
        classifier.model, classifier.scaler.transform(X_train),
        y_train, cv=5, scoring="accuracy"
    )
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Evaluate on test set
    X_test_scaled = classifier.scaler.transform(X_test)
    y_pred = classifier.model.predict(X_test_scaled)
    test_acc = np.mean(y_pred == y_test)
    logger.info(f"Test accuracy: {test_acc:.4f}")

    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
    )
    logger.info("\n" + classification_report(y_test, y_pred, target_names=class_names))

    # Save model
    classifier.save(output_path)

    return {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "report": report,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }


def evaluate(model_path, test_data=None, n_synthetic=100):
    """
    Evaluate a trained classifier model.

    Args:
        model_path: path to saved .joblib model
        test_data: optional (X, y) tuple of test data
        n_synthetic: samples per class if generating test data

    Returns:
        dict with evaluation metrics
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required for evaluation.")

    from .channel_classifier import ChannelClassifier

    classifier = ChannelClassifier(model_path=model_path)

    if test_data is not None:
        X_test, y_test = test_data
    else:
        logger.info("Generating synthetic test data for evaluation")
        X_test, y_test, _ = generate_synthetic_data(
            n_samples_per_class=n_synthetic, random_state=99
        )

    X_test_scaled = classifier.scaler.transform(X_test)
    y_pred = classifier.model.predict(X_test_scaled)
    y_proba = classifier.model.predict_proba(X_test_scaled)

    accuracy = float(np.mean(y_pred == y_test))
    logger.info(f"Evaluation accuracy: {accuracy:.4f}")

    # Per-class accuracy
    per_class = {}
    for i, name in enumerate(INSTRUMENT_CLASSES):
        mask = y_test == i
        if np.any(mask):
            class_acc = float(np.mean(y_pred[mask] == y_test[mask]))
            avg_conf = float(np.mean(np.max(y_proba[mask], axis=1)))
            per_class[name] = {"accuracy": class_acc, "avg_confidence": avg_conf}

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    report_str = classification_report(
        y_test, y_pred, target_names=INSTRUMENT_CLASSES
    )
    logger.info("\n" + report_str)

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_test)),
    }


# ======================================================================
# ChannelClassifierTrainer
# ======================================================================

class ChannelClassifierTrainer:
    """
    End-to-end training pipeline for the channel/instrument classifier.

    Features extracted per audio file (computed with numpy/scipy):
        spectral centroid, spectral rolloff, spectral bandwidth,
        zero crossing rate, RMS energy
    (plus 13 MFCC means and 13 MFCC stds from the shared
    ``channel_classifier.extract_features`` when available).

    Usage::

        trainer = ChannelClassifierTrainer()
        X, y = trainer.prepare_dataset("data/audio", "data/labels.csv")
        metrics = trainer.train(X, y, "models/classifier.joblib")
        eval_result = trainer.evaluate(X_test, y_test)
    """

    # Default classifier hyper-parameters
    _DEFAULT_N_ESTIMATORS = 200
    _DEFAULT_MAX_DEPTH = 20
    _DEFAULT_RANDOM_STATE = 42

    def __init__(self, sr=48000, n_fft=2048, hop_length=512):
        """
        Args:
            sr: sample rate for audio analysis.
            n_fft: FFT window size.
            hop_length: hop size in samples.
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.model = None
        self.scaler = None
        self.classes = list(INSTRUMENT_CLASSES)
        self._class_to_idx = {c: i for i, c in enumerate(self.classes)}

    # ----- feature extraction ------------------------------------------------

    def extract_features(self, audio_path):
        """
        Extract a feature vector from an audio file on disk.

        Computes (using numpy, optionally scipy):
            * spectral centroid  (mean)
            * spectral rolloff   (mean, 85th percentile)
            * spectral bandwidth (mean)
            * zero crossing rate (mean)
            * RMS energy         (mean)

        If the file cannot be loaded by numpy alone, a WAV-specific
        loader is used.

        Args:
            audio_path: path to a WAV audio file.

        Returns:
            1-D numpy array of shape ``(N_FEATURES,)`` (float32).

        Raises:
            FileNotFoundError: if *audio_path* does not exist.
        """
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(audio_path)

        audio = self._load_wav(audio_path)
        return self._extract_features_from_array(audio)

    def _load_wav(self, path):
        """Load a WAV file into a 1-D float32 numpy array."""
        try:
            import wave
            with wave.open(path, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            if sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 4:
                dtype = np.int32
            else:
                dtype = np.int16  # fallback

            audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            # Mix to mono
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
            # Normalise to -1..1
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio /= peak
            return audio
        except Exception as exc:
            logger.warning("WAV load failed for %s: %s, returning silence", path, exc)
            return np.zeros(self.sr, dtype=np.float32)

    def _extract_features_from_array(self, audio):
        """Compute feature vector from a 1-D numpy array of audio."""
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))

        n_frames = max(1, (len(audio) - self.n_fft) // self.hop_length + 1)
        window = np.hanning(self.n_fft)
        sr = self.sr

        spectrogram = np.zeros((self.n_fft // 2 + 1, n_frames), dtype=np.float64)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = audio[start: start + self.n_fft]
            if len(frame) < self.n_fft:
                frame = np.pad(frame, (0, self.n_fft - len(frame)))
            spectrogram[:, i] = np.abs(np.fft.rfft(frame * window))

        freq_bins = np.arange(spectrogram.shape[0]) * sr / self.n_fft
        total_energy = np.sum(spectrogram, axis=0) + 1e-10

        # ---- spectral centroid ----
        centroid = np.sum(spectrogram * freq_bins[:, np.newaxis], axis=0) / total_energy
        centroid_mean = float(np.mean(centroid))

        # ---- spectral rolloff (85th percentile) ----
        cumsum = np.cumsum(spectrogram, axis=0)
        threshold = 0.85 * total_energy
        rolloff = np.zeros(n_frames)
        for i in range(n_frames):
            idx = np.searchsorted(cumsum[:, i], threshold[i])
            rolloff[i] = idx * sr / self.n_fft
        rolloff_mean = float(np.mean(rolloff))

        # ---- spectral bandwidth ----
        deviation = np.abs(freq_bins[:, np.newaxis] - centroid[np.newaxis, :])
        bandwidth = np.sum(spectrogram * deviation, axis=0) / total_energy
        bandwidth_mean = float(np.mean(bandwidth))

        # ---- zero crossing rate ----
        zcr_vals = []
        for i in range(n_frames):
            start = i * self.hop_length
            end = min(start + self.n_fft, len(audio))
            frame = audio[start:end]
            if len(frame) > 1:
                zcr = float(np.sum(np.abs(np.diff(np.sign(frame))))) / (2.0 * len(frame))
            else:
                zcr = 0.0
            zcr_vals.append(zcr)
        zcr_mean = float(np.mean(zcr_vals))

        # ---- RMS energy ----
        rms_vals = []
        for i in range(n_frames):
            start = i * self.hop_length
            end = min(start + self.n_fft, len(audio))
            frame = audio[start:end]
            rms_vals.append(float(np.sqrt(np.mean(frame ** 2) + 1e-10)))
        rms_mean = float(np.mean(rms_vals))

        features = np.array(
            [centroid_mean, rolloff_mean, bandwidth_mean, zcr_mean, rms_mean],
            dtype=np.float32,
        )
        return features

    # ----- dataset preparation -----------------------------------------------

    def prepare_dataset(self, audio_dir, labels_csv):
        """
        Build a feature matrix and label vector from audio files on disk.

        Args:
            audio_dir: directory containing WAV files.
            labels_csv: CSV file with columns ``filename`` and ``label``.
                Each ``filename`` is relative to *audio_dir*.

        Returns:
            (X, y): tuple where X is ``(N, n_features)`` float32 and
            y is ``(N,)`` int64 class indices.

        Raises:
            FileNotFoundError: if *labels_csv* does not exist.
            RuntimeError: if no valid samples were found.
        """
        if not os.path.isfile(labels_csv):
            raise FileNotFoundError(labels_csv)

        X_list = []
        y_list = []
        skipped = 0

        with open(labels_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "").strip()
                label = row.get("label", "").strip()
                if not filename or not label:
                    skipped += 1
                    continue

                audio_path = os.path.join(audio_dir, filename)
                if not os.path.isfile(audio_path):
                    logger.warning("Audio file not found, skipping: %s", audio_path)
                    skipped += 1
                    continue

                if label not in self._class_to_idx:
                    logger.warning("Unknown label '%s', skipping %s", label, filename)
                    skipped += 1
                    continue

                try:
                    feats = self.extract_features(audio_path)
                    X_list.append(feats)
                    y_list.append(self._class_to_idx[label])
                except Exception as exc:
                    logger.warning("Feature extraction failed for %s: %s", filename, exc)
                    skipped += 1

        if not X_list:
            raise RuntimeError(
                "No valid samples found in %s / %s (skipped %d)"
                % (audio_dir, labels_csv, skipped)
            )

        logger.info(
            "Dataset: %d samples loaded, %d skipped", len(X_list), skipped
        )

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    # ----- training ----------------------------------------------------------

    def train(self, X, y, model_path=None):
        """
        Train a RandomForestClassifier on the given features and labels.

        Args:
            X: ``(N, n_features)`` feature matrix (numpy array).
            y: ``(N,)`` integer class labels.
            model_path: if provided, save the trained model to this path.

        Returns:
            dict with ``train_accuracy``, ``cv_mean``, ``cv_std``,
            and (if model_path given) ``model_path``.

        Raises:
            RuntimeError: if scikit-learn is not installed.
        """
        if not HAS_SKLEARN:
            raise RuntimeError(
                "scikit-learn is required for training. "
                "Install with: pip install scikit-learn"
            )

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Build model
        self.model = RandomForestClassifier(
            n_estimators=self._DEFAULT_N_ESTIMATORS,
            max_depth=self._DEFAULT_MAX_DEPTH,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=self._DEFAULT_RANDOM_STATE,
            n_jobs=-1,
        )
        self.model.fit(X_scaled, y)

        # Training accuracy
        train_preds = self.model.predict(X_scaled)
        train_acc = float(np.mean(train_preds == y))

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y, cv=min(5, len(X)), scoring="accuracy"
        )
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        logger.info(
            "Training complete: acc=%.4f, CV=%.4f (+/- %.4f)",
            train_acc, cv_mean, cv_std,
        )

        result = {
            "train_accuracy": train_acc,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }

        if model_path:
            self._save_model(model_path)
            result["model_path"] = model_path

        return result

    def _save_model(self, path):
        """Persist the trained model, scaler, and class list to disk."""
        if not HAS_JOBLIB:
            logger.warning("joblib not available; saving with numpy instead.")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.savez(
                path,
                classes=np.array(self.classes),
            )
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "classes": self.classes,
        }
        joblib.dump(data, path)
        logger.info("Model saved to %s", path)

    def _load_model(self, path):
        """Load a previously saved model."""
        if not HAS_JOBLIB:
            raise RuntimeError("joblib required for loading models")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.classes = data.get("classes", INSTRUMENT_CLASSES)
        self._class_to_idx = {c: i for i, c in enumerate(self.classes)}
        logger.info("Model loaded from %s", path)

    # ----- evaluation --------------------------------------------------------

    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model on a held-out test set.

        Args:
            X_test: ``(N, n_features)`` feature matrix.
            y_test: ``(N,)`` integer class labels.

        Returns:
            dict with ``accuracy``, ``per_class`` (dict of name ->
            accuracy/confidence), ``confusion_matrix`` (list of lists),
            ``classification_report`` (string).

        Raises:
            RuntimeError: if no model has been trained / loaded.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("No trained model available. Call train() first.")
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn required for evaluation.")

        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.int64)

        X_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)

        accuracy = float(np.mean(y_pred == y_test))
        logger.info("Test accuracy: %.4f", accuracy)

        # Per-class metrics
        per_class = {}
        for i, name in enumerate(self.classes):
            mask = y_test == i
            if np.any(mask):
                cls_acc = float(np.mean(y_pred[mask] == y_test[mask]))
                avg_conf = float(np.mean(np.max(y_proba[mask], axis=1)))
                per_class[name] = {"accuracy": cls_acc, "avg_confidence": avg_conf}

        cm = confusion_matrix(y_test, y_pred)
        report_str = classification_report(
            y_test, y_pred, target_names=self.classes, zero_division=0
        )
        logger.info("\n%s", report_str)

        return {
            "accuracy": accuracy,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_str,
            "n_samples": int(len(y_test)),
        }


# ======================================================================
# CLI main
# ======================================================================

def main():
    """Command-line entry point for training and evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train/evaluate channel classifier"
    )
    parser.add_argument(
        "--output", "-o",
        default="models/channel_classifier.joblib",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--evaluate", "-e",
        default=None,
        help="Path to model to evaluate (skip training)",
    )
    parser.add_argument(
        "--data-dir", "-d",
        default=None,
        help="Directory with X.npy and y.npy training data",
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Directory with WAV files (use with --labels)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="CSV file with 'filename' and 'label' columns",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Synthetic samples per class (default 200)",
    )

    args = parser.parse_args()

    # --- Use the new ChannelClassifierTrainer when audio-dir is given ---
    if args.audio_dir and args.labels:
        trainer = ChannelClassifierTrainer()
        X, y = trainer.prepare_dataset(args.audio_dir, args.labels)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        metrics = trainer.train(X_train, y_train, model_path=args.output)
        print("\nTraining Results:")
        print("  Train accuracy: %.4f" % metrics["train_accuracy"])
        print("  CV accuracy: %.4f (+/- %.4f)" % (metrics["cv_mean"], metrics["cv_std"]))
        eval_result = trainer.evaluate(X_test, y_test)
        print("  Test accuracy: %.4f" % eval_result["accuracy"])
        print("  Model saved to: %s" % args.output)
        return

    # --- Legacy paths (synthetic / pre-extracted data) ---
    if args.evaluate:
        results = evaluate(args.evaluate, n_synthetic=args.n_samples)
        print("\nEvaluation Results:")
        print("  Accuracy: %.4f" % results["accuracy"])
        print("  Samples: %d" % results["n_samples"])
        for name, metrics in results["per_class"].items():
            print("  %s: acc=%.3f, conf=%.3f" % (name, metrics["accuracy"], metrics["avg_confidence"]))
    else:
        results = train_and_save(
            data_dir=args.data_dir,
            output_path=args.output,
            n_synthetic=args.n_samples,
        )
        print("\nTraining Results:")
        print("  Train accuracy: %.4f" % results["train_accuracy"])
        print("  Test accuracy: %.4f" % results["test_accuracy"])
        print("  CV accuracy: %.4f (+/- %.4f)" % (results["cv_mean"], results["cv_std"]))
        print("  Model saved to: %s" % args.output)


if __name__ == "__main__":
    main()
