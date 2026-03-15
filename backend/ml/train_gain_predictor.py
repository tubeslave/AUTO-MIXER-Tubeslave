"""
Training script for the gain/pan predictor.

Provides:
- GainPredictorTrainer: sklearn-based training pipeline with feature
  engineering (LUFS, spectral centroid, dynamic range, channel-type
  one-hot encoding), separate gain and pan models, and evaluation.
- Synthetic training data generation from realistic mixing parameter
  distributions (for the PyTorch pipeline).
- PyTorch training loop with LR scheduling and early stopping.
- CLI interface supporting both sklearn and PyTorch workflows.

Usage:
    # sklearn-based trainer (no torch required)
    python -m backend.ml.train_gain_predictor --sklearn --output models/gain_predictor

    # PyTorch-based trainer
    python -m backend.ml.train_gain_predictor --output models/gain_pan_predictor.pt
"""

import csv
import numpy as np
import os
import sys
import struct
import logging
import argparse

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import joblib as _joblib

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Instrument-specific gain and pan distributions for synthetic data
# Based on typical live mixing practice
_INSTRUMENT_MIX_PARAMS = {
    "kick":             {"gain_mean": -8.0,  "gain_std": 3.0,  "pan_mean": 0.0,   "pan_std": 0.05},
    "snare":            {"gain_mean": -10.0, "gain_std": 3.0,  "pan_mean": 0.0,   "pan_std": 0.1},
    "hihat":            {"gain_mean": -18.0, "gain_std": 4.0,  "pan_mean": 0.3,   "pan_std": 0.15},
    "toms":             {"gain_mean": -14.0, "gain_std": 3.0,  "pan_mean": 0.0,   "pan_std": 0.3},
    "overheads":        {"gain_mean": -16.0, "gain_std": 4.0,  "pan_mean": 0.0,   "pan_std": 0.1},
    "bass_guitar":      {"gain_mean": -10.0, "gain_std": 3.0,  "pan_mean": 0.0,   "pan_std": 0.05},
    "electric_guitar":  {"gain_mean": -12.0, "gain_std": 4.0,  "pan_mean": -0.4,  "pan_std": 0.3},
    "acoustic_guitar":  {"gain_mean": -14.0, "gain_std": 3.0,  "pan_mean": 0.3,   "pan_std": 0.25},
    "keys":             {"gain_mean": -14.0, "gain_std": 3.0,  "pan_mean": -0.2,  "pan_std": 0.2},
    "vocals":           {"gain_mean": -6.0,  "gain_std": 3.0,  "pan_mean": 0.0,   "pan_std": 0.05},
    "brass":            {"gain_mean": -12.0, "gain_std": 3.0,  "pan_mean": 0.2,   "pan_std": 0.3},
    "strings":          {"gain_mean": -16.0, "gain_std": 4.0,  "pan_mean": 0.0,   "pan_std": 0.2},
    "percussion":       {"gain_mean": -16.0, "gain_std": 4.0,  "pan_mean": -0.3,  "pan_std": 0.3},
}

INSTRUMENT_CLASSES = [
    "kick", "snare", "hihat", "toms", "overheads",
    "bass_guitar", "electric_guitar", "acoustic_guitar",
    "keys", "vocals", "brass", "strings", "percussion",
]


def generate_synthetic_data(n_samples_per_class=300, random_state=42):
    """
    Generate synthetic training data for gain/pan prediction.

    Creates pairs of (audio features, target gain/pan) by:
    1. Sampling audio features from per-class distributions
    2. Sampling target gain and pan from mixing practice distributions

    Args:
        n_samples_per_class: samples per instrument class
        random_state: random seed

    Returns:
        features: (n_total, 34) feature matrix
        gain_targets: (n_total, 1) target gain in dB
        pan_targets: (n_total, 1) target pan (-1..+1)
    """
    # Re-use the classifier's feature distributions
    try:
        from .train_classifier import _CLASS_FEATURE_PARAMS, generate_synthetic_data as gen_clf_data
        X_features, y_labels, _ = gen_clf_data(
            n_samples_per_class=n_samples_per_class,
            random_state=random_state,
        )
    except ImportError:
        # Standalone fallback
        rng = np.random.RandomState(random_state)
        n_total = n_samples_per_class * len(INSTRUMENT_CLASSES)
        X_features = rng.randn(n_total, 34).astype(np.float32)
        y_labels = np.repeat(np.arange(len(INSTRUMENT_CLASSES)), n_samples_per_class)
        indices = rng.permutation(n_total)
        X_features = X_features[indices]
        y_labels = y_labels[indices]

    rng = np.random.RandomState(random_state + 1)
    n_total = len(X_features)
    gain_targets = np.zeros((n_total, 1), dtype=np.float32)
    pan_targets = np.zeros((n_total, 1), dtype=np.float32)

    for i in range(n_total):
        class_idx = y_labels[i]
        class_name = INSTRUMENT_CLASSES[class_idx]
        mix_params = _INSTRUMENT_MIX_PARAMS[class_name]

        # Sample gain and pan
        gain = rng.normal(mix_params["gain_mean"], mix_params["gain_std"])
        gain = np.clip(gain, -30.0, 12.0)
        gain_targets[i, 0] = gain

        pan = rng.normal(mix_params["pan_mean"], mix_params["pan_std"])
        pan = np.clip(pan, -1.0, 1.0)
        pan_targets[i, 0] = pan

    return X_features, gain_targets, pan_targets


def train_loop(model, train_loader, val_loader=None, epochs=100,
               learning_rate=1e-3, weight_decay=1e-4, patience=15):
    """
    Training loop with learning rate scheduling and early stopping.

    Args:
        model: GainPanPredictor instance
        train_loader: DataLoader for training data
        val_loader: optional DataLoader for validation
        epochs: maximum number of training epochs
        learning_rate: initial learning rate
        weight_decay: L2 regularization weight
        patience: epochs without improvement before early stopping

    Returns:
        dict with training history
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for training.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 3, min_lr=1e-6
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        for batch_features, batch_gain, batch_pan in train_loader:
            batch = {
                "features": batch_features,
                "gain_db": batch_gain,
                "pan": batch_pan,
            }
            loss = model.train_step(batch, optimizer)
            train_losses.append(loss)

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(float(avg_train_loss))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        # Validation phase
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_features, batch_gain, batch_pan in val_loader:
                    pred_gain, pred_pan = model(batch_features)
                    gain_loss = nn.functional.mse_loss(pred_gain, batch_gain)
                    pan_loss = nn.functional.mse_loss(pred_pan, batch_pan)
                    val_loss = 2.0 * gain_loss + pan_loss
                    val_losses.append(float(val_loss.item()))

            avg_val_loss = np.mean(val_losses)
            history["val_loss"].append(float(avg_val_loss))

            # LR scheduling
            scheduler.step(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch}/{epochs}: "
                    f"train_loss={avg_train_loss:.6f}, "
                    f"val_loss={avg_val_loss:.6f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                )

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break
        else:
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch}/{epochs}: train_loss={avg_train_loss:.6f}"
                )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model (val_loss={best_val_loss:.6f})")

    return history


def train_and_save(output_path="models/gain_pan_predictor.pt", n_synthetic=300,
                   epochs=100, batch_size=64, learning_rate=1e-3):
    """
    Full training pipeline: generate data -> train -> save.

    Args:
        output_path: path to save the trained model
        n_synthetic: samples per class for synthetic data
        epochs: training epochs
        batch_size: mini-batch size
        learning_rate: initial learning rate

    Returns:
        dict with training results
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for training.")

    from .gain_pan_predictor import GainPanPredictor

    # Generate data
    logger.info(f"Generating synthetic training data ({n_synthetic} per class)")
    X, gain_y, pan_y = generate_synthetic_data(n_samples_per_class=n_synthetic)

    # Split into train/val
    n_total = len(X)
    n_val = int(n_total * 0.2)
    indices = np.random.RandomState(42).permutation(n_total)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    gain_train = torch.tensor(gain_y[train_idx], dtype=torch.float32)
    pan_train = torch.tensor(pan_y[train_idx], dtype=torch.float32)

    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    gain_val = torch.tensor(gain_y[val_idx], dtype=torch.float32)
    pan_val = torch.tensor(pan_y[val_idx], dtype=torch.float32)

    train_dataset = TensorDataset(X_train, gain_train, pan_train)
    val_dataset = TensorDataset(X_val, gain_val, pan_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = X.shape[1]
    model = GainPanPredictor(input_dim=input_dim)
    logger.info(f"Model created: input_dim={input_dim}, params={sum(p.numel() for p in model.parameters())}")

    # Train
    history = train_loop(
        model, train_loader, val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred_gain, pred_pan = model(X_val)
        gain_mae = torch.mean(torch.abs(pred_gain - gain_val)).item()
        pan_mae = torch.mean(torch.abs(pred_pan - pan_val)).item()

    logger.info(f"Final evaluation: gain_MAE={gain_mae:.2f} dB, pan_MAE={pan_mae:.3f}")

    # Save
    model.save_model(output_path)

    return {
        "history": history,
        "gain_mae_db": float(gain_mae),
        "pan_mae": float(pan_mae),
        "n_train": int(len(train_idx)),
        "n_val": int(n_val),
        "model_path": output_path,
    }


# ======================================================================
# GainPredictorTrainer  (sklearn-based, no torch required)
# ======================================================================

class GainPredictorTrainer:
    """
    Sklearn-based training pipeline for gain and pan prediction.

    Features per channel (computed by :meth:`prepare_features`)::

        [lufs, spectral_centroid, dynamic_range, rms_db,
         one_hot_kick, one_hot_snare, ..., one_hot_percussion]

    This gives a 4 + N_CLASSES = 17 dimensional feature vector.

    Two separate models are trained:
        * **gain model** -- :class:`GradientBoostingRegressor` predicting
          target gain in dB.
        * **pan model** -- :class:`RandomForestRegressor` predicting
          target pan position (-1 .. +1).

    Usage::

        trainer = GainPredictorTrainer()
        X, y_gain, y_pan = trainer.generate_training_data()
        gain_result = trainer.train_gain_model(X, y_gain, "models/gain.joblib")
        pan_result  = trainer.train_pan_model(X, y_pan, "models/pan.joblib")
        eval_result = trainer.evaluate(trainer.gain_model, X_test, y_test_gain)
    """

    # The same instrument class list used elsewhere in the ML module.
    _INSTRUMENT_CLASSES = list(INSTRUMENT_CLASSES)
    _N_CLASSES = len(_INSTRUMENT_CLASSES)
    _N_SCALAR_FEATURES = 4  # lufs, spectral_centroid, dynamic_range, rms_db
    _N_FEATURES = _N_SCALAR_FEATURES + _N_CLASSES  # 4 + 13 = 17

    def __init__(self, sr=48000, n_fft=2048, hop_length=512):
        """
        Args:
            sr: sample rate for audio analysis.
            n_fft: FFT window size.
            hop_length: hop length in samples.
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.gain_model = None
        self.pan_model = None
        self.gain_scaler = None
        self.pan_scaler = None
        self._class_to_idx = {c: i for i, c in enumerate(self._INSTRUMENT_CLASSES)}

    # ----- feature helpers ---------------------------------------------------

    @classmethod
    def _one_hot(cls, instrument_type):
        """Return a one-hot vector of length ``_N_CLASSES`` for *instrument_type*."""
        vec = np.zeros(cls._N_CLASSES, dtype=np.float32)
        idx_map = {c: i for i, c in enumerate(cls._INSTRUMENT_CLASSES)}
        idx = idx_map.get(instrument_type, -1)
        if idx >= 0:
            vec[idx] = 1.0
        return vec

    @staticmethod
    def _estimate_lufs(audio, sr=48000):
        """
        Estimate integrated LUFS from a 1-D audio array (numpy only).

        This is a simplified K-weighted loudness estimate.  A full
        ITU-R BS.1770-4 implementation requires a pre-filter stage;
        here we approximate by RMS of the signal after a simple
        high-shelf emphasis.

        Returns:
            Estimated LUFS value (float).
        """
        audio = np.asarray(audio, dtype=np.float64)
        if len(audio) == 0:
            return -70.0

        # Simple high-shelf emphasis at ~1500 Hz (K-weighting approximation)
        alpha = 0.95
        emphasized = np.empty_like(audio)
        emphasized[0] = audio[0]
        for i in range(1, len(audio)):
            emphasized[i] = audio[i] - alpha * audio[i - 1]

        rms = np.sqrt(np.mean(emphasized ** 2) + 1e-12)
        lufs = -0.691 + 10.0 * np.log10(rms ** 2 + 1e-12)
        return float(lufs)

    @staticmethod
    def _estimate_dynamic_range(audio, frame_len=2048, hop=512):
        """
        Estimate dynamic range (dB) as difference between 95th and 5th
        percentile of per-frame RMS levels.
        """
        audio = np.asarray(audio, dtype=np.float64)
        if len(audio) < frame_len:
            return 0.0

        n_frames = max(1, (len(audio) - frame_len) // hop + 1)
        rms_db = np.empty(n_frames, dtype=np.float64)
        for i in range(n_frames):
            start = i * hop
            frame = audio[start: start + frame_len]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
            rms_db[i] = 20.0 * np.log10(rms + 1e-12)

        p95 = float(np.percentile(rms_db, 95))
        p5 = float(np.percentile(rms_db, 5))
        return max(0.0, p95 - p5)

    @staticmethod
    def _spectral_centroid_mean(audio, sr=48000, n_fft=2048, hop=512):
        """Compute mean spectral centroid (Hz) from a 1-D numpy array."""
        audio = np.asarray(audio, dtype=np.float64)
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))

        n_frames = max(1, (len(audio) - n_fft) // hop + 1)
        window = np.hanning(n_fft)
        freq_bins = np.arange(n_fft // 2 + 1) * sr / n_fft
        centroids = np.empty(n_frames, dtype=np.float64)

        for i in range(n_frames):
            start = i * hop
            frame = audio[start: start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            mag = np.abs(np.fft.rfft(frame * window))
            total = np.sum(mag) + 1e-10
            centroids[i] = np.sum(mag * freq_bins) / total

        return float(np.mean(centroids))

    # ----- public feature API ------------------------------------------------

    def prepare_features(self, channel_features):
        """
        Build a feature matrix from per-channel feature dicts.

        Each entry in *channel_features* must provide at least
        ``instrument_type`` (string).  Optional numeric keys:
        ``lufs``, ``spectral_centroid``, ``dynamic_range``, ``rms_db``.

        If numeric keys are missing they are set to sensible defaults.

        Args:
            channel_features: iterable of dicts, one per channel.

        Returns:
            X: ``(N, N_FEATURES)`` float32 array.
        """
        rows = []
        for feat in channel_features:
            lufs = float(feat.get("lufs", -20.0))
            centroid = float(feat.get("spectral_centroid", 1500.0))
            dynamic_range = float(feat.get("dynamic_range", 12.0))
            rms_db = float(feat.get("rms_db", -18.0))
            instrument = feat.get("instrument_type", "")
            one_hot = self._one_hot(instrument)

            row = np.concatenate([
                np.array([lufs, centroid, dynamic_range, rms_db], dtype=np.float32),
                one_hot,
            ])
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    def prepare_features_from_audio(self, audio, instrument_type):
        """
        Extract the feature vector from a raw audio array + label.

        Args:
            audio: 1-D numpy array.
            instrument_type: string label (e.g. "vocals").

        Returns:
            1-D ``(N_FEATURES,)`` float32 array.
        """
        audio = np.asarray(audio, dtype=np.float64)
        lufs = self._estimate_lufs(audio, sr=self.sr)
        centroid = self._spectral_centroid_mean(
            audio, sr=self.sr, n_fft=self.n_fft, hop=self.hop_length
        )
        dyn_range = self._estimate_dynamic_range(
            audio, frame_len=self.n_fft, hop=self.hop_length
        )
        rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
        rms_db = float(20.0 * np.log10(rms + 1e-12))

        one_hot = self._one_hot(instrument_type)
        return np.concatenate([
            np.array([lufs, centroid, dyn_range, rms_db], dtype=np.float32),
            one_hot,
        ])

    # ----- synthetic data generation -----------------------------------------

    def generate_training_data(self, n_samples_per_class=300,
                               random_state=42):
        """
        Generate synthetic training data for the sklearn pipeline.

        Returns:
            (X, y_gain, y_pan): feature matrix, gain targets, pan targets.
        """
        rng = np.random.RandomState(random_state)
        X_list = []
        gain_list = []
        pan_list = []

        for cls_name in self._INSTRUMENT_CLASSES:
            mix_params = _INSTRUMENT_MIX_PARAMS[cls_name]
            one_hot = self._one_hot(cls_name)

            for _ in range(n_samples_per_class):
                # Synthetic scalar features
                lufs = rng.normal(-18.0, 6.0)
                centroid = abs(rng.normal(1500.0, 800.0))
                dyn_range = abs(rng.normal(12.0, 6.0))
                rms_db = rng.normal(-18.0, 8.0)

                row = np.concatenate([
                    np.array([lufs, centroid, dyn_range, rms_db],
                             dtype=np.float32),
                    one_hot,
                ])
                X_list.append(row)

                # Target gain (correlated with LUFS + instrument offset)
                target_gain = rng.normal(
                    mix_params["gain_mean"], mix_params["gain_std"]
                )
                target_gain = float(np.clip(target_gain, -30.0, 12.0))
                gain_list.append(target_gain)

                # Target pan
                target_pan = rng.normal(
                    mix_params["pan_mean"], mix_params["pan_std"]
                )
                target_pan = float(np.clip(target_pan, -1.0, 1.0))
                pan_list.append(target_pan)

        X = np.array(X_list, dtype=np.float32)
        y_gain = np.array(gain_list, dtype=np.float32)
        y_pan = np.array(pan_list, dtype=np.float32)

        # Shuffle
        idx = rng.permutation(len(X))
        return X[idx], y_gain[idx], y_pan[idx]

    # ----- training ----------------------------------------------------------

    def train_gain_model(self, X, y_gain, model_path=None):
        """
        Train a GradientBoostingRegressor for gain prediction.

        Args:
            X: ``(N, N_FEATURES)`` feature matrix.
            y_gain: ``(N,)`` gain targets in dB.
            model_path: if given, persist the model to this path.

        Returns:
            dict with ``train_mae``, ``cv_mae_mean``, ``cv_mae_std``,
            and optionally ``model_path``.

        Raises:
            RuntimeError: if scikit-learn is not installed.
        """
        if not HAS_SKLEARN:
            raise RuntimeError(
                "scikit-learn is required for training.  "
                "Install with: pip install scikit-learn"
            )

        X = np.asarray(X, dtype=np.float32)
        y_gain = np.asarray(y_gain, dtype=np.float32).ravel()

        self.gain_scaler = StandardScaler()
        X_scaled = self.gain_scaler.fit_transform(X)

        self.gain_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
        )
        self.gain_model.fit(X_scaled, y_gain)

        # Training MAE
        train_pred = self.gain_model.predict(X_scaled)
        train_mae = float(mean_absolute_error(y_gain, train_pred))

        # Cross-validation
        cv_scores = cross_val_score(
            self.gain_model, X_scaled, y_gain,
            cv=min(5, len(X)),
            scoring="neg_mean_absolute_error",
        )
        cv_mae_mean = float(-cv_scores.mean())
        cv_mae_std = float(cv_scores.std())

        logger.info(
            "Gain model trained: MAE=%.3f dB, CV MAE=%.3f (+/- %.3f)",
            train_mae, cv_mae_mean, cv_mae_std,
        )

        result = {
            "train_mae": train_mae,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
        }

        if model_path:
            self._save_sklearn_model(
                model_path, self.gain_model, self.gain_scaler, "gain"
            )
            result["model_path"] = model_path

        return result

    def train_pan_model(self, X, y_pan, model_path=None):
        """
        Train a RandomForestRegressor for pan prediction.

        Args:
            X: ``(N, N_FEATURES)`` feature matrix.
            y_pan: ``(N,)`` pan targets in [-1, 1].
            model_path: if given, persist the model to this path.

        Returns:
            dict with ``train_mae``, ``cv_mae_mean``, ``cv_mae_std``,
            and optionally ``model_path``.

        Raises:
            RuntimeError: if scikit-learn is not installed.
        """
        if not HAS_SKLEARN:
            raise RuntimeError(
                "scikit-learn is required for training.  "
                "Install with: pip install scikit-learn"
            )

        X = np.asarray(X, dtype=np.float32)
        y_pan = np.asarray(y_pan, dtype=np.float32).ravel()

        self.pan_scaler = StandardScaler()
        X_scaled = self.pan_scaler.fit_transform(X)

        self.pan_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.pan_model.fit(X_scaled, y_pan)

        # Training MAE
        train_pred = self.pan_model.predict(X_scaled)
        train_mae = float(mean_absolute_error(y_pan, train_pred))

        # Cross-validation
        cv_scores = cross_val_score(
            self.pan_model, X_scaled, y_pan,
            cv=min(5, len(X)),
            scoring="neg_mean_absolute_error",
        )
        cv_mae_mean = float(-cv_scores.mean())
        cv_mae_std = float(cv_scores.std())

        logger.info(
            "Pan model trained: MAE=%.4f, CV MAE=%.4f (+/- %.4f)",
            train_mae, cv_mae_mean, cv_mae_std,
        )

        result = {
            "train_mae": train_mae,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
        }

        if model_path:
            self._save_sklearn_model(
                model_path, self.pan_model, self.pan_scaler, "pan"
            )
            result["model_path"] = model_path

        return result

    # ----- evaluation --------------------------------------------------------

    def evaluate(self, model, X_test, y_test, scaler=None):
        """
        Evaluate a trained regression model on held-out data.

        Args:
            model: a fitted sklearn regressor (gain or pan model).
            X_test: ``(N, N_FEATURES)`` feature matrix.
            y_test: ``(N,)`` ground-truth targets.
            scaler: optional ``StandardScaler`` to transform X_test.
                If *None*, tries ``self.gain_scaler`` then
                ``self.pan_scaler``.

        Returns:
            dict with ``mae``, ``rmse``, ``r2``, ``max_error``,
            ``n_samples``.

        Raises:
            RuntimeError: if scikit-learn is not installed.
        """
        if not HAS_SKLEARN:
            raise RuntimeError("scikit-learn required for evaluation.")

        X_test = np.asarray(X_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32).ravel()

        if scaler is not None:
            X_scaled = scaler.transform(X_test)
        elif self.gain_scaler is not None and model is self.gain_model:
            X_scaled = self.gain_scaler.transform(X_test)
        elif self.pan_scaler is not None and model is self.pan_model:
            X_scaled = self.pan_scaler.transform(X_test)
        else:
            # Fallback: no scaling
            X_scaled = X_test

        y_pred = model.predict(X_scaled)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        max_err = float(np.max(np.abs(y_test - y_pred)))

        logger.info(
            "Evaluation: MAE=%.4f, RMSE=%.4f, R2=%.4f, max_err=%.4f",
            mae, rmse, r2, max_err,
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "max_error": max_err,
            "n_samples": int(len(y_test)),
        }

    # ----- persistence -------------------------------------------------------

    def _save_sklearn_model(self, path, model, scaler, label):
        """Persist a sklearn model + scaler to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if HAS_JOBLIB:
            _joblib.dump(
                {"model": model, "scaler": scaler, "label": label},
                path,
            )
            logger.info("Saved %s model to %s (joblib)", label, path)
        else:
            # Fallback: numpy-only persistence of predictions is not
            # feasible for tree ensembles; warn and skip.
            logger.warning(
                "joblib not available — %s model NOT saved.  "
                "Install joblib to enable persistence.",
                label,
            )

    def load_gain_model(self, path):
        """Load a previously saved gain model."""
        data = self._load_sklearn_model(path)
        self.gain_model = data["model"]
        self.gain_scaler = data["scaler"]
        logger.info("Gain model loaded from %s", path)

    def load_pan_model(self, path):
        """Load a previously saved pan model."""
        data = self._load_sklearn_model(path)
        self.pan_model = data["model"]
        self.pan_scaler = data["scaler"]
        logger.info("Pan model loaded from %s", path)

    @staticmethod
    def _load_sklearn_model(path):
        """Load a joblib-persisted model dict."""
        if not HAS_JOBLIB:
            raise RuntimeError("joblib required for loading sklearn models")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return _joblib.load(path)

    # ----- convenience predict -----------------------------------------------

    def predict(self, X):
        """
        Predict gain and pan from a feature matrix using the sklearn models.

        Args:
            X: ``(N, N_FEATURES)`` float32 array.

        Returns:
            (gain_pred, pan_pred): tuple of 1-D numpy arrays.

        Raises:
            RuntimeError: if models have not been trained or loaded.
        """
        if self.gain_model is None:
            raise RuntimeError("Gain model not trained/loaded.")
        if self.pan_model is None:
            raise RuntimeError("Pan model not trained/loaded.")

        X = np.asarray(X, dtype=np.float32)
        X_g = self.gain_scaler.transform(X) if self.gain_scaler else X
        X_p = self.pan_scaler.transform(X) if self.pan_scaler else X
        gain = self.gain_model.predict(X_g)
        pan = self.pan_model.predict(X_p)
        return gain.astype(np.float32), pan.astype(np.float32)


# ======================================================================
# CLI main
# ======================================================================

def main():
    """Command-line entry point for training and evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train gain/pan predictor (sklearn or PyTorch)"
    )
    parser.add_argument(
        "--output", "-o",
        default="models/gain_pan_predictor.pt",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--sklearn", action="store_true",
        help="Use sklearn-based GainPredictorTrainer instead of PyTorch",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs (PyTorch only)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size (PyTorch only)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate (PyTorch only)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=300,
        help="Synthetic samples per class",
    )

    args = parser.parse_args()

    # ---- sklearn path ----
    if args.sklearn:
        if not HAS_SKLEARN:
            print("ERROR: scikit-learn is required.  pip install scikit-learn")
            sys.exit(1)

        trainer = GainPredictorTrainer()
        X, y_gain, y_pan = trainer.generate_training_data(
            n_samples_per_class=args.n_samples,
        )

        # Train/test split
        X_train, X_test, yg_train, yg_test, yp_train, yp_test = (
            train_test_split(
                X, y_gain, y_pan, test_size=0.2, random_state=42
            )
        )

        gain_out = args.output + "_gain.joblib"
        pan_out = args.output + "_pan.joblib"

        gain_res = trainer.train_gain_model(X_train, yg_train, gain_out)
        pan_res = trainer.train_pan_model(X_train, yp_train, pan_out)

        gain_eval = trainer.evaluate(trainer.gain_model, X_test, yg_test)
        pan_eval = trainer.evaluate(
            trainer.pan_model, X_test, yp_test, scaler=trainer.pan_scaler
        )

        print("\nGain Model Results:")
        print("  Train MAE : %.3f dB" % gain_res["train_mae"])
        print("  CV MAE    : %.3f (+/- %.3f)" % (gain_res["cv_mae_mean"], gain_res["cv_mae_std"]))
        print("  Test MAE  : %.3f dB" % gain_eval["mae"])
        print("  Test RMSE : %.3f dB" % gain_eval["rmse"])
        print("  Test R2   : %.4f" % gain_eval["r2"])

        print("\nPan Model Results:")
        print("  Train MAE : %.4f" % pan_res["train_mae"])
        print("  CV MAE    : %.4f (+/- %.4f)" % (pan_res["cv_mae_mean"], pan_res["cv_mae_std"]))
        print("  Test MAE  : %.4f" % pan_eval["mae"])
        print("  Test RMSE : %.4f" % pan_eval["rmse"])
        print("  Test R2   : %.4f" % pan_eval["r2"])
        return

    # ---- PyTorch path ----
    if not HAS_TORCH:
        print(
            "ERROR: PyTorch is required for the neural-network trainer.  "
            "Use --sklearn for the sklearn-based trainer."
        )
        sys.exit(1)

    results = train_and_save(
        output_path=args.output,
        n_synthetic=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print("\nTraining Results:")
    print("  Gain MAE: %.2f dB" % results["gain_mae_db"])
    print("  Pan MAE: %.3f" % results["pan_mae"])
    print("  Train samples: %d" % results["n_train"])
    print("  Val samples: %d" % results["n_val"])
    print("  Model saved to: %s" % results["model_path"])


if __name__ == "__main__":
    main()
