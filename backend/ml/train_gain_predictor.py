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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train gain/pan predictor")
    parser.add_argument(
        "--output", "-o",
        default="models/gain_pan_predictor.pt",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--n-samples", type=int, default=300,
        help="Synthetic samples per class",
    )

    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch is required for training. Install with: pip install torch")
        sys.exit(1)

    results = train_and_save(
        output_path=args.output,
        n_synthetic=args.n_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print(f"\nTraining Results:")
    print(f"  Gain MAE: {results['gain_mae_db']:.2f} dB")
    print(f"  Pan MAE: {results['pan_mae']:.3f}")
    print(f"  Train samples: {results['n_train']}")
    print(f"  Val samples: {results['n_val']}")
    print(f"  Model saved to: {results['model_path']}")
