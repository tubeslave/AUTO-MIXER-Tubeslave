"""
Squeeze-and-Excitation neural network for predicting gain (dB) and pan (-1..+1).

Takes audio features (MFCCs + spectral descriptors) as input and predicts
optimal mixing parameters. Uses channel attention via SE blocks to focus
on the most informative features.
"""

import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import feature extraction from our channel_classifier
try:
    from .channel_classifier import extract_features
except ImportError:
    extract_features = None


if HAS_TORCH:

    class SqueezeExcitation(nn.Module):
        """
        Squeeze-and-Excitation block for channel attention.

        Learns to re-weight feature channels by:
        1. Squeeze: global average pooling to aggregate spatial info
        2. Excitation: FC -> ReLU -> FC -> Sigmoid to produce channel weights
        3. Scale: multiply input by learned weights
        """

        def __init__(self, channels, reduction=4):
            """
            Args:
                channels: number of input channels/features
                reduction: reduction ratio for the bottleneck
            """
            super().__init__()
            mid = max(1, channels // reduction)
            self.fc1 = nn.Linear(channels, mid)
            self.fc2 = nn.Linear(mid, channels)

        def forward(self, x):
            """
            Args:
                x: (batch, channels) or (batch, channels, length)
            Returns:
                scaled_x: same shape as input
            """
            if x.dim() == 3:
                # (batch, channels, length) -> squeeze over length
                scale = x.mean(dim=2)  # (batch, channels)
            else:
                scale = x  # (batch, channels)

            scale = F.relu(self.fc1(scale))
            scale = torch.sigmoid(self.fc2(scale))

            if x.dim() == 3:
                scale = scale.unsqueeze(2)  # (batch, channels, 1)

            return x * scale

    class GainPanPredictor(nn.Module):
        """
        Neural network predicting gain (dB) and pan (-1..+1) from audio features.

        Architecture:
        - Input: feature vector (36 dims from extract_features)
        - FC(36, 128) -> BN -> ReLU -> SE
        - FC(128, 64) -> BN -> ReLU -> SE -> Dropout
        - FC(64, 32) -> ReLU
        - FC(32, 2) -> [gain_db (tanh scaled), pan (tanh)]

        The gain output is scaled to [-30, +12] dB range.
        The pan output is clamped to [-1, +1].
        """

        def __init__(self, input_dim=36, gain_range=(-30.0, 12.0)):
            """
            Args:
                input_dim: size of input feature vector
                gain_range: (min_db, max_db) output gain range
            """
            super().__init__()
            self.gain_min = gain_range[0]
            self.gain_max = gain_range[1]
            self.gain_center = (gain_range[0] + gain_range[1]) / 2.0
            self.gain_scale = (gain_range[1] - gain_range[0]) / 2.0

            self.fc1 = nn.Linear(input_dim, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.se1 = SqueezeExcitation(128, reduction=4)

            self.fc2 = nn.Linear(128, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.se2 = SqueezeExcitation(64, reduction=4)
            self.dropout = nn.Dropout(0.2)

            self.fc3 = nn.Linear(64, 32)
            self.fc_out = nn.Linear(32, 2)  # [gain, pan]

            # Initialize output layer with small weights
            nn.init.xavier_uniform_(self.fc_out.weight, gain=0.1)
            nn.init.zeros_(self.fc_out.bias)

        def forward(self, x):
            """
            Args:
                x: (batch, input_dim) feature vector

            Returns:
                gain_db: (batch, 1) predicted gain in dB
                pan: (batch, 1) predicted pan position
            """
            h = F.relu(self.bn1(self.fc1(x)))
            h = self.se1(h)

            h = F.relu(self.bn2(self.fc2(h)))
            h = self.se2(h)
            h = self.dropout(h)

            h = F.relu(self.fc3(h))
            out = self.fc_out(h)  # (batch, 2)

            # Scale outputs
            gain_raw = torch.tanh(out[:, 0:1])  # -1 to 1
            gain_db = gain_raw * self.gain_scale + self.gain_center

            pan = torch.tanh(out[:, 1:2])  # -1 to 1

            return gain_db, pan

        def predict(self, audio, sr=48000):
            """
            Predict gain and pan from raw audio.

            Args:
                audio: 1D numpy array
                sr: sample rate

            Returns:
                (gain_db, pan): tuple of floats
            """
            if extract_features is None:
                raise RuntimeError("channel_classifier.extract_features not available")

            self.eval()
            features = extract_features(audio, sr)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                gain_db, pan = self.forward(features_tensor)

            return float(gain_db.item()), float(pan.item())

        def train_step(self, batch, optimizer, loss_fn=None):
            """
            Perform a single training step.

            Args:
                batch: dict with keys:
                    'features': (batch, input_dim) tensor
                    'gain_db': (batch, 1) target gain
                    'pan': (batch, 1) target pan
                optimizer: torch optimizer
                loss_fn: optional custom loss (default: MSE)

            Returns:
                loss: float, total loss value
            """
            self.train()
            optimizer.zero_grad()

            features = batch["features"]
            target_gain = batch["gain_db"]
            target_pan = batch["pan"]

            pred_gain, pred_pan = self.forward(features)

            if loss_fn is not None:
                loss = loss_fn(pred_gain, target_gain, pred_pan, target_pan)
            else:
                gain_loss = F.mse_loss(pred_gain, target_gain)
                pan_loss = F.mse_loss(pred_pan, target_pan)
                # Weight gain loss higher as it's more critical
                loss = 2.0 * gain_loss + pan_loss

            loss.backward()
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            optimizer.step()

            return float(loss.item())

        def save_model(self, path):
            """Save model weights and architecture config."""
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            checkpoint = {
                "state_dict": self.state_dict(),
                "gain_min": self.gain_min,
                "gain_max": self.gain_max,
                "input_dim": self.fc1.in_features,
            }
            torch.save(checkpoint, path)
            logger.info(f"GainPanPredictor saved to {path}")

        def load_model(self, path):
            """Load model weights from file."""
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: {path}")

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.load_state_dict(checkpoint["state_dict"])
            self.gain_min = checkpoint.get("gain_min", -30.0)
            self.gain_max = checkpoint.get("gain_max", 12.0)
            self.gain_center = (self.gain_min + self.gain_max) / 2.0
            self.gain_scale = (self.gain_max - self.gain_min) / 2.0
            logger.info(f"GainPanPredictor loaded from {path}")

else:
    # Numpy-only fallback predictor

    class SqueezeExcitation:
        """Placeholder SE block for non-torch environments."""

        def __init__(self, channels, reduction=4):
            self.channels = channels

    class GainPanPredictor:
        """
        Numpy fallback gain/pan predictor using simple heuristics.
        Uses spectral analysis to estimate gain and stereo position.
        """

        def __init__(self, input_dim=36, gain_range=(-30.0, 12.0)):
            self.input_dim = input_dim
            self.gain_min = gain_range[0]
            self.gain_max = gain_range[1]
            logger.warning("GainPanPredictor using numpy fallback (no gradient support)")

        def predict(self, audio, sr=48000):
            """
            Predict gain and pan from raw audio using heuristics.

            Args:
                audio: 1D numpy array
                sr: sample rate

            Returns:
                (gain_db, pan): tuple of floats
            """
            audio = np.asarray(audio, dtype=np.float64)
            if len(audio) == 0:
                return -12.0, 0.0

            # Estimate appropriate gain from RMS
            rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
            rms_db = 20.0 * np.log10(rms + 1e-10)

            # Target RMS around -18 dBFS
            target_rms_db = -18.0
            gain_db = target_rms_db - rms_db
            gain_db = float(np.clip(gain_db, self.gain_min, self.gain_max))

            # Default center pan (no stereo information in mono)
            pan = 0.0

            return gain_db, pan

        def forward(self, x):
            """Numpy forward pass (returns zero-centered predictions)."""
            gain_db = np.zeros((x.shape[0], 1))
            pan = np.zeros((x.shape[0], 1))
            return gain_db, pan

        def train_step(self, batch, optimizer=None, loss_fn=None):
            """No-op training step for numpy fallback."""
            logger.warning("train_step called on numpy fallback - no training performed")
            return 0.0

        def save_model(self, path):
            """Save placeholder model."""
            np.savez(path, input_dim=self.input_dim,
                     gain_min=self.gain_min, gain_max=self.gain_max)
            logger.info(f"GainPanPredictor (numpy) saved to {path}")

        def load_model(self, path):
            """Load placeholder model."""
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: {path}")
            data = np.load(path, allow_pickle=True)
            self.input_dim = int(data.get("input_dim", 36))
            self.gain_min = float(data.get("gain_min", -30.0))
            self.gain_max = float(data.get("gain_max", 12.0))
            logger.info(f"GainPanPredictor (numpy) loaded from {path}")
