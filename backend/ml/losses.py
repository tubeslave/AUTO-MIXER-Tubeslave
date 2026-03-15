"""
Mixing-specific loss functions for training differentiable mixing models.

Provides:
- MultiResolutionSTFTLoss: spectral convergence + log magnitude at multiple FFT sizes
- SumAndDifferenceLoss: stereo correlation loss on (L+R) and (L-R)
- MixingLoss: weighted combination of the above
"""

import numpy as np
import math

# Try PyTorch, fall back to numpy-only implementations
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _numpy_stft(signal, fft_size, hop_size, win_size):
    """Compute STFT magnitude using numpy (fallback when torch unavailable)."""
    if win_size is None:
        win_size = fft_size
    window = np.hanning(win_size)
    # Pad signal so we get full frames
    pad_len = fft_size // 2
    padded = np.pad(signal, (pad_len, pad_len), mode="reflect")
    num_frames = 1 + (len(padded) - win_size) // hop_size
    frames = np.zeros((num_frames, fft_size))
    for i in range(num_frames):
        start = i * hop_size
        end = start + win_size
        frame = padded[start:end]
        if len(frame) < fft_size:
            frame = np.pad(frame, (0, fft_size - len(frame)))
        frames[i, :win_size] = frame[:win_size] * window
    spectrum = np.fft.rfft(frames, n=fft_size, axis=1)
    magnitude = np.abs(spectrum)
    return magnitude


def _numpy_spectral_convergence(pred_mag, target_mag):
    """Spectral convergence loss (numpy)."""
    diff_norm = np.sqrt(np.sum((target_mag - pred_mag) ** 2))
    target_norm = np.sqrt(np.sum(target_mag ** 2))
    if target_norm < 1e-8:
        return 0.0
    return diff_norm / target_norm


def _numpy_log_magnitude_loss(pred_mag, target_mag):
    """Log STFT magnitude loss (numpy)."""
    eps = 1e-7
    log_pred = np.log(pred_mag + eps)
    log_target = np.log(target_mag + eps)
    return np.mean(np.abs(log_target - log_pred))


if HAS_TORCH:

    class _STFTModule(nn.Module):
        """Single-resolution STFT loss computation."""

        def __init__(self, fft_size, hop_size, win_size):
            super().__init__()
            self.fft_size = fft_size
            self.hop_size = hop_size
            self.win_size = win_size if win_size is not None else fft_size
            self.register_buffer(
                "window", torch.hann_window(self.win_size)
            )

        def forward(self, pred, target):
            """
            Args:
                pred: (batch, samples) predicted audio
                target: (batch, samples) target audio
            Returns:
                sc_loss: spectral convergence loss
                mag_loss: log magnitude loss
            """
            # Compute STFT
            pred_stft = torch.stft(
                pred,
                n_fft=self.fft_size,
                hop_length=self.hop_size,
                win_length=self.win_size,
                window=self.window,
                return_complex=True,
            )
            target_stft = torch.stft(
                target,
                n_fft=self.fft_size,
                hop_length=self.hop_size,
                win_length=self.win_size,
                window=self.window,
                return_complex=True,
            )
            pred_mag = torch.abs(pred_stft)
            target_mag = torch.abs(target_stft)

            # Spectral convergence
            sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (
                torch.norm(target_mag, p="fro") + 1e-8
            )

            # Log magnitude loss
            eps = 1e-7
            log_pred = torch.log(pred_mag + eps)
            log_target = torch.log(target_mag + eps)
            mag_loss = F.l1_loss(log_pred, log_target)

            return sc_loss, mag_loss

    class MultiResolutionSTFTLoss(nn.Module):
        """
        Multi-resolution STFT loss combining spectral convergence
        and log magnitude losses across multiple FFT sizes.

        Used for audio quality assessment in differentiable mixing pipelines.
        """

        def __init__(
            self,
            fft_sizes=(512, 1024, 2048),
            hop_sizes=None,
            win_sizes=None,
            sc_weight=1.0,
            mag_weight=1.0,
        ):
            super().__init__()
            if hop_sizes is None:
                hop_sizes = [s // 4 for s in fft_sizes]
            if win_sizes is None:
                win_sizes = list(fft_sizes)
            assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)

            self.sc_weight = sc_weight
            self.mag_weight = mag_weight
            self.stft_modules = nn.ModuleList(
                [
                    _STFTModule(fft_s, hop_s, win_s)
                    for fft_s, hop_s, win_s in zip(fft_sizes, hop_sizes, win_sizes)
                ]
            )

        def forward(self, pred, target):
            """
            Args:
                pred: (batch, samples) or (batch, channels, samples)
                target: same shape as pred
            Returns:
                total_loss: weighted sum of all STFT losses
            """
            # Flatten to (batch, samples) if multichannel
            if pred.dim() == 3:
                batch, channels, samples = pred.shape
                pred = pred.reshape(batch * channels, samples)
                target = target.reshape(batch * channels, samples)

            total_sc = 0.0
            total_mag = 0.0
            for stft_mod in self.stft_modules:
                sc, mag = stft_mod(pred, target)
                total_sc += sc
                total_mag += mag

            n = len(self.stft_modules)
            return self.sc_weight * (total_sc / n) + self.mag_weight * (total_mag / n)

    class SumAndDifferenceLoss(nn.Module):
        """
        Loss for stereo signals comparing the sum (L+R) and difference (L-R)
        channels. Ensures correct stereo width and center-image balance.
        """

        def __init__(self, sum_weight=1.0, diff_weight=1.0):
            super().__init__()
            self.sum_weight = sum_weight
            self.diff_weight = diff_weight

        def forward(self, pred_stereo, target_stereo):
            """
            Args:
                pred_stereo: (batch, 2, samples) - predicted stereo audio
                target_stereo: (batch, 2, samples) - target stereo audio
            Returns:
                loss: weighted sum of L1 losses on sum and difference signals
            """
            # Extract left and right
            pred_left = pred_stereo[:, 0, :]
            pred_right = pred_stereo[:, 1, :]
            target_left = target_stereo[:, 0, :]
            target_right = target_stereo[:, 1, :]

            # Sum and difference
            pred_sum = pred_left + pred_right
            pred_diff = pred_left - pred_right
            target_sum = target_left + target_right
            target_diff = target_left - target_right

            sum_loss = F.l1_loss(pred_sum, target_sum)
            diff_loss = F.l1_loss(pred_diff, target_diff)

            return self.sum_weight * sum_loss + self.diff_weight * diff_loss

    class MixingLoss(nn.Module):
        """
        Combined mixing loss for training differentiable mixing models.
        Weights spectral STFT loss with stereo sum/difference loss.
        """

        def __init__(
            self,
            fft_sizes=(512, 1024, 2048),
            spectral_weight=1.0,
            stereo_weight=0.5,
            sum_weight=1.0,
            diff_weight=1.0,
            sc_weight=1.0,
            mag_weight=1.0,
        ):
            super().__init__()
            self.spectral_weight = spectral_weight
            self.stereo_weight = stereo_weight
            self.stft_loss = MultiResolutionSTFTLoss(
                fft_sizes=fft_sizes,
                sc_weight=sc_weight,
                mag_weight=mag_weight,
            )
            self.stereo_loss = SumAndDifferenceLoss(
                sum_weight=sum_weight,
                diff_weight=diff_weight,
            )

        def forward(self, pred, target, is_stereo=True):
            """
            Args:
                pred: predicted audio (batch, channels, samples) or (batch, samples)
                target: target audio, same shape
                is_stereo: if True, also compute stereo loss
            Returns:
                total_loss: combined loss
                loss_dict: breakdown of individual loss components
            """
            # STFT loss on mono or multichannel
            spec_loss = self.stft_loss(pred, target)

            loss_dict = {"spectral": spec_loss.item()}
            total = self.spectral_weight * spec_loss

            if is_stereo and pred.dim() == 3 and pred.shape[1] == 2:
                st_loss = self.stereo_loss(pred, target)
                loss_dict["stereo"] = st_loss.item()
                total = total + self.stereo_weight * st_loss
            else:
                loss_dict["stereo"] = 0.0

            loss_dict["total"] = total.item()
            return total, loss_dict

else:
    # Numpy fallback implementations (non-differentiable, for evaluation only)

    class MultiResolutionSTFTLoss:
        """Numpy fallback for multi-resolution STFT loss (non-differentiable)."""

        def __init__(
            self,
            fft_sizes=(512, 1024, 2048),
            hop_sizes=None,
            win_sizes=None,
            sc_weight=1.0,
            mag_weight=1.0,
        ):
            self.fft_sizes = list(fft_sizes)
            self.hop_sizes = hop_sizes or [s // 4 for s in fft_sizes]
            self.win_sizes = win_sizes or list(fft_sizes)
            self.sc_weight = sc_weight
            self.mag_weight = mag_weight

        def __call__(self, pred, target):
            """
            Args:
                pred: numpy array (samples,) or (channels, samples)
                target: numpy array, same shape
            Returns:
                total_loss: float
            """
            pred = np.atleast_2d(pred)
            target = np.atleast_2d(target)

            total_sc = 0.0
            total_mag = 0.0
            for fft_s, hop_s, win_s in zip(
                self.fft_sizes, self.hop_sizes, self.win_sizes
            ):
                for ch in range(pred.shape[0]):
                    pred_mag = _numpy_stft(pred[ch], fft_s, hop_s, win_s)
                    target_mag = _numpy_stft(target[ch], fft_s, hop_s, win_s)
                    total_sc += _numpy_spectral_convergence(pred_mag, target_mag)
                    total_mag += _numpy_log_magnitude_loss(pred_mag, target_mag)

            n = len(self.fft_sizes) * pred.shape[0]
            return self.sc_weight * (total_sc / n) + self.mag_weight * (total_mag / n)

    class SumAndDifferenceLoss:
        """Numpy fallback for stereo sum/difference loss."""

        def __init__(self, sum_weight=1.0, diff_weight=1.0):
            self.sum_weight = sum_weight
            self.diff_weight = diff_weight

        def __call__(self, pred_stereo, target_stereo):
            """
            Args:
                pred_stereo: (2, samples)
                target_stereo: (2, samples)
            Returns:
                loss: float
            """
            pred_sum = pred_stereo[0] + pred_stereo[1]
            pred_diff = pred_stereo[0] - pred_stereo[1]
            target_sum = target_stereo[0] + target_stereo[1]
            target_diff = target_stereo[0] - target_stereo[1]

            sum_loss = np.mean(np.abs(pred_sum - target_sum))
            diff_loss = np.mean(np.abs(pred_diff - target_diff))

            return self.sum_weight * sum_loss + self.diff_weight * diff_loss

    class MixingLoss:
        """Numpy fallback for combined mixing loss."""

        def __init__(
            self,
            fft_sizes=(512, 1024, 2048),
            spectral_weight=1.0,
            stereo_weight=0.5,
            sum_weight=1.0,
            diff_weight=1.0,
            sc_weight=1.0,
            mag_weight=1.0,
        ):
            self.spectral_weight = spectral_weight
            self.stereo_weight = stereo_weight
            self.stft_loss = MultiResolutionSTFTLoss(
                fft_sizes=fft_sizes,
                sc_weight=sc_weight,
                mag_weight=mag_weight,
            )
            self.stereo_loss = SumAndDifferenceLoss(
                sum_weight=sum_weight,
                diff_weight=diff_weight,
            )

        def __call__(self, pred, target, is_stereo=True):
            """
            Args:
                pred: numpy array
                target: numpy array
                is_stereo: bool
            Returns:
                total_loss: float
                loss_dict: dict
            """
            spec_loss = self.stft_loss(pred, target)
            loss_dict = {"spectral": float(spec_loss)}
            total = self.spectral_weight * spec_loss

            if is_stereo and pred.ndim >= 2 and pred.shape[0] == 2:
                st_loss = self.stereo_loss(pred, target)
                loss_dict["stereo"] = float(st_loss)
                total += self.stereo_weight * st_loss
            else:
                loss_dict["stereo"] = 0.0

            loss_dict["total"] = float(total)
            return total, loss_dict
