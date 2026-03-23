"""
Channel classifier -- identifies instrument type from audio features.
Uses a lightweight CNN on mel-spectrograms.
Based on approaches from Essentia/AudioSet instrument classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

INSTRUMENT_CLASSES = [
    'kick', 'snare', 'hi_hat', 'toms', 'overheads', 'room_mics',
    'bass_guitar', 'electric_guitar', 'acoustic_guitar',
    'keys_piano', 'synth', 'organ',
    'lead_vocal', 'backing_vocal', 'choir',
    'brass', 'woodwind', 'strings',
    'percussion', 'dj_playback', 'click_track',
    'ambient_mic', 'audience', 'unknown'
]

NUM_CLASSES = len(INSTRUMENT_CLASSES)


class ChannelClassifierNet(nn.Module):
    """Lightweight CNN for instrument classification from mel spectrograms."""

    def __init__(self, n_mels: int = 64, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class ChannelClassifier:
    """High-level classifier interface."""

    def __init__(self, model_path: Optional[str] = None, sample_rate: int = 48000,
                 n_mels: int = 64):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ChannelClassifierNet(n_mels=n_mels).to(self.device)
        self.model.eval()
        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def audio_to_mel(self, audio: np.ndarray) -> torch.Tensor:
        n_fft = 1024
        hop = 512
        audio_t = torch.from_numpy(audio.astype(np.float32)).to(self.device)
        if audio_t.dim() == 1:
            audio_t = audio_t.unsqueeze(0)
        spec = torch.stft(audio_t, n_fft, hop, n_fft,
                          torch.hann_window(n_fft).to(self.device),
                          return_complex=True)
        mag = torch.abs(spec)
        mel_basis = self._mel_filterbank(n_fft, self.n_mels).to(self.device)
        mel = torch.matmul(mel_basis, mag)
        mel_db = 20 * torch.log10(mel + 1e-8)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return mel_db

    def _mel_filterbank(self, n_fft: int, n_mels: int) -> torch.Tensor:
        low_freq = 0
        high_freq = self.sample_rate / 2
        mel_low = 2595 * np.log10(1 + low_freq / 700)
        mel_high = 2595 * np.log10(1 + high_freq / 700)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bins = np.floor((n_fft + 1) * hz_points / self.sample_rate).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(n_mels):
            for j in range(bins[i], bins[i + 1]):
                if bins[i + 1] > bins[i]:
                    fb[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
            for j in range(bins[i + 1], bins[i + 2]):
                if bins[i + 2] > bins[i + 1]:
                    fb[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])
        return torch.from_numpy(fb.astype(np.float32))

    def classify(self, audio: np.ndarray) -> Dict[str, float]:
        mel = self.audio_to_mel(audio)
        with torch.no_grad():
            logits = self.model(mel)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        return {cls: float(probs[i]) for i, cls in enumerate(INSTRUMENT_CLASSES)}

    def classify_top_k(self, audio: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        results = self.classify(audio)
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
