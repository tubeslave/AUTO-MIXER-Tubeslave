import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from live_runtime.feature_stream import (
    MainFeatureEvidence,
    Usb48FeatureExtractor,
    assemble_mix_features,
)


def _usb_block(frames=4096):
    sample_rate = 48_000
    t = np.arange(frames, dtype=np.float32) / sample_rate
    block = np.zeros((frames, 48), dtype=np.float32)
    block[:, 0] = 0.25 * np.sin(2.0 * np.pi * 1_000.0 * t)
    block[:, 47] = 0.10 * np.sin(2.0 * np.pi * 4_000.0 * t)
    return block


def test_usb48_extracts_selected_channel_evidence_without_decision_policy():
    extractor = Usb48FeatureExtractor()
    frame = extractor.extract(
        _usb_block(),
        selected_channels=[1, 48],
        channel_names={1: "Lead Vocal", 48: "Playback"},
        timestamp_s=12.0,
    )

    assert [item.channel for item in frame.channels] == [1, 48]
    assert [item.name for item in frame.channels] == ["Lead Vocal", "Playback"]
    assert frame.sample_rate == 48_000
    assert frame.source_channels == 48
    assert frame.timestamp_s == 12.0

    vocal, playback = frame.channels
    assert vocal.rms_dbfs < vocal.peak_dbfs < 0.0
    assert vocal.crest_db > 0.0
    assert vocal.activity == 1.0
    assert 900.0 < vocal.spectral_centroid_hz < 1_100.0
    assert vocal.harshness is not None
    assert 3_800.0 < playback.spectral_centroid_hz < 4_200.0


def test_usb48_silence_is_stable_and_does_not_invent_spectral_evidence():
    extractor = Usb48FeatureExtractor()
    frame = extractor.extract(
        np.zeros((1024, 48), dtype=np.float32),
        selected_channels=[7],
    )

    item = frame.channels[0]
    assert item.rms_dbfs == -120.0
    assert item.peak_dbfs == -120.0
    assert item.crest_db == 0.0
    assert item.activity == 0.0
    assert item.spectral_centroid_hz is None
    assert item.low_mid_ratio_db is None
    assert item.harshness is None


def test_usb48_rejects_invalid_transport_shape_and_nonfinite_audio():
    extractor = Usb48FeatureExtractor()

    with pytest.raises(ValueError, match="exactly 48"):
        extractor.extract(np.zeros((1024, 47), dtype=np.float32))

    bad = np.zeros((1024, 48), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        extractor.extract(bad)

    with pytest.raises(ValueError, match="outside 1..48"):
        extractor.extract(np.zeros((32, 48), dtype=np.float32), selected_channels=[49])


def test_usb_live_mvp_rejects_wrong_sample_rate_or_channel_count():
    with pytest.raises(ValueError, match="48000"):
        Usb48FeatureExtractor(sample_rate=44_100)
    with pytest.raises(ValueError, match="exactly 48"):
        Usb48FeatureExtractor(channel_count=32)


def test_mix_assembly_requires_explicit_coherent_main_evidence():
    frame = Usb48FeatureExtractor().extract(
        _usb_block(),
        selected_channels=[1],
        timestamp_s=20.0,
    )
    main = MainFeatureEvidence(
        rms_dbfs=-14.0,
        peak_dbfs=-3.0,
        crest_db=11.0,
        timestamp_s=20.1,
    )

    features = assemble_mix_features(frame, main)
    assert features.channels == frame.channels
    assert features.main_rms_dbfs == -14.0
    assert features.main_peak_dbfs == -3.0
    assert features.main_crest_db == 11.0
    assert features.timestamp_s == 20.0

    stale = MainFeatureEvidence(
        rms_dbfs=-14.0,
        peak_dbfs=-3.0,
        crest_db=11.0,
        timestamp_s=20.5,
    )
    with pytest.raises(ValueError, match="stale"):
        assemble_mix_features(frame, stale)


def test_mix_assembly_rejects_nonfinite_main_readback():
    frame = Usb48FeatureExtractor().extract(
        np.zeros((1024, 48), dtype=np.float32),
        selected_channels=[1],
        timestamp_s=1.0,
    )
    main = MainFeatureEvidence(
        rms_dbfs=-14.0,
        peak_dbfs=float("nan"),
        crest_db=11.0,
        timestamp_s=1.0,
    )
    with pytest.raises(ValueError, match="Main evidence"):
        assemble_mix_features(frame, main)
