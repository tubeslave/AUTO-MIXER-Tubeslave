import pytest


def test_classifier_training_rejects_resource_hyperparameters_before_allocation(monkeypatch):
    from ml import train_classifier as training

    def fail_if_dataset_allocates(*args, **kwargs):
        raise AssertionError("dataset allocation must not run for invalid hyperparameters")

    monkeypatch.setattr(training, "SyntheticInstrumentDataset", fail_if_dataset_allocates)
    invalid_requests = [
        {"n_epochs": 10**9},
        {"batch_size": 10**9},
        {"n_samples_per_class": 10**9},
        {"n_mels": 10**9},
        {"n_frames": 10**9},
        {"lr": float("nan")},
        {"n_epochs": 200, "n_samples_per_class": 1000},
        {"batch_size": 256, "n_mels": 256, "n_frames": 512},
    ]
    for overrides in invalid_requests:
        with pytest.raises(ValueError):
            training.train_classifier(**overrides)


def test_classifier_synthetic_dataset_enforces_joint_budget_and_keeps_small_input():
    from ml.train_classifier import SyntheticInstrumentDataset

    with pytest.raises(ValueError, match="synthetic dataset allocation limit exceeded"):
        SyntheticInstrumentDataset(n_samples_per_class=300, n_mels=256, n_frames=512)

    with pytest.raises(ValueError, match="dataset sample limit exceeded"):
        SyntheticInstrumentDataset(n_samples_per_class=1000, n_mels=8, n_frames=8)

    dataset = SyntheticInstrumentDataset(n_samples_per_class=1, n_mels=8, n_frames=8)
    assert len(dataset) > 1
    assert tuple(dataset[0][0].shape) == (1, 8, 8)


def test_mix_console_training_rejects_resource_hyperparameters_before_allocation(monkeypatch):
    from ml import train_mix_console as training

    def fail_if_dataset_allocates(*args, **kwargs):
        raise AssertionError("dataset allocation must not run for invalid hyperparameters")

    monkeypatch.setattr(training, "generate_synthetic_multitracks", fail_if_dataset_allocates)
    invalid_requests = [
        {"n_epochs": 10**9},
        {"n_channels": 10**9},
        {"audio_len": 10**9},
        {"n_samples": 10**9},
        {"sample_rate": 10**9},
        {"lr": float("inf")},
        {"n_epochs": 200, "n_samples": 4096},
        {"n_samples": 4096, "n_channels": 64, "audio_len": 262_144},
    ]
    for overrides in invalid_requests:
        with pytest.raises(ValueError):
            training.train_mix_console(**overrides)


def test_mix_console_generator_enforces_joint_budget_and_keeps_small_input():
    from ml.train_mix_console import generate_synthetic_multitracks

    with pytest.raises(ValueError, match="synthetic dataset allocation limit exceeded"):
        generate_synthetic_multitracks(4096, 64, 262_144)

    multitracks, references = generate_synthetic_multitracks(2, 1, 32)
    assert len(multitracks) == len(references) == 2
    assert multitracks[0].shape == (1, 32)
    assert references[0].shape == (32,)
