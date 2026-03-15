"""
AUTO MIXER Tubeslave - Machine Learning Module
===============================================
ML components for live concert automatic mixing:
- Channel classification (instrument type detection)
- Gain/pan prediction via neural networks
- Differentiable mixing console for gradient-based optimization
- Mix quality metrics
- EQ normalization and reference profiles
- Adaptive DRC with onset detection
- Subgroup/bus mixing hierarchy
"""

# Core utilities (always available - numpy only)
from .lufs_targets import get_target_lufs, INSTRUMENT_LUFS_OFFSETS, GENRE_MODIFIERS
from .reference_profiles import get_profile, PROFILES
from .drc_onset import compute_onset_threshold, get_attack_release
from .mix_quality import MixQualityMetric
from .eq_normalization import compute_spectral_profile, compute_correction

# ML models (may require sklearn/librosa)
try:
    from .channel_classifier import ChannelClassifier
except ImportError:
    ChannelClassifier = None

# PyTorch-based modules (optional)
try:
    from .losses import MixingLoss, MultiResolutionSTFTLoss, SumAndDifferenceLoss
except ImportError:
    MixingLoss = None
    MultiResolutionSTFTLoss = None
    SumAndDifferenceLoss = None

try:
    from .differentiable_console import DifferentiableMixingConsole
except ImportError:
    DifferentiableMixingConsole = None

try:
    from .gain_pan_predictor import GainPanPredictor
except ImportError:
    GainPanPredictor = None

try:
    from .subgroup_mixer import SubgroupMixer
except ImportError:
    SubgroupMixer = None

# Phase 8: Style transfer & processing graph (numpy-based, scipy optional)
try:
    from .neural_mix_extractor import NeuralMixExtractor
except ImportError:
    NeuralMixExtractor = None

try:
    from .style_transfer import StyleTransfer, StyleProfile
except ImportError:
    StyleTransfer = None
    StyleProfile = None

try:
    from .processing_graph import (
        ProcessingGraph,
        ProcessingNode,
        HPFNode,
        GateNode,
        EQNode,
        CompressorNode,
        FaderNode,
        PanNode,
        BusSendNode,
    )
except ImportError:
    ProcessingGraph = None
    ProcessingNode = None
    HPFNode = None
    GateNode = None
    EQNode = None
    CompressorNode = None
    FaderNode = None
    PanNode = None
    BusSendNode = None

__all__ = [
    "get_target_lufs",
    "INSTRUMENT_LUFS_OFFSETS",
    "GENRE_MODIFIERS",
    "get_profile",
    "PROFILES",
    "compute_onset_threshold",
    "get_attack_release",
    "MixQualityMetric",
    "compute_spectral_profile",
    "compute_correction",
    "ChannelClassifier",
    "MixingLoss",
    "MultiResolutionSTFTLoss",
    "SumAndDifferenceLoss",
    "DifferentiableMixingConsole",
    "GainPanPredictor",
    "SubgroupMixer",
    "NeuralMixExtractor",
    "StyleTransfer",
    "StyleProfile",
    "ProcessingGraph",
    "ProcessingNode",
    "HPFNode",
    "GateNode",
    "EQNode",
    "CompressorNode",
    "FaderNode",
    "PanNode",
    "BusSendNode",
]
