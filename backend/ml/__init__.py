"""
Machine Learning modules for AUTO-MIXER-Tubeslave.
All PyTorch-dependent imports are wrapped in try/except so the rest
of the application can load without GPU/torch installed.
"""
from .lufs_targets import LUFSTargetManager, LUFSTarget, GENRE_TARGETS, INSTRUMENT_PROFILES
from .drc_onset import DRCOnsetDetector, OnsetEvent
from .mix_quality import MixQualityAnalyzer, MixQualityScore
from .eq_normalization import EQNormalizer, EQProfile, EQBand, TARGET_CURVES
from .subgroup_mixer import SubgroupMixer, SubgroupConfig, DEFAULT_SUBGROUPS
from .reference_profiles import ReferenceProfileManager, ReferenceProfile

try:
    from .losses import MultiResolutionSTFTLoss, LoudnessLoss, MixConsistencyLoss
    from .differentiable_console import DifferentiableMixingConsole
    from .channel_classifier import ChannelClassifier, ChannelClassifierNet, INSTRUMENT_CLASSES
    from .gain_pan_predictor import GainPanPredictor, GainPanPredictorNet
    from .agent_training_service import AgentTrainingService
except ImportError:
    pass
