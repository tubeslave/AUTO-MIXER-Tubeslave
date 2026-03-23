"""
Advanced Compressor Profile System with Context-Aware Selection.

Provides instrument-specific profiles with context factors:
- Tempo (BPM) - affects attack/release timing
- Genre/Style - affects ratio, threshold, aggression
- Signal characteristics - affects parameter adaptation
- Task type - base, punch, control, gentle, aggressive, broadcast
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from signal_analysis import ChannelSignalFeatures

logger = logging.getLogger(__name__)


class Genre(Enum):
    """Musical genres with compression characteristics."""
    ROCK = "rock"
    POP = "pop"
    JAZZ = "jazz"
    CLASSICAL = "classical"
    ELECTRONIC = "electronic"
    METAL = "metal"
    COUNTRY = "country"
    BLUES = "blues"
    FOLK = "folk"
    REGGAE = "reggae"
    HIP_HOP = "hip_hop"
    R_AND_B = "r_and_b"
    LATIN = "latin"
    UNKNOWN = "unknown"


class Style(Enum):
    """Performance styles affecting compression."""
    ACOUSTIC = "acoustic"
    ELECTRIC = "electric"
    LIVE = "live"
    STUDIO = "studio"
    BROADCAST = "broadcast"
    LOUD = "loud"
    INTIMATE = "intimate"
    ENERGETIC = "energetic"
    SMOOTH = "smooth"


class TaskType(Enum):
    """Compression task types."""
    BASE = "base"
    PUNCH = "punch"
    CONTROL = "control"
    GENTLE = "gentle"
    AGGRESSIVE = "aggressive"
    BROADCAST = "broadcast"
    TRANSIENT = "transient"
    GLUE = "glue"


@dataclass
class ProfileContext:
    """Context information for profile selection."""
    instrument: str = "custom"
    task: TaskType = TaskType.BASE
    genre: Genre = Genre.UNKNOWN
    style: Style = Style.LIVE
    bpm: Optional[float] = None
    mix_density: float = 1.0  # 0.0 (sparse) to 2.0 (very dense)
    target_lufs: Optional[float] = None
    features: Optional[ChannelSignalFeatures] = None


@dataclass
class CompressorProfile:
    """Complete compressor profile with base parameters and adaptation rules."""
    name: str
    instrument: str
    task: TaskType
    
    # Base parameters
    threshold_db: float = -15.0
    ratio: float = 3.0
    attack_ms: float = 15.0
    release_ms: float = 150.0
    knee: int = 2
    makeup_gain_db: float = 0.0
    
    # Adaptation rules
    tempo_sensitive: bool = True  # Adjust attack/release based on BPM
    genre_modifiers: Dict[str, float] = field(default_factory=dict)  # Genre -> ratio multiplier
    style_modifiers: Dict[str, float] = field(default_factory=dict)  # Style -> threshold offset
    
    # Signal-based adaptation
    dynamic_range_target_db: float = 12.0  # Target dynamic range
    crest_factor_target_db: float = 10.0  # Target crest factor
    
    # BPM-based timing (if tempo_sensitive)
    attack_bpm_factor: float = 0.5  # Multiplier for BPM-based attack calculation
    release_bpm_factor: float = 2.0  # Multiplier for BPM-based release calculation
    
    # Description
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "instrument": self.instrument,
            "task": self.task.value,
            "threshold_db": self.threshold_db,
            "ratio": self.ratio,
            "attack_ms": self.attack_ms,
            "release_ms": self.release_ms,
            "knee": self.knee,
            "makeup_gain_db": self.makeup_gain_db,
            "tempo_sensitive": self.tempo_sensitive,
            "genre_modifiers": self.genre_modifiers,
            "style_modifiers": self.style_modifiers,
            "dynamic_range_target_db": self.dynamic_range_target_db,
            "crest_factor_target_db": self.crest_factor_target_db,
            "description": self.description,
        }


class ProfileLibrary:
    """Library of compressor profiles organized by instrument and context."""
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, CompressorProfile]] = {}  # instrument -> task -> profile
        self._load_default_profiles()
    
    def _load_default_profiles(self):
        """Load default profiles for common instruments."""
        
        # Kick Drum Profiles
        self._add_profile(CompressorProfile(
            name="Kick Base",
            instrument="kick",
            task=TaskType.BASE,
            threshold_db=-12.0,
            ratio=4.0,
            attack_ms=20.0,
            release_ms=80.0,
            knee=2,
            tempo_sensitive=True,
            genre_modifiers={"rock": 1.2, "metal": 1.3, "jazz": 0.8, "pop": 1.1},
            style_modifiers={"live": 0.0, "studio": -2.0},
            dynamic_range_target_db=8.0,
            description="Standard kick compression for most genres"
        ))
        
        self._add_profile(CompressorProfile(
            name="Kick Punch",
            instrument="kick",
            task=TaskType.PUNCH,
            threshold_db=-10.0,
            ratio=5.0,
            attack_ms=25.0,
            release_ms=70.0,
            knee=2,
            tempo_sensitive=True,
            genre_modifiers={"rock": 1.3, "metal": 1.4},
            dynamic_range_target_db=6.0,
            description="Aggressive kick compression for punch and attack"
        ))
        
        # Snare Profiles
        self._add_profile(CompressorProfile(
            name="Snare Base",
            instrument="snare",
            task=TaskType.BASE,
            threshold_db=-10.0,
            ratio=4.5,
            attack_ms=5.0,
            release_ms=100.0,
            knee=2,
            tempo_sensitive=True,
            genre_modifiers={"rock": 1.2, "metal": 1.3, "jazz": 0.7},
            dynamic_range_target_db=10.0,
            description="Standard snare compression"
        ))
        
        self._add_profile(CompressorProfile(
            name="Snare Transient",
            instrument="snare",
            task=TaskType.TRANSIENT,
            threshold_db=-8.0,
            ratio=4.0,
            attack_ms=8.0,
            release_ms=80.0,
            knee=2,
            tempo_sensitive=True,
            description="Preserve snare transients while controlling body"
        ))
        
        # Bass Profiles
        self._add_profile(CompressorProfile(
            name="Bass Base",
            instrument="bass",
            task=TaskType.BASE,
            threshold_db=-15.0,
            ratio=4.0,
            attack_ms=25.0,
            release_ms=200.0,
            knee=2,
            tempo_sensitive=True,
            genre_modifiers={"rock": 1.2, "metal": 1.3, "jazz": 0.8, "hip_hop": 1.1},
            dynamic_range_target_db=10.0,
            description="Standard bass compression"
        ))
        
        self._add_profile(CompressorProfile(
            name="Bass Glue",
            instrument="bass",
            task=TaskType.GLUE,
            threshold_db=-18.0,
            ratio=3.0,
            attack_ms=30.0,
            release_ms=250.0,
            knee=1,
            tempo_sensitive=True,
            description="Gentle bass compression for mix glue"
        ))
        
        # Lead Vocal Profiles
        self._add_profile(CompressorProfile(
            name="Vocal Base",
            instrument="leadVocal",
            task=TaskType.BASE,
            threshold_db=-18.0,
            ratio=3.0,
            attack_ms=12.0,
            release_ms=150.0,
            knee=2,
            tempo_sensitive=True,
            genre_modifiers={"pop": 1.2, "rock": 1.1, "jazz": 0.8, "hip_hop": 1.3},
            style_modifiers={"live": 0.0, "broadcast": -2.0, "studio": -1.0},
            dynamic_range_target_db=8.0,
            description="Standard lead vocal compression"
        ))
        
        self._add_profile(CompressorProfile(
            name="Vocal Broadcast",
            instrument="leadVocal",
            task=TaskType.BROADCAST,
            threshold_db=-20.0,
            ratio=4.0,
            attack_ms=8.0,
            release_ms=180.0,
            knee=3,
            tempo_sensitive=False,
            description="Broadcast-ready vocal compression with high consistency"
        ))
        
        self._add_profile(CompressorProfile(
            name="Vocal Control",
            instrument="leadVocal",
            task=TaskType.CONTROL,
            threshold_db=-22.0,
            ratio=4.0,
            attack_ms=10.0,
            release_ms=200.0,
            knee=3,
            tempo_sensitive=True,
            description="Tight vocal control for dynamic performances"
        ))
        
        # Electric Guitar Profiles
        self._add_profile(CompressorProfile(
            name="Guitar Base",
            instrument="electricGuitar",
            task=TaskType.BASE,
            threshold_db=-14.0,
            ratio=3.0,
            attack_ms=15.0,
            release_ms=180.0,
            knee=2,
            tempo_sensitive=True,
            genre_modifiers={"rock": 1.2, "metal": 1.3, "jazz": 0.8},
            description="Standard electric guitar compression"
        ))
        
        # Acoustic Guitar Profiles
        self._add_profile(CompressorProfile(
            name="Acoustic Guitar Base",
            instrument="acousticGuitar",
            task=TaskType.BASE,
            threshold_db=-18.0,
            ratio=3.0,
            attack_ms=15.0,
            release_ms=150.0,
            knee=2,
            tempo_sensitive=True,
            genre_modifiers={"folk": 0.9, "country": 1.0, "pop": 1.1},
            description="Standard acoustic guitar compression"
        ))
        
        # Backing Vocals
        self._add_profile(CompressorProfile(
            name="Back Vocal Base",
            instrument="backVocal",
            task=TaskType.BASE,
            threshold_db=-20.0,
            ratio=3.0,
            attack_ms=15.0,
            release_ms=180.0,
            knee=2,
            tempo_sensitive=True,
            description="Backing vocal compression"
        ))
        
        # Synth/Keys
        self._add_profile(CompressorProfile(
            name="Synth Base",
            instrument="synth",
            task=TaskType.BASE,
            threshold_db=-16.0,
            ratio=3.0,
            attack_ms=15.0,
            release_ms=150.0,
            knee=2,
            tempo_sensitive=True,
            genre_modifiers={"electronic": 1.2, "pop": 1.1},
            description="Synthesizer/keyboard compression"
        ))
        
        # Custom fallback
        self._add_profile(CompressorProfile(
            name="Custom Base",
            instrument="custom",
            task=TaskType.BASE,
            threshold_db=-15.0,
            ratio=3.0,
            attack_ms=15.0,
            release_ms=150.0,
            knee=2,
            tempo_sensitive=True,
            description="Default compression profile"
        ))
    
    def _add_profile(self, profile: CompressorProfile):
        """Add profile to library."""
        if profile.instrument not in self.profiles:
            self.profiles[profile.instrument] = {}
        self.profiles[profile.instrument][profile.task.value] = profile
    
    def get_profile(self, instrument: str, task: TaskType = TaskType.BASE) -> CompressorProfile:
        """Get profile for instrument and task, with fallback."""
        if instrument in self.profiles:
            if task.value in self.profiles[instrument]:
                return self.profiles[instrument][task.value]
            # Fallback to base task
            if TaskType.BASE.value in self.profiles[instrument]:
                return self.profiles[instrument][TaskType.BASE.value]
        
        # Fallback to custom
        if "custom" in self.profiles and TaskType.BASE.value in self.profiles["custom"]:
            return self.profiles["custom"][TaskType.BASE.value]
        
        # Last resort: create default
        return CompressorProfile(
            name="Default",
            instrument=instrument,
            task=task
        )
    
    def get_available_tasks(self, instrument: str) -> List[str]:
        """Get list of available tasks for instrument."""
        if instrument in self.profiles:
            return list(self.profiles[instrument].keys())
        return [TaskType.BASE.value]


class ProfileSelector:
    """Framework for selecting and adapting compressor profiles based on context."""
    
    def __init__(self, library: Optional[ProfileLibrary] = None):
        self.library = library or ProfileLibrary()
    
    def select_profile(
        self,
        instrument: str,
        context: ProfileContext,
        features: Optional[ChannelSignalFeatures] = None
    ) -> CompressorProfile:
        """
        Select appropriate profile based on instrument, context, and signal features.
        
        Args:
            instrument: Instrument type (from recognition)
            context: Profile context (task, genre, style, BPM, etc.)
            features: Signal features for intelligent selection
        
        Returns:
            Selected and adapted CompressorProfile
        """
        # Get base profile
        profile = self.library.get_profile(instrument, context.task)
        
        # Adapt based on context
        adapted = self._adapt_profile(profile, context, features)
        
        return adapted
    
    def _adapt_profile(
        self,
        profile: CompressorProfile,
        context: ProfileContext,
        features: Optional[ChannelSignalFeatures]
    ) -> CompressorProfile:
        """Adapt profile parameters based on context and signal features."""
        # Create a copy to avoid modifying original
        adapted = CompressorProfile(
            name=profile.name,
            instrument=profile.instrument,
            task=profile.task,
            threshold_db=profile.threshold_db,
            ratio=profile.ratio,
            attack_ms=profile.attack_ms,
            release_ms=profile.release_ms,
            knee=profile.knee,
            makeup_gain_db=profile.makeup_gain_db,
            tempo_sensitive=profile.tempo_sensitive,
            genre_modifiers=profile.genre_modifiers.copy(),
            style_modifiers=profile.style_modifiers.copy(),
            dynamic_range_target_db=profile.dynamic_range_target_db,
            crest_factor_target_db=profile.crest_factor_target_db,
            description=profile.description
        )
        
        # Apply genre modifier to ratio
        if context.genre != Genre.UNKNOWN and context.genre.value in adapted.genre_modifiers:
            multiplier = adapted.genre_modifiers[context.genre.value]
            adapted.ratio *= multiplier
        
        # Apply style modifier to threshold
        if context.style.value in adapted.style_modifiers:
            offset = adapted.style_modifiers[context.style.value]
            adapted.threshold_db += offset
        
        # Apply BPM-based timing adjustments
        if adapted.tempo_sensitive and context.bpm and context.bpm > 0:
            # Attack: faster tempo = slightly faster attack (but not too much)
            bpm_factor = min(1.5, max(0.7, 120.0 / context.bpm))
            adapted.attack_ms *= bpm_factor
            
            # Release: faster tempo = shorter release (musical timing)
            # Target: release should be around 1/16 note at song tempo
            release_base_ms = 60000.0 / context.bpm / 4.0  # 1/16 note
            if adapted.release_ms > release_base_ms * 2:
                adapted.release_ms = release_base_ms * 1.5
            elif adapted.release_ms < release_base_ms * 0.5:
                adapted.release_ms = release_base_ms * 0.75
        
        # Apply mix density modifier
        if context.mix_density > 1.0:
            # Denser mix: lower threshold, higher ratio
            density_factor = min(1.3, context.mix_density)
            adapted.threshold_db -= (density_factor - 1.0) * 3.0
            adapted.ratio *= density_factor
        elif context.mix_density < 1.0:
            # Sparse mix: higher threshold, lower ratio
            adapted.threshold_db += (1.0 - context.mix_density) * 2.0
            adapted.ratio *= max(0.8, context.mix_density)
        
        # Apply signal-based adaptations if features available
        if features:
            # Adjust threshold based on signal level
            signal_level = max(features.rms_db, features.lufs_momentary)
            if signal_level > -30:
                # Hot signal: raise threshold slightly
                adapted.threshold_db += 2.0
            elif signal_level < -50:
                # Quiet signal: lower threshold
                adapted.threshold_db -= 3.0
            
            # Adjust ratio based on dynamic range
            if features.dynamic_range_db > adapted.dynamic_range_target_db * 1.5:
                # Very dynamic: increase ratio
                adapted.ratio *= 1.2
            elif features.dynamic_range_db < adapted.dynamic_range_target_db * 0.7:
                # Less dynamic: decrease ratio
                adapted.ratio *= 0.9
            
            # Adjust attack based on transient characteristics
            if features.transient_strength > 0.7:
                # Strong transients: preserve with slower attack
                adapted.attack_ms *= 1.3
            elif features.transient_strength < 0.3:
                # Weak transients: can use faster attack
                adapted.attack_ms *= 0.8
        
        return adapted
    
    def detect_genre_from_features(
        self,
        features: ChannelSignalFeatures,
        instrument: str
    ) -> Genre:
        """
        Attempt to detect genre from signal features.
        This is a simple heuristic - can be enhanced with ML.
        """
        # Simple heuristics based on signal characteristics
        if instrument == "kick":
            if features.transient_strength > 0.8 and features.attack_time_ms < 5:
                return Genre.METAL
            elif features.transient_strength > 0.6:
                return Genre.ROCK
            else:
                return Genre.POP
        
        if instrument == "leadVocal":
            if features.spectral_centroid_hz > 3000:
                return Genre.POP
            elif features.dynamic_range_db > 20:
                return Genre.JAZZ
            else:
                return Genre.ROCK
        
        return Genre.UNKNOWN
    
    def detect_task_from_context(
        self,
        instrument: str,
        features: Optional[ChannelSignalFeatures],
        genre: Genre,
        style: Style
    ) -> TaskType:
        """
        Intelligently select task type based on context.
        """
        # Style-based selection
        if style == Style.BROADCAST:
            if instrument == "leadVocal":
                return TaskType.BROADCAST
        
        # Genre-based selection
        if genre in [Genre.METAL, Genre.ROCK]:
            if instrument in ["kick", "snare"]:
                return TaskType.PUNCH
        
        # Signal-based selection
        if features:
            if features.transient_strength > 0.7:
                if instrument in ["kick", "snare"]:
                    return TaskType.TRANSIENT
            
            if features.dynamic_range_db > 25:
                return TaskType.CONTROL
        
        return TaskType.BASE


# Global instance
_profile_library = ProfileLibrary()
_profile_selector = ProfileSelector(_profile_library)


def get_profile_for_context(
    instrument: str,
    task: str = "base",
    genre: str = "unknown",
    style: str = "live",
    bpm: Optional[float] = None,
    mix_density: float = 1.0,
    features: Optional[ChannelSignalFeatures] = None
) -> Dict[str, Any]:
    """
    High-level function to get adapted compressor profile.
    
    Args:
        instrument: Instrument type
        task: Task type (base, punch, control, etc.)
        genre: Genre (rock, pop, jazz, etc.)
        style: Style (live, studio, broadcast, etc.)
        bpm: Tempo in BPM
        mix_density: Mix density factor (0.0-2.0)
        features: Signal features for adaptation
    
    Returns:
        Dictionary with compressor parameters ready for use
    """
    try:
        task_enum = TaskType(task.lower())
    except ValueError:
        task_enum = TaskType.BASE
    
    try:
        genre_enum = Genre(genre.lower())
    except ValueError:
        genre_enum = Genre.UNKNOWN
    
    try:
        style_enum = Style(style.lower())
    except ValueError:
        style_enum = Style.LIVE
    
    context = ProfileContext(
        instrument=instrument,
        task=task_enum,
        genre=genre_enum,
        style=style_enum,
        bpm=bpm,
        mix_density=mix_density,
        features=features
    )
    
    profile = _profile_selector.select_profile(instrument, context, features)
    
    # Convert to dict format compatible with existing code
    return {
        "threshold": profile.threshold_db,
        "ratio": profile.ratio,
        "attack_ms": profile.attack_ms,
        "release_ms": profile.release_ms,
        "knee": profile.knee,
        "gain": profile.makeup_gain_db,
    }
