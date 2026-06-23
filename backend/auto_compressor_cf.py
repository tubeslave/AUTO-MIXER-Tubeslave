"""
Auto Compressor - CF-LUFS Adaptive Method

Based on debate results from Kimi AI:
- Crest Factor (CF) classification for attack/release/ratio
- LUFS/LRA for target setting
- Hard limits for safety
- Best practices from professional audio engineering

CF Classification:
- CF > 18 dB = PERCUSSION (attack 5ms, release 80ms, ratio 6:1)
- CF 12-18 dB = DRUMS (attack 10ms, release 150ms, ratio 4:1)
- CF 8-12 dB = VOCAL (attack 15ms, release AUTO, ratio 3:1)
- CF 5-8 dB = BASS (attack 40ms, release 250ms, ratio 5:1)
- CF 3-5 dB = PAD (attack 60ms, release 400ms, ratio 2:1)
- CF < 3 dB = FLAT (attack 100ms, release 1000ms, ratio 1.2:1)

Adaptive Formulas:
- Attack = 15 * (0.1 + 0.9 * cf_norm) ms
- Release = 100 * (1.2 - 0.4 * cf_norm) ms
- Ratio = 3.0 * (1.0 - 0.3 * cf_norm)

Hard Limits:
- Attack: 1-100ms
- Release: 10-2000ms
- Ratio: 1-20
- GR Max: 12dB
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
from enum import Enum

try:
    from processing_proposal_adapters import build_operator_correction_payload
except ImportError:  # pragma: no cover - package import compatibility
    from backend.processing_proposal_adapters import build_operator_correction_payload

logger = logging.getLogger(__name__)

AUTO_COMPRESSOR_CF_METHOD_ID = "auto_compressor_cf"
DYNAMICS_READ_REQUIRED_SOURCE = (
    "cached_or_physical_dynamics_read_required_before_promotion"
)


def _cf_fail_closed_contract_evidence(
    action: Dict[str, Any],
    *,
    target_lufs: float,
) -> Dict[str, Any]:
    return {
        "crest_factor_class": action.get("cf_class"),
        "lufs_lra_context": {
            "status": "required_before_live_promotion",
            "target_lufs": float(target_lufs),
            "lufs_readback_required": True,
            "lra_context_required": True,
        },
        "gain_reduction_estimate": {
            "status": "required_before_live_promotion",
            "meter_readback_required": True,
            "cf_db": action.get("cf_db"),
        },
        "bounded_makeup_headroom": {
            "status": "required_before_live_promotion",
            "makeup_gain_db": action.get("makeup_gain_db"),
            "true_peak_headroom_required": True,
        },
        "bypass_state": "readback_required_before_live_promotion",
    }


class CFClass(Enum):
    """Crest Factor classification for instrument types."""
    PERCUSSION = "percussion"    # CF > 18 dB
    DRUMS = "drums"              # CF 12-18 dB
    VOCAL = "vocal"              # CF 8-12 dB
    BASS = "bass"                # CF 5-8 dB
    PAD = "pad"                  # CF 3-5 dB
    FLAT = "flat"                # CF < 3 dB


@dataclass
class CFCompressorParams:
    """Compressor parameters derived from Crest Factor."""
    attack_ms: float
    release_ms: float
    ratio: float
    threshold_db: float
    knee_db: float
    makeup_gain_db: float
    cf_class: CFClass
    cf_db: float
    
    def to_dict(self) -> Dict:
        return {
            'attack_ms': self.attack_ms,
            'release_ms': self.release_ms,
            'ratio': self.ratio,
            'threshold_db': self.threshold_db,
            'knee_db': self.knee_db,
            'makeup_gain_db': self.makeup_gain_db,
            'cf_class': self.cf_class.value,
            'cf_db': self.cf_db
        }


class CFAnalyzer:
    """
    Crest Factor analyzer with temporal smoothing.
    
    RMS window: 10ms (480 samples @ 48kHz)
    Peak detector with decay: 10ms
    CF smoothed over 100ms
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # RMS window: 10ms
        self.rms_window_samples = int(0.01 * sample_rate)
        self.rms_buffer = np.zeros(self.rms_window_samples)
        self.rms_idx = 0
        
        # Peak detector with 10ms decay
        self.peak_decay = np.exp(-1.0 / (0.01 * sample_rate))
        self.current_peak = 0.0
        
        # CF smoothing: 100ms (10 frames @ 10ms)
        self.cf_history = deque(maxlen=10)
        
        # Current values
        self.rms_db = -60.0
        self.peak_db = -60.0
        self.cf_db = 0.0
        self.cf_smoothed_db = 0.0
        
        logger.info(f"CF Analyzer initialized: {self.rms_window_samples} samples RMS window")
    
    def process(self, audio: np.ndarray) -> float:
        """
        Process audio frame and return smoothed CF.
        
        Args:
            audio: Audio samples
            
        Returns:
            Smoothed Crest Factor in dB
        """
        # Compute RMS
        frame_size = len(audio)
        
        # Add to RMS buffer
        for sample in audio:
            self.rms_buffer[self.rms_idx] = sample ** 2
            self.rms_idx = (self.rms_idx + 1) % self.rms_window_samples
        
        rms_squared = np.mean(self.rms_buffer)
        rms = np.sqrt(rms_squared + 1e-10)
        self.rms_db = 20 * np.log10(rms)
        
        # Peak detector with decay
        frame_peak = np.max(np.abs(audio))
        if frame_peak > self.current_peak:
            self.current_peak = frame_peak
        else:
            self.current_peak *= self.peak_decay
            if self.current_peak < frame_peak:
                self.current_peak = frame_peak
        
        self.peak_db = 20 * np.log10(self.current_peak + 1e-10)
        
        # Compute Crest Factor
        if self.current_peak > 0 and rms > 0:
            cf_linear = self.current_peak / rms
            self.cf_db = 20 * np.log10(cf_linear)
        else:
            self.cf_db = 0.0
        
        # Temporal smoothing (100ms)
        self.cf_history.append(self.cf_db)
        if len(self.cf_history) > 0:
            self.cf_smoothed_db = np.median(self.cf_history)
        else:
            self.cf_smoothed_db = self.cf_db
        
        return self.cf_smoothed_db
    
    def classify_cf(self, cf_db: Optional[float] = None) -> CFClass:
        """
        Classify Crest Factor into instrument type.
        
        Args:
            cf_db: Crest Factor in dB (uses smoothed if None)
            
        Returns:
            CFClass enum
        """
        cf = cf_db if cf_db is not None else self.cf_smoothed_db
        
        if cf > 18:
            return CFClass.PERCUSSION
        elif cf > 12:
            return CFClass.DRUMS
        elif cf > 8:
            return CFClass.VOCAL
        elif cf > 5:
            return CFClass.BASS
        elif cf > 3:
            return CFClass.PAD
        else:
            return CFClass.FLAT
    
    def get_cf_normalized(self, cf_db: Optional[float] = None) -> float:
        """
        Get normalized CF for adaptive formulas.
        Maps CF 0-20 dB to 0-1 range.
        
        Args:
            cf_db: Crest Factor in dB
            
        Returns:
            Normalized CF (0-1)
        """
        cf = cf_db if cf_db is not None else self.cf_smoothed_db
        # Normalize: 0 dB -> 0, 20 dB -> 1
        normalized = (cf - 0) / 20.0
        return np.clip(normalized, 0.0, 1.0)
    
    def reset(self):
        """Reset analyzer state."""
        self.rms_buffer.fill(0)
        self.rms_idx = 0
        self.current_peak = 0.0
        self.cf_history.clear()
        self.rms_db = -60.0
        self.peak_db = -60.0
        self.cf_db = 0.0
        self.cf_smoothed_db = 0.0


class CFCompressorCalculator:
    """
    Calculate compressor parameters based on Crest Factor.
    
    Best Practices:
    - CF > 18 dB = PERCUSSION (attack 5ms, release 80ms, ratio 6:1)
    - CF 12-18 dB = DRUMS (attack 10ms, release 150ms, ratio 4:1)
    - CF 8-12 dB = VOCAL (attack 15ms, release AUTO, ratio 3:1)
    - CF 5-8 dB = BASS (attack 40ms, release 250ms, ratio 5:1)
    - CF 3-5 dB = PAD (attack 60ms, release 400ms, ratio 2:1)
    - CF < 3 dB = FLAT (attack 100ms, release 1000ms, ratio 1.2:1)
    
    Adaptive Formulas:
    - Attack = 15 * (0.1 + 0.9 * cf_norm) ms
    - Release = 100 * (1.2 - 0.4 * cf_norm) ms
    - Ratio = 3.0 * (1.0 - 0.3 * cf_norm)
    
    Hard Limits:
    - Attack: 1-100ms
    - Release: 10-2000ms
    - Ratio: 1-20
    """
    
    # Base parameters for each CF class
    BASE_PARAMS = {
        CFClass.PERCUSSION: {'attack': 5, 'release': 80, 'ratio': 6.0, 'knee': 0},
        CFClass.DRUMS: {'attack': 10, 'release': 150, 'ratio': 4.0, 'knee': 2},
        CFClass.VOCAL: {'attack': 15, 'release': 200, 'ratio': 3.0, 'knee': 3},
        CFClass.BASS: {'attack': 40, 'release': 250, 'ratio': 5.0, 'knee': 2},
        CFClass.PAD: {'attack': 60, 'release': 400, 'ratio': 2.0, 'knee': 6},
        CFClass.FLAT: {'attack': 100, 'release': 1000, 'ratio': 1.2, 'knee': 12},
    }
    
    # Hard limits
    MIN_ATTACK = 1.0
    MAX_ATTACK = 100.0
    MIN_RELEASE = 10.0
    MAX_RELEASE = 2000.0
    MIN_RATIO = 1.0
    MAX_RATIO = 20.0
    MAX_GR = 12.0  # Max gain reduction in dB
    
    def __init__(self, target_lufs: float = -18.0, headroom_db: float = 3.0):
        """
        Args:
            target_lufs: Target LUFS level
            headroom_db: Headroom for peak limiting
        """
        self.target_lufs = target_lufs
        self.headroom_db = headroom_db
    
    def calculate_params(
        self,
        cf_db: float,
        lufs_momentary: float,
        peak_db: float,
        cf_class: Optional[CFClass] = None
    ) -> CFCompressorParams:
        """
        Calculate compressor parameters based on CF and LUFS.
        
        Args:
            cf_db: Crest Factor in dB
            lufs_momentary: Momentary LUFS level
            peak_db: Peak level in dB
            cf_class: Pre-computed CF class (optional)
            
        Returns:
            CFCompressorParams with all settings
        """
        # Determine CF class
        if cf_class is None:
            cf_class = self._classify_cf(cf_db)
        
        # Get base params for class
        base = self.BASE_PARAMS[cf_class]
        
        # Compute normalized CF (0-1)
        cf_norm = np.clip((cf_db - 0) / 20.0, 0.0, 1.0)
        
        # Adaptive formulas
        # Attack: faster for high CF (percussive), slower for low CF
        attack_ms = 15 * (0.1 + 0.9 * cf_norm)
        
        # Release: faster for high CF, slower for low CF
        release_ms = 100 * (1.2 - 0.4 * cf_norm)
        
        # Ratio: higher for high CF (percussive), lower for low CF (sustained).
        # C-05 FIX: Original formula was inverted — (1.0 - 0.3 * cf_norm)
        # made high CF produce LOW ratio (soft), opposite of correct behaviour.
        # Percussion (cf_norm → 1) must get AGGRESSIVE compression (high ratio).
        # New formula: ratio grows with cf_norm, range 1.5 × to 4.5 ×.
        ratio = 3.0 * (1.0 + 0.5 * cf_norm)
        
        # Blend with base params (50% adaptive, 50% base)
        attack_ms = 0.5 * attack_ms + 0.5 * base['attack']
        release_ms = 0.5 * release_ms + 0.5 * base['release']
        ratio = 0.5 * ratio + 0.5 * base['ratio']
        knee_db = base['knee']
        
        # Apply hard limits
        attack_ms = np.clip(attack_ms, self.MIN_ATTACK, self.MAX_ATTACK)
        release_ms = np.clip(release_ms, self.MIN_RELEASE, self.MAX_RELEASE)
        ratio = np.clip(ratio, self.MIN_RATIO, self.MAX_RATIO)
        
        # Calculate threshold based on LUFS
        # Target: bring loud signals down to target_lufs
        # Threshold should be below peak but above target
        threshold_db = min(peak_db - self.headroom_db, lufs_momentary + 3)
        
        # Calculate makeup gain
        # Compensate for gain reduction to maintain LUFS
        estimated_gr = self._estimate_gr(lufs_momentary, threshold_db, ratio)
        makeup_gain_db = min(estimated_gr * 0.7, 6.0)  # Compensate 70% of GR, max +6dB
        
        return CFCompressorParams(
            attack_ms=attack_ms,
            release_ms=release_ms,
            ratio=ratio,
            threshold_db=threshold_db,
            knee_db=knee_db,
            makeup_gain_db=makeup_gain_db,
            cf_class=cf_class,
            cf_db=cf_db
        )
    
    def _classify_cf(self, cf_db: float) -> CFClass:
        """Classify Crest Factor."""
        if cf_db > 18:
            return CFClass.PERCUSSION
        elif cf_db > 12:
            return CFClass.DRUMS
        elif cf_db > 8:
            return CFClass.VOCAL
        elif cf_db > 5:
            return CFClass.BASS
        elif cf_db > 3:
            return CFClass.PAD
        else:
            return CFClass.FLAT
    
    def _estimate_gr(self, lufs: float, threshold: float, ratio: float) -> float:
        """
        Estimate gain reduction in dB.
        
        Simple model: GR = (LUFS - threshold) * (ratio - 1) / ratio
        """
        if lufs <= threshold:
            return 0.0
        
        over_db = lufs - threshold
        gr = over_db * (ratio - 1) / ratio
        
        return min(gr, self.MAX_GR)


class AutoCFCompressorController:
    """
    Automatic compressor controller based on Crest Factor.
    
    Main features:
    - Real-time CF analysis
    - Adaptive parameter calculation
    - Smooth transitions
    - Hard limits for safety
    """
    
    def __init__(
        self,
        mixer_client=None,
        sample_rate: int = 48000,
        target_lufs: float = -18.0,
        update_interval_ms: float = 100.0
    ):
        """
        Args:
            mixer_client: OSC mixer client
            sample_rate: Audio sample rate
            target_lufs: Target LUFS level
            update_interval_ms: How often to update parameters
        """
        self.mixer_client = mixer_client
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.update_interval_ms = update_interval_ms
        
        # Components
        self.cf_analyzer = CFAnalyzer(sample_rate)
        self.calculator = CFCompressorCalculator(target_lufs)
        
        # State
        self.is_active = False
        self.current_channel: Optional[int] = None
        self.current_params: Optional[CFCompressorParams] = None
        self.live_write_allowed = False
        self.write_policy = "proposal_only_quarantined"
        self.last_write_attempted = False
        self.last_write_blocked = False
        self.last_write_count = 0
        self.last_blocked_calls: List[Dict[str, Any]] = []
        self.last_apply_result: Dict[str, Any] = {}
        
        logger.info("AutoCFCompressorController initialized")

    def _cf_compressor_safety_summary(
        self,
        *,
        reason: str,
        channel: Optional[int],
        blocked_calls: List[Dict[str, Any]],
        proposed_action: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "summary_type": "auto_cf_compressor_fail_closed_review",
            "status": "blocked",
            "channel": channel,
            "reason": reason,
            "write_policy": self.write_policy,
            "blocked_call_count": len(blocked_calls),
            "blocked_calls": list(blocked_calls),
            "threshold_db": proposed_action.get("threshold_db"),
            "ratio": proposed_action.get("ratio"),
            "attack_ms": proposed_action.get("attack_ms"),
            "release_ms": proposed_action.get("release_ms"),
            "makeup_gain_db": proposed_action.get("makeup_gain_db"),
            "cf_db": proposed_action.get("cf_db"),
            "cf_class": proposed_action.get("cf_class"),
            "headroom_check": "required_before_apply",
            "feedback_safety": "required_before_apply",
            "true_peak_headroom": "required_before_apply",
            "gain_reduction_review": "required_before_apply",
            "makeup_gain_safety": "required_before_apply",
            "readback_required_before_promotion": True,
            "rollback_required_before_promotion": True,
            "operator_review_required": True,
            "read_only_or_dry_run_only": True,
            "live_write": False,
            "queue_mutation": False,
            "can_apply_live": False,
            "can_import_to_queue": False,
            "live_apply_allowed": False,
            "queue_import_allowed": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "queue_action": "not_queueable_until_promoted",
            "mutates_mixer": False,
            "can_authorize_live": False,
        }
    
    def process_audio(
        self,
        audio: np.ndarray,
        lufs_momentary: float,
        peak_db: float
    ) -> Optional[CFCompressorParams]:
        """
        Process audio and calculate new compressor parameters.
        
        Args:
            audio: Audio samples
            lufs_momentary: Current LUFS level
            peak_db: Current peak level
            
        Returns:
            New parameters if changed significantly, None otherwise
        """
        if not self.is_active:
            return None
        
        # Analyze CF
        cf_db = self.cf_analyzer.process(audio)
        cf_class = self.cf_analyzer.classify_cf(cf_db)
        
        # Calculate new params
        new_params = self.calculator.calculate_params(
            cf_db=cf_db,
            lufs_momentary=lufs_momentary,
            peak_db=peak_db,
            cf_class=cf_class
        )
        
        # Check if params changed significantly
        if self.current_params is not None:
            attack_change = abs(new_params.attack_ms - self.current_params.attack_ms)
            ratio_change = abs(new_params.ratio - self.current_params.ratio)
            threshold_change = abs(new_params.threshold_db - self.current_params.threshold_db)
            
            # Only update if significant change
            if attack_change < 2 and ratio_change < 0.3 and threshold_change < 1:
                return None
        
        self.current_params = new_params
        return new_params
    
    def _blocked_apply_result(
        self,
        reason: str,
        blocked_calls: List[Dict[str, Any]],
        *,
        channel: Optional[int] = None,
        proposed_action: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self.last_write_attempted = bool(blocked_calls)
        self.last_write_blocked = bool(blocked_calls)
        self.last_write_count = 0
        self.last_blocked_calls = list(blocked_calls)
        result = {
            "applied": False,
            "blocked": bool(blocked_calls),
            "calls": [],
            "blocked_calls": list(blocked_calls),
            "blocked_call_count": len(blocked_calls),
            "reason": reason,
            "write_policy": self.write_policy,
            "live_write_allowed": self.live_write_allowed,
            "can_apply_live": False,
            "can_import_to_queue": False,
            "queue_import_allowed": False,
            "live_apply_allowed": False,
            "read_only_or_dry_run_only": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "queue_action": "not_queueable_until_promoted",
            "live_write": False,
            "queue_mutation": False,
            "mixer_client_present": self.mixer_client is not None,
            "mutates_mixer": False,
        }
        if channel is not None:
            result["channel"] = channel
        blockers = [
            "unsupported_value_type:compressor",
            "not_in_supervised_fader_gain_executor",
            "hardware_readback_and_rollback_not_proven",
            reason,
        ]
        if not blocked_calls:
            blockers.append("missing_cf_compressor_apply_calls")
        action = proposed_action or {
            "action_type": reason,
            "channel": channel,
            "blocked_calls": list(blocked_calls),
            "blocked_call_count": len(blocked_calls),
        }
        cf_compressor_safety_summary = self._cf_compressor_safety_summary(
            reason=reason,
            channel=channel,
            blocked_calls=blocked_calls,
            proposed_action=action,
        )
        contract_evidence = _cf_fail_closed_contract_evidence(
            action,
            target_lufs=self.target_lufs,
        )
        action = {
            **action,
            **contract_evidence,
            "cf_compressor_safety_summary": cf_compressor_safety_summary,
        }
        proposal = {
            "id": f"{AUTO_COMPRESSOR_CF_METHOD_ID}:{channel or 'unknown'}:{reason}",
            "method_id": AUTO_COMPRESSOR_CF_METHOD_ID,
            "source": AUTO_COMPRESSOR_CF_METHOD_ID,
            "title": f"Ch {channel or '?'}: CF compressor blocked proposal evidence",
            "reason": (
                "Crest-factor compressor generated blocked dynamics write evidence. "
                "It is operator review only until dynamics readback, makeup/headroom "
                "safety and rollback proof exist."
            ),
            "target": channel,
            "kind": "compressor",
            "severity": "medium" if blocked_calls else "low",
            "confidence": None,
            "queue_action": result["queue_action"],
            "can_import_to_queue": False,
            "can_apply_live": False,
            "queue_import_allowed": False,
            "live_apply_allowed": False,
            "read_only_or_dry_run_only": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "live_write": False,
            "queue_mutation": False,
            "mutates_mixer": False,
            "blockers": blockers,
            "proposed_action": action,
            "evidence": {
                "blocked_calls": list(blocked_calls),
                "blocked_call_count": len(blocked_calls),
                "write_policy": self.write_policy,
                "old_value_source": DYNAMICS_READ_REQUIRED_SOURCE,
                "rollback_value_source": DYNAMICS_READ_REQUIRED_SOURCE,
                "cf_compressor_safety_summary": cf_compressor_safety_summary,
                **contract_evidence,
                "live_write": False,
                "queue_mutation": False,
                "can_apply_live": False,
                "can_import_to_queue": False,
                "queue_import_allowed": False,
                "live_apply_allowed": False,
                "read_only_or_dry_run_only": True,
                "live_authority_granted": False,
                "readiness_closure_allowed": False,
                "queue_action": "not_queueable_until_promoted",
                "mutates_mixer": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
            },
        }
        result["cf_compressor_safety_summary"] = cf_compressor_safety_summary
        result["operator_correction"] = build_operator_correction_payload(
            proposal=proposal,
            method_id=AUTO_COMPRESSOR_CF_METHOD_ID,
            target=channel,
            proposed_action=action,
            blockers=blockers,
        )
        self.last_apply_result = result
        if blocked_calls:
            logger.warning(
                "AutoCFCompressor live writes are quarantined; %s commands kept as evidence",
                len(blocked_calls),
            )
        return result

    def _build_param_calls(self, channel: int, params: CFCompressorParams) -> List[Dict[str, Any]]:
        return [
            {
                "cmd": "set_compressor_on",
                "channel": channel,
                "on": 1,
                "semantic": "enable_before_cf_params",
            },
            {
                "cmd": "set_compressor",
                "channel": channel,
                "threshold": float(params.threshold_db),
                "ratio": str(params.ratio),
                "attack": float(params.attack_ms),
                "release": float(params.release_ms),
                "knee": int(round(params.knee_db)),
                "gain": float(params.makeup_gain_db),
                "cf_db": float(params.cf_db),
                "cf_class": params.cf_class.value,
                "semantic": "apply_cf_compressor_params",
            },
        ]

    def apply_params(self, channel: int, params: CFCompressorParams) -> Dict[str, Any]:
        """
        Return blocked compressor parameter evidence without mutating a mixer.
        
        Args:
            channel: Mixer channel number
            params: Compressor parameters
        """
        blocked_calls = self._build_param_calls(channel, params)
        result = self._blocked_apply_result(
            "auto_cf_compressor_direct_apply_quarantined",
            blocked_calls,
            channel=channel,
            proposed_action={
                "action_type": "auto_cf_compressor_apply_blocked",
                "channel": channel,
                "threshold_db": float(params.threshold_db),
                "ratio": float(params.ratio),
                "attack_ms": float(params.attack_ms),
                "release_ms": float(params.release_ms),
                "knee_db": float(params.knee_db),
                "makeup_gain_db": float(params.makeup_gain_db),
                "cf_db": float(params.cf_db),
                "cf_class": params.cf_class.value,
                "blocked_calls": list(blocked_calls),
                "blocked_call_count": len(blocked_calls),
            },
        )
        result["planned_params"] = {
            "threshold": float(params.threshold_db),
            "ratio": float(params.ratio),
            "attack_ms": float(params.attack_ms),
            "release_ms": float(params.release_ms),
            "knee_db": float(params.knee_db),
            "makeup_gain_db": float(params.makeup_gain_db),
            "cf_db": float(params.cf_db),
            "cf_class": params.cf_class.value,
        }
        logger.info(
            "CF compressor params kept as blocked evidence for ch%s: CF=%.1fdB (%s), attack=%.1fms, release=%.1fms, ratio=%.1f:1, thr=%.1fdB",
            channel,
            params.cf_db,
            params.cf_class.value,
            params.attack_ms,
            params.release_ms,
            params.ratio,
            params.threshold_db,
        )
        return result
    
    def reset(self):
        """Reset controller state."""
        self.cf_analyzer.reset()
        self.current_params = None
        self.is_active = False
        logger.info("AutoCFCompressorController reset")


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("CF-LUFS Auto Compressor Test")
    print("=" * 70)
    
    # Test different CF values
    test_cases = [
        # (cf_db, lufs, peak, description)
        (22.0, -12.0, -1.0, "Percussion (Snare)"),
        (15.0, -15.0, -3.0, "Drums (Kick)"),
        (10.0, -18.0, -6.0, "Vocal"),
        (6.0, -20.0, -8.0, "Bass"),
        (4.0, -22.0, -10.0, "Pad (Synth)"),
        (2.0, -24.0, -12.0, "Flat (Sine wave)"),
    ]
    
    calculator = CFCompressorCalculator()
    
    for cf_db, lufs, peak, description in test_cases:
        cf_class = calculator._classify_cf(cf_db)
        params = calculator.calculate_params(cf_db, lufs, peak, cf_class)
        
        print(f"\n{description}:")
        print(f"  CF = {cf_db:.1f} dB ({cf_class.value})")
        print(f"  LUFS = {lufs:.1f} dB, Peak = {peak:.1f} dB")
        print(f"  → Attack: {params.attack_ms:.1f} ms")
        print(f"  → Release: {params.release_ms:.1f} ms")
        print(f"  → Ratio: {params.ratio:.1f}:1")
        print(f"  → Threshold: {params.threshold_db:.1f} dB")
        print(f"  → Knee: {params.knee_db:.1f} dB")
        print(f"  → Makeup Gain: {params.makeup_gain_db:.1f} dB")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
