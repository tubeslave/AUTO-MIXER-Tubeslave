"""
Channel name recognition and AutoFOH source classification.

The legacy ``recognize_instrument``/``scan_and_recognize`` API is preserved,
but the implementation now returns richer source/stem metadata so the
existing soundcheck engine can make safer choices.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from autofoh_models import ControlType, SourceRole, StemRole

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoleProfile:
    source_role: SourceRole
    legacy_preset: Optional[str]
    stem_roles: Tuple[StemRole, ...]
    allowed_controls: Tuple[ControlType, ...]
    priority: float


@dataclass(frozen=True)
class ChannelClassification:
    channel_name: str
    source_role: SourceRole
    legacy_preset: Optional[str]
    stem_roles: Tuple[StemRole, ...]
    allowed_controls: Tuple[ControlType, ...]
    priority: float
    confidence: float
    match_type: str
    matched_pattern: Optional[str] = None

    @property
    def recognized(self) -> bool:
        return self.source_role != SourceRole.UNKNOWN

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.channel_name,
            "source_role": self.source_role.value,
            "preset": self.legacy_preset,
            "recognized": self.recognized,
            "stem_roles": [stem.value for stem in self.stem_roles],
            "allowed_controls": [control.value for control in self.allowed_controls],
            "priority": self.priority,
            "confidence": self.confidence,
            "match_type": self.match_type,
            "matched_pattern": self.matched_pattern,
        }


SAFE_MUSICAL_CONTROLS = (
    ControlType.GAIN,
    ControlType.HPF,
    ControlType.EQ,
    ControlType.COMPRESSOR,
    ControlType.FADER,
    ControlType.PAN,
    ControlType.FX_SEND,
    ControlType.FEEDBACK_NOTCH,
)

LIMITED_FX_CONTROLS = (
    ControlType.EQ,
    ControlType.FADER,
    ControlType.FEEDBACK_NOTCH,
)

EMERGENCY_ONLY_CONTROLS = (
    ControlType.FEEDBACK_NOTCH,
    ControlType.EMERGENCY_FADER,
)


DEFAULT_ROLE_PROFILES: Dict[SourceRole, RoleProfile] = {
    SourceRole.KICK: RoleProfile(SourceRole.KICK, "kick", (StemRole.KICK, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.72),
    SourceRole.KICK_IN: RoleProfile(SourceRole.KICK_IN, "kick", (StemRole.KICK, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.74),
    SourceRole.KICK_OUT: RoleProfile(SourceRole.KICK_OUT, "kick", (StemRole.KICK, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.74),
    SourceRole.SNARE: RoleProfile(SourceRole.SNARE, "snare", (StemRole.SNARE, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.7),
    SourceRole.SNARE_TOP: RoleProfile(SourceRole.SNARE_TOP, "snare", (StemRole.SNARE, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.72),
    SourceRole.SNARE_BOTTOM: RoleProfile(SourceRole.SNARE_BOTTOM, "snare", (StemRole.SNARE, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.72),
    SourceRole.TOM: RoleProfile(SourceRole.TOM, "tom", (StemRole.TOMS, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.68),
    SourceRole.RACK_TOM: RoleProfile(SourceRole.RACK_TOM, "tom", (StemRole.TOMS, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.69),
    SourceRole.FLOOR_TOM: RoleProfile(SourceRole.FLOOR_TOM, "tom", (StemRole.TOMS, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.69),
    SourceRole.OVERHEAD: RoleProfile(SourceRole.OVERHEAD, "overheads", (StemRole.CYMBALS, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.6),
    SourceRole.OH_L: RoleProfile(SourceRole.OH_L, "overheads", (StemRole.CYMBALS, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.61),
    SourceRole.OH_R: RoleProfile(SourceRole.OH_R, "overheads", (StemRole.CYMBALS, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.61),
    SourceRole.HIHAT: RoleProfile(SourceRole.HIHAT, "hihat", (StemRole.CYMBALS, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.58),
    SourceRole.RIDE: RoleProfile(SourceRole.RIDE, "ride", (StemRole.CYMBALS, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.58),
    SourceRole.CYMBALS: RoleProfile(SourceRole.CYMBALS, "cymbals", (StemRole.CYMBALS, StemRole.DRUMS), SAFE_MUSICAL_CONTROLS, 0.55),
    SourceRole.ROOM: RoleProfile(SourceRole.ROOM, "room", (StemRole.DRUMS,), LIMITED_FX_CONTROLS, 0.35),
    SourceRole.BASS: RoleProfile(SourceRole.BASS, "bass", (StemRole.BASS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.75),
    SourceRole.BASS_DI: RoleProfile(SourceRole.BASS_DI, "bass", (StemRole.BASS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.76),
    SourceRole.BASS_MIC: RoleProfile(SourceRole.BASS_MIC, "bass", (StemRole.BASS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.76),
    SourceRole.SYNTH_BASS: RoleProfile(SourceRole.SYNTH_BASS, "bass", (StemRole.BASS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.73),
    SourceRole.GUITAR: RoleProfile(SourceRole.GUITAR, "electricGuitar", (StemRole.GUITARS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.62),
    SourceRole.ELECTRIC_GUITAR: RoleProfile(SourceRole.ELECTRIC_GUITAR, "electricGuitar", (StemRole.GUITARS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.63),
    SourceRole.ACOUSTIC_GUITAR: RoleProfile(SourceRole.ACOUSTIC_GUITAR, "acousticGuitar", (StemRole.GUITARS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.63),
    SourceRole.LEAD_GUITAR: RoleProfile(SourceRole.LEAD_GUITAR, "electricGuitar", (StemRole.GUITARS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.7),
    SourceRole.RHYTHM_GUITAR: RoleProfile(SourceRole.RHYTHM_GUITAR, "electricGuitar", (StemRole.GUITARS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.66),
    SourceRole.KEYS: RoleProfile(SourceRole.KEYS, "synth", (StemRole.KEYS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.58),
    SourceRole.PIANO: RoleProfile(SourceRole.PIANO, "synth", (StemRole.KEYS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.6),
    SourceRole.ORGAN: RoleProfile(SourceRole.ORGAN, "synth", (StemRole.KEYS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.58),
    SourceRole.SYNTH: RoleProfile(SourceRole.SYNTH, "synth", (StemRole.KEYS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.58),
    SourceRole.PAD: RoleProfile(SourceRole.PAD, "synth", (StemRole.KEYS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.55),
    SourceRole.LEAD_SYNTH: RoleProfile(SourceRole.LEAD_SYNTH, "synth", (StemRole.KEYS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.65),
    SourceRole.ACCORDION: RoleProfile(SourceRole.ACCORDION, "accordion", (StemRole.KEYS, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.58),
    SourceRole.LEAD_VOCAL: RoleProfile(SourceRole.LEAD_VOCAL, "leadVocal", (StemRole.LEAD,), SAFE_MUSICAL_CONTROLS, 1.0),
    SourceRole.BACKING_VOCAL: RoleProfile(SourceRole.BACKING_VOCAL, "backVocal", (StemRole.BGV,), SAFE_MUSICAL_CONTROLS, 0.82),
    SourceRole.PLAYBACK: RoleProfile(SourceRole.PLAYBACK, "playback", (StemRole.PLAYBACK, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.52),
    SourceRole.TRACKS: RoleProfile(SourceRole.TRACKS, "playback", (StemRole.PLAYBACK, StemRole.MUSIC), SAFE_MUSICAL_CONTROLS, 0.52),
    SourceRole.MUSIC: RoleProfile(SourceRole.MUSIC, "playback", (StemRole.MUSIC,), LIMITED_FX_CONTROLS, 0.45),
    SourceRole.CLICK: RoleProfile(SourceRole.CLICK, None, (StemRole.UNKNOWN,), EMERGENCY_ONLY_CONTROLS, 0.05),
    SourceRole.TALKBACK: RoleProfile(SourceRole.TALKBACK, None, (StemRole.UNKNOWN,), EMERGENCY_ONLY_CONTROLS, 0.05),
    SourceRole.FX_RETURN: RoleProfile(SourceRole.FX_RETURN, "custom", (StemRole.FX,), LIMITED_FX_CONTROLS, 0.32),
    SourceRole.REVERB_RETURN: RoleProfile(SourceRole.REVERB_RETURN, "custom", (StemRole.FX,), LIMITED_FX_CONTROLS, 0.32),
    SourceRole.DELAY_RETURN: RoleProfile(SourceRole.DELAY_RETURN, "custom", (StemRole.FX,), LIMITED_FX_CONTROLS, 0.32),
    SourceRole.BUS_DRUMS: RoleProfile(SourceRole.BUS_DRUMS, "drums_bus", (StemRole.DRUMS,), LIMITED_FX_CONTROLS, 0.25),
    SourceRole.BUS_VOCAL: RoleProfile(SourceRole.BUS_VOCAL, "vocal_bus", (StemRole.LEAD, StemRole.BGV), LIMITED_FX_CONTROLS, 0.25),
    SourceRole.BUS_INSTRUMENT: RoleProfile(SourceRole.BUS_INSTRUMENT, "instrument_bus", (StemRole.MUSIC,), LIMITED_FX_CONTROLS, 0.25),
    SourceRole.UNKNOWN: RoleProfile(SourceRole.UNKNOWN, None, (StemRole.UNKNOWN,), EMERGENCY_ONLY_CONTROLS, 0.0),
}


LEGACY_PRESET_TO_ROLE = {
    "kick": SourceRole.KICK,
    "snare": SourceRole.SNARE,
    "tom": SourceRole.TOM,
    "hihat": SourceRole.HIHAT,
    "ride": SourceRole.RIDE,
    "cymbals": SourceRole.CYMBALS,
    "overheads": SourceRole.OVERHEAD,
    "room": SourceRole.ROOM,
    "bass": SourceRole.BASS,
    "electricGuitar": SourceRole.ELECTRIC_GUITAR,
    "acousticGuitar": SourceRole.ACOUSTIC_GUITAR,
    "accordion": SourceRole.ACCORDION,
    "synth": SourceRole.SYNTH,
    "playback": SourceRole.PLAYBACK,
    "leadVocal": SourceRole.LEAD_VOCAL,
    "backVocal": SourceRole.BACKING_VOCAL,
    "drums_bus": SourceRole.BUS_DRUMS,
    "vocal_bus": SourceRole.BUS_VOCAL,
    "instrument_bus": SourceRole.BUS_INSTRUMENT,
    "custom": SourceRole.UNKNOWN,
}


EXACT_ALIASES: Dict[str, SourceRole] = {
    "k": SourceRole.KICK,
    "kik": SourceRole.KICK,
    "kick": SourceRole.KICK,
    "kick in": SourceRole.KICK_IN,
    "kick out": SourceRole.KICK_OUT,
    "bd": SourceRole.KICK,
    "bd in": SourceRole.KICK_IN,
    "bd out": SourceRole.KICK_OUT,
    "sn": SourceRole.SNARE,
    "sd": SourceRole.SNARE,
    "snare": SourceRole.SNARE,
    "snare top": SourceRole.SNARE_TOP,
    "snare bottom": SourceRole.SNARE_BOTTOM,
    "rack tom": SourceRole.RACK_TOM,
    "floor tom": SourceRole.FLOOR_TOM,
    "oh": SourceRole.OVERHEAD,
    "oh l": SourceRole.OH_L,
    "oh r": SourceRole.OH_R,
    "hh": SourceRole.HIHAT,
    "ride": SourceRole.RIDE,
    "vox": SourceRole.LEAD_VOCAL,
    "vx": SourceRole.LEAD_VOCAL,
    "lead vox": SourceRole.LEAD_VOCAL,
    "ld vox": SourceRole.LEAD_VOCAL,
    "main vox": SourceRole.LEAD_VOCAL,
    "bgv": SourceRole.BACKING_VOCAL,
    "pb": SourceRole.PLAYBACK,
    "playback": SourceRole.PLAYBACK,
    "tracks": SourceRole.TRACKS,
    "trax": SourceRole.TRACKS,
    "click": SourceRole.CLICK,
    "talkback": SourceRole.TALKBACK,
    "fx": SourceRole.FX_RETURN,
    "verb": SourceRole.REVERB_RETURN,
    "rev": SourceRole.REVERB_RETURN,
    "delay": SourceRole.DELAY_RETURN,
    "keys": SourceRole.KEYS,
    "piano": SourceRole.PIANO,
    "organ": SourceRole.ORGAN,
    "synth": SourceRole.SYNTH,
    "pad": SourceRole.PAD,
    "lead synth": SourceRole.LEAD_SYNTH,
    "bass": SourceRole.BASS,
    "bass di": SourceRole.BASS_DI,
    "bass mic": SourceRole.BASS_MIC,
    "synth bass": SourceRole.SYNTH_BASS,
    "gtr": SourceRole.GUITAR,
    "egtr": SourceRole.ELECTRIC_GUITAR,
    "agtr": SourceRole.ACOUSTIC_GUITAR,
    "ac gtr": SourceRole.ACOUSTIC_GUITAR,
    "fx return": SourceRole.FX_RETURN,
    "reverb return": SourceRole.REVERB_RETURN,
    "delay return": SourceRole.DELAY_RETURN,
}


REGEX_RULES: Sequence[Tuple[re.Pattern[str], SourceRole, float]] = (
    (re.compile(r"\b(kick|bd|bass\s*drum)\s*(in|inside)\b", re.IGNORECASE), SourceRole.KICK_IN, 0.94),
    (re.compile(r"\b(kick|bd|bass\s*drum)\s*(out|outside)\b", re.IGNORECASE), SourceRole.KICK_OUT, 0.94),
    (re.compile(r"\b(snare|sd|sn)\s*(top|t)\b", re.IGNORECASE), SourceRole.SNARE_TOP, 0.93),
    (re.compile(r"\b(snare|sd|sn)\s*(bottom|btm|bot|b)\b", re.IGNORECASE), SourceRole.SNARE_BOTTOM, 0.93),
    (re.compile(r"\b(floor)\s*tom\b", re.IGNORECASE), SourceRole.FLOOR_TOM, 0.91),
    (re.compile(r"\b(rack)\s*tom\b", re.IGNORECASE), SourceRole.RACK_TOM, 0.91),
    (re.compile(r"\b(tom)\b", re.IGNORECASE), SourceRole.TOM, 0.87),
    (re.compile(r"\b(oh|over[\s-]?head|overhead)\s*l\b", re.IGNORECASE), SourceRole.OH_L, 0.9),
    (re.compile(r"\b(oh|over[\s-]?head|overhead)\s*r\b", re.IGNORECASE), SourceRole.OH_R, 0.9),
    (re.compile(r"\b(oh|over[\s-]?head|overhead)s?\b", re.IGNORECASE), SourceRole.OVERHEAD, 0.85),
    (re.compile(r"\b(hi[\s-]?hat|hh)\b", re.IGNORECASE), SourceRole.HIHAT, 0.9),
    (re.compile(r"\b(ride)\b", re.IGNORECASE), SourceRole.RIDE, 0.9),
    (re.compile(r"\b(crash|splash|china|cymbal|cymbals)\b", re.IGNORECASE), SourceRole.CYMBALS, 0.85),
    (re.compile(r"\b(bass)\s*di\b", re.IGNORECASE), SourceRole.BASS_DI, 0.93),
    (re.compile(r"\b(bass)\s*mic\b", re.IGNORECASE), SourceRole.BASS_MIC, 0.93),
    (re.compile(r"\b(synth)\s*bass\b", re.IGNORECASE), SourceRole.SYNTH_BASS, 0.92),
    (re.compile(r"\b(bass)\b(?!\s*drum)", re.IGNORECASE), SourceRole.BASS, 0.87),
    (re.compile(r"\b(lead)\s*gtr\b|\b(lead)\s*guitar\b", re.IGNORECASE), SourceRole.LEAD_GUITAR, 0.9),
    (re.compile(r"\b(rhythm)\s*gtr\b|\b(rhythm)\s*guitar\b", re.IGNORECASE), SourceRole.RHYTHM_GUITAR, 0.9),
    (re.compile(r"\b(acoustic|agtr|ac[\s-]?gtr)\b", re.IGNORECASE), SourceRole.ACOUSTIC_GUITAR, 0.9),
    (re.compile(r"\b(electric|egtr|e[\s-]?gtr)\b", re.IGNORECASE), SourceRole.ELECTRIC_GUITAR, 0.9),
    (re.compile(r"\b(gtr|guitar)\b", re.IGNORECASE), SourceRole.GUITAR, 0.84),
    (re.compile(r"\b(keys|keyboard|kbd)\b", re.IGNORECASE), SourceRole.KEYS, 0.88),
    (re.compile(r"\b(piano|grand)\b", re.IGNORECASE), SourceRole.PIANO, 0.88),
    (re.compile(r"\b(organ|b3|hammond)\b", re.IGNORECASE), SourceRole.ORGAN, 0.88),
    (re.compile(r"\b(lead)\s*synth\b", re.IGNORECASE), SourceRole.LEAD_SYNTH, 0.88),
    (re.compile(r"\b(pad)\b", re.IGNORECASE), SourceRole.PAD, 0.84),
    (re.compile(r"\b(synth)\b", re.IGNORECASE), SourceRole.SYNTH, 0.82),
    (re.compile(r"\b(accordion|bayan|accord)\b", re.IGNORECASE), SourceRole.ACCORDION, 0.9),
    (re.compile(r"\b(lead|ld|main)\s*(vox|voc|vocal|vx)\b", re.IGNORECASE), SourceRole.LEAD_VOCAL, 0.94),
    (re.compile(r"\b(back|bgv|bvox|backing)\s*(vox|voc|vocal)?\b", re.IGNORECASE), SourceRole.BACKING_VOCAL, 0.9),
    (re.compile(r"\b(vox|voc|vocal|vx)\b", re.IGNORECASE), SourceRole.LEAD_VOCAL, 0.86),
    (re.compile(r"\b(playback|tracks|track|trax|pb|stems)\b", re.IGNORECASE), SourceRole.PLAYBACK, 0.9),
    (re.compile(r"\b(music)\b", re.IGNORECASE), SourceRole.MUSIC, 0.82),
    (re.compile(r"\b(click)\b", re.IGNORECASE), SourceRole.CLICK, 0.95),
    (re.compile(r"\b(talkback|tb)\b", re.IGNORECASE), SourceRole.TALKBACK, 0.95),
    (re.compile(r"\b(verb|rev|reverb)\b", re.IGNORECASE), SourceRole.REVERB_RETURN, 0.86),
    (re.compile(r"\b(delay|dly)\b", re.IGNORECASE), SourceRole.DELAY_RETURN, 0.86),
    (re.compile(r"\b(fx|fx\s*return|return)\b", re.IGNORECASE), SourceRole.FX_RETURN, 0.8),
    (re.compile(r"\b(drums?)\s*(bus|grp|group)\b", re.IGNORECASE), SourceRole.BUS_DRUMS, 0.86),
    (re.compile(r"\b(vocals?|vox)\s*(bus|grp|group)\b", re.IGNORECASE), SourceRole.BUS_VOCAL, 0.86),
    (re.compile(r"\b(instr|instrument)\s*(bus|grp|group)\b", re.IGNORECASE), SourceRole.BUS_INSTRUMENT, 0.86),
)


VOCAL_NAME_TOKENS = {
    "katya", "sergey", "slava", "dima", "masha", "sasha", "pasha",
    "vova", "andrey", "alex", "misha", "natasha", "olga", "tanya",
    "vlad", "ivan", "max", "nikita", "dasha", "anya", "lena",
    "maria", "anna", "elena",
}


def _normalize_name(channel_name: str) -> str:
    text = channel_name.lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _role_from_value(value: str) -> SourceRole:
    try:
        return SourceRole(str(value).strip().lower())
    except ValueError:
        return SourceRole.UNKNOWN


def _stem_tuple(values: Optional[Iterable[str]], fallback: Tuple[StemRole, ...]) -> Tuple[StemRole, ...]:
    if not values:
        return fallback
    stems = []
    for value in values:
        try:
            stems.append(StemRole[str(value).strip().upper()])
        except KeyError:
            try:
                stems.append(StemRole(str(value).strip().upper()))
            except ValueError:
                continue
    return tuple(stems) or fallback


def _controls_tuple(values: Optional[Iterable[str]], fallback: Tuple[ControlType, ...]) -> Tuple[ControlType, ...]:
    if not values:
        return fallback
    controls = []
    for value in values:
        try:
            controls.append(ControlType(str(value).strip().lower()))
        except ValueError:
            continue
    return tuple(controls) or fallback


def _build_role_profile(
    role: SourceRole,
    classifier_config: Optional[Mapping[str, object]] = None,
) -> RoleProfile:
    profile = DEFAULT_ROLE_PROFILES.get(role, DEFAULT_ROLE_PROFILES[SourceRole.UNKNOWN])
    overrides = (classifier_config or {}).get("role_overrides", {})
    if isinstance(overrides, Mapping):
        role_override = overrides.get(role.value)
        if isinstance(role_override, Mapping):
            profile = replace(
                profile,
                legacy_preset=role_override.get("legacy_preset", profile.legacy_preset),
                stem_roles=_stem_tuple(role_override.get("stem_roles"), profile.stem_roles),
                allowed_controls=_controls_tuple(
                    role_override.get("allowed_controls"),
                    profile.allowed_controls,
                ),
                priority=float(role_override.get("priority", profile.priority)),
            )
    return profile


def _classification_from_role(
    channel_name: str,
    role: SourceRole,
    confidence: float,
    match_type: str,
    matched_pattern: Optional[str] = None,
    classifier_config: Optional[Mapping[str, object]] = None,
) -> ChannelClassification:
    profile = _build_role_profile(role, classifier_config=classifier_config)
    return ChannelClassification(
        channel_name=channel_name,
        source_role=profile.source_role,
        legacy_preset=profile.legacy_preset,
        stem_roles=profile.stem_roles,
        allowed_controls=profile.allowed_controls,
        priority=profile.priority,
        confidence=float(confidence),
        match_type=match_type,
        matched_pattern=matched_pattern,
    )


def classification_from_legacy_preset(
    legacy_preset: Optional[str],
    channel_name: str = "",
    confidence: float = 0.5,
    match_type: str = "legacy",
) -> ChannelClassification:
    role = LEGACY_PRESET_TO_ROLE.get(legacy_preset or "", SourceRole.UNKNOWN)
    return _classification_from_role(
        channel_name=channel_name or (legacy_preset or ""),
        role=role,
        confidence=confidence,
        match_type=match_type,
        matched_pattern=legacy_preset,
    )


def classify_channel_name(
    channel_name: str,
    classifier_config: Optional[Mapping[str, object]] = None,
) -> ChannelClassification:
    if not channel_name:
        return _classification_from_role(
            channel_name="",
            role=SourceRole.UNKNOWN,
            confidence=0.0,
            match_type="empty",
            classifier_config=classifier_config,
        )

    normalized = _normalize_name(channel_name)

    override_rules = (classifier_config or {}).get("name_overrides", [])
    if isinstance(override_rules, Sequence):
        for rule in override_rules:
            if not isinstance(rule, Mapping):
                continue
            pattern = str(rule.get("pattern", "")).strip()
            if not pattern:
                continue
            if re.search(pattern, normalized, re.IGNORECASE):
                role = _role_from_value(str(rule.get("source_role", "unknown")))
                confidence = float(rule.get("confidence", 0.99))
                return _classification_from_role(
                    channel_name=channel_name,
                    role=role,
                    confidence=confidence,
                    match_type="override",
                    matched_pattern=pattern,
                    classifier_config=classifier_config,
                )

    role = EXACT_ALIASES.get(normalized)
    if role is not None:
        return _classification_from_role(
            channel_name=channel_name,
            role=role,
            confidence=0.98,
            match_type="exact_alias",
            matched_pattern=normalized,
            classifier_config=classifier_config,
        )

    for pattern, role, confidence in REGEX_RULES:
        if pattern.search(normalized):
            return _classification_from_role(
                channel_name=channel_name,
                role=role,
                confidence=confidence,
                match_type="regex",
                matched_pattern=pattern.pattern,
                classifier_config=classifier_config,
            )

    token_set = set(normalized.split())
    if token_set & VOCAL_NAME_TOKENS:
        return _classification_from_role(
            channel_name=channel_name,
            role=SourceRole.LEAD_VOCAL,
            confidence=0.62,
            match_type="heuristic_name",
            matched_pattern="vocal_name_token",
            classifier_config=classifier_config,
        )

    return _classification_from_role(
        channel_name=channel_name,
        role=SourceRole.UNKNOWN,
        confidence=0.1,
        match_type="unknown",
        classifier_config=classifier_config,
    )


def recognize_instrument(channel_name: str) -> Optional[str]:
    """Backward-compatible legacy preset lookup."""
    classification = classify_channel_name(channel_name)
    return classification.legacy_preset


def recognize_instrument_spectral_fallback(
    channel_name: str,
    centroid_hz: float = 0.0,
    energy_bands: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    """
    Conservative spectral fallback used only when the name is unclear.
    """
    if energy_bands is None:
        energy_bands = {}
    low = energy_bands.get("low_100_300", 0.0)
    mid = energy_bands.get("mid_1k_4k", 0.0)
    high = energy_bands.get("high_4k_10k", 0.0)
    if centroid_hz <= 0 and not energy_bands:
        return None
    if centroid_hz < 180 and low > 0.5:
        return "kick"
    if 180 <= centroid_hz < 700 and low > 0.3:
        return "bass"
    if 1800 <= centroid_hz < 5500 and mid > 0.4:
        return "leadVocal"
    if centroid_hz >= 6000 and high > 0.4:
        return "hihat"
    return None


def scan_and_recognize(
    channel_names: Dict[int, str],
    classifier_config: Optional[Mapping[str, object]] = None,
) -> Dict[int, Dict]:
    """Classify a batch of channel names."""
    results = {}

    for channel_num, name in channel_names.items():
        classification = classify_channel_name(name, classifier_config=classifier_config)
        results[channel_num] = classification.to_dict()

        if classification.recognized:
            logger.info(
                "Channel %s '%s' -> %s (conf=%.2f, stems=%s)",
                channel_num,
                name,
                classification.source_role.value,
                classification.confidence,
                ",".join(stem.value for stem in classification.stem_roles),
            )
        else:
            logger.debug(
                "Channel %s '%s' -> unknown (conf=%.2f)",
                channel_num,
                name,
                classification.confidence,
            )

    recognized_count = sum(1 for result in results.values() if result["recognized"])
    logger.info("Recognition complete: %s/%s channels recognized", recognized_count, len(results))
    return results


AVAILABLE_PRESETS = {
    "kick": "Kick",
    "snare": "Snare",
    "tom": "Tom",
    "hihat": "Hi-Hat",
    "ride": "Ride",
    "cymbals": "Cymbals",
    "overheads": "Overheads",
    "room": "Room",
    "bass": "Bass",
    "electricGuitar": "Electric Guitar",
    "acousticGuitar": "Acoustic Guitar",
    "accordion": "Accordion",
    "synth": "Synth / Keys",
    "playback": "Playback",
    "leadVocal": "Lead Vocal",
    "backVocal": "Back Vocal",
    "drums_bus": "Drums Bus",
    "vocal_bus": "Vocal Bus",
    "instrument_bus": "Instrument Bus",
    "custom": "Custom",
}
