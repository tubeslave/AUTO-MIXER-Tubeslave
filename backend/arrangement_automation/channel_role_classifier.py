"""Channel role classifier based on channel names and conservative defaults."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

from .models import ChannelRoleInfo


DEFAULT_ROLE_LIMITS = {
    "lead_vocal": {"min_db": -4.0, "max_db": 4.0, "attack_sec": 1.0, "release_sec": 3.0},
    "backing_vocal": {"min_db": -5.0, "max_db": 5.0, "attack_sec": 1.5, "release_sec": 4.0},
    "kick": {"min_db": -3.0, "max_db": 3.0, "attack_sec": 2.0, "release_sec": 6.0},
    "snare": {"min_db": -3.0, "max_db": 3.0, "attack_sec": 2.0, "release_sec": 6.0},
    "tom": {"min_db": -3.0, "max_db": 3.0, "attack_sec": 2.0, "release_sec": 6.0},
    "overhead": {"min_db": -4.0, "max_db": 4.0, "attack_sec": 2.5, "release_sec": 6.0},
    "bass": {"min_db": -3.0, "max_db": 3.0, "attack_sec": 2.0, "release_sec": 6.0},
    "rhythm_guitar": {"min_db": -6.0, "max_db": 3.0, "attack_sec": 2.0, "release_sec": 5.0},
    "lead_guitar": {"min_db": -6.0, "max_db": 5.0, "attack_sec": 1.5, "release_sec": 4.0},
    "acoustic_guitar": {"min_db": -5.0, "max_db": 4.0, "attack_sec": 2.0, "release_sec": 5.0},
    "keys": {"min_db": -5.0, "max_db": 3.0, "attack_sec": 2.0, "release_sec": 5.0},
    "pad": {"min_db": -5.0, "max_db": 3.0, "attack_sec": 3.0, "release_sec": 6.0},
    "solo_instrument": {"min_db": -5.0, "max_db": 5.0, "attack_sec": 1.5, "release_sec": 4.0},
    "fx": {"min_db": -5.0, "max_db": 3.0, "attack_sec": 3.0, "release_sec": 6.0},
    "unknown": {"min_db": -2.0, "max_db": 2.0, "attack_sec": 3.0, "release_sec": 6.0},
}

ROLE_GROUPS = {
    "lead_vocal": "vocals",
    "backing_vocal": "vocals",
    "kick": "drums",
    "snare": "drums",
    "tom": "drums",
    "overhead": "drums",
    "bass": "foundation",
    "rhythm_guitar": "guitars",
    "lead_guitar": "guitars",
    "acoustic_guitar": "guitars",
    "keys": "keys",
    "pad": "keys",
    "solo_instrument": "foreground",
    "fx": "fx",
    "unknown": "unknown",
}

ROLE_PRIORITIES = {
    "lead_vocal": 1,
    "solo_instrument": 2,
    "lead_guitar": 3,
    "kick": 4,
    "snare": 4,
    "bass": 4,
    "backing_vocal": 5,
    "rhythm_guitar": 6,
    "acoustic_guitar": 6,
    "keys": 6,
    "tom": 7,
    "overhead": 8,
    "pad": 8,
    "fx": 9,
    "unknown": 10,
}


KEYWORDS: Dict[str, Iterable[str]] = {
    "backing_vocal": (
        "bv", "bvox", "back vox", "back vocal", "backing", "choir", "chorus vox",
        "harmony", "гармония", "бэк", "бек", "подпев", "хор",
    ),
    "lead_vocal": (
        "lead vox", "lead vocal", "vox", "vocal", "voice", "main vox", "main vocal",
        "вокал", "лид вокал", "главный вокал", "голос",
    ),
    "kick": ("kick", "bd", "bass drum", "бочка", "кик"),
    "snare": ("snare", "sd", "sn", "малый", "рабочий"),
    "tom": ("tom", "floor", "rack tom", "том", "томы"),
    "overhead": (
        "oh", "overhead", "ovh", "cym", "cymbal", "hihat", "hat", "ride",
        "оверхед", "тарел",
    ),
    "bass": ("bass", "bs", "бас", "басуха", "sub bass"),
    "lead_guitar": (
        "lead gtr", "lead guitar", "solo gtr", "solo guitar",
        "лид гитара", "соло гитара",
    ),
    "acoustic_guitar": (
        "ac gtr", "acoustic", "акуст", "акустическая", "ак гитара",
    ),
    "rhythm_guitar": (
        "gtr", "guitar", "egtr", "el gtr", "electric guitar",
        "гитара", "ритм гитара",
    ),
    "keys": (
        "keys", "key", "piano", "synth", "organ", "accordion",
        "клавиши", "пиано", "рояль", "синт", "аккордеон", "баян",
    ),
    "pad": (
        "pad", "pads", "strings", "string", "playback", "backing track", "track",
        "пэд", "пады", "стринги", "минус", "фонограмма",
    ),
    "solo_instrument": (
        "solo", "sax", "trumpet", "violin", "flute",
        "соло", "сакс", "труба", "скрипка",
    ),
    "fx": ("fx", "effect", "reverb", "delay", "amb", "эффект", "ревер", "дилей"),
}


class ChannelRoleClassifier:
    """Classify channel musical role from name, channel number, and optional features."""

    def __init__(self, role_limits: Optional[Dict[str, Dict[str, float]]] = None):
        self.role_limits = dict(DEFAULT_ROLE_LIMITS)
        if role_limits:
            for role, values in role_limits.items():
                merged = dict(self.role_limits.get(role, self.role_limits["unknown"]))
                merged.update(values)
                self.role_limits[role] = merged

    def classify(
        self,
        channel_id: int,
        channel_name: str,
        audio_features: Optional[Dict[str, float]] = None,
    ) -> ChannelRoleInfo:
        """Return a role classification with limits and musical priority."""
        normalized = self._normalize(channel_name)
        role = "unknown"
        confidence = 0.25

        for candidate, keywords in KEYWORDS.items():
            if self._contains_keyword(normalized, keywords):
                role = candidate
                confidence = 0.9
                break

        if role == "unknown" and audio_features:
            role, confidence = self._classify_by_features(audio_features)

        limits = self.role_limits.get(role, self.role_limits["unknown"])
        return ChannelRoleInfo(
            channel_id=channel_id,
            channel_name=channel_name or f"Ch {channel_id}",
            role=role,
            confidence=confidence,
            group=ROLE_GROUPS.get(role, "unknown"),
            priority=ROLE_PRIORITIES.get(role, ROLE_PRIORITIES["unknown"]),
            allowed_gain_range_db=(float(limits["min_db"]), float(limits["max_db"])),
            attack_time_sec=float(limits["attack_sec"]),
            release_time_sec=float(limits["release_sec"]),
        )

    @staticmethod
    def _normalize(name: str) -> str:
        text = (name or "").lower().replace("ё", "е")
        text = re.sub(r"[^a-zа-я0-9]+", " ", text)
        return f" {text.strip()} "

    @staticmethod
    def _contains_keyword(normalized_name: str, keywords: Iterable[str]) -> bool:
        compact = normalized_name.strip()
        for keyword in keywords:
            norm_kw = ChannelRoleClassifier._normalize(keyword).strip()
            if len(norm_kw) <= 3:
                if re.search(rf"(^|\s){re.escape(norm_kw)}($|\s)", compact):
                    return True
            elif norm_kw in compact:
                return True
        return False

    @staticmethod
    def _classify_by_features(features: Dict[str, float]) -> tuple[str, float]:
        centroid = float(features.get("spectral_centroid", 0.0) or 0.0)
        if centroid and centroid < 180.0:
            return "kick", 0.45
        if 180.0 <= centroid < 450.0:
            return "bass", 0.4
        if 450.0 <= centroid < 1200.0:
            return "tom", 0.35
        return "unknown", 0.25
