"""Голосовой помощник саундчека — этап 0 (Ф10, DEC-0010).

Фразовый движок на событиях per-channel саундчека + локальный TTS.
Этап 0: только локальный выход (macOS `say`), НИКАКИХ записей на пульт и
маршрутизации в мониторы — это этап 3 за операторской разблокировкой
(write-family talkback_routing, класс A).

Языки: ru/en (выбор придёт из frontend). Фразы кэшируются в WAV.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

TTS_TIMEOUT_SEC = 30.0
DEFAULT_VOICES = {"ru": "Milena", "en": "Samantha"}
PHRASE_PEAK_CAP_DBFS = -20.0  # талкбэк-кап: фразы никогда не громче

# Локализация инструментов: ключи — нормализованные пресеты/имена движка.
INSTRUMENT_NAMES = {
    "kick": {"ru": "бас-бочка", "en": "kick drum"},
    "snare": {"ru": "малый барабан", "en": "snare"},
    "snare_bottom": {"ru": "нижний микрофон малого", "en": "snare bottom"},
    "rack_tom": {"ru": "том", "en": "rack tom"},
    "floor_tom": {"ru": "напольный том", "en": "floor tom"},
    "overhead": {"ru": "тарелки", "en": "overheads"},
    "hihat": {"ru": "хай-хэт", "en": "hi-hat"},
    "bass_guitar": {"ru": "бас-гитара", "en": "bass guitar"},
    "electric_guitar": {"ru": "гитара", "en": "guitar"},
    "acoustic_guitar": {"ru": "акустическая гитара", "en": "acoustic guitar"},
    "accordion": {"ru": "аккордеон", "en": "accordion"},
    "keys": {"ru": "клавиши", "en": "keys"},
    "lead_vocal": {"ru": "вокал", "en": "lead vocal"},
    "backing_vocal": {"ru": "бэк-вокал", "en": "backing vocals"},
    "playback": {"ru": "плейбек", "en": "playback"},
}

PHRASES = {
    "soundcheck_start": {
        "ru": "Начинаем саундчек по инструментам.",
        "en": "Starting the per-instrument soundcheck.",
    },
    "waiting": {
        "ru": "Система ждёт: {instrument}. Играй.",
        "en": "Waiting for {instrument}. Play, please.",
    },
    "signal_confirmed": {
        "ru": "{instrument}: сигнал есть, настраиваю.",
        "en": "{instrument}: got signal, tuning now.",
    },
    "channel_done": {
        "ru": "{instrument} готово. Спасибо.",
        "en": "{instrument} done. Thank you.",
    },
    "skipped": {
        "ru": "{instrument}: пропускаем, идём дальше.",
        "en": "{instrument}: skipped, moving on.",
    },
    "timeout": {
        "ru": "{instrument}: сигнала нет, пропускаю. Проверьте линию.",
        "en": "{instrument}: no signal, skipping. Check the line.",
    },
    "repeat": {
        "ru": "Ещё раз: {instrument}, пожалуйста.",
        "en": "One more time: {instrument}, please.",
    },
    "soundcheck_done": {
        "ru": "Саундчек по инструментам завершён. Спасибо всем.",
        "en": "Per-instrument soundcheck complete. Thank you all.",
    },
}

# Маппинг per_channel_status движка -> ключ фразы.
STATUS_TO_PHRASE = {
    "waiting_for_signal": "waiting",
    "signal": "signal_confirmed",
    "done": "channel_done",
    "skipped": "skipped",
    "timeout": "timeout",
    "repeat_prev": "repeat",
}


def normalize_instrument(preset_or_name: str) -> str:
    key = str(preset_or_name or "").strip().lower().replace(" ", "_")
    return key if key in INSTRUMENT_NAMES else ""


def instrument_display(preset_or_name: str, lang: str, raw_name: str = "") -> str:
    """Локализованное имя инструмента; fallback — сырое имя канала."""
    key = normalize_instrument(preset_or_name)
    if key:
        return INSTRUMENT_NAMES[key].get(lang) or INSTRUMENT_NAMES[key]["en"]
    return str(raw_name or preset_or_name or ("канал" if lang == "ru" else "channel"))


def render_phrase(phrase_key: str, lang: str, instrument: str = "") -> str:
    lang = lang if lang in ("ru", "en") else "en"
    template = PHRASES.get(phrase_key, {}).get(lang)
    if template is None:
        raise KeyError(f"unknown phrase/lang: {phrase_key}/{lang}")
    return template.format(instrument=instrument)


class SayTTS:
    """macOS `say` бэкенд (этап 0). Кэширует фразы в AIFF/WAV."""

    def __init__(self, cache_dir: Path, voices: Optional[Mapping[str, str]] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voices = dict(voices or DEFAULT_VOICES)

    @staticmethod
    def available() -> bool:
        return sys.platform == "darwin" and shutil.which("say") is not None

    def synth(self, text: str, lang: str) -> Path:
        voice = self.voices.get(lang, self.voices.get("en", "Samantha"))
        digest = hashlib.sha256(f"{voice}|{text}".encode("utf-8")).hexdigest()[:16]
        out = self.cache_dir / f"phrase_{lang}_{digest}.aiff"
        if out.exists():
            return out
        subprocess.run(
            ["say", "-v", voice, "-o", str(out), text],
            check=True,
            timeout=TTS_TIMEOUT_SEC,
            stdin=subprocess.DEVNULL,
            capture_output=True,
        )
        return out


class VoiceAssistant:
    """Событие per-channel саундчека -> произнесённая фраза (этап 0: локально).

    Готов к подключению как канальный callback движка:
    engine.on_channel_update = assistant.handle_channel_update
    Никаких записей на пульт: воспроизведение только в локальный выход.
    """

    def __init__(
        self,
        tts: SayTTS,
        lang: str = "ru",
        enabled: bool = True,
        player: Optional[Any] = None,
    ):
        self.tts = tts
        self.lang = lang if lang in ("ru", "en") else "en"
        self.enabled = bool(enabled)
        self._player = player  # callable(path) | None -> afplay
        self.spoken_log: list[dict[str, str]] = []

    def set_language(self, lang: str) -> None:
        if lang in ("ru", "en"):
            self.lang = lang

    def _play(self, path: Path) -> None:
        if self._player is not None:
            self._player(path)
            return
        subprocess.run(
            ["afplay", str(path)],
            check=False,
            timeout=TTS_TIMEOUT_SEC,
            stdin=subprocess.DEVNULL,
            capture_output=True,
        )

    def speak(self, phrase_key: str, instrument: str = "") -> Optional[Path]:
        if not self.enabled:
            return None
        try:
            text = render_phrase(phrase_key, self.lang, instrument)
            path = self.tts.synth(text, self.lang)
            self._play(path)
            self.spoken_log.append({"phrase": phrase_key, "text": text})
            return path
        except Exception:
            logger.exception("Voice assistant: speak failed (%s)", phrase_key)
            return None

    def handle_channel_update(self, ch: int, payload: Mapping[str, Any]) -> None:
        """Callback-совместим с on_channel_update per-channel движка."""
        status = str(payload.get("per_channel_status") or "")
        phrase_key = STATUS_TO_PHRASE.get(status)
        if not phrase_key:
            return
        instrument = instrument_display(
            str(payload.get("preset") or ""),
            self.lang,
            raw_name=str(payload.get("name") or ""),
        )
        self.speak(phrase_key, instrument)
