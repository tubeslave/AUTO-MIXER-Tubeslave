"""Тесты фразового движка голосового помощника (Ф10 этап 0, DEC-0010)."""

import sys
from pathlib import Path

import pytest

from voice_assistant import (
    DEFAULT_VOICES,
    PHRASES,
    STATUS_TO_PHRASE,
    SayTTS,
    VoiceAssistant,
    instrument_display,
    render_phrase,
)


class TestPhrases:
    def test_all_phrases_have_both_languages(self):
        for key, variants in PHRASES.items():
            assert set(variants) == {"ru", "en"}, key
            for lang, template in variants.items():
                assert template.strip(), (key, lang)

    def test_all_engine_statuses_map_to_known_phrases(self):
        for status, phrase_key in STATUS_TO_PHRASE.items():
            assert phrase_key in PHRASES, status

    def test_render_waiting_ru(self):
        text = render_phrase("waiting", "ru", "бас-бочка")
        assert text == "Система ждёт: бас-бочка. Играй."

    def test_render_waiting_en(self):
        text = render_phrase("waiting", "en", "kick drum")
        assert "Waiting for kick drum" in text

    def test_unknown_language_falls_back_to_english(self):
        assert render_phrase("waiting", "de", "kick") == render_phrase(
            "waiting", "en", "kick"
        )

    def test_unknown_phrase_raises(self):
        with pytest.raises(KeyError):
            render_phrase("no_such_phrase", "ru")


class TestInstrumentDisplay:
    def test_known_preset_localized(self):
        assert instrument_display("kick", "ru") == "бас-бочка"
        assert instrument_display("kick", "en") == "kick drum"

    def test_spaces_normalized(self):
        assert instrument_display("Floor Tom", "ru") == "напольный том"

    def test_unknown_preset_falls_back_to_raw_name(self):
        assert instrument_display("custom", "ru", raw_name="SNARE T") == "SNARE T"


class FakeTTS:
    def __init__(self):
        self.calls = []

    def synth(self, text, lang):
        self.calls.append((text, lang))
        return Path("/tmp/fake.aiff")


class TestVoiceAssistant:
    def _assistant(self, lang="ru"):
        played = []
        tts = FakeTTS()
        assistant = VoiceAssistant(tts, lang=lang, player=played.append)
        return assistant, tts, played

    def test_waiting_event_speaks_localized_phrase(self):
        assistant, tts, played = self._assistant("ru")
        assistant.handle_channel_update(8, {
            "per_channel_status": "waiting_for_signal",
            "preset": "kick",
            "name": "KICK",
        })
        assert tts.calls == [("Система ждёт: бас-бочка. Играй.", "ru")]
        assert len(played) == 1
        assert assistant.spoken_log[-1]["phrase"] == "waiting"

    def test_unknown_status_is_silent(self):
        assistant, tts, played = self._assistant()
        assistant.handle_channel_update(1, {"per_channel_status": "analyzing"})
        assert tts.calls == [] and played == []

    def test_disabled_assistant_is_silent(self):
        assistant, tts, played = self._assistant()
        assistant.enabled = False
        assistant.handle_channel_update(1, {
            "per_channel_status": "waiting_for_signal", "preset": "kick",
        })
        assert tts.calls == [] and played == []

    def test_language_switch(self):
        assistant, tts, _ = self._assistant("ru")
        assistant.set_language("en")
        assistant.handle_channel_update(1, {
            "per_channel_status": "skipped", "preset": "accordion",
        })
        assert tts.calls[-1][1] == "en"
        assert "accordion" in tts.calls[-1][0]

    def test_tts_failure_does_not_raise(self):
        class BrokenTTS:
            def synth(self, text, lang):
                raise RuntimeError("no voice")

        assistant = VoiceAssistant(BrokenTTS(), lang="ru", player=lambda p: None)
        assistant.handle_channel_update(1, {
            "per_channel_status": "waiting_for_signal", "preset": "kick",
        })
        assert assistant.spoken_log == []


@pytest.mark.skipif(
    not (sys.platform == "darwin" and SayTTS.available()),
    reason="macOS say недоступен",
)
class TestSayTTSIntegration:
    def test_synth_creates_cached_file(self, tmp_path):
        tts = SayTTS(tmp_path)
        out = tts.synth("Проверка связи.", "ru")
        assert out.exists() and out.stat().st_size > 1000
        again = tts.synth("Проверка связи.", "ru")
        assert again == out  # кэш

    def test_default_voices_installed(self):
        import subprocess
        listing = subprocess.run(
            ["say", "-v", "?"], capture_output=True, text=True, timeout=15,
        ).stdout
        assert DEFAULT_VOICES["ru"] in listing
