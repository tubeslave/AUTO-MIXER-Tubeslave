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


class TestEngineWiring:
    """Контракт подключения к per-channel движку (Ф10 этап 0)."""

    def _engine(self):
        from auto_soundcheck_engine import AutoSoundcheckEngine
        return AutoSoundcheckEngine(
            num_channels=2, auto_discover=False, mode="per_channel",
        )

    def _info(self, preset="kick", name="KICK"):
        from auto_soundcheck_engine import ChannelInfo
        return ChannelInfo(channel=3, name=name, preset=preset)

    def test_emit_pc_status_feeds_voice_assistant(self):
        engine = self._engine()
        events = []

        class FakeAssistant:
            def handle_channel_update(self, ch, payload):
                events.append((ch, payload))

        engine.voice_assistant = FakeAssistant()
        engine._emit_pc_status(3, self._info(), "waiting_for_signal")
        assert events and events[0][0] == 3
        assert events[0][1]["per_channel_status"] == "waiting_for_signal"
        assert events[0][1]["preset"] == "kick"

    def test_voice_failure_does_not_break_soundcheck(self):
        engine = self._engine()

        class BrokenAssistant:
            def handle_channel_update(self, ch, payload):
                raise RuntimeError("voice down")

        engine.voice_assistant = BrokenAssistant()
        engine._emit_pc_status(3, self._info(), "skipped")  # не должно бросить
        assert engine._pc_results[3] == "skipped"

    def test_assistant_defaults_to_none(self):
        engine = self._engine()
        assert engine.voice_assistant is None
        engine._emit_pc_status(3, self._info(), "signal")
        assert engine._pc_results[3] == "signal"


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
