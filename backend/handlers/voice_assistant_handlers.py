"""WS-поверхность голосового помощника v1 (Ф10, DEC-0010, контракт CLD-0031).

Команды: set_voice_language, set_voice_enabled, get_voice_assistant_status.
Статус: voice_assistant_status (schema automixer.voice_assistant.v1) —
ответ запросившему + broadcast при изменении. Поля v1 не переименовываются.

Этапы 0-2: routing строго "local_only" — фразы играют в локальный выход
Mac; маршрутизация в мониторные шины ("monitor_buses") — этап 3 за
операторской разблокировкой (write-family talkback_routing, класс A).
Помощник по умолчанию ВЫКЛЮЧЕН и включается только явной командой.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

VOICE_ASSISTANT_SCHEMA = "automixer.voice_assistant.v1"
SPOKEN_LOG_TAIL = 50


def register_handlers(server):
    def _get_assistant():
        assistant = getattr(server, "voice_assistant", None)
        if assistant is None:
            from voice_assistant import SayTTS, VoiceAssistant

            cache_dir = Path.home() / ".automixer" / "tts_cache"
            assistant = VoiceAssistant(
                SayTTS(cache_dir), lang="ru", enabled=False,
            )
            server.voice_assistant = assistant
        return assistant

    def _sync_engine_attachment(assistant):
        """Включён — слушает per-channel события движка; выключен — отцеплен."""
        engine = getattr(server, "auto_soundcheck_engine", None)
        if engine is None:
            return
        engine.voice_assistant = assistant if assistant.enabled else None

    def _status_payload(assistant):
        from voice_assistant import SayTTS

        return {
            "type": "voice_assistant_status",
            "schema": VOICE_ASSISTANT_SCHEMA,
            "lang": assistant.lang,
            "enabled": bool(assistant.enabled),
            "available": SayTTS.available(),
            "spoken_log": list(assistant.spoken_log[-SPOKEN_LOG_TAIL:]),
            "backend": "say",
            "routing": "local_only",
        }

    async def _reply_and_broadcast(websocket, assistant, changed: bool):
        payload = _status_payload(assistant)
        await server.send_to_client(websocket, payload)
        if changed:
            try:
                await server.broadcast(payload)
            except Exception:
                logger.exception("voice_assistant_status broadcast failed")

    async def handle_set_voice_language(websocket, data):
        assistant = _get_assistant()
        lang = data.get("lang")
        changed = lang in ("ru", "en") and lang != assistant.lang
        if lang in ("ru", "en"):
            assistant.set_language(lang)
        # невалидный lang — молчаливое игнорирование, статус без изменений
        await _reply_and_broadcast(websocket, assistant, changed)

    async def handle_set_voice_enabled(websocket, data):
        assistant = _get_assistant()
        enabled = data.get("enabled")
        changed = isinstance(enabled, bool) and enabled != assistant.enabled
        if isinstance(enabled, bool):
            assistant.enabled = enabled
            _sync_engine_attachment(assistant)
        await _reply_and_broadcast(websocket, assistant, changed)

    async def handle_get_voice_assistant_status(websocket, data):
        await server.send_to_client(
            websocket, _status_payload(_get_assistant())
        )

    return {
        "set_voice_language": handle_set_voice_language,
        "set_voice_enabled": handle_set_voice_enabled,
        "get_voice_assistant_status": handle_get_voice_assistant_status,
    }
