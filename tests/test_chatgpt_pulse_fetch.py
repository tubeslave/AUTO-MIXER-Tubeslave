import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "chatgpt_pulse_review"))

import fetch_chatgpt_pulse as pulse


def test_sanitize_text_redacts_secret_like_excerpts() -> None:
    text = "\n".join(
        [
            "Authorization: Bearer abc.def-123",
            "OPENAI_API_KEY=sk-proj-abc1234567890abcdefghijklmnop",
            "ANTHROPIC_API_KEY=anthropic-secret",
            "HF_TOKEN='hf-secret'",
            'DATABASE_PASSWORD="db-secret"',
            '"accessToken": "browser-secret",',
            "export CLIENT_SECRET=client-secret",
            "FEATURE_TOKENIZER=safe-tokenizer",
        ]
    )

    sanitized = pulse.sanitize_text(text)

    for secret in (
        "abc.def-123",
        "sk-proj-",
        "anthropic-secret",
        "hf-secret",
        "db-secret",
        "browser-secret",
        "client-secret",
    ):
        assert secret not in sanitized
    assert "FEATURE_TOKENIZER=safe-tokenizer" in sanitized
    assert "<redacted>" in sanitized


def test_extract_conversation_text_sanitizes_message_parts() -> None:
    payload = {
        "mapping": {
            "user": {
                "message": {
                    "author": {"role": "user"},
                    "content": {
                        "parts": [
                            {
                                "text": "Here is HF_TOKEN=hf-secret and https://sdmntpr.example/image.png",
                            }
                        ]
                    },
                }
            },
            "assistant": {
                "message": {
                    "author": {"role": "assistant"},
                    "content": {
                        "parts": [
                            {
                                "text": 'Use {"apiKey": "assistant-secret"} in the config.',
                            }
                        ]
                    },
                }
            },
        }
    }

    extracted = pulse.extract_conversation_text(payload)

    assert "hf-secret" not in extracted["user_excerpt"]
    assert "[redacted-image-url]" in extracted["user_excerpt"]
    assert "assistant-secret" not in extracted["assistant_excerpt"]
    assert "<redacted>" in extracted["assistant_excerpt"]
