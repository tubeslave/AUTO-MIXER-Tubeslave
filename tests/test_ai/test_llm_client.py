"""
Tests for backend.ai.llm_client — OllamaClient, PerplexityClient, and
FallbackChain.

All HTTP calls are mocked via unittest.mock. No real LLM servers required.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.ai.llm_client import (
    OllamaClient,
    PerplexityClient,
    FallbackChain,
    _sync_post_json,
    _HAS_HTTPX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# OllamaClient tests
# ---------------------------------------------------------------------------

class TestOllamaClient:

    def test_default_attributes(self):
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama3.1:8b"
        assert client.timeout == 60.0
        assert client.max_retries == 2
        assert "live-sound" in client.system_prompt.lower() or "mixing" in client.system_prompt.lower()

    def test_custom_attributes(self):
        client = OllamaClient(
            base_url="http://myhost:9999",
            model="mistral",
            timeout=30.0,
            max_retries=5,
            system_prompt="Custom prompt",
        )
        assert client.base_url == "http://myhost:9999"
        assert client.model == "mistral"
        assert client.timeout == 30.0
        assert client.max_retries == 5
        assert client.system_prompt == "Custom prompt"

    def test_trailing_slash_stripped(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    def test_chat_success_mocked(self):
        """Mock a successful Ollama chat response."""
        client = OllamaClient(max_retries=0)

        async def _mock_chat():
            # Mock the internal HTTP call
            with patch.object(client, "_get_client") as mock_get:
                mock_resp = MagicMock()
                mock_resp.json.return_value = {
                    "message": {"content": "Boost kick at 60 Hz."}
                }
                mock_resp.raise_for_status = MagicMock()

                mock_http = AsyncMock()
                mock_http.post = AsyncMock(return_value=mock_resp)
                mock_get.return_value = mock_http

                if _HAS_HTTPX:
                    result = await client.chat("How to EQ a kick?")
                    return result
                else:
                    return None

        if _HAS_HTTPX:
            result = _run(_mock_chat())
            if result is not None:
                assert "kick" in result.lower() or "60" in result

    def test_is_available_when_server_down(self):
        """is_available should return False when the server is unreachable."""
        client = OllamaClient(base_url="http://127.0.0.1:1")  # Invalid port

        async def _check():
            return await client.is_available()

        result = _run(_check())
        assert result is False

    def test_close_when_no_client(self):
        """Closing without ever creating a client should not raise."""
        client = OllamaClient()

        async def _close():
            await client.close()

        _run(_close())  # Should not raise


# ---------------------------------------------------------------------------
# PerplexityClient tests
# ---------------------------------------------------------------------------

class TestPerplexityClient:

    def test_default_attributes(self):
        client = PerplexityClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.model == "sonar"
        assert client.timeout == 30.0
        assert client.max_retries == 2

    def test_api_key_from_env(self):
        """Falls back to environment variable when no key is passed."""
        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": "env-key"}):
            client = PerplexityClient()
            assert client.api_key == "env-key"

    def test_no_api_key_raises_on_chat(self):
        """chat() should raise RuntimeError when no API key is configured."""
        client = PerplexityClient(api_key="")
        with patch.dict("os.environ", {"PERPLEXITY_API_KEY": ""}, clear=False):
            client.api_key = ""

            async def _chat():
                return await client.chat("test prompt")

            with pytest.raises(RuntimeError, match="API key"):
                _run(_chat())

    def test_is_available_with_key(self):
        client = PerplexityClient(api_key="test-key")

        async def _check():
            return await client.is_available()

        assert _run(_check()) is True

    def test_is_available_without_key(self):
        client = PerplexityClient(api_key="")

        async def _check():
            return await client.is_available()

        assert _run(_check()) is False

    def test_close_when_no_client(self):
        client = PerplexityClient(api_key="test")

        async def _close():
            await client.close()

        _run(_close())  # Should not raise


# ---------------------------------------------------------------------------
# FallbackChain tests
# ---------------------------------------------------------------------------

class TestFallbackChain:

    def test_empty_chain_raises(self):
        chain = FallbackChain(clients=[])

        async def _chat():
            return await chain.chat("test")

        with pytest.raises(RuntimeError, match="All LLM clients failed"):
            _run(_chat())

    def test_first_client_succeeds(self):
        client1 = AsyncMock()
        client1.chat = AsyncMock(return_value="Response from client 1")
        client1.is_available = AsyncMock(return_value=True)

        client2 = AsyncMock()
        client2.chat = AsyncMock(return_value="Response from client 2")

        chain = FallbackChain(clients=[client1, client2])

        async def _chat():
            return await chain.chat("test prompt")

        result = _run(_chat())
        assert result == "Response from client 1"
        client1.chat.assert_awaited_once()
        client2.chat.assert_not_awaited()

    def test_falls_through_on_failure(self):
        client1 = AsyncMock()
        client1.chat = AsyncMock(side_effect=RuntimeError("Ollama down"))
        client1.is_available = AsyncMock(return_value=True)

        client2 = AsyncMock()
        client2.chat = AsyncMock(return_value="Fallback response")
        client2.is_available = AsyncMock(return_value=True)

        chain = FallbackChain(clients=[client1, client2])

        async def _chat():
            return await chain.chat("test prompt")

        result = _run(_chat())
        assert result == "Fallback response"

    def test_skips_unavailable_client(self):
        client1 = AsyncMock()
        client1.is_available = AsyncMock(return_value=False)
        client1.chat = AsyncMock(return_value="Should not be called")

        client2 = AsyncMock()
        client2.is_available = AsyncMock(return_value=True)
        client2.chat = AsyncMock(return_value="Available client response")

        chain = FallbackChain(clients=[client1, client2])

        async def _chat():
            return await chain.chat("test")

        result = _run(_chat())
        assert result == "Available client response"
        client1.chat.assert_not_awaited()

    def test_last_successful_client_tracked(self):
        client1 = AsyncMock()
        client1.__class__.__name__ = "OllamaClient"
        client1.is_available = AsyncMock(return_value=True)
        client1.chat = AsyncMock(return_value="Ok")

        chain = FallbackChain(clients=[client1])
        assert chain.last_successful_client is None

        async def _chat():
            return await chain.chat("test")

        _run(_chat())
        assert chain.last_successful_client is not None

    def test_all_clients_fail_raises(self):
        client1 = AsyncMock()
        client1.is_available = AsyncMock(return_value=True)
        client1.chat = AsyncMock(side_effect=RuntimeError("fail1"))

        client2 = AsyncMock()
        client2.is_available = AsyncMock(return_value=True)
        client2.chat = AsyncMock(side_effect=RuntimeError("fail2"))

        chain = FallbackChain(clients=[client1, client2])

        async def _chat():
            return await chain.chat("test")

        with pytest.raises(RuntimeError, match="All LLM clients failed"):
            _run(_chat())

    def test_close_all_clients(self):
        client1 = AsyncMock()
        client1.close = AsyncMock()
        client2 = AsyncMock()
        client2.close = AsyncMock()

        chain = FallbackChain(clients=[client1, client2])

        async def _close():
            await chain.close()

        _run(_close())
        client1.close.assert_awaited_once()
        client2.close.assert_awaited_once()

    def test_timeout_per_client(self):
        chain = FallbackChain(clients=[], timeout_per_client=42.0)
        assert chain.timeout_per_client == 42.0
