"""
LLM client abstraction for the AI mixing agent.

Provides async clients for:
- Ollama (local, self-hosted LLM)
- Perplexity (cloud API with live search grounding)
- FallbackChain (tries multiple clients in priority order)

All clients share a common async interface: chat(prompt) -> str
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# httpx import (optional but strongly preferred)
# ---------------------------------------------------------------------------

_HAS_HTTPX = False
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    httpx = None  # type: ignore[assignment]
    logger.warning(
        "httpx not installed; LLM clients will fall back to urllib. "
        "Install httpx for async support: pip install httpx"
    )

# Fallback: use urllib for synchronous HTTP if httpx is unavailable
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _async_sleep(seconds: float) -> None:
    """Async sleep wrapper."""
    await asyncio.sleep(seconds)


def _sync_post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """
    Synchronous POST with JSON body, using urllib as fallback.

    Returns parsed JSON response dict.
    """
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)

    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


# ---------------------------------------------------------------------------
# OllamaClient
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Async client for Ollama (local LLM server).

    Connects to the Ollama REST API at localhost:11434 by default.
    Supports any model available on the Ollama instance.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: float = 60.0,
        max_retries: int = 2,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            base_url: Ollama server URL.
            model: Model name (must be pulled on server).
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts on failure.
            system_prompt: Optional system prompt prepended to every request.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_prompt = system_prompt or (
            "You are an expert live-sound engineer assistant for the "
            "Behringer Wing Rack digital mixer. Provide concise, actionable "
            "mixing advice with specific parameter values."
        )
        self._client: Optional[Any] = None

    async def _get_client(self) -> Any:
        """Lazily create httpx async client."""
        if _HAS_HTTPX and self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def chat(self, prompt: str) -> str:
        """
        Send a chat message to Ollama and return the response text.

        Args:
            prompt: User message / question.

        Returns:
            Model response as a string.

        Raises:
            RuntimeError: If all retries fail.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        url = f"{self.base_url}/api/chat"

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                if _HAS_HTTPX:
                    client = await self._get_client()
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                else:
                    # Synchronous fallback
                    data = await asyncio.get_event_loop().run_in_executor(
                        None, _sync_post_json, url, payload, None, self.timeout,
                    )

                content = data.get("message", {}).get("content", "")
                if content:
                    return content.strip()
                raise RuntimeError("Empty response from Ollama")

            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Ollama attempt {attempt}/{self.max_retries + 1} failed: {exc}"
                )
                if attempt <= self.max_retries:
                    await _async_sleep(min(2 ** attempt, 8))

        raise RuntimeError(
            f"Ollama failed after {self.max_retries + 1} attempts: {last_error}"
        )

    async def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        url = f"{self.base_url}/api/tags"
        try:
            if _HAS_HTTPX:
                client = await self._get_client()
                resp = await client.get(url)
                return resp.status_code == 200
            else:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# PerplexityClient
# ---------------------------------------------------------------------------

class PerplexityClient:
    """
    Async client for Perplexity AI API.

    Uses the Perplexity chat completions endpoint with search-grounded
    responses. Requires PERPLEXITY_API_KEY environment variable.
    """

    API_URL = "https://api.perplexity.ai/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "sonar",
        timeout: float = 30.0,
        max_retries: int = 2,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            api_key: Perplexity API key. Falls back to PERPLEXITY_API_KEY env var.
            model: Perplexity model name.
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts.
            system_prompt: Optional system prompt.
        """
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_prompt = system_prompt or (
            "You are an expert live-sound engineer. When answering questions "
            "about mixing, provide specific numeric values for EQ frequencies, "
            "compression ratios, and other parameters. Reference professional "
            "standards and best practices."
        )
        self._client: Optional[Any] = None

    async def _get_client(self) -> Any:
        """Lazily create httpx async client."""
        if _HAS_HTTPX and self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def chat(self, prompt: str) -> str:
        """
        Send a chat message to Perplexity and return the response.

        Args:
            prompt: User message / question.

        Returns:
            Model response as a string.

        Raises:
            RuntimeError: If API key is missing or all retries fail.
        """
        if not self.api_key:
            raise RuntimeError(
                "Perplexity API key not set. "
                "Set PERPLEXITY_API_KEY environment variable."
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 2):
            try:
                if _HAS_HTTPX:
                    client = await self._get_client()
                    resp = await client.post(
                        self.API_URL, json=payload, headers=headers,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                else:
                    data = await asyncio.get_event_loop().run_in_executor(
                        None,
                        _sync_post_json,
                        self.API_URL,
                        payload,
                        headers,
                        self.timeout,
                    )

                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    if content:
                        return content.strip()
                raise RuntimeError("Empty response from Perplexity")

            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"Perplexity attempt {attempt}/{self.max_retries + 1} "
                    f"failed: {exc}"
                )
                if attempt <= self.max_retries:
                    await _async_sleep(min(2 ** attempt, 8))

        raise RuntimeError(
            f"Perplexity failed after {self.max_retries + 1} attempts: "
            f"{last_error}"
        )

    async def is_available(self) -> bool:
        """Check if API key is configured (does not make a network call)."""
        return bool(self.api_key)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# FallbackChain
# ---------------------------------------------------------------------------

class FallbackChain:
    """
    Tries multiple LLM clients in order, returning the first successful response.

    Typical usage:
        chain = FallbackChain([OllamaClient(), PerplexityClient()])
        response = await chain.chat("How should I EQ a kick drum?")
    """

    def __init__(
        self,
        clients: Optional[List[Any]] = None,
        timeout_per_client: float = 60.0,
    ):
        """
        Args:
            clients: Ordered list of LLM client instances (each must have
                     an async chat(prompt) -> str method).
            timeout_per_client: Max time per client attempt in seconds.
        """
        self.clients = clients or []
        self.timeout_per_client = timeout_per_client
        self._last_successful_client: Optional[str] = None

    @property
    def last_successful_client(self) -> Optional[str]:
        """Name of the last client that returned a successful response."""
        return self._last_successful_client

    async def chat(self, prompt: str) -> str:
        """
        Send prompt through the fallback chain.

        Tries each client in order. Returns the first successful response.

        Args:
            prompt: User message / question.

        Returns:
            Response text from the first successful client.

        Raises:
            RuntimeError: If all clients fail.
        """
        errors: List[str] = []

        for client in self.clients:
            client_name = type(client).__name__
            try:
                # Check availability first (fast fail)
                if hasattr(client, "is_available"):
                    available = await client.is_available()
                    if not available:
                        logger.info(f"{client_name} not available, skipping")
                        errors.append(f"{client_name}: not available")
                        continue

                result = await asyncio.wait_for(
                    client.chat(prompt),
                    timeout=self.timeout_per_client,
                )
                self._last_successful_client = client_name
                logger.info(f"FallbackChain: response from {client_name}")
                return result

            except asyncio.TimeoutError:
                msg = f"{client_name}: timed out after {self.timeout_per_client}s"
                logger.warning(msg)
                errors.append(msg)
            except Exception as exc:
                msg = f"{client_name}: {exc}"
                logger.warning(msg)
                errors.append(msg)

        raise RuntimeError(
            f"All LLM clients failed. Errors:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    async def close(self) -> None:
        """Close all underlying clients."""
        for client in self.clients:
            if hasattr(client, "close"):
                try:
                    await client.close()
                except Exception as exc:
                    logger.warning(f"Error closing {type(client).__name__}: {exc}")
