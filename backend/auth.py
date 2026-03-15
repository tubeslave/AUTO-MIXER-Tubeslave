"""
Authentication, TLS, and rate limiting for AUTO-MIXER-Tubeslave.

Provides token-based auth, SSL context creation, and token-bucket rate limiting.
"""

import hashlib
import hmac
import logging
import os
import secrets
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

SECRET_KEY = os.environ.get("AUTOMIXER_SECRET_KEY", secrets.token_hex(32))


@dataclass
class TokenInfo:
    """Metadata for an issued token."""
    token_hash: str
    created_at: float
    expires_at: float
    client_id: str
    scopes: Set[str] = field(default_factory=lambda: {"read", "write"})


class TokenAuth:
    """Token-based authentication for WebSocket connections."""

    def __init__(self, secret_key: Optional[str] = None,
                 token_ttl: float = 86400.0):
        self._secret = (secret_key or SECRET_KEY).encode("utf-8")
        self._token_ttl = token_ttl
        self._tokens: Dict[str, TokenInfo] = {}
        self._lock = threading.Lock()

    def generate_token(self, client_id: str = "",
                       scopes: Optional[Set[str]] = None) -> str:
        """Generate a new authentication token."""
        raw = secrets.token_urlsafe(32)
        token_hash = self._hash_token(raw)
        now = time.time()
        info = TokenInfo(
            token_hash=token_hash,
            created_at=now,
            expires_at=now + self._token_ttl,
            client_id=client_id or secrets.token_hex(4),
            scopes=scopes or {"read", "write"},
        )
        with self._lock:
            self._tokens[token_hash] = info
        logger.info(f"Token generated for client {info.client_id}")
        return raw

    def validate_token(self, token: str) -> Optional[TokenInfo]:
        """Validate a token and return its info, or None if invalid."""
        token_hash = self._hash_token(token)
        with self._lock:
            info = self._tokens.get(token_hash)
        if info is None:
            return None
        if time.time() > info.expires_at:
            self.revoke_token(token)
            return None
        return info

    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        token_hash = self._hash_token(token)
        with self._lock:
            return self._tokens.pop(token_hash, None) is not None

    def cleanup_expired(self) -> int:
        """Remove all expired tokens. Returns count of removed tokens."""
        now = time.time()
        with self._lock:
            expired = [h for h, info in self._tokens.items() if now > info.expires_at]
            for h in expired:
                del self._tokens[h]
        return len(expired)

    def _hash_token(self, token: str) -> str:
        """Create HMAC hash of a token for storage."""
        return hmac.new(self._secret, token.encode("utf-8"), hashlib.sha256).hexdigest()


class RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, rate: float = 100.0, burst: int = 200):
        """
        Args:
            rate: Tokens added per second.
            burst: Maximum bucket size.
        """
        self._rate = rate
        self._burst = burst
        self._buckets: Dict[str, float] = {}
        self._last_check: Dict[str, float] = {}
        self._lock = threading.Lock()

    def check(self, client_id: str) -> bool:
        """Check if a request is allowed for the given client."""
        now = time.time()
        with self._lock:
            if client_id not in self._buckets:
                self._buckets[client_id] = float(self._burst)
                self._last_check[client_id] = now

            elapsed = now - self._last_check[client_id]
            self._last_check[client_id] = now

            self._buckets[client_id] = min(
                self._burst,
                self._buckets[client_id] + elapsed * self._rate,
            )

            if self._buckets[client_id] >= 1.0:
                self._buckets[client_id] -= 1.0
                return True
            return False

    def reset(self, client_id: str) -> None:
        """Reset a client's rate limit bucket."""
        with self._lock:
            self._buckets.pop(client_id, None)
            self._last_check.pop(client_id, None)


class TLSConfig:
    """TLS/SSL configuration helper."""

    def __init__(self, cert_path: Optional[str] = None,
                 key_path: Optional[str] = None,
                 ca_path: Optional[str] = None):
        self.cert_path = cert_path
        self.key_path = key_path
        self.ca_path = ca_path

    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create and return an SSL context, or None if no certs configured."""
        if not self.cert_path or not self.key_path:
            return None

        if not os.path.isfile(self.cert_path):
            logger.error(f"Certificate file not found: {self.cert_path}")
            return None

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(self.cert_path, self.key_path)
        if self.ca_path and os.path.isfile(self.ca_path):
            ctx.load_verify_locations(self.ca_path)
            ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        logger.info("TLS context created")
        return ctx


class AuthMiddleware:
    """Authentication middleware for WebSocket connections."""

    def __init__(self, token_auth: TokenAuth,
                 rate_limiter: Optional[RateLimiter] = None):
        self._token_auth = token_auth
        self._rate_limiter = rate_limiter or RateLimiter()

    def authenticate(self, token: str) -> Optional[TokenInfo]:
        """Authenticate a WebSocket connection."""
        info = self._token_auth.validate_token(token)
        if info is None:
            logger.warning("Authentication failed: invalid token")
            return None
        if not self._rate_limiter.check(info.client_id):
            logger.warning(f"Rate limit exceeded for client {info.client_id}")
            return None
        return info

    def has_scope(self, token_info: TokenInfo, scope: str) -> bool:
        """Check if a token has the required scope."""
        return scope in token_info.scopes
