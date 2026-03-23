"""
Simple authentication for the WebSocket API.
Supports token-based auth and IP allowlisting.
"""
import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

class AuthManager:
    """Manages API authentication."""

    def __init__(self, secret_key: Optional[str] = None, enabled: bool = False):
        self.enabled = enabled
        self.secret_key = secret_key or secrets.token_hex(32)
        self._tokens: Dict[str, dict] = {}
        self._allowed_ips: Set[str] = {'127.0.0.1', '::1', 'localhost'}
        self._rate_limits: Dict[str, List[float]] = {}
        self._max_requests_per_minute = 120
        logger.info(f"Auth manager initialized (enabled={enabled})")

    def generate_token(self, client_id: str, expires_hours: float = 24.0) -> str:
        """Generate an authentication token."""
        token = secrets.token_urlsafe(32)
        self._tokens[token] = {
            'client_id': client_id,
            'created': time.time(),
            'expires': time.time() + expires_hours * 3600,
        }
        logger.info(f"Token generated for client: {client_id}")
        return token

    def validate_token(self, token: str) -> bool:
        """Validate an authentication token."""
        if not self.enabled:
            return True
        if token not in self._tokens:
            return False
        info = self._tokens[token]
        if time.time() > info['expires']:
            del self._tokens[token]
            return False
        return True

    def revoke_token(self, token: str):
        """Revoke a token."""
        self._tokens.pop(token, None)

    def add_allowed_ip(self, ip: str):
        """Add an IP to the allowlist."""
        self._allowed_ips.add(ip)

    def check_ip(self, ip: str) -> bool:
        """Check if an IP is allowed."""
        if not self.enabled:
            return True
        return ip in self._allowed_ips

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()
        if client_id not in self._rate_limits:
            self._rate_limits[client_id] = []
        # Clean old entries
        self._rate_limits[client_id] = [
            t for t in self._rate_limits[client_id] if now - t < 60
        ]
        if len(self._rate_limits[client_id]) >= self._max_requests_per_minute:
            return False
        self._rate_limits[client_id].append(now)
        return True

    def authenticate(self, token: Optional[str] = None, ip: Optional[str] = None,
                    client_id: Optional[str] = None) -> bool:
        """Full authentication check."""
        if not self.enabled:
            return True
        if ip and not self.check_ip(ip):
            logger.warning(f"Rejected IP: {ip}")
            return False
        if token and not self.validate_token(token):
            logger.warning(f"Invalid token from {ip or 'unknown'}")
            return False
        if client_id and not self.check_rate_limit(client_id):
            logger.warning(f"Rate limit exceeded: {client_id}")
            return False
        return True

    def create_signed_message(self, payload: str) -> str:
        """Create HMAC-signed message."""
        signature = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{payload}|{signature}"

    def verify_signed_message(self, message: str) -> Optional[str]:
        """Verify HMAC-signed message, return payload if valid."""
        if '|' not in message:
            return None
        payload, signature = message.rsplit('|', 1)
        expected = hmac.new(
            self.secret_key.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        if hmac.compare_digest(signature, expected):
            return payload
        return None

    def cleanup_expired(self):
        """Remove expired tokens."""
        now = time.time()
        expired = [t for t, info in self._tokens.items() if now > info['expires']]
        for token in expired:
            del self._tokens[token]
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired tokens")
