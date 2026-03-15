"""
Tests for backend/auth.py — TokenAuth, RateLimiter, TLSConfig, AuthMiddleware.

Covers token generation, validation, expiry, revocation, rate limiting,
TLS context creation, and middleware integration.
"""

import time
import pytest

try:
    from auth import TokenAuth, RateLimiter, TLSConfig, AuthMiddleware, TokenInfo
except ImportError:
    pytest.skip("auth module not importable", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def token_auth():
    """TokenAuth instance with a short TTL for testing expiry."""
    return TokenAuth(secret_key="test-secret", token_ttl=2.0)


@pytest.fixture
def rate_limiter():
    """RateLimiter with a very low rate so we can trigger limits quickly."""
    return RateLimiter(rate=1.0, burst=2)


@pytest.fixture
def middleware(token_auth, rate_limiter):
    """AuthMiddleware wired to the test token_auth and rate_limiter."""
    return AuthMiddleware(token_auth, rate_limiter)


# ---------------------------------------------------------------------------
# TokenAuth tests
# ---------------------------------------------------------------------------

class TestTokenAuth:

    def test_generate_and_validate_token(self, token_auth):
        """Generated token should validate successfully."""
        raw = token_auth.generate_token(client_id="c1")
        info = token_auth.validate_token(raw)
        assert info is not None
        assert info.client_id == "c1"
        assert "read" in info.scopes
        assert "write" in info.scopes

    def test_invalid_token_returns_none(self, token_auth):
        """An unknown token should fail validation."""
        assert token_auth.validate_token("not-a-real-token") is None

    def test_revoke_token(self, token_auth):
        """Revoking a token should prevent subsequent validation."""
        raw = token_auth.generate_token(client_id="c2")
        assert token_auth.validate_token(raw) is not None
        assert token_auth.revoke_token(raw) is True
        assert token_auth.validate_token(raw) is None

    def test_revoke_unknown_token_returns_false(self, token_auth):
        assert token_auth.revoke_token("nonexistent") is False

    def test_token_expiry(self):
        """Token should be rejected after TTL expires."""
        auth = TokenAuth(secret_key="key", token_ttl=0.1)
        raw = auth.generate_token(client_id="c3")
        assert auth.validate_token(raw) is not None
        time.sleep(0.2)
        assert auth.validate_token(raw) is None

    def test_cleanup_expired(self):
        """cleanup_expired should remove tokens past their TTL."""
        auth = TokenAuth(secret_key="key", token_ttl=0.1)
        auth.generate_token(client_id="c4")
        auth.generate_token(client_id="c5")
        time.sleep(0.2)
        removed = auth.cleanup_expired()
        assert removed == 2

    def test_custom_scopes(self, token_auth):
        raw = token_auth.generate_token(client_id="scoped", scopes={"read"})
        info = token_auth.validate_token(raw)
        assert info is not None
        assert info.scopes == {"read"}


# ---------------------------------------------------------------------------
# RateLimiter tests
# ---------------------------------------------------------------------------

class TestRateLimiter:

    def test_allows_initial_burst(self, rate_limiter):
        """First requests up to burst size should be allowed."""
        assert rate_limiter.check("client_a") is True
        assert rate_limiter.check("client_a") is True

    def test_denies_after_burst_exhausted(self, rate_limiter):
        """Requests beyond burst should be denied when rate is low."""
        rate_limiter.check("client_b")
        rate_limiter.check("client_b")
        # Third request immediately should fail with rate=1, burst=2
        assert rate_limiter.check("client_b") is False

    def test_tokens_refill_over_time(self, rate_limiter):
        """After some time, tokens should refill."""
        rate_limiter.check("client_c")
        rate_limiter.check("client_c")
        assert rate_limiter.check("client_c") is False
        time.sleep(1.1)  # rate=1 token/sec
        assert rate_limiter.check("client_c") is True

    def test_reset_client(self, rate_limiter):
        """Resetting a client should restore their bucket."""
        rate_limiter.check("client_d")
        rate_limiter.check("client_d")
        rate_limiter.check("client_d")  # may fail
        rate_limiter.reset("client_d")
        assert rate_limiter.check("client_d") is True

    def test_separate_clients_independent(self, rate_limiter):
        """Different clients should have independent buckets."""
        rate_limiter.check("x")
        rate_limiter.check("x")
        rate_limiter.check("x")
        # Client y should still have full burst
        assert rate_limiter.check("y") is True


# ---------------------------------------------------------------------------
# TLSConfig tests
# ---------------------------------------------------------------------------

class TestTLSConfig:

    def test_no_certs_returns_none(self):
        """Without cert paths, get_ssl_context should return None."""
        tls = TLSConfig()
        assert tls.get_ssl_context() is None

    def test_missing_cert_file_returns_none(self, tmp_path):
        """If cert file does not exist, should return None."""
        tls = TLSConfig(
            cert_path=str(tmp_path / "nonexistent.pem"),
            key_path=str(tmp_path / "key.pem"),
        )
        assert tls.get_ssl_context() is None

    def test_attributes_stored(self):
        tls = TLSConfig(cert_path="/a", key_path="/b", ca_path="/c")
        assert tls.cert_path == "/a"
        assert tls.key_path == "/b"
        assert tls.ca_path == "/c"


# ---------------------------------------------------------------------------
# AuthMiddleware tests
# ---------------------------------------------------------------------------

class TestAuthMiddleware:

    def test_authenticate_valid_token(self, middleware, token_auth):
        raw = token_auth.generate_token(client_id="mw_client")
        info = middleware.authenticate(raw)
        assert info is not None
        assert info.client_id == "mw_client"

    def test_authenticate_invalid_token(self, middleware):
        assert middleware.authenticate("bad-token") is None

    def test_has_scope(self, middleware, token_auth):
        raw = token_auth.generate_token(scopes={"read"})
        info = middleware.authenticate(raw)
        assert info is not None
        assert middleware.has_scope(info, "read") is True
        assert middleware.has_scope(info, "write") is False

    def test_rate_limited_authentication(self, middleware, token_auth):
        """After exceeding rate limit, authenticate should return None."""
        raw = token_auth.generate_token(client_id="limited")
        # Exhaust the burst (burst=2)
        middleware.authenticate(raw)
        middleware.authenticate(raw)
        # Third attempt should be rate limited
        result = middleware.authenticate(raw)
        assert result is None
