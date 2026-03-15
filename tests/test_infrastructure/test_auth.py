"""Tests for auth module."""
import pytest
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from auth import AuthManager


class TestAuthManager:
    """Tests for the AuthManager class."""

    def test_init_defaults(self):
        """AuthManager initializes with expected defaults."""
        am = AuthManager()
        assert am.enabled is False
        # Secret key is auto-generated as a 64-char hex string
        assert len(am.secret_key) == 64
        assert '127.0.0.1' in am._allowed_ips
        assert '::1' in am._allowed_ips

    def test_generate_and_validate_token(self):
        """Generated tokens are valid until revoked or expired."""
        am = AuthManager(enabled=True)
        token = am.generate_token('client_1', expires_hours=1.0)
        assert isinstance(token, str)
        assert len(token) > 20
        assert am.validate_token(token) is True

    def test_validate_token_when_disabled(self):
        """Token validation always returns True when auth is disabled."""
        am = AuthManager(enabled=False)
        assert am.validate_token('bogus_token_that_doesnt_exist') is True

    def test_revoke_token(self):
        """Revoking a token makes it invalid on subsequent validation."""
        am = AuthManager(enabled=True)
        token = am.generate_token('client_2')
        assert am.validate_token(token) is True
        am.revoke_token(token)
        assert am.validate_token(token) is False

    def test_ip_allowlist(self):
        """check_ip allows listed IPs and rejects unlisted ones when enabled."""
        am = AuthManager(enabled=True)
        assert am.check_ip('127.0.0.1') is True
        assert am.check_ip('10.0.0.99') is False

        am.add_allowed_ip('10.0.0.99')
        assert am.check_ip('10.0.0.99') is True

    def test_rate_limiting(self):
        """Rate limiter rejects clients exceeding max_requests_per_minute."""
        am = AuthManager(enabled=True)
        am._max_requests_per_minute = 5

        client = 'rate_test_client'
        for _ in range(5):
            assert am.check_rate_limit(client) is True

        # 6th request within the same minute should be rejected
        assert am.check_rate_limit(client) is False

    def test_hmac_sign_and_verify(self):
        """create_signed_message and verify_signed_message round-trip correctly."""
        am = AuthManager(secret_key='test_secret_key_1234')
        payload = 'channel:5:gain:-3.0'
        signed = am.create_signed_message(payload)
        assert '|' in signed
        assert signed.startswith(payload + '|')

        # Verification returns the original payload
        verified = am.verify_signed_message(signed)
        assert verified == payload

    def test_hmac_verify_rejects_tampered_message(self):
        """verify_signed_message returns None for tampered messages."""
        am = AuthManager(secret_key='test_secret_key_1234')
        signed = am.create_signed_message('original_payload')

        # Tamper with the payload portion
        tampered = 'tampered_payload' + signed[signed.index('|'):]
        assert am.verify_signed_message(tampered) is None

        # No pipe separator
        assert am.verify_signed_message('no_pipe_here') is None

    def test_cleanup_expired_tokens(self):
        """cleanup_expired removes tokens past their expiration time."""
        am = AuthManager(enabled=True)
        token = am.generate_token('expiry_client', expires_hours=0.0)
        # Token was created with immediate expiry (expires = time.time() + 0)
        # Wait a tiny bit to ensure it's expired
        time.sleep(0.01)
        am.cleanup_expired()
        assert token not in am._tokens

    def test_authenticate_full_flow(self):
        """authenticate() checks IP, token, and rate limit in sequence."""
        am = AuthManager(enabled=True)
        token = am.generate_token('full_flow_client')

        # Valid everything
        assert am.authenticate(token=token, ip='127.0.0.1', client_id='full_flow_client') is True

        # Invalid IP
        assert am.authenticate(token=token, ip='192.168.0.1', client_id='full_flow_client') is False

        # Invalid token
        assert am.authenticate(token='bad_token', ip='127.0.0.1', client_id='full_flow_client') is False

        # When disabled, everything passes
        am_disabled = AuthManager(enabled=False)
        assert am_disabled.authenticate(token='any', ip='any', client_id='any') is True
