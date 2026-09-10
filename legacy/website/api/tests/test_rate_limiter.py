"""
Tests for Rate Limiter Module

Run with: pytest website/api/tests/test_rate_limiter.py -v
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from fastapi import Request
from fastapi.testclient import TestClient


class TestInMemoryRateLimiter:
    """Tests for the in-memory rate limiter fallback."""
    
    def test_basic_rate_limiting(self):
        """Test that requests are properly rate limited."""
        from ..rate_limiter import InMemoryRateLimiter
        
        limiter = InMemoryRateLimiter()
        key = "test_user_1"
        
        # First 5 requests should succeed
        for i in range(5):
            is_limited, remaining = limiter.is_rate_limited(key, max_requests=5, window_seconds=60)
            assert not is_limited, f"Request {i+1} should not be limited"
            assert remaining == 5 - i - 1
        
        # 6th request should be limited
        is_limited, remaining = limiter.is_rate_limited(key, max_requests=5, window_seconds=60)
        assert is_limited, "Request 6 should be limited"
        assert remaining == 0
    
    def test_different_keys_independent(self):
        """Test that different keys have independent limits."""
        from ..rate_limiter import InMemoryRateLimiter
        
        limiter = InMemoryRateLimiter()
        
        # Use up limit for user1
        for _ in range(5):
            limiter.is_rate_limited("user1", max_requests=5, window_seconds=60)
        
        # user2 should still be able to make requests
        is_limited, remaining = limiter.is_rate_limited("user2", max_requests=5, window_seconds=60)
        assert not is_limited
        assert remaining == 4
    
    def test_window_expiry(self):
        """Test that rate limits reset after window expires."""
        from ..rate_limiter import InMemoryRateLimiter
        
        limiter = InMemoryRateLimiter()
        key = "test_user_2"
        
        # Use up the limit with 1-second window
        for _ in range(3):
            limiter.is_rate_limited(key, max_requests=3, window_seconds=1)
        
        # Should be limited now
        is_limited, _ = limiter.is_rate_limited(key, max_requests=3, window_seconds=1)
        assert is_limited
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Should be able to make requests again
        is_limited, remaining = limiter.is_rate_limited(key, max_requests=3, window_seconds=1)
        assert not is_limited
        assert remaining == 2
    
    def test_get_requests_in_window(self):
        """Test counting requests in a window."""
        from ..rate_limiter import InMemoryRateLimiter
        
        limiter = InMemoryRateLimiter()
        key = "test_user_3"
        
        # Make 3 requests
        for _ in range(3):
            limiter.is_rate_limited(key, max_requests=10, window_seconds=60)
        
        count = limiter.get_requests_in_window(key, window_seconds=60)
        assert count == 3


class TestKeyFunctions:
    """Tests for rate limit key generation functions."""
    
    def test_hash_ip(self):
        """Test IP hashing for privacy."""
        from ..rate_limiter import hash_ip
        
        ip1 = "192.168.1.1"
        ip2 = "192.168.1.2"
        
        hash1 = hash_ip(ip1)
        hash2 = hash_ip(ip2)
        
        # Should be 16 characters
        assert len(hash1) == 16
        assert len(hash2) == 16
        
        # Different IPs should have different hashes
        assert hash1 != hash2
        
        # Same IP should have same hash
        assert hash_ip(ip1) == hash1
    
    def test_get_client_ip_direct(self):
        """Test getting client IP from direct connection."""
        from ..rate_limiter import get_client_ip
        
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {}
        mock_request.client.host = "10.0.0.1"
        
        ip = get_client_ip(mock_request)
        
        # Should return hashed IP
        assert len(ip) == 16
    
    def test_get_client_ip_forwarded(self):
        """Test getting client IP from X-Forwarded-For header."""
        from ..rate_limiter import get_client_ip
        
        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"X-Forwarded-For": "203.0.113.1, 10.0.0.1"}
        mock_request.client.host = "10.0.0.1"
        
        ip = get_client_ip(mock_request)
        
        # Should use the first IP in the chain
        from ..rate_limiter import hash_ip
        expected = hash_ip("203.0.113.1")
        assert ip == expected
    
    def test_get_user_key_authenticated(self):
        """Test getting rate limit key for authenticated user."""
        from ..rate_limiter import get_user_key
        
        mock_user = MagicMock()
        mock_user.id = 42
        
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()
        mock_request.state.user = mock_user
        mock_request.headers = {}
        mock_request.client.host = "10.0.0.1"
        
        key = get_user_key(mock_request)
        
        assert key == "user:42"
    
    def test_get_user_key_anonymous(self):
        """Test getting rate limit key for anonymous user."""
        from ..rate_limiter import get_user_key
        
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()
        mock_request.state.user = None
        mock_request.headers = {}
        mock_request.client.host = "10.0.0.1"
        
        key = get_user_key(mock_request)
        
        assert key.startswith("ip:")


class TestTierDetermination:
    """Tests for rate limit tier determination."""
    
    def test_anonymous_tier(self):
        """Test that unauthenticated requests get anonymous tier."""
        from ..rate_limiter import get_user_tier
        
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()
        mock_request.state.user = None
        
        tier = get_user_tier(mock_request)
        assert tier == "anonymous"
    
    def test_admin_tier(self):
        """Test that admin users get unlimited tier."""
        from ..rate_limiter import get_user_tier
        
        mock_user = MagicMock()
        mock_user.is_admin = True
        mock_user.rate_limit_tier = "standard"
        
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()
        mock_request.state.user = mock_user
        
        tier = get_user_tier(mock_request)
        assert tier == "unlimited"
    
    def test_standard_tier(self):
        """Test that regular users get standard tier."""
        from ..rate_limiter import get_user_tier
        
        mock_user = MagicMock()
        mock_user.is_admin = False
        mock_user.rate_limit_tier = "standard"
        
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()
        mock_request.state.user = mock_user
        
        tier = get_user_tier(mock_request)
        assert tier == "standard"
    
    def test_premium_tier(self):
        """Test that premium users get premium tier."""
        from ..rate_limiter import get_user_tier
        
        mock_user = MagicMock()
        mock_user.is_admin = False
        mock_user.rate_limit_tier = "premium"
        
        mock_request = MagicMock(spec=Request)
        mock_request.state = MagicMock()
        mock_request.state.user = mock_user
        
        tier = get_user_tier(mock_request)
        assert tier == "premium"


class TestRateLimitConfiguration:
    """Tests for rate limit configuration."""
    
    def test_rate_limits_defined(self):
        """Test that all expected tiers are defined."""
        from ..rate_limiter import RATE_LIMITS
        
        assert "anonymous" in RATE_LIMITS
        assert "standard" in RATE_LIMITS
        assert "premium" in RATE_LIMITS
        assert "unlimited" in RATE_LIMITS
    
    def test_rate_limit_values(self):
        """Test that rate limits have correct structure."""
        from ..rate_limiter import RATE_LIMITS
        
        for tier, limits in RATE_LIMITS.items():
            assert len(limits) == 3, f"Tier {tier} should have 3 limits"
            per_minute, per_hour, burst = limits
            assert per_minute > 0
            assert per_hour > per_minute
            assert burst > 0
    
    def test_tier_ordering(self):
        """Test that higher tiers have more generous limits."""
        from ..rate_limiter import RATE_LIMITS
        
        anonymous = RATE_LIMITS["anonymous"]
        standard = RATE_LIMITS["standard"]
        premium = RATE_LIMITS["premium"]
        
        # Per-minute limits should increase
        assert anonymous[0] < standard[0] < premium[0]
        
        # Per-hour limits should increase
        assert anonymous[1] < standard[1] < premium[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
