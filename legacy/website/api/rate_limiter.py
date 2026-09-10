"""
Rate Limiting Module for NeuroShard Website API

Production-ready rate limiting with:
- Redis backend for distributed deployments
- In-memory fallback for development
- Per-user limits based on tier
- Abuse detection and logging

User Rate Limit Tiers (for authenticated endpoints like /api/chat):
- Standard: 10/minute, 100/hour (default for new users)
- Premium: 30/minute, 500/hour (upgraded users)
- Unlimited: No limits (admin users)

Note: The /api/chat endpoint REQUIRES authentication.
Anonymous users cannot access chat - they get 401 Unauthorized.
"""

import os
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Tuple
from functools import wraps

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Redis imports with fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# RATE LIMIT CONFIGURATION
# =============================================================================

# Rate limits by tier (requests per time window)
# Users are assigned a tier in the database (default: "standard")
# Admins can upgrade users to "premium" or "unlimited" via admin panel
RATE_LIMITS = {
    # Tier: (per_minute, per_hour, burst_per_5s)
    "standard": (10, 100, 3),    # Default for new users
    "premium": (30, 500, 5),     # Upgraded users (set by admin)
    "unlimited": (1000, 10000, 100),  # Admin users (auto-assigned)
}

# Valid tiers that can be assigned to users
VALID_USER_TIERS = ["standard", "premium", "unlimited"]

# Global limits (DDoS protection - applies to all requests)
GLOBAL_RATE_LIMITS = {
    "per_minute": 500,
    "per_second": 50,
}

# Endpoints with custom limits
ENDPOINT_LIMITS = {
    "/api/chat": {
        # Chat requires authentication - no anonymous access
        "standard": "10/minute",
        "premium": "30/minute",
        "unlimited": "1000/minute",
    },
    "/api/auth/token": {
        "default": "5/minute",  # Login attempts (by IP)
    },
    "/api/auth/signup": {
        "default": "3/minute",  # Signup attempts (by IP)
    },
}

# =============================================================================
# REDIS CONNECTION
# =============================================================================

def get_redis_client() -> Optional[redis.Redis]:
    """
    Get Redis client for rate limiting storage.
    Returns None if Redis is not available or not configured.
    """
    if not REDIS_AVAILABLE:
        logger.warning("Redis library not installed, using in-memory rate limiting")
        return None
    
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        # Try individual connection params
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD")
        redis_db = int(os.getenv("REDIS_DB", "0"))
        
        try:
            client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                db=redis_db,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )
            # Test connection
            client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            return client
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory rate limiting")
            return None
    else:
        try:
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            logger.info("Connected to Redis via URL")
            return client
        except Exception as e:
            logger.warning(f"Redis URL connection failed: {e}, using in-memory rate limiting")
            return None


# Global Redis client (initialized on first use)
_redis_client: Optional[redis.Redis] = None
_redis_checked = False


def get_redis() -> Optional[redis.Redis]:
    """Get or create Redis client singleton."""
    global _redis_client, _redis_checked
    if not _redis_checked:
        _redis_client = get_redis_client()
        _redis_checked = True
    return _redis_client


# =============================================================================
# KEY FUNCTIONS
# =============================================================================

def hash_ip(ip: str) -> str:
    """Hash IP address for privacy-preserving storage."""
    # Use first 16 chars of SHA256 for reasonable uniqueness
    return hashlib.sha256(ip.encode()).hexdigest()[:16]


def get_client_ip(request: Request) -> str:
    """
    Get the real client IP, handling proxies and load balancers.
    Returns hashed IP for privacy.
    """
    # Check common proxy headers
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP in the chain (original client)
        ip = forwarded.split(",")[0].strip()
    else:
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            ip = real_ip
        else:
            # Fallback to direct connection
            ip = request.client.host if request.client else "unknown"
    
    return hash_ip(ip)


def get_user_key(request: Request) -> str:
    """
    Get rate limit key based on user identity.
    Uses user_id if authenticated, otherwise IP.
    """
    # Check if user is attached to request (set by dependency)
    user = getattr(request.state, "user", None)
    if user and hasattr(user, "id"):
        return f"user:{user.id}"
    
    # Fall back to IP-based limiting
    return f"ip:{get_client_ip(request)}"


def get_user_tier(request: Request) -> str:
    """
    Determine the rate limit tier for the current request.
    
    For authenticated endpoints (like /api/chat):
    - Returns user's tier from database (standard/premium/unlimited)
    - Admins automatically get "unlimited"
    
    For unauthenticated requests:
    - Returns "standard" as fallback (IP-based limiting still applies)
    """
    user = getattr(request.state, "user", None)
    
    if not user:
        # No user attached - use standard limits (for non-auth endpoints)
        # The actual auth check happens in the endpoint via Depends()
        return "standard"
    
    if hasattr(user, "is_admin") and user.is_admin:
        return "unlimited"
    
    if hasattr(user, "rate_limit_tier"):
        tier = user.rate_limit_tier
        if tier in RATE_LIMITS:
            return tier
    
    return "standard"


# =============================================================================
# SLOWAPI LIMITER SETUP
# =============================================================================

def create_limiter() -> Limiter:
    """
    Create and configure the SlowAPI limiter.
    Uses Redis if available, otherwise in-memory.
    """
    redis_client = get_redis()
    
    if redis_client:
        # Use Redis storage
        storage_uri = os.getenv("REDIS_URL")
        if not storage_uri:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = os.getenv("REDIS_PORT", "6379")
            redis_password = os.getenv("REDIS_PASSWORD", "")
            if redis_password:
                storage_uri = f"redis://:{redis_password}@{redis_host}:{redis_port}"
            else:
                storage_uri = f"redis://{redis_host}:{redis_port}"
        
        return Limiter(
            key_func=get_user_key,
            storage_uri=storage_uri,
            strategy="fixed-window",
            headers_enabled=True,  # Add X-RateLimit headers
        )
    else:
        # Use in-memory storage (development/single instance)
        return Limiter(
            key_func=get_user_key,
            storage_uri="memory://",
            strategy="fixed-window",
            headers_enabled=True,
        )


# Global limiter instance
limiter = create_limiter()


# =============================================================================
# RATE LIMIT EXCEPTION HANDLER
# =============================================================================

async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """
    Custom handler for rate limit exceeded errors.
    Logs the event and returns a proper JSON response.
    """
    from . import models, database
    
    # Extract rate limit info
    limit_str = str(exc.detail) if hasattr(exc, "detail") else "Rate limit exceeded"
    
    # Log the event
    client_ip = get_client_ip(request)
    user = getattr(request.state, "user", None)
    user_id = user.id if user else None
    
    logger.warning(
        f"Rate limit exceeded: ip={client_ip}, user_id={user_id}, "
        f"endpoint={request.url.path}, limit={limit_str}"
    )
    
    # Store in database for analysis (async would be better but keeping it simple)
    try:
        db = next(database.get_db())
        event = models.RateLimitEvent(
            user_id=user_id,
            client_ip=client_ip,
            endpoint=request.url.path,
            limit_type="request",
            limit_value=limit_str,
            user_agent=request.headers.get("User-Agent", "")[:200],
        )
        db.add(event)
        db.commit()
        db.close()
    except Exception as e:
        logger.error(f"Failed to log rate limit event: {e}")
    
    # Calculate retry-after from the limit string
    retry_after = 60  # Default to 1 minute
    if "minute" in limit_str.lower():
        retry_after = 60
    elif "hour" in limit_str.lower():
        retry_after = 3600
    elif "second" in limit_str.lower():
        retry_after = 5
    
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": {
                "code": "RATE_LIMITED",
                "message": "Too many requests. Please slow down.",
                "detail": limit_str,
                "retry_after": retry_after,
            }
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Reset": str(int(time.time()) + retry_after),
        }
    )


# =============================================================================
# CUSTOM RATE LIMIT DECORATORS
# =============================================================================

def get_dynamic_limit(endpoint: str) -> Callable:
    """
    Returns a function that determines the rate limit based on user tier.
    This allows different limits for different user types.
    """
    endpoint_config = ENDPOINT_LIMITS.get(endpoint, {})
    
    def dynamic_limit_func(request: Request) -> str:
        tier = get_user_tier(request)
        
        # Check endpoint-specific limits first
        if tier in endpoint_config:
            return endpoint_config[tier]
        if "default" in endpoint_config:
            return endpoint_config["default"]
        
        # Fall back to global tier limits
        limits = RATE_LIMITS.get(tier, RATE_LIMITS["standard"])
        return f"{limits[0]}/minute"
    
    return dynamic_limit_func


def rate_limit_chat(func):
    """
    Decorator for chat endpoint with tiered rate limiting.
    Applies both per-user and burst protection limits.
    """
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        # Get user tier for appropriate limits
        tier = get_user_tier(request)
        limits = RATE_LIMITS.get(tier, RATE_LIMITS["standard"])
        per_minute, per_hour, burst = limits
        
        # Check limits using the limiter
        key = get_user_key(request)
        
        # Apply the decorated function
        return await func(request, *args, **kwargs)
    
    return wrapper


# =============================================================================
# IN-MEMORY RATE LIMITER (Fallback)
# =============================================================================

class InMemoryRateLimiter:
    """
    Simple in-memory rate limiter for development or single-instance deployments.
    Uses sliding window algorithm.
    """
    
    def __init__(self):
        self.requests: dict = {}  # key -> list of timestamps
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
    
    def _cleanup(self):
        """Remove old entries to prevent memory growth."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff = now - 3600  # Keep last hour
        keys_to_delete = []
        
        for key, timestamps in self.requests.items():
            self.requests[key] = [ts for ts in timestamps if ts > cutoff]
            if not self.requests[key]:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.requests[key]
        
        self.last_cleanup = now
    
    def is_rate_limited(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, int]:
        """
        Check if a key is rate limited.
        
        Returns:
            (is_limited, remaining_requests)
        """
        self._cleanup()
        
        now = time.time()
        cutoff = now - window_seconds
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Filter to requests within window
        self.requests[key] = [ts for ts in self.requests[key] if ts > cutoff]
        
        current_count = len(self.requests[key])
        
        if current_count >= max_requests:
            return True, 0
        
        # Record this request
        self.requests[key].append(now)
        return False, max_requests - current_count - 1
    
    def get_requests_in_window(self, key: str, window_seconds: int) -> int:
        """Get the number of requests in the time window."""
        now = time.time()
        cutoff = now - window_seconds
        
        if key not in self.requests:
            return 0
        
        return len([ts for ts in self.requests[key] if ts > cutoff])


# Global in-memory limiter instance (used as fallback)
memory_limiter = InMemoryRateLimiter()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_rate_limit(
    request: Request,
    max_per_minute: int = 10,
    max_per_hour: int = 100,
    burst_per_5s: int = 3,
) -> Tuple[bool, str, int]:
    """
    Manual rate limit check (for use in endpoint logic).
    
    Returns:
        (is_limited, limit_type, remaining)
    """
    key = get_user_key(request)
    redis_client = get_redis()
    
    if redis_client:
        # Use Redis for distributed rate limiting
        pipe = redis_client.pipeline()
        now = int(time.time())
        
        # Check burst (5 second window)
        burst_key = f"ratelimit:burst:{key}"
        pipe.incr(burst_key)
        pipe.expire(burst_key, 5)
        
        # Check per-minute
        minute_key = f"ratelimit:minute:{key}:{now // 60}"
        pipe.incr(minute_key)
        pipe.expire(minute_key, 120)  # Keep for 2 minutes
        
        # Check per-hour
        hour_key = f"ratelimit:hour:{key}:{now // 3600}"
        pipe.incr(hour_key)
        pipe.expire(hour_key, 7200)  # Keep for 2 hours
        
        results = pipe.execute()
        burst_count = results[0]
        minute_count = results[2]
        hour_count = results[4]
        
        if burst_count > burst_per_5s:
            return True, "burst", 0
        if minute_count > max_per_minute:
            return True, "per_minute", 0
        if hour_count > max_per_hour:
            return True, "per_hour", 0
        
        return False, "", max_per_minute - minute_count
    
    else:
        # Use in-memory limiter
        is_limited, remaining = memory_limiter.is_rate_limited(
            f"{key}:minute", max_per_minute, 60
        )
        if is_limited:
            return True, "per_minute", 0
        
        is_limited, _ = memory_limiter.is_rate_limited(
            f"{key}:hour", max_per_hour, 3600
        )
        if is_limited:
            return True, "per_hour", 0
        
        is_limited, _ = memory_limiter.is_rate_limited(
            f"{key}:burst", burst_per_5s, 5
        )
        if is_limited:
            return True, "burst", 0
        
        return False, "", remaining


def get_rate_limit_status(request: Request) -> dict:
    """
    Get current rate limit status for a request.
    Useful for debugging and showing users their limits.
    """
    key = get_user_key(request)
    tier = get_user_tier(request)
    limits = RATE_LIMITS.get(tier, RATE_LIMITS["standard"])
    
    redis_client = get_redis()
    
    if redis_client:
        now = int(time.time())
        minute_key = f"ratelimit:minute:{key}:{now // 60}"
        hour_key = f"ratelimit:hour:{key}:{now // 3600}"
        
        minute_count = int(redis_client.get(minute_key) or 0)
        hour_count = int(redis_client.get(hour_key) or 0)
    else:
        minute_count = memory_limiter.get_requests_in_window(f"{key}:minute", 60)
        hour_count = memory_limiter.get_requests_in_window(f"{key}:hour", 3600)
    
    return {
        "tier": tier,
        "limits": {
            "per_minute": limits[0],
            "per_hour": limits[1],
            "burst_per_5s": limits[2],
        },
        "usage": {
            "requests_this_minute": minute_count,
            "requests_this_hour": hour_count,
        },
        "remaining": {
            "this_minute": max(0, limits[0] - minute_count),
            "this_hour": max(0, limits[1] - hour_count),
        }
    }
