from sqlalchemy import Boolean, Column, Integer, String, DateTime, ForeignKey, Float, Text, Index
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)  # Admin flag
    
    # Wallet - PUBLIC ONLY (never store private keys!)
    # node_id is the public wallet address (32-char hex from ECDSA public key)
    # This is derived from the user's mnemonic/token but is safe to store
    node_id = Column(String, unique=True, index=True, nullable=True)
    wallet_id = Column(String, nullable=True)  # First 16 chars of node_id (display only)
    
    # Waitlist status - users must be approved before they can create wallets
    waitlist_approved = Column(Boolean, default=False)
    waitlist_id = Column(Integer, ForeignKey("waitlist_entries.id"), nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime, nullable=True)
    
    # Chat usage stats (aggregate for quick lookups)
    chat_count = Column(Integer, default=0)  # Total number of chat requests
    total_tokens_used = Column(Integer, default=0)  # Total tokens consumed
    total_neuro_spent_chat = Column(Float, default=0.0)  # Total NEURO spent on chat
    last_chat_at = Column(DateTime, nullable=True)  # Last chat timestamp
    
    # Rate limit tracking (for per-user limits)
    rate_limit_tier = Column(String, default="standard")  # standard, premium, unlimited
    is_rate_limited = Column(Boolean, default=False)  # Temporarily blocked
    rate_limit_until = Column(DateTime, nullable=True)  # When rate limit expires
    
    # Relationship to refresh tokens
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    
    # Relationship to waitlist entry
    waitlist_entry = relationship("WaitlistEntry", back_populates="user", foreign_keys=[waitlist_id])
    
    # Relationship to chat interactions
    chat_interactions = relationship("ChatInteraction", back_populates="user", cascade="all, delete-orphan")


class RefreshToken(Base):
    """
    Store refresh tokens for secure token management.
    Allows for token revocation and tracking active sessions.
    """
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token_id = Column(String, unique=True, index=True)  # Unique identifier for the token
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    revoked = Column(Boolean, default=False)
    
    # Relationship back to user
    user = relationship("User", back_populates="refresh_tokens")


class WaitlistEntry(Base):
    """
    Waitlist entries for users who want to join the NeuroShard network.
    Collects hardware specs and generates referral codes.
    Users can submit multiple applications (e.g., different hardware).
    """
    __tablename__ = "waitlist_entries"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, index=True)  # Not unique - allow multiple entries per email
    
    # Hardware specifications
    gpu_model = Column(String, nullable=True)  # e.g., "RTX 4090", "M2 Pro", "None"
    gpu_vram = Column(Integer, nullable=True)  # VRAM in GB
    ram_gb = Column(Integer, nullable=False)  # System RAM in GB
    internet_speed = Column(Integer, nullable=True)  # Mbps
    operating_system = Column(String, nullable=True)  # Windows, macOS, Linux
    
    # Calculated estimates
    estimated_daily_neuro = Column(Float, default=0.0)
    hardware_tier = Column(String, default="basic")  # basic, standard, pro, elite
    hardware_score = Column(Integer, default=0)  # 0-100 score
    
    # Referral system - "Neuro Link"
    referral_code = Column(String, unique=True, index=True)  # Unique 8-char code
    referred_by = Column(String, nullable=True)  # referral_code of referrer
    referral_count = Column(Integer, default=0)  # Number of successful referrals
    referral_bonus_percent = Column(Float, default=0.0)  # Bonus % from referrals
    
    # Status
    status = Column(String, default="pending")  # pending, approved, rejected, converted
    position = Column(Integer, nullable=True)  # Position in waitlist queue
    priority_score = Column(Integer, default=0)  # Higher = earlier access
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    approved_at = Column(DateTime, nullable=True)
    converted_at = Column(DateTime, nullable=True)  # When they completed full signup
    
    # Admin notes
    admin_notes = Column(Text, nullable=True)
    
    # Email tracking
    confirmation_email_sent = Column(Boolean, default=False)
    approval_email_sent = Column(Boolean, default=False)
    
    # Relationship to user (after conversion)
    user = relationship("User", back_populates="waitlist_entry", uselist=False, foreign_keys="User.waitlist_id")


class ChatInteraction(Base):
    """
    Track every chat interaction for analytics and abuse detection.
    
    Used for:
    - Usage analytics (popular times, average response times)
    - Abuse detection (spam patterns, unusual activity)
    - Billing verification (ensure NEURO charges match actual usage)
    - User experience monitoring (error rates, latency)
    """
    __tablename__ = "chat_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), index=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Request details
    prompt_length = Column(Integer, nullable=False)  # Characters in prompt
    max_tokens_requested = Column(Integer, default=50)  # User requested max
    
    # Response details
    response_length = Column(Integer, nullable=True)  # Characters in response
    tokens_used = Column(Integer, nullable=True)  # Actual tokens (prompt + response)
    
    # Cost tracking
    neuro_cost = Column(Float, nullable=True)  # NEURO charged
    fee_burned = Column(Float, nullable=True)  # 5% fee that was burned
    
    # Performance metrics
    response_time_ms = Column(Integer, nullable=True)  # Total latency
    node_response_time_ms = Column(Integer, nullable=True)  # Just node inference time
    
    # Status
    success = Column(Boolean, default=True)
    error_code = Column(String, nullable=True)  # HTTP status or error type
    error_message = Column(Text, nullable=True)  # Error details (truncated)
    
    # Node routing (for debugging)
    target_node_url = Column(String, nullable=True)  # Which node handled it
    nodes_tried = Column(Integer, default=1)  # How many nodes were tried
    
    # Client info (for abuse detection)
    client_ip = Column(String, nullable=True)  # Hashed or masked for privacy
    user_agent = Column(String, nullable=True)  # Browser/client info
    
    # Relationship to user
    user = relationship("User", back_populates="chat_interactions")
    
    # Indexes for common queries
    __table_args__ = (
        Index('ix_chat_user_created', 'user_id', 'created_at'),
        Index('ix_chat_created_success', 'created_at', 'success'),
    )


class RateLimitEvent(Base):
    """
    Log rate limit violations for monitoring and security.
    
    Helps identify:
    - Abuse patterns
    - DDoS attempts
    - Compromised accounts
    - Need for rate limit adjustments
    """
    __tablename__ = "rate_limit_events"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Who was limited
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Null if not authenticated
    client_ip = Column(String, nullable=False, index=True)  # Hashed IP
    
    # What was limited
    endpoint = Column(String, nullable=False)  # /api/chat, etc.
    limit_type = Column(String, nullable=False)  # per_minute, per_hour, burst, global
    limit_value = Column(String, nullable=False)  # e.g., "10/minute"
    
    # When
    created_at = Column(DateTime, server_default=func.now(), index=True)
    
    # Context
    request_count = Column(Integer, nullable=True)  # Requests in window
    user_agent = Column(String, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('ix_ratelimit_ip_created', 'client_ip', 'created_at'),
        Index('ix_ratelimit_user_created', 'user_id', 'created_at'),
    )

