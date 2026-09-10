from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    is_admin: bool = False
    node_id: Optional[str] = None  # Public wallet address (safe to expose)
    wallet_id: Optional[str] = None  # Short display ID
    waitlist_approved: bool = False

    class Config:
        from_attributes = True

class UserAdmin(User):
    """Extended user info for admin view"""
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

class WalletCreate(BaseModel):
    """Response when creating a new wallet - SHOW ONLY ONCE!"""
    mnemonic: str  # 12-word seed phrase (MUST be saved by user!)
    token: str  # Node token (PRIVATE - derived from mnemonic)
    node_id: str  # Public wallet address
    wallet_id: str  # Short display ID
    
class WalletConnect(BaseModel):
    """Request to connect wallet using mnemonic or token"""
    secret: str  # Can be either mnemonic phrase or token
    
class WalletInfo(BaseModel):
    """Public wallet information"""
    node_id: str
    wallet_id: str
    balance: float = 0.0

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int  # seconds until access token expires

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class TokenData(BaseModel):
    email: Optional[str] = None


# =============================================================================
# WAITLIST SCHEMAS
# =============================================================================

class WaitlistSignup(BaseModel):
    """Request to join the waitlist"""
    email: EmailStr
    
    # Hardware specifications
    gpu_model: Optional[str] = None  # e.g., "RTX 4090", "M2 Pro", "None/CPU Only"
    gpu_vram: Optional[int] = None  # VRAM in GB
    ram_gb: int = Field(..., ge=4, le=1024, description="System RAM in GB")
    internet_speed: Optional[int] = Field(None, ge=1, le=10000, description="Internet speed in Mbps")
    operating_system: Optional[str] = None
    
    # Referral
    referral_code: Optional[str] = None  # Code from referrer


class WaitlistResponse(BaseModel):
    """Response after successful waitlist signup"""
    id: int
    email: str
    referral_code: str  # Their unique "Neuro Link" code
    referral_url: str  # Full URL with referral code
    position: int  # Position in queue
    estimated_daily_neuro: float
    hardware_tier: str
    hardware_score: int
    priority_score: int
    status: str
    message: str

    class Config:
        from_attributes = True


class WaitlistStatus(BaseModel):
    """Check status of waitlist entry"""
    email: str
    status: str  # pending, approved, rejected, converted
    position: Optional[int]
    estimated_daily_neuro: float
    hardware_tier: str
    referral_code: str
    referral_count: int
    referral_bonus_percent: float
    priority_score: int
    created_at: datetime
    approved_at: Optional[datetime]

    class Config:
        from_attributes = True


class WaitlistAdminView(BaseModel):
    """Admin view of waitlist entry"""
    id: int
    email: str
    gpu_model: Optional[str]
    gpu_vram: Optional[int]
    ram_gb: int
    internet_speed: Optional[int]
    operating_system: Optional[str]
    estimated_daily_neuro: float
    hardware_tier: str
    hardware_score: int
    referral_code: str
    referred_by: Optional[str]
    referral_count: int
    referral_bonus_percent: float
    status: str
    position: Optional[int]
    priority_score: int
    created_at: datetime
    approved_at: Optional[datetime]
    converted_at: Optional[datetime]
    admin_notes: Optional[str]
    confirmation_email_sent: bool
    approval_email_sent: bool

    class Config:
        from_attributes = True


class WaitlistApproval(BaseModel):
    """Admin approval/rejection of waitlist entry"""
    waitlist_id: int
    action: str = Field(..., pattern="^(approve|reject)$")
    admin_notes: Optional[str] = None


class WaitlistStats(BaseModel):
    """Statistics about the waitlist"""
    total_entries: int
    pending_entries: int
    approved_entries: int
    rejected_entries: int
    converted_entries: int
    total_referrals: int
    avg_hardware_score: float
    tier_distribution: dict  # {"basic": 10, "standard": 25, ...}

