"""
Waitlist API endpoints for NeuroShard
Handles waitlist signups, referral tracking, and admin approval.
"""

import os
import secrets
import string
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from . import models, schemas, database
from .dependencies import get_current_admin_user
from .email_service import send_waitlist_confirmation_email, send_waitlist_approval_email

router = APIRouter(prefix="/api/waitlist", tags=["waitlist"])

WEBSITE_URL = os.getenv("WEBSITE_URL", "https://neuroshard.com")


def generate_referral_code(length: int = 8) -> str:
    """Generate a unique, URL-safe referral code."""
    chars = string.ascii_uppercase + string.digits
    # Exclude confusing characters
    chars = chars.replace('0', '').replace('O', '').replace('I', '').replace('1', '').replace('L', '')
    return ''.join(secrets.choice(chars) for _ in range(length))


def calculate_hardware_score(
    gpu_model: Optional[str],
    gpu_vram: Optional[int],
    ram_gb: int,
    internet_speed: Optional[int]
) -> tuple[int, str, float]:
    """
    Calculate hardware score, tier, and estimated daily NEURO earnings.
    
    Returns: (hardware_score, hardware_tier, estimated_daily_neuro)
    """
    score = 0
    
    # GPU scoring (0-50 points)
    gpu_score = 0
    if gpu_model and gpu_model.lower() not in ["none", "cpu only", "no gpu", ""]:
        gpu_model_lower = gpu_model.lower()
        
        # NVIDIA GPUs
        if any(x in gpu_model_lower for x in ["4090", "a100", "h100"]):
            gpu_score = 50
        elif any(x in gpu_model_lower for x in ["4080", "3090", "a6000", "a40"]):
            gpu_score = 45
        elif any(x in gpu_model_lower for x in ["4070", "3080", "a5000"]):
            gpu_score = 40
        elif any(x in gpu_model_lower for x in ["4060", "3070", "a4000"]):
            gpu_score = 35
        elif any(x in gpu_model_lower for x in ["3060", "2080", "a2000"]):
            gpu_score = 30
        elif any(x in gpu_model_lower for x in ["2070", "1080", "t4"]):
            gpu_score = 25
        elif any(x in gpu_model_lower for x in ["2060", "1070"]):
            gpu_score = 20
        elif any(x in gpu_model_lower for x in ["1060", "1650", "1660"]):
            gpu_score = 15
        # AMD GPUs
        elif any(x in gpu_model_lower for x in ["7900", "6900"]):
            gpu_score = 45
        elif any(x in gpu_model_lower for x in ["7800", "6800"]):
            gpu_score = 40
        elif any(x in gpu_model_lower for x in ["7700", "6700"]):
            gpu_score = 35
        elif any(x in gpu_model_lower for x in ["7600", "6600"]):
            gpu_score = 30
        # Apple Silicon
        elif any(x in gpu_model_lower for x in ["m3 ultra", "m2 ultra"]):
            gpu_score = 45
        elif any(x in gpu_model_lower for x in ["m3 max", "m2 max"]):
            gpu_score = 40
        elif any(x in gpu_model_lower for x in ["m3 pro", "m2 pro"]):
            gpu_score = 35
        elif any(x in gpu_model_lower for x in ["m3", "m2", "m1 max"]):
            gpu_score = 30
        elif any(x in gpu_model_lower for x in ["m1 pro"]):
            gpu_score = 25
        elif any(x in gpu_model_lower for x in ["m1"]):
            gpu_score = 20
        else:
            # Generic GPU
            gpu_score = 15
        
        # VRAM bonus
        if gpu_vram:
            if gpu_vram >= 24:
                gpu_score = min(50, gpu_score + 5)
            elif gpu_vram >= 16:
                gpu_score = min(50, gpu_score + 3)
            elif gpu_vram >= 12:
                gpu_score = min(50, gpu_score + 2)
    
    score += gpu_score
    
    # RAM scoring (0-25 points)
    if ram_gb >= 128:
        score += 25
    elif ram_gb >= 64:
        score += 22
    elif ram_gb >= 32:
        score += 18
    elif ram_gb >= 16:
        score += 12
    elif ram_gb >= 8:
        score += 6
    else:
        score += 2
    
    # Internet scoring (0-25 points)
    if internet_speed:
        if internet_speed >= 1000:
            score += 25
        elif internet_speed >= 500:
            score += 22
        elif internet_speed >= 200:
            score += 18
        elif internet_speed >= 100:
            score += 14
        elif internet_speed >= 50:
            score += 10
        elif internet_speed >= 25:
            score += 6
        else:
            score += 3
    else:
        score += 10  # Default/unknown
    
    # Determine tier
    if score >= 80:
        tier = "elite"
        base_daily = 15.0
    elif score >= 60:
        tier = "pro"
        base_daily = 8.0
    elif score >= 40:
        tier = "standard"
        base_daily = 4.0
    else:
        tier = "basic"
        base_daily = 1.5
    
    # Calculate estimated daily NEURO (with some variance for realism)
    # This is gamified - actual earnings depend on network participation
    estimated_daily = base_daily * (1 + (score - 50) * 0.02)  # ±2% per point from 50
    estimated_daily = max(0.5, estimated_daily)  # Minimum 0.5 NEURO/day
    
    return score, tier, round(estimated_daily, 2)


def calculate_priority_score(
    hardware_score: int,
    referral_count: int,
    referred_by: Optional[str]
) -> int:
    """
    Calculate priority score for waitlist ordering.
    Higher = earlier access.
    """
    priority = hardware_score  # Base from hardware
    
    # Referral bonuses
    priority += referral_count * 10  # +10 per referral
    
    # Being referred gives a small bonus
    if referred_by:
        priority += 5
    
    return priority


@router.post("/signup", response_model=schemas.WaitlistResponse)
async def waitlist_signup(
    signup_data: schemas.WaitlistSignup,
    db: Session = Depends(database.get_db)
):
    """
    Join the waitlist with hardware specifications.
    
    This is the first step in the signup flow:
    1. User submits hardware specs
    2. We calculate estimated earnings and assign a tier
    3. User gets a unique referral code ("Neuro Link")
    4. Admin approves -> user can complete registration
    
    Note: Multiple entries with the same email are allowed (e.g., different hardware configs).
    """
    
    # Check if email already has a user account (already completed registration)
    existing_user = db.query(models.User).filter(
        models.User.email == signup_data.email
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="This email is already registered. Please log in instead."
        )
    
    # Generate unique referral code
    referral_code = generate_referral_code()
    while db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.referral_code == referral_code
    ).first():
        referral_code = generate_referral_code()
    
    # Calculate hardware metrics
    hardware_score, hardware_tier, estimated_daily = calculate_hardware_score(
        gpu_model=signup_data.gpu_model,
        gpu_vram=signup_data.gpu_vram,
        ram_gb=signup_data.ram_gb,
        internet_speed=signup_data.internet_speed
    )
    
    # Validate referral code if provided
    referred_by = None
    if signup_data.referral_code:
        referrer = db.query(models.WaitlistEntry).filter(
            models.WaitlistEntry.referral_code == signup_data.referral_code
        ).first()
        if referrer:
            referred_by = signup_data.referral_code
            # Increment referrer's count
            referrer.referral_count += 1
            # Update referrer's bonus (5% per referral, max 50%)
            referrer.referral_bonus_percent = min(50.0, referrer.referral_count * 5.0)
            # Recalculate referrer's priority
            referrer.priority_score = calculate_priority_score(
                referrer.hardware_score,
                referrer.referral_count,
                referrer.referred_by
            )
    
    # Calculate priority
    priority_score = calculate_priority_score(
        hardware_score=hardware_score,
        referral_count=0,
        referred_by=referred_by
    )
    
    # Get current position (count of pending + 1)
    pending_count = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.status == "pending"
    ).count()
    position = pending_count + 1
    
    # Create waitlist entry
    waitlist_entry = models.WaitlistEntry(
        email=signup_data.email,
        gpu_model=signup_data.gpu_model,
        gpu_vram=signup_data.gpu_vram,
        ram_gb=signup_data.ram_gb,
        internet_speed=signup_data.internet_speed,
        operating_system=signup_data.operating_system,
        estimated_daily_neuro=estimated_daily,
        hardware_tier=hardware_tier,
        hardware_score=hardware_score,
        referral_code=referral_code,
        referred_by=referred_by,
        priority_score=priority_score,
        position=position,
        status="pending"
    )
    
    db.add(waitlist_entry)
    db.commit()
    db.refresh(waitlist_entry)
    
    # Send confirmation email
    email_sent = send_waitlist_confirmation_email(
        to_email=signup_data.email,
        referral_code=referral_code,
        position=position,
        hardware_tier=hardware_tier,
        hardware_score=hardware_score,
        estimated_daily_neuro=estimated_daily,
        gpu_model=signup_data.gpu_model,
        ram_gb=signup_data.ram_gb,
        internet_speed=signup_data.internet_speed
    )
    
    waitlist_entry.confirmation_email_sent = email_sent
    db.commit()
    
    referral_url = f"{WEBSITE_URL}/join?ref={referral_code}"
    
    return schemas.WaitlistResponse(
        id=waitlist_entry.id,
        email=waitlist_entry.email,
        referral_code=referral_code,
        referral_url=referral_url,
        position=position,
        estimated_daily_neuro=estimated_daily,
        hardware_tier=hardware_tier,
        hardware_score=hardware_score,
        priority_score=priority_score,
        status="pending",
        message="Welcome to the network! Your hardware has been registered."
    )


@router.get("/status", response_model=schemas.WaitlistStatus)
async def get_waitlist_status(
    email: str = Query(..., description="Email to check status for"),
    db: Session = Depends(database.get_db)
):
    """Check waitlist status by email."""
    
    entry = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.email == email
    ).first()
    
    if not entry:
        raise HTTPException(
            status_code=404,
            detail="Email not found in waitlist. Please sign up first."
        )
    
    return schemas.WaitlistStatus(
        email=entry.email,
        status=entry.status,
        position=entry.position,
        estimated_daily_neuro=entry.estimated_daily_neuro,
        hardware_tier=entry.hardware_tier,
        referral_code=entry.referral_code,
        referral_count=entry.referral_count,
        referral_bonus_percent=entry.referral_bonus_percent,
        priority_score=entry.priority_score,
        created_at=entry.created_at,
        approved_at=entry.approved_at
    )


@router.get("/check-referral")
async def check_referral_code(
    code: str = Query(..., description="Referral code to validate"),
    db: Session = Depends(database.get_db)
):
    """Validate a referral code and get referrer info."""
    
    referrer = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.referral_code == code
    ).first()
    
    if not referrer:
        return {"valid": False, "message": "Invalid referral code"}
    
    return {
        "valid": True,
        "referrer_tier": referrer.hardware_tier,
        "message": f"Referred by a {referrer.hardware_tier} tier member"
    }


# =============================================================================
# ADMIN ENDPOINTS
# =============================================================================

@router.get("/admin/list", response_model=List[schemas.WaitlistAdminView])
async def admin_list_waitlist(
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    current_admin: models.User = Depends(get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Admin: Get all waitlist entries with filtering."""
    
    query = db.query(models.WaitlistEntry)
    
    if status_filter:
        query = query.filter(models.WaitlistEntry.status == status_filter)
    
    # Order by priority (highest first), then by created_at
    query = query.order_by(
        models.WaitlistEntry.priority_score.desc(),
        models.WaitlistEntry.created_at.asc()
    )
    
    entries = query.offset(offset).limit(limit).all()
    
    return [schemas.WaitlistAdminView.model_validate(e) for e in entries]


@router.get("/admin/stats", response_model=schemas.WaitlistStats)
async def admin_waitlist_stats(
    current_admin: models.User = Depends(get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Admin: Get waitlist statistics."""
    
    total = db.query(models.WaitlistEntry).count()
    pending = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.status == "pending"
    ).count()
    approved = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.status == "approved"
    ).count()
    rejected = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.status == "rejected"
    ).count()
    converted = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.status == "converted"
    ).count()
    
    # Total referrals
    total_referrals = db.query(func.sum(models.WaitlistEntry.referral_count)).scalar() or 0
    
    # Average hardware score
    avg_score = db.query(func.avg(models.WaitlistEntry.hardware_score)).scalar() or 0
    
    # Tier distribution
    tier_counts = db.query(
        models.WaitlistEntry.hardware_tier,
        func.count(models.WaitlistEntry.id)
    ).group_by(models.WaitlistEntry.hardware_tier).all()
    
    tier_distribution = {tier: count for tier, count in tier_counts}
    
    return schemas.WaitlistStats(
        total_entries=total,
        pending_entries=pending,
        approved_entries=approved,
        rejected_entries=rejected,
        converted_entries=converted,
        total_referrals=total_referrals,
        avg_hardware_score=round(avg_score, 1),
        tier_distribution=tier_distribution
    )


@router.post("/admin/approve")
async def admin_approve_waitlist(
    approval: schemas.WaitlistApproval,
    current_admin: models.User = Depends(get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Admin: Approve or reject a waitlist entry."""
    
    entry = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.id == approval.waitlist_id
    ).first()
    
    if not entry:
        raise HTTPException(status_code=404, detail="Waitlist entry not found")
    
    if entry.status not in ["pending"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot {approval.action} entry with status '{entry.status}'"
        )
    
    if approval.action == "approve":
        entry.status = "approved"
        entry.approved_at = datetime.utcnow()
        
        # Send approval email
        email_sent = send_waitlist_approval_email(
            to_email=entry.email,
            referral_code=entry.referral_code
        )
        entry.approval_email_sent = email_sent
        
        message = f"Approved {entry.email}. Approval email {'sent' if email_sent else 'failed to send'}."
        
    elif approval.action == "reject":
        entry.status = "rejected"
        message = f"Rejected {entry.email}."
    
    if approval.admin_notes:
        entry.admin_notes = approval.admin_notes
    
    db.commit()
    
    return {
        "success": True,
        "message": message,
        "entry_id": entry.id,
        "new_status": entry.status
    }


@router.post("/admin/bulk-approve")
async def admin_bulk_approve(
    count: int = Query(..., ge=1, le=100, description="Number of entries to approve"),
    tier_filter: Optional[str] = Query(None, description="Only approve specific tier"),
    current_admin: models.User = Depends(get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Admin: Bulk approve top N entries by priority score."""
    
    query = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.status == "pending"
    )
    
    if tier_filter:
        query = query.filter(models.WaitlistEntry.hardware_tier == tier_filter)
    
    # Get top entries by priority
    entries = query.order_by(
        models.WaitlistEntry.priority_score.desc()
    ).limit(count).all()
    
    approved_count = 0
    emails_sent = 0
    
    for entry in entries:
        entry.status = "approved"
        entry.approved_at = datetime.utcnow()
        
        if send_waitlist_approval_email(entry.email, entry.referral_code):
            entry.approval_email_sent = True
            emails_sent += 1
        
        approved_count += 1
    
    db.commit()
    
    return {
        "success": True,
        "approved_count": approved_count,
        "emails_sent": emails_sent,
        "message": f"Approved {approved_count} entries, sent {emails_sent} emails."
    }


@router.get("/admin/entry/{entry_id}", response_model=schemas.WaitlistAdminView)
async def admin_get_entry(
    entry_id: int,
    current_admin: models.User = Depends(get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Admin: Get detailed view of a single waitlist entry."""
    
    entry = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.id == entry_id
    ).first()
    
    if not entry:
        raise HTTPException(status_code=404, detail="Waitlist entry not found")
    
    return schemas.WaitlistAdminView.model_validate(entry)


@router.patch("/admin/entry/{entry_id}/notes")
async def admin_update_notes(
    entry_id: int,
    notes: str,
    current_admin: models.User = Depends(get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Admin: Update admin notes for a waitlist entry."""
    
    entry = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.id == entry_id
    ).first()
    
    if not entry:
        raise HTTPException(status_code=404, detail="Waitlist entry not found")
    
    entry.admin_notes = notes
    db.commit()
    
    return {"success": True, "message": "Notes updated"}


@router.post("/admin/test-email")
async def admin_test_email(
    to_email: str = Query(..., description="Email address to send test to"),
    current_admin: models.User = Depends(get_current_admin_user)
):
    """
    Admin: Send a test email to verify SES is configured correctly.
    """
    import os
    
    # Check if AWS credentials are configured
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    sender = os.getenv("SENDER_EMAIL", "noreply@neuroshard.com")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    config_status = {
        "aws_access_key_id": "✓ Set" if aws_key else "✗ Missing",
        "aws_secret_access_key": "✓ Set" if aws_secret else "✗ Missing",
        "sender_email": sender,
        "aws_region": region,
    }
    
    if not aws_key or not aws_secret:
        return {
            "success": False,
            "message": "AWS credentials not configured",
            "config": config_status
        }
    
    # Try to send test email
    success = send_waitlist_confirmation_email(
        to_email=to_email,
        referral_code="TEST1234",
        position=1,
        hardware_tier="standard",
        hardware_score=75,
        estimated_daily_neuro=5.0,
        gpu_model="Test GPU",
        ram_gb=32,
        internet_speed=100
    )
    
    return {
        "success": success,
        "message": "Test email sent successfully!" if success else "Failed to send email - check CloudWatch logs",
        "config": config_status,
        "sent_to": to_email
    }

