from fastapi import FastAPI, Depends, HTTPException, status, Header, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import requests
import asyncio
import time
import logging
from typing import Optional, List, Dict
from dotenv import load_dotenv
from . import models, schemas, database, auth_utils, dependencies, downloads, ledger
from . import waitlist as waitlist_module
from .wallet import wallet_manager
from .rate_limiter import (
    limiter,
    rate_limit_exceeded_handler,
    check_rate_limit,
    get_rate_limit_status,
    get_user_tier,
    get_client_ip,
    RATE_LIMITS,
    VALID_USER_TIERS,
)
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import random
import os
import uuid
import math
import hashlib

from neuroshard.core.economics import calculate_stake_multiplier

logger = logging.getLogger(__name__)

# ECDSA node_id derivation (matches neuroshard/core/crypto.py)
def derive_ecdsa_node_id(token: str) -> str:
    """
    Derive ECDSA node_id from token.
    
    This matches the derivation in neuroshard/core/crypto.py:
    1. private_key = SHA256(token)
    2. public_key = ECDSA_derive(private_key) on secp256k1
    3. node_id = SHA256(public_key)[:32]
    
    Since we don't want to import the full crypto module here,
    we compute it directly using the cryptography library.
    """
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    
    # Derive private key from token
    private_key_bytes = hashlib.sha256(token.encode()).digest()
    
    # Create ECDSA private key
    private_key = ec.derive_private_key(
        int.from_bytes(private_key_bytes, 'big'),
        ec.SECP256K1(),
        default_backend()
    )
    
    # Get compressed public key
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.CompressedPoint
    )
    
    # node_id = SHA256(public_key)[:32]
    return hashlib.sha256(public_key_bytes).hexdigest()[:32]

# Load environment variables
load_dotenv()

# Initialize SQLite Database for Users (Decoupled from Tracker)
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(
    title="NeuroShard API",
    description="API for NeuroShard distributed AI network",
    version="1.0.0",
)

# Configure CORS
cors_origins = os.getenv("CORS_ORIGINS", "https://neuroshard.com,http://localhost:5173,http://localhost:8000,http://127.0.0.1:5173").split(",")
allow_origins = [origin.strip() for origin in cors_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiter to app state
app.state.limiter = limiter

# Register rate limit exception handler
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.include_router(downloads.router, prefix="/api/downloads", tags=["downloads"])
app.include_router(ledger.router)
app.include_router(waitlist_module.router)

# DECENTRALIZED: No more central reward loop hitting Postgres or Tracker
# Credits are now handled by the LedgerManager on individual nodes.
# The website database (sqlite/postgres) only stores "purchased" credits or initial grants.
# Real-time mining rewards live in the node's local wallet.

@app.on_event("startup")
async def startup_event():
    """
    Application startup tasks:
    1. Run database migrations for new tracking tables
    2. Initialize Redis connection for rate limiting
    3. Log startup info
    """
    logger.info("NeuroShard API starting up...")
    
    # Run database migrations
    try:
        from .migrations import migrate, verify_schema
        status = verify_schema()
        if status.get("status") == "needs_migration":
            logger.info("Database needs migration, applying...")
            migrate()
            logger.info("Database migrations completed.")
        else:
            logger.info("Database schema is up to date.")
    except Exception as e:
        logger.warning(f"Migration check failed (may be first run): {e}")
    
    # Log rate limiter status
    from .rate_limiter import get_redis
    redis_client = get_redis()
    if redis_client:
        logger.info("Rate limiter: Using Redis backend")
    else:
        logger.info("Rate limiter: Using in-memory backend (not recommended for production)")
    
    logger.info("NeuroShard API started successfully.")

class ChatRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50


@app.post("/api/chat")
@limiter.limit("10/minute")  # Base rate limit (overridden by tier)
@limiter.limit("3/5seconds")  # Burst protection
async def chat_proxy(
    request: Request,
    response: Response,
    req: ChatRequest,
    current_user: models.User = Depends(dependencies.get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Proxy chat requests to an available Entry Node in the swarm. 
    Requires Authentication and NEURO balance.
    
    Rate Limits:
    - Standard users: 10 requests/minute, 100/hour
    - Premium users: 30 requests/minute, 500/hour  
    - Burst protection: 3 requests per 5 seconds
    
    Fee Structure (from economics.py):
    - Cost: 0.1 NEURO per 1M tokens (INFERENCE_REWARD_PER_MILLION)
    - Fee: 5% burned (deflationary)
    """
    import sqlite3
    
    # Start timing for metrics
    start_time = time.time()
    
    # Get client info for tracking
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("User-Agent", "")[:200]
    
    # Attach user to request state for rate limiter
    request.state.user = current_user
    
    # Create interaction record (will be updated with results)
    interaction = models.ChatInteraction(
        user_id=current_user.id,
        prompt_length=len(req.prompt),
        max_tokens_requested=req.max_new_tokens,
        client_ip=client_ip,
        user_agent=user_agent,
    )
    db.add(interaction)
    db.flush()  # Get ID but don't commit yet
    
    # Initialize tracking variables
    target_node_used = None
    nodes_tried = 0
    resp = None
    error_code = None
    error_message = None
    
    try:
        # Check if user is temporarily rate limited (database flag)
        if current_user.is_rate_limited:
            if current_user.rate_limit_until and current_user.rate_limit_until > datetime.utcnow():
                remaining = (current_user.rate_limit_until - datetime.utcnow()).seconds
                interaction.success = False
                interaction.error_code = "429"
                interaction.error_message = "User temporarily rate limited"
                db.commit()
                raise HTTPException(
                    status_code=429,
                    detail=f"You are temporarily rate limited. Try again in {remaining} seconds.",
                    headers={"Retry-After": str(remaining)}
                )
            else:
                # Rate limit expired, clear it
                current_user.is_rate_limited = False
                current_user.rate_limit_until = None
        
        # Apply tiered rate limiting based on user status
        tier = get_user_tier(request)
        limits = RATE_LIMITS.get(tier, RATE_LIMITS["standard"])
        per_minute, per_hour, burst = limits
        
        is_limited, limit_type, remaining = check_rate_limit(
            request,
            max_per_minute=per_minute,
            max_per_hour=per_hour,
            burst_per_5s=burst,
        )
        
        if is_limited:
            # Log rate limit event
            rate_event = models.RateLimitEvent(
                user_id=current_user.id,
                client_ip=client_ip,
                endpoint="/api/chat",
                limit_type=limit_type,
                limit_value=f"{per_minute}/minute, {per_hour}/hour",
                user_agent=user_agent,
            )
            db.add(rate_event)
            
            interaction.success = False
            interaction.error_code = "429"
            interaction.error_message = f"Rate limit exceeded: {limit_type}"
            db.commit()
            
            retry_after = 60 if limit_type == "per_minute" else (5 if limit_type == "burst" else 3600)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({limit_type}). Please wait before trying again.",
                headers={"Retry-After": str(retry_after)}
            )
        
        # Check NEURO balance from ledger
        if not current_user.node_id:
            interaction.success = False
            interaction.error_code = "400"
            interaction.error_message = "No wallet connected"
            db.commit()
            raise HTTPException(
                status_code=400,
                detail="Wallet required. Please create or connect a wallet in the dashboard."
            )
        
        # Get NEURO balance from new NEUROLedger format
        node_id = current_user.node_id
        
        # Check LEDGER_DB_PATH first, then construct from LEDGER_DATA_DIR
        ledger_db_path = os.getenv("LEDGER_DB_PATH")
        if not ledger_db_path:
            ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
            # Check for observer_ledger.db first (shared from observer container)
            observer_ledger = os.path.join(ledger_data_dir, "observer_ledger.db")
            if os.path.exists(observer_ledger):
                ledger_db_path = observer_ledger
            else:
                ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
        
        # Fee burn constants (from whitepaper)
        FEE_BURN_RATE = 0.05  # 5% burned
        BURN_ADDRESS = "BURN_0x0000000000000000000000000000000000000000"
        
        neuro_balance = 0.0
        if os.path.exists(ledger_db_path):
            def _read_balance():
                try:
                    conn = sqlite3.connect(ledger_db_path, check_same_thread=False)
                    cursor = conn.cursor()
                    try:
                        cursor.execute("SELECT balance FROM balances WHERE node_id = ?", (node_id,))
                        row = cursor.fetchone()
                        if row:
                            return row[0]
                    except sqlite3.OperationalError:
                        cursor.execute("SELECT balance FROM credits WHERE node_id = ?", (node_id,))
                        row = cursor.fetchone()
                        if row:
                            return row[0]
                    conn.close()
                except Exception as e:
                    logger.error(f"Ledger read error: {e}")
                return 0.0
            neuro_balance = await asyncio.to_thread(_read_balance)
        
        # Pricing Model: Based on economics.py - INFERENCE_REWARD_PER_MILLION per 1M tokens + 5% fee
        try:
            from neuroshard.core.economics import INFERENCE_REWARD_PER_MILLION
            price_per_million = INFERENCE_REWARD_PER_MILLION  # 0.1 NEURO per 1M tokens
        except ImportError:
            price_per_million = 0.1  # Fallback
        
        prompt_tokens = len(req.prompt) // 4
        total_estimated_tokens = prompt_tokens + req.max_new_tokens
        base_cost = (total_estimated_tokens / 1_000_000.0) * price_per_million
        base_cost = max(0.00001, base_cost)  # Minimum cost (lowered)
        
        fee = base_cost * FEE_BURN_RATE
        total_cost = base_cost + fee
        
        if neuro_balance < total_cost:
            interaction.success = False
            interaction.error_code = "402"
            interaction.error_message = "Insufficient NEURO balance"
            db.commit()
            raise HTTPException(
                status_code=402, 
                detail=f"Insufficient NEURO. Request costs {total_cost:.6f} NEURO ({total_estimated_tokens} tokens + 5% fee), but you have {neuro_balance:.6f} NEURO. Keep your node running to earn more!"
            )

        # Find Entry Nodes (Decentralized Discovery)
        tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
        
        def _fetch_peers():
            try:
                peers_resp = requests.get(f"{tracker_url}/peers", params={"layer_needed": 0}, timeout=2)
                if peers_resp.status_code == 200:
                    return [p['url'] for p in peers_resp.json()]
            except Exception as e:
                logger.error(f"Tracker error: {e}")
            return []
        
        entry_nodes = await asyncio.to_thread(_fetch_peers)
            
        if not entry_nodes:
            interaction.success = False
            interaction.error_code = "503"
            interaction.error_message = "No entry nodes available"
            db.commit()
            raise HTTPException(status_code=503, detail="No Entry Nodes available in the swarm. Please wait for nodes to register.")
        
        # Try multiple nodes with retry logic for robustness
        last_error = None
        random.shuffle(entry_nodes)  # Randomize order for load balancing
        
        def _call_swarm_nodes():
            """Run blocking HTTP calls to swarm nodes in a thread to avoid blocking the event loop."""
            nonlocal nodes_tried, target_node_used
            _last_error = None
            _resp = None
            
            for target_node in entry_nodes[:3]:
                nodes_tried += 1
                
                node_url = target_node
                if "localhost" in node_url or "127.0.0.1" in node_url:
                    node_url = node_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")

                logger.info(f"Forwarding chat to Node: {node_url}")
                target_node_used = node_url

                try:
                    node_start = time.time()
                    _resp = requests.post(f"{node_url}/generate_text", json={
                        "prompt": req.prompt,
                        "max_new_tokens": req.max_new_tokens
                    }, timeout=30)
                    node_time = int((time.time() - node_start) * 1000)
                    interaction.node_response_time_ms = node_time
                    
                    if _resp.status_code == 200:
                        return _resp, None
                    else:
                        _last_error = f"Node returned {_resp.status_code}"
                        logger.warning(f"Node {node_url} returned {_resp.status_code}, trying next...")
                        continue
                except requests.exceptions.Timeout:
                    _last_error = f"Node {node_url} timed out"
                    logger.warning(f"Node {node_url} timed out, trying next...")
                    continue
                except requests.exceptions.ConnectionError as e:
                    _last_error = f"Could not connect to {node_url}"
                    logger.warning(f"Could not connect to {node_url}: {e}, trying next...")
                    continue
                except Exception as e:
                    _last_error = str(e)
                    logger.warning(f"Error with {node_url}: {e}, trying next...")
                    continue
            
            return _resp, _last_error
        
        resp, last_error = await asyncio.to_thread(_call_swarm_nodes)
        
        if last_error and (resp is None or resp.status_code != 200):
            interaction.success = False
            interaction.error_code = "503"
            interaction.error_message = f"All nodes failed: {last_error}"
            interaction.nodes_tried = nodes_tried
            interaction.response_time_ms = int((time.time() - start_time) * 1000)
            db.commit()
            raise HTTPException(
                status_code=503, 
                detail=f"All entry nodes failed. Last error: {last_error}"
            ) 
        
        if resp.status_code == 200:
            result = resp.json()
            response_text = result.get("text", result.get("result", ""))
            
            # Deduct NEURO with fee burn (deflationary mechanism)
            if os.path.exists(ledger_db_path):
                def _deduct_balance():
                    try:
                        conn = sqlite3.connect(ledger_db_path, check_same_thread=False)
                        cursor = conn.cursor()
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='balances'")
                        use_new_format = cursor.fetchone() is not None
                        if use_new_format:
                            cursor.execute("""
                                UPDATE balances SET 
                                    balance = balance - ?,
                                    total_spent = COALESCE(total_spent, 0) + ?
                                WHERE node_id = ?
                            """, (total_cost, total_cost, node_id))
                            cursor.execute("""
                                INSERT INTO balances (node_id, balance, total_earned, created_at)
                                VALUES (?, ?, ?, ?)
                                ON CONFLICT(node_id) DO UPDATE SET
                                    balance = balance + ?
                            """, (BURN_ADDRESS, fee, fee, time.time(), fee))
                            cursor.execute("""
                                UPDATE global_stats SET
                                    total_burned = COALESCE(total_burned, 0) + ?,
                                    updated_at = ?
                                WHERE id = 1
                            """, (fee, time.time()))
                            logger.info(f"NEURO spent: {base_cost:.6f} + {fee:.6f} fee (burned)")
                        else:
                            cursor.execute("UPDATE credits SET balance = balance - ? WHERE node_id = ?", (total_cost, node_id))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        logger.error(f"Ledger deduct error: {e}")
                await asyncio.to_thread(_deduct_balance)

            # Calculate actual tokens used (estimate based on response)
            actual_tokens = prompt_tokens + (len(response_text) // 4)
            
            # Update interaction record with success
            interaction.success = True
            interaction.response_length = len(response_text)
            interaction.tokens_used = actual_tokens
            interaction.neuro_cost = total_cost
            interaction.fee_burned = fee
            interaction.target_node_url = target_node_used
            interaction.nodes_tried = nodes_tried
            interaction.completed_at = datetime.utcnow()
            interaction.response_time_ms = int((time.time() - start_time) * 1000)
            
            # Update user aggregate stats
            current_user.chat_count = (current_user.chat_count or 0) + 1
            current_user.total_tokens_used = (current_user.total_tokens_used or 0) + actual_tokens
            current_user.total_neuro_spent_chat = (current_user.total_neuro_spent_chat or 0) + total_cost
            current_user.last_chat_at = datetime.utcnow()
            
            db.commit()
            
            return result
        else:
            interaction.success = False
            interaction.error_code = str(resp.status_code)
            interaction.error_message = resp.text[:500] if resp.text else "Unknown error"
            interaction.nodes_tried = nodes_tried
            interaction.response_time_ms = int((time.time() - start_time) * 1000)
            db.commit()
            raise HTTPException(status_code=502, detail=f"Node Error: {resp.text}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat Proxy Error: {e}")
        interaction.success = False
        interaction.error_code = "500"
        interaction.error_message = str(e)[:500]
        interaction.response_time_ms = int((time.time() - start_time) * 1000)
        try:
            db.commit()
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auth/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth_utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_utils.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    # Create refresh token with unique ID for revocation support
    token_id = auth_utils.generate_refresh_token_id()
    refresh_token_expires = timedelta(days=auth_utils.REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_token = auth_utils.create_refresh_token(
        data={"sub": user.email, "token_id": token_id}, expires_delta=refresh_token_expires
    )
    
    # Store refresh token in database for revocation tracking
    db_refresh_token = models.RefreshToken(
        token_id=token_id,
        user_id=user.id,
        expires_at=datetime.utcnow() + refresh_token_expires
    )
    db.add(db_refresh_token)
    
    # Update last login timestamp
    user.last_login = datetime.utcnow()
    db.commit()
    
    return {
        "access_token": access_token, 
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES * 60  # seconds
    }


@app.post("/api/auth/token/refresh", response_model=schemas.Token)
async def refresh_access_token(
    request: schemas.RefreshTokenRequest,
    db: Session = Depends(database.get_db)
):
    """
    Refresh an access token using a valid refresh token.
    
    This allows clients to get a new access token without re-authenticating
    with username/password, as long as the refresh token is still valid.
    """
    # Verify the refresh token JWT
    payload = auth_utils.verify_refresh_token(request.refresh_token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    email = payload.get("sub")
    token_id = payload.get("token_id")
    
    if not email or not token_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if token exists and is not revoked
    db_token = db.query(models.RefreshToken).filter(
        models.RefreshToken.token_id == token_id
    ).first()
    
    if not db_token or db_token.revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get the user
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create new access token
    access_token_expires = timedelta(minutes=auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_utils.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    # Rotate refresh token (create new one, revoke old one)
    new_token_id = auth_utils.generate_refresh_token_id()
    refresh_token_expires = timedelta(days=auth_utils.REFRESH_TOKEN_EXPIRE_DAYS)
    new_refresh_token = auth_utils.create_refresh_token(
        data={"sub": user.email, "token_id": new_token_id}, expires_delta=refresh_token_expires
    )
    
    # Revoke old token and create new one
    db_token.revoked = True
    new_db_token = models.RefreshToken(
        token_id=new_token_id,
        user_id=user.id,
        expires_at=datetime.utcnow() + refresh_token_expires
    )
    db.add(new_db_token)
    db.commit()
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "expires_in": auth_utils.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }


@app.post("/api/auth/logout")
async def logout(
    request: schemas.RefreshTokenRequest,
    db: Session = Depends(database.get_db)
):
    """
    Logout by revoking the refresh token.
    
    This prevents the refresh token from being used to generate new access tokens.
    The client should also clear any stored tokens locally.
    """
    payload = auth_utils.verify_refresh_token(request.refresh_token)
    if payload:
        token_id = payload.get("token_id")
        if token_id:
            db_token = db.query(models.RefreshToken).filter(
                models.RefreshToken.token_id == token_id
            ).first()
            if db_token:
                db_token.revoked = True
                db.commit()
    
    return {"message": "Successfully logged out"}

@app.post("/api/auth/signup", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    """
    Create new user account.
    
    WAITLIST FLOW:
    - User must first join waitlist and be approved
    - Upon approval, they receive an email with signup link
    - This endpoint checks waitlist status before allowing account creation
    
    NOTE: This does NOT create a wallet automatically.
    User must call /wallet/create or /wallet/connect after signup.
    """
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Check if user is on the waitlist and approved
    waitlist_entry = db.query(models.WaitlistEntry).filter(
        models.WaitlistEntry.email == user.email
    ).first()
    
    if not waitlist_entry:
        raise HTTPException(
            status_code=403,
            detail="You must join the waitlist first. Please register your hardware at /join"
        )
    
    if waitlist_entry.status == "pending":
        raise HTTPException(
            status_code=403,
            detail="Your waitlist application is still pending approval. We'll email you when you're approved."
        )
    
    if waitlist_entry.status == "rejected":
        raise HTTPException(
            status_code=403,
            detail="Your waitlist application was not approved. Please contact support."
        )
    
    if waitlist_entry.status == "converted":
        raise HTTPException(
            status_code=400,
            detail="This waitlist entry has already been used to create an account."
        )
    
    # User is approved - allow account creation
    hashed_password = auth_utils.get_password_hash(user.password)
    
    db_user = models.User(
        email=user.email, 
        hashed_password=hashed_password,
        node_id=None,  # Will be set when wallet is created/connected
        wallet_id=None,
        waitlist_approved=True,
        waitlist_id=waitlist_entry.id
    )
    db.add(db_user)
    
    # Mark waitlist entry as converted
    waitlist_entry.status = "converted"
    waitlist_entry.converted_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/api/wallet/create", response_model=schemas.WalletCreate)
async def create_wallet(
    current_user: models.User = Depends(dependencies.get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Generate a NEW wallet with BIP39 mnemonic seed phrase.
    
    ⚠️  CRITICAL: The mnemonic is shown ONLY ONCE!
    User MUST save it - we don't store private keys in the database.
    
    Similar to MetaMask wallet creation.
    """
    # Check if user already has a wallet
    if current_user.node_id:
        raise HTTPException(
            status_code=400,
            detail="Wallet already exists. Use /wallet/recover to import a different wallet."
        )
    
    # Generate new wallet
    wallet = wallet_manager.create_wallet()
    
    # Save ONLY public info to database
    current_user.node_id = wallet['node_id']
    current_user.wallet_id = wallet['wallet_id']
    db.commit()
    
    # Return everything INCLUDING the mnemonic (shown only this once!)
    return schemas.WalletCreate(**wallet)

@app.post("/api/wallet/connect", response_model=schemas.WalletInfo)
async def connect_wallet(
    wallet_data: schemas.WalletConnect,
    current_user: models.User = Depends(dependencies.get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Connect/Import wallet using mnemonic seed phrase or node token.
    
    This allows users to:
    - Import existing wallet from another device
    - Recover wallet from backup
    - Switch wallets
    
    Similar to MetaMask "Import Wallet" feature.
    """
    try:
        # Try to recover wallet from the secret (mnemonic or token)
        secret = wallet_data.secret.strip()
        
        # Check if it's a mnemonic (12 words) or a token (hex string)
        if len(secret.split()) == 12:
            # It's a mnemonic
            wallet = wallet_manager.recover_wallet(secret)
        else:
            # It's a raw token - derive node_id
            wallet = {
                'token': secret,
                'node_id': wallet_manager.token_to_node_id(secret),
                'wallet_id': wallet_manager.token_to_node_id(secret)[:16]
            }
        
        # Check if this wallet is already used by another user
        existing_user = db.query(models.User).filter(
            models.User.node_id == wallet['node_id'],
            models.User.id != current_user.id
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="This wallet is already connected to another account"
            )
        
        # Save ONLY public info to database
        current_user.node_id = wallet['node_id']
        current_user.wallet_id = wallet['wallet_id']
        db.commit()
        
        # Return only public info
        return schemas.WalletInfo(
            node_id=wallet['node_id'],
            wallet_id=wallet['wallet_id'],
            balance=0.0  # Will be fetched from ledger
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to connect wallet: {str(e)}")

@app.get("/api/users/me/wallet")
async def read_user_wallet(current_user: models.User = Depends(dependencies.get_current_user)):
    """Get user's public wallet info (no private keys)"""
    if not current_user.node_id:
        return {
            "connected": False,
            "node_id": None,
            "wallet_id": None,
            "message": "No wallet connected. Create or import a wallet."
        }
    
    return {
        "connected": True,
        "node_id": current_user.node_id,
        "wallet_id": current_user.wallet_id
    }

# Legacy endpoint for backwards compatibility
@app.get("/api/users/me/token")
async def read_user_token_legacy(current_user: models.User = Depends(dependencies.get_current_user)):
    """
    DEPRECATED: Use /users/me/wallet instead.
    This endpoint no longer returns the private token (security improvement).
    """
    return {
        "node_id": current_user.node_id,
        "wallet_id": current_user.wallet_id,
        "deprecated": True,
        "message": "Private keys are no longer stored in database. Use /users/me/wallet"
    }

@app.get("/api/users/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(dependencies.get_current_user)):
    return current_user

@app.get("/api/stats")
async def get_global_stats():
    """Proxy request to the central tracker to get network stats."""
    try:
        tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
        response = requests.get(f"{tracker_url}/stats", timeout=2)
        if response.status_code == 200:
            return response.json()
        else:
            return {"active_nodes": 0, "model_size": "142B", "total_tps": 0, "avg_latency": "N/A"}
    except Exception as e:
        return {"active_nodes": 0, "model_size": "142B", "total_tps": 0, "avg_latency": "N/A"}


# =============================================================================
# NETWORK STATUS - Native NeuroShard Architecture
# =============================================================================

@app.get("/api/network/status")
async def get_network_status():
    """
    Get comprehensive network status for the native NeuroShard architecture.
    
    Returns:
    - Node counts and health
    - Quorum statistics (training groups)
    - Network training progress
    """
    try:
        tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
        response = requests.get(f"{tracker_url}/network/status", timeout=3)
        if response.status_code == 200:
            return response.json()
        else:
            # Fallback to basic stats
            basic_response = requests.get(f"{tracker_url}/stats", timeout=2)
            if basic_response.status_code == 200:
                basic = basic_response.json()
                return {
                    "status": "healthy" if basic.get("active_nodes", 0) > 0 else "no_nodes",
                    "nodes": {
                        "active": basic.get("active_nodes", 0),
                        "total_tps": basic.get("total_tps", 0)
                    },
                    "quorums": {
                        "total": 0,
                        "active": 0
                    },
                    "network": {
                        "architecture": "NeuroShard",
                        "training_mode": "quorum-based"
                    }
                }
            return {"status": "unavailable"}
    except Exception as e:
        logger.error(f"Network status error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/network/quorums")
async def get_network_quorums(
    speed_tier: Optional[str] = None,
    lifecycle: Optional[str] = None,
    limit: int = 50
):
    """
    Get list of active quorums in the network.
    
    Quorums are speed-matched training groups that work together
    as a complete pipeline for distributed training.
    
    Args:
        speed_tier: Filter by speed tier (tier1-tier5)
        lifecycle: Filter by lifecycle state (forming, active)
        limit: Max number of quorums to return
    """
    try:
        tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
        params = {"limit": limit}
        if speed_tier:
            params["speed_tier"] = speed_tier
        if lifecycle:
            params["lifecycle"] = lifecycle
        
        response = requests.get(f"{tracker_url}/quorums", params=params, timeout=3)
        if response.status_code == 200:
            return response.json()
        return {"quorums": [], "total": 0}
    except Exception as e:
        logger.error(f"Quorums fetch error: {e}")
        return {"quorums": [], "total": 0, "error": str(e)}


@app.get("/api/network/quorums/{quorum_id}")
async def get_quorum_detail(quorum_id: str):
    """Get details of a specific quorum."""
    try:
        tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
        response = requests.get(f"{tracker_url}/quorums/{quorum_id}", timeout=3)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise HTTPException(status_code=404, detail="Quorum not found")
        return {"error": "Failed to fetch quorum"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quorum detail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/peers")
async def get_admin_peers(
    current_admin: models.User = Depends(dependencies.get_current_admin_user)
):
    """
    Get network peers from tracker - Admin Only.
    Returns actual peer data including last_seen timestamps for status display.
    """
    try:
        tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
        response = requests.get(f"{tracker_url}/peers", timeout=5)
        if response.status_code == 200:
            peers = response.json()
            # Return the actual peer data from tracker
            return [
                {
                    "url": p.get("url", ""),
                    "shard_range": p.get("shard_range", "unknown"),
                    "last_seen": p.get("last_seen", 0),
                    "tps": p.get("tps", 0),
                    "latency": p.get("latency", 0),
                    "is_entry": p.get("shard_range", "").startswith("dynamic-") and "0" in p.get("shard_range", "").split("-")[0] if p.get("shard_range") else False,
                    "is_exit": False,
                }
                for p in peers
            ]
        else:
            return []
    except Exception as e:
        logger.error(f"Failed to fetch peers from tracker: {e}")
        return []


# =============================================================================
# ADMIN USER MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/admin/users")
async def get_admin_users(
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """
    Get all registered users - Admin Only.
    Returns user list with registration info and rate limit tier.
    """
    users = db.query(models.User).order_by(models.User.created_at.desc()).all()
    
    return [
        {
            "id": user.id,
            "email": user.email,
            "is_active": user.is_active,
            "is_admin": user.is_admin,
            "has_wallet": user.node_id is not None,
            "wallet_id": user.wallet_id if user.node_id else None,
            "node_id": user.node_id if user.node_id else None,
            # Rate limiting
            "rate_limit_tier": user.rate_limit_tier or "standard",
            "is_rate_limited": user.is_rate_limited or False,
            # Chat stats
            "chat_count": user.chat_count or 0,
            "total_tokens_used": user.total_tokens_used or 0,
            # Timestamps
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "last_chat_at": user.last_chat_at.isoformat() if user.last_chat_at else None,
        }
        for user in users
    ]


@app.get("/api/admin/users/{user_id}")
async def get_admin_user_detail(
    user_id: int,
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Get detailed info about a specific user - Admin Only."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "email": user.email,
        "is_active": user.is_active,
        "is_admin": user.is_admin,
        "node_id": user.node_id,
        "wallet_id": user.wallet_id,
        # Rate limiting
        "rate_limit_tier": user.rate_limit_tier or "standard",
        "is_rate_limited": user.is_rate_limited or False,
        "rate_limit_until": user.rate_limit_until.isoformat() if user.rate_limit_until else None,
        # Chat stats
        "chat_count": user.chat_count or 0,
        "total_tokens_used": user.total_tokens_used or 0,
        "total_neuro_spent_chat": round(user.total_neuro_spent_chat or 0, 6),
        # Timestamps
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "last_chat_at": user.last_chat_at.isoformat() if user.last_chat_at else None,
        # Available tiers for dropdown
        "available_tiers": VALID_USER_TIERS,
    }


@app.patch("/api/admin/users/{user_id}/toggle-admin")
async def toggle_user_admin(
    user_id: int,
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Toggle admin status for a user - Admin Only."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent removing own admin status
    if user.id == current_admin.id:
        raise HTTPException(status_code=400, detail="Cannot modify your own admin status")
    
    user.is_admin = not user.is_admin
    db.commit()
    
    return {
        "id": user.id,
        "email": user.email,
        "is_admin": user.is_admin,
        "message": f"User {'promoted to' if user.is_admin else 'demoted from'} admin"
    }


@app.patch("/api/admin/users/{user_id}/toggle-active")
async def toggle_user_active(
    user_id: int,
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Toggle active status for a user - Admin Only."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent deactivating own account
    if user.id == current_admin.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
    
    user.is_active = not user.is_active
    db.commit()
    
    return {
        "id": user.id,
        "email": user.email,
        "is_active": user.is_active,
        "message": f"User {'activated' if user.is_active else 'deactivated'}"
    }


@app.delete("/api/admin/users/{user_id}")
async def delete_user(
    user_id: int,
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """
    Permanently delete a user - Admin Only.
    This also deletes associated refresh tokens.
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent deleting own account
    if user.id == current_admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    # Store email for response
    deleted_email = user.email
    
    # Reset waitlist entry status if exists so they can sign up again
    if user.waitlist_id:
        waitlist_entry = db.query(models.WaitlistEntry).filter(
            models.WaitlistEntry.id == user.waitlist_id
        ).first()
        
        if waitlist_entry:
            waitlist_entry.status = "approved"
            waitlist_entry.converted_at = None
            # We keep the entry so they don't lose their referral code/stats
    
    # Delete associated refresh tokens first
    db.query(models.RefreshToken).filter(models.RefreshToken.user_id == user_id).delete()
    
    # Delete the user
    db.delete(user)
    db.commit()
    
    return {
        "success": True,
        "message": f"User {deleted_email} has been permanently deleted"
    }


@app.get("/api/admin/stats")
async def get_admin_stats(
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """Get admin dashboard statistics - Admin Only."""
    total_users = db.query(models.User).count()
    active_users = db.query(models.User).filter(models.User.is_active == True).count()
    admin_users = db.query(models.User).filter(models.User.is_admin == True).count()
    users_with_wallets = db.query(models.User).filter(models.User.node_id != None).count()
    
    return {
        "total_users": total_users,
        "users_with_wallets": users_with_wallets,
        "active_users": active_users,
        "admin_users": admin_users
    }


# =============================================================================
# CHAT ANALYTICS ENDPOINTS
# =============================================================================

@app.get("/api/admin/chat-stats")
async def get_admin_chat_stats(
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db),
    hours: int = 24,
):
    """
    Get comprehensive chat analytics for admin dashboard.
    
    Returns:
    - Total interactions
    - Success/failure rates
    - Average response times
    - Token usage
    - NEURO spent/burned
    - Top users
    - Rate limit events
    - Hourly breakdown
    """
    from sqlalchemy import func, and_
    
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    # Overall stats
    total_interactions = db.query(models.ChatInteraction).filter(
        models.ChatInteraction.created_at >= cutoff
    ).count()
    
    successful = db.query(models.ChatInteraction).filter(
        and_(
            models.ChatInteraction.created_at >= cutoff,
            models.ChatInteraction.success == True
        )
    ).count()
    
    failed = total_interactions - successful
    
    # Aggregate metrics
    metrics = db.query(
        func.sum(models.ChatInteraction.tokens_used).label('total_tokens'),
        func.sum(models.ChatInteraction.neuro_cost).label('total_neuro_spent'),
        func.sum(models.ChatInteraction.fee_burned).label('total_fee_burned'),
        func.avg(models.ChatInteraction.response_time_ms).label('avg_response_time'),
        func.avg(models.ChatInteraction.node_response_time_ms).label('avg_node_time'),
        func.count(func.distinct(models.ChatInteraction.user_id)).label('unique_users'),
    ).filter(
        models.ChatInteraction.created_at >= cutoff
    ).first()
    
    # Top users by chat count
    top_users = db.query(
        models.User.email,
        models.User.id,
        models.User.chat_count,
        models.User.total_tokens_used,
        models.User.total_neuro_spent_chat,
    ).order_by(models.User.chat_count.desc()).limit(10).all()
    
    # Rate limit events
    rate_limit_count = db.query(models.RateLimitEvent).filter(
        models.RateLimitEvent.created_at >= cutoff
    ).count()
    
    # Rate limit by type
    rate_limit_by_type = db.query(
        models.RateLimitEvent.limit_type,
        func.count(models.RateLimitEvent.id).label('count')
    ).filter(
        models.RateLimitEvent.created_at >= cutoff
    ).group_by(models.RateLimitEvent.limit_type).all()
    
    # Error breakdown
    error_breakdown = db.query(
        models.ChatInteraction.error_code,
        func.count(models.ChatInteraction.id).label('count')
    ).filter(
        and_(
            models.ChatInteraction.created_at >= cutoff,
            models.ChatInteraction.success == False,
            models.ChatInteraction.error_code != None
        )
    ).group_by(models.ChatInteraction.error_code).all()
    
    # Hourly breakdown (last 24 hours)
    hourly_stats = []
    for i in range(min(hours, 24)):
        hour_start = datetime.utcnow() - timedelta(hours=i+1)
        hour_end = datetime.utcnow() - timedelta(hours=i)
        
        hour_count = db.query(models.ChatInteraction).filter(
            and_(
                models.ChatInteraction.created_at >= hour_start,
                models.ChatInteraction.created_at < hour_end
            )
        ).count()
        
        hourly_stats.append({
            "hour": hour_start.strftime("%Y-%m-%d %H:00"),
            "count": hour_count
        })
    
    # All-time totals
    all_time = db.query(
        func.sum(models.User.chat_count).label('total_chats'),
        func.sum(models.User.total_tokens_used).label('total_tokens'),
        func.sum(models.User.total_neuro_spent_chat).label('total_neuro'),
    ).first()
    
    return {
        "period_hours": hours,
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_interactions": total_interactions,
            "successful": successful,
            "failed": failed,
            "success_rate": round(successful / max(1, total_interactions) * 100, 2),
            "unique_users": metrics.unique_users or 0,
        },
        "performance": {
            "avg_response_time_ms": round(metrics.avg_response_time or 0, 2),
            "avg_node_time_ms": round(metrics.avg_node_time or 0, 2),
        },
        "usage": {
            "total_tokens": metrics.total_tokens or 0,
            "total_neuro_spent": round(metrics.total_neuro_spent or 0, 6),
            "total_fee_burned": round(metrics.total_fee_burned or 0, 6),
        },
        "rate_limiting": {
            "total_events": rate_limit_count,
            "by_type": {r[0]: r[1] for r in rate_limit_by_type},
        },
        "errors": {
            "total": failed,
            "by_code": {str(e[0]): e[1] for e in error_breakdown},
        },
        "top_users": [
            {
                "email": u.email,
                "user_id": u.id,
                "chat_count": u.chat_count or 0,
                "total_tokens": u.total_tokens_used or 0,
                "neuro_spent": round(u.total_neuro_spent_chat or 0, 6),
            }
            for u in top_users
        ],
        "hourly_breakdown": hourly_stats[::-1],  # Oldest first
        "all_time": {
            "total_chats": all_time.total_chats or 0,
            "total_tokens": all_time.total_tokens or 0,
            "total_neuro_spent": round(all_time.total_neuro or 0, 6),
        }
    }


@app.get("/api/admin/rate-limit-events")
async def get_rate_limit_events(
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db),
    hours: int = 24,
    limit: int = 100,
):
    """
    Get recent rate limit events for monitoring potential abuse.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    events = db.query(models.RateLimitEvent).filter(
        models.RateLimitEvent.created_at >= cutoff
    ).order_by(models.RateLimitEvent.created_at.desc()).limit(limit).all()
    
    return {
        "period_hours": hours,
        "total_events": len(events),
        "events": [
            {
                "id": e.id,
                "user_id": e.user_id,
                "client_ip": e.client_ip,
                "endpoint": e.endpoint,
                "limit_type": e.limit_type,
                "limit_value": e.limit_value,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ]
    }


@app.patch("/api/admin/users/{user_id}/rate-limit")
async def set_user_rate_limit(
    user_id: int,
    tier: str = "standard",
    block_minutes: int = 0,
    current_admin: models.User = Depends(dependencies.get_current_admin_user),
    db: Session = Depends(database.get_db)
):
    """
    Set rate limit tier or temporarily block a user - Admin Only.
    
    Tiers:
    - standard: 10 requests/minute, 100/hour (default for new users)
    - premium: 30 requests/minute, 500/hour (upgraded users)
    - unlimited: No rate limits (for trusted users)
    
    block_minutes: If > 0, temporarily block user from chat for this many minutes
    """
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if tier not in VALID_USER_TIERS:
        raise HTTPException(status_code=400, detail=f"Invalid tier. Must be one of: {VALID_USER_TIERS}")
    
    user.rate_limit_tier = tier
    
    if block_minutes > 0:
        user.is_rate_limited = True
        user.rate_limit_until = datetime.utcnow() + timedelta(minutes=block_minutes)
    else:
        user.is_rate_limited = False
        user.rate_limit_until = None
    
    db.commit()
    
    return {
        "user_id": user_id,
        "email": user.email,
        "rate_limit_tier": user.rate_limit_tier,
        "is_rate_limited": user.is_rate_limited,
        "rate_limit_until": user.rate_limit_until.isoformat() if user.rate_limit_until else None,
    }


@app.get("/api/users/me/chat-stats")
async def get_my_chat_stats(
    request: Request,
    current_user: models.User = Depends(dependencies.get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Get current user's chat statistics and rate limit status.
    """
    from sqlalchemy import func, and_
    
    # Attach user to request for rate limit status
    request.state.user = current_user
    
    # Recent interactions (last 24 hours)
    cutoff = datetime.utcnow() - timedelta(hours=24)
    
    recent_stats = db.query(
        func.count(models.ChatInteraction.id).label('count'),
        func.sum(models.ChatInteraction.tokens_used).label('tokens'),
        func.sum(models.ChatInteraction.neuro_cost).label('neuro_spent'),
        func.avg(models.ChatInteraction.response_time_ms).label('avg_time'),
    ).filter(
        and_(
            models.ChatInteraction.user_id == current_user.id,
            models.ChatInteraction.created_at >= cutoff,
        )
    ).first()
    
    # Get rate limit status
    rate_status = get_rate_limit_status(request)
    
    # Recent interactions list
    recent_interactions = db.query(models.ChatInteraction).filter(
        models.ChatInteraction.user_id == current_user.id
    ).order_by(models.ChatInteraction.created_at.desc()).limit(10).all()
    
    return {
        "user_id": current_user.id,
        "all_time": {
            "chat_count": current_user.chat_count or 0,
            "total_tokens": current_user.total_tokens_used or 0,
            "neuro_spent": round(current_user.total_neuro_spent_chat or 0, 6),
            "last_chat": current_user.last_chat_at.isoformat() if current_user.last_chat_at else None,
        },
        "last_24h": {
            "chat_count": recent_stats.count or 0,
            "tokens_used": recent_stats.tokens or 0,
            "neuro_spent": round(recent_stats.neuro_spent or 0, 6),
            "avg_response_time_ms": round(recent_stats.avg_time or 0, 2),
        },
        "rate_limit": rate_status,
        "recent_interactions": [
            {
                "id": i.id,
                "created_at": i.created_at.isoformat() if i.created_at else None,
                "prompt_length": i.prompt_length,
                "response_length": i.response_length,
                "tokens_used": i.tokens_used,
                "neuro_cost": round(i.neuro_cost or 0, 6),
                "response_time_ms": i.response_time_ms,
                "success": i.success,
                "error_code": i.error_code,
            }
            for i in recent_interactions
        ]
    }


@app.get("/api/users/me/rate-limit-status")
async def get_my_rate_limit_status(
    request: Request,
    current_user: models.User = Depends(dependencies.get_current_user)
):
    """
    Get current rate limit status for the authenticated user.
    Useful for showing users how many requests they have remaining.
    """
    request.state.user = current_user
    status = get_rate_limit_status(request)
    
    # Add user-specific info
    status["is_blocked"] = current_user.is_rate_limited
    if current_user.rate_limit_until:
        status["blocked_until"] = current_user.rate_limit_until.isoformat()
        remaining_seconds = (current_user.rate_limit_until - datetime.utcnow()).total_seconds()
        status["blocked_remaining_seconds"] = max(0, int(remaining_seconds))
    
    return status


@app.get("/api/users/me/is_admin")
async def check_is_admin(current_user: models.User = Depends(dependencies.get_current_user)):
    """Check if current user is an admin."""
    return {"is_admin": current_user.is_admin}


@app.get("/api/node/neuro")
async def get_node_neuro(node_id: str = None, token: str = None):
    """
    Get NEURO token balance and stats from the distributed ledger.
    
    Can accept either:
    - node_id: Public wallet address (preferred)
    - token: DEPRECATED - Private token (for backwards compatibility)
    
    Returns full account info including:
    - balance: Current spendable balance
    - total_earned: Lifetime earnings
    - total_spent: Lifetime spending
    - stake: Currently staked amount
    - stake_multiplier: Reward multiplier from staking
    """
    import sqlite3
    import time as time_module
    
    # Accept either node_id or token (derive node_id from token if needed)
    if not node_id and not token:
        raise HTTPException(status_code=400, detail="Either node_id or token required")
    
    if token and not node_id:
        # Legacy: Derive ECDSA node_id from token
        node_id = derive_ecdsa_node_id(token)
    
    # Try to query ledger database (if available on server)
    # Check LEDGER_DB_PATH first, then construct from LEDGER_DATA_DIR
    ledger_db_path = os.getenv("LEDGER_DB_PATH")
    if not ledger_db_path:
        ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
        # Check for observer_ledger.db first (shared from observer container)
        observer_ledger = os.path.join(ledger_data_dir, "observer_ledger.db")
        if os.path.exists(observer_ledger):
            ledger_db_path = observer_ledger
        else:
            ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
    
    # Account info
    balance = 0.0
    total_earned = 0.0
    total_spent = 0.0
    stake = 0.0
    stake_multiplier = 1.0
    stake_locked_until = 0.0
    proof_count = 0
    source = "ledger"
    
    # Global stats
    total_burned = 0.0
    circulating_supply = 0.0

    if os.path.exists(ledger_db_path):
        try:
            conn = sqlite3.connect(ledger_db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # Check which table format exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='balances'")
            use_new_format = cursor.fetchone() is not None
            
            if use_new_format:
                # New NEUROLedger format
                cursor.execute("""
                    SELECT balance, total_earned, total_spent, proof_count 
                    FROM balances WHERE node_id = ?
                """, (node_id,))
                row = cursor.fetchone()
                if row:
                    balance, total_earned, total_spent, proof_count = row
                    total_earned = total_earned or 0.0
                    total_spent = total_spent or 0.0
                    proof_count = proof_count or 0
                
                # Get stake info
                cursor.execute("""
                    SELECT amount, locked_until FROM stakes WHERE node_id = ?
                """, (node_id,))
                stake_row = cursor.fetchone()
                if stake_row:
                    stake_amount = stake_row[0]
                    stake_locked_until = stake_row[1] or 0.0
                    # Only set stake if it's actually a positive value
                    if stake_amount is not None and stake_amount > 0:
                        stake = float(stake_amount)
                        # Calculate multiplier with DIMINISHING RETURNS (from economics module)
                        # Multiplier applies even after unlock (until unstaked)
                        stake_multiplier = calculate_stake_multiplier(stake)
                    else:
                        # Ensure stake is 0 and multiplier is 1.0 if no valid stake
                        stake = 0.0
                        stake_multiplier = 1.0
                        stake_locked_until = 0.0
                
                # Get global burn stats
                cursor.execute("""
                    SELECT total_minted, total_burned FROM global_stats WHERE id = 1
                """)
                stats_row = cursor.fetchone()
                if stats_row:
                    total_minted = stats_row[0] or 0.0
                    total_burned = stats_row[1] or 0.0
                    circulating_supply = total_minted - total_burned
                
                source = "neuro_ledger"
            else:
                # Legacy format
                cursor.execute("SELECT balance FROM credits WHERE node_id = ?", (node_id,))
                row = cursor.fetchone()
                if row:
                    balance = row[0]
                source = "legacy_ledger"
                
            conn.close()
        except Exception as e:
             source = f"error_db: {e}"
    else:
        source = "unavailable"
        
    # NOTE: Legacy tracker stake lookup removed
    # Stake info now comes from ledger (NEUROLedger.stakes table)
    # Tracker no longer stores private node tokens for security reasons

    # Ensure stake_multiplier is 1.0 if stake is 0 or None
    if stake is None or stake <= 0:
        stake = 0.0
        stake_multiplier = 1.0
        stake_locked_until = 0.0
    
    return {
        "neuro_balance": round(balance, 6),
        "total_earned": round(total_earned, 6),
        "total_spent": round(total_spent, 6),
        "staked_balance": round(stake, 2),
        "stake_multiplier": round(stake_multiplier, 2),
        "stake_locked_until": stake_locked_until if stake_locked_until > 0 else None,
        "proof_count": proof_count,
        "node_id": node_id,
        "source": source,
        # Global network stats
        "network": {
            "total_burned": round(total_burned, 6),
            "circulating_supply": round(circulating_supply, 6),
            "burn_rate": "5%"
        }
    }

@app.get("/api/training/global")
async def get_global_training_status():
    """
    Get global LLM training status from GOSSIP DATA (ledger).
    
    DECENTRALIZED: We don't query nodes directly (they may be behind NAT).
    Instead, we read from the observer's ledger which receives PoNW proofs via gossip.
    
    This shows whether the distributed training is actually working:
    - Is the model improving?
    - Are nodes converging to the same weights?
    - What's the network-wide loss?
    """
    import sqlite3
    import time
    
    # Default response structure
    training_status = {
        "is_training": False,
        "training_verified": False,
        "is_converging": True,
        "global_loss": 0.0,
        "loss_trend": "unknown",
        "hash_agreement_rate": 1.0,
        "total_nodes_training": 0,
        "total_training_steps": 0,
        "total_tokens_trained": 0,
        "data_shards_covered": 0,
        "sync_success_rate": 0.0,
        "diloco": {
            "enabled": True,
            "inner_steps_config": 500,
            "outer_steps_completed": 0,
        },
        "nodes": [],
        "source": "gossip"
    }
    
    try:
        # Read training data from observer's ledger (populated via gossip)
        # Check LEDGER_DB_PATH first, then LEDGER_DATA_DIR
        ledger_db_path = os.getenv("LEDGER_DB_PATH")
        if not ledger_db_path:
            ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
            # Check for observer_ledger.db first (shared from observer container)
            observer_ledger = os.path.join(ledger_data_dir, "observer_ledger.db")
            if os.path.exists(observer_ledger):
                ledger_db_path = observer_ledger
            else:
                ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
        
        if not os.path.exists(ledger_db_path):
            training_status["source"] = f"ledger_not_found:{ledger_db_path}"
            return training_status
        
        conn = sqlite3.connect(ledger_db_path)
        cursor = conn.cursor()
        
        # Check if proof_history table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='proof_history'")
        if not cursor.fetchone():
            conn.close()
            training_status["source"] = "no_proof_history_table"
            return training_status
        
        # Get training proofs from last 5 minutes (recent activity)
        recent_cutoff = time.time() - 300  # 5 minutes
        
        # Aggregate training stats by node (table is proof_history, not ponw_proofs)
        cursor.execute("""
            SELECT 
                node_id, 
                SUM(training_batches) as total_batches,
                SUM(tokens_processed) as total_tokens,
                COUNT(*) as proof_count,
                MAX(timestamp) as last_seen
            FROM proof_history 
            WHERE timestamp > ? AND proof_type = 'training'
            GROUP BY node_id
            ORDER BY total_batches DESC
        """, (recent_cutoff,))
        
        training_nodes = []
        total_batches_all = 0
        total_tokens_all = 0
        loss_values = []
        
        for row in cursor.fetchall():
            node_id, total_batches, total_tokens, proof_count, last_seen = row
            batches = int(total_batches or 0)
            tokens = int(total_tokens or 0)
            
            # Get the most recent loss for this node (separate query)
            # Also check if this node is a validator (has_lm_head=1) for proper cross-entropy loss
            cursor.execute("""
                SELECT current_loss, has_lm_head FROM proof_history 
                WHERE node_id = ? AND proof_type = 'training' AND current_loss IS NOT NULL
                ORDER BY timestamp DESC LIMIT 1
            """, (node_id,))
            loss_row = cursor.fetchone()
            latest_loss = loss_row[0] if loss_row else None
            is_validator = bool(loss_row[1]) if loss_row and len(loss_row) > 1 else False
            
            training_nodes.append({
                "node_id": (node_id or "unknown")[:12],
                "training_rounds": batches,
                "proofs_submitted": proof_count,
                "last_active": last_seen,
                "current_loss": latest_loss,
                "is_validator": is_validator,
            })
            total_batches_all += batches
            total_tokens_all += tokens
            # Include all valid loss values (validators have true cross-entropy, workers have proxy loss)
            if latest_loss is not None and latest_loss > 0:
                loss_values.append(latest_loss)
        
        # Get total training steps from all time
        cursor.execute("""
            SELECT COUNT(DISTINCT node_id), SUM(training_batches), SUM(tokens_processed)
            FROM proof_history 
            WHERE proof_type = 'training'
        """)
        row = cursor.fetchone()
        all_time_nodes = row[0] or 0
        all_time_batches = int(row[1] or 0)
        all_time_tokens = int(row[2] or 0)
        
        # Get unique proofs from recent activity
        cursor.execute("""
            SELECT COUNT(DISTINCT signature) 
            FROM proof_history 
            WHERE timestamp > ?
        """, (recent_cutoff,))
        unique_proofs = cursor.fetchone()[0] or 0
        
        # Get total proofs count from global_stats (actual schema)
        max_epoch = 0
        try:
            cursor.execute("SELECT total_proofs FROM global_stats WHERE id = 1")
            row = cursor.fetchone()
            if row:
                # Estimate outer steps: total_proofs / expected_proofs_per_sync
                total_proofs = int(row[0] or 0)
                # Each outer step = ~500 inner steps = ~8 proofs (assuming 60s proof interval)
                max_epoch = total_proofs // 8 if total_proofs > 0 else 0
        except Exception as e:
            pass  # Table might not exist or have different schema
        
        # Count unique data shards (based on unique node_ids that have done training)
        try:
            cursor.execute("SELECT COUNT(DISTINCT node_id) FROM proof_history WHERE proof_type = 'training'")
            data_shards = cursor.fetchone()[0] or 0
        except:
            data_shards = 0
        
        # =====================================================================
        # DECENTRALIZED LOSS TREND: Compare recent PoNW proofs
        # This is the blockchain-style verification - we look at the CONSENSUS
        # of loss values from signed proofs over time, not a central authority.
        #
        # IMPORTANT: Only compare proofs within a RECENT window (last 6 hours).
        # Comparing across all-time history is misleading because:
        # - Old proofs may be from different model architectures/sizes
        # - New nodes joining with random weights temporarily spike loss
        # - The "early loss" should be from the current session, not weeks ago
        # =====================================================================
        loss_improvement = None
        early_loss_avg = None
        recent_loss_avg = None
        
        try:
            # Use a sliding window of the last 6 hours for trend analysis.
            # IMPORTANT: Only compare loss for nodes that were active in BOTH
            # the early and recent periods. A newly-joined node with high initial
            # loss should NOT drag down the "improvement" metric — each node is
            # individually improving, and the network average is misleading when
            # nodes join at different times.
            window_start = time.time() - (6 * 3600)  # 6 hours ago
            
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
                FROM proof_history 
                WHERE proof_type = 'training' 
                    AND current_loss IS NOT NULL 
                    AND current_loss > 0 
                    AND current_loss < 20
                    AND timestamp > ?
            """, (window_start,))
            row = cursor.fetchone()
            first_timestamp = row[0] if row else None
            proof_count_in_window = row[2] if row else 0
            
            if first_timestamp and proof_count_in_window >= 6:
                time_range = time.time() - first_timestamp
                early_cutoff = first_timestamp + (time_range * 0.3)
                recent_cutoff = time.time() - (time_range * 0.3)
                
                # Find nodes that were active in BOTH the early and recent periods
                # This prevents a new node's high initial loss from polluting the trend
                cursor.execute("""
                    SELECT node_id FROM proof_history
                    WHERE proof_type = 'training'
                        AND current_loss IS NOT NULL AND current_loss > 0 AND current_loss < 20
                        AND timestamp > ? AND timestamp < ?
                    GROUP BY node_id HAVING COUNT(*) >= 2
                    INTERSECT
                    SELECT node_id FROM proof_history
                    WHERE proof_type = 'training'
                        AND current_loss IS NOT NULL AND current_loss > 0 AND current_loss < 20
                        AND timestamp > ?
                    GROUP BY node_id HAVING COUNT(*) >= 2
                """, (window_start, early_cutoff, recent_cutoff))
                stable_nodes = [r[0] for r in cursor.fetchall()]
                
                if stable_nodes:
                    placeholders = ','.join('?' * len(stable_nodes))
                    
                    # Early loss — only from nodes that were in both periods
                    cursor.execute(f"""
                        SELECT AVG(current_loss), COUNT(*)
                        FROM proof_history 
                        WHERE proof_type = 'training' 
                            AND current_loss IS NOT NULL 
                            AND current_loss > 0 
                            AND current_loss < 20
                            AND timestamp > ?
                            AND timestamp < ?
                            AND node_id IN ({placeholders})
                    """, [window_start, early_cutoff] + stable_nodes)
                    early_row = cursor.fetchone()
                    early_loss_avg = early_row[0] if early_row and early_row[0] else None
                    early_count = early_row[1] if early_row else 0
                    
                    # Recent loss — only from nodes that were in both periods
                    cursor.execute(f"""
                        SELECT AVG(current_loss), COUNT(*)
                        FROM proof_history 
                        WHERE proof_type = 'training' 
                            AND current_loss IS NOT NULL 
                            AND current_loss > 0 
                            AND current_loss < 20
                            AND timestamp > ?
                            AND node_id IN ({placeholders})
                    """, [recent_cutoff] + stable_nodes)
                    recent_row = cursor.fetchone()
                    recent_loss_avg = recent_row[0] if recent_row and recent_row[0] else None
                    recent_count = recent_row[1] if recent_row else 0
                    
                    # Calculate improvement from PoNW proof consensus
                    if early_loss_avg and recent_loss_avg and early_count >= 3 and recent_count >= 3:
                        loss_improvement = ((early_loss_avg - recent_loss_avg) / early_loss_avg) * 100
        except Exception as e:
            logger.warning(f"Error calculating loss trend from proofs: {e}")
        
        conn.close()
        
        # Populate response
        training_status["total_nodes_training"] = len(training_nodes)
        training_status["is_training"] = len(training_nodes) > 0
        training_status["total_training_steps"] = all_time_batches
        training_status["total_tokens_trained"] = all_time_tokens
        training_status["data_shards_covered"] = data_shards
        training_status["nodes"] = training_nodes[:10]
        
        # Calculate global loss (average of all nodes with valid loss)
        if loss_values:
            training_status["global_loss"] = round(sum(loss_values) / len(loss_values), 4)
        
        # Determine trend from HISTORICAL PROOF COMPARISON (decentralized consensus)
        if loss_improvement is not None:
            training_status["loss_improvement_percent"] = round(loss_improvement, 2)
            training_status["early_loss"] = round(early_loss_avg, 4) if early_loss_avg else None
            training_status["recent_loss"] = round(recent_loss_avg, 4) if recent_loss_avg else None
            
            if loss_improvement > 10:
                training_status["loss_trend"] = "improving"
                training_status["training_verified"] = True
            elif loss_improvement > 2:
                training_status["loss_trend"] = "improving"
                training_status["training_verified"] = True
            elif loss_improvement > -2:
                training_status["loss_trend"] = "stable"
                # Stable after significant training = converged
                training_status["training_verified"] = all_time_batches > 1000
            elif loss_improvement > -10:
                training_status["loss_trend"] = "needs_attention"
                training_status["training_verified"] = False
            else:
                training_status["loss_trend"] = "degrading"
                training_status["training_verified"] = False
        elif len(training_nodes) > 0:
            # Training is happening but not enough historical data yet
            training_status["loss_trend"] = "initializing"
            training_status["training_verified"] = False
        
        # Converging = we have loss data AND trend is positive or stable
        training_status["is_converging"] = (
            len(loss_values) > 0 and 
            training_status.get("loss_trend") in ["improving", "stable", "initializing"]
        )
        
        # DiLoCo info
        training_status["diloco"]["outer_steps_completed"] = max_epoch
        steps_until_sync = 500 - (all_time_batches % 500) if all_time_batches > 0 else 500
        training_status["diloco"]["steps_until_sync"] = steps_until_sync
        
        # Calculate sync success rate based on recent proof activity
        if len(training_nodes) > 0:
            # Success rate = unique proofs in last 5 min / expected proofs (1 per node per minute)
            expected_proofs = len(training_nodes) * 5  # 5 minutes * 1 proof/minute/node
            training_status["sync_success_rate"] = min(1.0, unique_proofs / max(1, expected_proofs))
        
        training_status["source"] = "gossip_ledger"
        
    except Exception as e:
        training_status["source"] = f"error: {str(e)}"
    
    return training_status


@app.get("/api/training/history")
async def get_training_loss_history():
    """
    Get historical loss data to verify the model is ACTUALLY improving.
    
    This endpoint answers: "Is the distributed LLM training working?"
    
    Returns:
    - loss_history: List of (timestamp, step, loss) points
    - loss_trend: "improving", "stable", or "degrading"
    - improvement_percent: How much loss has decreased
    - training_verified: True if we can confirm model is learning
    
    A healthy training run should show:
    - Loss decreasing over time (not monotonically, but trending down)
    - improvement_percent > 0 (ideally 10%+ after 1000+ steps)
    """
    import sqlite3
    import time
    
    result = {
        "loss_history": [],
        "loss_trend": "unknown",
        "improvement_percent": 0.0,
        "training_verified": False,
        "analysis": {},
        "source": "unavailable"
    }
    
    try:
        ledger_db_path = os.getenv("LEDGER_DB_PATH")
        if not ledger_db_path:
            ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
            # Check for observer_ledger.db first (shared from observer container)
            observer_ledger = os.path.join(ledger_data_dir, "observer_ledger.db")
            if os.path.exists(observer_ledger):
                ledger_db_path = observer_ledger
            else:
                ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
        
        if not os.path.exists(ledger_db_path):
            result["source"] = f"ledger_not_found:{ledger_db_path}"
            return result
        
        conn = sqlite3.connect(ledger_db_path)
        cursor = conn.cursor()
        
        # Get loss values over time from proof_history
        # Group by 10-minute intervals to see trends
        cursor.execute("""
            SELECT 
                (timestamp / 600) * 600 as time_bucket,
                AVG(current_loss) as avg_loss,
                MIN(current_loss) as min_loss,
                MAX(current_loss) as max_loss,
                SUM(training_batches) as total_batches,
                COUNT(*) as proof_count
            FROM proof_history 
            WHERE proof_type = 'training' 
                AND current_loss IS NOT NULL 
                AND current_loss > 0
                AND current_loss < 100
            GROUP BY time_bucket
            ORDER BY time_bucket ASC
            LIMIT 144
        """)
        
        loss_history = []
        for row in cursor.fetchall():
            time_bucket, avg_loss, min_loss, max_loss, total_batches, proof_count = row
            loss_history.append({
                "timestamp": time_bucket,
                "avg_loss": round(avg_loss, 4) if avg_loss else None,
                "min_loss": round(min_loss, 4) if min_loss else None,
                "max_loss": round(max_loss, 4) if max_loss else None,
                "batches": int(total_batches or 0),
                "proofs": int(proof_count or 0)
            })
        
        result["loss_history"] = loss_history
        
        # Analyze the trend
        if len(loss_history) >= 3:
            losses = [h["avg_loss"] for h in loss_history if h["avg_loss"]]
            
            if len(losses) >= 3:
                # Compare first third to last third
                third = len(losses) // 3
                first_third = losses[:third] if third > 0 else losses[:1]
                last_third = losses[-third:] if third > 0 else losses[-1:]
                
                first_avg = sum(first_third) / len(first_third)
                last_avg = sum(last_third) / len(last_third)
                
                if first_avg > 0:
                    improvement = (first_avg - last_avg) / first_avg * 100
                    result["improvement_percent"] = round(improvement, 2)
                    
                    if improvement > 10:
                        result["loss_trend"] = "improving_strongly"
                        result["training_verified"] = True
                    elif improvement > 2:
                        result["loss_trend"] = "improving"
                        result["training_verified"] = True
                    elif improvement > -2:
                        result["loss_trend"] = "stable"
                        # Stable can be ok for converged model
                        result["training_verified"] = len(losses) > 10
                    elif improvement > -10:
                        result["loss_trend"] = "degrading_slightly"
                    else:
                        result["loss_trend"] = "degrading"
                
                # Detailed analysis
                result["analysis"] = {
                    "data_points": len(losses),
                    "first_avg_loss": round(first_avg, 4),
                    "last_avg_loss": round(last_avg, 4),
                    "min_loss_seen": round(min(losses), 4),
                    "max_loss_seen": round(max(losses), 4),
                    "loss_variance": round(sum((l - sum(losses)/len(losses))**2 for l in losses) / len(losses), 4),
                }
                
                # Get total steps and tokens
                cursor.execute("""
                    SELECT SUM(training_batches), SUM(tokens_processed)
                    FROM proof_history WHERE proof_type = 'training'
                """)
                row = cursor.fetchone()
                result["analysis"]["total_batches"] = int(row[0] or 0) if row else 0
                result["analysis"]["total_tokens"] = int(row[1] or 0) if row else 0
        
        conn.close()
        result["source"] = "gossip_ledger"
        
    except Exception as e:
        result["source"] = f"error: {str(e)}"
    
    return result


@app.get("/api/neuro/stats")
async def get_neuro_global_stats():
    """
    Get global NEURO token statistics.
    
    Returns:
    - total_minted: Total NEURO ever created through PoNW
    - total_burned: Total NEURO burned through fee mechanism
    - circulating_supply: total_minted - total_burned
    - burn_rate: Current burn rate (5%)
    - total_proofs: Number of PoNW proofs processed
    - total_transactions: Number of transfers
    """
    import sqlite3
    
    # Check LEDGER_DB_PATH first, then construct from LEDGER_DATA_DIR
    ledger_db_path = os.getenv("LEDGER_DB_PATH")
    if not ledger_db_path:
        ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
        ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
    
    stats = {
        "total_minted": 0.0,
        "total_burned": 0.0,
        "circulating_supply": 0.0,
        "total_proofs": 0,
        "total_transactions": 0,
        "burn_rate": "5%",
        "source": "unavailable"
    }
    
    if os.path.exists(ledger_db_path):
        try:
            conn = sqlite3.connect(ledger_db_path, check_same_thread=False)
            cursor = conn.cursor()
            
            # Check for new format
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='global_stats'")
            if cursor.fetchone():
                cursor.execute("""
                    SELECT total_minted, total_burned, total_transferred, total_proofs, total_transactions
                    FROM global_stats WHERE id = 1
                """)
                row = cursor.fetchone()
                if row:
                    total_minted = row[0] or 0.0
                    total_burned = row[1] or 0.0
                    stats["total_minted"] = round(total_minted, 6)
                    stats["total_burned"] = round(total_burned, 6)
                    stats["circulating_supply"] = round(total_minted - total_burned, 6)
                    stats["total_proofs"] = row[3] or 0
                    stats["total_transactions"] = row[4] or 0
                    stats["source"] = "neuro_ledger"
            else:
                # Legacy - estimate from credits table
                cursor.execute("SELECT SUM(balance) FROM credits")
                row = cursor.fetchone()
                if row and row[0]:
                    stats["circulating_supply"] = round(row[0], 6)
                    stats["total_minted"] = stats["circulating_supply"]  # No burn in legacy
                stats["source"] = "legacy_ledger"
            
            conn.close()
        except Exception as e:
            stats["source"] = f"error: {e}"
    
    return stats


@app.get("/api/users/me/node_status")
async def get_my_node_status(current_user: models.User = Depends(dependencies.get_current_user)):
    """Check if the current user's node is active."""
    if not current_user.node_id:
        return {"active": False, "detail": "No wallet connected"}
        
    # Check if user's node_id appears in active peers
    tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
    try:
        resp = requests.get(f"{tracker_url}/peers", timeout=2)
        if resp.status_code == 200:
            peers = resp.json()
            
            # Check each peer's node_token to see if it matches this user's node_id
            # node_id = derive_from(node_token), so we check if any peer's derived node_id matches
            from .wallet import WalletManager
            wallet_manager = WalletManager()
            
            for peer in peers:
                peer_token = peer.get("node_token")
                if peer_token:
                    try:
                        # Derive node_id from peer's token
                        peer_node_id = wallet_manager.token_to_node_id(peer_token)
                        if peer_node_id == current_user.node_id:
                            return {"active": True, "node_id": current_user.node_id}
                    except:
                        pass  # Skip peers with invalid tokens
            
            return {"active": False, "node_id": current_user.node_id, "detail": "Node not running"}
        else:
            return {"active": False, "detail": "Tracker error"}
    except Exception as e:
        logger.error(f"Node status check failed: {e}")
        return {"active": False, "detail": "Tracker unreachable"}


# =============================================================================
# PROTECTED WHITEPAPER ENDPOINT
# =============================================================================

from fastapi.responses import FileResponse

@app.get("/api/whitepaper/pdf")
async def get_whitepaper_pdf(current_user: models.User = Depends(dependencies.get_current_user)):
    """
    Serve the whitepaper PDF to authenticated users only.
    
    This protects our technical documentation from public access while
    allowing registered users to view and download it.
    """
    # Path to the whitepaper PDF (stored in protected location)
    whitepaper_path = os.path.join(os.path.dirname(__file__), "..", "protected", "whitepaper.pdf")
    
    # Fallback to public location if protected doesn't exist yet
    if not os.path.exists(whitepaper_path):
        whitepaper_path = os.path.join(os.path.dirname(__file__), "..", "public", "whitepaper.pdf")
    
    if not os.path.exists(whitepaper_path):
        raise HTTPException(status_code=404, detail="Whitepaper not found")
    
    return FileResponse(
        whitepaper_path,
        media_type="application/pdf",
        filename="NeuroShard_Whitepaper.pdf",
        headers={
            "Content-Disposition": "inline; filename=NeuroShard_Whitepaper.pdf"
        }
    )

@app.get("/api/whitepaper/info")
async def get_whitepaper_info(current_user: models.User = Depends(dependencies.get_current_user)):
    """Get metadata about the whitepaper."""
    return {
        "title": "NeuroShard: A Decentralized Architecture for Collective Intelligence",
        "version": "1.0",
        "date": "November 2025",
        "authors": "LZ",
        "sections": [
            "Introduction",
            "NeuroLLM Architecture",
            "Decentralized Training",
            "System Architecture",
            "Proof of Neural Work (PoNW)",
            "Robustness and Anti-Poisoning",
            "Governance and The NeuroDAO",
            "NEURO Token Economics",
            "Security Considerations",
            "Checkpoint System",
            "Implementation",
            "Vision and Roadmap"
        ],
        "access_level": "registered_users_only"
    }


# =============================================================================
# HEALTH CHECK & SYSTEM ENDPOINTS
# =============================================================================

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    Returns system status and component health.
    """
    from .rate_limiter import get_redis
    
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {}
    }
    
    # Check database
    try:
        db = next(database.get_db())
        db.execute("SELECT 1")
        health["components"]["database"] = {"status": "healthy"}
        db.close()
    except Exception as e:
        health["components"]["database"] = {"status": "unhealthy", "error": str(e)}
        health["status"] = "degraded"
    
    # Check Redis (for rate limiting)
    redis_client = get_redis()
    if redis_client:
        try:
            redis_client.ping()
            health["components"]["redis"] = {"status": "healthy"}
        except Exception as e:
            health["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
            # Redis being down is not critical - fallback to in-memory
    else:
        health["components"]["redis"] = {"status": "not_configured", "note": "Using in-memory rate limiting"}
    
    # Check tracker connectivity
    tracker_url = os.getenv("TRACKER_URL", "http://tracker:3000")
    try:
        resp = requests.get(f"{tracker_url}/stats", timeout=2)
        if resp.status_code == 200:
            health["components"]["tracker"] = {"status": "healthy"}
        else:
            health["components"]["tracker"] = {"status": "degraded", "code": resp.status_code}
    except Exception as e:
        health["components"]["tracker"] = {"status": "unreachable", "error": str(e)[:100]}
    
    return health


@app.get("/api/admin/migrations/status")
async def get_migration_status(
    current_admin: models.User = Depends(dependencies.get_current_admin_user)
):
    """
    Get current database migration status - Admin Only.
    """
    from .migrations import verify_schema
    
    status = verify_schema()
    return status


@app.post("/api/admin/migrations/run")
async def run_migrations(
    current_admin: models.User = Depends(dependencies.get_current_admin_user)
):
    """
    Manually run database migrations - Admin Only.
    """
    from .migrations import migrate, verify_schema
    
    before = verify_schema()
    
    try:
        migrate()
        after = verify_schema()
        
        return {
            "success": True,
            "before": before,
            "after": after,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "before": before,
        }


@app.post("/api/admin/cleanup")
async def cleanup_old_data(
    days: int = 30,
    current_admin: models.User = Depends(dependencies.get_current_admin_user)
):
    """
    Clean up old tracking data to prevent database bloat - Admin Only.
    
    Args:
        days: Delete data older than this many days (default 30)
    """
    from .migrations import cleanup_old_data as do_cleanup
    
    result = do_cleanup(days)
    return result


@app.get("/api/admin/rate-limiter/status")
async def get_rate_limiter_status(
    current_admin: models.User = Depends(dependencies.get_current_admin_user)
):
    """
    Get rate limiter configuration and status - Admin Only.
    """
    from .rate_limiter import get_redis, RATE_LIMITS, GLOBAL_RATE_LIMITS, ENDPOINT_LIMITS
    
    redis_client = get_redis()
    
    return {
        "backend": "redis" if redis_client else "memory",
        "redis_connected": redis_client is not None,
        "tiers": RATE_LIMITS,
        "global_limits": GLOBAL_RATE_LIMITS,
        "endpoint_limits": ENDPOINT_LIMITS,
    }


# =============================================================================
# EPOCH CHAIN EXPLORER API
# =============================================================================
# These endpoints expose the chained PoNW epoch system for transparency.
# Anyone can verify the epoch chain without running a full node.

@app.get("/api/epochs/latest")
async def get_latest_epoch():
    """
    Get the latest finalized epoch.
    
    Returns:
        The most recent finalized epoch with its chain linkage.
    """
    import sqlite3
    
    # Try to query ledger database
    ledger_db_path = os.getenv("LEDGER_DB_PATH")
    if not ledger_db_path:
        ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
        observer_ledger = os.path.join(ledger_data_dir, "observer_ledger.db")
        if os.path.exists(observer_ledger):
            ledger_db_path = observer_ledger
        else:
            ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
    
    if not os.path.exists(ledger_db_path):
        raise HTTPException(status_code=503, detail="Ledger database not available")
    
    try:
        with sqlite3.connect(ledger_db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            
            epoch = conn.execute("""
                SELECT * FROM epochs 
                WHERE finalized = 1 
                ORDER BY epoch_id DESC 
                LIMIT 1
            """).fetchone()
            
            if not epoch:
                return {
                    "epoch_id": None,
                    "message": "No finalized epochs yet"
                }
            
            return {
                "epoch_id": epoch["epoch_id"],
                "epoch_hash": epoch["epoch_hash"],
                "prev_epoch_hash": epoch["prev_epoch_hash"],
                "timestamp_start": epoch["timestamp_start"],
                "timestamp_end": epoch["timestamp_end"],
                "model_state_hash_start": epoch["model_state_hash_start"],
                "model_state_hash_end": epoch["model_state_hash_end"],
                "proofs_merkle_root": epoch["proofs_merkle_root"],
                "gradient_commitments_root": epoch["gradient_commitments_root"],
                "proof_count": epoch["proof_count"],
                "total_reward": epoch["total_reward"],
                "total_batches": epoch["total_batches"],
                "average_loss": epoch["average_loss"],
                "proposer_node_id": epoch["proposer_node_id"],
                "proposer_signature": epoch["proposer_signature"],
            }
    except Exception as e:
        logger.error(f"Error getting latest epoch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/epochs/{epoch_id}")
async def get_epoch(epoch_id: int):
    """
    Get a specific epoch by ID.
    
    Args:
        epoch_id: The epoch number to retrieve
        
    Returns:
        The epoch data with chain linkage verification.
    """
    import sqlite3
    
    # Try to query ledger database
    ledger_db_path = os.getenv("LEDGER_DB_PATH")
    if not ledger_db_path:
        ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
        observer_ledger = os.path.join(ledger_data_dir, "observer_ledger.db")
        if os.path.exists(observer_ledger):
            ledger_db_path = observer_ledger
        else:
            ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
    
    if not os.path.exists(ledger_db_path):
        raise HTTPException(status_code=503, detail="Ledger database not available")
    
    try:
        with sqlite3.connect(ledger_db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            
            epoch = conn.execute(
                "SELECT * FROM epochs WHERE epoch_id = ?",
                (epoch_id,)
            ).fetchone()
            
            if not epoch:
                raise HTTPException(status_code=404, detail=f"Epoch {epoch_id} not found")
            
            # Also get chain verification info
            prev_epoch = conn.execute(
                "SELECT epoch_hash, model_state_hash_end FROM epochs WHERE epoch_id = ?",
                (epoch_id - 1,)
            ).fetchone() if epoch_id > 0 else None
            
            chain_valid = True
            chain_message = "OK"
            
            if prev_epoch:
                if epoch["prev_epoch_hash"] != prev_epoch["epoch_hash"]:
                    chain_valid = False
                    chain_message = "prev_epoch_hash mismatch"
                elif epoch["model_state_hash_start"] != prev_epoch["model_state_hash_end"]:
                    chain_valid = False
                    chain_message = "model_state discontinuity"
            
            return {
                "epoch_id": epoch["epoch_id"],
                "epoch_hash": epoch["epoch_hash"],
                "prev_epoch_hash": epoch["prev_epoch_hash"],
                "timestamp_start": epoch["timestamp_start"],
                "timestamp_end": epoch["timestamp_end"],
                "model_state_hash_start": epoch["model_state_hash_start"],
                "model_state_hash_end": epoch["model_state_hash_end"],
                "proofs_merkle_root": epoch["proofs_merkle_root"],
                "gradient_commitments_root": epoch["gradient_commitments_root"],
                "proof_count": epoch["proof_count"],
                "total_reward": epoch["total_reward"],
                "total_batches": epoch["total_batches"],
                "average_loss": epoch["average_loss"],
                "proposer_node_id": epoch["proposer_node_id"],
                "proposer_signature": epoch["proposer_signature"],
                "finalized": bool(epoch["finalized"]),
                "chain_verification": {
                    "valid": chain_valid,
                    "message": chain_message,
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting epoch {epoch_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/epochs")
async def list_epochs(
    offset: int = 0,
    limit: int = 20,
    order: str = "desc"
):
    """
    List finalized epochs with pagination.
    
    Args:
        offset: Number of epochs to skip
        limit: Maximum epochs to return (max 100)
        order: Sort order ('asc' or 'desc')
        
    Returns:
        List of epochs with chain info.
    """
    import sqlite3
    
    limit = min(limit, 100)  # Cap at 100
    order_sql = "DESC" if order.lower() == "desc" else "ASC"
    
    # Try to query ledger database
    ledger_db_path = os.getenv("LEDGER_DB_PATH")
    if not ledger_db_path:
        ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
        observer_ledger = os.path.join(ledger_data_dir, "observer_ledger.db")
        if os.path.exists(observer_ledger):
            ledger_db_path = observer_ledger
        else:
            ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
    
    if not os.path.exists(ledger_db_path):
        raise HTTPException(status_code=503, detail="Ledger database not available")
    
    try:
        with sqlite3.connect(ledger_db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get total count
            total = conn.execute(
                "SELECT COUNT(*) FROM epochs WHERE finalized = 1"
            ).fetchone()[0]
            
            # Get epochs
            epochs = conn.execute(f"""
                SELECT epoch_id, epoch_hash, prev_epoch_hash, 
                       timestamp_start, timestamp_end,
                       proof_count, total_reward, total_batches,
                       proposer_node_id, finalized
                FROM epochs 
                WHERE finalized = 1 
                ORDER BY epoch_id {order_sql}
                LIMIT ? OFFSET ?
            """, (limit, offset)).fetchall()
            
            return {
                "total": total,
                "offset": offset,
                "limit": limit,
                "epochs": [
                    {
                        "epoch_id": e["epoch_id"],
                        "epoch_hash": e["epoch_hash"][:16] + "...",
                        "prev_epoch_hash": e["prev_epoch_hash"][:16] + "..." if e["prev_epoch_hash"] else None,
                        "timestamp_start": e["timestamp_start"],
                        "timestamp_end": e["timestamp_end"],
                        "proof_count": e["proof_count"],
                        "total_reward": e["total_reward"],
                        "total_batches": e["total_batches"],
                        "proposer_node_id": e["proposer_node_id"][:16] + "...",
                    }
                    for e in epochs
                ]
            }
    except Exception as e:
        logger.error(f"Error listing epochs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/epochs/chain/verify")
async def verify_epoch_chain(
    start_epoch: int = 0,
    end_epoch: Optional[int] = None
):
    """
    Verify the epoch chain integrity.
    
    This enables trustless verification of the PoNW chain:
    - Each epoch's prev_hash matches the previous epoch's hash
    - Model state shows progression (end of prev = start of current)
    - Epoch hashes are correctly computed
    
    Args:
        start_epoch: First epoch to verify
        end_epoch: Last epoch to verify (defaults to latest)
        
    Returns:
        Chain verification result with any errors found.
    """
    import sqlite3
    import hashlib
    
    # Try to query ledger database
    ledger_db_path = os.getenv("LEDGER_DB_PATH")
    if not ledger_db_path:
        ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
        observer_ledger = os.path.join(ledger_data_dir, "observer_ledger.db")
        if os.path.exists(observer_ledger):
            ledger_db_path = observer_ledger
        else:
            ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
    
    if not os.path.exists(ledger_db_path):
        raise HTTPException(status_code=503, detail="Ledger database not available")
    
    try:
        with sqlite3.connect(ledger_db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get end epoch if not specified
            if end_epoch is None:
                result = conn.execute(
                    "SELECT MAX(epoch_id) FROM epochs WHERE finalized = 1"
                ).fetchone()
                end_epoch = result[0] if result[0] else 0
            
            # Limit verification range
            if end_epoch - start_epoch > 1000:
                return {
                    "valid": False,
                    "error": "Range too large. Max 1000 epochs per request."
                }
            
            # Get all epochs in range
            epochs = conn.execute("""
                SELECT * FROM epochs 
                WHERE epoch_id >= ? AND epoch_id <= ?
                ORDER BY epoch_id ASC
            """, (start_epoch, end_epoch)).fetchall()
            
            if not epochs:
                return {
                    "valid": True,
                    "message": "No epochs in range",
                    "range": [start_epoch, end_epoch]
                }
            
            # Verify chain
            errors = []
            prev_epoch = None
            
            genesis_hash = "0x0000000000000000000000000000000000000000000000000000000000000000"
            
            for epoch in epochs:
                # Recompute epoch hash
                payload = (
                    f"{epoch['epoch_id']}:"
                    f"{epoch['prev_epoch_hash']}:"
                    f"{epoch['timestamp_start']}:{epoch['timestamp_end']}:"
                    f"{epoch['model_state_hash_start']}:{epoch['model_state_hash_end']}:"
                    f"{epoch['proofs_merkle_root']}:"
                    f"{epoch['proof_count']}:{epoch['total_reward']:.6f}:"
                    f"{epoch['total_batches']}:{epoch['average_loss']:.6f}:"
                    f"{epoch['gradient_commitments_root']}:"
                    f"{epoch['proposer_node_id']}"
                )
                computed_hash = hashlib.sha256(payload.encode()).hexdigest()
                
                if computed_hash != epoch['epoch_hash']:
                    errors.append({
                        "epoch_id": epoch['epoch_id'],
                        "error": "Hash mismatch",
                        "computed": computed_hash[:16] + "...",
                        "stored": epoch['epoch_hash'][:16] + "..."
                    })
                
                # Check chain linkage
                if prev_epoch:
                    if epoch['prev_epoch_hash'] != prev_epoch['epoch_hash']:
                        errors.append({
                            "epoch_id": epoch['epoch_id'],
                            "error": "prev_epoch_hash broken"
                        })
                    
                    if epoch['model_state_hash_start'] != prev_epoch['model_state_hash_end']:
                        errors.append({
                            "epoch_id": epoch['epoch_id'],
                            "error": "Model state discontinuity"
                        })
                    
                    if epoch['timestamp_start'] < prev_epoch['timestamp_end']:
                        errors.append({
                            "epoch_id": epoch['epoch_id'],
                            "error": "Timestamp overlap"
                        })
                
                prev_epoch = epoch
            
            return {
                "valid": len(errors) == 0,
                "range": [start_epoch, end_epoch],
                "epochs_verified": len(epochs),
                "errors": errors if errors else None,
                "message": "Chain integrity verified" if not errors else f"Found {len(errors)} errors"
            }
            
    except Exception as e:
        logger.error(f"Error verifying epoch chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/epochs/{epoch_id}/proofs")
async def get_epoch_proofs(
    epoch_id: int,
    offset: int = 0,
    limit: int = 50
):
    """
    Get proofs included in a specific epoch.
    
    Args:
        epoch_id: The epoch to query
        offset: Number of proofs to skip
        limit: Maximum proofs to return
        
    Returns:
        List of proofs in the epoch.
    """
    import sqlite3
    
    limit = min(limit, 100)
    
    # Calculate time range for the epoch
    epoch_start = epoch_id * 60  # 60-second epochs
    epoch_end = epoch_start + 60
    
    # Try to query ledger database
    ledger_db_path = os.getenv("LEDGER_DB_PATH")
    if not ledger_db_path:
        ledger_data_dir = os.getenv("LEDGER_DATA_DIR", "/data")
        observer_ledger = os.path.join(ledger_data_dir, "observer_ledger.db")
        if os.path.exists(observer_ledger):
            ledger_db_path = observer_ledger
        else:
            ledger_db_path = os.path.join(ledger_data_dir, "node_ledger.db")
    
    if not os.path.exists(ledger_db_path):
        raise HTTPException(status_code=503, detail="Ledger database not available")
    
    try:
        with sqlite3.connect(ledger_db_path, timeout=10.0) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get proofs for this time range
            proofs = conn.execute("""
                SELECT signature, node_id, proof_type, timestamp,
                       uptime_seconds, tokens_processed, training_batches,
                       reward_amount, current_loss
                FROM proof_history
                WHERE timestamp >= ? AND timestamp < ?
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
            """, (epoch_start, epoch_end, limit, offset)).fetchall()
            
            # Get total count
            total = conn.execute("""
                SELECT COUNT(*) FROM proof_history
                WHERE timestamp >= ? AND timestamp < ?
            """, (epoch_start, epoch_end)).fetchone()[0]
            
            return {
                "epoch_id": epoch_id,
                "total_proofs": total,
                "offset": offset,
                "limit": limit,
                "proofs": [
                    {
                        "signature": p["signature"][:16] + "...",
                        "node_id": p["node_id"][:16] + "...",
                        "proof_type": p["proof_type"],
                        "timestamp": p["timestamp"],
                        "uptime_seconds": p["uptime_seconds"],
                        "tokens_processed": p["tokens_processed"],
                        "training_batches": p["training_batches"],
                        "reward_amount": p["reward_amount"],
                        "current_loss": p["current_loss"],
                    }
                    for p in proofs
                ]
            }
            
    except Exception as e:
        logger.error(f"Error getting epoch proofs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
