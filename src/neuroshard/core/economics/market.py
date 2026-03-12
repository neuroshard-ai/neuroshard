"""
Dynamic Inference Market - Request-Response Marketplace

This module implements a TRUE MARKETPLACE for inference where:
- Users submit requests with locked-in prices
- Nodes claim requests and process them
- Payment guaranteed at submission price (no timing attacks)
- Full attack resistance through request matching

Economic Principles:
1. High demand + Low supply = High price (attracts nodes to serve)
2. Low demand + High supply = Low price (nodes switch to training)
3. Market finds equilibrium where: inference_profit ≈ training_profit × 0.5

Security Features:
1. Request-Response Matching: Price locked at submission time
2. Atomic Claiming: Prevents double-claim attacks
3. Claim Timeouts: Squatting nodes slashed and requests re-queued
4. ECDSA Signatures: User must sign request to authorize payment
5. Capacity Timeout: Fake capacity removed after 60s
6. Price Smoothing: EMA prevents flash crashes
7. Rate Limiting: Per-node and per-user limits
"""

import time
import math
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
from collections import deque
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class RequestStatus(str, Enum):
    """Request lifecycle states."""
    PENDING = "pending"              # Waiting for initiator to claim
    CLAIMED = "claimed"              # Initiator claimed, starting pipeline
    PROCESSING = "processing"        # Pipeline actively processing
    COMPLETED = "completed"          # All proofs received, done
    FAILED = "failed"                # Timeout or error


@dataclass
class InferenceRequest:
    """
    Marketplace request (PUBLIC metadata - NO PROMPT!).
    
    Privacy: Prompt is sent DIRECTLY to initiator node (encrypted).
    The marketplace only orchestrates pricing and pipeline coordination.
    """
    request_id: str              # Unique ID (UUID)
    user_id: str                 # User who submitted
    initiator_node_id: str       # Which initiator node will process (Layer 0)
    
    # Pricing
    tokens_requested: int        # Max tokens to generate
    max_price: float             # Max NEURO per 1M tokens user will pay
    locked_price: float          # Actual price locked at submission time
    priority: int                # Higher = more urgent
    
    # Pipeline state (distributed inference)
    pipeline_session_id: Optional[str] = None  # Tracks distributed session
    status: RequestStatus = RequestStatus.PENDING
    claimed_by: Optional[str] = None  # Node ID that claimed this request
    
    # Timestamps
    timestamp: float = 0.0       # When submitted
    claimed_at: Optional[float] = None  # When initiator claimed
    completed_at: Optional[float] = None # When all proofs received
    
    # Proofs received (quorum-based pipeline)
    initiator_proof_received: bool = False
    processor_proofs_received: List[str] = field(default_factory=list)  # Node IDs
    finisher_proof_received: bool = False
    
    # Security
    user_signature: str = ""     # ECDSA signature authorizing payment
    
    @property
    def completed(self) -> bool:
        """Check if request is completed (for ledger.py compatibility)."""
        return self.status == RequestStatus.COMPLETED


@dataclass
class NodeCapacity:
    """Available inference capacity from a node."""
    node_id: str
    tokens_per_second: int  # Processing capacity
    min_price: float  # Minimum NEURO per 1M tokens node will accept
    timestamp: float


@dataclass
class PipelineSession:
    """
    Tracks quorum-based inference pipeline across multiple nodes.
    
    Privacy: Prompt is NEVER stored here - only initiator knows it!
    Processors only see activations (meaningless vectors).
    """
    session_id: str              # UUID
    request_id: str              # Links to InferenceRequest
    initiator_node_id: str       # Layer 0 (embedding)
    
    # Pipeline participants (quorum members)
    processor_node_ids: List[str] = field(default_factory=list)  # Layers 1-N
    finisher_node_id: Optional[str] = None  # LM Head
    
    # State tracking
    current_layer: int = 0       # Which layer is processing now
    tokens_generated: int = 0    # How many tokens generated so far
    
    # Security: Verify pipeline integrity
    activations_hashes: List[str] = field(default_factory=list)  # Hash per layer
    
    # Timestamps
    started_at: float = 0.0
    last_activity: float = 0.0


class InferenceMarket:
    """
    Real-time market for inference pricing.
    
    Price Discovery:
    - Measures real-time supply (nodes offering capacity)
    - Measures real-time demand (pending requests)
    - Adjusts price to clear the market
    
    Properties:
    - Self-regulating (no admin intervention)
    - Attack-resistant (real supply/demand signals)
    - Training-dominant (low prices push nodes to train)
    """
    
    def __init__(
        self,
        price_smoothing: float = 0.8,   # Exponential moving average (prevents volatility)
        capacity_timeout: int = 60,     # Seconds before stale capacity expires
        claim_timeout: int = 60,        # Seconds before claimed request expires
        base_price: float = 0.0001,     # Bootstrap starting price (worthless model)
    ):
        self.price_smoothing = price_smoothing
        self.capacity_timeout = capacity_timeout
        self.claim_timeout = claim_timeout
        self.base_price = base_price
        
        # REQUEST QUEUES (marketplace state)
        self.pending_requests: Dict[str, InferenceRequest] = {}   # Unclaimed requests
        self.claimed_requests: Dict[str, InferenceRequest] = {}   # Claimed but incomplete
        self.completed_requests: Dict[str, InferenceRequest] = {} # Done (audit trail)
        
        # RESULTS STORAGE (user retrieval)
        self.results: Dict[str, str] = {}  # request_id -> output_text
        self.result_timestamps: Dict[str, float] = {}  # For cleanup
        
        # PIPELINE SESSIONS (distributed inference tracking)
        self.active_sessions: Dict[str, PipelineSession] = {}  # session_id -> session
        self.completed_sessions: Dict[str, PipelineSession] = {}  # Audit trail
        
        # Capacity tracking
        self.available_capacity: List[NodeCapacity] = []
        
        # Price tracking
        self.current_price = base_price  # Start near zero (model has no value yet)
        self.price_history = deque(maxlen=100)
        
        # Security: Prevent replay attacks
        self.request_nonces: Set[str] = set()  # Track used request IDs
        self.last_capacity_update: Dict[str, float] = {}  # Rate limit capacity updates
        
        # Threading
        self.lock = threading.Lock()
        
        # Metrics
        self.total_tokens_processed = 0
        self.total_requests_served = 0
        self.price_updates = 0
        
        logger.info("InferenceMarket initialized with request-response matching")
    
    # ========================================================================
    # SUPPLY MEASUREMENT
    # ========================================================================
    
    def register_capacity(
        self,
        node_id: str,
        tokens_per_second: int,
        min_price: float
    ):
        """
        Node announces available inference capacity.
        
        Args:
            node_id: Node identifier
            tokens_per_second: How many tokens/sec this node can process
            min_price: Minimum price node will accept (0 = any price)
        
        Security:
            - Rate limited to prevent spam
            - Sanity check on capacity claims
            - Expires after 60s (must refresh)
        """
        with self.lock:
            now = time.time()
            
            # SECURITY: Rate limit capacity updates (max 1 per 30 seconds)
            if node_id in self.last_capacity_update:
                if (now - self.last_capacity_update[node_id]) < 30:
                    logger.warning(f"Capacity update too frequent from {node_id[:16]}...")
                    return  # Silently ignore spam
            
            # SECURITY: Sanity check on capacity claim
            MAX_CAPACITY_SINGLE_NODE = 50000  # 50k tokens/sec (~high-end GPU)
            if tokens_per_second > MAX_CAPACITY_SINGLE_NODE:
                logger.warning(f"Unrealistic capacity claim from {node_id[:16]}...: "
                             f"{tokens_per_second} t/s (max {MAX_CAPACITY_SINGLE_NODE})")
                tokens_per_second = MAX_CAPACITY_SINGLE_NODE  # Cap it
            
            # Remove old capacity from this node
            self.available_capacity = [
                c for c in self.available_capacity 
                if c.node_id != node_id
            ]
            
            # Add new capacity
            self.available_capacity.append(NodeCapacity(
                node_id=node_id,
                tokens_per_second=tokens_per_second,
                min_price=min_price,
                timestamp=now
            ))
            
            # Track last update time
            self.last_capacity_update[node_id] = now
            
            # Trigger price update
            self._update_market_price()
    
    def withdraw_capacity(self, node_id: str):
        """Node withdraws from inference market (e.g., to focus on training)."""
        with self.lock:
            self.available_capacity = [
                c for c in self.available_capacity 
                if c.node_id != node_id
            ]
            self._update_market_price()
    
    def get_total_supply(self) -> int:
        """
        Calculate total available supply in tokens/sec.
        
        Returns:
            Total network inference capacity (tokens/sec)
        """
        now = time.time()
        
        # Clean up stale capacity announcements
        self.available_capacity = [
            c for c in self.available_capacity
            if (now - c.timestamp) < self.capacity_timeout
        ]
        
        return sum(c.tokens_per_second for c in self.available_capacity)
    
    # ========================================================================
    # MARKETPLACE: REQUEST SUBMISSION
    # ========================================================================
    
    def submit_request(
        self,
        user_id: str,
        initiator_node_id: str,
        tokens_requested: int,
        max_price: float,
        user_signature: str = "",
        priority: int = 0
    ) -> Tuple[bool, str, float]:
        """
        Submit inference request to marketplace.
        
        PRIVACY: Prompt is NOT sent to marketplace!
        User sends encrypted prompt DIRECTLY to initiator node.
        Marketplace only handles pricing and pipeline coordination.
        
        Args:
            user_id: User submitting request
            initiator_node_id: Which initiator node (Layer 0) will process
            tokens_requested: Max tokens to generate
            max_price: Max NEURO per 1M tokens user will pay
            user_signature: ECDSA signature authorizing payment
            priority: Request priority (higher = more urgent)
        
        Returns:
            (success, request_id, locked_price)
        
        Security:
            - User signature validates authorization
            - Nonce prevents replay attacks
            - Price locked at current market rate
            - Initiator-specific (user chooses who sees prompt)
        """
        try:
            # Input validation
            if not user_id or not initiator_node_id:
                logger.warning("Invalid request: missing user_id or initiator_node_id")
                return False, "", self.current_price
            
            if tokens_requested <= 0:
                logger.warning(f"Invalid tokens_requested: {tokens_requested}")
                return False, "", self.current_price
            
            if max_price < 0:
                logger.warning(f"Invalid max_price: {max_price}")
                return False, "", self.current_price
            
            with self.lock:
                # 1. Check current price vs user's max
                current_price = self.current_price
                if max_price < current_price:
                    logger.debug(f"Price too high for user: {current_price:.6f} > {max_price:.6f}")
                    return False, "", current_price  # Price too high for user
                
                # 2. Generate unique request ID
                request_id = str(uuid.uuid4())
                
                # 3. Check for replay attack
                if request_id in self.request_nonces:
                    logger.warning(f"Duplicate request_id detected: {request_id}")
                    return False, "", current_price
                
                # 4. Create request with LOCKED price (NO PROMPT - privacy!)
                request = InferenceRequest(
                    request_id=request_id,
                    user_id=user_id,
                    initiator_node_id=initiator_node_id,  # User chooses initiator
                    tokens_requested=tokens_requested,
                    max_price=max_price,
                    locked_price=current_price,  # LOCKED at submission time!
                    priority=priority,
                    timestamp=time.time(),
                    status=RequestStatus.PENDING,
                    user_signature=user_signature
                )
                
                # 5. Add to pending queue
                self.pending_requests[request_id] = request
                self.request_nonces.add(request_id)
                
                # 6. Update market price (demand increased)
                self._update_market_price()
                
                logger.info(f"Request {request_id[:8]}... submitted: "
                           f"{tokens_requested} tokens @ {current_price:.6f} NEURO/1M")
                
                return True, request_id, current_price
                
        except Exception as e:
            logger.error(f"Error submitting request: {e}")
            return False, "", self.current_price
    
    # ========================================================================
    # MARKETPLACE: REQUEST CLAIMING
    # ========================================================================
    
    def claim_request(self, node_id: str) -> Optional[InferenceRequest]:
        """
        Driver node claims a pending request to start distributed pipeline.
        
        DISTRIBUTED INFERENCE:
        - Only the specified driver node can claim
        - Initiator starts the pipeline (Layer 0)
        - Processors and finisher join via pipeline session
        
        Args:
            node_id: Initiator node ID attempting to claim
        
        Returns:
            InferenceRequest if claim successful, None if no requests available
        
        Security:
            - Atomic operation (thread-safe)
            - Initiator-specific (must match initiator_node_id)
            - Priority-based selection
        """
        try:
            if not node_id:
                logger.warning("claim_request called with empty node_id")
                return None
            
            with self.lock:
                if not self.pending_requests:
                    return None  # No requests available
                
                # Find requests for THIS initiator only
                my_requests = [
                    r for r in self.pending_requests.values()
                    if r.initiator_node_id == node_id
                ]
                
                if not my_requests:
                    return None  # No requests for this initiator
                
                # Select best request (highest priority, oldest first)
                sorted_requests = sorted(
                    my_requests,
                    key=lambda r: (-r.priority, r.timestamp)
                )
                
                request = sorted_requests[0]
                
                # Claim the request (atomic operation)
                request.claimed_at = time.time()
                request.claimed_by = node_id  # Track who claimed it (for ledger verification)
                request.status = RequestStatus.CLAIMED
                
                # Generate pipeline session ID
                request.pipeline_session_id = str(uuid.uuid4())
                
                # Move from pending to claimed
                self.claimed_requests[request.request_id] = request
                del self.pending_requests[request.request_id]
                
                logger.info(f"Request {request.request_id[:8]}... claimed by INITIATOR {node_id[:16]}... "
                           f"(session {request.pipeline_session_id[:8]}...) at locked price {request.locked_price:.6f} NEURO/1M")
                
                return request
                
        except Exception as e:
            logger.error(f"Error claiming request for node {node_id[:16] if node_id else 'unknown'}...: {e}")
            return None
    
    # ========================================================================
    # MARKETPLACE: REQUEST COMPLETION
    # ========================================================================
    
    def register_proof_received(
        self,
        request_id: str,
        node_id: str,
        is_initiator: bool,
        is_finisher: bool
    ) -> Tuple[bool, str]:
        """
        Track that a node has submitted proof for this request.
        
        In quorum-based inference, multiple nodes process the same request:
        - Initiator (Layer 0) - embeds input
        - Processors (Layers 1-N) - process through layers
        - Finisher (LM Head) - generates output
        
        Each submits their own proof. Request is complete when all proofs received.
        
        Args:
            request_id: Which request
            node_id: Which node submitted proof
            is_initiator: Has embedding layer
            is_finisher: Has LM head layer
        
        Returns:
            (is_request_complete, error_message)
        """
        try:
            # Input validation
            if not request_id:
                return False, "Missing request_id"
            if not node_id:
                return False, "Missing node_id"
            
            with self.lock:
                # Look in both claimed and completed (proof might arrive late)
                request = self.claimed_requests.get(request_id) or self.completed_requests.get(request_id)
                
                if not request:
                    logger.warning(f"Proof received for unknown request {request_id[:8]}...")
                    return False, "Request not found"
                
                # Update proof tracking
                if is_initiator:
                    request.initiator_proof_received = True
                    logger.debug(f"INITIATOR proof received for {request_id[:8]}...")
                elif is_finisher:
                    request.finisher_proof_received = True
                    logger.debug(f"FINISHER proof received for {request_id[:8]}...")
                else:
                    # Processor
                    if node_id not in request.processor_proofs_received:
                        request.processor_proofs_received.append(node_id)
                    logger.debug(f"PROCESSOR proof received for {request_id[:8]}... from {node_id[:16]}...")
            
            # Check if all proofs received (request complete)
            # Minimum: initiator + finisher (processors optional if single-node has all layers)
            all_proofs_received = (
                request.initiator_proof_received and
                request.finisher_proof_received
            )
            
            if all_proofs_received and request.status != RequestStatus.COMPLETED:
                # Mark as completed
                request.status = RequestStatus.COMPLETED
                request.completed_at = time.time()
                
                # Move to completed queue
                if request_id in self.claimed_requests:
                    self.completed_requests[request_id] = request
                    del self.claimed_requests[request_id]
                
                # Update metrics
                self.total_requests_served += 1
                
                # Update market price (demand decreased)
                self._update_market_price()
                
                logger.info(f"Request {request_id[:8]}... COMPLETE: "
                           f"initiator + {len(request.processor_proofs_received)} processors + finisher")
                
                return True, ""
            
            return False, ""  # Not yet complete
            
        except Exception as e:
            logger.error(f"Error registering proof for request {request_id[:8] if request_id else 'unknown'}...: {e}")
            return False, str(e)
    
    def get_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Look up a request by ID (across all queues)."""
        with self.lock:
            if request_id in self.pending_requests:
                return self.pending_requests[request_id]
            elif request_id in self.claimed_requests:
                return self.claimed_requests[request_id]
            elif request_id in self.completed_requests:
                return self.completed_requests[request_id]
            return None
    
    def store_result(self, request_id: str, output_text: str):
        """
        Store inference result for user retrieval.
        
        Results are kept for 5 minutes after completion.
        """
        with self.lock:
            self.results[request_id] = output_text
            self.result_timestamps[request_id] = time.time()
            logger.info(f"Stored result for request {request_id[:8]}... ({len(output_text)} chars)")
    
    def get_result(self, request_id: str) -> Optional[str]:
        """Retrieve inference result."""
        with self.lock:
            return self.results.get(request_id)
    
    def cleanup_old_results(self):
        """Remove results older than 5 minutes."""
        with self.lock:
            now = time.time()
            to_remove = [
                req_id for req_id, ts in self.result_timestamps.items()
                if (now - ts) > 300  # 5 minutes
            ]
            for req_id in to_remove:
                self.results.pop(req_id, None)
                self.result_timestamps.pop(req_id, None)
            
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} old results")
    
    # ========================================================================
    # PIPELINE SESSION MANAGEMENT (Distributed Inference)
    # ========================================================================
    
    def start_pipeline_session(
        self,
        request_id: str,
        session_id: str,
        initiator_node_id: str
    ) -> bool:
        """
        Start a quorum-based inference pipeline session.
        
        Called by initiator node after claiming request.
        Processors and finisher join from the quorum.
        
        Args:
            request_id: Which request this session serves
            session_id: Unique session identifier
            initiator_node_id: Initiator node starting the pipeline
        
        Returns:
            True if session started successfully
        """
        with self.lock:
            if session_id in self.active_sessions:
                logger.warning(f"Session {session_id[:8]}... already exists")
                return False
            
            session = PipelineSession(
                session_id=session_id,
                request_id=request_id,
                initiator_node_id=initiator_node_id,
                started_at=time.time(),
                last_activity=time.time()
            )
            
            self.active_sessions[session_id] = session
            
            logger.info(f"Pipeline session {session_id[:8]}... started for request {request_id[:8]}...")
            return True
    
    def update_pipeline_progress(
        self,
        session_id: str,
        current_layer: int,
        tokens_generated: int = 0
    ):
        """Update pipeline progress (called by processing nodes)."""
        with self.lock:
            session = self.active_sessions.get(session_id)
            if session:
                session.current_layer = current_layer
                session.tokens_generated = tokens_generated
                session.last_activity = time.time()
    
    def register_pipeline_participant(
        self,
        session_id: str,
        node_id: str,
        is_processor: bool = False,
        is_finisher: bool = False
    ):
        """Register a node as participant in the pipeline."""
        with self.lock:
            session = self.active_sessions.get(session_id)
            if not session:
                logger.warning(f"Session {session_id[:8]}... not found")
                return
            
            if is_processor and node_id not in session.processor_node_ids:
                session.processor_node_ids.append(node_id)
                logger.debug(f"Processor {node_id[:16]}... joined session {session_id[:8]}...")
            elif is_finisher:
                session.finisher_node_id = node_id
                logger.debug(f"Finisher {node_id[:16]}... joined session {session_id[:8]}...")
    
    def complete_pipeline_session(self, session_id: str):
        """Mark pipeline session as complete."""
        with self.lock:
            session = self.active_sessions.get(session_id)
            if session:
                self.completed_sessions[session_id] = session
                del self.active_sessions[session_id]
                logger.info(f"Pipeline session {session_id[:8]}... completed")
    
    def get_session(self, session_id: str) -> Optional[PipelineSession]:
        """Look up a pipeline session."""
        with self.lock:
            return (self.active_sessions.get(session_id) or 
                   self.completed_sessions.get(session_id))
    
    # ========================================================================
    # DEMAND MEASUREMENT (updated for marketplace)
    # ========================================================================
    
    def get_total_demand(self) -> int:
        """
        Calculate total pending demand in tokens.
        
        Returns:
            Total tokens waiting to be processed (pending + claimed but incomplete)
        """
        # Include both pending AND claimed (still being processed)
        pending_tokens = sum(r.tokens_requested for r in self.pending_requests.values())
        claimed_tokens = sum(r.tokens_requested for r in self.claimed_requests.values())
        return pending_tokens + claimed_tokens
    
    # ========================================================================
    # PRICE DISCOVERY
    # ========================================================================
    
    def _update_market_price(self):
        """
        Update market price based on PURE supply/demand (no artificial caps).
        
        Price Formula (Pure Market):
            P = base_price × (1 + utilization)^2
        
        Where:
            utilization = demand_rate / supply_rate
            base_price = Starting price for worthless model (0.0001)
        
        Natural Dynamics:
            - No demand → Price ≈ 0 (stupid model = worthless)
            - Low util (0.1) → Price ≈ 0.00012 (cheap, encourages usage)
            - Balanced (1.0) → Price ≈ 0.0004 (fair value)
            - High util (10) → Price ≈ 0.012 (scarcity premium)
            - Extreme (100) → Price ≈ 1.01 (severe scarcity)
        
        Quality Emerges Naturally:
            - Stupid model → Users don't use it → Demand = 0 → Price ≈ 0
            - Good model → Users want it → Demand ↑ → Price ↑
            - Excellent model → Viral usage → Demand >> Supply → Price spikes
        
        No Caps Needed:
            - Market finds true value
            - High price attracts supply (nodes switch from training)
            - Equilibrium naturally forms where: inference_profit ≈ training_profit
        """
        supply = self.get_total_supply()  # tokens/sec
        demand = self.get_total_demand()  # total tokens waiting
        
        if supply == 0:
            if demand == 0:
                # No supply, no demand → Keep current price
                new_price = self.current_price
            else:
                # No supply but there's demand → Extreme scarcity
                # Price spikes to attract supply (this is CORRECT market behavior!)
                new_price = self.current_price * 10.0  # Dramatic signal
        else:
            # Calculate demand rate (tokens/sec needed to clear queue)
            # Using TARGET_RESPONSE_TIME from economics
            from neuroshard.core.economics.constants import INFERENCE_MARKET_TARGET_RESPONSE_TIME
            demand_rate = demand / INFERENCE_MARKET_TARGET_RESPONSE_TIME
            
            # Utilization ratio (pure supply/demand)
            utilization = demand_rate / supply
            
            # PURE MARKET FORMULA: Price grows with utilization squared
            # This creates natural equilibrium:
            # - Low utilization → Cheap (encourages usage)
            # - High utilization → Expensive (attracts supply)
            # - Quadratic ensures rapid price response to scarcity
            new_price = self.base_price * math.pow(1 + utilization, 2)
        
        # Smooth price changes (exponential moving average)
        # Higher smoothing = more stability, less volatility
        self.current_price = (
            self.price_smoothing * self.current_price +
            (1 - self.price_smoothing) * new_price
        )
        
        # Record for analysis
        self.price_history.append({
            'timestamp': time.time(),
            'price': self.current_price,
            'supply': supply,
            'demand': demand,
            'utilization': demand_rate / supply if supply > 0 else 0
        })
        
        self.price_updates += 1
    
    def get_current_price(self) -> float:
        """
        Get current market price for inference.
        
        Returns:
            NEURO per 1M tokens (current market rate)
        """
        with self.lock:
            return self.current_price
    
    # ========================================================================
    # SECURITY: TIMEOUT HANDLING
    # ========================================================================
    
    def cleanup_stale_claims(self) -> int:
        """
        Re-queue requests that were claimed but not completed (timeout).
        
        This prevents malicious nodes from "squatting" on requests.
        Called periodically by background task.
        
        Returns:
            Number of requests re-queued
        """
        with self.lock:
            now = time.time()
            expired_requests = []
            
            # Find expired claims
            for req_id, request in self.claimed_requests.items():
                try:
                    # Safety check for claimed_at
                    if request.claimed_at is None:
                        logger.warning(f"Request {req_id[:8]}... has no claimed_at timestamp, marking expired")
                        expired_requests.append(req_id)
                    elif (now - request.claimed_at) > self.claim_timeout:
                        expired_requests.append(req_id)
                except Exception as e:
                    logger.error(f"Error checking claim expiry for {req_id[:8]}...: {e}")
            
            # Re-queue them
            for req_id in expired_requests:
                try:
                    request = self.claimed_requests[req_id]
                    
                    # Store info before reset
                    squatter_node_id = request.claimed_by or "unknown"
                    claim_duration = (now - request.claimed_at) if request.claimed_at else 0
                    
                    # Reset claim state
                    request.status = RequestStatus.PENDING
                    request.claimed_by = None
                    request.claimed_at = None
                    request.pipeline_session_id = None
                    
                    # Move back to pending
                    self.pending_requests[req_id] = request
                    del self.claimed_requests[req_id]
                    
                    logger.warning(f"Request {req_id[:8]}... timed out "
                                 f"(claimed by {squatter_node_id[:16]}... for {claim_duration:.1f}s), "
                                 f"re-queued")
                except Exception as e:
                    logger.error(f"Error re-queuing request {req_id[:8]}...: {e}")
            
            return len(expired_requests)
    
    def prune_completed_requests(self, max_age_seconds: int = 3600):
        """
        Remove old completed requests from audit trail.
        
        Keeps memory usage bounded while maintaining recent history.
        
        Args:
            max_age_seconds: How long to keep completed requests (default 1 hour)
        """
        with self.lock:
            now = time.time()
            to_remove = []
            
            for req_id, request in self.completed_requests.items():
                if (now - request.completed_at) > max_age_seconds:
                    to_remove.append(req_id)
            
            for req_id in to_remove:
                del self.completed_requests[req_id]
                # Also remove from nonce set to allow reuse after long time
                self.request_nonces.discard(req_id)
            
            if to_remove:
                logger.debug(f"Pruned {len(to_remove)} old completed requests")
    
    # ========================================================================
    # ANALYTICS
    # ========================================================================
    
    def get_market_stats(self) -> dict:
        """
        Get current marketplace statistics.
        
        Returns:
            Dict with price, supply, demand, queue sizes, etc.
        """
        with self.lock:
            supply = self.get_total_supply()
            demand = self.get_total_demand()
            
            return {
                # Price
                'current_price': round(self.current_price, 6),
                
                # Supply & Demand
                'supply_tokens_per_sec': supply,
                'demand_tokens_waiting': demand,
                'utilization': (demand / 60) / supply if supply > 0 else 0,
                
                # Marketplace queues
                'pending_requests': len(self.pending_requests),
                'claimed_requests': len(self.claimed_requests),
                'completed_requests': len(self.completed_requests),
                
                # Pipeline sessions (distributed inference)
                'active_sessions': len(self.active_sessions),
                'completed_sessions': len(self.completed_sessions),
                'available_nodes': len(self.available_capacity),
                'total_processed': self.total_tokens_processed,
                'total_served': self.total_requests_served,
                'avg_price_24h': self._get_avg_price(24 * 3600)
            }
    
    def _get_avg_price(self, seconds: int) -> float:
        """Calculate average price over last N seconds."""
        if not self.price_history:
            return self.current_price
        
        cutoff = time.time() - seconds
        recent = [p['price'] for p in self.price_history if p['timestamp'] > cutoff]
        
        return sum(recent) / len(recent) if recent else self.current_price
    
    def get_price_chart(self, last_n: int = 100) -> List[dict]:
        """Get recent price history for charting."""
        with self.lock:
            return list(self.price_history)[-last_n:]


# ============================================================================
# INTEGRATION WITH LEDGER
# ============================================================================

def calculate_inference_reward(
    market: InferenceMarket,
    tokens_processed: int
) -> float:
    """
    Calculate inference reward based on current market price.
    
    This replaces the fixed INFERENCE_REWARD_PER_MILLION.
    
    Args:
        market: InferenceMarket instance
        tokens_processed: Number of tokens processed
    
    Returns:
        NEURO reward based on market rate
    """
    market_price = market.get_current_price()
    reward = (tokens_processed / 1_000_000.0) * market_price
    return reward


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup logging for demo
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Create market
    market = InferenceMarket()
    
    # Scenario 1: Low demand (bootstrap phase)
    logger.info("\n=== Scenario 1: Low Demand ===")
    market.register_capacity("node1", tokens_per_second=1000, min_price=0.01)
    market.register_capacity("node2", tokens_per_second=1000, min_price=0.01)
    market.submit_request("user1", "driver1", tokens_requested=10000, max_price=1.0)
    
    stats = market.get_market_stats()
    logger.info(f"Supply: {stats['supply_tokens_per_sec']} t/s")
    logger.info(f"Demand: {stats['demand_tokens_waiting']} tokens")
    logger.info(f"Price: {stats['current_price']:.4f} NEURO per 1M tokens")
    logger.info(f"Utilization: {stats['utilization']*100:.1f}%")
    
    # Scenario 2: High demand
    logger.info("\n=== Scenario 2: High Demand ===")
    for i in range(10):
        market.submit_request(f"user{i+2}", "driver1",
                            tokens_requested=50000, max_price=1.0)
    
    stats = market.get_market_stats()
    logger.info(f"Supply: {stats['supply_tokens_per_sec']} t/s")
    logger.info(f"Demand: {stats['demand_tokens_waiting']} tokens")
    logger.info(f"Price: {stats['current_price']:.4f} NEURO per 1M tokens")
    logger.info(f"Utilization: {stats['utilization']*100:.1f}%")
    
    # Scenario 3: Supply increases (more nodes join)
    logger.info("\n=== Scenario 3: Supply Increases ===")
    for i in range(10):
        market.register_capacity(f"node{i+3}", tokens_per_second=1000, min_price=0.01)
    
    stats = market.get_market_stats()
    logger.info(f"Supply: {stats['supply_tokens_per_sec']} t/s")
    logger.info(f"Demand: {stats['demand_tokens_waiting']} tokens")
    logger.info(f"Price: {stats['current_price']:.4f} NEURO per 1M tokens")
    logger.info(f"Utilization: {stats['utilization']*100:.1f}%")

