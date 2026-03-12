"""
NEURO Token Economics - Centralized Configuration

This module defines ALL economic constants for the NeuroShard network.
All values are documented and should be referenced from here, not hardcoded elsewhere.

=============================================================================
DESIGN PRINCIPLES
=============================================================================

1. TRAINING DOMINATES: Training rewards are the highest to incentivize 
   actual model improvement over passive participation.

2. WORK BEFORE STAKE: Base rewards come from actual work (compute).
   Staking only provides a multiplier, not base rewards.

3. DIMINISHING RETURNS: Stake multipliers use logarithmic scaling to
   prevent rich-get-richer dynamics.

4. DEFLATIONARY: 5% of all spending is burned, creating scarcity.

5. SECURITY: Multiple caps and limits prevent economic attacks.

=============================================================================
"""

# =============================================================================
# REWARD RATES
# =============================================================================

# UPTIME REWARD (minimal - discourages idle farming)
# Goal: Minimal passive income, forces users to train for real rewards
# NOTE: Only awarded when node is in a quorum or actively available
UPTIME_REWARD_PER_MINUTE = 0.0001   # 0.0001 NEURO per minute
                                     # ~0.14 NEURO/day idle (80% reduction)

# TRAINING REWARD (dominant - this is the core value!)
# Goal: Strong incentive for active training, covers electricity + profit
# IMPORTANT: This is PER BATCH PER LAYER - nodes holding more layers earn more!
TRAINING_REWARD_PER_BATCH_PER_LAYER = 0.0005  # 0.0005 NEURO per batch per layer
                                               # Node with 10 layers, 60 batches/min:
                                               # = 0.0005 × 10 × 60 = 0.3 NEURO/min = ~432 NEURO/day

# ASYNC TRAINING REWARD (for offline gradient contributors)
# Goal: Reward async contributors fairly but less than real-time pipeline
# Async gradients are less valuable because they may be stale
ASYNC_TRAINING_REWARD_PER_BATCH_PER_LAYER = 0.0003  # 60% of pipeline rate
                                                     # Lower because:
                                                     # 1. Not time-critical
                                                     # 2. May be stale
                                                     # 3. Less coordination overhead

# ASYNC FRESHNESS DECAY - gradients lose value over time
# Newer gradients are more valuable for training convergence
ASYNC_FRESHNESS_DECAY = {
    3600: 1.0,      # < 1 hour: full value (100%)
    86400: 0.7,     # < 1 day: 70% value
    604800: 0.5,    # < 1 week: 50% value
    float('inf'): 0.3,  # > 1 week: 30% value (still has some diversity value)
}

# DATA SERVING REWARD (for nodes serving training shards)
DATA_REWARD_PER_SAMPLE = 0.00001    # 0.00001 NEURO per data sample served

# INFERENCE REWARD (PURE MARKET-BASED PRICING)
# Goal: Let supply/demand discover the true price
# No artificial caps - market determines value based on model quality via demand
# Quality is implicit: stupid model = no demand = low price, good model = high demand = high price

# Dynamic Market Parameters
INFERENCE_MARKET_PRICE_SMOOTHING = 0.8    # EMA smoothing (higher = smoother price changes)
INFERENCE_MARKET_CAPACITY_TIMEOUT = 60    # Seconds before stale capacity expires
INFERENCE_MARKET_TARGET_RESPONSE_TIME = 60  # Target seconds to serve requests
INFERENCE_MARKET_BASE_PRICE = 0.0001      # Starting price (bootstrap with worthless model)

# =============================================================================
# QUORUM ROLE DISTRIBUTION
# =============================================================================
# In quorum-based architecture, roles are:
# - INITIATOR: Holds embedding layer (Layer 0), starts pipeline
# - PROCESSOR: Holds middle layers, forwards activations
# - FINISHER: Holds LM head, computes loss/output

# Role shares for inference rewards (must sum to 1.0)
INITIATOR_SHARE = 0.15              # 15% - Embedding, batch initiation
PROCESSOR_SHARE = 0.70              # 70% - Heavy computation (split by layers)
FINISHER_SHARE = 0.15               # 15% - LM head, output generation

# =============================================================================
# ROLE BONUSES (ADDITIVE - applied to base reward)
# =============================================================================
# These are ADDITIVE bonuses, not multiplicative!
# Final multiplier = 1.0 + INITIATOR_BONUS + FINISHER_BONUS + TRAINING_BONUS
# Example: Initiator + Finisher + Training = 1.0 + 0.2 + 0.3 + 0.1 = 1.6x
#
# This prevents exponential reward growth and is more intuitive.

INITIATOR_BONUS = 0.2               # +20% bonus for holding embedding layer
FINISHER_BONUS = 0.3                # +30% bonus for holding LM head + loss calc
TRAINING_BONUS = 0.1                # +10% bonus when actively training

# NOTE: Layer scaling is handled by per-layer reward calculation
# (TRAINING_REWARD_PER_BATCH_PER_LAYER × layers_held)
# No separate layer bonus needed - reward scales directly with layers held

# =============================================================================
# REPUTATION-BASED BONUSES
# =============================================================================
# Nodes with high uptime and success rates receive bonus rewards.
# This incentivizes reliable participation in the network.

# Reputation bonus (up to 10% extra based on reputation score)
REPUTATION_BONUS_MAX = 0.10         # Maximum 10% bonus for perfect reputation
MIN_UPTIME_FOR_PIPELINE = 0.90      # Minimum 90% uptime required for pipeline mode
MIN_UPTIME_FOR_QUORUM = 0.80        # Minimum 80% uptime to join a quorum

# =============================================================================
# SCARCITY-BASED INCENTIVES
# =============================================================================
# Nodes holding under-replicated layers receive bonus rewards.
# This incentivizes balanced layer distribution across the network.

# Base scarcity bonus multiplier
SCARCITY_BONUS_BASE = 0.5           # 50% base bonus for under-replicated layers
SCARCITY_BONUS_MAX = 1.0            # Max 100% bonus (2x rewards)

# Target replicas per layer (scales with network size)
# See get_target_replicas() for dynamic calculation
TARGET_REPLICAS_BASE = 3            # Minimum replicas per layer
TARGET_REPLICAS_SCALE = 0.1         # Scale factor (10% of nodes)


def get_target_replicas(network_size: int) -> int:
    """
    Get target number of replicas per layer based on network size.
    
    Uses dynamic calculation: max(TARGET_REPLICAS_BASE, network_size * TARGET_REPLICAS_SCALE)
    
    Examples:
        - 10 nodes -> 3 replicas (minimum)
        - 50 nodes -> 5 replicas
        - 100 nodes -> 10 replicas
        - 500 nodes -> 50 replicas
    
    Args:
        network_size: Number of nodes in network
        
    Returns:
        Target number of replicas per layer
    """
    scaled = int(network_size * TARGET_REPLICAS_SCALE)
    return max(TARGET_REPLICAS_BASE, scaled)


def compute_scarcity_bonus(layer_id: int, replicas: int, network_size: int) -> float:
    """
    Compute scarcity bonus multiplier for a layer.
    
    Nodes holding under-replicated layers receive bonus rewards.
    This incentivizes nodes to hold rare layers, improving network resilience.
    
    Bonus Calculation:
    - If replicas >= target: 1.0 (no bonus)
    - If replicas < target: 1.0 + SCARCITY_BONUS_BASE * (target - replicas) / target
    - Capped at 1.0 + SCARCITY_BONUS_MAX (2x rewards)
    
    Args:
        layer_id: Layer identifier
        replicas: Current number of nodes holding this layer
        network_size: Total number of nodes in network
        
    Returns:
        Reward multiplier (1.0 to 1.0 + SCARCITY_BONUS_MAX)
    
    Examples:
        >>> compute_scarcity_bonus(5, 3, 100)  # Target is 10 replicas
        1.35  # 35% bonus (under-replicated)
        >>> compute_scarcity_bonus(5, 10, 100)  # At target
        1.0  # No bonus
        >>> compute_scarcity_bonus(5, 1, 100)  # Critical shortage
        1.5  # Max bonus (capped)
    """
    target = get_target_replicas(network_size)
    
    if replicas >= target:
        return 1.0  # No bonus - layer is well replicated
    
    # Calculate bonus based on shortage
    shortage_ratio = (target - replicas) / target
    bonus = SCARCITY_BONUS_BASE * shortage_ratio
    
    # Cap at maximum bonus
    return min(1.0 + bonus, 1.0 + SCARCITY_BONUS_MAX)


def get_layer_scarcity_scores(
    layer_counts: dict,  # {layer_id: replica_count}
    network_size: int,
) -> dict:
    """
    Get scarcity scores for all layers.
    
    Args:
        layer_counts: Dict mapping layer_id to number of replicas
        network_size: Total number of nodes
        
    Returns:
        Dict mapping layer_id to (scarcity_bonus, is_under_replicated)
    """
    target = get_target_replicas(network_size)
    scores = {}
    
    for layer_id, replicas in layer_counts.items():
        bonus = compute_scarcity_bonus(layer_id, replicas, network_size)
        is_under = replicas < target
        scores[layer_id] = {
            "bonus": bonus,
            "under_replicated": is_under,
            "replicas": replicas,
            "target": target,
            "shortage": max(0, target - replicas),
        }
    
    return scores

# =============================================================================
# STAKING ECONOMICS
# =============================================================================

# Stake multiplier formula: 1.0 + STAKING_BASE_BONUS * log2(1 + stake / STAKING_UNIT)
# Uses logarithmic diminishing returns to prevent whale dominance
STAKING_BASE_BONUS = 0.1            # Base 10% bonus coefficient
STAKING_UNIT = 1000.0               # Staking calculated per 1000 NEURO

# Staking limits
MIN_STAKE_AMOUNT = 1.0              # Minimum stake amount (1 NEURO)
MAX_STAKE_AMOUNT = 10_000_000.0     # Maximum stake amount (10M NEURO)
MIN_STAKE_DURATION_DAYS = 1         # Minimum lock period (1 day)
MAX_STAKE_DURATION_DAYS = 365       # Maximum lock period (1 year)

# =============================================================================
# VALIDATOR REQUIREMENTS (DYNAMIC SCALING)
# =============================================================================

# Base validator stake - used when network is small
VALIDATOR_BASE_STAKE = 100.0        # Starting minimum (100 NEURO)
VALIDATOR_MIN_MEMORY_MB = 2000      # Minimum memory for Validator (2GB)

# Dynamic stake tiers - scales with network size for security
# As more validators join, stake requirement increases to maintain security
VALIDATOR_STAKE_TIERS = [
    # (max_validators, required_stake)
    (2, 0.0),          # Bootstrap: 0-2 validators → FREE (network needs to start!)
    (10, 100.0),       # Early: 3-10 validators → 100 NEURO
    (50, 250.0),       # Growing: 11-50 validators → 250 NEURO  
    (200, 500.0),      # Established: 51-200 validators → 500 NEURO
    (1000, 1000.0),    # Mature: 201-1000 validators → 1,000 NEURO
    (float('inf'), 2500.0),  # Large scale: 1000+ validators → 2,500 NEURO
]

# Legacy constant for backward compatibility (uses dynamic function below)
VALIDATOR_MIN_STAKE = VALIDATOR_BASE_STAKE

# Validation rewards
VALIDATION_FEE_PER_PROOF = 0.001    # 0.001 NEURO per proof validated
VALIDATION_CONSENSUS_THRESHOLD = 0.66  # 66% stake-weighted agreement required

# Validator selection
VALIDATOR_ROTATION_ENABLED = True   # Enable random validator selection
VALIDATOR_SELECTION_RANDOMNESS = 0.3  # 30% randomness in selection

# Remote proof security (limits impact of fake stake claims)
REMOTE_STAKE_MULTIPLIER_CAP = 1.5   # Max multiplier for remote proofs


def get_dynamic_validator_stake(num_validators: int) -> float:
    """
    Get the required validator stake based on current network size.
    
    Scales automatically to maintain security as network grows:
    - Few validators: Low barrier (100 NEURO) for accessibility
    - Many validators: Higher barrier (2500 NEURO) for security
    
    Args:
        num_validators: Current number of validators in the network
        
    Returns:
        Required stake in NEURO
        
    Examples:
        >>> get_dynamic_validator_stake(5)    # Bootstrap
        100.0
        >>> get_dynamic_validator_stake(30)   # Growing
        250.0
        >>> get_dynamic_validator_stake(100)  # Established
        500.0
        >>> get_dynamic_validator_stake(500)  # Mature
        1000.0
        >>> get_dynamic_validator_stake(2000) # Large scale
        2500.0
    """
    for max_validators, required_stake in VALIDATOR_STAKE_TIERS:
        if num_validators <= max_validators:
            return required_stake
    
    # Fallback to highest tier
    return VALIDATOR_STAKE_TIERS[-1][1]


def get_validator_stake_info(num_validators: int) -> dict:
    """
    Get detailed validator stake information for current network state.
    
    Returns:
        Dict with current requirement, next tier, and progress info
    """
    current_stake = get_dynamic_validator_stake(num_validators)
    
    # Find current and next tier
    current_tier_idx = 0
    for i, (max_val, stake) in enumerate(VALIDATOR_STAKE_TIERS):
        if num_validators <= max_val:
            current_tier_idx = i
            break
    
    # Get next tier info
    next_tier = None
    validators_until_increase = None
    if current_tier_idx < len(VALIDATOR_STAKE_TIERS) - 1:
        current_max = VALIDATOR_STAKE_TIERS[current_tier_idx][0]
        next_stake = VALIDATOR_STAKE_TIERS[current_tier_idx + 1][1]
        validators_until_increase = current_max - num_validators + 1
        next_tier = {
            "stake": next_stake,
            "at_validators": current_max + 1,
            "validators_away": validators_until_increase,
        }
    
    return {
        "current_stake_required": current_stake,
        "num_validators": num_validators,
        "tier": current_tier_idx + 1,
        "total_tiers": len(VALIDATOR_STAKE_TIERS),
        "next_tier": next_tier,
        "tiers": [
            {"max_validators": max_v if max_v != float('inf') else "unlimited", "stake": s}
            for max_v, s in VALIDATOR_STAKE_TIERS
        ],
    }

# =============================================================================
# FEE BURN (Deflationary Mechanism)
# =============================================================================

FEE_BURN_RATE = 0.05                # 5% of spending fees are burned
BURN_ADDRESS = "BURN_0x0000000000000000000000000000000000000000"

# =============================================================================
# ANTI-CHEAT LIMITS
# =============================================================================

MAX_UPTIME_PER_PROOF = 120          # Max 2 minutes per proof (prevents inflation)
MAX_TOKENS_PER_MINUTE = 1_000_000   # Max 1M tokens/minute (modern GPUs can do this)
MAX_PROOFS_PER_HOUR = 120           # Max 2 proofs per minute sustained
PROOF_FRESHNESS_WINDOW = 300        # Proofs valid for 5 minutes

# =============================================================================
# SLASHING
# =============================================================================

SLASH_AMOUNT = 10.0                 # NEURO slashed for fraud
WHISTLEBLOWER_REWARD_RATE = 0.5     # 50% of slash goes to reporter
VALIDATOR_SLASH_MULTIPLIER = 2.0    # Validators slashed 2x for bad validation

# =============================================================================
# SUPPLY LIMITS
# =============================================================================

# There is NO hard cap on total supply - NEURO is minted through PoNW
# However, deflationary mechanics (burn) and diminishing rewards create scarcity
MAX_REWARD_PER_PROOF = 100.0        # Cap on single proof reward (sanity check)
MAX_DAILY_MINT_PER_NODE = 10_000.0  # Cap on daily minting per node

# =============================================================================
# GENESIS
# =============================================================================

GENESIS_SUPPLY = 0.0                # Zero pre-mine - all NEURO is earned
GENESIS_SIGNATURE = "GENESIS_BLOCK"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

import math


def calculate_stake_multiplier(stake: float) -> float:
    """
    Calculate stake multiplier with diminishing returns.
    
    Formula: 1.0 + STAKING_BASE_BONUS * log2(1 + stake / STAKING_UNIT)
    
    Examples:
    - 0 NEURO = 1.00x
    - 1,000 NEURO = 1.10x
    - 2,000 NEURO = 1.16x
    - 10,000 NEURO = 1.35x
    - 100,000 NEURO = 1.66x
    """
    if stake <= 0:
        return 1.0
    
    return 1.0 + STAKING_BASE_BONUS * math.log2(1 + stake / STAKING_UNIT)


def calculate_async_freshness(age_seconds: float) -> float:
    """
    Calculate freshness multiplier for async gradient contributions.
    
    Newer gradients are more valuable for training convergence.
    Older gradients still have diversity value but are discounted.
    
    Args:
        age_seconds: Time since gradient was computed
        
    Returns:
        Freshness multiplier (0.3 to 1.0)
        
    Examples:
        >>> calculate_async_freshness(1800)   # 30 minutes
        1.0
        >>> calculate_async_freshness(43200)  # 12 hours
        0.7
        >>> calculate_async_freshness(259200) # 3 days
        0.5
        >>> calculate_async_freshness(1000000) # > 1 week
        0.3
    """
    for threshold, freshness in sorted(ASYNC_FRESHNESS_DECAY.items()):
        if age_seconds < threshold:
            return freshness
    # Fallback to minimum freshness
    return 0.3


def calculate_training_reward(
    batches: int,
    layers_held: int,
    is_async: bool = False,
    age_seconds: float = 0.0,
) -> float:
    """
    Calculate training reward for a node.
    
    This is the canonical function for computing training rewards.
    Use this instead of manually multiplying constants.
    
    Args:
        batches: Number of training batches processed
        layers_held: Number of layers this node holds
        is_async: Whether this is an async contribution (vs pipeline)
        age_seconds: Age of contribution in seconds (for async freshness)
        
    Returns:
        Base training reward in NEURO (before multipliers)
        
    Examples:
        >>> calculate_training_reward(60, 10)  # Pipeline: 60 batches, 10 layers
        0.3  # 60 × 0.0005 × 10 = 0.3 NEURO
        >>> calculate_training_reward(60, 10, is_async=True, age_seconds=7200)  # 2hr old async
        0.126  # 60 × 0.0003 × 10 × 0.7 = 0.126 NEURO
    """
    if layers_held <= 0:
        return 0.0
    
    if is_async:
        base_rate = ASYNC_TRAINING_REWARD_PER_BATCH_PER_LAYER
        freshness = calculate_async_freshness(age_seconds)
        return batches * base_rate * layers_held * freshness
    else:
        base_rate = TRAINING_REWARD_PER_BATCH_PER_LAYER
        return batches * base_rate * layers_held


def calculate_role_bonus(
    has_embedding: bool = False,
    has_lm_head: bool = False,
    is_training: bool = False,
) -> float:
    """
    Calculate the role bonus multiplier (ADDITIVE).
    
    The multiplier is 1.0 + sum of applicable bonuses.
    
    Args:
        has_embedding: Node holds embedding layer (Layer 0)
        has_lm_head: Node holds LM head layer
        is_training: Node is actively training
        
    Returns:
        Role multiplier (1.0 to 1.6)
        
    Examples:
        >>> calculate_role_bonus()  # Plain processor
        1.0
        >>> calculate_role_bonus(has_embedding=True)  # Initiator
        1.2
        >>> calculate_role_bonus(has_lm_head=True)  # Finisher
        1.3
        >>> calculate_role_bonus(has_embedding=True, has_lm_head=True, is_training=True)  # Solo training
        1.6  # 1.0 + 0.2 + 0.3 + 0.1
    """
    multiplier = 1.0
    
    if has_embedding:
        multiplier += INITIATOR_BONUS
    
    if has_lm_head:
        multiplier += FINISHER_BONUS
    
    if is_training:
        multiplier += TRAINING_BONUS
    
    return multiplier


def calculate_burn_amount(spend_amount: float) -> float:
    """Calculate the burn amount for a transaction."""
    return spend_amount * FEE_BURN_RATE


def is_valid_stake_amount(amount: float) -> tuple:
    """
    Validate a stake amount.
    
    Returns: (is_valid, error_message)
    """
    if amount < MIN_STAKE_AMOUNT:
        return False, f"Minimum stake is {MIN_STAKE_AMOUNT} NEURO"
    if amount > MAX_STAKE_AMOUNT:
        return False, f"Maximum stake is {MAX_STAKE_AMOUNT:,.0f} NEURO"
    return True, ""


def is_valid_stake_duration(days: int) -> tuple:
    """
    Validate a stake duration.
    
    Returns: (is_valid, error_message)
    """
    if days < MIN_STAKE_DURATION_DAYS:
        return False, f"Minimum lock period is {MIN_STAKE_DURATION_DAYS} day(s)"
    if days > MAX_STAKE_DURATION_DAYS:
        return False, f"Maximum lock period is {MAX_STAKE_DURATION_DAYS} days"
    return True, ""


def is_eligible_validator(
    stake: float, 
    memory_mb: float = None,
    num_validators: int = 0
) -> tuple:
    """
    Check if a node is eligible to be a validator.
    
    Uses DYNAMIC stake requirement based on network size:
    - Few validators (1-10): 100 NEURO
    - Growing (11-50): 250 NEURO
    - Established (51-200): 500 NEURO
    - Mature (201-1000): 1,000 NEURO
    - Large scale (1000+): 2,500 NEURO
    
    Args:
        stake: Amount of NEURO staked
        memory_mb: Available memory in MB
        num_validators: Current number of validators in network (for dynamic scaling)
    
    Returns: (is_eligible, reason)
    """
    # Get dynamic stake requirement
    required_stake = get_dynamic_validator_stake(num_validators)
    
    if stake < required_stake:
        return False, f"Insufficient stake: {stake:.2f} < {required_stake:.0f} NEURO (current network requires {required_stake:.0f} with {num_validators} validators)"
    
    if memory_mb is not None and memory_mb < VALIDATOR_MIN_MEMORY_MB:
        return False, f"Insufficient memory: {memory_mb:.0f}MB < {VALIDATOR_MIN_MEMORY_MB}MB"
    
    return True, f"Eligible with {stake:.2f} NEURO staked (requirement: {required_stake:.0f} NEURO)"


# =============================================================================
# SUMMARY TABLE (for reference)
# =============================================================================
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        NEURO ECONOMICS SUMMARY                                ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ EARNING NEURO                                                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Activity          │ Rate                    │ Daily (Active, 10 layers)        ║
║───────────────────┼─────────────────────────┼──────────────────────────────────║
║ Pipeline Training │ 0.0005 NEURO/batch/layer│ ~432 NEURO (60 batch/min)        ║
║ Async Training    │ 0.0003 NEURO/batch/layer│ ~259 NEURO (with freshness)      ║
║ Inference         │ DYNAMIC (market-based)  │ Supply/demand driven             ║
║ Data Serving      │ 0.00001 NEURO/sample    │ Variable                         ║
║ Uptime (idle)     │ 0.0001 NEURO/min        │ ~0.14 NEURO (minimal)            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ REALISTIC DAILY EARNINGS (10 layers, with bonuses)                            ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Idle node         │ ~0.14 NEURO/day         │ Just uptime (unprofitable)       ║
║ Async (1 layer)   │ ~15-25 NEURO/day        │ Raspberry Pi, stale gradients    ║
║ Pipeline (5 layer)│ ~200-300 NEURO/day      │ Gaming PC in quorum              ║
║ Pipeline (10 layer│ ~400-600 NEURO/day      │ Server, full model segment       ║
║ Solo (all layers) │ ~500-800 NEURO/day      │ Full model + all bonuses         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ ROLE BONUSES (ADDITIVE - sum together)                                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Initiator         │ +20%                    │ Holds embedding layer (Layer 0)  ║
║ Finisher          │ +30%                    │ Holds LM head layer              ║
║ Training          │ +10%                    │ When actively training           ║
║ Example (all 3)   │ 1.6x total              │ 1.0 + 0.2 + 0.3 + 0.1 = 1.6x    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ OTHER MULTIPLIERS                                                              ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Staking           │ log2(1 + stake/1000)    │ 1.10x @ 1K, 1.66x @ 100K        ║
║ Reputation        │ up to +10%              │ Based on uptime/success rate     ║
║ Scarcity          │ up to +100%             │ For under-replicated layers      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ ASYNC FRESHNESS DECAY                                                          ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ < 1 hour          │ 100% value              │ Fresh gradients, full reward     ║
║ < 1 day           │ 70% value               │ Slightly stale                   ║
║ < 1 week          │ 50% value               │ Moderately stale                 ║
║ > 1 week          │ 30% value               │ Old but still diversity value    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ REQUIREMENTS                                                                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Validator         │ 100+ NEURO stake        │ + 2GB memory (scales w/network)  ║
║ Stake Min/Max     │ 1 / 10,000,000 NEURO    │                                  ║
║ Lock Period       │ 1 - 365 days            │                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ FEES & BURNS                                                                   ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║ Transaction Fee   │ 5% burned               │ Deflationary                     ║
║ Fraud Slash       │ 10 NEURO                │ 50% to reporter, 50% burned      ║
║ Validator Slash   │ 20 NEURO (2x)           │ 100% burned                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

