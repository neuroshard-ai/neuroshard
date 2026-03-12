# neuroshard/core/economics/__init__.py
"""
Token economics components for NeuroShard.

- constants: Token economics constants
- ledger: NEUROLedger, PoNWProof
- wallet: Wallet
- market: InferenceMarket
"""

__all__ = [
    'NEURO_DECIMALS',
    'NEURO_TOTAL_SUPPLY',
    'NEUROLedger',
    'PoNWProof',
    'ProofType',
    'Wallet',
    'InferenceMarket',
    'MIN_STAKE_AMOUNT',
    'MAX_STAKE_AMOUNT',
    'MIN_STAKE_DURATION_DAYS',
    'MAX_STAKE_DURATION_DAYS',
    'TRAINING_REWARD_PER_BATCH_PER_LAYER',
    'calculate_stake_multiplier',
    'calculate_training_reward',
    'calculate_role_bonus',
    'is_valid_stake_amount',
    'is_valid_stake_duration',
    'calculate_burn_amount',
    'is_eligible_validator',
]

def __getattr__(name):
    """Lazy loading of submodules."""
    if name in ('NEURO_DECIMALS', 'NEURO_TOTAL_SUPPLY', 'VALIDATOR_MIN_STAKE', 
                'VALIDATOR_MIN_MEMORY_MB', 'TRAINING_REWARD_PER_BATCH_PER_LAYER',
                'DATA_REWARD_PER_SAMPLE', 'INFERENCE_MARKET_TARGET_RESPONSE_TIME',
                'MIN_STAKE_AMOUNT', 'MAX_STAKE_AMOUNT', 
                'MIN_STAKE_DURATION_DAYS', 'MAX_STAKE_DURATION_DAYS',
                'calculate_stake_multiplier', 'calculate_training_reward',
                'calculate_role_bonus', 'is_valid_stake_amount', 
                'is_valid_stake_duration', 'calculate_burn_amount', 
                'is_eligible_validator'):
        from neuroshard.core.economics import constants
        return getattr(constants, name)
    elif name in ('NEUROLedger', 'PoNWProof', 'ProofType', 'LedgerEntry', 'sign_proof'):
        from neuroshard.core.economics import ledger
        return getattr(ledger, name)
    elif name == 'Wallet':
        from neuroshard.core.economics.wallet import Wallet
        return Wallet
    elif name in ('InferenceMarket', 'RequestStatus'):
        from neuroshard.core.economics import market
        return getattr(market, name)
    raise AttributeError(f"module 'neuroshard.core.economics' has no attribute '{name}'")
