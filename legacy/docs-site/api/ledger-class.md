# NEUROLedger Class

Complete reference for the NEUROLedger Python class.

## Class Definition

```python
class NEUROLedger:
    """
    Client for NEURO token operations.
    
    Provides methods for:
    - Balance queries
    - Token transfers
    - Staking management
    - Transaction history
    - Reward tracking
    """
    
    def __init__(self, node: NeuroNode):
        """
        Initialize ledger client.
        
        Args:
            node: Connected NeuroNode instance
        """
```

## Constructor

```python
from neuroshard import NeuroNode, NEUROLedger

node = NeuroNode("http://localhost:8000", api_token="TOKEN")
ledger = NEUROLedger(node)
```

## Balance Methods

### get_balance()

```python
def get_balance(self) -> Balance:
    """
    Get wallet balance.
    
    Returns:
        Balance with available, staked, and pending amounts
    """
```

**Example:**
```python
balance = ledger.get_balance()

print(f"Address: {balance.address}")
print(f"Available: {balance.available:.2f} NEURO")
print(f"Staked: {balance.staked:.2f} NEURO")
print(f"Pending: {balance.pending:.2f} NEURO")
print(f"Total: {balance.total:.2f} NEURO")
```

**Balance Type:**
```python
@dataclass
class Balance:
    address: str
    available: float
    staked: float
    pending: float
    total: float
```

## Transfer Methods

### send()

```python
def send(
    self,
    to: str,
    amount: float,
    memo: Optional[str] = None
) -> Transaction:
    """
    Send NEURO to another address.
    
    Args:
        to: Recipient address
        amount: Amount in NEURO
        memo: Optional transaction memo
    
    Returns:
        Transaction details
    
    Raises:
        InsufficientBalanceError: If balance too low
        ValidationError: If invalid address
    """
```

**Example:**
```python
tx = ledger.send(
    to="0xabcd1234...",
    amount=100.0,
    memo="Payment for services"
)

print(f"Transaction ID: {tx.id}")
print(f"Status: {tx.status}")
print(f"Fee: {tx.fee:.4f} NEURO")
```

**Transaction Type:**
```python
@dataclass
class Transaction:
    id: str
    type: str  # "send", "receive", "reward", "stake", "unstake"
    from_address: Optional[str]
    to_address: Optional[str]
    amount: float
    fee: float
    memo: Optional[str]
    status: str  # "pending", "confirmed", "failed"
    timestamp: datetime
    details: Optional[Dict[str, Any]]
```

### get_transactions()

```python
def get_transactions(
    self,
    limit: int = 10,
    offset: int = 0,
    type: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> TransactionList:
    """
    Get transaction history.
    
    Args:
        limit: Maximum transactions to return
        offset: Pagination offset
        type: Filter by type ("send", "receive", "reward", "stake")
        start_date: Filter from date
        end_date: Filter to date
    
    Returns:
        List of transactions with total count
    """
```

**Example:**
```python
# Get recent transactions
transactions = ledger.get_transactions(limit=20)

for tx in transactions.items:
    amount_str = f"+{tx.amount:.2f}" if tx.amount > 0 else f"{tx.amount:.2f}"
    print(f"{tx.timestamp}: {tx.type:8} {amount_str:>12} NEURO")

print(f"\nTotal transactions: {transactions.total}")

# Filter by type
rewards = ledger.get_transactions(type="reward", limit=100)
print(f"Reward transactions: {rewards.total}")
```

**TransactionList Type:**
```python
@dataclass
class TransactionList:
    items: List[Transaction]
    total: int
    limit: int
    offset: int
```

## Staking Methods

### stake()

```python
def stake(
    self,
    amount: float,
    duration_days: int
) -> StakeResult:
    """
    Stake NEURO tokens.
    
    Args:
        amount: Amount to stake (min 1000 NEURO)
        duration_days: Lock duration (min 7, max 365)
    
    Returns:
        Stake result with new multiplier
    
    Raises:
        InsufficientBalanceError: If balance too low
        ValidationError: If invalid amount or duration
    """
```

**Example:**
```python
result = ledger.stake(
    amount=10000.0,
    duration_days=30
)

print(f"Staked: {result.amount:.2f} NEURO")
print(f"Duration: {result.duration_days} days")
print(f"Unlock date: {result.unlock_date}")
print(f"Multiplier: {result.multiplier:.2f}x")
```

**StakeResult Type:**
```python
@dataclass
class StakeResult:
    amount: float
    duration_days: int
    start_date: date
    unlock_date: date
    multiplier: float
    new_balance: Balance
```

### unstake()

```python
def unstake(self, amount: float) -> UnstakeResult:
    """
    Request unstaking.
    
    Note: Unstaking has a 7-day cooldown period.
    
    Args:
        amount: Amount to unstake
    
    Returns:
        Unstake result with availability date
    
    Raises:
        ValidationError: If amount exceeds staked balance
    """
```

**Example:**
```python
result = ledger.unstake(amount=5000.0)

print(f"Unstaking: {result.amount:.2f} NEURO")
print(f"Cooldown: {result.cooldown_days} days")
print(f"Available: {result.available_date}")
```

**UnstakeResult Type:**
```python
@dataclass
class UnstakeResult:
    amount: float
    cooldown_days: int
    available_date: date
    remaining_stake: float
```

### get_stake_info()

```python
def get_stake_info(self) -> StakeInfo:
    """
    Get current staking information.
    
    Returns:
        Stake details including multiplier
    """
```

**Example:**
```python
stake = ledger.get_stake_info()

if stake.amount > 0:
    print(f"Staked: {stake.amount:.2f} NEURO")
    print(f"Duration: {stake.duration_days} days")
    print(f"Start: {stake.start_date}")
    print(f"Unlock: {stake.unlock_date}")
    print(f"Days remaining: {stake.days_remaining}")
    print(f"Multiplier: {stake.multiplier:.2f}x")
    
    if stake.pending_unstake > 0:
        print(f"Pending unstake: {stake.pending_unstake:.2f} NEURO")
else:
    print("No active stake")
```

**StakeInfo Type:**
```python
@dataclass
class StakeInfo:
    amount: float
    duration_days: int
    start_date: Optional[date]
    unlock_date: Optional[date]
    days_remaining: int
    multiplier: float
    pending_unstake: float
    pending_available_date: Optional[date]
```

### add_stake()

```python
def add_stake(self, amount: float) -> StakeResult:
    """
    Add to existing stake.
    
    New amount inherits remaining lock duration.
    
    Args:
        amount: Amount to add
    
    Returns:
        Updated stake result
    """
```

**Example:**
```python
result = ledger.add_stake(amount=5000.0)
print(f"New total stake: {result.amount:.2f} NEURO")
print(f"New multiplier: {result.multiplier:.2f}x")
```

### extend_stake()

```python
def extend_stake(self, additional_days: int) -> StakeResult:
    """
    Extend stake duration.
    
    Args:
        additional_days: Days to add (total max 365)
    
    Returns:
        Updated stake result
    """
```

**Example:**
```python
result = ledger.extend_stake(additional_days=60)
print(f"New unlock date: {result.unlock_date}")
print(f"New multiplier: {result.multiplier:.2f}x")
```

## Reward Methods

### get_rewards()

```python
def get_rewards(
    self,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> RewardSummary:
    """
    Get reward history and statistics.
    
    Args:
        start_date: Start of period (default: 30 days ago)
        end_date: End of period (default: today)
    
    Returns:
        Reward summary with daily breakdown
    """
```

**Example:**
```python
from datetime import date, timedelta

# Last 30 days
rewards = ledger.get_rewards()

print(f"Total: {rewards.total:.2f} NEURO")

# By day
for day in rewards.by_day[-7:]:  # Last 7 days
    print(f"{day.date}: {day.amount:.2f} NEURO ({day.proofs} proofs)")

# By type
print("\nBy type:")
for reward_type, amount in rewards.by_type.items():
    print(f"  {reward_type}: {amount:.2f} NEURO")

# Custom period
last_month = ledger.get_rewards(
    start_date=date.today() - timedelta(days=30),
    end_date=date.today()
)
```

**RewardSummary Type:**
```python
@dataclass
class RewardSummary:
    total: float
    by_day: List[DailyReward]
    by_type: Dict[str, float]

@dataclass
class DailyReward:
    date: date
    amount: float
    proofs: int
```

### get_reward_rate()

```python
def get_reward_rate(self) -> RewardRate:
    """
    Get current reward earning rate.
    
    Returns:
        Current rate with multipliers
    """
```

**Example:**
```python
rate = ledger.get_reward_rate()

print(f"Base rate: {rate.base_rate:.4f} NEURO/proof")
print(f"Role: {rate.role} ({rate.role_multiplier:.1f}x)")
print(f"Stake: {rate.stake_multiplier:.2f}x")
print(f"Effective rate: {rate.effective_rate:.4f} NEURO/proof")
print(f"Estimated hourly: {rate.estimated_hourly:.2f} NEURO")
print(f"Estimated daily: {rate.estimated_daily:.2f} NEURO")
```

**RewardRate Type:**
```python
@dataclass
class RewardRate:
    base_rate: float
    role: str
    role_multiplier: float
    stake_multiplier: float
    reputation_multiplier: float
    effective_rate: float
    estimated_hourly: float
    estimated_daily: float
```

## Utility Methods

### estimate_stake_multiplier()

```python
def estimate_stake_multiplier(
    self,
    amount: float,
    duration_days: int
) -> float:
    """
    Estimate multiplier for stake parameters.
    
    Args:
        amount: Stake amount
        duration_days: Lock duration
    
    Returns:
        Estimated multiplier
    """
```

**Example:**
```python
# Compare options
for amount in [1000, 5000, 10000, 50000]:
    for days in [7, 30, 90, 180, 365]:
        mult = ledger.estimate_stake_multiplier(amount, days)
        print(f"{amount:>6} NEURO × {days:>3}d = {mult:.2f}x")
```

### validate_address()

```python
def validate_address(self, address: str) -> bool:
    """
    Validate NEURO address format.
    
    Args:
        address: Address to validate
    
    Returns:
        True if valid
    """
```

**Example:**
```python
if ledger.validate_address("0xabcd..."):
    print("Valid address")
else:
    print("Invalid address format")
```

## Error Handling

```python
from neuroshard.exceptions import (
    InsufficientBalanceError,
    InvalidAddressError,
    StakeLockedError,
    MinimumStakeError,
)

try:
    tx = ledger.send(to="0xabc...", amount=100)
except InsufficientBalanceError as e:
    print(f"Need {e.required:.2f} NEURO, have {e.available:.2f}")
except InvalidAddressError:
    print("Invalid recipient address")

try:
    result = ledger.stake(amount=500, duration_days=30)
except MinimumStakeError as e:
    print(f"Minimum stake is {e.minimum:.2f} NEURO")

try:
    result = ledger.unstake(amount=5000)
except StakeLockedError as e:
    print(f"Stake locked until {e.unlock_date}")
```

## Async Version

```python
from neuroshard import AsyncNeuroNode, AsyncNEUROLedger

async def main():
    node = AsyncNeuroNode("http://localhost:8000", api_token="TOKEN")
    ledger = AsyncNEUROLedger(node)
    
    balance = await ledger.get_balance()
    print(f"Balance: {balance.available} NEURO")
    
    rewards = await ledger.get_rewards()
    print(f"Total rewards: {rewards.total} NEURO")

import asyncio
asyncio.run(main())
```

## Examples

### Auto-Compound Rewards

```python
from neuroshard import NeuroNode, NEUROLedger
import time

node = NeuroNode("http://localhost:8000", api_token="TOKEN")
ledger = NEUROLedger(node)

def auto_compound():
    """Add available balance to stake."""
    balance = ledger.get_balance()
    stake_info = ledger.get_stake_info()
    
    # Keep 100 NEURO liquid
    stakeable = balance.available - 100
    
    if stakeable >= 100 and stake_info.amount > 0:
        result = ledger.add_stake(stakeable)
        print(f"Added {stakeable:.2f} NEURO to stake")
        print(f"New multiplier: {result.multiplier:.2f}x")
        return True
    return False

# Run periodically
while True:
    auto_compound()
    time.sleep(3600)  # Check hourly
```

### Reward Dashboard

```python
from neuroshard import NeuroNode, NEUROLedger
from datetime import date, timedelta

node = NeuroNode("http://localhost:8000", api_token="TOKEN")
ledger = NEUROLedger(node)

def show_dashboard():
    balance = ledger.get_balance()
    stake = ledger.get_stake_info()
    rate = ledger.get_reward_rate()
    rewards = ledger.get_rewards()
    
    print("=" * 50)
    print("          NEURO DASHBOARD")
    print("=" * 50)
    print(f"Balance:    {balance.available:>12.2f} NEURO")
    print(f"Staked:     {balance.staked:>12.2f} NEURO")
    print(f"Pending:    {balance.pending:>12.2f} NEURO")
    print(f"Total:      {balance.total:>12.2f} NEURO")
    print("-" * 50)
    print(f"Multiplier: {stake.multiplier:>12.2f}x")
    print(f"Role:       {rate.role:>12}")
    print(f"Rate:       {rate.effective_rate:>12.4f} NEURO/proof")
    print("-" * 50)
    print(f"Today:      {rewards.by_day[-1].amount if rewards.by_day else 0:>12.2f} NEURO")
    print(f"This week:  {sum(d.amount for d in rewards.by_day[-7:]):>12.2f} NEURO")
    print(f"This month: {rewards.total:>12.2f} NEURO")
    print("=" * 50)

show_dashboard()
```

## Next Steps

- [NeuroNode Class](/api/neuronode-class) — Node operations
- [Staking Guide](/economics/staking) — Staking strategies
- [Reward System](/economics/rewards) — How rewards work

