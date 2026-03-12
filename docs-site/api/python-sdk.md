# Python SDK

High-level Python client for interacting with NeuroShard nodes.

## Installation

```bash
pip install neuroshard-ai
```

Or from source:

```bash
git clone https://github.com/neuroshard-ai/neuroshard
cd neuroshard
pip install -e .
```

## Quick Start

```python
from neuroshard import NeuroNode, NEUROLedger

# Connect to local node (default port is 8000)
node = NeuroNode("http://localhost:8000", api_token="YOUR_TOKEN")

# Check status
status = node.get_status()
print(f"Node: {status.node_id}")
print(f"Role: {status.role}")
print(f"Layers: {status.layers}")
print(f"Peers: {status.peer_count}")

# Run inference
response = node.inference(
    prompt="Explain quantum computing in simple terms.",
    max_tokens=100,
    temperature=0.7
)
print(response.text)

# Check balance
ledger = NEUROLedger(node)
balance = ledger.get_balance()
print(f"Balance: {balance.available} NEURO")
```

## NeuroNode Class

### Constructor

```python
node = NeuroNode(
    url: str,                    # Node URL
    api_token: str = None,       # API token
    timeout: float = 30.0,       # Request timeout
    retry_attempts: int = 3,     # Retry count
    verify_ssl: bool = True      # SSL verification
)
```

### Methods

#### get_status()

Get node status.

```python
status = node.get_status()

# Returns NodeStatus:
# - node_id: str
# - version: str
# - uptime_seconds: int
# - status: str ("running", "syncing", "error")
# - role: str ("driver", "worker", "validator")
# - layers: List[int]
# - peer_count: int
# - training: TrainingStatus
# - resources: ResourceStatus
```

#### get_metrics()

Get performance metrics.

```python
metrics = node.get_metrics()

# Returns Metrics:
# - inference: InferenceMetrics
# - training: TrainingMetrics
# - network: NetworkMetrics
# - rewards: RewardMetrics
```

#### inference()

Run inference request.

```python
response = node.inference(
    prompt: str,                 # Input text
    max_tokens: int = 100,       # Max output tokens
    temperature: float = 1.0,    # Sampling temperature
    top_p: float = 1.0,          # Nucleus sampling
    top_k: int = 50,             # Top-k sampling
    stop: List[str] = None,      # Stop sequences
    stream: bool = False         # Enable streaming
)

# Returns InferenceResponse:
# - id: str
# - text: str
# - tokens_generated: int
# - finish_reason: str
# - usage: TokenUsage
# - cost: Cost
# - timing: Timing
```

#### inference_stream()

Stream inference response.

```python
for chunk in node.inference_stream(
    prompt="Write a poem about AI.",
    max_tokens=100
):
    print(chunk.token, end="", flush=True)
    
# Yields InferenceChunk:
# - token: str
# - index: int
# - logprob: float (optional)
```

#### get_peers()

List connected peers.

```python
peers = node.get_peers()

# Returns List[PeerInfo]:
# - id: str
# - address: str
# - role: str
# - layers: List[int]
# - latency_ms: float
```

#### get_layers()

List assigned layers.

```python
layers = node.get_layers()

# Returns List[LayerInfo]:
# - index: int
# - type: str
# - memory_mb: int
# - status: str
```

#### get_config()

Get node configuration.

```python
config = node.get_config()

# Returns NodeConfig
```

#### update_config()

Update configuration.

```python
node.update_config({
    "training": {
        "batch_size": 16
    }
})
```

## NEUROLedger Class

### Constructor

```python
ledger = NEUROLedger(node: NeuroNode)
```

### Methods

#### get_balance()

Get wallet balance.

```python
balance = ledger.get_balance()

# Returns Balance:
# - address: str
# - available: float
# - staked: float
# - pending: float
# - total: float
```

#### send()

Send NEURO to another address.

```python
tx = ledger.send(
    to: str,            # Recipient address
    amount: float,      # Amount in NEURO
    memo: str = None    # Optional memo
)

# Returns Transaction:
# - id: str
# - from_address: str
# - to_address: str
# - amount: float
# - fee: float
# - status: str
# - timestamp: datetime
```

#### get_transactions()

Get transaction history.

```python
transactions = ledger.get_transactions(
    limit: int = 10,
    offset: int = 0,
    type: str = None    # "reward", "send", "receive", "stake"
)

# Returns List[Transaction]
```

#### stake()

Stake NEURO tokens.

```python
result = ledger.stake(
    amount: float,           # Amount to stake
    duration_days: int       # Lock duration
)

# Returns StakeResult:
# - amount: float
# - duration_days: int
# - start_date: date
# - unlock_date: date
# - multiplier: float
```

#### unstake()

Request unstaking.

```python
result = ledger.unstake(amount: float)

# Returns UnstakeResult:
# - amount: float
# - cooldown_days: int
# - available_date: date
```

#### get_stake_info()

Get staking information.

```python
stake = ledger.get_stake_info()

# Returns StakeInfo:
# - amount: float
# - duration_days: int
# - start_date: date
# - unlock_date: date
# - multiplier: float
# - pending_unstake: float
```

#### get_rewards()

Get reward history.

```python
rewards = ledger.get_rewards(
    start_date: date = None,
    end_date: date = None
)

# Returns RewardSummary:
# - total: float
# - by_day: List[DailyReward]
# - by_type: Dict[str, float]
```

## Async Support

All methods have async versions:

```python
import asyncio
from neuroshard import AsyncNeuroNode, AsyncNEUROLedger

async def main():
    node = AsyncNeuroNode("http://localhost:8000", api_token="YOUR_TOKEN")
    
    # Async status
    status = await node.get_status()
    
    # Async inference
    response = await node.inference("Hello, AI!", max_tokens=50)
    print(response.text)
    
    # Async streaming
    async for chunk in node.inference_stream("Write a story."):
        print(chunk.token, end="")

asyncio.run(main())
```

## Error Handling

```python
from neuroshard import (
    NeuroNode,
    NeuroShardError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
    NodeOfflineError
)

node = NeuroNode("http://localhost:8000", api_token="YOUR_TOKEN")

try:
    response = node.inference("Hello!")
except AuthenticationError:
    print("Invalid API token")
except InsufficientBalanceError as e:
    print(f"Need {e.required} NEURO, have {e.available}")
except RateLimitError as e:
    print(f"Rate limited, retry in {e.retry_after} seconds")
except NodeOfflineError:
    print("Node is offline")
except NeuroShardError as e:
    print(f"Error: {e.message}")
```

## Configuration

### Environment Variables

```bash
export NEUROSHARD_URL="http://localhost:8000"
export NEUROSHARD_TOKEN="your_api_token"
export NEUROSHARD_TIMEOUT="30"
```

```python
from neuroshard import NeuroNode

# Uses environment variables
node = NeuroNode.from_env()
```

### Config File

```yaml
# ~/.neuroshard/config.yaml
url: http://localhost:8000
token: your_api_token
timeout: 30
retry_attempts: 3
```

```python
from neuroshard import NeuroNode

# Uses config file
node = NeuroNode.from_config()
```

## Examples

### Monitor Node

```python
import time
from neuroshard import NeuroNode

node = NeuroNode("http://localhost:8000", api_token="YOUR_TOKEN")

while True:
    metrics = node.get_metrics()
    
    print(f"\r GPU: {metrics.resources.gpu_percent:.1f}% | "
          f"Peers: {metrics.network.peer_count} | "
          f"Rewards: {metrics.rewards.earned_today:.2f} NEURO", end="")
    
    time.sleep(1)
```

### Batch Inference

```python
import asyncio
from neuroshard import AsyncNeuroNode

async def batch_inference(prompts):
    node = AsyncNeuroNode("http://localhost:8000", api_token="YOUR_TOKEN")
    
    # Run all prompts concurrently
    tasks = [node.inference(p, max_tokens=100) for p in prompts]
    responses = await asyncio.gather(*tasks)
    
    return [r.text for r in responses]

prompts = [
    "Explain quantum computing.",
    "What is machine learning?",
    "How does blockchain work?"
]

results = asyncio.run(batch_inference(prompts))
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}")
    print(f"A: {result}\n")
```

### Auto-Stake Rewards

```python
from neuroshard import NeuroNode, NEUROLedger
import schedule
import time

node = NeuroNode("http://localhost:8000", api_token="YOUR_TOKEN")
ledger = NEUROLedger(node)

def auto_stake():
    balance = ledger.get_balance()
    stake_info = ledger.get_stake_info()
    
    # Stake any available balance over 100 NEURO
    stakeable = balance.available - 100  # Keep 100 NEURO liquid
    
    if stakeable > 0:
        result = ledger.stake(
            amount=stakeable,
            duration_days=30
        )
        print(f"Staked {stakeable} NEURO, new multiplier: {result.multiplier}x")

# Run daily
schedule.every().day.at("00:00").do(auto_stake)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Type Definitions

All types are available in `neuroshard.types`:

```python
from neuroshard.types import (
    NodeStatus,
    Metrics,
    InferenceResponse,
    InferenceChunk,
    PeerInfo,
    LayerInfo,
    Balance,
    Transaction,
    StakeInfo,
    RewardSummary
)
```

## Next Steps

- [NeuroNode Class](/api/neuronode-class) — Detailed class reference
- [NEUROLedger Class](/api/ledger-class) — Ledger API reference
- [Running a Node](/guide/running-a-node) — Start your node

