# NeuroNode Class

Complete reference for the NeuroNode Python class.

## Class Definition

```python
class NeuroNode:
    """
    Client for interacting with a NeuroShard node.
    
    Provides methods for:
    - Node status and metrics
    - Inference requests
    - Peer management
    - Configuration
    """
    
    def __init__(
        self,
        url: str,
        api_token: Optional[str] = None,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        verify_ssl: bool = True
    ):
        """
        Initialize NeuroNode client.
        
        Args:
            url: Node URL (e.g., "http://localhost:8000")
            api_token: API authentication token
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            verify_ssl: Whether to verify SSL certificates
        """
```

## Constructor Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | Required | Node URL |
| `api_token` | str | None | API token |
| `timeout` | float | 30.0 | Request timeout |
| `retry_attempts` | int | 3 | Retry count |
| `verify_ssl` | bool | True | SSL verification |

## Factory Methods

### from_env()

Create from environment variables.

```python
# Set environment variables
# NEUROSHARD_URL=http://localhost:8000
# NEUROSHARD_TOKEN=your_token

node = NeuroNode.from_env()
```

### from_config()

Create from config file.

```python
# Uses ~/.neuroshard/config.yaml
node = NeuroNode.from_config()

# Or specify path
node = NeuroNode.from_config("/path/to/config.yaml")
```

## Status Methods

### get_status()

```python
def get_status(self) -> NodeStatus:
    """
    Get current node status.
    
    Returns:
        NodeStatus object with node information
    """
```

**Example:**
```python
status = node.get_status()

print(f"Node ID: {status.node_id}")
print(f"Version: {status.version}")
print(f"Uptime: {status.uptime_seconds} seconds")
print(f"Status: {status.status}")
print(f"Role: {status.role}")
print(f"Layers: {status.layers}")
print(f"Peers: {status.peer_count}")

# Training info
if status.training.enabled:
    print(f"Epoch: {status.training.epoch}")
    print(f"Step: {status.training.step}")
    print(f"Loss: {status.training.loss:.4f}")

# Resources
print(f"GPU Memory: {status.resources.gpu_memory_used / 1e9:.1f}GB / {status.resources.gpu_memory_total / 1e9:.1f}GB")
print(f"CPU: {status.resources.cpu_percent:.1f}%")
```

**NodeStatus Type:**
```python
@dataclass
class NodeStatus:
    node_id: str
    version: str
    uptime_seconds: int
    status: str  # "running", "syncing", "error"
    role: str    # "driver", "worker", "validator"
    layers: List[int]
    peer_count: int
    training: TrainingStatus
    resources: ResourceStatus

@dataclass
class TrainingStatus:
    enabled: bool
    epoch: int
    step: int
    loss: float

@dataclass
class ResourceStatus:
    gpu_memory_used: int
    gpu_memory_total: int
    cpu_percent: float
    ram_used: int
```

### get_metrics()

```python
def get_metrics(self) -> Metrics:
    """
    Get performance metrics.
    
    Returns:
        Metrics object with detailed statistics
    """
```

**Example:**
```python
metrics = node.get_metrics()

# Inference metrics
print(f"Total requests: {metrics.inference.requests_total}")
print(f"Requests/min: {metrics.inference.requests_per_minute:.1f}")
print(f"Avg latency: {metrics.inference.avg_latency_ms:.1f}ms")
print(f"P99 latency: {metrics.inference.p99_latency_ms:.1f}ms")

# Training metrics
print(f"Steps: {metrics.training.steps_total}")
print(f"Steps/hour: {metrics.training.steps_per_hour}")

# Network metrics
print(f"Sent: {metrics.network.bytes_sent / 1e9:.2f}GB")
print(f"Received: {metrics.network.bytes_received / 1e9:.2f}GB")

# Rewards
print(f"Today: {metrics.rewards.earned_today:.2f} NEURO")
print(f"Total: {metrics.rewards.earned_total:.2f} NEURO")
```

### get_health()

```python
def get_health(self) -> HealthStatus:
    """
    Check node health.
    
    Returns:
        HealthStatus with component checks
    """
```

**Example:**
```python
health = node.get_health()

if health.healthy:
    print("✓ Node is healthy")
else:
    print("✗ Node has issues")
    
for component, status in health.checks.items():
    icon = "✓" if status == "ok" else "✗"
    print(f"  {icon} {component}: {status}")
```

## Inference Methods

### inference()

```python
def inference(
    self,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    stop: Optional[List[str]] = None,
    stream: bool = False
) -> InferenceResponse:
    """
    Run inference request.
    
    Args:
        prompt: Input text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0-2)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling
        stop: Stop sequences
        stream: Enable streaming (use inference_stream instead)
    
    Returns:
        InferenceResponse with generated text
    """
```

**Example:**
```python
response = node.inference(
    prompt="What is the meaning of life?",
    max_tokens=200,
    temperature=0.7,
    stop=["\n\n"]
)

print(f"Response: {response.text}")
print(f"Tokens: {response.tokens_generated}")
print(f"Finish reason: {response.finish_reason}")
print(f"Cost: {response.cost.amount:.4f} {response.cost.currency}")
print(f"Latency: {response.timing.total_ms}ms")
```

**InferenceResponse Type:**
```python
@dataclass
class InferenceResponse:
    id: str
    text: str
    tokens_generated: int
    finish_reason: str  # "stop", "length", "error"
    usage: TokenUsage
    cost: Cost
    timing: Timing

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass
class Cost:
    amount: float
    currency: str  # "NEURO"

@dataclass
class Timing:
    queue_ms: int
    inference_ms: int
    total_ms: int
```

### inference_stream()

```python
def inference_stream(
    self,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    stop: Optional[List[str]] = None
) -> Iterator[InferenceChunk]:
    """
    Stream inference response.
    
    Args:
        Same as inference()
    
    Yields:
        InferenceChunk for each token
    """
```

**Example:**
```python
print("Response: ", end="")

for chunk in node.inference_stream("Write a haiku about AI."):
    print(chunk.token, end="", flush=True)
    
print()  # Newline at end
```

**InferenceChunk Type:**
```python
@dataclass
class InferenceChunk:
    token: str
    index: int
    logprob: Optional[float] = None
    finish_reason: Optional[str] = None
```

## Network Methods

### get_peers()

```python
def get_peers(self) -> List[PeerInfo]:
    """
    List connected peers.
    
    Returns:
        List of peer information
    """
```

**Example:**
```python
peers = node.get_peers()

for peer in peers:
    print(f"{peer.id}: {peer.address}")
    print(f"  Role: {peer.role}")
    print(f"  Layers: {peer.layers}")
    print(f"  Latency: {peer.latency_ms}ms")
```

**PeerInfo Type:**
```python
@dataclass
class PeerInfo:
    id: str
    address: str
    role: str
    layers: List[int]
    latency_ms: float
    connected_since: datetime
```

### get_layers()

```python
def get_layers(self) -> List[LayerInfo]:
    """
    List assigned layers.
    
    Returns:
        List of layer information
    """
```

**Example:**
```python
layers = node.get_layers()

for layer in layers:
    print(f"Layer {layer.index}: {layer.type}")
    print(f"  Memory: {layer.memory_mb}MB")
    print(f"  Status: {layer.status}")
```

### get_network()

```python
def get_network(self) -> NetworkInfo:
    """
    Get network statistics.
    
    Returns:
        Network-wide information
    """
```

**Example:**
```python
network = node.get_network()

print(f"Total nodes: {network.nodes.total}")
print(f"  Drivers: {network.nodes.drivers}")
print(f"  Workers: {network.nodes.workers}")
print(f"  Validators: {network.nodes.validators}")

print(f"Architecture: {network.architecture.layers}L × {network.architecture.hidden_dim}H")
print(f"Parameters: {network.architecture.parameters}")

print(f"Training epoch: {network.training.epoch}")
print(f"Loss: {network.training.loss:.4f}")
```

## Configuration Methods

### get_config()

```python
def get_config(self) -> NodeConfig:
    """
    Get node configuration.
    
    Returns:
        Current configuration
    """
```

### update_config()

```python
def update_config(self, updates: Dict[str, Any]) -> ConfigUpdateResult:
    """
    Update configuration.
    
    Args:
        updates: Configuration updates (nested dict)
    
    Returns:
        Result with updated fields and restart requirement
    """
```

**Example:**
```python
result = node.update_config({
    "training": {
        "batch_size": 16,
        "learning_rate": 0.0002
    }
})

print(f"Updated: {result.updated}")
if result.restart_required:
    print("Restart required for changes to take effect")
```

## Proof Methods

### get_proofs()

```python
def get_proofs(
    self,
    limit: int = 10,
    offset: int = 0
) -> ProofHistory:
    """
    Get proof submission history.
    
    Returns:
        Proof history with statistics
    """
```

**Example:**
```python
history = node.get_proofs(limit=10)

for proof in history.proofs:
    status = "✓" if proof.verified else "✗"
    print(f"{status} {proof.type} layer {proof.layer}: {proof.reward:.2f} NEURO")

print(f"\nTotal submitted: {history.stats.total_submitted}")
print(f"Total verified: {history.stats.total_verified}")
print(f"Total reward: {history.stats.total_reward:.2f} NEURO")
```

## Error Handling

```python
from neuroshard.exceptions import (
    NeuroShardError,       # Base exception
    AuthenticationError,   # Invalid token
    ConnectionError,       # Can't connect
    TimeoutError,          # Request timeout
    RateLimitError,        # Too many requests
    InsufficientBalanceError,  # Not enough NEURO
    ValidationError,       # Invalid request
)

try:
    response = node.inference("Hello")
except AuthenticationError:
    print("Check your API token")
except ConnectionError:
    print("Node is offline")
except TimeoutError:
    print("Request timed out")
except RateLimitError as e:
    print(f"Rate limited, retry in {e.retry_after}s")
except NeuroShardError as e:
    print(f"Error: {e.code}: {e.message}")
```

## Context Manager

```python
from neuroshard import NeuroNode

# Auto-close on exit
with NeuroNode("http://localhost:8000", api_token="TOKEN") as node:
    status = node.get_status()
    response = node.inference("Hello!")
```

## Thread Safety

NeuroNode is thread-safe. You can share one instance across threads:

```python
from concurrent.futures import ThreadPoolExecutor
from neuroshard import NeuroNode

node = NeuroNode("http://localhost:8000", api_token="TOKEN")

def process_prompt(prompt):
    return node.inference(prompt, max_tokens=50)

prompts = ["Question 1", "Question 2", "Question 3"]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_prompt, prompts))
```

## Async Version

See [AsyncNeuroNode](/api/python-sdk#async-support) for async support.

## Next Steps

- [NEUROLedger Class](/api/ledger-class) — Wallet operations
- [Python SDK](/api/python-sdk) — SDK overview
- [HTTP Endpoints](/api/http-endpoints) — REST API

