# HTTP Endpoints

Complete reference for the NeuroShard HTTP REST API.

## Base URL

```
http://localhost:8000/api/v1
```

The default port is 8000. For remote access, use the node's public IP/domain.

::: tip Port Configuration
Set a custom port with `neuroshard --port 9000`. The gRPC port is always HTTP port + 1000.
:::

## Node Status

### GET /status

Get current node status.

**Request:**
```bash
curl http://localhost:8000/api/v1/status
```

**Response:**
```json
{
  "node_id": "node_abc123",
  "version": "0.1.0",
  "uptime_seconds": 86400,
  "status": "running",
  "role": "worker",
  "layers": [4, 5, 6, 7],
  "peer_count": 12,
  "training": {
    "enabled": true,
    "epoch": 1542,
    "step": 234567,
    "loss": 2.34
  },
  "resources": {
    "gpu_memory_used": 7234567890,
    "gpu_memory_total": 8589934592,
    "cpu_percent": 45.2,
    "ram_used": 4294967296
  }
}
```

### GET /metrics

Get performance metrics.

**Request:**
```bash
curl http://localhost:8000/api/v1/metrics
```

**Response:**
```json
{
  "timestamp": "2024-12-04T12:00:00Z",
  "inference": {
    "requests_total": 15234,
    "requests_per_minute": 42.5,
    "avg_latency_ms": 234.5,
    "p99_latency_ms": 456.7,
    "tokens_generated": 1234567
  },
  "training": {
    "steps_total": 234567,
    "steps_per_hour": 1234,
    "gradients_submitted": 4567,
    "gradients_accepted": 4500
  },
  "network": {
    "bytes_sent": 12345678901,
    "bytes_received": 23456789012,
    "active_connections": 15,
    "rpc_calls": 456789
  },
  "rewards": {
    "earned_today": 2847.32,
    "earned_total": 234567.89,
    "pending": 123.45
  }
}
```

### GET /health

Health check endpoint.

**Request:**
```bash
curl http://localhost:8000/api/v1/health
```

**Response:**
```json
{
  "healthy": true,
  "checks": {
    "gpu": "ok",
    "network": "ok",
    "storage": "ok",
    "peers": "ok"
  }
}
```

## Inference

### POST /inference

Submit an inference request.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms.",
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
  }'
```

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Input text |
| `max_tokens` | int | No | 100 | Maximum tokens to generate |
| `temperature` | float | No | 1.0 | Sampling temperature (0-2) |
| `top_p` | float | No | 1.0 | Nucleus sampling threshold |
| `top_k` | int | No | 50 | Top-k sampling |
| `stream` | bool | No | false | Enable streaming response |
| `stop` | array | No | [] | Stop sequences |

**Response:**
```json
{
  "id": "inf_abc123",
  "text": "Quantum computing is a type of computation that...",
  "tokens_generated": 156,
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 156,
    "total_tokens": 168
  },
  "cost": {
    "amount": 0.0168,
    "currency": "NEURO"
  },
  "timing": {
    "queue_ms": 12,
    "inference_ms": 2345,
    "total_ms": 2357
  }
}
```

### POST /inference (Streaming)

Stream inference response.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a poem about AI.",
    "max_tokens": 100,
    "stream": true
  }'
```

**Response (Server-Sent Events):**
```
data: {"token": "In", "index": 0}

data: {"token": " silicon", "index": 1}

data: {"token": " dreams", "index": 2}

...

data: {"token": "[DONE]", "finish_reason": "stop", "usage": {...}}
```

## Wallet

### GET /wallet/balance

Get wallet balance.

**Request:**
```bash
curl http://localhost:8000/api/v1/wallet/balance \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "address": "0x1234...abcd",
  "balances": {
    "available": 12345.67,
    "staked": 10000.00,
    "pending": 234.56,
    "total": 22580.23
  },
  "staking": {
    "amount": 10000.00,
    "duration_days": 30,
    "start_date": "2024-11-04",
    "unlock_date": "2024-12-04",
    "multiplier": 1.50
  }
}
```

### POST /wallet/send

Send NEURO to another address.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/wallet/send \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "to": "0xabcd...1234",
    "amount": 100.0,
    "memo": "Payment for services"
  }'
```

**Response:**
```json
{
  "transaction_id": "tx_abc123",
  "from": "0x1234...abcd",
  "to": "0xabcd...1234",
  "amount": 100.0,
  "fee": 0.1,
  "memo": "Payment for services",
  "status": "confirmed",
  "timestamp": "2024-12-04T12:00:00Z"
}
```

### GET /wallet/transactions

Get transaction history.

**Request:**
```bash
curl "http://localhost:8000/api/v1/wallet/transactions?limit=10&offset=0" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "transactions": [
    {
      "id": "tx_abc123",
      "type": "reward",
      "amount": 123.45,
      "timestamp": "2024-12-04T12:00:00Z",
      "details": {
        "proof_id": "proof_xyz",
        "role": "worker"
      }
    },
    {
      "id": "tx_def456",
      "type": "send",
      "amount": -100.0,
      "to": "0xabcd...1234",
      "timestamp": "2024-12-04T11:00:00Z"
    }
  ],
  "total": 156,
  "limit": 10,
  "offset": 0
}
```

### POST /wallet/stake

Stake NEURO tokens.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/wallet/stake \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 10000.0,
    "duration_days": 30
  }'
```

**Response:**
```json
{
  "success": true,
  "stake": {
    "amount": 10000.0,
    "duration_days": 30,
    "start_date": "2024-12-04",
    "unlock_date": "2025-01-03",
    "multiplier": 1.50
  },
  "new_balance": {
    "available": 2345.67,
    "staked": 10000.0
  }
}
```

### POST /wallet/unstake

Request unstaking.

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/wallet/unstake \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 5000.0
  }'
```

**Response:**
```json
{
  "success": true,
  "unstake": {
    "amount": 5000.0,
    "cooldown_days": 7,
    "available_date": "2024-12-11"
  }
}
```

## Network

### GET /peers

List connected peers.

**Request:**
```bash
curl http://localhost:8000/api/v1/peers \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "peers": [
    {
      "id": "peer_abc123",
      "address": "192.168.1.100:9000",
      "role": "worker",
      "layers": [0, 1, 2, 3],
      "latency_ms": 23,
      "connected_since": "2024-12-04T10:00:00Z"
    },
    {
      "id": "peer_def456",
      "address": "192.168.1.101:9000",
      "role": "validator",
      "layers": [28, 29, 30, 31],
      "latency_ms": 45,
      "connected_since": "2024-12-04T09:00:00Z"
    }
  ],
  "total": 12
}
```

### GET /layers

List assigned layers.

**Request:**
```bash
curl http://localhost:8000/api/v1/layers \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "layers": [
    {
      "index": 4,
      "type": "transformer",
      "memory_mb": 512,
      "status": "active"
    },
    {
      "index": 5,
      "type": "transformer",
      "memory_mb": 512,
      "status": "active"
    }
  ],
  "total_layers": 32,
  "my_layer_count": 4
}
```

### GET /network

Get network statistics.

**Request:**
```bash
curl http://localhost:8000/api/v1/network
```

**Response:**
```json
{
  "nodes": {
    "total": 156,
    "drivers": 12,
    "workers": 132,
    "validators": 12
  },
  "architecture": {
    "layers": 32,
    "hidden_dim": 3072,
    "parameters": "9.2B"
  },
  "capacity": {
    "total_memory_gb": 1234,
    "used_memory_gb": 890
  },
  "training": {
    "epoch": 1542,
    "global_step": 234567,
    "loss": 2.34
  }
}
```

## Configuration

### GET /config

Get node configuration.

**Request:**
```bash
curl http://localhost:8000/api/v1/config \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "node_id": "node_abc123",
  "port": 8000,
  "grpc_port": 9000,
  "tracker_url": "wss://tracker.neuroshard.com:8765",
  "training": {
    "enabled": true,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "diloco_steps": 500
  },
  "resources": {
    "max_memory_gb": 8,
    "cpu_threads": 4
  }
}
```

### PATCH /config

Update configuration.

**Request:**
```bash
curl -X PATCH http://localhost:8000/api/v1/config \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "training": {
      "batch_size": 16
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "updated": ["training.batch_size"],
  "restart_required": false
}
```

## Training Verification

### GET /api/training/global

Get global training verification status. This endpoint answers: "Is the distributed training actually working?"

**Request:**
```bash
curl http://localhost:8000/api/training/global
```

**Response:**
```json
{
  "global_avg_loss": 0.1234,
  "global_min_loss": 0.0456,
  "hash_agreement_rate": 1.0,
  "total_nodes_training": 3,
  "successful_syncs": 15,
  "failed_syncs": 0,
  "sync_success_rate": 1.0,
  "global_steps": 1500,
  "global_tokens": 12800000,
  "data_shards_covered": [0, 1, 2, 5, 8],
  "is_converging": true,
  "training_verified": true,
  "loss_trend": "improving",
  "local": {
    "node_id": "abc123def456...",
    "training_rounds": 500,
    "current_loss": 0.1234,
    "is_training": true
  },
  "diloco": {
    "enabled": true,
    "inner_step_count": 123,
    "inner_steps_total": 500,
    "progress": 0.246,
    "outer_step_count": 3
  }
}
```

**Key Metrics:**

| Field | Description | Good Value |
|-------|-------------|------------|
| `training_verified` | Confirmed model is improving | `true` |
| `is_converging` | Network trending toward same model | `true` |
| `hash_agreement_rate` | % of nodes with same model hash | `1.0` (100%) |
| `sync_success_rate` | % of gradient syncs that succeeded | `> 0.5` |
| `loss_trend` | Direction of loss over time | `"improving"` |

::: warning Hash Agreement
If `hash_agreement_rate` < 100%, nodes have diverged and training is NOT coordinated. Check gradient synchronization.
:::

### GET /api/training/verify

Quick verification endpoint - simple yes/no answer.

**Request:**
```bash
curl http://localhost:8000/api/training/verify
```

**Response (Success):**
```json
{
  "is_working": true,
  "reason": "Training verified! Loss is decreasing and network is converging.",
  "metrics": {
    "loss_trend": "improving",
    "hash_agreement": "100.0%",
    "global_loss": 0.1234
  }
}
```

**Response (Problem):**
```json
{
  "is_working": false,
  "reason": "Network NOT converging! Only 50.0% hash agreement.",
  "action": "Nodes have diverged. Check gradient sync is working."
}
```

## Proofs

### GET /proofs

Get proof history.

**Request:**
```bash
curl "http://localhost:8000/api/v1/proofs?limit=10" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "proofs": [
    {
      "id": "proof_abc123",
      "type": "forward",
      "layer": 5,
      "timestamp": "2024-12-04T12:00:00Z",
      "verified": true,
      "reward": 1.23
    }
  ],
  "stats": {
    "total_submitted": 45678,
    "total_verified": 45500,
    "total_reward": 56789.01
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "Additional context"
    }
  }
}
```

### Common Errors

**401 Unauthorized:**
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired API token"
  }
}
```

**429 Rate Limited:**
```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Too many requests",
    "details": {
      "retry_after": 60
    }
  }
}
```

**402 Payment Required:**
```json
{
  "error": {
    "code": "INSUFFICIENT_BALANCE",
    "message": "Not enough NEURO for this operation",
    "details": {
      "required": 100.0,
      "available": 50.0
    }
  }
}
```

## Next Steps

- [gRPC Services](/api/grpc-services) — Node-to-node protocol
- [Python SDK](/api/python-sdk) — Python client

