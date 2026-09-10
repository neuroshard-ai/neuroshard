# gRPC Services

Low-level gRPC API for node-to-node communication.

## Protocol Buffers

### Service Definition

```protobuf
syntax = "proto3";

package neuroshard;

service NeuroShardService {
  // Inference
  rpc StreamInference (stream InferenceRequest) returns (stream InferenceResponse);
  rpc UnaryInference (InferenceRequest) returns (InferenceResponse);
  
  // Weights & Training
  rpc GetWeights (WeightRequest) returns (WeightResponse);
  rpc GossipGradient (GossipGradientRequest) returns (GossipGradientResponse);
  rpc GetCheckpoint (GetCheckpointRequest) returns (GetCheckpointResponse);
  rpc GetCheckpointInfo (GetCheckpointInfoRequest) returns (GetCheckpointInfoResponse);
  
  // Proofs & Economics
  rpc GossipProof (GossipProofRequest) returns (GossipProofResponse);
  rpc GossipTransaction (GossipTransactionRequest) returns (GossipTransactionResponse);
  rpc GossipStake (GossipStakeRequest) returns (GossipStakeResponse);
  rpc RequestProofValidation (ProofValidationRequest) returns (ProofValidationResponse);
  rpc GossipValidationVote (ValidationVoteRequest) returns (ValidationVoteResponse);
  
  // Pipeline Parallelism
  rpc PipelineForward (PipelineForwardRequest) returns (PipelineForwardResponse);
  rpc PipelineBackward (PipelineBackwardRequest) returns (PipelineBackwardResponse);
  rpc GetShardInfo (GetShardInfoRequest) returns (GetShardInfoResponse);
  
  // DHT (Peer Discovery)
  rpc DHTPing (DHTPingRequest) returns (DHTPingResponse);
  rpc DHTStore (DHTStoreRequest) returns (DHTStoreResponse);
  rpc DHTFindNode (DHTFindNodeRequest) returns (DHTFindNodeResponse);
  rpc DHTFindValue (DHTFindValueRequest) returns (DHTFindValueResponse);
  
  // Swarm Routing
  rpc SwarmForward (SwarmForwardRequest) returns (SwarmForwardResponse);
  rpc GetSwarmStatus (SwarmStatusRequest) returns (SwarmStatusResponse);
  rpc UpdatePeerCapacity (UpdatePeerCapacityRequest) returns (UpdatePeerCapacityResponse);
}
```

## Connection

### Default Port

The gRPC port is HTTP port + 1000. For the default HTTP port of 8000:

```
localhost:9000
```

### Channel Options

```python
import grpc

channel = grpc.insecure_channel(
    'localhost:9000',  # HTTP port (8000) + 1000
    options=[
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.keepalive_time_ms', 30000),
        ('grpc.keepalive_timeout_ms', 10000),
        ('grpc.keepalive_permit_without_calls', True),
    ]
)
```

::: tip Port Calculation
If your node runs on HTTP port 9000 (via `--port 9000`), the gRPC port will be 10000.
:::

## Pipeline Operations

### PipelineForward

Forward hidden states through this node's layers (for distributed inference).

**Request:**
```protobuf
message PipelineForwardRequest {
  string session_id = 1;
  string request_id = 2;
  
  bytes hidden_states = 3;         // Serialized tensor [batch, seq, hidden]
  repeated int64 hidden_shape = 4;
  bytes attention_mask = 5;
  bytes position_ids = 6;
  
  repeated bytes past_key_values = 7;  // KV cache
  bool use_cache = 8;
  
  int32 source_shard = 9;
  int32 target_shard = 10;
  
  bytes training_labels = 11;      // For training (Driver only)
  string sender_url = 12;          // For backward pass routing
}
```

**Response:**
```protobuf
message PipelineForwardResponse {
  string request_id = 1;
  bool success = 2;
  string error_message = 3;
  
  bytes hidden_states = 4;
  repeated int64 hidden_shape = 5;
  repeated bytes past_key_values = 6;
  
  bool is_final = 7;               // True if this is the last shard
  bytes logits = 8;                // Only if is_final
  repeated int64 logits_shape = 9;
  
  double loss = 10;                // Returned by Validator
}
```

**Python Example:**
```python
import grpc
from protos import neuroshard_pb2, neuroshard_pb2_grpc
from neuroshard.utils.serialization import serialize_tensor, deserialize_tensor

# Connect
channel = grpc.insecure_channel('localhost:9000')
stub = neuroshard_pb2_grpc.NeuroShardServiceStub(channel)

# Prepare request
hidden_states = torch.randn(1, 128, 1024)

request = neuroshard_pb2.PipelineForwardRequest(
    session_id="sess_123",
    request_id="req_456",
    hidden_states=serialize_tensor(hidden_states).encode('utf-8'),
    hidden_shape=[1, 128, 1024],
    target_shard=1,
    use_cache=True,
)

# Call
response = stub.PipelineForward(request)

if response.success:
    output = deserialize_tensor(response.hidden_states.decode('utf-8'))
```

### PipelineBackward

Propagate gradients back to previous node.

**Request:**
```protobuf
message PipelineBackwardRequest {
  string session_id = 1;
  string request_id = 2;
  
  bytes grad_output = 3;           // Gradient w.r.t output
  repeated int64 grad_shape = 4;
  
  int32 target_shard = 5;
}
```

**Response:**
```protobuf
message PipelineBackwardResponse {
  bool success = 1;
  string error_message = 2;
}
```

## Training Operations

### GossipGradient

Exchange gradients for distributed training.

**Request:**
```protobuf
message GossipGradientRequest {
  string node_id = 1;              // Sender node ID
  int32 round_id = 2;              // Training round ID
  string model_hash = 3;           // Model hash for consistency check
  double timestamp = 4;
  
  int32 batch_size = 5;
  double loss = 6;
  
  map<string, bytes> layer_gradients = 7;  // Compressed gradients per layer
  
  string signature = 8;            // Proof signature
  int32 ttl = 9;                   // Time-to-live for forwarding
}
```

**Response:**
```protobuf
message GossipGradientResponse {
  bool accepted = 1;
  string reason = 2;
  int32 current_round = 3;         // Receiver's current round (for sync)
}
```

**Python Example:**
```python
# Compress gradients
compressed_grads = {}
for name, grad in pseudo_gradients.items():
    compressed_grads[name] = lz4.frame.compress(grad.numpy().tobytes())

request = neuroshard_pb2.GossipGradientRequest(
    node_id=my_node_id,
    round_id=current_round,
    model_hash=model_hash,
    timestamp=time.time(),
    batch_size=8,
    loss=current_loss,
    layer_gradients=compressed_grads,
)

response = stub.GossipGradient(request)
if response.accepted:
    print(f"Gradient accepted, peer at round {response.current_round}")
```

### GetWeights

Get model weights from a peer.

**Request:**
```protobuf
message WeightRequest {
  string shard_range = 1;          // Which layers to get
}
```

**Response:**
```protobuf
message WeightResponse {
  bytes weights_data = 1;          // Serialized state_dict
}
```

## Checkpoint Operations

### GetCheckpoint

Download checkpoint from peer.

**Request:**
```protobuf
message GetCheckpointRequest {
  string model_hash = 1;           // Optional: specific checkpoint hash
  int32 min_version = 2;           // Optional: minimum version number
}
```

**Response:**
```protobuf
message GetCheckpointResponse {
  bool success = 1;
  string error_message = 2;
  
  int32 version = 3;               // Checkpoint version (training round)
  string model_hash = 4;
  string phase = 5;                // Model phase (bootstrap, early, etc.)
  
  bytes checkpoint_data = 6;       // Serialized checkpoint (compressed)
  int64 total_size = 7;
}
```

**Python Example:**
```python
request = neuroshard_pb2.GetCheckpointRequest(
    min_version=100  # Get checkpoint at least version 100
)

response = stub.GetCheckpoint(request)

if response.success:
    # Decompress and load
    checkpoint = lz4.frame.decompress(response.checkpoint_data)
    model.load_state_dict(torch.load(io.BytesIO(checkpoint)))
    print(f"Loaded checkpoint v{response.version}")
```

### GetCheckpointInfo

Get checkpoint metadata without downloading.

**Request:**
```protobuf
message GetCheckpointInfoRequest {
  // Empty - just get current info
}
```

**Response:**
```protobuf
message GetCheckpointInfoResponse {
  int32 version = 1;               // Current training round
  string model_hash = 2;
  string phase = 3;                // Model phase
  int64 params = 4;                // Number of parameters
  double loss = 5;                 // Current loss
}
```

## DHT Operations (Peer Discovery)

### DHTPing

Health check for DHT nodes.

**Request:**
```protobuf
message DHTPingRequest {
  DHTNodeInfo sender = 1;
}

message DHTNodeInfo {
  bytes node_id = 1;               // 20-byte ID (160 bits)
  string ip = 2;
  int32 port = 3;
}
```

**Response:**
```protobuf
message DHTPingResponse {
  DHTNodeInfo responder = 1;
}
```

**Python Example:**
```python
import hashlib

# Generate node ID from public key
node_id = hashlib.sha1(public_key).digest()

request = neuroshard_pb2.DHTPingRequest(
    sender=neuroshard_pb2.DHTNodeInfo(
        node_id=node_id,
        ip="192.168.1.100",
        port=9000
    )
)

response = stub.DHTPing(request)
print(f"Peer: {response.responder.ip}:{response.responder.port}")
```

### DHTFindNode

Find K closest nodes to a target ID.

**Request:**
```protobuf
message DHTFindNodeRequest {
  DHTNodeInfo sender = 1;
  bytes target_id = 2;             // ID to search for
}
```

**Response:**
```protobuf
message DHTFindNodeResponse {
  DHTNodeInfo responder = 1;
  repeated DHTNodeInfo nodes = 2;  // K closest nodes
}
```

### DHTFindValue

Find a value in the DHT.

**Request:**
```protobuf
message DHTFindValueRequest {
  DHTNodeInfo sender = 1;
  bytes key = 2;
}
```

**Response:**
```protobuf
message DHTFindValueResponse {
  DHTNodeInfo responder = 1;
  string value = 2;                // If found
  repeated DHTNodeInfo nodes = 3;  // If not found (K closest nodes)
  bool found = 4;
}
```

### DHTStore

Store a value in the DHT.

**Request:**
```protobuf
message DHTStoreRequest {
  DHTNodeInfo sender = 1;
  bytes key = 2;
  string value = 3;                // e.g., "ip:port"
}
```

**Response:**
```protobuf
message DHTStoreResponse {
  DHTNodeInfo responder = 1;
  bool success = 2;
}
```

## Error Handling

### gRPC Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| `OK` | Success | - |
| `UNAVAILABLE` | Peer offline | Retry with backoff |
| `DEADLINE_EXCEEDED` | Timeout | Increase timeout or try different peer |
| `RESOURCE_EXHAUSTED` | Peer overloaded | Back off, try later |
| `NOT_FOUND` | Layer not on peer | Route to different peer |
| `INVALID_ARGUMENT` | Bad request | Check request format |
| `INTERNAL` | Server error | Log and retry |

### Retry Logic

```python
import grpc
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.1, max=2)
)
def forward_with_retry(stub, request):
    try:
        return stub.Forward(request, timeout=5.0)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            raise  # Retry
        elif e.code() == grpc.StatusCode.NOT_FOUND:
            raise ValueError("Layer not found")  # Don't retry
        else:
            raise
```

## Performance Tips

### Connection Pooling

```python
class GrpcPool:
    def __init__(self, max_per_peer=3):
        self.pools = {}
        self.max_per_peer = max_per_peer
    
    def get_stub(self, address):
        if address not in self.pools:
            self.pools[address] = []
        
        pool = self.pools[address]
        
        # Reuse existing
        for channel, stub in pool:
            if channel._channel.check_connectivity_state(True) == grpc.ChannelConnectivity.READY:
                return stub
        
        # Create new
        if len(pool) < self.max_per_peer:
            channel = grpc.insecure_channel(address)
            stub = neuroshard_pb2_grpc.NeuroShardStub(channel)
            pool.append((channel, stub))
            return stub
        
        # Wait for available
        return pool[0][1]
```

### Compression

Always compress large tensors:

```python
import lz4.frame

def compress_tensor(tensor):
    return lz4.frame.compress(
        tensor.cpu().numpy().tobytes(),
        compression_level=1  # Fast compression
    )

def decompress_tensor(data, shape, dtype=torch.float32):
    decompressed = lz4.frame.decompress(data)
    return torch.frombuffer(decompressed, dtype=dtype).view(shape)
```

### Streaming for Large Data

Use streaming for checkpoints and large gradients:

```python
def stream_checkpoint(stub, checkpoint_id, layers):
    """Stream checkpoint in chunks."""
    CHUNK_SIZE = 4 * 1024 * 1024  # 4MB chunks
    
    def chunk_generator():
        for layer_idx in layers:
            for name, param in model.layers[layer_idx].named_parameters():
                data = param.data.cpu().numpy().tobytes()
                total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
                
                for i in range(total_chunks):
                    chunk_data = data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
                    yield neuroshard_pb2.CheckpointChunk(
                        checkpoint_id=checkpoint_id,
                        layer_idx=layer_idx,
                        param_name=name,
                        data=chunk_data,
                        chunk_idx=i,
                        total_chunks=total_chunks
                    )
    
    response = stub.PutCheckpoint(chunk_generator())
    return response.success
```

## Next Steps

- [Python SDK](/api/python-sdk) — High-level client
- [P2P Network](/architecture/p2p-network) — Network architecture
- [HTTP Endpoints](/api/http-endpoints) — REST API

