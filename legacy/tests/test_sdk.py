"""
Tests for the NeuroShard Python SDK.

Tests the SDK client classes without requiring a running node.
Uses mocking to simulate API responses.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, date, timedelta
import json

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from neuroshard.sdk.client import NeuroNode, NEUROLedger, AsyncNeuroNode, AsyncNEUROLedger
from neuroshard.sdk.types import (
    NodeStatus,
    Metrics,
    InferenceResponse,
    InferenceChunk,
    PeerInfo,
    LayerInfo,
    Balance,
    Transaction,
    StakeInfo,
    StakeResult,
    UnstakeResult,
    RewardSummary,
    TrainingStatus,
    ResourceStatus,
    TokenUsage,
    Cost,
    Timing,
)
from neuroshard.sdk.errors import (
    NeuroShardError,
    AuthenticationError,
    InsufficientBalanceError,
    RateLimitError,
    NodeOfflineError,
    InvalidRequestError,
    NotFoundError,
)


class TestNeuroNodeClient(unittest.TestCase):
    """Tests for NeuroNode SDK client."""

    def setUp(self):
        """Set up test fixtures."""
        self.node = NeuroNode(
            url="http://localhost:8000",
            api_token="test_token_123",
            timeout=10.0,
            retry_attempts=1,  # Single attempt for tests
        )

    def test_init_strips_trailing_slash(self):
        """Test that URL trailing slash is stripped."""
        node = NeuroNode("http://localhost:8000/", api_token="token")
        self.assertEqual(node.url, "http://localhost:8000")

    def test_init_sets_auth_header(self):
        """Test that authorization header is set."""
        self.assertEqual(
            self.node._session.headers["Authorization"],
            "Bearer test_token_123"
        )

    def test_from_env(self):
        """Test creating NeuroNode from environment variables."""
        with patch.dict(os.environ, {
            "NEUROSHARD_URL": "http://custom:9000",
            "NEUROSHARD_TOKEN": "env_token",
            "NEUROSHARD_TIMEOUT": "60"
        }):
            node = NeuroNode.from_env()
            self.assertEqual(node.url, "http://custom:9000")
            self.assertEqual(node.api_token, "env_token")
            self.assertEqual(node.timeout, 60.0)

    @patch('requests.Session.request')
    def test_get_status(self, mock_request):
        """Test get_status method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "node_id": "node_abc123",
            "version": "0.1.0",
            "uptime_seconds": 3600,
            "status": "running",
            "role": "worker",
            "layers": [4, 5, 6, 7],
            "peer_count": 12,
            "has_embedding": False,
            "has_lm_head": False,
            "training": {
                "enabled": True,
                "epoch": 100,
                "step": 5000,
                "loss": 2.5
            },
            "resources": {
                "gpu_memory_used": 4000000000,
                "gpu_memory_total": 8000000000,
                "cpu_percent": 45.0,
                "ram_used": 2000000000,
                "ram_total": 16000000000
            }
        }
        mock_request.return_value = mock_response

        status = self.node.get_status()

        self.assertIsInstance(status, NodeStatus)
        self.assertEqual(status.node_id, "node_abc123")
        self.assertEqual(status.version, "0.1.0")
        self.assertEqual(status.uptime_seconds, 3600)
        self.assertEqual(status.status, "running")
        self.assertEqual(status.role, "worker")
        self.assertEqual(status.layers, [4, 5, 6, 7])
        self.assertEqual(status.peer_count, 12)
        self.assertEqual(status.training.enabled, True)
        self.assertEqual(status.training.loss, 2.5)
        self.assertEqual(status.resources.gpu_memory_used, 4000000000)

    @patch('requests.Session.request')
    def test_get_metrics(self, mock_request):
        """Test get_metrics method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "timestamp": "2024-12-05T12:00:00Z",
            "inference": {
                "requests_total": 1000,
                "requests_per_minute": 10.5,
                "avg_latency_ms": 150.0,
                "p99_latency_ms": 500.0,
                "tokens_generated": 50000
            },
            "training": {
                "steps_total": 10000,
                "steps_per_hour": 200.0,
                "gradients_submitted": 500,
                "gradients_accepted": 480
            },
            "network": {
                "bytes_sent": 1000000,
                "bytes_received": 2000000,
                "active_connections": 10,
                "rpc_calls": 5000,
                "peer_count": 15
            },
            "rewards": {
                "earned_today": 100.5,
                "earned_total": 5000.0,
                "pending": 25.0
            }
        }
        mock_request.return_value = mock_response

        metrics = self.node.get_metrics()

        self.assertIsInstance(metrics, Metrics)
        self.assertEqual(metrics.inference.requests_total, 1000)
        self.assertEqual(metrics.training.steps_total, 10000)
        self.assertEqual(metrics.network.peer_count, 15)
        self.assertEqual(metrics.rewards.earned_today, 100.5)

    @patch('requests.Session.request')
    def test_inference(self, mock_request):
        """Test inference method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "inf_123",
            "text": "This is the generated text.",
            "tokens_generated": 10,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15
            },
            "cost": {
                "amount": 0.015,
                "currency": "NEURO"
            },
            "timing": {
                "queue_ms": 10,
                "inference_ms": 200,
                "total_ms": 210
            }
        }
        mock_request.return_value = mock_response

        response = self.node.inference(
            prompt="Hello, AI!",
            max_tokens=100,
            temperature=0.7
        )

        self.assertIsInstance(response, InferenceResponse)
        self.assertEqual(response.id, "inf_123")
        self.assertEqual(response.text, "This is the generated text.")
        self.assertEqual(response.tokens_generated, 10)
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage.total_tokens, 15)
        self.assertEqual(response.cost.amount, 0.015)
        self.assertEqual(response.timing.total_ms, 210)

        # Verify the request was made with correct parameters
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        self.assertEqual(call_args.kwargs['json']['prompt'], "Hello, AI!")
        self.assertEqual(call_args.kwargs['json']['max_tokens'], 100)
        self.assertEqual(call_args.kwargs['json']['temperature'], 0.7)

    @patch('requests.Session.request')
    def test_inference_stream(self, mock_request):
        """Test inference_stream method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"token": "Hello", "index": 0}',
            b'data: {"token": " world", "index": 1}',
            b'data: {"token": "!", "index": 2}',
            b'data: [DONE]',
        ]
        mock_request.return_value = mock_response

        chunks = list(self.node.inference_stream("Say hello"))

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].token, "Hello")
        self.assertEqual(chunks[0].index, 0)
        self.assertEqual(chunks[1].token, " world")
        self.assertEqual(chunks[2].token, "!")

    @patch('requests.Session.request')
    def test_get_peers(self, mock_request):
        """Test get_peers method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "peers": [
                {
                    "id": "peer_1",
                    "address": "192.168.1.100:9000",
                    "role": "worker",
                    "layers": [0, 1, 2, 3],
                    "latency_ms": 25.0,
                    "connected_since": "2024-12-05T10:00:00Z"
                },
                {
                    "id": "peer_2",
                    "address": "192.168.1.101:9000",
                    "role": "validator",
                    "layers": [28, 29, 30, 31],
                    "latency_ms": 40.0,
                    "connected_since": "2024-12-05T09:00:00Z"
                }
            ]
        }
        mock_request.return_value = mock_response

        peers = self.node.get_peers()

        self.assertEqual(len(peers), 2)
        self.assertIsInstance(peers[0], PeerInfo)
        self.assertEqual(peers[0].id, "peer_1")
        self.assertEqual(peers[0].role, "worker")
        self.assertEqual(peers[1].role, "validator")

    @patch('requests.Session.request')
    def test_get_layers(self, mock_request):
        """Test get_layers method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "layers": [
                {"index": 4, "type": "transformer", "memory_mb": 512, "status": "active"},
                {"index": 5, "type": "transformer", "memory_mb": 512, "status": "active"},
            ]
        }
        mock_request.return_value = mock_response

        layers = self.node.get_layers()

        self.assertEqual(len(layers), 2)
        self.assertIsInstance(layers[0], LayerInfo)
        self.assertEqual(layers[0].index, 4)
        self.assertEqual(layers[0].type, "transformer")

    @patch('requests.Session.request')
    def test_authentication_error(self, mock_request):
        """Test that 401 raises AuthenticationError."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_request.return_value = mock_response

        with self.assertRaises(AuthenticationError):
            self.node.get_status()

    @patch('requests.Session.request')
    def test_rate_limit_error(self, mock_request):
        """Test that 429 raises RateLimitError."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "30"}
        mock_request.return_value = mock_response

        with self.assertRaises(RateLimitError) as ctx:
            self.node.get_status()
        
        self.assertEqual(ctx.exception.retry_after, 30)

    @patch('requests.Session.request')
    def test_not_found_error(self, mock_request):
        """Test that 404 raises NotFoundError."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_request.return_value = mock_response

        with self.assertRaises(NotFoundError):
            self.node.get_status()

    @patch('requests.Session.request')
    def test_insufficient_balance_error(self, mock_request):
        """Test that INSUFFICIENT_BALANCE error is parsed."""
        mock_response = Mock()
        mock_response.status_code = 402
        mock_response.json.return_value = {
            "error": {
                "code": "INSUFFICIENT_BALANCE",
                "message": "Not enough NEURO",
                "details": {
                    "required": 100.0,
                    "available": 50.0
                }
            }
        }
        mock_request.return_value = mock_response

        with self.assertRaises(InsufficientBalanceError) as ctx:
            self.node.inference("test")
        
        self.assertEqual(ctx.exception.required, 100.0)
        self.assertEqual(ctx.exception.available, 50.0)

    @patch('requests.Session.request')
    def test_connection_error_raises_node_offline(self, mock_request):
        """Test that connection error raises NodeOfflineError."""
        import requests
        mock_request.side_effect = requests.exceptions.ConnectionError()

        with self.assertRaises(NodeOfflineError):
            self.node.get_status()


class TestNEUROLedgerClient(unittest.TestCase):
    """Tests for NEUROLedger SDK client."""

    def setUp(self):
        """Set up test fixtures."""
        self.node = NeuroNode(
            url="http://localhost:8000",
            api_token="test_token_123",
            retry_attempts=1,
        )
        self.ledger = NEUROLedger(self.node)

    @patch('requests.Session.request')
    def test_get_balance(self, mock_request):
        """Test get_balance method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "address": "0x1234abcd",
            "balances": {
                "available": 1000.0,
                "staked": 500.0,
                "pending": 50.0,
                "total": 1550.0
            }
        }
        mock_request.return_value = mock_response

        balance = self.ledger.get_balance()

        self.assertIsInstance(balance, Balance)
        self.assertEqual(balance.address, "0x1234abcd")
        self.assertEqual(balance.available, 1000.0)
        self.assertEqual(balance.staked, 500.0)
        self.assertEqual(balance.pending, 50.0)
        self.assertEqual(balance.total, 1550.0)

    @patch('requests.Session.request')
    def test_send(self, mock_request):
        """Test send method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "transaction_id": "tx_123",
            "from": "0x1234",
            "to": "0x5678",
            "amount": 100.0,
            "fee": 0.1,
            "status": "confirmed",
            "timestamp": "2024-12-05T12:00:00Z",
            "memo": "Payment"
        }
        mock_request.return_value = mock_response

        tx = self.ledger.send(to="0x5678", amount=100.0, memo="Payment")

        self.assertIsInstance(tx, Transaction)
        self.assertEqual(tx.id, "tx_123")
        self.assertEqual(tx.amount, 100.0)
        self.assertEqual(tx.fee, 0.1)
        self.assertEqual(tx.status, "confirmed")

    @patch('requests.Session.request')
    def test_get_transactions(self, mock_request):
        """Test get_transactions method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "transactions": [
                {
                    "id": "tx_1",
                    "type": "reward",
                    "amount": 10.0,
                    "fee": 0.0,
                    "status": "confirmed",
                    "timestamp": "2024-12-05T12:00:00Z"
                },
                {
                    "id": "tx_2",
                    "type": "send",
                    "amount": -50.0,
                    "to": "0x5678",
                    "fee": 0.1,
                    "status": "confirmed",
                    "timestamp": "2024-12-05T11:00:00Z"
                }
            ]
        }
        mock_request.return_value = mock_response

        transactions = self.ledger.get_transactions(limit=10)

        self.assertEqual(len(transactions), 2)
        self.assertEqual(transactions[0].id, "tx_1")
        self.assertEqual(transactions[0].type, "reward")

    @patch('requests.Session.request')
    def test_stake(self, mock_request):
        """Test stake method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "stake": {
                "amount": 1000.0,
                "duration_days": 30,
                "start_date": "2024-12-05",
                "unlock_date": "2025-01-04",
                "multiplier": 1.5
            }
        }
        mock_request.return_value = mock_response

        result = self.ledger.stake(amount=1000.0, duration_days=30)

        self.assertIsInstance(result, StakeResult)
        self.assertEqual(result.amount, 1000.0)
        self.assertEqual(result.duration_days, 30)
        self.assertEqual(result.multiplier, 1.5)

    @patch('requests.Session.request')
    def test_unstake(self, mock_request):
        """Test unstake method."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "unstake": {
                "amount": 500.0,
                "cooldown_days": 7,
                "available_date": "2024-12-12"
            }
        }
        mock_request.return_value = mock_response

        result = self.ledger.unstake(amount=500.0)

        self.assertIsInstance(result, UnstakeResult)
        self.assertEqual(result.amount, 500.0)
        self.assertEqual(result.cooldown_days, 7)


class TestSDKTypes(unittest.TestCase):
    """Tests for SDK data types."""

    def test_node_status_creation(self):
        """Test NodeStatus creation with required fields."""
        status = NodeStatus(
            node_id="node_123",
            version="0.1.0",
            uptime_seconds=3600,
            status="running",
            role="worker",
            layers=[0, 1, 2],
            peer_count=5,
            training=TrainingStatus(),
            resources=ResourceStatus(),
        )
        self.assertEqual(status.node_id, "node_123")
        self.assertEqual(status.uptime_seconds, 3600)
        self.assertEqual(status.status, "running")
        self.assertEqual(status.layers, [0, 1, 2])
        self.assertEqual(status.peer_count, 5)

    def test_inference_response_creation(self):
        """Test InferenceResponse creation."""
        response = InferenceResponse(
            id="inf_123",
            text="Generated text",
            tokens_generated=5,
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=3,
                completion_tokens=5,
                total_tokens=8
            ),
            cost=Cost(amount=0.01, currency="NEURO"),
            timing=Timing(queue_ms=5, inference_ms=100, total_ms=105)
        )
        self.assertEqual(response.text, "Generated text")
        self.assertEqual(response.usage.total_tokens, 8)

    def test_balance_total_calculation(self):
        """Test Balance struct."""
        balance = Balance(
            address="0x123",
            available=100.0,
            staked=50.0,
            pending=10.0,
            total=160.0
        )
        self.assertEqual(balance.total, 160.0)


class TestSDKErrors(unittest.TestCase):
    """Tests for SDK error classes."""

    def test_neuroshard_error_str(self):
        """Test NeuroShardError string representation."""
        error = NeuroShardError("Something went wrong", code="TEST_ERROR")
        self.assertEqual(str(error), "[TEST_ERROR] Something went wrong")

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError()
        self.assertEqual(error.code, "UNAUTHORIZED")

    def test_insufficient_balance_error(self):
        """Test InsufficientBalanceError."""
        error = InsufficientBalanceError(required=100.0, available=50.0)
        self.assertEqual(error.required, 100.0)
        self.assertEqual(error.available, 50.0)

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError(retry_after=60)
        self.assertEqual(error.retry_after, 60)

    def test_node_offline_error(self):
        """Test NodeOfflineError."""
        error = NodeOfflineError("http://localhost:8000")
        self.assertIn("localhost:8000", str(error))


class TestAsyncNeuroNode(unittest.TestCase):
    """Tests for AsyncNeuroNode client."""

    def test_init(self):
        """Test AsyncNeuroNode initialization."""
        node = AsyncNeuroNode(
            url="http://localhost:8000",
            api_token="test_token",
            timeout=30.0
        )
        self.assertEqual(node.url, "http://localhost:8000")
        self.assertEqual(node.api_token, "test_token")
        self.assertEqual(node.timeout, 30.0)

    def test_session_not_initialized_raises_error(self):
        """Test that using client without context manager raises error."""
        node = AsyncNeuroNode("http://localhost:8000")
        
        # Calling _request without __aenter__ should raise RuntimeError
        import asyncio
        
        async def test():
            with self.assertRaises(RuntimeError):
                await node._request("GET", "/api/v1/status")
        
        asyncio.run(test())


if __name__ == "__main__":
    unittest.main()

