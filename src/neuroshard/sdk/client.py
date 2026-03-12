"""
NeuroShard SDK Client

High-level Python client for interacting with NeuroShard nodes.

Usage:
    from neuroshard import NeuroNode, NEUROLedger
    
    node = NeuroNode("http://localhost:8000", api_token="YOUR_TOKEN")
    status = node.get_status()
    response = node.inference("Hello!", max_tokens=50)
    
    ledger = NEUROLedger(node)
    balance = ledger.get_balance()
"""

import os
import time
import json
import requests
from typing import Optional, List, Iterator, Dict, Any, Generator
from datetime import datetime, date, timedelta

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
    DailyReward,
    NodeConfig,
    TrainingStatus,
    ResourceStatus,
    InferenceMetrics,
    TrainingMetrics,
    NetworkMetrics,
    RewardMetrics,
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
    ForbiddenError,
    InternalError,
)


class NeuroNode:
    """
    High-level client for interacting with a NeuroShard node.
    
    Example:
        node = NeuroNode("http://localhost:8000", api_token="YOUR_TOKEN")
        
        # Check status
        status = node.get_status()
        print(f"Node: {status.node_id}")
        
        # Run inference
        response = node.inference("Explain quantum computing.", max_tokens=100)
        print(response.text)
    """
    
    def __init__(
        self,
        url: str,
        api_token: Optional[str] = None,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        verify_ssl: bool = True,
    ):
        """
        Initialize NeuroNode client.
        
        Args:
            url: Node URL (e.g., "http://localhost:8000")
            api_token: API token for authentication
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            verify_ssl: Whether to verify SSL certificates
        """
        self.url = url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.verify_ssl = verify_ssl
        
        self._session = requests.Session()
        if api_token:
            self._session.headers["Authorization"] = f"Bearer {api_token}"
        self._session.headers["Content-Type"] = "application/json"
    
    @classmethod
    def from_env(cls) -> "NeuroNode":
        """
        Create a NeuroNode from environment variables.
        
        Uses:
            NEUROSHARD_URL: Node URL
            NEUROSHARD_TOKEN: API token
            NEUROSHARD_TIMEOUT: Request timeout (optional)
        """
        url = os.environ.get("NEUROSHARD_URL", "http://localhost:8000")
        token = os.environ.get("NEUROSHARD_TOKEN")
        timeout = float(os.environ.get("NEUROSHARD_TIMEOUT", "30"))
        
        return cls(url=url, api_token=token, timeout=timeout)
    
    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "NeuroNode":
        """
        Create a NeuroNode from a config file.
        
        Args:
            config_path: Path to config file. Defaults to ~/.neuroshard/config.json
        """
        if config_path is None:
            config_path = os.path.expanduser("~/.neuroshard/config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config = json.load(f)
        
        return cls(
            url=config.get("url", "http://localhost:8000"),
            api_token=config.get("token"),
            timeout=config.get("timeout", 30.0),
            retry_attempts=config.get("retry_attempts", 3),
        )
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        stream: bool = False,
    ) -> dict:
        """Make an HTTP request to the node."""
        url = f"{self.url}{endpoint}"
        
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    stream=stream,
                )
                
                # Handle errors
                if response.status_code == 401:
                    raise AuthenticationError()
                elif response.status_code == 403:
                    raise ForbiddenError()
                elif response.status_code == 404:
                    raise NotFoundError("Resource", endpoint)
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise RateLimitError(retry_after)
                elif response.status_code >= 500:
                    raise InternalError()
                elif response.status_code >= 400:
                    try:
                        error_data = response.json()
                        error_info = error_data.get("error", {})
                        code = error_info.get("code", "UNKNOWN")
                        message = error_info.get("message", response.text)
                        
                        if code == "INSUFFICIENT_BALANCE":
                            details = error_info.get("details", {})
                            raise InsufficientBalanceError(
                                required=details.get("required", 0),
                                available=details.get("available", 0),
                            )
                        raise InvalidRequestError(message)
                    except (json.JSONDecodeError, KeyError):
                        raise InvalidRequestError(response.text)
                
                if stream:
                    return response
                
                return response.json()
                
            except requests.exceptions.ConnectionError as e:
                last_error = NodeOfflineError(self.url)
                if attempt < self.retry_attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise last_error
            except requests.exceptions.Timeout:
                last_error = NeuroShardError("Request timeout", code="TIMEOUT")
                if attempt < self.retry_attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise last_error
            except NeuroShardError:
                raise
            except Exception as e:
                raise NeuroShardError(str(e))
        
        if last_error:
            raise last_error
        raise NeuroShardError("Request failed after retries")
    
    def get_status(self) -> NodeStatus:
        """
        Get current node status.
        
        Returns:
            NodeStatus with node info, layers, peers, training status, etc.
        """
        data = self._request("GET", "/api/v1/status")
        
        return NodeStatus(
            node_id=data.get("node_id", ""),
            version=data.get("version", ""),
            uptime_seconds=data.get("uptime_seconds", 0),
            status=data.get("status", "unknown"),
            role=data.get("role", "unknown"),
            layers=data.get("layers", []),
            peer_count=data.get("peer_count", 0),
            has_embedding=data.get("has_embedding", False),
            has_lm_head=data.get("has_lm_head", False),
            training=TrainingStatus(
                enabled=data.get("training", {}).get("enabled", False),
                epoch=data.get("training", {}).get("epoch", 0),
                step=data.get("training", {}).get("step", 0),
                loss=data.get("training", {}).get("loss", 0.0),
            ),
            resources=ResourceStatus(
                gpu_memory_used=data.get("resources", {}).get("gpu_memory_used", 0),
                gpu_memory_total=data.get("resources", {}).get("gpu_memory_total", 0),
                cpu_percent=data.get("resources", {}).get("cpu_percent", 0.0),
                ram_used=data.get("resources", {}).get("ram_used", 0),
                ram_total=data.get("resources", {}).get("ram_total", 0),
            ),
        )
    
    def get_metrics(self) -> Metrics:
        """
        Get performance metrics.
        
        Returns:
            Metrics with inference, training, network, and reward stats.
        """
        data = self._request("GET", "/api/v1/metrics")
        
        return Metrics(
            timestamp=data.get("timestamp", ""),
            inference=InferenceMetrics(
                requests_total=data.get("inference", {}).get("requests_total", 0),
                requests_per_minute=data.get("inference", {}).get("requests_per_minute", 0.0),
                avg_latency_ms=data.get("inference", {}).get("avg_latency_ms", 0.0),
                p99_latency_ms=data.get("inference", {}).get("p99_latency_ms", 0.0),
                tokens_generated=data.get("inference", {}).get("tokens_generated", 0),
            ),
            training=TrainingMetrics(
                steps_total=data.get("training", {}).get("steps_total", 0),
                steps_per_hour=data.get("training", {}).get("steps_per_hour", 0.0),
                gradients_submitted=data.get("training", {}).get("gradients_submitted", 0),
                gradients_accepted=data.get("training", {}).get("gradients_accepted", 0),
            ),
            network=NetworkMetrics(
                bytes_sent=data.get("network", {}).get("bytes_sent", 0),
                bytes_received=data.get("network", {}).get("bytes_received", 0),
                active_connections=data.get("network", {}).get("active_connections", 0),
                rpc_calls=data.get("network", {}).get("rpc_calls", 0),
                peer_count=data.get("network", {}).get("peer_count", 0),
            ),
            rewards=RewardMetrics(
                earned_today=data.get("rewards", {}).get("earned_today", 0.0),
                earned_total=data.get("rewards", {}).get("earned_total", 0.0),
                pending=data.get("rewards", {}).get("pending", 0.0),
            ),
        )
    
    def inference(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> InferenceResponse:
        """
        Run an inference request.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            stop: Stop sequences
            stream: Enable streaming (use inference_stream for streaming)
        
        Returns:
            InferenceResponse with generated text, usage, cost, timing.
        """
        if stream:
            # For streaming, collect all chunks
            chunks = list(self.inference_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
            ))
            text = "".join(c.token for c in chunks)
            return InferenceResponse(
                id="stream",
                text=text,
                tokens_generated=len(chunks),
                finish_reason="stop",
                usage=TokenUsage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(chunks),
                    total_tokens=len(prompt.split()) + len(chunks),
                ),
                cost=Cost(amount=0.0, currency="NEURO"),
                timing=Timing(),
            )
        
        data = self._request("POST", "/api/v1/inference", data={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop or [],
            "stream": False,
        })
        
        return InferenceResponse(
            id=data.get("id", ""),
            text=data.get("text", ""),
            tokens_generated=data.get("tokens_generated", 0),
            finish_reason=data.get("finish_reason", "stop"),
            usage=TokenUsage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
                total_tokens=data.get("usage", {}).get("total_tokens", 0),
            ),
            cost=Cost(
                amount=data.get("cost", {}).get("amount", 0.0),
                currency=data.get("cost", {}).get("currency", "NEURO"),
            ),
            timing=Timing(
                queue_ms=data.get("timing", {}).get("queue_ms", 0.0),
                inference_ms=data.get("timing", {}).get("inference_ms", 0.0),
                total_ms=data.get("timing", {}).get("total_ms", 0.0),
            ),
        )
    
    def inference_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
    ) -> Generator[InferenceChunk, None, None]:
        """
        Stream inference response token by token.
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            stop: Stop sequences
        
        Yields:
            InferenceChunk for each generated token.
        """
        response = self._request("POST", "/api/v1/inference", data={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop or [],
            "stream": True,
        }, stream=True)
        
        index = 0
        for line in response.iter_lines():
            if not line:
                continue
            
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                
                try:
                    data = json.loads(data_str)
                    token = data.get("token", "")
                    if token == "[DONE]":
                        break
                    
                    yield InferenceChunk(
                        token=token,
                        index=index,
                        logprob=data.get("logprob"),
                    )
                    index += 1
                except json.JSONDecodeError:
                    continue
    
    def get_peers(self) -> List[PeerInfo]:
        """
        List connected peers.
        
        Returns:
            List of PeerInfo with peer details.
        """
        data = self._request("GET", "/api/v1/peers")
        
        peers = []
        for p in data.get("peers", []):
            peers.append(PeerInfo(
                id=p.get("id", ""),
                address=p.get("address", ""),
                role=p.get("role", "worker"),
                layers=p.get("layers", []),
                latency_ms=p.get("latency_ms", 0.0),
                connected_since=p.get("connected_since"),
            ))
        return peers
    
    def get_layers(self) -> List[LayerInfo]:
        """
        List assigned layers.
        
        Returns:
            List of LayerInfo with layer details.
        """
        data = self._request("GET", "/api/v1/layers")
        
        layers = []
        for l in data.get("layers", []):
            layers.append(LayerInfo(
                index=l.get("index", 0),
                type=l.get("type", "transformer"),
                memory_mb=l.get("memory_mb", 0),
                status=l.get("status", "active"),
            ))
        return layers
    
    def get_config(self) -> NodeConfig:
        """
        Get node configuration.
        
        Returns:
            NodeConfig with current settings.
        """
        data = self._request("GET", "/api/v1/config")
        
        return NodeConfig(
            node_id=data.get("node_id", ""),
            port=data.get("port", 8000),
            grpc_port=data.get("grpc_port", 9000),
            tracker_url=data.get("tracker_url", ""),
            training=data.get("training", {}),
            resources=data.get("resources", {}),
        )
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update node configuration.
        
        Args:
            updates: Dictionary of configuration updates.
        
        Returns:
            True if update was successful.
        """
        data = self._request("PATCH", "/api/v1/config", data=updates)
        return data.get("success", False)
    
    def health(self) -> Dict[str, Any]:
        """
        Check node health.
        
        Returns:
            Health status dictionary.
        """
        return self._request("GET", "/api/v1/health")


class NEUROLedger:
    """
    Client for NEURO token operations.
    
    Example:
        node = NeuroNode("http://localhost:8000", api_token="YOUR_TOKEN")
        ledger = NEUROLedger(node)
        
        balance = ledger.get_balance()
        print(f"Balance: {balance.available} NEURO")
    """
    
    def __init__(self, node: NeuroNode):
        """
        Initialize NEUROLedger.
        
        Args:
            node: NeuroNode instance to use for API calls.
        """
        self.node = node
    
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse ISO timestamp, handling Z suffix."""
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str)
    
    def get_balance(self) -> Balance:
        """
        Get wallet balance.
        
        Returns:
            Balance with available, staked, pending, and total amounts.
        """
        data = self.node._request("GET", "/api/v1/wallet/balance")
        
        balances = data.get("balances", data)
        return Balance(
            address=data.get("address", ""),
            available=balances.get("available", 0.0),
            staked=balances.get("staked", 0.0),
            pending=balances.get("pending", 0.0),
            total=balances.get("total", 0.0),
        )
    
    def send(
        self,
        to: str,
        amount: float,
        memo: Optional[str] = None,
    ) -> Transaction:
        """
        Send NEURO to another address.
        
        Args:
            to: Recipient address
            amount: Amount in NEURO
            memo: Optional transaction memo
        
        Returns:
            Transaction with details.
        """
        data = self.node._request("POST", "/api/v1/wallet/send", data={
            "to": to,
            "amount": amount,
            "memo": memo or "",
        })
        
        return Transaction(
            id=data.get("transaction_id", ""),
            from_address=data.get("from", ""),
            to_address=data.get("to", ""),
            amount=data.get("amount", 0.0),
            fee=data.get("fee", 0.0),
            status=data.get("status", "pending"),
            timestamp=self._parse_timestamp(data.get("timestamp", datetime.now().isoformat())),
            type="transfer",
            memo=data.get("memo"),
        )
    
    def get_transactions(
        self,
        limit: int = 10,
        offset: int = 0,
        type: Optional[str] = None,
    ) -> List[Transaction]:
        """
        Get transaction history.
        
        Args:
            limit: Maximum transactions to return
            offset: Offset for pagination
            type: Filter by type ("reward", "send", "receive", "stake")
        
        Returns:
            List of Transaction objects.
        """
        params = {"limit": limit, "offset": offset}
        if type:
            params["type"] = type
        
        data = self.node._request("GET", "/api/v1/wallet/transactions", params=params)
        
        transactions = []
        for t in data.get("transactions", []):
            transactions.append(Transaction(
                id=t.get("id", ""),
                from_address=t.get("from", ""),
                to_address=t.get("to", ""),
                amount=t.get("amount", 0.0),
                fee=t.get("fee", 0.0),
                status=t.get("status", "confirmed"),
                timestamp=self._parse_timestamp(t.get("timestamp", datetime.now().isoformat())),
                type=t.get("type", "transfer"),
                memo=t.get("memo"),
            ))
        return transactions
    
    def stake(self, amount: float, duration_days: int) -> StakeResult:
        """
        Stake NEURO tokens.
        
        Args:
            amount: Amount to stake
            duration_days: Lock duration in days
        
        Returns:
            StakeResult with stake details and multiplier.
        """
        data = self.node._request("POST", "/api/v1/wallet/stake", data={
            "amount": amount,
            "duration_days": duration_days,
        })
        
        stake = data.get("stake", {})
        return StakeResult(
            amount=stake.get("amount", amount),
            duration_days=stake.get("duration_days", duration_days),
            start_date=date.fromisoformat(stake.get("start_date", date.today().isoformat())),
            unlock_date=date.fromisoformat(stake.get("unlock_date", (date.today() + timedelta(days=duration_days)).isoformat())),
            multiplier=stake.get("multiplier", 1.0),
        )
    
    def unstake(self, amount: float) -> UnstakeResult:
        """
        Request unstaking.
        
        Args:
            amount: Amount to unstake
        
        Returns:
            UnstakeResult with cooldown info.
        """
        data = self.node._request("POST", "/api/v1/wallet/unstake", data={
            "amount": amount,
        })
        
        unstake = data.get("unstake", {})
        return UnstakeResult(
            amount=unstake.get("amount", amount),
            cooldown_days=unstake.get("cooldown_days", 7),
            available_date=date.fromisoformat(unstake.get("available_date", (date.today() + timedelta(days=7)).isoformat())),
        )
    
    def get_stake_info(self) -> StakeInfo:
        """
        Get current staking information.
        
        Returns:
            StakeInfo with stake amount, duration, and multiplier.
        """
        data = self.node._request("GET", "/api/v1/stake/info")
        
        return StakeInfo(
            amount=data.get("stake", 0.0),
            duration_days=data.get("duration_days", 0),
            start_date=date.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            unlock_date=date.fromisoformat(data["unlock_date"]) if data.get("unlock_date") else None,
            multiplier=data.get("stake_multiplier", 1.0),
            pending_unstake=data.get("pending_unstake", 0.0),
        )
    
    def get_rewards(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> RewardSummary:
        """
        Get reward history.
        
        Args:
            start_date: Filter from date
            end_date: Filter to date
        
        Returns:
            RewardSummary with totals and daily breakdown.
        """
        params = {}
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        data = self.node._request("GET", "/api/v1/wallet/rewards", params=params)
        
        by_day = []
        for d in data.get("by_day", []):
            by_day.append(DailyReward(
                date=date.fromisoformat(d.get("date", date.today().isoformat())),
                amount=d.get("amount", 0.0),
                proofs=d.get("proofs", 0),
            ))
        
        return RewardSummary(
            total=data.get("total", 0.0),
            by_day=by_day,
            by_type=data.get("by_type", {}),
        )


# ============================================================================
# ASYNC CLIENTS
# ============================================================================

class AsyncNeuroNode:
    """
    Async version of NeuroNode.
    
    Example:
        async with AsyncNeuroNode("http://localhost:8000", api_token="TOKEN") as node:
            status = await node.get_status()
            response = await node.inference("Hello!")
    """
    
    def __init__(
        self,
        url: str,
        api_token: Optional[str] = None,
        timeout: float = 30.0,
        retry_attempts: int = 3,
    ):
        self.url = url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self._session = None
    
    async def __aenter__(self):
        import aiohttp
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        self._session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """Make an async HTTP request."""
        import aiohttp
        
        if not self._session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        url = f"{self.url}{endpoint}"
        
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                async with self._session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                ) as response:
                    if response.status == 401:
                        raise AuthenticationError()
                    elif response.status == 403:
                        raise ForbiddenError()
                    elif response.status == 404:
                        raise NotFoundError("Resource", endpoint)
                    elif response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        raise RateLimitError(retry_after)
                    elif response.status >= 500:
                        raise InternalError()
                    elif response.status >= 400:
                        text = await response.text()
                        raise InvalidRequestError(text)
                    
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                last_error = NodeOfflineError(self.url)
                if attempt < self.retry_attempts - 1:
                    import asyncio
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise last_error
        
        if last_error:
            raise last_error
        raise NeuroShardError("Request failed after retries")
    
    async def get_status(self) -> NodeStatus:
        """Get current node status."""
        data = await self._request("GET", "/api/v1/status")
        
        return NodeStatus(
            node_id=data.get("node_id", ""),
            version=data.get("version", ""),
            uptime_seconds=data.get("uptime_seconds", 0),
            status=data.get("status", "unknown"),
            role=data.get("role", "unknown"),
            layers=data.get("layers", []),
            peer_count=data.get("peer_count", 0),
            has_embedding=data.get("has_embedding", False),
            has_lm_head=data.get("has_lm_head", False),
            training=TrainingStatus(
                enabled=data.get("training", {}).get("enabled", False),
                epoch=data.get("training", {}).get("epoch", 0),
                step=data.get("training", {}).get("step", 0),
                loss=data.get("training", {}).get("loss", 0.0),
            ),
            resources=ResourceStatus(
                gpu_memory_used=data.get("resources", {}).get("gpu_memory_used", 0),
                gpu_memory_total=data.get("resources", {}).get("gpu_memory_total", 0),
                cpu_percent=data.get("resources", {}).get("cpu_percent", 0.0),
                ram_used=data.get("resources", {}).get("ram_used", 0),
                ram_total=data.get("resources", {}).get("ram_total", 0),
            ),
        )
    
    async def inference(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
    ) -> InferenceResponse:
        """Run an inference request."""
        data = await self._request("POST", "/api/v1/inference", data={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop or [],
            "stream": False,
        })
        
        return InferenceResponse(
            id=data.get("id", ""),
            text=data.get("text", ""),
            tokens_generated=data.get("tokens_generated", 0),
            finish_reason=data.get("finish_reason", "stop"),
            usage=TokenUsage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
                total_tokens=data.get("usage", {}).get("total_tokens", 0),
            ),
            cost=Cost(
                amount=data.get("cost", {}).get("amount", 0.0),
                currency=data.get("cost", {}).get("currency", "NEURO"),
            ),
            timing=Timing(
                queue_ms=data.get("timing", {}).get("queue_ms", 0.0),
                inference_ms=data.get("timing", {}).get("inference_ms", 0.0),
                total_ms=data.get("timing", {}).get("total_ms", 0.0),
            ),
        )
    
    async def get_peers(self) -> List[PeerInfo]:
        """List connected peers."""
        data = await self._request("GET", "/api/v1/peers")
        
        peers = []
        for p in data.get("peers", []):
            peers.append(PeerInfo(
                id=p.get("id", ""),
                address=p.get("address", ""),
                role=p.get("role", "worker"),
                layers=p.get("layers", []),
                latency_ms=p.get("latency_ms", 0.0),
            ))
        return peers
    
    async def get_layers(self) -> List[LayerInfo]:
        """List assigned layers."""
        data = await self._request("GET", "/api/v1/layers")
        
        layers = []
        for l in data.get("layers", []):
            layers.append(LayerInfo(
                index=l.get("index", 0),
                type=l.get("type", "transformer"),
                memory_mb=l.get("memory_mb", 0),
                status=l.get("status", "active"),
            ))
        return layers


class AsyncNEUROLedger:
    """
    Async version of NEUROLedger.
    
    Example:
        async with AsyncNeuroNode("http://localhost:8000", api_token="TOKEN") as node:
            ledger = AsyncNEUROLedger(node)
            balance = await ledger.get_balance()
    """
    
    def __init__(self, node: AsyncNeuroNode):
        self.node = node
    
    async def get_balance(self) -> Balance:
        """Get wallet balance."""
        data = await self.node._request("GET", "/api/v1/wallet/balance")
        
        balances = data.get("balances", data)
        return Balance(
            address=data.get("address", ""),
            available=balances.get("available", 0.0),
            staked=balances.get("staked", 0.0),
            pending=balances.get("pending", 0.0),
            total=balances.get("total", 0.0),
        )
    
    async def send(self, to: str, amount: float, memo: Optional[str] = None) -> Transaction:
        """Send NEURO to another address."""
        data = await self.node._request("POST", "/api/v1/wallet/send", data={
            "to": to,
            "amount": amount,
            "memo": memo or "",
        })
        
        return Transaction(
            id=data.get("transaction_id", ""),
            from_address=data.get("from", ""),
            to_address=data.get("to", ""),
            amount=data.get("amount", 0.0),
            fee=data.get("fee", 0.0),
            status=data.get("status", "pending"),
            timestamp=self._parse_timestamp(data.get("timestamp", datetime.now().isoformat())),
            type="transfer",
            memo=data.get("memo"),
        )
    
    async def get_stake_info(self) -> StakeInfo:
        """Get current staking information."""
        data = await self.node._request("GET", "/api/v1/stake/info")
        
        return StakeInfo(
            amount=data.get("stake", 0.0),
            multiplier=data.get("stake_multiplier", 1.0),
        )

