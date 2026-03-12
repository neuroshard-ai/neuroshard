"""
Activation Buffer System - Decoupling Compute from Communication

Implements priority queues for incoming and outgoing activations with:
- Priority-based scheduling (inference > training forward > backward)
- Soft overflow handling ("Don't Stop" logic)
- Metrics for buffer fill rate and wait times

Key Directive: "The GPU must never wait for network packets. 
Forward pass can run 100 steps ahead of backward if necessary."

Metric: Buffer Fill Rate
- Empty (< 10%) = Starved (BAD - GPU idle)
- Full (> 90%) = Backpressured (triggers soft overflow)
"""

import asyncio
import heapq
import logging
import threading
import time
import torch
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from neuroshard.core.swarm.router import SwarmRouter

logger = logging.getLogger(__name__)


class ActivationPriority(IntEnum):
    """
    Priority levels for activation processing.
    
    Lower value = higher priority (processed first).
    
    Rationale:
    - INFERENCE_URGENT: Paid inference, user waiting - highest priority
    - INFERENCE_NORMAL: Standard inference requests
    - TRAINING_FORWARD: Training forward passes need timely processing
    - TRAINING_BACKWARD: Backward passes can be slightly delayed
    - GRADIENT_SYNC: Gradient gossip can wait longest
    """
    INFERENCE_URGENT = 0    # Paid inference, user waiting
    INFERENCE_NORMAL = 10   # Standard inference
    TRAINING_FORWARD = 20   # Training forward pass
    TRAINING_BACKWARD = 30  # Training backward pass (can be delayed)
    GRADIENT_SYNC = 40      # Gradient gossip


@dataclass(order=True)
class ActivationPacket:
    """
    A unit of work for the compute engine.
    
    Packets are ordered by (priority, sequence_num) for heap operations.
    Lower priority value = processed first. Sequence_num breaks ties (FIFO).
    """
    priority: int
    sequence_num: int = field(default=0)  # For stable FIFO ordering within priority
    timestamp: float = field(compare=False, default_factory=time.time)
    
    # Payload
    session_id: str = field(compare=False, default="")
    micro_batch_id: int = field(compare=False, default=0)
    tensor_data: Optional[torch.Tensor] = field(compare=False, default=None)
    
    # Routing info
    source_node: str = field(compare=False, default="")
    target_layer: int = field(compare=False, default=0)
    is_backward: bool = field(compare=False, default=False)
    
    # Training metadata
    requires_grad: bool = field(compare=False, default=False)
    grad_output: Optional[torch.Tensor] = field(compare=False, default=None)
    
    # For tracking
    created_at: float = field(compare=False, default_factory=time.time)
    
    def wait_time_ms(self) -> float:
        """Time since packet was created, in milliseconds."""
        return (time.time() - self.created_at) * 1000
    
    @classmethod
    def create_forward(
        cls,
        session_id: str,
        micro_batch_id: int,
        tensor: torch.Tensor,
        source_node: str,
        target_layer: int,
        is_inference: bool = False,
        urgent: bool = False,
    ) -> 'ActivationPacket':
        """Factory for forward pass packets."""
        if is_inference:
            priority = ActivationPriority.INFERENCE_URGENT if urgent else ActivationPriority.INFERENCE_NORMAL
        else:
            priority = ActivationPriority.TRAINING_FORWARD
            
        return cls(
            priority=priority,
            session_id=session_id,
            micro_batch_id=micro_batch_id,
            tensor_data=tensor,
            source_node=source_node,
            target_layer=target_layer,
            is_backward=False,
            requires_grad=not is_inference,
        )
    
    @classmethod
    def create_backward(
        cls,
        session_id: str,
        micro_batch_id: int,
        grad_tensor: torch.Tensor,
        source_node: str,
        target_layer: int,
    ) -> 'ActivationPacket':
        """Factory for backward pass packets."""
        return cls(
            priority=ActivationPriority.TRAINING_BACKWARD,
            session_id=session_id,
            micro_batch_id=micro_batch_id,
            tensor_data=grad_tensor,
            source_node=source_node,
            target_layer=target_layer,
            is_backward=True,
            requires_grad=True,
            grad_output=grad_tensor,
        )


class ActivationBuffer:
    """
    Priority queue for incoming activations.
    
    Decouples network I/O from GPU computation:
    - Network receiver pushes packets
    - GPU compute thread pops next-priority item
    
    Thread-safe: Uses locks for multi-producer/single-consumer pattern.
    
    Metrics:
    - fill_rate: 0.0 = starved (bad), 1.0 = full (backpressured)
    - avg_wait_time: How long packets wait before processing
    """
    
    # Default buffer size (100 packets ≈ 100 micro-batches)
    DEFAULT_MAX_SIZE = 100
    
    # Starvation/backpressure thresholds
    STARVED_THRESHOLD = 0.1       # < 10% full = starved
    BACKPRESSURE_THRESHOLD = 0.9  # > 90% full = backpressured
    
    def __init__(self, max_size: int = DEFAULT_MAX_SIZE):
        """
        Initialize activation buffer.
        
        Args:
            max_size: Maximum number of packets in buffer
        """
        self.max_size = max_size
        self._queue: List[ActivationPacket] = []
        self._lock = threading.RLock()  # Reentrant lock to allow nested calls
        self._event_not_empty = threading.Event()
        self._event_not_full = threading.Event()
        self._event_not_full.set()  # Initially not full
        
        # Sequence counter for ordering packets with same priority
        self._sequence_counter = 0
        
        # Metrics
        self.packets_in = 0
        self.packets_out = 0
        self.packets_dropped = 0
        self.total_wait_time = 0.0
        
        # Priority breakdown
        self._priority_counts: Dict[int, int] = {}
        
    @property
    def fill_rate(self) -> float:
        """Current fill rate (0.0 to 1.0)."""
        with self._lock:
            return len(self._queue) / self.max_size if self.max_size > 0 else 0.0
    
    @property
    def is_starved(self) -> bool:
        """True if buffer is starving (< 10% full)."""
        return self.fill_rate < self.STARVED_THRESHOLD
    
    @property
    def is_backpressured(self) -> bool:
        """True if buffer is backpressured (>= 90% full)."""
        return self.fill_rate >= self.BACKPRESSURE_THRESHOLD
    
    def put(self, packet: ActivationPacket, timeout: Optional[float] = None) -> bool:
        """
        Add activation to buffer.
        
        Args:
            packet: Activation packet to add
            timeout: Max time to wait if buffer full (None = blocking, 0 = non-blocking)
            
        Returns:
            True if added successfully, False if timeout expired
        """
        # Wait for space if buffer is full
        if timeout == 0:
            # Non-blocking
            with self._lock:
                if len(self._queue) >= self.max_size:
                    return False
        else:
            # Blocking with optional timeout
            if not self._event_not_full.wait(timeout):
                return False
        
        with self._lock:
            # Double-check after acquiring lock
            if len(self._queue) >= self.max_size:
                return False
            
            # Add sequence number for stable ordering
            self._sequence_counter += 1
            packet.sequence_num = self._sequence_counter
            
            heapq.heappush(self._queue, packet)
            self.packets_in += 1
            
            # Track priority distribution
            self._priority_counts[packet.priority] = (
                self._priority_counts.get(packet.priority, 0) + 1
            )
            
            # Update events
            self._event_not_empty.set()
            if len(self._queue) >= self.max_size:
                self._event_not_full.clear()
            
            return True
    
    def put_nowait(self, packet: ActivationPacket) -> bool:
        """
        Non-blocking put - returns False immediately if full.
        
        Used by soft overflow logic.
        """
        return self.put(packet, timeout=0)
    
    def get(self, timeout: Optional[float] = None) -> Optional[ActivationPacket]:
        """
        Get highest-priority activation.
        
        Args:
            timeout: Max time to wait if buffer empty (None = blocking, 0 = non-blocking)
            
        Returns:
            Highest priority packet, or None if timeout expired
        """
        # Wait for data if buffer is empty
        if timeout == 0:
            # Non-blocking
            with self._lock:
                if not self._queue:
                    return None
        else:
            # Blocking with optional timeout
            if not self._event_not_empty.wait(timeout):
                return None
        
        with self._lock:
            # Double-check after acquiring lock
            if not self._queue:
                return None
            
            packet = heapq.heappop(self._queue)
            
            # Track metrics
            wait_time = time.time() - packet.timestamp
            self.total_wait_time += wait_time
            self.packets_out += 1
            
            # Update events
            self._event_not_full.set()
            if not self._queue:
                self._event_not_empty.clear()
            
            return packet
    
    def get_nowait(self) -> Optional[ActivationPacket]:
        """
        Non-blocking get - returns None immediately if empty.
        
        Used by compute engine for non-blocking iteration.
        """
        return self.get(timeout=0)
    
    def peek(self) -> Optional[ActivationPacket]:
        """Peek at next packet without removing it."""
        with self._lock:
            if self._queue:
                return self._queue[0]
            return None
    
    def clear(self) -> int:
        """Clear buffer, returns number of packets cleared."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            self.packets_dropped += count
            self._event_not_full.set()
            self._event_not_empty.clear()
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            avg_wait = (
                self.total_wait_time / max(1, self.packets_out) * 1000
            )  # Convert to ms
            
            return {
                "fill_rate": self.fill_rate,
                "queue_size": len(self._queue),
                "max_size": self.max_size,
                "packets_in": self.packets_in,
                "packets_out": self.packets_out,
                "packets_dropped": self.packets_dropped,
                "avg_wait_time_ms": avg_wait,
                "is_starved": self.is_starved,
                "is_backpressured": self.is_backpressured,
                "priority_breakdown": dict(self._priority_counts),
            }


class OutboundBuffer:
    """
    Buffer for outgoing activations with async network send.
    
    Supports the "Soft Overflow" mechanism:
    - When full, compute engine can check without blocking
    - Enables DiLoCo-style local-only steps during congestion
    
    Thread-safe async queue with metrics.
    """
    
    DEFAULT_MAX_SIZE = 50
    
    # Default soft overflow thresholds
    DEFAULT_SOFT_LIMIT = 0.9   # 90% - start warning
    DEFAULT_HARD_LIMIT = 0.99  # 99% - critical
    
    def __init__(
        self, 
        max_size: int = DEFAULT_MAX_SIZE,
        soft_overflow_threshold: float = DEFAULT_SOFT_LIMIT,
        hard_overflow_threshold: float = DEFAULT_HARD_LIMIT,
    ):
        """
        Initialize outbound buffer.
        
        Args:
            max_size: Maximum number of pending outbound packets
            soft_overflow_threshold: Fill rate to trigger soft overflow (0-1)
            hard_overflow_threshold: Fill rate to trigger hard overflow (0-1)
        """
        self.max_size = max_size
        self.soft_limit = soft_overflow_threshold
        self.hard_limit = hard_overflow_threshold
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        
        # Track send status
        self.send_in_progress = 0
        self.packets_sent = 0
        self.packets_failed = 0
        self.send_retries = 0
        
        # Soft overflow tracking
        self.soft_overflow_count = 0
        self.hard_overflow_count = 0
        
        # Send loop state
        self._send_task: Optional[asyncio.Task] = None
        self._running = False
        
    @property
    def fill_rate(self) -> float:
        """Current fill rate (0.0 to 1.0)."""
        return self._queue.qsize() / self.max_size if self.max_size > 0 else 0.0
    
    @property
    def is_soft_overflow(self) -> bool:
        """True if buffer is in soft overflow (>= soft_limit)."""
        return self.fill_rate >= self.soft_limit
    
    @property
    def is_hard_overflow(self) -> bool:
        """True if buffer is in hard overflow (>= hard_limit)."""
        return self.fill_rate >= self.hard_limit
    
    def check_pressure(self) -> str:
        """
        Check current backpressure level.
        
        Returns:
            "ok" - normal operation
            "soft_overflow" - buffer almost full, consider local accumulation
            "hard_overflow" - buffer critical, must discard
        """
        if self.is_hard_overflow:
            return "hard_overflow"
        elif self.is_soft_overflow:
            return "soft_overflow"
        else:
            return "ok"
    
    async def put(self, packet: ActivationPacket, timeout: Optional[float] = None):
        """
        Queue packet for sending.
        
        Args:
            packet: Packet to send
            timeout: Max time to wait if full (None = blocking)
            
        Raises:
            asyncio.TimeoutError: If timeout expires while buffer full
            asyncio.QueueFull: If full and timeout=0
        """
        if timeout == 0:
            self._queue.put_nowait(packet)
        elif timeout is not None:
            await asyncio.wait_for(self._queue.put(packet), timeout=timeout)
        else:
            await self._queue.put(packet)
    
    def put_nowait(self, packet: ActivationPacket) -> bool:
        """
        Non-blocking put - returns False if full.
        
        Used for soft overflow checking.
        """
        try:
            self._queue.put_nowait(packet)
            return True
        except asyncio.QueueFull:
            return False
    
    async def get(self) -> ActivationPacket:
        """Get next packet to send (blocks if empty)."""
        return await self._queue.get()
    
    def start_send_loop(self, swarm_router: 'SwarmRouter'):
        """Start the background send loop."""
        if self._send_task is not None:
            return
        self._running = True
        self._send_task = asyncio.create_task(self._send_loop(swarm_router))
        logger.info("OutboundBuffer send loop started")
        
    async def stop_send_loop(self):
        """Stop the background send loop."""
        self._running = False
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
            self._send_task = None
        logger.info("OutboundBuffer send loop stopped")
        
    async def _send_loop(self, swarm_router: 'SwarmRouter'):
        """
        Continuously send packets to peers via swarm router.
        
        Uses failover routing for resilience.
        """
        while self._running:
            try:
                packet = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
                
            self.send_in_progress += 1
            
            try:
                result = await swarm_router.send_with_failover(
                    tensor=packet.tensor_data,
                    target_layer=packet.target_layer,
                    session_id=packet.session_id,
                    metadata={
                        "source_node": packet.source_node,
                        "micro_batch_id": packet.micro_batch_id,
                        "priority": packet.priority,
                        "is_backward": packet.is_backward,
                    }
                )
                
                if result.success:
                    self.packets_sent += 1
                else:
                    self.packets_failed += 1
                    logger.warning(f"Failed to send packet: {result.error}")
                    
                # Track retries (failover attempts - 1)
                if result.attempts > 1:
                    self.send_retries += result.attempts - 1
                    
            except Exception as e:
                self.packets_failed += 1
                logger.error(f"Send error: {e}")
            finally:
                self.send_in_progress -= 1
                self._queue.task_done()
    
    async def flush(self, timeout: float = 30.0):
        """Wait for all pending packets to be sent."""
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Flush timeout after {timeout}s, {self._queue.qsize()} packets remaining")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get outbound buffer statistics."""
        return {
            "fill_rate": self.fill_rate,
            "queue_size": self._queue.qsize(),
            "max_size": self.max_size,
            "send_in_progress": self.send_in_progress,
            "packets_sent": self.packets_sent,
            "packets_failed": self.packets_failed,
            "send_retries": self.send_retries,
            "soft_overflow_count": self.soft_overflow_count,
            "hard_overflow_count": self.hard_overflow_count,
            "is_soft_overflow": self.is_soft_overflow,
            "is_hard_overflow": self.is_hard_overflow,
            "pressure": self.check_pressure(),
        }


class BufferMetricsCollector:
    """
    Collects and aggregates metrics from activation buffers.
    
    Provides unified view for monitoring dashboard.
    """
    
    def __init__(
        self,
        inbound: ActivationBuffer,
        outbound: OutboundBuffer,
        collect_interval: float = 5.0,
    ):
        self.inbound = inbound
        self.outbound = outbound
        self.collect_interval = collect_interval
        
        # Historical metrics
        self.history: List[Dict[str, Any]] = []
        self.max_history = 720  # 1 hour at 5s intervals
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start metrics collection."""
        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        
    async def stop(self):
        """Stop metrics collection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
    async def _collect_loop(self):
        """Periodically collect buffer metrics."""
        while self._running:
            try:
                await asyncio.sleep(self.collect_interval)
                
                snapshot = {
                    "timestamp": time.time(),
                    "inbound": self.inbound.get_stats(),
                    "outbound": self.outbound.get_stats(),
                }
                
                self.history.append(snapshot)
                
                # Trim history
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]
                    
                # Log warnings
                if self.inbound.is_starved:
                    logger.warning(
                        f"Inbound buffer STARVED: {self.inbound.fill_rate:.1%} full"
                    )
                if self.outbound.is_soft_overflow:
                    logger.warning(
                        f"Outbound buffer OVERFLOW: {self.outbound.fill_rate:.1%} full"
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.history:
            return {
                "inbound": self.inbound.get_stats(),
                "outbound": self.outbound.get_stats(),
            }
            
        # Calculate averages over history
        avg_inbound_fill = sum(h["inbound"]["fill_rate"] for h in self.history) / len(self.history)
        avg_outbound_fill = sum(h["outbound"]["fill_rate"] for h in self.history) / len(self.history)
        
        return {
            "current": {
                "inbound": self.inbound.get_stats(),
                "outbound": self.outbound.get_stats(),
            },
            "averages": {
                "inbound_fill_rate": avg_inbound_fill,
                "outbound_fill_rate": avg_outbound_fill,
            },
            "history_length": len(self.history),
        }

