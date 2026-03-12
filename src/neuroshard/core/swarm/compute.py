"""
Compute Engine - Decoupled GPU Worker with Soft Overflow

Implements async compute loop with:
- Priority-based activation processing
- Interleaved 1F1B schedule (forward/backward interleaving)
- Soft overflow handling ("Don't Stop" logic)
- DiLoCo-style local gradient accumulation during congestion

Key Directive: "If outbound.full(): Do not await. Instead: 
accumulate_gradient_locally() and discard the activation. 
Treat it as a DiLoCo-style local-only training step."

CRITICAL: GPU must NEVER wait for network. Compute loop must always make progress.
"""

import asyncio
import logging
import time
import torch
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Any, List, TYPE_CHECKING

from neuroshard.core.swarm.buffers import (
    ActivationBuffer,
    OutboundBuffer,
    ActivationPacket,
    ActivationPriority,
)

if TYPE_CHECKING:
    from neuroshard.core.swarm.diloco import DiLoCoTrainer

logger = logging.getLogger(__name__)


class StepOutcome(Enum):
    """Outcome of a compute step."""
    SENT = "sent"               # Normal: activation sent to next peer
    LOCAL_ONLY = "local_only"   # Soft overflow: accumulated locally, activation discarded
    DROPPED = "dropped"         # Critical overflow: couldn't even accumulate


@dataclass
class ComputeStats:
    """Statistics for compute engine."""
    total_steps: int = 0
    forward_count: int = 0
    backward_count: int = 0
    local_only_steps: int = 0
    dropped_steps: int = 0
    total_compute_time_ms: float = 0.0
    total_queue_time_ms: float = 0.0
    
    @property
    def local_only_rate(self) -> float:
        """Fraction of steps that were local-only due to overflow."""
        if self.total_steps == 0:
            return 0.0
        return self.local_only_steps / self.total_steps
    
    @property
    def drop_rate(self) -> float:
        """Fraction of steps that were completely dropped."""
        if self.total_steps == 0:
            return 0.0
        return self.dropped_steps / self.total_steps
    
    @property
    def avg_compute_time_ms(self) -> float:
        """Average compute time per step."""
        if self.total_steps == 0:
            return 0.0
        return self.total_compute_time_ms / self.total_steps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_steps": self.total_steps,
            "forward_count": self.forward_count,
            "backward_count": self.backward_count,
            "local_only_steps": self.local_only_steps,
            "dropped_steps": self.dropped_steps,
            "local_only_rate": self.local_only_rate,
            "drop_rate": self.drop_rate,
            "avg_compute_time_ms": self.avg_compute_time_ms,
        }


class ComputeEngine:
    """
    Decoupled GPU compute worker with Soft Overflow handling.
    
    Architecture:
    
        ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
        │ InboundQueue│ ──→ │ GPU Compute │ ──→ │OutboundQueue│
        │ (Priority)  │     │ (Never Wait)│     │ (Async Send)│
        └─────────────┘     └─────────────┘     └─────────────┘
    
    Key Behaviors:
    
    1. PULLS from inbound buffer (priority queue)
    2. COMPUTES forward/backward pass on GPU
    3. PUSHES to outbound buffer (if not congested)
    
    CRITICAL: Never waits for network - GPU must never stall!
    
    Soft Overflow Logic ("Don't Stop" Mechanism):
    =============================================
    When outbound buffer is full (network congestion):
    
    1. DO NOT await outbound.put() - this would stall GPU
    2. Instead, accumulate gradients locally (DiLoCo style)
    3. Discard activation (don't try to send)
    4. Continue processing next packet
    
    Rationale: Better to treat a step as "local-only training" than to
    halt the GPU waiting for network. DiLoCo outer optimizer syncs later.
    
    Interleaved 1F1B Schedule:
    ==========================
    For 4 micro-batches: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 ...
    
    Start backward passes BEFORE all forwards complete, overlapping
    backward compute with forward network latency.
    """
    
    # Soft overflow thresholds
    OUTBOUND_SOFT_LIMIT = 0.9    # Start soft overflow at 90% full
    OUTBOUND_HARD_LIMIT = 0.99   # Hard limit - must discard
    
    # Scheduling parameters
    DEFAULT_NUM_MICRO_BATCHES = 4
    DEFAULT_WARMUP_STEPS = 4     # Forward steps before interleaving
    
    def __init__(
        self,
        model: Any,  # DynamicNeuroLLM
        inbound: ActivationBuffer,
        outbound: OutboundBuffer,
        diloco_trainer: Optional['DiLoCoTrainer'] = None,
        num_micro_batches: int = DEFAULT_NUM_MICRO_BATCHES,
        node_id: str = "",
    ):
        """
        Initialize compute engine.
        
        Args:
            model: Neural network model (DynamicNeuroLLM)
            inbound: Buffer for incoming activations
            outbound: Buffer for outgoing activations
            diloco_trainer: Optional DiLoCo trainer for local accumulation
            num_micro_batches: Number of micro-batches for 1F1B schedule
            node_id: This node's identifier
        """
        self.model = model
        self.inbound = inbound
        self.outbound = outbound
        self.diloco = diloco_trainer
        self.num_micro_batches = num_micro_batches
        self.node_id = node_id
        
        # Device handling
        self.device = getattr(model, 'device', 'cpu')
        if hasattr(model, 'device'):
            self.device = model.device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Interleaved 1F1B state
        self.pending_backwards: Dict[int, ActivationPacket] = {}
        self.saved_activations: Dict[int, torch.Tensor] = {}  # For backward pass
        
        # Soft overflow state
        self.local_gradient_buffer: Dict[str, torch.Tensor] = {}
        
        # Statistics
        self.stats = ComputeStats()
        
        # State
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._on_forward_complete: Optional[callable] = None
        self._on_backward_complete: Optional[callable] = None
        
    def _check_outbound_pressure(self) -> str:
        """
        Check outbound buffer pressure level.
        
        Returns:
            "ok" - can send normally
            "soft_overflow" - buffer almost full, use local accumulation
            "hard_overflow" - buffer completely full, must discard
        """
        fill_rate = self.outbound.fill_rate
        
        if fill_rate >= self.OUTBOUND_HARD_LIMIT:
            return "hard_overflow"
        elif fill_rate >= self.OUTBOUND_SOFT_LIMIT:
            return "soft_overflow"
        else:
            return "ok"
    
    async def start(self):
        """Start the compute loop."""
        if self.running:
            return
            
        self.running = True
        self._task = asyncio.create_task(self.run())
        logger.info(f"ComputeEngine started on {self.device}")
        
    async def stop(self):
        """Stop the compute loop gracefully."""
        self.running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            
        logger.info("ComputeEngine stopped")
        
    async def run(self):
        """
        Main compute loop with Interleaved 1F1B schedule and Soft Overflow.
        
        This is the heart of the async engine. It:
        1. Pulls packets from inbound queue
        2. Processes forward/backward with interleaving
        3. Handles network congestion gracefully
        """
        logger.info("ComputeEngine run loop started")
        
        while self.running:
            try:
                # Non-blocking get from inbound buffer
                packet = self.inbound.get(timeout=0.01)
                
                if packet is None:
                    # Buffer empty - GPU potentially starved
                    # Small sleep to avoid busy-wait
                    await asyncio.sleep(0.001)
                    continue
                
                # Process the packet
                await self._process_packet(packet)
                
                # Interleaved 1F1B: After warmup, interleave backwards
                if self.stats.forward_count >= self.num_micro_batches:
                    await self._try_interleaved_backward()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ComputeEngine error: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # Prevent tight error loop
                
        # Cleanup
        await self._flush_pending()
        logger.info("ComputeEngine run loop ended")
        
    async def _process_packet(self, packet: ActivationPacket):
        """Process a single activation packet."""
        self.stats.total_steps += 1
        start_time = time.time()
        
        # Track queue wait time
        queue_time = (start_time - packet.timestamp) * 1000
        self.stats.total_queue_time_ms += queue_time
        
        if packet.is_backward:
            outcome = await self._process_backward(packet)
        else:
            outcome = await self._process_forward_with_overflow(packet)
            
        # Track compute time
        compute_time = (time.time() - start_time) * 1000
        self.stats.total_compute_time_ms += compute_time
        
        # Update stats based on outcome
        if outcome == StepOutcome.LOCAL_ONLY:
            self.stats.local_only_steps += 1
        elif outcome == StepOutcome.DROPPED:
            self.stats.dropped_steps += 1
            
        # Periodic logging
        if self.stats.total_steps % 100 == 0:
            self._log_stats()
            
    def _log_stats(self):
        """Log current statistics."""
        logger.info(
            f"ComputeEngine: steps={self.stats.total_steps}, "
            f"forward={self.stats.forward_count}, backward={self.stats.backward_count}, "
            f"local_only={self.stats.local_only_rate:.1%}, "
            f"dropped={self.stats.drop_rate:.1%}, "
            f"avg_compute={self.stats.avg_compute_time_ms:.1f}ms"
        )
        
    async def _process_forward_with_overflow(self, packet: ActivationPacket) -> StepOutcome:
        """
        Process forward pass with soft overflow handling.
        
        The "Don't Stop" Logic:
        1. Always compute forward pass (GPU never waits)
        2. Check outbound pressure AFTER compute
        3. If congested: accumulate locally, skip sending
        4. If ok: queue for outbound
        """
        # ALWAYS compute forward - GPU must never stall
        try:
            with torch.no_grad() if not packet.requires_grad else torch.enable_grad():
                input_tensor = packet.tensor_data.to(self.device)
                
                # Forward through model layers
                if hasattr(self.model, 'forward_my_layers'):
                    output, _ = self.model.forward_my_layers(input_tensor)
                else:
                    output = self.model(input_tensor)
                    
                # Save activation for potential backward pass
                if packet.requires_grad:
                    # Store the raw INPUT to this node's layers so we can
                    # recompute the forward pass with gradient tracking during backward.
                    self.saved_activations[packet.micro_batch_id] = input_tensor.detach().clone()
                    
        except Exception as e:
            logger.error(f"Forward pass error: {e}")
            return StepOutcome.DROPPED
            
        self.stats.forward_count += 1
        
        # Check backpressure AFTER compute
        pressure = self._check_outbound_pressure()
        
        if pressure == "ok":
            # Normal path: queue activation for sending
            return await self._queue_forward_output(packet, output)
        elif pressure == "soft_overflow":
            # SOFT OVERFLOW: Network congested
            return self._handle_soft_overflow(packet, output)
        else:
            # HARD OVERFLOW: Critical congestion
            return self._handle_hard_overflow(packet, output)
            
    async def _queue_forward_output(
        self, 
        packet: ActivationPacket, 
        output: torch.Tensor
    ) -> StepOutcome:
        """Queue forward output for sending."""
        # Determine next layer
        if hasattr(self.model, 'my_layer_ids'):
            next_layer = max(self.model.my_layer_ids) + 1
        else:
            next_layer = packet.target_layer + 1
            
        outbound_packet = ActivationPacket(
            priority=packet.priority,
            session_id=packet.session_id,
            micro_batch_id=packet.micro_batch_id,
            tensor_data=output.cpu(),
            source_node=self.node_id,
            target_layer=next_layer,
            requires_grad=packet.requires_grad,
        )
        
        # Non-blocking put with short timeout
        try:
            await asyncio.wait_for(
                self.outbound.put(outbound_packet),
                timeout=0.01  # 10ms max wait
            )
            return StepOutcome.SENT
        except asyncio.TimeoutError:
            # Couldn't send in time - fall through to soft overflow
            return self._handle_soft_overflow(packet, output)
        except asyncio.QueueFull:
            return self._handle_soft_overflow(packet, output)
            
    def _handle_soft_overflow(
        self, 
        packet: ActivationPacket, 
        output: torch.Tensor
    ) -> StepOutcome:
        """
        Handle soft overflow: accumulate locally, discard activation.
        
        This implements DiLoCo "local training" behavior during congestion.
        """
        logger.debug(
            f"Soft overflow at step {self.stats.total_steps}: "
            f"accumulating locally (outbound: {self.outbound.fill_rate:.1%})"
        )
        
        self.outbound.soft_overflow_count += 1
        
        # Accumulate gradient locally if training
        if packet.requires_grad and self.model.training:
            self._accumulate_local_gradient(output, packet)
            
        # Discard activation - don't try to send
        del output
        
        return StepOutcome.LOCAL_ONLY
        
    def _handle_hard_overflow(
        self, 
        packet: ActivationPacket, 
        output: torch.Tensor
    ) -> StepOutcome:
        """
        Handle hard overflow: must drop step entirely.
        """
        logger.warning(
            f"Hard overflow at step {self.stats.total_steps}: "
            f"dropping step (outbound: {self.outbound.fill_rate:.1%})"
        )
        
        self.outbound.hard_overflow_count += 1
        
        del output
        return StepOutcome.DROPPED
        
    def _accumulate_local_gradient(
        self, 
        output: torch.Tensor, 
        packet: ActivationPacket
    ):
        """
        Accumulate gradient locally during soft overflow.
        
        This is the DiLoCo "local training" behavior - gradients are
        accumulated locally and will be synced during next outer step.
        """
        if not self.model.training:
            return
            
        # If we have upstream gradient, compute local gradient
        if packet.grad_output is not None:
            try:
                output.backward(packet.grad_output.to(self.device))
                
                # Track in DiLoCo trainer if available
                if self.diloco:
                    self.diloco.stats.inner_step_count += 1
            except Exception as e:
                logger.debug(f"Local gradient accumulation error: {e}")
                
    async def _process_backward(self, packet: ActivationPacket) -> StepOutcome:
        """
        Process backward pass.

        We stored the INPUT tensor to our layers (not the output) during forward.
        To get a proper autograd graph we re-run forward with requires_grad=True,
        then call backward through our local layers so their parameters receive
        real gradients.  This is the standard "recompute-and-backward" pattern
        for pipeline parallelism across process boundaries.
        """
        try:
            saved_input = self.saved_activations.pop(packet.micro_batch_id, None)

            if saved_input is not None and packet.grad_output is not None:
                grad_output = packet.grad_output.to(self.device)

                # Re-run forward with gradient tracking so autograd can
                # compute ∂loss/∂(local params).
                recomputed_input = saved_input.to(self.device).requires_grad_(True)

                if hasattr(self.model, 'forward_my_layers'):
                    recomputed_output, _ = self.model.forward_my_layers(recomputed_input)
                else:
                    recomputed_output = self.model(recomputed_input)

                # Backward through local layers using the upstream gradient.
                recomputed_output.backward(grad_output)

                # grad_input can be passed upstream if needed
                grad_input = recomputed_input.grad

        except Exception as e:
            logger.error(f"Backward pass error: {e}")
            return StepOutcome.DROPPED

        self.stats.backward_count += 1

        # Backward passes also respect soft overflow for gradient sending
        pressure = self._check_outbound_pressure()

        if pressure != "ok":
            logger.debug(f"Backward gradient local accumulation")
            return StepOutcome.LOCAL_ONLY

        return StepOutcome.SENT
        
    async def _try_interleaved_backward(self):
        """
        Interleaved 1F1B: Process oldest pending backward if available.
        
        This interleaves backward passes with forward passes to hide
        network latency and maintain GPU utilization.
        """
        if not self.pending_backwards:
            return
            
        # Get oldest micro-batch that needs backward
        oldest_mb = min(self.pending_backwards.keys())
        packet = self.pending_backwards.pop(oldest_mb)
        
        await self._process_backward(packet)
        
    async def _flush_pending(self):
        """Process any remaining pending backward passes."""
        while self.pending_backwards:
            oldest_mb = min(self.pending_backwards.keys())
            packet = self.pending_backwards.pop(oldest_mb)
            await self._process_backward(packet)
            
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        return {
            **self.stats.to_dict(),
            "outbound_fill_rate": self.outbound.fill_rate,
            "inbound_fill_rate": self.inbound.fill_rate,
            "pending_backwards": len(self.pending_backwards),
            "saved_activations": len(self.saved_activations),
            "device": str(self.device),
        }


class InferenceEngine:
    """
    Simplified compute engine for inference-only workloads.
    
    No gradient handling, no backward passes, no soft overflow.
    Just fast forward pass processing.
    """
    
    def __init__(
        self,
        model: Any,
        inbound: ActivationBuffer,
        outbound: OutboundBuffer,
        node_id: str = "",
    ):
        self.model = model
        self.inbound = inbound
        self.outbound = outbound
        self.node_id = node_id
        
        # Device
        self.device = getattr(model, 'device', torch.device('cpu'))
        
        # Stats
        self.requests_processed = 0
        self.total_latency_ms = 0.0
        
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start inference loop."""
        self.running = True
        self._task = asyncio.create_task(self._run())
        
    async def stop(self):
        """Stop inference loop."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
    async def _run(self):
        """Main inference loop."""
        while self.running:
            try:
                packet = self.inbound.get(timeout=0.01)
                if packet is None:
                    await asyncio.sleep(0.001)
                    continue
                    
                start = time.time()
                
                with torch.no_grad():
                    input_tensor = packet.tensor_data.to(self.device)
                    output = self.model(input_tensor)
                    
                # Queue output
                outbound_packet = ActivationPacket(
                    priority=packet.priority,
                    session_id=packet.session_id,
                    micro_batch_id=packet.micro_batch_id,
                    tensor_data=output.cpu(),
                    source_node=self.node_id,
                    target_layer=packet.target_layer + 1,
                )
                
                await self.outbound.put(outbound_packet)
                
                self.requests_processed += 1
                self.total_latency_ms += (time.time() - start) * 1000
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Inference error: {e}")
                
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_latency = (
            self.total_latency_ms / max(1, self.requests_processed)
        )
        return {
            "requests_processed": self.requests_processed,
            "avg_latency_ms": avg_latency,
            "inbound_fill": self.inbound.fill_rate,
            "outbound_fill": self.outbound.fill_rate,
        }

