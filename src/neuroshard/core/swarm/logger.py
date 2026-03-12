"""
Swarm Logger - Structured logging for NeuroShard Swarm Architecture

This module provides:
- Structured JSON logging for metrics and events
- Periodic summary statistics (reduce log spam)
- Role-specific log prefixes (DRIVER, WORKER, VALIDATOR)
- Log level filtering by component
- Training/Inference/Swarm log separation

Usage:
    from neuroshard.core.swarm import SwarmLogger, LogCategory
    
    logger = SwarmLogger("my_node")
    logger.log_training_step(round=100, loss=0.5, tokens=1000)
    logger.log_diloco_sync(inner_steps=500, outer_step=1)
"""

import json
import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
from collections import defaultdict
from datetime import datetime


class LogCategory(Enum):
    """Log categories for filtering and routing."""
    SWARM = "swarm"
    TRAINING = "training"
    INFERENCE = "inference"
    DILOCO = "diloco"
    HEARTBEAT = "heartbeat"
    ROUTING = "routing"
    CHECKPOINT = "checkpoint"
    SYSTEM = "system"


class NodeRole(Enum):
    """Node roles for log prefixes."""
    DRIVER = "DRIVER"
    WORKER = "WORKER"
    VALIDATOR = "VALIDATOR"
    FULL = "FULL"  # Has both embedding and LM head


@dataclass
class LogStats:
    """Accumulated statistics for periodic summaries."""
    training_steps: int = 0
    total_loss: float = 0.0
    tokens_processed: int = 0
    diloco_syncs: int = 0
    activations_sent: int = 0
    activations_received: int = 0
    activations_dropped: int = 0
    local_only_steps: int = 0
    heartbeats_sent: int = 0
    heartbeats_received: int = 0
    peer_updates: int = 0
    checkpoints_saved: int = 0
    
    def reset(self):
        """Reset all stats."""
        self.training_steps = 0
        self.total_loss = 0.0
        self.tokens_processed = 0
        self.diloco_syncs = 0
        self.activations_sent = 0
        self.activations_received = 0
        self.activations_dropped = 0
        self.local_only_steps = 0
        self.heartbeats_sent = 0
        self.heartbeats_received = 0
        self.peer_updates = 0
        self.checkpoints_saved = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SwarmLogger:
    """
    Structured logger for NeuroShard Swarm Architecture.
    
    Features:
    - Role-specific prefixes (DRIVER, WORKER, VALIDATOR)
    - JSON-structured event logging
    - Periodic summary statistics
    - Configurable log levels per category
    """
    
    # Summary interval in seconds
    SUMMARY_INTERVAL = 60
    
    def __init__(
        self,
        node_id: str,
        role: Optional[NodeRole] = None,
        log_level: int = logging.INFO,
        enable_json: bool = False,
        summary_interval: int = 60,
    ):
        """
        Initialize SwarmLogger.
        
        Args:
            node_id: Node identifier (first 8 chars used in prefix)
            role: Node role (DRIVER, WORKER, VALIDATOR, FULL)
            log_level: Base log level
            enable_json: Enable JSON-structured logging
            summary_interval: Seconds between periodic summaries
        """
        self.node_id = node_id
        self.node_id_short = node_id[:8] if node_id else "unknown"
        self.role = role or NodeRole.WORKER
        self.enable_json = enable_json
        self.summary_interval = summary_interval
        
        # Create logger
        self.logger = logging.getLogger(f"neuroshard.swarm.{self.node_id_short}")
        self.logger.setLevel(log_level)
        
        # Category-specific log levels
        self.category_levels: Dict[LogCategory, int] = defaultdict(lambda: log_level)
        
        # Accumulated stats for summaries
        self.stats = LogStats()
        self._stats_lock = threading.Lock()
        
        # Last summary time
        self.last_summary_time = time.time()
        
        # Start summary thread
        self._running = True
        self._summary_thread = threading.Thread(target=self._summary_loop, daemon=True)
        self._summary_thread.start()
    
    def stop(self):
        """Stop the logger and summary thread."""
        self._running = False
    
    def set_role(self, has_embedding: bool, has_lm_head: bool):
        """Set role based on layer assignment."""
        if has_embedding and has_lm_head:
            self.role = NodeRole.FULL
        elif has_embedding:
            self.role = NodeRole.DRIVER
        elif has_lm_head:
            self.role = NodeRole.VALIDATOR
        else:
            self.role = NodeRole.WORKER
    
    def set_category_level(self, category: LogCategory, level: int):
        """Set log level for a specific category."""
        self.category_levels[category] = level
    
    def _get_prefix(self) -> str:
        """Get log prefix based on role."""
        return f"[{self.role.value}:{self.node_id_short}]"
    
    def _format_message(
        self,
        category: LogCategory,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format log message."""
        prefix = self._get_prefix()
        
        if self.enable_json and data:
            # JSON structured logging
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "node_id": self.node_id_short,
                "role": self.role.value,
                "category": category.value,
                "message": message,
                **data
            }
            return json.dumps(log_entry)
        elif data:
            # Human-readable with data
            data_str = ", ".join(f"{k}={v}" for k, v in data.items())
            return f"{prefix} [{category.value}] {message} ({data_str})"
        else:
            return f"{prefix} [{category.value}] {message}"
    
    def _log(
        self,
        level: int,
        category: LogCategory,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Internal log method."""
        # Check category-specific level
        if level < self.category_levels[category]:
            return
        
        formatted = self._format_message(category, message, data)
        self.logger.log(level, formatted)
    
    # ==================== TRAINING LOGS ====================
    
    def log_training_step(
        self,
        round: int,
        loss: float,
        tokens: int = 0,
        duration_ms: float = 0,
    ):
        """Log a training step completion."""
        with self._stats_lock:
            self.stats.training_steps += 1
            self.stats.total_loss += loss
            self.stats.tokens_processed += tokens
        
        self._log(
            logging.INFO,
            LogCategory.TRAINING,
            f"Step #{round} complete",
            {"loss": f"{loss:.4f}", "tokens": tokens, "duration_ms": f"{duration_ms:.1f}"}
        )
    
    def log_training_waiting(self, reason: str = "data"):
        """Log training waiting state."""
        self._log(
            logging.DEBUG,
            LogCategory.TRAINING,
            f"Waiting for {reason}",
        )
    
    # ==================== DiLoCo LOGS ====================
    
    def log_diloco_progress(self, inner_step: int, inner_total: int):
        """Log DiLoCo inner step progress (only on milestones)."""
        # Only log at 10%, 25%, 50%, 75%, 90%, 100%
        progress = inner_step / inner_total
        milestones = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        
        for milestone in milestones:
            if abs(progress - milestone) < 0.01:
                self._log(
                    logging.INFO,
                    LogCategory.DILOCO,
                    f"Inner progress: {int(progress * 100)}%",
                    {"inner_step": inner_step, "inner_total": inner_total}
                )
                break
    
    def log_diloco_sync(self, inner_steps: int, outer_step: int, duration_ms: float = 0):
        """Log DiLoCo outer sync completion."""
        with self._stats_lock:
            self.stats.diloco_syncs += 1
        
        self._log(
            logging.INFO,
            LogCategory.DILOCO,
            f"Outer sync #{outer_step} complete",
            {"inner_steps": inner_steps, "duration_ms": f"{duration_ms:.1f}"}
        )
    
    # ==================== SWARM LOGS ====================
    
    def log_activation_sent(self, target_layer: int, target_node: str):
        """Log activation sent to peer."""
        with self._stats_lock:
            self.stats.activations_sent += 1
        
        self._log(
            logging.DEBUG,
            LogCategory.ROUTING,
            f"Sent activation to layer {target_layer}",
            {"target_node": target_node[:8]}
        )
    
    def log_activation_received(self, source_node: str, layer: int):
        """Log activation received from peer."""
        with self._stats_lock:
            self.stats.activations_received += 1
        
        self._log(
            logging.DEBUG,
            LogCategory.ROUTING,
            f"Received activation for layer {layer}",
            {"source_node": source_node[:8]}
        )
    
    def log_soft_overflow(self, step: int, buffer_fill: float):
        """Log soft overflow (local accumulation)."""
        with self._stats_lock:
            self.stats.local_only_steps += 1
        
        self._log(
            logging.WARNING,
            LogCategory.SWARM,
            f"Soft overflow at step {step}",
            {"buffer_fill": f"{buffer_fill:.1%}"}
        )
    
    def log_hard_overflow(self, step: int, buffer_fill: float):
        """Log hard overflow (dropped step)."""
        with self._stats_lock:
            self.stats.activations_dropped += 1
        
        self._log(
            logging.ERROR,
            LogCategory.SWARM,
            f"Hard overflow at step {step} - step dropped",
            {"buffer_fill": f"{buffer_fill:.1%}"}
        )
    
    def log_failover(self, from_node: str, to_node: str, reason: str):
        """Log routing failover."""
        self._log(
            logging.WARNING,
            LogCategory.ROUTING,
            f"Failover from {from_node[:8]} to {to_node[:8]}",
            {"reason": reason}
        )
    
    # ==================== HEARTBEAT LOGS ====================
    
    def log_heartbeat_sent(self, peers: int):
        """Log heartbeat broadcast."""
        with self._stats_lock:
            self.stats.heartbeats_sent += 1
        
        self._log(
            logging.DEBUG,
            LogCategory.HEARTBEAT,
            f"Heartbeat sent to {peers} peers",
        )
    
    def log_heartbeat_received(self, from_node: str):
        """Log heartbeat received."""
        with self._stats_lock:
            self.stats.heartbeats_received += 1
        
        self._log(
            logging.DEBUG,
            LogCategory.HEARTBEAT,
            f"Heartbeat from {from_node[:8]}",
        )
    
    def log_peer_update(self, node_id: str, capacity: int):
        """Log peer capacity update."""
        with self._stats_lock:
            self.stats.peer_updates += 1
        
        self._log(
            logging.DEBUG,
            LogCategory.HEARTBEAT,
            f"Peer {node_id[:8]} capacity updated",
            {"available_mb": capacity}
        )
    
    # ==================== CHECKPOINT LOGS ====================
    
    def log_checkpoint_saved(self, path: str, size_mb: float):
        """Log checkpoint saved."""
        with self._stats_lock:
            self.stats.checkpoints_saved += 1
        
        self._log(
            logging.INFO,
            LogCategory.CHECKPOINT,
            f"Checkpoint saved",
            {"path": path, "size_mb": f"{size_mb:.1f}"}
        )
    
    def log_checkpoint_restored(self, path: str, round: int):
        """Log checkpoint restored."""
        self._log(
            logging.INFO,
            LogCategory.CHECKPOINT,
            f"Checkpoint restored from round {round}",
            {"path": path}
        )
    
    # ==================== SUMMARY LOGS ====================
    
    def _summary_loop(self):
        """Background thread for periodic summaries."""
        while self._running:
            time.sleep(self.summary_interval)
            self.log_summary()
    
    def log_summary(self):
        """Log periodic summary of accumulated stats."""
        with self._stats_lock:
            stats = self.stats.to_dict()
            
            # Calculate averages
            avg_loss = (
                stats["total_loss"] / stats["training_steps"]
                if stats["training_steps"] > 0
                else 0.0
            )
            
            # Build summary message
            summary_parts = []
            
            if stats["training_steps"] > 0:
                summary_parts.append(
                    f"Training: {stats['training_steps']} steps, "
                    f"avg_loss={avg_loss:.4f}, "
                    f"tokens={stats['tokens_processed']}"
                )
            
            if stats["diloco_syncs"] > 0:
                summary_parts.append(f"DiLoCo: {stats['diloco_syncs']} syncs")
            
            if stats["activations_sent"] > 0 or stats["activations_received"] > 0:
                summary_parts.append(
                    f"Routing: sent={stats['activations_sent']}, "
                    f"recv={stats['activations_received']}, "
                    f"dropped={stats['activations_dropped']}"
                )
            
            if stats["local_only_steps"] > 0:
                summary_parts.append(f"Overflow: {stats['local_only_steps']} local-only steps")
            
            if stats["heartbeats_sent"] > 0:
                summary_parts.append(
                    f"Heartbeat: sent={stats['heartbeats_sent']}, "
                    f"recv={stats['heartbeats_received']}"
                )
            
            if stats["checkpoints_saved"] > 0:
                summary_parts.append(f"Checkpoints: {stats['checkpoints_saved']} saved")
            
            # Log summary if there's anything to report
            if summary_parts:
                self._log(
                    logging.INFO,
                    LogCategory.SYSTEM,
                    f"[SUMMARY] " + " | ".join(summary_parts),
                    {"interval_seconds": self.summary_interval}
                )
            
            # Reset stats
            self.stats.reset()
    
    # ==================== CONVENIENCE METHODS ====================
    
    def info(self, message: str, category: LogCategory = LogCategory.SYSTEM, **data):
        """Log info message."""
        self._log(logging.INFO, category, message, data if data else None)
    
    def warning(self, message: str, category: LogCategory = LogCategory.SYSTEM, **data):
        """Log warning message."""
        self._log(logging.WARNING, category, message, data if data else None)
    
    def error(self, message: str, category: LogCategory = LogCategory.SYSTEM, **data):
        """Log error message."""
        self._log(logging.ERROR, category, message, data if data else None)
    
    def debug(self, message: str, category: LogCategory = LogCategory.SYSTEM, **data):
        """Log debug message."""
        self._log(logging.DEBUG, category, message, data if data else None)


# Global logger instance (can be initialized later)
_swarm_logger: Optional[SwarmLogger] = None


def get_swarm_logger() -> Optional[SwarmLogger]:
    """Get the global swarm logger instance."""
    return _swarm_logger


def init_swarm_logger(
    node_id: str,
    role: Optional[NodeRole] = None,
    **kwargs
) -> SwarmLogger:
    """Initialize the global swarm logger."""
    global _swarm_logger
    _swarm_logger = SwarmLogger(node_id, role, **kwargs)
    return _swarm_logger

