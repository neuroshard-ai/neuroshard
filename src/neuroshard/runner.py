"""
NeuroShard Node Runner

This is the main entry point for running a NeuroShard node.
The node participates in:
1. Training NeuroLLM (our own model, trained from scratch by the network)
2. Inference (generating text using the collective model)
3. Earning NEURO tokens through Proof of Neural Work

TRULY DECENTRALIZED:
- No fixed model phases
- Model size grows with network capacity
- Each node contributes based on available memory
- More memory = more layers = more NEURO rewards
"""

import argparse
import uvicorn
import threading
import torch  # Imported early for API endpoints
import time
import requests
import logging
import logging.handlers  # For RotatingFileHandler
import sys
import os
import socket
import uuid
import hashlib
import math
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

from neuroshard.core.model.dynamic import (
    DynamicNeuroNode,
    create_dynamic_node,
    ContributionMode,
    SpeedTier,
    select_contribution_mode,
    check_layer_growth,
    LayerGrowthManager,
)
from neuroshard.core.model.tokenizer import get_neuro_tokenizer
from neuroshard.core.network.p2p import P2PManager

# Native NeuroShard Architecture Imports
from neuroshard.core.swarm.factory import (
    SwarmEnabledDynamicNode,
    SwarmNodeConfig,
    create_swarm_node,
    create_swarm_node_with_p2p,
)
from neuroshard.core.swarm.quorum import (
    Quorum,
    QuorumMember,
    QuorumLifecycle,
    QuorumType,
    QuorumRole,
    QuorumRegistry,
    QuorumFormationService,
    QuorumTrainer,
    QuorumInferenceRouter,
    AsyncTrainer,
)
from neuroshard.core.consensus.verifier import (
    ProofVerifier,
    PipelineProof,
    CohortSyncProof,
)
from neuroshard.core.network.dht_protocol import DHTProtocol
from neuroshard.core.economics.constants import (
    is_valid_stake_amount,
    is_valid_stake_duration,
    VALIDATOR_BASE_STAKE,
    get_dynamic_validator_stake,
    get_validator_stake_info,
)
from neuroshard.ui.app import STATE, templates
from neuroshard.utils.serialization import deserialize_tensor, serialize_tensor
from neuroshard.grpc_server import start_grpc_background
from neuroshard.version import __version__

# Safe print for Windows frozen GUI mode (where stdout may be None)
_original_print = print

def _safe_print(*args, **kwargs):
    """Print that works even when stdout is None (Windows GUI)."""
    try:
        if sys.stdout is not None:
            _original_print(*args, **kwargs)
    except (AttributeError, OSError, ValueError):
        pass  # Silently ignore - logging will capture it

# Override print globally in this module
print = _safe_print

# Global shutdown flag for clean exit from GUI
_SHUTDOWN_REQUESTED = threading.Event()
_UVICORN_SERVER = None  # Global reference to uvicorn server for shutdown

def request_shutdown():
    """Request graceful shutdown of the node. Called from GUI when stopping."""
    global _UVICORN_SERVER, NEURO_NODE, P2P, QUORUM_TRAINER, CURRENT_QUORUM
    logger.info("[NODE] Shutdown requested...")
    _SHUTDOWN_REQUESTED.set()
    
    # Stop QuorumTrainer FIRST (exit training loop, release GPU)
    # Must stop BEFORE saving checkpoint — otherwise save_checkpoint's .cpu()
    # calls deadlock with the training thread's CUDA forward/backward passes.
    if QUORUM_TRAINER:
        try:
            logger.info("[NODE] Stopping QuorumTrainer...")
            QUORUM_TRAINER.stop()
            logger.info("[NODE] QuorumTrainer stopped.")
        except Exception as e:
            logger.error(f"[NODE] QuorumTrainer shutdown error: {e}")
    
    # Now save checkpoint (GPU is free, no training thread competing)
    if NEURO_NODE:
        try:
            base = getattr(NEURO_NODE, 'base_node', NEURO_NODE)
            if hasattr(base, 'save_checkpoint'):
                logger.info("[NODE] Saving checkpoint...")
                base.save_checkpoint()
                logger.info("[NODE] Checkpoint saved.")
        except Exception as e:
            logger.error(f"[NODE] Checkpoint save error: {e}")
    
    # Leave current quorum gracefully
    if CURRENT_QUORUM and QUORUM_FORMATION:
        try:
            logger.info(f"[NODE] Leaving quorum {CURRENT_QUORUM.quorum_id[:8]}...")
            # Leave quorum via formation service (has the leave_quorum method)
            node_id = NEURO_NODE.node_id if NEURO_NODE else ""
            QUORUM_FORMATION.leave_quorum(node_id, CURRENT_QUORUM.quorum_id)
        except Exception as e:
            logger.error(f"[NODE] Quorum leave error: {e}")
    
    # Stop gRPC server first (releases port)
    try:
        from neuroshard.grpc_server import stop_grpc
        stop_grpc(timeout=3.0)
    except Exception as e:
        logger.error(f"[NODE] gRPC shutdown error: {e}")
    
    # Stop the node first (sets is_running = False)
    if NEURO_NODE:
        try:
            logger.info("[NODE] Stopping node...")
            # Get base node for SwarmEnabledDynamicNode
            base = getattr(NEURO_NODE, 'base_node', NEURO_NODE)
            if hasattr(base, 'stop'):
                base.stop()
            if hasattr(NEURO_NODE, 'stop') and NEURO_NODE != base:
                NEURO_NODE.stop()
        except Exception as e:
            logger.error(f"[NODE] Node stop error: {e}")
    
    # Stop swarm components if enabled
    if NEURO_NODE and hasattr(NEURO_NODE, 'stop_swarm_sync'):
        try:
            logger.info("[NODE] Stopping swarm components...")
            NEURO_NODE.stop_swarm_sync()
            logger.info("[NODE] Swarm components stopped.")
        except Exception as e:
            logger.error(f"[NODE] Swarm shutdown error: {e}")
    
    # Save checkpoint before shutting down
    if NEURO_NODE:
        try:
            logger.info("[NODE] Saving checkpoint before shutdown...")
            # Force synchronous save during shutdown to ensure it completes
            NEURO_NODE._save_checkpoint(async_save=False)
            logger.info("[NODE] Checkpoint saved.")
        except Exception as e:
            logger.error(f"[NODE] Failed to save checkpoint: {e}")
        
        # Wait for any ongoing async saves to complete (if lock exists)
        try:
            from neuroshard.core.model.dynamic import DynamicNeuroNode
            if hasattr(DynamicNeuroNode, '_checkpoint_save_lock'):
                # Try to acquire the lock (will wait if async save in progress)
                if DynamicNeuroNode._checkpoint_save_lock.acquire(timeout=30):
                    DynamicNeuroNode._checkpoint_save_lock.release()
                    logger.info("[NODE] All checkpoint saves completed.")
        except Exception as e:
            logger.debug(f"[NODE] Checkpoint lock not available: {e}")
        
        # CRITICAL: Free memory by deleting model and data
        try:
            logger.info("[NODE] Freeing memory...")
            
            # Clear genesis data
            if hasattr(NEURO_NODE, 'genesis_loader') and NEURO_NODE.genesis_loader:
                NEURO_NODE.genesis_loader.loaded_shards.clear()
                NEURO_NODE.genesis_loader._prefetch_ready.clear()
                NEURO_NODE.genesis_loader.current_dataset = None
            
            # Get base node (for SwarmEnabledDynamicNode) or use directly
            base = getattr(NEURO_NODE, 'base_node', NEURO_NODE)
            
            # Delete optimizer (holds 2x model params in memory for Adam)
            if hasattr(base, 'optimizer') and base.optimizer is not None:
                del base.optimizer
            
            # Delete model
            if hasattr(base, 'model') and base.model is not None:
                del base.model
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear GPU cache if applicable
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("[NODE] Cleared CUDA cache")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                logger.info("[NODE] Cleared MPS cache")
            
            logger.info("[NODE] Memory freed")
        except Exception as e:
            logger.error(f"[NODE] Memory cleanup error: {e}")
    
    # Stop P2P manager (stops background threads)
    if P2P:
        try:
            P2P.stop()
        except Exception as e:
            logger.error(f"[NODE] P2P stop error: {e}")
    
    # Stop uvicorn server
    if _UVICORN_SERVER:
        logger.info("[NODE] Stopping HTTP server...")
        _UVICORN_SERVER.should_exit = True
    
    # FORCE EXIT: Always force exit after 3 seconds regardless
    # This handles nohup, daemon, and any other run mode
    def force_exit():
        import time as t_module
        import os
        import signal
        t_module.sleep(3.0)
        logger.warning("[NODE] Force exiting (server didn't stop gracefully)...")
        # Try SIGTERM first (graceful)
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            pass
        t_module.sleep(0.5)
        # If still running, force kill
        logger.warning("[NODE] Forcing process termination...")
        os._exit(0)  # Force exit without cleanup
    
    # Use non-daemon thread to ensure force_exit runs to completion
    force_thread = threading.Thread(target=force_exit, daemon=False)
    force_thread.start()
    logger.info("[NODE] Force exit scheduled in 3 seconds...")
    
    # Reset globals so next run starts fresh
    NEURO_NODE = None
    P2P = None
    _UVICORN_SERVER = None

# Configure Logging - ensure all loggers use our format
# Clear any existing handlers first to prevent duplicates
root_logger = logging.getLogger()
root_logger.handlers = []  # Clear existing handlers
root_logger.setLevel(logging.INFO)

# --- In-memory log buffer for dashboard ---
from collections import deque
from datetime import datetime

# Circular buffer to store recent logs (max 500 entries)
_LOG_BUFFER = deque(maxlen=500)
_LOG_BUFFER_LOCK = threading.Lock()

class MemoryLogHandler(logging.Handler):
    """Custom handler that stores logs in memory for dashboard API."""
    
    # Auto-incrementing log ID for reliable polling
    _log_id_counter = 0
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # Store both display timestamp and epoch for sorting
            epoch_ms = int(record.created * 1000)
            timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
            
            # Determine log type for filtering
            log_type = 'info'
            msg_lower = msg.lower()
            if 'neuro' in msg_lower and ('earned' in msg_lower or 'reward' in msg_lower or '+' in msg):
                log_type = 'neuro'
            elif 'error' in msg_lower or record.levelno >= logging.ERROR:
                log_type = 'error'
            elif 'training' in msg_lower or 'diloco' in msg_lower or 'gradient' in msg_lower or 'batch' in msg_lower:
                log_type = 'training'
            elif record.levelno >= logging.WARNING:
                log_type = 'warning'
            
            with _LOG_BUFFER_LOCK:
                MemoryLogHandler._log_id_counter += 1
                _LOG_BUFFER.append({
                    'id': MemoryLogHandler._log_id_counter,
                    'epoch': epoch_ms,
                    'timestamp': timestamp,
                    'message': msg,
                    'type': log_type,
                    'level': record.levelname,
                })
        except Exception:
            pass  # Never fail logging

# Windows GUI apps (frozen) may have None stdout/stderr
# Create a safe handler that won't crash
def _create_safe_handler():
    """Create a logging handler that works even when stdout is None (Windows GUI)."""
    # Check if stdout is usable
    if sys.stdout is not None and hasattr(sys.stdout, 'write'):
        try:
            # Test if it's actually writable
            sys.stdout.write('')
            sys.stdout.flush()
            return logging.StreamHandler(sys.stdout)
        except (AttributeError, OSError, ValueError):
            pass
    
    # Fallback: log to file in .neuroshard directory
    log_dir = os.path.join(os.path.expanduser("~"), ".neuroshard")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "node.log")
    
    # Rotate logs - keep last 5MB
    return logging.handlers.RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2, encoding='utf-8')

handler = _create_safe_handler()
handler.setFormatter(logging.Formatter('[%(asctime)s] [NODE] %(message)s', datefmt='%H:%M:%S'))
root_logger.addHandler(handler)

# Add memory handler for dashboard logs API
memory_handler = MemoryLogHandler()
memory_handler.setFormatter(logging.Formatter('%(message)s'))
memory_handler.setLevel(logging.INFO)
root_logger.addHandler(memory_handler)

# Also configure neuroshard module loggers explicitly
for module in ['neuroshard.core.p2p', 'neuroshard.core.ledger', 'neuroshard.core.dynamic_model',
               'neuroshard.core.distributed_training', 'neuroshard.core.dht_service']:
    module_logger = logging.getLogger(module)
    module_logger.setLevel(logging.INFO)
    module_logger.propagate = True  # Propagate to root logger

# Create logger for this module
logger = logging.getLogger(__name__)

# --- Main API App ---
node_app = FastAPI(title="NeuroShard Node", version=__version__)
# Serve dashboard at root
from fastapi.responses import HTMLResponse

@node_app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serve the main dashboard at root."""
    return templates.TemplateResponse("index.html", {"request": request})

# Shared State
NEURO_NODE: Optional[DynamicNeuroNode] = None
P2P: Optional[P2PManager] = None
SESSION_TIMESTAMPS = {}

# Quorum System Components
QUORUM_REGISTRY: Optional[QuorumRegistry] = None
QUORUM_FORMATION: Optional[QuorumFormationService] = None
QUORUM_TRAINER: Optional[QuorumTrainer] = None
QUORUM_INFERENCE_ROUTER: Optional[QuorumInferenceRouter] = None
PROOF_VERIFIER: Optional[ProofVerifier] = None
DHT_PROTOCOL: Optional[DHTProtocol] = None
CURRENT_QUORUM: Optional[Quorum] = None

# Contribution Mode (per whitepaper: PIPELINE, ASYNC, DATA, VERIFY, INFERENCE, IDLE)
CURRENT_CONTRIBUTION_MODE: ContributionMode = ContributionMode.IDLE
ASYNC_TRAINER = None  # AsyncTrainer instance for nodes in ASYNC mode
LAYER_GROWTH_MANAGER: Optional[LayerGrowthManager] = None

def get_app():
    return node_app


class InferenceRequest(BaseModel):
    tensor_data: str
    request_id: str
    session_id: Optional[str] = None
    sender_reputation: float = 100.0


class TextGenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9


class TrainingDataRequest(BaseModel):
    text: str
    apply_dp: bool = True  # Apply differential privacy


# ==================== INFERENCE ENDPOINTS ====================

@node_app.post("/generate_text")
async def generate_text(req: TextGenerationRequest):
    """
    Generate text using NeuroLLM.
    
    Note: Early in the network's life, output will be mostly random.
    As more users train the model, quality will improve!
    """
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    try:
        generated = NEURO_NODE.generate(
            prompt=req.prompt,
            max_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )
        
        STATE["processed_count"] = STATE.get("processed_count", 0) + 1
        STATE["token_count"] = NEURO_NODE.total_tokens_processed
        
        return {
            "text": generated,
            "my_layers": NEURO_NODE.my_layer_ids,
            "total_training_rounds": NEURO_NODE.total_training_rounds,
            "note": "Quality improves as more users train the model!"
        }
        
    except Exception as e:
        return {"error": str(e)}


@node_app.post("/forward")
async def forward(req: InferenceRequest):
    """Forward pass for distributed inference pipeline."""
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    try:
        STATE["processed_count"] = STATE.get("processed_count", 0) + 1
        
        if req.session_id:
            SESSION_TIMESTAMPS[req.session_id] = time.time()
        
        # Deserialize input
        input_tensor = deserialize_tensor(req.tensor_data)
        
        # Forward through NeuroLLM
        output = NEURO_NODE.forward(input_tensor, session_id=req.session_id)
        
        # Update token count
        STATE["token_count"] = NEURO_NODE.total_tokens_processed
        
        # Return result (NeuroLLM is always a full model, no pipeline needed)
        return {"result": serialize_tensor(output)}
        
    except Exception as e:
        return {"error": str(e)}


# ==================== TRAINING ENDPOINTS ====================

@node_app.post("/contribute_data")
async def contribute_training_data(req: TrainingDataRequest):
    """
    Contribute training data to help train NeuroLLM.
    
    Your data is processed locally with differential privacy.
    You earn NEURO tokens for contributing!
    """
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    if not NEURO_NODE.enable_training:
        raise HTTPException(status_code=400, detail="Training not enabled on this node")
    
    try:
        tokens_added = NEURO_NODE.contribute_training_data(req.text, apply_dp=req.apply_dp)
        
        data_stats = NEURO_NODE.data_manager.get_stats() if NEURO_NODE.data_manager else {}
        
        return {
            "success": True,
            "message": "Data added to training buffer",
            "tokens_added": tokens_added or 0,
            "buffer_size": data_stats.get("buffer_size", 0),
            "total_tokens": data_stats.get("total_tokens", 0),
        }
        
    except Exception as e:
        return {"error": str(e)}


@node_app.post("/train_step")
async def trigger_train_step():
    """Manually trigger a training step (for testing)."""
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    loss = NEURO_NODE.train_step()
    
    if loss is None:
        return {"success": False, "message": "Not enough training data in buffer"}
    
    return {
        "success": True,
        "loss": loss,
        "total_training_rounds": NEURO_NODE.total_training_rounds
    }


@node_app.get("/training_status")
async def get_training_status():
    """Get current training status."""
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    # Sanitize loss for JSON
    current_loss = NEURO_NODE.current_loss
    if math.isinf(current_loss) or math.isnan(current_loss):
        current_loss = None
    
    return {
        "training_enabled": NEURO_NODE.enable_training,
        "total_training_rounds": NEURO_NODE.total_training_rounds,
        "current_loss": current_loss,
        "training_contributions": NEURO_NODE.training_contribution_count,
        "data_buffer": NEURO_NODE.data_manager.get_stats() if NEURO_NODE.data_manager else None,
        "my_layers": NEURO_NODE.my_layer_ids,
    }


@node_app.get("/api/training/global")
async def get_global_training_status():
    """
    Get GLOBAL training verification status.
    
    This endpoint answers the question: "Is the distributed training ACTUALLY working?"
    
    Key metrics:
    - training_verified: True if we can confirm the model is improving
    - is_converging: True if the network appears to be converging
    - hash_agreement_rate: % of nodes with the same model hash (should be 100%)
    - global_avg_loss: Average loss across all network nodes
    - sync_success_rate: % of gradient syncs that succeeded
    
    If hash_agreement_rate < 100%, nodes have diverged and training is NOT coordinated!
    """
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    # Get global status from swarm-enabled node
    if hasattr(NEURO_NODE, 'get_global_training_status'):
        global_status = NEURO_NODE.get_global_training_status()
    else:
        # Fallback for non-swarm nodes
        global_status = {
            "error": "Node does not support global training tracking",
            "training_verified": False,
            "is_converging": False,
        }
    
    # Add local context (sanitize float values for JSON)
    current_loss = NEURO_NODE.current_loss
    if math.isinf(current_loss) or math.isnan(current_loss):
        current_loss = None
    
    # Get model hash from global tracker if available
    model_hash = ""
    if hasattr(NEURO_NODE, '_global_tracker') and NEURO_NODE._global_tracker:
        local_status = NEURO_NODE._global_tracker.get_local_status()
        model_hash = local_status.get('model_hash', '')
    
    global_status["local"] = {
        "node_id": NEURO_NODE.node_id[:16],
        "training_rounds": NEURO_NODE.total_training_rounds,
        "current_loss": current_loss,
        "is_training": NEURO_NODE.enable_training,
        "model_hash": model_hash,
    }
    
    # Add DiLoCo status if available
    if hasattr(NEURO_NODE, 'get_diloco_progress'):
        global_status["diloco"] = NEURO_NODE.get_diloco_progress()
    
    return global_status


@node_app.get("/api/training/verify")
async def verify_training():
    """
    Quick verification endpoint - answers: "Is training working?"
    
    Returns a simple yes/no with explanation.
    """
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    if not NEURO_NODE.enable_training:
        return {
            "is_working": False,
            "reason": "Training is not enabled on this node",
            "action": "Start the node with --train flag",
        }
    
    # Check if we have enough training data
    if NEURO_NODE.total_training_rounds < 10:
        return {
            "is_working": "insufficient_data",
            "reason": f"Only {NEURO_NODE.total_training_rounds} training steps completed",
            "action": "Wait for more training steps (need 10+ for verification)",
        }
    
    # Get global status
    if hasattr(NEURO_NODE, 'get_global_training_status'):
        global_status = NEURO_NODE.get_global_training_status()
        
        is_working = global_status.get("training_verified", False)
        is_converging = global_status.get("is_converging", False)
        hash_agreement = global_status.get("hash_agreement_rate", 0)
        
        if is_working and is_converging:
            return {
                "is_working": True,
                "reason": "Training verified! Loss is decreasing and network is converging.",
                "metrics": {
                    "loss_trend": global_status.get("loss_trend", "unknown"),
                    "hash_agreement": f"{hash_agreement*100:.1f}%",
                    "global_loss": global_status.get("global_avg_loss", 0),
                },
            }
        elif not is_converging and hash_agreement < 0.5:
            return {
                "is_working": False,
                "reason": f"Network NOT converging! Only {hash_agreement*100:.1f}% hash agreement.",
                "action": "Nodes have diverged. Check gradient sync is working.",
            }
        else:
            return {
                "is_working": "partial",
                "reason": "Training running but not yet verified as improving.",
                "action": "Continue training - need more data for verification.",
            }
    
    # Fallback: check if loss is decreasing
    loss = NEURO_NODE.current_loss
    if loss < 1.0:
        return {
            "is_working": True,
            "reason": f"Loss is {loss:.4f} which is reasonable for training.",
        }
    else:
        return {
            "is_working": "unknown",
            "reason": "Cannot verify without global tracker.",
            "action": "Check loss values in logs - should be decreasing.",
        }


@node_app.get("/api/training/history")
async def get_local_training_history():
    """
    Get LOCAL loss history to verify model is improving.
    
    Returns loss checkpoints recorded during training.
    Use this to see if YOUR node's training is working.
    """
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    result = {
        "total_steps": NEURO_NODE.total_training_rounds,
        "current_loss": NEURO_NODE.current_loss if NEURO_NODE.current_loss != float('inf') else None,
        "loss_checkpoints": [],
        "loss_trend": "unknown",
        "improvement_percent": 0.0,
        "training_verified": False,
        "analysis": {},
    }
    
    # Get loss checkpoints from global tracker
    if hasattr(NEURO_NODE, '_global_tracker') and NEURO_NODE._global_tracker:
        tracker = NEURO_NODE._global_tracker
        
        # Get loss checkpoints (list of (step, loss) tuples)
        checkpoints = getattr(tracker, '_loss_checkpoints', [])
        result["loss_checkpoints"] = [
            {"step": step, "loss": round(loss, 4)} 
            for step, loss in checkpoints
        ]
        
        # Analyze trend
        if len(checkpoints) >= 5:
            losses = [loss for _, loss in checkpoints]
            
            # Compare first 20% to last 20%
            n = len(losses)
            first_n = max(1, n // 5)
            first_avg = sum(losses[:first_n]) / first_n
            last_avg = sum(losses[-first_n:]) / first_n
            
            if first_avg > 0:
                improvement = (first_avg - last_avg) / first_avg * 100
                result["improvement_percent"] = round(improvement, 2)
                
                if improvement > 10:
                    result["loss_trend"] = "improving_strongly"
                    result["training_verified"] = True
                elif improvement > 2:
                    result["loss_trend"] = "improving"
                    result["training_verified"] = True
                elif improvement > -2:
                    result["loss_trend"] = "stable"
                    result["training_verified"] = n > 20  # Stable after many steps = converged
                elif improvement > -10:
                    result["loss_trend"] = "degrading_slightly"
                else:
                    result["loss_trend"] = "degrading"
            
            result["analysis"] = {
                "data_points": n,
                "first_avg_loss": round(first_avg, 4),
                "last_avg_loss": round(last_avg, 4),
                "min_loss_seen": round(min(losses), 4),
                "max_loss_seen": round(max(losses), 4),
                "expected_initial_loss": "~10-11 (random init for 50k vocab)",
                "good_loss_range": "< 4.0 (perplexity < 55)",
                "great_loss_range": "< 2.5 (perplexity < 12)",
            }
    else:
        result["analysis"]["note"] = "Global tracker not initialized - restart node to enable"
    
    return result


# ==================== CONTRIBUTION MODE ENDPOINT ====================

@node_app.get("/api/contribution_mode")
async def get_contribution_mode():
    """
    Get this node's current contribution mode.
    
    Modes per whitepaper:
    - pipeline: Real-time quorum member (synchronous training)
    - async: Offline training, submit gradients periodically
    - data: Store and serve Genesis training data
    - verify: Re-execute proofs for verification
    - inference: Serve inference requests only
    - idle: Available but not actively contributing
    
    Returns current mode and relevant details.
    """
    return {
        "mode": CURRENT_CONTRIBUTION_MODE.value if CURRENT_CONTRIBUTION_MODE else "idle",
        "speed_tier": STATE.get("speed_tier", "unknown"),
        "has_quorum": CURRENT_QUORUM is not None and CURRENT_QUORUM.lifecycle == QuorumLifecycle.ACTIVE if CURRENT_QUORUM else False,
        "quorum_id": CURRENT_QUORUM.quorum_id[:16] if CURRENT_QUORUM else None,
        "training_mode": STATE.get("training_mode", "unknown"),
        "training_status": STATE.get("training_status", "unknown"),
        "description": {
            "pipeline": "Real-time training in speed-matched quorum",
            "async": "Offline training with periodic gradient submission",
            "data": "Serving Genesis training data shards",
            "verify": "Verifying PoNW proofs from other nodes",
            "inference": "Serving inference requests only",
            "idle": "Available but not actively contributing",
        }.get(CURRENT_CONTRIBUTION_MODE.value if CURRENT_CONTRIBUTION_MODE else "idle", "Unknown mode"),
    }


# ==================== STATS & PONW ENDPOINTS ====================

@node_app.get("/api/stats")
async def get_api_stats():
    """Endpoint for GUI to fetch local node stats."""
    import math
    import asyncio
    import os
    
    # Yield to event loop to ensure responsiveness
    await asyncio.sleep(0)
    
    # Get actual system resource usage
    system_stats = {}
    try:
        import psutil
        # CPU usage (system-wide percentage)
        system_stats["cpu_percent"] = psutil.cpu_percent(interval=None)  # Non-blocking
        
        # Memory usage (system-wide)
        mem = psutil.virtual_memory()
        system_stats["ram_used_gb"] = round(mem.used / (1024**3), 2)
        system_stats["ram_total_gb"] = round(mem.total / (1024**3), 2)
        system_stats["ram_percent"] = mem.percent
        
        # Process-specific memory
        process = psutil.Process(os.getpid())
        system_stats["process_ram_mb"] = round(process.memory_info().rss / (1024**2), 1)
    except Exception:
        pass
    
    # Start with basic stats from STATE
    stats = {
        "peer_count": STATE.get("peer_count", 0),
        "processed_count": STATE.get("processed_count", 0),
        "training_status": STATE.get("training_status", "idle"),
        # Actual system resource usage
        "system": system_stats,
        # Resource throttle info
        "throttle": {
            "cpu_ratio": STATE.get("throttle_cpu_ratio", 1.0),
            "ram_ratio": STATE.get("throttle_ram_ratio", 1.0),
            "effective": STATE.get("throttle_effective", 1.0),
            "interval_seconds": STATE.get("throttle_interval", 2.0),
            "max_steps_per_min": STATE.get("throttle_max_steps", 30),
        },
    }
    
    if NEURO_NODE:
        # Run get_stats in executor to not block event loop
        loop = asyncio.get_event_loop()
        node_stats = await loop.run_in_executor(None, NEURO_NODE.get_stats)
        
        # Handle infinity/None values (not JSON serializable)
        current_loss = node_stats.get("current_loss")
        if current_loss is None or (isinstance(current_loss, float) and (math.isinf(current_loss) or math.isnan(current_loss))):
            current_loss = None  # Use None for JSON compatibility
        
        # Determine role string for display
        has_embedding = node_stats.get("has_embedding", False)
        has_lm_head = node_stats.get("has_lm_head", False)
        if has_embedding and has_lm_head:
            role = "Full Node (Driver + Validator)"
        elif has_embedding:
            role = "Driver"
        elif has_lm_head:
            role = "Validator"
        else:
            role = "Worker"
        
        stats.update({
            # My contribution
            "my_layers": node_stats.get("my_layers", []),
            "my_params_m": node_stats.get("my_params", 0) / 1e6,
            "has_embedding": has_embedding,
            "has_lm_head": has_lm_head,
            "role": role,
            "available_memory_mb": node_stats.get("available_memory_mb", 0),
            "reward_multiplier": node_stats.get("reward_multiplier", 1.0),
            
            # Network stats
            "network_layers": node_stats.get("network_layers", 0),
            "network_params_m": node_stats.get("network_params", 0) / 1e6,
            "network_nodes": node_stats.get("network_nodes", 0),
            "contribution_ratio": node_stats.get("contribution_ratio", 0),
            
            # Training stats - use CUMULATIVE values from NEURO_NODE, not delta from STATE
            "training_enabled": NEURO_NODE.enable_training,
            "training_rounds": node_stats.get("total_training_rounds", 0),
            "token_count": node_stats.get("total_tokens_processed", 0),  # Cumulative tokens
            "current_loss": current_loss,
            "data_buffer_size": node_stats.get("data_buffer_size", 0),
            
            # Data shard stats (if Driver)
            "shard_stats": node_stats.get("shard_stats", {}),
            
            # Device info
            "device": NEURO_NODE.device,
            
            # Instance info (for multi-node support)
            "instance_id": getattr(NEURO_NODE, 'instance_id', None),
        })
        
        # Add DiLoCo progress
        diloco = NEURO_NODE.get_diloco_progress()
        if diloco.get("enabled", False):
            stats["diloco"] = {
                "inner_step": diloco.get("inner_step_count", 0),
                "inner_total": diloco.get("inner_steps_total", 500),
                "progress": diloco.get("progress", 0.0),
                "outer_step": diloco.get("outer_step_count", 0),
            }
    else:
        # Node not ready yet or observer mode
        stats["token_count"] = 0
        stats["training_rounds"] = 0
        stats["observer_mode"] = STATE.get("observer_mode", False)
        
        # Observer mode specific stats
        if STATE.get("observer_mode"):
            stats["role"] = "Observer"
            stats["training_enabled"] = False
            stats["my_layers"] = []
            stats["my_params_m"] = 0
    
    # Add version
    stats["version"] = __version__
    
    # Add current config settings (for UI sliders)
    stats["config"] = {
        "cpu_threads": STATE.get("config_cpu_threads"),
        "memory_mb": STATE.get("config_memory_mb"),
        "storage_mb": STATE.get("config_storage_mb", 100),  # Default 100MB
    }
    
    # Architecture V2 - Contribution Mode & Quorum Status
    stats["contribution_mode"] = STATE.get("contribution_mode", "idle")
    stats["speed_tier"] = STATE.get("speed_tier", "tier5")
    stats["training_mode"] = STATE.get("training_mode", "disabled")
    
    # Quorum status
    stats["quorum_id"] = STATE.get("quorum_id")
    stats["quorum_lifecycle"] = STATE.get("quorum_lifecycle", "none")
    stats["quorum_members"] = STATE.get("quorum_members", 0)
    stats["quorum_batches"] = STATE.get("quorum_batches", 0)
    stats["quorum_sync_round"] = STATE.get("quorum_sync_round", 0)
    
    # Async training stats (if in async mode)
    stats["async_batches"] = STATE.get("async_batches", 0)
    stats["async_syncs"] = STATE.get("async_syncs", 0)
    
    # Layer growth status
    stats["growth_phase"] = STATE.get("growth_phase", "none")
    stats["growth_target_layers"] = STATE.get("growth_target_layers")
    
    return stats


@node_app.get("/api/node/architecture")
async def get_node_architecture():
    """
    Get this node's current architecture.
    
    Used by other nodes to query network architecture when rejoining.
    This enables smart architecture reconciliation across the network.
    """
    if not NEURO_NODE or not NEURO_NODE.model:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    arch = NEURO_NODE.model.architecture
    
    return {
        "hidden_dim": arch.hidden_dim,
        "intermediate_dim": arch.intermediate_dim,
        "num_layers": arch.num_layers,
        "num_heads": arch.num_heads,
        "num_kv_heads": arch.num_kv_heads,
        "estimated_params": arch.estimate_params(),
        "estimated_memory_mb": arch.estimate_memory_mb(),
        "architecture_version": getattr(NEURO_NODE.layer_pool, 'architecture_version', 1),
    }


@node_app.get("/api/market")
async def get_market_stats():
    """
    Get real-time inference market statistics.
    
    Returns current price, supply, demand, utilization.
    """
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    return P2P.ledger.get_inference_market_stats()


@node_app.post("/api/market/register")
async def register_inference_capacity(
    tokens_per_second: int,
    min_price: float = 0.0
):
    """
    Register this node's inference capacity with the market.
    
    Nodes should call this when idle/available to serve inference.
    Call withdraw endpoint when switching to training.
    """
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    P2P.ledger.register_inference_capacity(
        tokens_per_second=tokens_per_second,
        min_price=min_price
    )
    
    return {"status": "registered", "tokens_per_second": tokens_per_second, "min_price": min_price}


@node_app.post("/api/market/withdraw")
async def withdraw_inference_capacity():
    """
    Withdraw this node from inference market.
    
    Call this when switching to training.
    """
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    P2P.ledger.withdraw_inference_capacity()
    
    return {"status": "withdrawn"}


# ==================== DISTRIBUTED INFERENCE MARKETPLACE ====================

class MarketplaceSubmitRequest(BaseModel):
    """User submits inference request to marketplace."""
    prompt: str
    max_tokens: int = 100
    max_price: float = 1.0
    driver_node_id: Optional[str] = None  # Optional: specify driver, else round-robin


class DriverPromptRequest(BaseModel):
    """User sends encrypted prompt directly to driver."""
    encrypted_prompt: str
    user_id: str


@node_app.post("/api/market/submit")
async def submit_marketplace_request(req: MarketplaceSubmitRequest):
    """
    Submit inference request to marketplace (USER API).
    
    Flow:
    1. User submits request with prompt
    2. Marketplace locks price, assigns driver
    3. User sends encrypted prompt to driver
    4. Driver processes, returns result
    
    Returns:
        request_id, locked_price, driver_node_id
    """
    if not NEURO_NODE or not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    if not hasattr(P2P.ledger, 'inference_market'):
        raise HTTPException(status_code=503, detail="Marketplace not available")
    
    market = P2P.ledger.inference_market
    
    # Choose driver (round-robin if not specified)
    driver_node_id = req.driver_node_id
    
    if not driver_node_id:
        # Find a driver node from layer pool
        if NEURO_NODE.layer_pool:
            route = NEURO_NODE.layer_pool.get_pipeline_route()
            if route and len(route) > 0:
                # First layer should be embedding (driver)
                driver_node_id = route[0][1].split(':')[0] if ':' in route[0][1] else NEURO_NODE.node_id
            else:
                # Fallback to this node if it's a driver
                if NEURO_NODE.model.has_embedding:
                    driver_node_id = NEURO_NODE.node_id
                else:
                    raise HTTPException(status_code=503, detail="No driver nodes available")
        else:
            # Single node mode
            driver_node_id = NEURO_NODE.node_id
    
    # Sign request with node's ECDSA key (authorizes payment)
    from neuroshard.core.crypto.ecdsa import sign_message
    signature_payload = f"{NEURO_NODE.node_id}:{driver_node_id}:{req.max_tokens}:{req.max_price}"
    user_signature = sign_message(signature_payload, NEURO_NODE.node_token)
    
    # Submit to marketplace (without prompt - privacy!)
    success, request_id, locked_price = market.submit_request(
        user_id=NEURO_NODE.node_id,  # For testing, use node ID as user ID
        driver_node_id=driver_node_id,
        tokens_requested=req.max_tokens,
        max_price=req.max_price,
        user_signature=user_signature,
        priority=0
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Request rejected (price too high or market full)")
    
    # Encrypt prompt for driver
    from neuroshard.core.network.encrypted_channel import PromptEncryption
    encrypted_prompt = PromptEncryption.encrypt_prompt(req.prompt, request_id)
    
    # If we are the driver, add to our own queue
    if driver_node_id == NEURO_NODE.node_id and hasattr(NEURO_NODE, 'prompt_queue'):
        from neuroshard.core.network.encrypted_channel import EncryptedPrompt
        import time
        
        NEURO_NODE.prompt_queue.add_prompt(EncryptedPrompt(
            request_id=request_id,
            encrypted_data=encrypted_prompt,
            timestamp=time.time(),
            user_id=NEURO_NODE.node_id
        ))
        logger.info(f"[API] ✓ Added encrypted prompt to local driver queue")
    
    return {
        "request_id": request_id,
        "locked_price": locked_price,
        "driver_node_id": driver_node_id,
        "encrypted_prompt": encrypted_prompt,  # User should send this to driver
        "instructions": f"POST encrypted_prompt to /api/driver/prompt/{request_id} on driver node"
    }


@node_app.post("/api/driver/prompt/{request_id}")
async def submit_encrypted_prompt(request_id: str, req: DriverPromptRequest):
    """
    User sends encrypted prompt to driver node (PRIVACY CHANNEL).
    
    This endpoint is called on the DRIVER node, not the marketplace.
    Prompt is encrypted - only driver can decrypt it.
    """
    if not NEURO_NODE or not NEURO_NODE.model.has_embedding:
        raise HTTPException(status_code=403, detail="This node is not a driver")
    
    if not hasattr(NEURO_NODE, 'prompt_queue'):
        raise HTTPException(status_code=503, detail="Driver not initialized")
    
    # Add to prompt queue
    from neuroshard.core.network.encrypted_channel import EncryptedPrompt
    import time
    
    prompt = EncryptedPrompt(
        request_id=request_id,
        encrypted_data=req.encrypted_prompt,
        timestamp=time.time(),
        user_id=req.user_id
    )
    
    success = NEURO_NODE.prompt_queue.add_prompt(prompt)
    
    if not success:
        raise HTTPException(status_code=503, detail="Prompt queue full")
    
    return {
        "status": "success",
        "message": f"Encrypted prompt queued for request {request_id[:8]}...",
        "queue_position": len(NEURO_NODE.prompt_queue.prompts)
    }


@node_app.get("/api/market/request/{request_id}")
async def get_request_status(request_id: str):
    """
    Get status of inference request.
    
    Returns:
        status, progress, eta, result (if completed)
    """
    if not NEURO_NODE or not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    if not hasattr(P2P.ledger, 'inference_market'):
        raise HTTPException(status_code=503, detail="Marketplace not available")
    
    market = P2P.ledger.inference_market
    request = market.get_request(request_id)
    
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    
    # Get result from marketplace storage
    result_text = market.get_result(request_id)
    
    return {
        "request_id": request_id,
        "status": request.status,
        "locked_price": request.locked_price,
        "tokens_requested": request.tokens_requested,
        "driver_node_id": request.driver_node_id,
        "pipeline_session_id": request.pipeline_session_id,
        "result": result_text,
        "completed": request.status == "completed" and result_text is not None
    }


@node_app.get("/api/ponw")
async def get_ponw_proof():
    """
    Get Proof of Neural Work for this node.
    
    This proves the node actually contributed compute for training/inference.
    Used for NEURO token rewards.
    """
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    return NEURO_NODE.get_ponw_proof()


# =============================================================================
# Pipeline Proofs (PoNW)
# =============================================================================

@node_app.get("/api/ponw")
async def get_ponw_proof():
    """
    Get Proof of Neural Work with quorum context.
    
    Returns a PipelineProof that includes:
    - Quorum membership information
    - Pipeline position and layers processed
    - DiLoCo sync round for cross-quorum verification
    - Signed proof for cryptographic verification
    """
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    # Get base proof from node
    base_proof = NEURO_NODE.get_ponw_proof()
    
    # Build proof with quorum context
    proof_response = {
        **base_proof,
        "version": 2,
        "quorum_id": CURRENT_QUORUM.quorum_id if CURRENT_QUORUM else None,
        "quorum_lifecycle": CURRENT_QUORUM.lifecycle.value if CURRENT_QUORUM else None,
        "training_mode": STATE.get("training_mode", "unknown"),
    }
    
    # Add QuorumTrainer stats if active
    if QUORUM_TRAINER:
        proof_response["quorum_batches"] = QUORUM_TRAINER.total_batches
        proof_response["quorum_sync_round"] = QUORUM_TRAINER.sync_round
        proof_response["quorum_loss"] = QUORUM_TRAINER.current_loss
    
    # Add role information from quorum
    if CURRENT_QUORUM and NEURO_NODE:
        member = CURRENT_QUORUM.get_member(NEURO_NODE.node_id)
        if member:
            proof_response["quorum_role"] = member.role.value
            proof_response["member_batches"] = member.batches_processed
            proof_response["member_reputation"] = member.reputation
    
    return proof_response


@node_app.post("/api/ponw/verify")
async def verify_ponw_proof(proof: dict):
    """
    Verify a PoNW proof using the ProofVerifier.
    
    Supports optimistic verification with challenge mechanism.
    """
    if not PROOF_VERIFIER:
        raise HTTPException(status_code=503, detail="Proof verifier not initialized")
    
    try:
        # Create PipelineProof from the submitted data
        pipeline_proof = PipelineProof(
            proof_id=proof.get("proof_id", ""),
            node_id=proof.get("node_id", ""),
            quorum_id=proof.get("quorum_id", ""),
            micro_batch_id=proof.get("micro_batch_id", ""),
            layer_range=(
                proof.get("layer_start", 0),
                proof.get("layer_end", 1),
            ),
            activation_hash=proof.get("activation_hash", ""),
            gradient_hash=proof.get("gradient_hash", ""),
            timestamp=proof.get("timestamp", time.time()),
        )
        
        # Use optimistic acceptance
        accepted, reason = PROOF_VERIFIER.accept_proof_optimistic(pipeline_proof)
        
        return {
            "accepted": accepted,
            "reason": reason,
            "proof_id": pipeline_proof.proof_id,
            "verification_mode": "optimistic",
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid proof: {e}")


@node_app.get("/api/neuro")
async def get_neuro_balance():
    """
    Get NEURO token balance and account info for this node.
    
    Returns:
    - balance: Current spendable balance
    - total_earned: Lifetime earnings from PoNW
    - total_spent: Lifetime spending
    - stake: Currently staked amount
    - stake_multiplier: Reward multiplier from staking
    """
    # Use local reference to avoid race condition during shutdown
    p2p = P2P
    if not p2p or not p2p.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    ledger = p2p.ledger
    
    try:
        # Use NEUROLedger API (no fallbacks)
        account_info = ledger.get_account_info()
        burn_stats = ledger.get_burn_stats()
        
        # Get node IDs
        wallet_id = ledger.node_id
        node_id = p2p.node_id
        
        return {
            "balance": round(account_info.get("balance", 0.0), 6),
            "total_earned": round(account_info.get("total_earned", 0.0), 6),
            "total_spent": round(account_info.get("total_spent", 0.0), 6),
            "stake": round(account_info.get("stake", 0.0), 2),
            "stake_multiplier": round(account_info.get("stake_multiplier", 1.0), 2),
            "proof_count": account_info.get("proof_count", 0),
            "wallet_id": wallet_id,
            "node_id": node_id,
            "network": {
                "total_burned": round(burn_stats.get("total_burned", 0.0), 6),
                "circulating_supply": round(burn_stats.get("circulating_supply", 0.0), 6),
                "burn_rate": "5%"
            }
        }
    except Exception as e:
        # Handle shutdown race condition gracefully
        raise HTTPException(status_code=503, detail=f"Service shutting down: {e}")


# ==================== LEDGER EXPLORER ENDPOINTS (DHT-BASED, TRULY DECENTRALIZED) ====================
# These endpoints query the DHT for proofs, not local DB.
# This means ANY node will return the SAME data - fully decentralized!

@node_app.get("/api/ledger/balance/{wallet_id}")
async def get_wallet_balance_from_dht(wallet_id: str):
    """
    Get wallet balance by querying the DHT (decentralized).
    
    This is the TRUSTLESS way to check balance:
    1. Query DHT for all proofs from this wallet
    2. Verify ECDSA signature on each proof
    3. Sum verified rewards = balance
    
    Any node in the network will return the same answer.
    """
    p2p = P2P
    if not p2p or not p2p.dht:
        raise HTTPException(status_code=503, detail="DHT not available")
    
    try:
        from neuroshard.core.network.dht_proof_store import DHTProofStore
        
        dht_store = DHTProofStore(p2p.dht)
        
        # Query DHT for proofs (with signature verification)
        verified_proofs, metadata = dht_store.retrieve_proofs_from_dht(
            wallet_id=wallet_id,
            max_proofs=1000,
            verify_signatures=True
        )
        
        if not metadata.get("found", False):
            return {
                "wallet_id": wallet_id,
                "balance": 0.0,
                "proof_count": 0,
                "message": "No proofs found in DHT (new wallet or network syncing)",
                "source": "dht"
            }
        
        # Calculate balance from verified proofs
        total_balance = sum(p.reward for p in verified_proofs)
        
        return {
            "wallet_id": wallet_id,
            "balance": round(total_balance, 6),
            "proof_count": len(verified_proofs),
            "total_proofs_in_dht": metadata.get("total_proofs", 0),
            "verification_failures": metadata.get("verification_failures", 0),
            "source": "dht",
            "trustless": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DHT query failed: {e}")


@node_app.get("/api/ledger/network")
async def get_network_ledger_stats():
    """
    Get network-wide ledger statistics by querying DHT.
    
    TRULY DECENTRALIZED: This queries the DHT for known wallets,
    not a local database. Any node will return the same data.
    
    Note: For efficiency, we query wallets that have announced to the tracker.
    The balance for each is verified via DHT signature verification.
    """
    p2p = P2P
    if not p2p or not p2p.dht:
        raise HTTPException(status_code=503, detail="DHT not available")
    
    try:
        from neuroshard.core.network.dht_proof_store import DHTProofStore
        import requests
        
        dht_store = DHTProofStore(p2p.dht)
        
        # Get list of known wallets from tracker (for discovery)
        # In a fully decentralized system, this would come from DHT announcements
        wallets = []
        total_supply = 0.0
        
        # Query peers from tracker to get their wallet IDs
        tracker_url = p2p.tracker_url or "http://tracker:3000"
        try:
            resp = requests.get(f"{tracker_url}/peers", timeout=5)
            if resp.status_code == 200:
                peers = resp.json()
                seen_wallets = set()
                
                for peer in peers:
                    token = peer.get("node_token")
                    if token and token not in seen_wallets:
                        seen_wallets.add(token)
                        # Derive wallet_id from token (first 16 chars of SHA256)
                        import hashlib
                        wallet_id = hashlib.sha256(token.encode()).hexdigest()[:16]
                        
                        # Query DHT for this wallet's balance (trustless)
                        verified_proofs, metadata = dht_store.retrieve_proofs_from_dht(
                            wallet_id=wallet_id,
                            max_proofs=100,
                            verify_signatures=True
                        )
                        
                        if verified_proofs:
                            balance = sum(p.reward for p in verified_proofs)
                            wallets.append({
                                "wallet_id": wallet_id + "...",
                                "balance": round(balance, 6),
                                "proof_count": len(verified_proofs),
                                "verified": True
                            })
                            total_supply += balance
        except Exception as e:
            logger.debug(f"Tracker query failed: {e}")
        
        # Also include wallets from local DHT storage (nodes we've seen)
        # This makes it work even without tracker
        if hasattr(p2p.dht, 'storage'):
            for key, value in list(p2p.dht.storage.items())[:50]:
                try:
                    # Check if this is a proofs key
                    if isinstance(value, str) and '"reward"' in value:
                        import json
                        proofs = json.loads(value)
                        if isinstance(proofs, dict) and 'node_id' in proofs:
                            proofs = [proofs]
                        if isinstance(proofs, list) and proofs:
                            wallet_id = proofs[0].get('node_id', '')[:16]
                            if wallet_id and not any(w['wallet_id'].startswith(wallet_id) for w in wallets):
                                balance = sum(p.get('reward', 0) for p in proofs if isinstance(p, dict))
                                wallets.append({
                                    "wallet_id": wallet_id + "...",
                                    "balance": round(balance, 6),
                                    "proof_count": len(proofs),
                                    "verified": False  # From local storage, not re-verified
                                })
                                total_supply += balance
                except:
                    pass
        
        return {
            "wallets": wallets,
            "wallet_count": len(wallets),
            "total_supply": round(total_supply, 6),
            "source": "dht",
            "trustless": True,
            "note": "Balances verified via ECDSA signatures from DHT"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DHT query failed: {e}")


@node_app.get("/api/ledger/proofs/{wallet_id}")
async def get_wallet_proofs_from_dht(wallet_id: str, limit: int = 50):
    """
    Get proof history for a wallet from DHT (decentralized).
    
    Returns the actual proofs with their signatures - fully verifiable.
    """
    p2p = P2P
    if not p2p or not p2p.dht:
        raise HTTPException(status_code=503, detail="DHT not available")
    
    try:
        from neuroshard.core.network.dht_proof_store import DHTProofStore
        
        dht_store = DHTProofStore(p2p.dht)
        
        # Query DHT for proofs
        verified_proofs, metadata = dht_store.retrieve_proofs_from_dht(
            wallet_id=wallet_id,
            max_proofs=limit,
            verify_signatures=True
        )
        
        proofs = []
        for p in verified_proofs:
            proofs.append({
                "timestamp": p.timestamp,
                "type": p.proof_type,
                "reward": round(p.reward, 6),
                "training_batches": p.training_batches,
                "tokens_processed": p.tokens_processed,
                "layers_held": p.layers_held,
                "signature": p.signature[:32] + "..." if p.signature else None,
                "verified": True
            })
        
        return {
            "wallet_id": wallet_id,
            "proofs": proofs,
            "count": len(proofs),
            "total_in_dht": metadata.get("total_proofs", 0),
            "source": "dht"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DHT query failed: {e}")


# ==================== STAKING ENDPOINTS ====================

class StakeRequest(BaseModel):
    amount: float
    duration_days: int = 30


@node_app.post("/api/stake")
async def stake_neuro(req: StakeRequest):
    """
    Stake NEURO tokens for reward multiplier.
    
    Staking provides:
    - 10% bonus per 1000 NEURO staked (diminishing returns)
    - Tokens locked for specified duration
    - 100+ NEURO stake unlocks Validator role (computes real cross-entropy loss)
    
    Example: Stake 2000 NEURO for 30 days = ~1.16x multiplier on all rewards
    """
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    # Validate using centralized economics
    is_valid, error = is_valid_stake_amount(req.amount)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    is_valid, error = is_valid_stake_duration(req.duration_days)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    success, message = P2P.ledger.stake(req.amount, req.duration_days)
    
    if success:
        account = P2P.ledger.get_account_info()
        new_stake = account.get("stake", 0.0)
        
        # Get dynamic validator stake requirement based on network size
        num_validators = 0
        if NEURO_NODE and hasattr(NEURO_NODE, 'layer_pool') and NEURO_NODE.layer_pool:
            last_layer = max(0, NEURO_NODE.layer_pool.current_num_layers - 1)
            num_validators = len(NEURO_NODE.layer_pool.layer_assignments.get(last_layer, []))
        
        required_stake = get_dynamic_validator_stake(num_validators)
        
        # Check if we should upgrade to Validator (no restart needed!)
        validator_upgraded = False
        if new_stake >= required_stake and NEURO_NODE:
            # Check if not already a validator
            if hasattr(NEURO_NODE, 'model') and NEURO_NODE.model and not NEURO_NODE.model.has_lm_head:
                # Upgrade the model to have LM head
                if NEURO_NODE.model.initialize_lm_head():
                    validator_upgraded = True
                    logger.info(f"Node upgraded to VALIDATOR! Now computing real cross-entropy loss.")
        
        response = {
            "success": True,
            "message": message,
            "new_stake": new_stake,
            "new_multiplier": account.get("stake_multiplier", 1.0),
            "locked_until": account.get("stake_locked_until", 0.0),
            "validator_stake_required": required_stake,
            "num_validators": num_validators,
        }
        
        if validator_upgraded:
            response["validator_upgrade"] = True
            response["message"] += " Upgraded to VALIDATOR! Now computing real training loss."
        elif new_stake < required_stake:
            response["validator_progress"] = f"{new_stake:.0f}/{required_stake:.0f} NEURO ({new_stake/required_stake*100:.1f}%)"
        
        return response
    else:
        raise HTTPException(status_code=400, detail=message)


@node_app.post("/api/unstake")
async def unstake_neuro():
    """
    Unstake NEURO tokens (if lock period expired).
    
    Returns staked tokens to balance.
    Note: If remaining stake drops below validator requirement, node is demoted to Worker.
    """
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    success, amount, message = P2P.ledger.unstake()
    
    if success:
        # Check if we need to demote from Validator
        validator_demoted = False
        account = P2P.ledger.get_account_info()
        remaining_stake = account.get("stake", 0.0)
        
        # Get current network size for dynamic stake calculation
        num_validators = 0
        if NEURO_NODE and hasattr(NEURO_NODE, 'layer_pool') and NEURO_NODE.layer_pool:
            last_layer = max(0, NEURO_NODE.layer_pool.current_num_layers - 1)
            num_validators = len(NEURO_NODE.layer_pool.layer_assignments.get(last_layer, []))
        
        required_stake = get_dynamic_validator_stake(num_validators)
        
        # Check if we were a validator and now don't qualify
        if NEURO_NODE and hasattr(NEURO_NODE, 'model') and NEURO_NODE.model:
            if NEURO_NODE.model.has_lm_head and remaining_stake < required_stake:
                # Demote from validator
                if NEURO_NODE.model.disable_lm_head():
                    validator_demoted = True
                    # Also update layer pool
                    if NEURO_NODE.layer_pool:
                        NEURO_NODE.layer_pool.demote_from_validator(NEURO_NODE.node_id)
                    logger.warning(f"Node demoted from Validator: stake {remaining_stake:.0f} < {required_stake:.0f} required")
        
        response = {
            "success": True,
            "message": message,
            "amount_unstaked": amount,
            "remaining_stake": remaining_stake,
        }
        
        if validator_demoted:
            response["validator_demoted"] = True
            response["message"] += f" WARNING: Demoted from Validator (need {required_stake:.0f} NEURO, have {remaining_stake:.0f})"
        
        return response
    else:
        raise HTTPException(status_code=400, detail=message)


@node_app.get("/api/stake/info")
async def get_stake_info():
    """Get current staking information with dynamic validator requirements."""
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    account = P2P.ledger.get_account_info()
    
    # Get current network size for dynamic stake calculation
    num_validators = 0
    if NEURO_NODE and hasattr(NEURO_NODE, 'layer_pool') and NEURO_NODE.layer_pool:
        last_layer = max(0, NEURO_NODE.layer_pool.current_num_layers - 1)
        num_validators = len(NEURO_NODE.layer_pool.layer_assignments.get(last_layer, []))
    
    return {
        "stake": account.get("stake", 0.0),
        "stake_multiplier": account.get("stake_multiplier", 1.0),
        "stake_locked_until": account.get("stake_locked_until", 0.0),
        "balance": account.get("balance", 0.0),
        "staking_info": {
            "bonus_per_1000": "10% (diminishing)",
            "min_lock_days": 1,
            "max_lock_days": 365,
            "validator_stake": get_validator_stake_info(num_validators),
        }
    }


class ThrottleUpdateRequest(BaseModel):
    cpu_threads: Optional[int] = None
    memory_mb: Optional[int] = None
    storage_mb: Optional[int] = None


@node_app.post("/api/throttle")
async def update_throttle(req: ThrottleUpdateRequest):
    """
    Update training throttle settings while node is running.
    
    This allows the GUI to change CPU/RAM/Storage limits without restarting.
    Changes take effect within 5 seconds.
    """
    updated = {}
    
    if req.cpu_threads is not None:
        STATE["config_cpu_threads"] = req.cpu_threads
        updated["cpu_threads"] = req.cpu_threads
    
    if req.memory_mb is not None:
        STATE["config_memory_mb"] = req.memory_mb
        updated["memory_mb"] = req.memory_mb
    
    if req.storage_mb is not None:
        STATE["config_storage_mb"] = req.storage_mb
        updated["storage_mb"] = req.storage_mb
        # Update genesis loader if it exists
        if NEURO_NODE and hasattr(NEURO_NODE, 'genesis_loader') and NEURO_NODE.genesis_loader:
            NEURO_NODE.genesis_loader.max_storage_mb = req.storage_mb
            NEURO_NODE.genesis_loader.max_shards = max(1, int(req.storage_mb / 10))
            logger.info(f"[NODE] Updated storage limit: {req.storage_mb}MB ({NEURO_NODE.genesis_loader.max_shards} shards)")
    
    return {
        "success": True,
        "updated": updated,
        "message": "Settings updated. Changes take effect within 5 seconds.",
        "current_throttle": {
            "cpu_ratio": STATE.get("throttle_cpu_ratio", 1.0),
            "ram_ratio": STATE.get("throttle_ram_ratio", 1.0),
            "effective": STATE.get("throttle_effective", 1.0),
        }
    }


@node_app.get("/api/validator/info")
async def get_validator_info():
    """
    Get validator eligibility and status.
    
    Validators require:
    - Minimum 100 NEURO staked
    - LM Head layer assignment (last layer)
    
    Validators earn:
    - 30% bonus on rewards (up from 20%)
    - 0.001 NEURO per proof validated
    """
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    validator_info = P2P.ledger.get_validator_info()
    
    # Add role info from node
    if NEURO_NODE:
        validator_info["has_lm_head"] = NEURO_NODE.model.has_lm_head if NEURO_NODE.model else False
        validator_info["is_active_validator"] = (
            validator_info["is_eligible_validator"] and 
            validator_info.get("has_lm_head", False)
        )
    
    return validator_info


# ==================== SWARM ENDPOINTS ====================

@node_app.get("/api/swarm")
async def get_swarm_status():
    """
    Get Swarm architecture status.
    
    Returns buffer fill rates, heartbeat peers, routing stats.
    """
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    # Get swarm status from node
    swarm_status = NEURO_NODE.get_swarm_status()
    
    return swarm_status


@node_app.get("/api/diloco")
async def get_diloco_progress():
    """
    Get DiLoCo training progress.
    
    Returns inner step count, sync progress, outer step count.
    """
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    return NEURO_NODE.get_diloco_progress()


@node_app.get("/api/model_info")
async def get_model_info():
    """Get information about the NeuroLLM model."""
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    stats = NEURO_NODE.get_stats()
    
    # Get architecture info
    arch_info = {}
    if NEURO_NODE.layer_pool and NEURO_NODE.layer_pool.current_architecture:
        arch = NEURO_NODE.layer_pool.current_architecture
        arch_info = {
            "hidden_dim": arch.hidden_dim,
            "num_layers": arch.num_layers,
            "num_heads": arch.num_heads,
            "vocab_size": arch.vocab_size,
            "architecture_version": NEURO_NODE.layer_pool.architecture_version,
            "total_params": arch.estimate_params(),
        }
    
    # Sanitize loss for JSON
    model_loss = NEURO_NODE.current_loss
    if math.isinf(model_loss) or math.isnan(model_loss):
        model_loss = None
    
    return {
        "model_name": "NeuroLLM",
        "description": "The People's Language Model - trained from scratch by the network",
        "architecture": arch_info,  # NEW: Show current architecture
        "my_layers": stats.get("my_layers", []),
        "my_params": stats.get("my_params", 0),
        "network_layers": stats.get("network_layers", 0),
        "network_nodes": stats.get("network_nodes", 0),
        "total_training_rounds": NEURO_NODE.total_training_rounds,
        "current_loss": model_loss,
        "note": "This model is trained collaboratively. Quality improves as more users contribute!"
    }


@node_app.get("/api/network")
async def get_network_info():
    """Get network capacity and layer distribution."""
    if not NEURO_NODE or not NEURO_NODE.layer_pool:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    capacity = NEURO_NODE.layer_pool.get_network_capacity()
    
    return {
        "total_nodes": capacity.total_nodes,
        "total_memory_mb": capacity.total_memory_mb,
        "max_possible_layers": capacity.max_layers,
        "current_layers": capacity.assigned_layers,
        "layer_coverage": capacity.layer_coverage,
        "my_contribution": NEURO_NODE.model.get_my_contribution() if NEURO_NODE.model else {},
    }


@node_app.get("/api/logs")
async def get_logs(since_id: Optional[int] = None, limit: int = 100):
    """
    Get recent logs from the node.
    
    Args:
        since_id: Return logs with ID greater than this (for polling). 
                  Use 0 or omit to get all available logs on initial load.
        limit: Maximum number of logs to return (default 100)
    
    Returns:
        List of log entries with id, epoch, timestamp, message, type, and level
    """
    with _LOG_BUFFER_LOCK:
        logs = list(_LOG_BUFFER)
    
    # If since_id is provided, filter to only logs with ID > since_id
    if since_id is not None and since_id > 0:
        logs = [log for log in logs if log.get('id', 0) > since_id]
    
    # Limit results (take most recent)
    if len(logs) > limit:
        logs = logs[-limit:]
    
    # Get the latest log ID for next poll
    latest_id = logs[-1]['id'] if logs else (since_id or 0)
    
    return {
        "logs": logs,
        "total": len(_LOG_BUFFER),
        "latest_id": latest_id,  # Client should use this for next poll
    }


@node_app.post("/api/shutdown")
async def shutdown_node():
    """
    Gracefully shutdown the node.
    
    Saves checkpoint and stops all components cleanly.
    """
    logger.info("[NODE] Shutdown requested via API")
    
    # Use a background thread for shutdown (more reliable than asyncio.create_task)
    def do_shutdown():
        import time
        time.sleep(0.5)  # Brief delay to allow response to be sent
        request_shutdown()
    
    shutdown_thread = threading.Thread(target=do_shutdown, daemon=False)
    shutdown_thread.start()
    
    return {
        "status": "shutting_down",
        "message": "Node will shutdown in 0.5 seconds. Checkpoint will be saved."
    }


@node_app.get("/api/checkpoint/info")
async def get_checkpoint_info():
    """Get checkpoint info for P2P sync."""
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    return NEURO_NODE.get_checkpoint_info()


@node_app.get("/api/checkpoint/download")
async def download_checkpoint():
    """Download checkpoint (for P2P sync via HTTP fallback)."""
    import io
    import zlib
    from fastapi.responses import Response
    
    if not NEURO_NODE or not NEURO_NODE.model:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    try:
        # Serialize checkpoint for my layers only
        buffer = io.BytesIO()
        
        # Collect layer state dicts
        layer_states = {
            layer_id: layer.state_dict()
            for layer_id, layer in NEURO_NODE.model.my_layers.items()
        }
        
        checkpoint = {
            "layer_ids": NEURO_NODE.my_layer_ids,
            "layers": layer_states,
            "has_embedding": NEURO_NODE.model.has_embedding,
            "has_lm_head": NEURO_NODE.model.has_lm_head,
            "version": NEURO_NODE.total_training_rounds,
        }
        
        if NEURO_NODE.model.embedding:
            checkpoint["embedding"] = NEURO_NODE.model.embedding.state_dict()
        if NEURO_NODE.model.lm_head:
            checkpoint["lm_head"] = NEURO_NODE.model.lm_head.state_dict()
        if NEURO_NODE.model.final_norm:
            checkpoint["final_norm"] = NEURO_NODE.model.final_norm.state_dict()
        
        torch.save(checkpoint, buffer)
        
        # Compress
        raw_data = buffer.getvalue()
        compressed = zlib.compress(raw_data, level=6)
        
        return Response(
            content=compressed,
            media_type="application/octet-stream",
            headers={
                "X-Checkpoint-Version": str(checkpoint["version"]),
                "X-Layer-IDs": ",".join(map(str, NEURO_NODE.my_layer_ids)),
                "X-Original-Size": str(len(raw_data)),
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== API v1 ENDPOINTS (SDK Compatible) ====================

class InferenceV1Request(BaseModel):
    """Inference request matching SDK expectations."""
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    stop: List[str] = []
    stream: bool = False


class SendNEURORequest(BaseModel):
    """Send NEURO request."""
    to: str
    amount: float
    memo: str = ""


@node_app.get("/api/v1/status")
async def get_status_v1():
    """
    Get current node status (SDK compatible).
    
    Returns status in format expected by NeuroNode SDK.
    """
    import math
    import psutil
    import os
    
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    # Get node stats
    stats = NEURO_NODE.get_stats()
    
    # Determine role
    has_embedding = stats.get("has_embedding", False)
    has_lm_head = stats.get("has_lm_head", False)
    if has_embedding and has_lm_head:
        role = "full"
    elif has_embedding:
        role = "driver"
    elif has_lm_head:
        role = "validator"
    else:
        role = "worker"
    
    # Get system resources
    try:
        mem = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        gpu_used = 0
        gpu_total = 0
        
        if torch.cuda.is_available():
            gpu_used = torch.cuda.memory_allocated()
            gpu_total = torch.cuda.get_device_properties(0).total_memory
        
        resources = {
            "gpu_memory_used": gpu_used,
            "gpu_memory_total": gpu_total,
            "cpu_percent": psutil.cpu_percent(),
            "ram_used": mem.used,
            "ram_total": mem.total,
        }
    except Exception:
        resources = {}
    
    # Handle infinity loss
    loss = stats.get("current_loss", 0.0)
    if math.isinf(loss) or math.isnan(loss):
        loss = 0.0
    
    return {
        "node_id": NEURO_NODE.node_id,
        "version": __version__,
        "uptime_seconds": int(time.time() - getattr(NEURO_NODE, '_start_time', time.time())),
        "status": STATE.get("training_status", "running"),
        "role": role,
        "layers": stats.get("my_layers", []),
        "peer_count": STATE.get("peer_count", 0),
        "has_embedding": has_embedding,
        "has_lm_head": has_lm_head,
        "training": {
            "enabled": NEURO_NODE.enable_training,
            "epoch": 0,  # Not tracked separately
            "step": stats.get("total_training_rounds", 0),
            "loss": loss,
        },
        "resources": resources,
    }


@node_app.get("/api/v1/metrics")
async def get_metrics_v1():
    """
    Get performance metrics (SDK compatible).
    """
    import math
    from datetime import datetime
    
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    stats = NEURO_NODE.get_stats()
    
    # Get balance info for rewards
    earned_total = 0.0
    pending = 0.0
    if P2P and P2P.ledger:
        account = P2P.ledger.get_account_info()
        earned_total = account.get("total_earned", 0.0)
        pending = 0.0  # Could track pending proofs
    
    return {
        "timestamp": datetime.now().isoformat(),
        "inference": {
            "requests_total": STATE.get("processed_count", 0),
            "requests_per_minute": 0.0,  # Would need tracking
            "avg_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "tokens_generated": stats.get("total_tokens_processed", 0),
        },
        "training": {
            "steps_total": stats.get("total_training_rounds", 0),
            "steps_per_hour": 0.0,
            "gradients_submitted": 0,
            "gradients_accepted": 0,
        },
        "network": {
            "bytes_sent": 0,
            "bytes_received": 0,
            "active_connections": STATE.get("peer_count", 0),
            "rpc_calls": 0,
            "peer_count": STATE.get("peer_count", 0),
        },
        "rewards": {
            "earned_today": 0.0,  # Would need daily tracking
            "earned_total": earned_total,
            "pending": pending,
        },
    }


@node_app.get("/api/v1/health")
async def health_check_v1():
    """Health check endpoint (SDK compatible)."""
    checks = {
        "node": "ok" if NEURO_NODE else "error",
        "network": "ok" if P2P else "error",
        "model": "ok" if NEURO_NODE and NEURO_NODE.model else "error",
    }
    
    # Check GPU
    try:
        if torch.cuda.is_available():
            checks["gpu"] = "ok"
        else:
            checks["gpu"] = "cpu_only"
    except Exception:
        checks["gpu"] = "unknown"
    
    healthy = all(v == "ok" for k, v in checks.items() if k != "gpu")
    
    return {
        "healthy": healthy,
        "checks": checks,
    }


@node_app.post("/api/v1/inference")
async def inference_v1(req: InferenceV1Request):
    """
    Run inference (SDK compatible).
    
    Supports both streaming and non-streaming modes.
    """
    from fastapi.responses import StreamingResponse
    import uuid
    
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    start_time = time.time()
    request_id = f"inf_{uuid.uuid4().hex[:12]}"
    
    if req.stream:
        # Streaming response
        async def generate_stream():
            try:
                # Generate tokens one at a time
                text = NEURO_NODE.generate(
                    prompt=req.prompt,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
                
                # Emit tokens
                tokens = text.split()
                for i, token in enumerate(tokens):
                    yield f"data: {json.dumps({'token': token + ' ', 'index': i})}\n\n"
                
                # Final message
                yield f"data: {json.dumps({'token': '[DONE]', 'finish_reason': 'stop'})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
        )
    
    # Non-streaming response
    try:
        text = NEURO_NODE.generate(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        
        end_time = time.time()
        inference_ms = (end_time - start_time) * 1000
        
        # Count tokens (simple approximation)
        prompt_tokens = len(req.prompt.split())
        completion_tokens = len(text.split())
        
        STATE["processed_count"] = STATE.get("processed_count", 0) + 1
        
        return {
            "id": request_id,
            "text": text,
            "tokens_generated": completion_tokens,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "cost": {
                "amount": completion_tokens * 0.000001,  # Approximate
                "currency": "NEURO",
            },
            "timing": {
                "queue_ms": 0,
                "inference_ms": inference_ms,
                "total_ms": inference_ms,
            },
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Quorum-Based Inference API
# =============================================================================

@node_app.get("/api/inference/quorums")
async def discover_inference_quorums():
    """
    Discover available inference quorums.
    
    Returns a list of quorums that can handle inference requests,
    sorted by score (latency, price, reputation).
    """
    if not QUORUM_INFERENCE_ROUTER:
        raise HTTPException(status_code=503, detail="Quorum inference router not initialized")
    
    try:
        quorums = QUORUM_INFERENCE_ROUTER.discover_quorums()
        
        return {
            "quorums": [
                {
                    "quorum_id": q.quorum_id,
                    "speed_tier": q.speed_tier,
                    "initiator_endpoint": q.initiator_endpoint,
                    "estimated_latency_ms": q.estimated_latency_ms,
                    "price_per_token": q.price_per_token,
                    "available_capacity": q.available_capacity,
                }
                for q in quorums
            ],
            "total": len(quorums),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class QuorumInferenceRequest(BaseModel):
    """Request for quorum-based inference."""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    quorum_id: Optional[str] = None  # Optional: specific quorum to use


@node_app.post("/api/inference/quorum")
async def quorum_inference(req: QuorumInferenceRequest):
    """
    Run inference through the quorum system.
    
    If quorum_id is specified, routes to that quorum's initiator.
    Otherwise, discovers available quorums and selects the best one.
    
    For local inference (when this node is part of a quorum), uses
    the local model directly.
    """
    import uuid
    
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    start_time = time.time()
    request_id = f"q_{uuid.uuid4().hex[:12]}"
    
    # Check if we should route to another quorum or handle locally
    if QUORUM_INFERENCE_ROUTER and CURRENT_QUORUM:
        # We're part of a quorum - check if we can handle it
        if CURRENT_QUORUM.lifecycle == QuorumLifecycle.ACTIVE:
            # Route to quorum initiator (may be us)
            member = CURRENT_QUORUM.get_member(NEURO_NODE.node_id)
            if member and member.role == QuorumRole.INITIATOR:
                # We are the initiator - handle locally
                pass
            elif req.quorum_id and req.quorum_id != CURRENT_QUORUM.quorum_id:
                # User requested a different quorum
                quorums = QUORUM_INFERENCE_ROUTER.discover_quorums()
                target = next((q for q in quorums if q.quorum_id == req.quorum_id), None)
                if target:
                    return {
                        "id": request_id,
                        "redirect": True,
                        "target_endpoint": target.initiator_endpoint,
                        "message": f"Route to quorum {req.quorum_id[:8]}...",
                    }
    
    # Handle inference locally
    try:
        text = NEURO_NODE.generate(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        
        end_time = time.time()
        inference_ms = (end_time - start_time) * 1000
        
        prompt_tokens = len(req.prompt.split())
        completion_tokens = len(text.split())
        
        STATE["processed_count"] = STATE.get("processed_count", 0) + 1
        
        return {
            "id": request_id,
            "text": text,
            "quorum_id": CURRENT_QUORUM.quorum_id if CURRENT_QUORUM else None,
            "tokens_generated": completion_tokens,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "cost": {
                "amount": completion_tokens * 0.000001,
                "currency": "NEURO",
            },
            "timing": {
                "queue_ms": 0,
                "inference_ms": inference_ms,
                "total_ms": inference_ms,
            },
            "version": "quorum",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@node_app.get("/api/quorum/status")
async def get_quorum_status():
    """
    Get current quorum status for this node.
    """
    if not CURRENT_QUORUM:
        return {
            "in_quorum": False,
            "training_mode": STATE.get("training_mode", "unknown"),
        }
    
    member = CURRENT_QUORUM.get_member(NEURO_NODE.node_id) if NEURO_NODE else None
    
    return {
        "in_quorum": True,
        "quorum_id": CURRENT_QUORUM.quorum_id,
        "lifecycle": CURRENT_QUORUM.lifecycle.value,
        "speed_tier": CURRENT_QUORUM.speed_tier,
        "members": len(CURRENT_QUORUM.members),
        "is_complete": CURRENT_QUORUM.is_complete,
        "my_role": member.role.value if member else None,
        "my_layers": list(member.layer_range) if member else None,
        "total_batches": CURRENT_QUORUM.total_batches,
        "session_remaining_seconds": max(0, CURRENT_QUORUM.session_end - time.time()),
        "training_mode": STATE.get("training_mode", "unknown"),
        "quorum_batches": STATE.get("quorum_batches", 0),
        "quorum_loss": STATE.get("quorum_loss"),
    }


@node_app.get("/api/v1/wallet/balance")
async def get_wallet_balance_v1():
    """Get wallet balance (SDK compatible)."""
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    account = P2P.ledger.get_account_info()
    
    return {
        "address": P2P.ledger.node_id,
        "balances": {
            "available": account.get("balance", 0.0),
            "staked": account.get("stake", 0.0),
            "pending": 0.0,
            "total": account.get("balance", 0.0) + account.get("stake", 0.0),
        },
        "staking": {
            "amount": account.get("stake", 0.0),
            "duration_days": 30,
            "multiplier": account.get("stake_multiplier", 1.0),
        },
    }


@node_app.post("/api/v1/wallet/send")
async def send_neuro_v1(req: SendNEURORequest):
    """Send NEURO tokens (SDK compatible)."""
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    success, message, tx = P2P.ledger.transfer(req.to, req.amount, req.memo)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    return {
        "transaction_id": tx.tx_id if tx else "",
        "from": P2P.ledger.node_id,
        "to": req.to,
        "amount": req.amount,
        "fee": tx.fee if tx else 0.0,
        "memo": req.memo,
        "status": "confirmed",
        "timestamp": datetime.now().isoformat() if 'datetime' in dir() else time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@node_app.get("/api/v1/wallet/transactions")
async def get_transactions_v1(limit: int = 10, offset: int = 0, type: Optional[str] = None):
    """Get transaction history (SDK compatible)."""
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    # Get recent proofs as transactions
    import sqlite3
    transactions = []
    
    try:
        with sqlite3.connect(P2P.ledger.db_path) as conn:
            query = """
                SELECT signature, node_id, proof_type, timestamp, reward_amount
                FROM proof_history
                WHERE node_id = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """
            rows = conn.execute(query, (P2P.ledger.node_id, limit, offset)).fetchall()
            
            for sig, node_id, ptype, ts, reward in rows:
                transactions.append({
                    "id": sig[:16] if sig else "",
                    "type": "reward",
                    "amount": reward,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                    "details": {
                        "proof_type": ptype,
                    },
                })
    except Exception:
        pass
    
    return {
        "transactions": transactions,
        "total": len(transactions),
        "limit": limit,
        "offset": offset,
    }


@node_app.post("/api/v1/wallet/stake")
async def stake_neuro_v1(req: StakeRequest):
    """Stake NEURO tokens (SDK compatible)."""
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    success, message = P2P.ledger.stake(req.amount, req.duration_days)
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    account = P2P.ledger.get_account_info()
    
    from datetime import date, timedelta
    start = date.today()
    unlock = start + timedelta(days=req.duration_days)
    
    return {
        "success": True,
        "stake": {
            "amount": req.amount,
            "duration_days": req.duration_days,
            "start_date": start.isoformat(),
            "unlock_date": unlock.isoformat(),
            "multiplier": account.get("stake_multiplier", 1.0),
        },
        "new_balance": {
            "available": account.get("balance", 0.0),
            "staked": account.get("stake", 0.0),
        },
    }


@node_app.post("/api/v1/wallet/unstake")
async def unstake_neuro_v1(amount: float = None):
    """Request unstaking (SDK compatible)."""
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    success, unstaked_amount, message = P2P.ledger.unstake()
    
    if not success:
        raise HTTPException(status_code=400, detail=message)
    
    from datetime import date, timedelta
    available = date.today() + timedelta(days=7)
    
    return {
        "success": True,
        "unstake": {
            "amount": unstaked_amount,
            "cooldown_days": 7,
            "available_date": available.isoformat(),
        },
    }


@node_app.get("/api/v1/wallet/rewards")
async def get_rewards_v1(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get reward history (SDK compatible)."""
    if not P2P or not P2P.ledger:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    account = P2P.ledger.get_account_info()
    
    return {
        "total": account.get("total_earned", 0.0),
        "by_day": [],  # Would need daily tracking
        "by_type": {
            "uptime": 0.0,
            "inference": 0.0,
            "training": 0.0,
        },
    }


@node_app.get("/api/v1/peers")
async def get_peers_v1():
    """List connected peers (SDK compatible)."""
    if not P2P:
        raise HTTPException(status_code=503, detail="P2P not available")
    
    peers = []
    for peer_url, peer_info in P2P.known_peers.items():
        # Parse peer info
        peer_id = peer_info.get("id", peer_url)
        role = "worker"
        layers = []
        
        if isinstance(peer_info, dict):
            if peer_info.get("has_embedding"):
                role = "driver"
            elif peer_info.get("has_lm_head"):
                role = "validator"
            layers = peer_info.get("layers", [])
        
        peers.append({
            "id": peer_id,
            "address": peer_url,
            "role": role,
            "layers": layers,
            "latency_ms": 0.0,
            "connected_since": None,
        })
    
    return {
        "peers": peers,
        "total": len(peers),
    }


@node_app.get("/api/v1/layers")
async def get_layers_v1():
    """List assigned layers (SDK compatible)."""
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    layers = []
    for layer_id in NEURO_NODE.my_layer_ids:
        layer_type = "transformer"
        if layer_id == 0 and NEURO_NODE.model.has_embedding:
            layer_type = "embedding"
        
        layers.append({
            "index": layer_id,
            "type": layer_type,
            "memory_mb": 0,  # Would need per-layer tracking
            "status": "active",
        })
    
    # Add LM head if present
    if NEURO_NODE.model.has_lm_head:
        layers.append({
            "index": max(NEURO_NODE.my_layer_ids) + 1 if NEURO_NODE.my_layer_ids else 0,
            "type": "lm_head",
            "memory_mb": 0,
            "status": "active",
        })
    
    return {
        "layers": layers,
        "total_layers": len(NEURO_NODE.my_layer_ids),
        "my_layer_count": len(NEURO_NODE.my_layer_ids),
    }


@node_app.get("/api/v1/config")
async def get_config_v1():
    """Get node configuration (SDK compatible)."""
    if not NEURO_NODE:
        raise HTTPException(status_code=503, detail="Node not ready")
    
    port = STATE.get("port", 8000)
    
    return {
        "node_id": NEURO_NODE.node_id,
        "port": port,
        "grpc_port": port + 1000,
        "tracker_url": "https://neuroshard.com/api/tracker",
        "training": {
            "enabled": NEURO_NODE.enable_training,
            "batch_size": 8,
            "learning_rate": 0.0001,
            "diloco_steps": STATE.get("diloco_inner_steps", 500),
        },
        "resources": {
            "max_memory_mb": STATE.get("config_memory_mb"),
            "cpu_threads": STATE.get("config_cpu_threads"),
        },
    }


@node_app.patch("/api/v1/config")
async def update_config_v1(updates: dict):
    """Update node configuration (SDK compatible)."""
    updated = []
    
    if "training" in updates:
        training = updates["training"]
        if "batch_size" in training:
            updated.append("training.batch_size")
        if "diloco_steps" in training:
            STATE["diloco_inner_steps"] = training["diloco_steps"]
            updated.append("training.diloco_steps")
    
    if "resources" in updates:
        resources = updates["resources"]
        if "max_memory_mb" in resources:
            STATE["config_memory_mb"] = resources["max_memory_mb"]
            updated.append("resources.max_memory_mb")
        if "cpu_threads" in resources:
            STATE["config_cpu_threads"] = resources["cpu_threads"]
            updated.append("resources.cpu_threads")
    
    return {
        "success": True,
        "updated": updated,
        "restart_required": False,
    }


# ==================== UTILITY FUNCTIONS ====================

def get_public_ip():
    """Attempt to get the public IP address of this node."""
    try:
        services = [
            'https://api.ipify.org',
            'https://ifconfig.me/ip',
            'https://icanhazip.com'
        ]
        for service in services:
            try:
                return requests.get(service, timeout=3).text.strip()
            except:
                continue
    except Exception:
        pass
    return None


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def _get_speed_tier(node, benchmark: bool = True) -> SpeedTier:
    """
    Determine the speed tier for a node based on hardware and benchmarking.
    
    Speed tiers match training throughput for efficient quorum formation:
    - T1: Enterprise GPUs (H100, A100) - fastest (<10ms/layer)
    - T2: Consumer GPUs (RTX 4090, 3090) (10-50ms/layer)
    - T3: Mid-range GPUs (RTX 3080, 4070) (50-200ms/layer)
    - T4: Entry GPUs or fast CPU (RTX 3060, Apple M2) (200-1000ms/layer)
    - T5: CPU-only or low-memory - slowest (>1000ms/layer)
    
    Args:
        node: The NeuroShard node
        benchmark: If True, run actual forward pass benchmarking (recommended)
    
    Returns:
        SpeedTier enum value
    """
    # Try benchmark-based classification first (most accurate)
    if benchmark:
        benchmark_tier = _benchmark_speed_tier(node)
        if benchmark_tier:
            return benchmark_tier
    
    # Fall back to heuristic-based classification
    try:
        device = getattr(node, 'device', 'cpu')
        memory_mb = getattr(node, 'memory_limit_mb', 4096)
        
        if device == 'cuda':
            # Check GPU type and memory
            gpu_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ""
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
            
            # Enterprise tier
            if any(x in gpu_name for x in ['h100', 'a100', 'h200', 'a6000']):
                return SpeedTier.T1
            # High-end consumer
            elif any(x in gpu_name for x in ['4090', '3090', 'a5000']) or gpu_memory >= 20:
                return SpeedTier.T2
            # Mid-range
            elif any(x in gpu_name for x in ['4080', '3080', '4070', 'a4000']) or gpu_memory >= 10:
                return SpeedTier.T3
            # Entry-level GPU
            else:
                return SpeedTier.T4
        
        elif device == 'mps':
            # Apple Silicon - tier based on memory
            if memory_mb >= 32768:  # 32GB+ unified memory
                return SpeedTier.T3
            elif memory_mb >= 16384:  # 16GB
                return SpeedTier.T4
            else:
                return SpeedTier.T5
        
        else:
            # CPU-only
            return SpeedTier.T5
            
    except Exception as e:
        logger.warning(f"Could not determine speed tier: {e}")
        return SpeedTier.T5  # Conservative default


def _benchmark_speed_tier(node) -> Optional[SpeedTier]:
    """
    Benchmark actual forward pass latency to determine speed tier.
    
    Runs 5 forward passes through a single transformer layer and
    measures average latency to classify the node.
    
    Thresholds (per layer forward pass):
    - T1: <10ms (enterprise GPUs)
    - T2: 10-50ms (high-end consumer)
    - T3: 50-200ms (mid-range)
    - T4: 200-1000ms (entry-level)
    - T5: >1000ms (slow CPU)
    
    Returns:
        SpeedTier or None if benchmarking fails
    """
    try:
        import torch
        import time
        
        # Check if model is available
        if not hasattr(node, 'model') or node.model is None:
            logger.debug("[BENCHMARK] No model available for benchmarking")
            return None
        
        model = node.model
        device = getattr(node, 'device', 'cpu')
        
        # Get a single transformer block to benchmark
        # Try common layer access patterns
        layer = None
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layer = model.transformer.h[0]  # GPT-2 style
        elif hasattr(model, 'layers'):
            layer = model.layers[0]  # LLaMA style
        elif hasattr(model, 'blocks'):
            layer = model.blocks[0]  # Other style
        
        if layer is None:
            logger.debug("[BENCHMARK] Could not find transformer layer")
            return None
        
        # Create test input (batch_size=1, seq_len=32)
        config = model.config if hasattr(model, 'config') else None
        hidden_dim = getattr(config, 'n_embd', 1024) if config else 1024
        test_input = torch.randn(1, 32, hidden_dim)
        
        # Move to device
        if device == 'cuda' and torch.cuda.is_available():
            test_input = test_input.cuda()
            layer = layer.cuda()
            torch.cuda.synchronize()
        elif device == 'mps':
            test_input = test_input.to('mps')
            layer = layer.to('mps')
        
        # Warm-up run
        with torch.no_grad():
            _ = layer(test_input)
        
        # Benchmark 5 forward passes
        latencies = []
        with torch.no_grad():
            for _ in range(5):
                if device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = layer(test_input)
                if device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms
        
        avg_latency_ms = sum(latencies) / len(latencies)
        logger.info(f"[BENCHMARK] Average layer forward pass: {avg_latency_ms:.2f}ms")
        
        # Classify based on latency thresholds
        if avg_latency_ms < 10:
            return SpeedTier.T1
        elif avg_latency_ms < 50:
            return SpeedTier.T2
        elif avg_latency_ms < 200:
            return SpeedTier.T3
        elif avg_latency_ms < 1000:
            return SpeedTier.T4
        else:
            return SpeedTier.T5
            
    except Exception as e:
        logger.debug(f"[BENCHMARK] Benchmarking failed: {e}")
        return None


# =============================================================================
# DEFAULT BOOTSTRAP NODES (like Bitcoin DNS seeds, IPFS bootstrap nodes)
# These are well-known nodes that help new nodes discover the network.
# They are NOT special or trusted - just reliable entry points.
# Once connected, DHT takes over and tracker/seeds are no longer needed.
#
# Priority order:
# 1. User-specified --seed-peers (highest priority)
# 2. NEUROSHARD_BOOTSTRAP_NODES environment variable (comma-separated)
# 3. DEFAULT_BOOTSTRAP_NODES (fallback)
# =============================================================================
def get_bootstrap_nodes() -> List[str]:
    """Get bootstrap nodes from environment or use defaults."""
    # Check for user-configured bootstrap nodes via environment
    env_nodes = os.environ.get("NEUROSHARD_BOOTSTRAP_NODES", "")
    if env_nodes:
        nodes = [n.strip() for n in env_nodes.split(",") if n.strip()]
        if nodes:
            return nodes
    
    # Default bootstrap nodes
    return [
        # NeuroShard official observer/bootstrap node
        "observer.neuroshard.com:8001",
        # Add more community-run bootstrap nodes here as network grows
    ]

DEFAULT_BOOTSTRAP_NODES = get_bootstrap_nodes()

def run_node(
    port: int,
    tracker: str = "https://neuroshard.com/api/tracker",
    node_token: Optional[str] = None,
    announce_ip: str = None,
    announce_port: int = None,
    enable_training: bool = True,
    observer_mode: bool = False,
    available_memory_mb: Optional[float] = None,
    max_storage_mb: float = 100.0,
    max_cpu_threads: Optional[int] = None,
    diloco_inner_steps: int = 500,
    device: str = "auto",
    seed_peers: Optional[List[str]] = None,
):
    """
    Start a NeuroShard node.
    
    TRULY DECENTRALIZED:
    - No fixed phases or model sizes
    - Node contributes based on available memory
    - More memory = more layers = more NEURO rewards
    - DHT-based peer discovery with optional seed peers
    
    MULTI-NODE SUPPORT:
    - Same token on multiple machines/ports is now supported
    - Each instance gets unique network identity (for layers)
    - Earnings accumulate to the same NEURO wallet
    
    Args:
        port: HTTP port
        tracker: Tracker URL for initial peer bootstrap
        node_token: Authentication token
        enable_training: Whether to participate in training
        available_memory_mb: Override memory detection (for testing)
        max_storage_mb: Maximum disk space for training data shards
        max_cpu_threads: Maximum CPU threads to use for training
        seed_peers: List of seed peer addresses (e.g., ["1.2.3.4:8000"]) for DHT bootstrap
    """
    global NEURO_NODE, P2P
    
    # CRITICAL: Clear shutdown flag from previous run (for GUI restart support)
    _SHUTDOWN_REQUESTED.clear()
    
    # Reset STATE for fresh start (important for GUI restart)
    STATE.clear()
    STATE.update({
        "shard_range": "Unknown",
        "peer_count": 0,
        "processed_count": 0,
        "training_updates": 0,
        "token_count": 0,
        "training_batches": 0,
        "assigned_layers": [],
        "has_embedding": False,
        "has_lm_head": False,
    })
    
    logger.info(f"Starting NeuroShard Node {__version__} on Port {port}")
    
    # Multi-node detection and info
    from neuroshard.utils.hardware import get_instance_id, get_machine_id
    instance_id = get_instance_id(port)
    machine_id = get_machine_id()
    
    logger.info(f"Machine ID: {machine_id}")
    logger.info(f"Instance ID: {instance_id} (machine:port unique)")
    
    if node_token:
        wallet_id = hashlib.sha256(node_token.encode()).hexdigest()[:16]
        logger.info(f"Wallet ID: {wallet_id}... (NEURO earnings go here)")
        logger.info("=" * 50)
        logger.info("MULTI-NODE INFO:")
        logger.info("  Same token on multiple machines? Each gets unique assignment")
        logger.info("=" * 50)
    logger.info(f"Dashboard available at http://localhost:{port}/")
    logger.info(f"Max training data storage: {max_storage_mb}MB")
    
    # Thread configuration
    # Note: For GUI mode, this is already set in gui_runner.py wrapper
    # For CLI mode, we do our best here (may fail if torch already initialized)
    if max_cpu_threads:
        logger.info(f"Limiting CPU threads to: {max_cpu_threads}")
        
        # Set environment variables (these always work)
        import os
        os.environ['OMP_NUM_THREADS'] = str(max_cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(max_cpu_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(max_cpu_threads)
        
        # Try to set PyTorch threads (may fail if already set)
        try:
            torch.set_num_threads(max_cpu_threads)
            torch.set_num_interop_threads(max(1, max_cpu_threads // 2))
        except RuntimeError:
            # Already configured (likely by GUI wrapper or torch initialized)
            pass
        
        # Lower process priority (to not hog system resources)
        try:
            if sys.platform == 'win32':
                # Windows: Use SetPriorityClass
                import ctypes
                kernel32 = ctypes.windll.kernel32
                BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
                kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
                logger.info("Process priority lowered (Windows BELOW_NORMAL)")
            elif hasattr(os, 'nice'):
                # Unix/Mac: Use nice
                os.nice(10)
                logger.info("Process priority lowered (nice=10)")
        except Exception:
            pass
    
    if node_token:
        logger.info(f"Authenticated with Token: {node_token[:8]}...")
    
    # FULLY DECENTRALIZED INITIALIZATION ORDER:
    # 1. Setup networking FIRST (so DHT is available for layer discovery)
    # 2. Initialize P2P BEFORE creating the node
    # 3. Create node WITH P2P connected (uses DHT for network discovery)
    # This ensures layer assignment can use DHT to detect existing nodes!
    
    token_for_id = node_token or str(uuid.uuid4())
    
    # 1. Setup networking FIRST
    from neuroshard.core.network.nat import NATTraverser
    nat = NATTraverser()
    
    ip_addr = announce_ip or nat.discover_public_ip() or get_public_ip() or get_local_ip()
    local_ip = nat.local_ip  # LAN IP (e.g. 192.168.x.x) for same-network peer detection
    
    # UPnP mapping
    nat.attempt_upnp_mapping(port, "TCP", "NeuroShard HTTP")
    nat.attempt_upnp_mapping(port + 1000, "TCP", "NeuroShard gRPC")
    
    final_announce_port = announce_port or port
    logger.info(f"Announcing as: {ip_addr}:{final_announce_port} (LAN: {local_ip}:{final_announce_port})")
    
    my_url = f"http://{ip_addr}:{final_announce_port}"
    
    # 2. Initialize P2P BEFORE creating the node
    # Use temporary shard_range "unassigned" - will be updated after layer assignment
    # This prevents premature announcement of layer 0 while keeping DHT available for discovery!
    # Pass training_enabled so peers know if we can participate in training pipeline
    # Pass observer_mode to disable proof generation (for explorer nodes)
    P2P = P2PManager(my_url, "unassigned", tracker, node_token=node_token, training_enabled=enable_training, observer_mode=observer_mode)
    P2P.state_ref = STATE
    P2P.public_ip = ip_addr  # Store public IP for LAN peer detection
    P2P.local_ip = local_ip  # Store LAN IP for same-network communication
    
    # Wire LAN IP into DHT so announce() includes both public + local addresses
    if P2P.dht:
        P2P.dht._local_lan_ip = local_ip
    
    # CRITICAL: Synchronously fetch peers and populate routing table BEFORE node creation!
    # The background thread might not have run yet, so we do it explicitly here.
    logger.info("DHT bootstrapping... (discovering existing nodes)")
    import time
    import hashlib as hashlib_module  # Avoid shadowing issues
    
    # Track peers we'll ping for bidirectional DHT connectivity
    peers_to_ping = []
    
    try:
        import requests
        from urllib.parse import urlparse
        from neuroshard.core.network.dht import Node
        
        # Fetch ALL peers from tracker
        resp = requests.get(f"{tracker}/peers", params={"limit": 100}, timeout=5)
        if resp.status_code == 200:
            peers = resp.json()
            peer_count = 0
            for p in peers:
                if p.get("url") != my_url:
                    P2P.known_peers[p["url"]] = p
                    # Add to DHT routing table so layer lookups can find them!
                    if P2P.routing_table:
                        try:
                            p_parsed = urlparse(p["url"])
                            p_ip = p_parsed.hostname
                            p_port = p_parsed.port or 80
                            if p_ip and p_ip != "None":
                                p_id = int(hashlib_module.sha1(f"{p['url']}".encode()).hexdigest(), 16)
                                peer_node = Node(p_id, p_ip, p_port)
                                P2P.routing_table.add_contact(peer_node)
                                peers_to_ping.append((peer_node, p["url"]))
                                peer_count += 1
                        except Exception as e:
                            logger.debug(f"Failed to add peer {p.get('url')}: {e}")
            if peer_count > 0:
                logger.info(f"DHT: Added {peer_count} peers from tracker")
    except Exception as e:
        logger.debug(f"Tracker peer discovery failed: {e}")
    
    # CRITICAL: Ping discovered peers to establish BIDIRECTIONAL DHT connectivity
    # Without this, the peer won't know about us and we can't discover each other's data
    if P2P.dht and peers_to_ping:
        logger.info(f"DHT: Pinging {len(peers_to_ping)} peers for bidirectional connectivity...")
        ping_success = 0
        for peer_node, peer_url in peers_to_ping[:10]:  # Limit to first 10 to avoid delays
            try:
                if P2P.dht.ping(peer_node):
                    ping_success += 1
                    logger.debug(f"  ✓ Pinged {peer_node.ip}:{peer_node.port}")
            except Exception as e:
                logger.debug(f"  ✗ Ping failed for {peer_url}: {e}")
        if ping_success > 0:
            logger.info(f"DHT: Established connectivity with {ping_success} peers")
    
    # =========================================================================
    # SEED PEERS: Direct DHT bootstrap without tracker (fully decentralized)
    # Priority: 1) User-specified seed_peers, 2) Default bootstrap nodes
    # This is how Bitcoin DNS seeds and IPFS bootstrap nodes work.
    # =========================================================================
    
    # Determine which bootstrap nodes to use
    bootstrap_nodes = seed_peers if seed_peers else []
    
    # If no user-specified seeds AND we didn't get peers from tracker,
    # fall back to default bootstrap nodes
    if not bootstrap_nodes and not peers_to_ping:
        bootstrap_nodes = DEFAULT_BOOTSTRAP_NODES
        if bootstrap_nodes:
            logger.info(f"DHT: Using {len(bootstrap_nodes)} default bootstrap nodes...")
    
    if bootstrap_nodes:
        if seed_peers:
            logger.info(f"DHT: Bootstrapping from {len(bootstrap_nodes)} user-specified seed peers...")
        seed_count = 0
        ping_success = 0
        for peer_addr in bootstrap_nodes:
            try:
                # Parse ip:port (handle both hostname:port and ip:port)
                if ':' in peer_addr:
                    peer_ip, peer_port_str = peer_addr.rsplit(':', 1)
                    peer_port = int(peer_port_str)
                else:
                    peer_ip = peer_addr
                    peer_port = 8000  # Default port
                
                # Resolve hostname to IP if needed
                import socket
                try:
                    resolved_ip = socket.gethostbyname(peer_ip)
                    if resolved_ip != peer_ip:
                        logger.debug(f"Resolved {peer_ip} -> {resolved_ip}")
                        peer_ip = resolved_ip
                except socket.gaierror:
                    logger.debug(f"Could not resolve {peer_ip}, using as-is")
                
                # Generate DHT node ID from address
                peer_url = f"http://{peer_ip}:{peer_port}"
                peer_id = int(hashlib_module.sha1(peer_url.encode()).hexdigest(), 16)
                
                # Add to DHT routing table
                if P2P.routing_table:
                    from neuroshard.core.network.dht import Node
                    peer_node = Node(peer_id, peer_ip, peer_port)
                    P2P.routing_table.add_contact(peer_node)
                    seed_count += 1
                    
                    # Ping the peer to establish bidirectional connection
                    # This lets the peer add US to their routing table
                    if P2P.dht:
                        try:
                            if P2P.dht.ping(peer_node):
                                ping_success += 1
                                logger.info(f"  ✓ Connected to {peer_ip}:{peer_port}")
                            else:
                                logger.debug(f"  ✗ Ping failed for {peer_ip}:{peer_port}")
                        except Exception as e:
                            logger.debug(f"  ✗ Ping to {peer_addr} failed: {e}")
            except Exception as e:
                logger.debug(f"Bootstrap node '{peer_addr}' unavailable: {e}")
        
        if ping_success > 0:
            logger.info(f"DHT: Connected to {ping_success}/{seed_count} bootstrap nodes")
        elif seed_count > 0:
            logger.info(f"DHT: Added {seed_count} bootstrap nodes (connectivity pending)")
    
    # Additional wait to let DHT stabilize
    time.sleep(1)
    
    # =========================================================================
    # OBSERVER MODE: Skip model initialization for explorer/block-explorer nodes
    # =========================================================================
    if observer_mode:
        logger.info("=" * 50)
        logger.info("  NeuroShard OBSERVER Node")
        logger.info("=" * 50)
        logger.info("[OBSERVER] Starting in observer mode (no model, no training)")
        logger.info("[OBSERVER] Syncing ledger from network...")
        logger.info("[OBSERVER] Ledger explorer API available at /api/ledger/*")
        
        # Store observer state
        STATE["observer_mode"] = True
        STATE["node_id"] = P2P.ledger_node_id if P2P and P2P.ledger else "observer"
        STATE["assigned_layers"] = []
        STATE["has_embedding"] = False
        STATE["has_lm_head"] = False
        STATE["wallet_id"] = P2P.ledger.node_id if P2P and P2P.ledger else None
        
        # =========================================================================
        # EPOCH MANAGER: Track and verify the chained epoch system
        # =========================================================================
        from neuroshard.core.consensus.epoch import EpochManager
        
        # Initialize epoch manager for the observer
        epoch_manager = EpochManager(
            node_id=STATE["node_id"],
            crypto=P2P.ledger.crypto if P2P and P2P.ledger else None,
            dht=P2P.dht if P2P else None,
            ledger=P2P.ledger if P2P else None,
        )
        epoch_manager.start()
        STATE["epoch_manager"] = epoch_manager
        logger.info("[OBSERVER] Epoch manager started - tracking chained PoNW epochs")
        
        # Start the gRPC server for receiving proof broadcasts
        # Observer mode: pass None for NEURO_NODE since we don't have a model
        start_grpc_background(port, None, P2P, None)
        logger.info(f"[OBSERVER] gRPC server started on port {port + 1000}")
        
        # Log startup complete
        logger.info("=" * 50)
        logger.info("[OBSERVER] NeuroShard Observer Ready!")
        logger.info(f"   Dashboard: http://localhost:{port}/")
        logger.info(f"   Ledger API: http://localhost:{port}/api/ledger/")
        logger.info(f"   Epoch Chain: Tracking at 60-second intervals")
        logger.info("=" * 50)

        # Start the HTTP server
        # Note: uvicorn is already imported at module level (line 18)
        import asyncio as asyncio_lib
        
        async def run_observer():
            """Run observer with background tasks."""
            config = uvicorn.Config(node_app, host="0.0.0.0", port=port, log_level="warning")
            server = uvicorn.Server(config)
            
            # Start background task for stats updates
            async def update_stats():
                while True:
                    try:
                        import psutil
                        proc = psutil.Process()
                        STATE["memory_mb"] = proc.memory_info().rss / (1024 * 1024)
                        STATE["system_cpu"] = psutil.cpu_percent()
                        
                        # Also update epoch stats
                        if "epoch_manager" in STATE:
                            STATE["epoch_info"] = STATE["epoch_manager"].get_current_epoch_info()
                    except:
                        pass
                    await asyncio_lib.sleep(30)
            
            # Start background task for DHT/tracker announcements
            async def announce_loop():
                """Periodic announce to tracker and DHT so nodes can discover us."""
                first_announce = True
                while True:
                    try:
                        # Announce to DHT and tracker
                        P2P._announce_once(verbose=first_announce)
                        first_announce = False
                    except Exception as e:
                        logger.debug(f"[OBSERVER] Announce error: {e}")
                    await asyncio_lib.sleep(30)  # Announce every 30 seconds
            
            # Start background task for epoch chain verification
            async def epoch_verification_loop():
                """Periodically verify epoch chain integrity and sync from network."""
                while True:
                    try:
                        if "epoch_manager" in STATE:
                            em = STATE["epoch_manager"]
                            
                            # Get latest finalized epoch
                            info = em.get_current_epoch_info()
                            logger.info(
                                f"[EPOCH] Current: #{info['epoch_id']}, "
                                f"proofs: {info['proof_count']}, "
                                f"finalized: {info['latest_finalized']}, "
                                f"chain length: {info['chain_length']}"
                            )
                            
                            # Verify recent epoch chain (last 10 epochs)
                            # Note: Gap-tolerant - empty epochs are expected and don't break the chain
                            if info['latest_finalized'] > 0:
                                start_id = max(0, info['latest_finalized'] - 10)
                                is_valid, result_msg = em.verify_epoch_chain(start_id, info['latest_finalized'])
                                if is_valid:
                                    logger.debug(f"[EPOCH] Chain verified: epochs {start_id}-{info['latest_finalized']} ✓ ({result_msg})")
                                else:
                                    logger.warning(f"[EPOCH] Chain verification FAILED: {result_msg}")
                    except Exception as e:
                        logger.error(f"[EPOCH] Verification error: {e}")
                    
                    await asyncio_lib.sleep(120)  # Check every 2 minutes
            
            asyncio_lib.create_task(update_stats())
            asyncio_lib.create_task(announce_loop())
            asyncio_lib.create_task(epoch_verification_loop())
            await server.serve()
        
        asyncio_lib.run(run_observer())
        return  # Exit after server stops
    
    # =========================================================================
    # FULL NODE MODE: Initialize model, training, and quorum system
    # =========================================================================
    logger.info(f"Initializing NeuroShard Node (training={enable_training}, DiLoCo steps={diloco_inner_steps})...")
    
    # 3. Create swarm config
    swarm_config = SwarmNodeConfig(
        diloco_inner_steps=diloco_inner_steps,
    )
    
    # 4. Create node WITH P2P already available
    # This allows layer assignment to use DHT for network discovery!
    NEURO_NODE = create_swarm_node_with_p2p(
        node_token=token_for_id,
        port=port,
        tracker_url=tracker,
        config=swarm_config,
        available_memory_mb=available_memory_mb,
        enable_training=enable_training,
        max_storage_mb=max_storage_mb,
        max_cpu_threads=max_cpu_threads,
        device=device,
        p2p_manager=P2P,  # Pass P2P so DHT is available during layer assignment!
    )
    
    STATE["diloco_inner_steps"] = diloco_inner_steps
    
    logger.info(f"NeuroLLM loaded: {NEURO_NODE.model.get_num_params() / 1e6:.1f}M parameters")
    logger.info(f"Assigned layers: {NEURO_NODE.my_layer_ids}")
    logger.info(f"Embedding: {NEURO_NODE.model.has_embedding}, LM Head: {NEURO_NODE.model.has_lm_head}")
    logger.info(f"DiLoCo: inner_steps={diloco_inner_steps}")
    
    # EARLY NETWORK WARNING
    num_layers = len(NEURO_NODE.my_layer_ids)
    if num_layers > 50:
        logger.warning("⚠️  EARLY NETWORK NOTICE ⚠️")
        logger.warning(f"You're holding {num_layers} layers because the network is small.")
        logger.warning("This is TEMPORARY - as more nodes join, the model will be sharded.")
    
    # Show initial memory usage
    try:
        import psutil
        process = psutil.Process()
        process_mem_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Current memory usage: {process_mem_mb:.0f}MB / {available_memory_mb or '?'}MB allocated")
    except Exception:
        pass
    
    # 5. Update P2P shard_range with actual assigned layers
    # IMPORTANT: Only announce layer 0 if we have embedding (are actual DRIVER)
    # Non-training nodes hold layer 0 weights for redundancy but should NOT
    # announce it, otherwise training nodes will think a DRIVER already exists!
    layer_ids = NEURO_NODE.my_layer_ids
    if layer_ids:
        # If we don't have embedding, skip layer 0 in announcements
        if not NEURO_NODE.model.has_embedding and 0 in layer_ids:
            announce_layers = [l for l in layer_ids if l > 0]
            if announce_layers:
                start_layer = min(announce_layers)
                end_layer = max(announce_layers)
            else:
                start_layer = 1
                end_layer = 1
            logger.info(f"[P2P] Skipping layer 0 announcement (no embedding - redundancy only)")
        else:
            start_layer = min(layer_ids)
            end_layer = max(layer_ids)
        shard_range = f"{start_layer}-{end_layer}"
    else:
        shard_range = "unassigned"
        start_layer = -1
        end_layer = -1
    P2P.shard_range = shard_range
    P2P.start_layer = start_layer
    P2P.end_layer = end_layer
    STATE["shard_range"] = shard_range
    logger.info(f"P2P shard_range: {shard_range} (layers {layer_ids})")
    
    # CRITICAL: Re-announce with correct layers IMMEDIATELY after assignment!
    # The initial announce happened with "0-0" before model creation.
    # This corrects it so other nodes can discover our actual layers.
    P2P._announce_once(verbose=True)
    
    # Set node role info for PoNW reward calculation
    STATE["assigned_layers"] = NEURO_NODE.my_layer_ids
    STATE["has_embedding"] = NEURO_NODE.model.has_embedding
    STATE["has_lm_head"] = NEURO_NODE.model.has_lm_head
    STATE["current_loss"] = NEURO_NODE.current_loss if NEURO_NODE.current_loss != float('inf') else None
    
    logger.info(f"Connected to P2P network for distributed training")
    
    # 4a. Set up ROLE VERIFICATION to prevent fake Validator/Driver claims
    # This is CRITICAL for security - nodes can't claim roles they don't have
    def verify_node_role(node_id: str, claimed_embed: bool, claimed_head: bool):
        """
        Verify that a node actually holds the layers it claims.
        
        Uses THREE sources for verification (defense in depth):
        1. Local layer_pool (authoritative for nodes we know)
        2. DHT lookup (for remote nodes we don't have in local pool)
        3. Tracker query (fallback for unverifiable claims)
        
        Returns: (is_valid, actual_has_embedding, actual_has_lm_head)
        """
        import json
        import hashlib
        
        # 1. LOCAL VERIFICATION (fastest, most authoritative)
        if NEURO_NODE.layer_pool:
            layer_0_holders = [a.node_id for a in NEURO_NODE.layer_pool.get_layer_holders(0)]
            last_layer = max(0, NEURO_NODE.layer_pool.current_num_layers - 1)
            last_layer_holders = [a.node_id for a in NEURO_NODE.layer_pool.get_layer_holders(last_layer)]
            
            # Check if we know this node locally
            all_known_nodes = set(layer_0_holders + last_layer_holders)
            for assignments in NEURO_NODE.layer_pool.layer_assignments.values():
                for a in assignments:
                    all_known_nodes.add(a.node_id)
            
            if node_id in all_known_nodes:
                # We know this node - verify against local data
                actual_embed = node_id in layer_0_holders
                actual_head = node_id in last_layer_holders
                
                is_valid = True
                if claimed_head and not actual_head:
                    is_valid = False
                if claimed_embed and not actual_embed:
                    is_valid = False
                
                return is_valid, actual_embed, actual_head
        
        # 2. HEARTBEAT/PEER_STATS VERIFICATION (from swarm router)
        # Heartbeats contain node_id AND layer_range - this is the best source for remote nodes!
        # Note: swarm_components contains SwarmComponents (router, buffers, etc.)
        if hasattr(NEURO_NODE, 'swarm_components') and NEURO_NODE.swarm_components and hasattr(NEURO_NODE.swarm_components, 'swarm_router'):
            router = NEURO_NODE.swarm_components.swarm_router
            if hasattr(router, 'peer_stats') and node_id in router.peer_stats:
                peer = router.peer_stats[node_id]
                layer_range = peer.layer_range  # (start, end) tuple
                
                # Get last layer from our layer pool
                last_layer = max(0, NEURO_NODE.layer_pool.current_num_layers - 1) if NEURO_NODE.layer_pool else 0
                
                # Driver = holds layer 0
                actual_embed = layer_range[0] == 0
                # Validator = holds last layer  
                actual_head = last_layer in range(layer_range[0], layer_range[1])
                
                is_valid = True
                if claimed_head and not actual_head:
                    is_valid = False
                if claimed_embed and not actual_embed:
                    is_valid = False
                
                logger.debug(f"Role verification via heartbeat: {node_id[:16]}... "
                           f"layers={layer_range}, embed={actual_embed}, head={actual_head}")
                return is_valid, actual_embed, actual_head
        
        # 3. FALLBACK: For unknown nodes, use CONSERVATIVE verification
        # NOTE: DHT stores IP:port not node_id, so we can't verify roles via DHT alone
        # If we can't verify, we have two options:
        # a) REJECT all unknown claims (secure but might reject valid proofs)
        # b) ACCEPT but cap rewards (economic security)
        # 
        # We use option (b) - the proof is ACCEPTED but role bonuses are NOT applied
        # This is handled in _calculate_reward by checking verified roles
        
        # For now, if we can't verify, return "claims not verified"
        # The reward calculation should treat unverified claims as false
        logger.debug(f"Role verification: Node {node_id[:16]}... not in local pool, claims unverifiable")
        
        # Return: valid=True (don't reject), but actual roles = False (no bonus)
        # This allows the proof through but without Validator/Driver bonuses
        return True, False, False
    
    P2P.ledger.set_role_verifier(verify_node_role)
    logger.info("Role verification enabled - fake Validator/Driver claims will be REJECTED")
    
    # Set model interface for training work verification
    P2P.ledger.set_model_interface(NEURO_NODE)
    
    # 4b. Start Swarm components
    if hasattr(NEURO_NODE, 'start_swarm_sync'):
        logger.info("[SWARM] Starting swarm components...")
        NEURO_NODE.start_swarm_sync()
        logger.info("[SWARM] Swarm components started")
    
    # =========================================================================
    # NATIVE ARCHITECTURE: Initialize Quorum-Based Training System
    # =========================================================================
    global QUORUM_REGISTRY, QUORUM_FORMATION, QUORUM_TRAINER, QUORUM_INFERENCE_ROUTER
    global PROOF_VERIFIER, DHT_PROTOCOL, CURRENT_QUORUM
    
    logger.info("[QUORUM] Initializing quorum system...")
    
    # Initialize DHT Protocol for peer discovery
    try:
        from neuroshard.core.network.dht import Node as DHTNode
        # Convert node_id (hex string) to int for DHT
        node_id_int = int(NEURO_NODE.node_id[:32], 16) if NEURO_NODE.node_id else 0
        local_node = DHTNode(node_id=node_id_int, ip=ip_addr, port=port)
        
        if P2P.routing_table:
            DHT_PROTOCOL = DHTProtocol(
                local_node=local_node,
                routing_table=P2P.routing_table,
                port=port,
            )
            logger.info("[QUORUM] DHT Protocol initialized")
        else:
            logger.warning("[QUORUM] No routing table available, DHT Protocol disabled")
            DHT_PROTOCOL = None
    except Exception as e:
        logger.warning(f"[QUORUM] DHT Protocol init failed: {e}, using fallback")
        DHT_PROTOCOL = None
    
    # Initialize Quorum Registry (DHT-backed)
    QUORUM_REGISTRY = QuorumRegistry(
        dht_protocol=DHT_PROTOCOL,
    )
    logger.info("[QUORUM] QuorumRegistry initialized")
    
    # Initialize Quorum Formation Service
    QUORUM_FORMATION = QuorumFormationService(
        registry=QUORUM_REGISTRY,
        layer_pool=NEURO_NODE.layer_pool,
        dht_protocol=DHT_PROTOCOL,
        p2p_manager=P2P,  # For decentralized peer discovery via known_peers
    )
    logger.info("[QUORUM] QuorumFormationService initialized")
    
    # Initialize Quorum Inference Router
    QUORUM_INFERENCE_ROUTER = QuorumInferenceRouter(
        registry=QUORUM_REGISTRY,
        dht_protocol=DHT_PROTOCOL,
    )
    logger.info("[QUORUM] QuorumInferenceRouter initialized")
    
    # Initialize Proof Verifier for PoNW
    PROOF_VERIFIER = ProofVerifier(
        optimistic=True,  # Optimistic acceptance for liveness
        network_size=len(P2P.known_peers) + 1,
    )
    logger.info("[QUORUM] ProofVerifier initialized (optimistic mode)")
    
    # Determine speed tier from hardware
    speed_tier = _get_speed_tier(NEURO_NODE)
    STATE["speed_tier"] = speed_tier.value  # Store string for JSON serialization
    STATE["speed_tier_enum"] = speed_tier    # Keep enum for internal use
    logger.info(f"[QUORUM] Node speed tier: {speed_tier.value}")
    
    # Form or join a quorum if training is enabled
    if enable_training:
        logger.info("[QUORUM] Attempting quorum formation for training...")
        
        # Get gRPC endpoint
        grpc_port = port + 1000  # Convention: gRPC is HTTP port + 1000
        grpc_endpoint = f"{ip_addr}:{grpc_port}"
        
        # Total layers from model
        total_layers = len(NEURO_NODE.my_layer_ids) if NEURO_NODE.my_layer_ids else 1
        if NEURO_NODE.layer_pool:
            total_layers = NEURO_NODE.layer_pool.current_num_layers
        
        # Form or join quorum
        CURRENT_QUORUM = QUORUM_FORMATION.form_quorum(
            initiator_node_id=NEURO_NODE.node_id,
            initiator_endpoint=grpc_endpoint,
            initiator_layers=(min(layer_ids) if layer_ids else 0, max(layer_ids) + 1 if layer_ids else 1),
            initiator_speed_tier=speed_tier.value,
            total_layers=total_layers,
        )
        
        if CURRENT_QUORUM:
            STATE["quorum_id"] = CURRENT_QUORUM.quorum_id
            STATE["quorum_lifecycle"] = CURRENT_QUORUM.lifecycle.value
            STATE["quorum_members"] = len(CURRENT_QUORUM.members)
            
            if CURRENT_QUORUM.lifecycle == QuorumLifecycle.ACTIVE:
                logger.info(f"[QUORUM] Joined ACTIVE quorum {CURRENT_QUORUM.quorum_id[:8]}... "
                           f"with {len(CURRENT_QUORUM.members)} members")
                
                # Ensure genesis_loader is initialized for initiator nodes (holds embedding)
                # Pipeline mode needs training data just like async mode
                base_node = getattr(NEURO_NODE, 'base_node', NEURO_NODE)
                if not hasattr(base_node, 'genesis_loader') or base_node.genesis_loader is None:
                    try:
                        from neuroshard.core.training.distributed import GenesisDataLoader
                        from neuroshard.core.model.tokenizer import get_neuro_tokenizer
                        logger.info("[GENESIS] Initializing data loader for pipeline training...")
                        base_node.genesis_loader = GenesisDataLoader(
                            NEURO_NODE.node_id,
                            get_neuro_tokenizer(),
                            max_storage_mb=max_storage_mb
                        )
                        logger.info(f"[GENESIS] Data loader ready: {base_node.genesis_loader.total_shards} shards available")
                    except Exception as e:
                        logger.warning(f"[GENESIS] Failed to initialize data loader: {e}")
                
                # Initialize QuorumTrainer
                QUORUM_TRAINER = QuorumTrainer(
                    quorum=CURRENT_QUORUM,
                    node_id=NEURO_NODE.node_id,
                    model=NEURO_NODE.model,
                    optimizer=NEURO_NODE.optimizer,
                    genesis_loader=getattr(base_node, 'genesis_loader', None),
                    dht_protocol=DHT_PROTOCOL,
                )
                # Wire inbound buffer to gRPC handler for pipeline training
                from neuroshard.grpc_server import GRPC_SERVICER as _grpc_svc
                if _grpc_svc and QUORUM_TRAINER.inbound_buffer:
                    _grpc_svc.quorum_inbound_buffer = QUORUM_TRAINER.inbound_buffer
                    _grpc_svc._active_quorum_id = CURRENT_QUORUM.quorum_id
                logger.info("[QUORUM] QuorumTrainer initialized")
            else:
                logger.info(f"[QUORUM] Quorum {CURRENT_QUORUM.quorum_id[:8]}... is FORMING "
                           f"(partial layer coverage) — will use ASYNC training mode")
        else:
            logger.info("[QUORUM] No quorum formed yet, will retry in background...")
    else:
        logger.info("[QUORUM] Training disabled, skipping quorum formation")
    
    # =========================================================================
    # SELECT CONTRIBUTION MODE (per whitepaper)
    # =========================================================================
    global CURRENT_CONTRIBUTION_MODE, LAYER_GROWTH_MANAGER
    
    # Determine contribution mode based on current state
    has_layers = bool(NEURO_NODE.my_layer_ids)
    has_quorum = CURRENT_QUORUM is not None and CURRENT_QUORUM.lifecycle == QuorumLifecycle.ACTIVE
    has_genesis_data = hasattr(NEURO_NODE, 'genesis_loader') and NEURO_NODE.genesis_loader is not None
    has_stake = P2P.ledger.get_account_info().get("stake", 0.0) >= 10.0 if P2P and P2P.ledger else False
    
    CURRENT_CONTRIBUTION_MODE = select_contribution_mode(
        speed_tier=speed_tier,
        has_layers=has_layers,
        has_quorum=has_quorum,
        has_genesis_data=has_genesis_data,
        has_stake=has_stake,
        network_needs_verifiers=False,  # TODO: Query from network state
        network_needs_data_providers=False,
    )
    
    STATE["contribution_mode"] = CURRENT_CONTRIBUTION_MODE.value
    logger.info(f"[MODE] Contribution mode: {CURRENT_CONTRIBUTION_MODE.value}")
    
    # T5 nodes can't do real-time pipeline, log this
    if speed_tier == SpeedTier.T5:
        logger.info("[MODE] T5 (slow) node - using async training mode")
    
    # Initialize AsyncTrainer if mode is ASYNC and training is enabled
    global ASYNC_TRAINER
    if enable_training and CURRENT_CONTRIBUTION_MODE == ContributionMode.ASYNC:
        logger.info("[ASYNC] Initializing AsyncTrainer for async contribution mode...")
        
        # Ensure genesis_loader is initialized (it's lazy by default)
        # Note: We set on base_node because SwarmEnabledDynamicNode wraps it
        base_node = getattr(NEURO_NODE, 'base_node', NEURO_NODE)
        if not hasattr(base_node, 'genesis_loader') or base_node.genesis_loader is None:
            try:
                from neuroshard.core.training.distributed import GenesisDataLoader
                from neuroshard.core.model.tokenizer import get_neuro_tokenizer
                logger.info("[GENESIS] Initializing data loader for async training...")
                base_node.genesis_loader = GenesisDataLoader(
                    NEURO_NODE.node_id,
                    get_neuro_tokenizer(),
                    max_storage_mb=max_storage_mb
                )
                logger.info(f"[GENESIS] Data loader ready: {base_node.genesis_loader.total_shards} shards available")
            except Exception as e:
                logger.warning(f"[GENESIS] Failed to initialize data loader: {e}")
        
        ASYNC_TRAINER = AsyncTrainer(
            node_id=NEURO_NODE.node_id,
            model=NEURO_NODE.model,
            optimizer=NEURO_NODE.optimizer,
            genesis_loader=base_node.genesis_loader,
            dht_protocol=DHT_PROTOCOL,
        )
        
        # Set up callback for P2P to get fresh training stats
        # This bypasses timing issues where STATE isn't updated yet
        def get_fresh_training_stats():
            if ASYNC_TRAINER:
                return ASYNC_TRAINER.get_stats()
            return None
        
        # Callback to notify AsyncTrainer when a proof is accepted
        def on_proof_accepted():
            if ASYNC_TRAINER:
                ASYNC_TRAINER.reset_proof_period()
        
        if P2P:
            P2P.get_fresh_training_stats = get_fresh_training_stats
            P2P.on_proof_accepted = on_proof_accepted
        logger.info("[ASYNC] AsyncTrainer initialized")
    
    # Initialize LayerGrowthManager for monitoring network growth
    LAYER_GROWTH_MANAGER = LayerGrowthManager(dht_protocol=DHT_PROTOCOL)
    logger.info("[GROWTH] LayerGrowthManager initialized")
    
    logger.info("[QUORUM] initialization complete")
    # =========================================================================
    
    # 5. Start gRPC Server
    start_grpc_background(port, NEURO_NODE, P2P, None)
    
    # 5b. Register quorum callbacks on gRPC servicer
    # This allows the ProposeQuorum RPC handler to set up the quorum on this side
    # when another node proposes a pipeline quorum.
    from neuroshard.grpc_server import GRPC_SERVICER
    
    def _on_quorum_proposed(quorum_id, proposer_info, my_info, total_layers):
        """Callback: another node proposed a pipeline quorum and we accepted."""
        global CURRENT_QUORUM, QUORUM_TRAINER, CURRENT_CONTRIBUTION_MODE, ASYNC_TRAINER
        
        logger.info(f"[QUORUM] Setting up accepted quorum {quorum_id[:8]}... "
                    f"(proposed by {proposer_info['node_id'][:8]})")
        
        try:
            # Stop any running async trainer
            if ASYNC_TRAINER and ASYNC_TRAINER.running:
                logger.info("[ASYNC] Stopping AsyncTrainer (joining pipeline quorum)")
                ASYNC_TRAINER.stop()
            
            # Stop any existing quorum trainer
            if QUORUM_TRAINER and QUORUM_TRAINER.running:
                QUORUM_TRAINER.stop()
                QUORUM_TRAINER = None
            
            # Determine quorum type: SYNC if both sides are full replicas
            all_layers = set(range(total_layers))
            my_layers = set(range(my_info['layer_range'][0], my_info['layer_range'][1]))
            proposer_layers = set(range(proposer_info['layer_range'][0], proposer_info['layer_range'][1]))
            both_full_replicas = (my_layers >= all_layers) and (proposer_layers >= all_layers)
            q_type = QuorumType.SYNC if both_full_replicas else QuorumType.PIPELINE
            
            # Create the quorum with both members
            quorum = Quorum(
                quorum_id=quorum_id,
                speed_tier=my_info['speed_tier'],
                quorum_type=q_type,
                arch_version=1,
                vocab_version=1,
            )
            
            # Add proposer
            proposer_member = QuorumMember(
                node_id=proposer_info['node_id'],
                endpoint=proposer_info['endpoint'],
                layer_range=tuple(proposer_info['layer_range']),
                speed_tier=proposer_info['speed_tier'],
            )
            quorum.add_member(proposer_member)
            
            # Add ourselves
            my_member = QuorumMember(
                node_id=my_info['node_id'],
                endpoint=my_info['endpoint'],
                layer_range=tuple(my_info['layer_range']),
                speed_tier=my_info['speed_tier'],
            )
            quorum.add_member(my_member)
            
            # Check completeness
            if quorum.layer_coverage >= all_layers:
                quorum.lifecycle = QuorumLifecycle.ACTIVE
            else:
                quorum.lifecycle = QuorumLifecycle.FORMING
            
            # Register quorum
            QUORUM_REGISTRY.register_quorum(quorum)
            CURRENT_QUORUM = quorum
            
            if quorum.lifecycle == QuorumLifecycle.ACTIVE:
                # Initialize QuorumTrainer
                base_node = getattr(NEURO_NODE, 'base_node', NEURO_NODE)
                if not hasattr(base_node, 'genesis_loader') or base_node.genesis_loader is None:
                    try:
                        from neuroshard.core.training.distributed import GenesisDataLoader
                        from neuroshard.core.model.tokenizer import get_neuro_tokenizer
                        base_node.genesis_loader = GenesisDataLoader(
                            NEURO_NODE.node_id,
                            get_neuro_tokenizer(),
                            max_storage_mb=max_storage_mb
                        )
                    except Exception as e:
                        logger.warning(f"[GENESIS] Data loader init failed: {e}")
                
                QUORUM_TRAINER = QuorumTrainer(
                    quorum=CURRENT_QUORUM,
                    node_id=NEURO_NODE.node_id,
                    model=NEURO_NODE.model,
                    optimizer=NEURO_NODE.optimizer,
                    genesis_loader=getattr(base_node, 'genesis_loader', None),
                    dht_protocol=DHT_PROTOCOL,
                )
                
                # Wire the inbound buffer to the gRPC handler
                if GRPC_SERVICER and QUORUM_TRAINER.inbound_buffer:
                    GRPC_SERVICER.quorum_inbound_buffer = QUORUM_TRAINER.inbound_buffer
                    GRPC_SERVICER._active_quorum_id = quorum_id
                    GRPC_SERVICER._active_quorum_type = getattr(quorum, 'quorum_type', QuorumType.PIPELINE).value
                
                QUORUM_TRAINER.start()
                CURRENT_CONTRIBUTION_MODE = ContributionMode.PIPELINE
                STATE["contribution_mode"] = "pipeline"
                STATE["training_mode"] = "quorum"
                STATE["training_status"] = "active"
                STATE["quorum_id"] = quorum_id
                STATE["quorum_lifecycle"] = "active"
                STATE["quorum_members"] = len(quorum.members)
                
                logger.info(f"[QUORUM] Pipeline quorum {quorum_id[:8]}... ACTIVE! "
                           f"Role: {QUORUM_TRAINER.member.role.value}, "
                           f"layers: {my_info['layer_range']}")
            else:
                logger.info(f"[QUORUM] Quorum {quorum_id[:8]}... is FORMING (partial coverage)")
                
        except Exception as e:
            logger.error(f"[QUORUM] Failed to set up accepted quorum: {e}")
            import traceback
            traceback.print_exc()
    
    if GRPC_SERVICER:
        GRPC_SERVICER.on_quorum_proposed = _on_quorum_proposed
        logger.info("[QUORUM] Registered ProposeQuorum callback on gRPC servicer")
    
    # 6. Background tasks - Native Quorum-Based System (no legacy fallbacks)
    def background_tasks():
        """
        Background Task Loop
        
        Training is ONLY done via QuorumTrainer:
        - Form/join a quorum with speed-matched peers
        - QuorumTrainer handles all training in its own thread
        - DiLoCo sync happens across quorums automatically
        - PoNW proofs are generated by the trainer
        
        This loop handles:
        - Quorum monitoring and reformation
        - State updates for dashboard
        - Marketplace and housekeeping tasks
        """
        global QUORUM_TRAINER, CURRENT_QUORUM, QUORUM_FORMATION
        global CURRENT_CONTRIBUTION_MODE, ASYNC_TRAINER, LAYER_GROWTH_MANAGER
        
        import psutil
        
        # Store config
        STATE["config_cpu_threads"] = max_cpu_threads
        STATE["config_memory_mb"] = available_memory_mb
        STATE["config_storage_mb"] = max_storage_mb
        
        # Start training based on contribution mode
        if enable_training:
            if CURRENT_CONTRIBUTION_MODE == ContributionMode.PIPELINE and QUORUM_TRAINER and CURRENT_QUORUM:
                if CURRENT_QUORUM.lifecycle == QuorumLifecycle.ACTIVE:
                    logger.info("[QUORUM] Starting QuorumTrainer (PIPELINE mode)...")
                    QUORUM_TRAINER.start()
                    STATE["training_mode"] = "quorum"
                    STATE["training_status"] = "active"
                else:
                    STATE["training_mode"] = "forming"
                    STATE["training_status"] = "waiting_for_quorum"
            elif CURRENT_CONTRIBUTION_MODE == ContributionMode.ASYNC and ASYNC_TRAINER:
                logger.info("[ASYNC] Starting AsyncTrainer (ASYNC mode)...")
                ASYNC_TRAINER.start()
                STATE["training_mode"] = "async"
                STATE["training_status"] = "async_training"
            else:
                STATE["training_mode"] = "forming"
                STATE["training_status"] = "waiting_for_quorum"
        else:
            STATE["training_mode"] = "disabled"
            STATE["training_status"] = "disabled"
        
        # Timing
        last_quorum_check = 0
        last_memory_report = 0
        last_heartbeat = 0
        last_layer_growth_check = 0
        last_tokens = NEURO_NODE.total_tokens_processed if NEURO_NODE else 0
        last_training_rounds = NEURO_NODE.total_training_rounds if NEURO_NODE else 0
        last_quorum_batches = 0  # Track last quorum batch count for delta calculation
        last_async_batches = 0   # Track last async batch count for delta calculation
        forming_retry_count = 0   # Track consecutive FORMING retries for backoff
        last_forming_retry = 0    # Timestamp of last FORMING retry
        last_sync_peer_check = 0  # Timestamp of last sync peer discovery check
        
        QUORUM_CHECK_INTERVAL = 10  # Check quorum every 10 seconds
        MEMORY_REPORT_INTERVAL = 60
        HEARTBEAT_INTERVAL = 30
        LAYER_GROWTH_CHECK_INTERVAL = 60  # Check layer growth every 60 seconds
        CHECKPOINT_SAVE_INTERVAL = 300   # Save checkpoint every 5 minutes (crash protection)
        last_checkpoint_save = time.time()  # Don't save immediately on startup
        SYNC_PEER_CHECK_INTERVAL = 30    # Check for new sync peers every 30s
        FORMING_RETRY_BASE = 10          # Base retry interval for FORMING pipeline quorums
        FORMING_RETRY_MAX = 60           # Max retry interval (cap backoff at 60s)
        
        logger.info("[QUORUM] Background task loop started")
        
        while not _SHUTDOWN_REQUESTED.is_set():
            now = time.time()
            
            # Update peer count
            STATE["peer_count"] = len(P2P.known_peers)
            
            # =================================================================
            # QUORUM MANAGEMENT (every 10 seconds)
            # =================================================================
            if now - last_quorum_check >= QUORUM_CHECK_INTERVAL:
                last_quorum_check = now
                
                if CURRENT_QUORUM:
                    STATE["quorum_id"] = CURRENT_QUORUM.quorum_id
                    STATE["quorum_lifecycle"] = CURRENT_QUORUM.lifecycle.value
                    STATE["quorum_members"] = len(CURRENT_QUORUM.members)
                    
                    # Quorum became ACTIVE - start trainer
                    if CURRENT_QUORUM.lifecycle == QuorumLifecycle.ACTIVE:
                        if not QUORUM_TRAINER:
                            logger.info("[QUORUM] Quorum ACTIVE, creating QuorumTrainer...")
                            # Ensure genesis loader is available
                            _base = getattr(NEURO_NODE, 'base_node', NEURO_NODE)
                            if not hasattr(_base, 'genesis_loader') or _base.genesis_loader is None:
                                try:
                                    from neuroshard.core.training.distributed import GenesisDataLoader
                                    from neuroshard.core.model.tokenizer import get_neuro_tokenizer
                                    _base.genesis_loader = GenesisDataLoader(
                                        NEURO_NODE.node_id,
                                        get_neuro_tokenizer(),
                                        max_storage_mb=max_storage_mb
                                    )
                                except Exception as e:
                                    logger.warning(f"[GENESIS] Data loader init failed: {e}")
                            QUORUM_TRAINER = QuorumTrainer(
                                quorum=CURRENT_QUORUM,
                                node_id=NEURO_NODE.node_id,
                                model=NEURO_NODE.model,
                                optimizer=NEURO_NODE.optimizer,
                                genesis_loader=getattr(_base, 'genesis_loader', None),
                                dht_protocol=DHT_PROTOCOL,
                            )
                            # Wire inbound buffer to gRPC handler
                            from neuroshard.grpc_server import GRPC_SERVICER as _gs
                            if _gs and QUORUM_TRAINER.inbound_buffer:
                                _gs.quorum_inbound_buffer = QUORUM_TRAINER.inbound_buffer
                                _gs._active_quorum_id = CURRENT_QUORUM.quorum_id
                                _gs._active_quorum_type = getattr(CURRENT_QUORUM, 'quorum_type', None)
                                if _gs._active_quorum_type:
                                    _gs._active_quorum_type = _gs._active_quorum_type.value
                            QUORUM_TRAINER.start()
                            STATE["training_mode"] = "quorum"
                            STATE["training_status"] = "active"
                            forming_retry_count = 0  # Reset backoff on success
                            logger.info("[QUORUM] QuorumTrainer started!")
                        elif not QUORUM_TRAINER.running:
                            QUORUM_TRAINER.start()
                            STATE["training_status"] = "active"
                        
                        # SYNC quorums: periodically discover new sync peers
                        # SYNC quorums stay alive — we just add new members, never dissolve.
                        quorum_type = getattr(CURRENT_QUORUM, 'quorum_type', None)
                        if (quorum_type == QuorumType.SYNC
                                and enable_training and QUORUM_FORMATION
                                and now - last_sync_peer_check >= SYNC_PEER_CHECK_INTERVAL):
                            last_sync_peer_check = now
                            try:
                                layer_ids = NEURO_NODE.my_layer_ids
                                total_layers = NEURO_NODE.layer_pool.current_num_layers if NEURO_NODE.layer_pool else 1
                                if layer_ids:
                                    # Look for peers that hold any layers (potential sync partners)
                                    peers = QUORUM_FORMATION._find_compatible_peers(
                                        STATE.get("speed_tier", "tier5"),
                                        {0},  # Just check for layer 0 holders (full replicas have layer 0)
                                    )
                                    # Filter out ourselves and nodes already in our quorum
                                    my_node_id = NEURO_NODE.node_id
                                    existing_endpoints = {m.endpoint for m in CURRENT_QUORUM.members}
                                    existing_node_ids = {m.node_id for m in CURRENT_QUORUM.members}
                                    # Also filter by IP — peer found via tracker may use our public IP
                                    my_ips = {ip_addr}
                                    if local_ip:
                                        my_ips.add(local_ip)
                                    new_peers = []
                                    for p in peers:
                                        ep = p.get("endpoint", "")
                                        # Extract IP from endpoint (ip:port)
                                        peer_ip = ep.rsplit(":", 1)[0] if ":" in ep else ep
                                        if peer_ip in my_ips:
                                            continue  # That's us
                                        if p.get("node_id") in existing_node_ids:
                                            continue  # Already in quorum
                                        if ep in existing_endpoints:
                                            continue  # Already in quorum
                                        new_peers.append(p)
                                    if new_peers:
                                        logger.info(f"[QUORUM] SYNC: Found {len(new_peers)} new sync peer(s) "
                                                   f"— proposing quorum join")
                            except Exception as e:
                                logger.debug(f"[QUORUM] Sync peer-check error: {e}")
                    
                    # Quorum still FORMING — retry with exponential backoff
                    # Only PIPELINE quorums can be FORMING (SYNC quorums are always ACTIVE)
                    elif CURRENT_QUORUM.lifecycle == QuorumLifecycle.FORMING:
                        if enable_training and QUORUM_FORMATION:
                            # Exponential backoff: 10s, 20s, 40s, 60s (capped)
                            retry_interval = min(
                                FORMING_RETRY_BASE * (2 ** forming_retry_count),
                                FORMING_RETRY_MAX,
                            )
                            
                            if now - last_forming_retry >= retry_interval:
                                last_forming_retry = now
                                forming_retry_count += 1
                                layer_ids = NEURO_NODE.my_layer_ids
                                total_layers = NEURO_NODE.layer_pool.current_num_layers if NEURO_NODE.layer_pool else 1
                                
                                if layer_ids:
                                    # Dissolve the incomplete quorum and retry
                                    # form_quorum() will discover any new peers that have
                                    # appeared since the last attempt
                                    CURRENT_QUORUM = None
                                    if QUORUM_TRAINER and QUORUM_TRAINER.running:
                                        QUORUM_TRAINER.stop()
                                        QUORUM_TRAINER = None
                                    
                                    grpc_port = port + 1000
                                    grpc_endpoint = f"{ip_addr}:{grpc_port}"
                                    CURRENT_QUORUM = QUORUM_FORMATION.form_quorum(
                                        initiator_node_id=NEURO_NODE.node_id,
                                        initiator_endpoint=grpc_endpoint,
                                        initiator_layers=(min(layer_ids), max(layer_ids) + 1),
                                        initiator_speed_tier=STATE.get("speed_tier", "tier5"),
                                        total_layers=total_layers,
                                    )
                                    
                                    # Reset backoff if we formed an ACTIVE quorum
                                    if CURRENT_QUORUM and CURRENT_QUORUM.lifecycle == QuorumLifecycle.ACTIVE:
                                        forming_retry_count = 0
                    
                    # Quorum dissolved - stop trainer and reform
                    elif CURRENT_QUORUM.lifecycle in [QuorumLifecycle.DISSOLVED, QuorumLifecycle.DISSOLVING]:
                        if QUORUM_TRAINER and QUORUM_TRAINER.running:
                            logger.info("[QUORUM] Quorum dissolving, stopping QuorumTrainer...")
                            QUORUM_TRAINER.stop()
                            QUORUM_TRAINER = None
                            STATE["training_status"] = "reforming"
                        
                        # Try to form new quorum
                        if enable_training and QUORUM_FORMATION:
                            logger.info("[QUORUM] Reforming quorum...")
                            forming_retry_count = 0  # Reset backoff on dissolve/reform
                            grpc_port = port + 1000
                            grpc_endpoint = f"{ip_addr}:{grpc_port}"
                            layer_ids = NEURO_NODE.my_layer_ids
                            total_layers = NEURO_NODE.layer_pool.current_num_layers if NEURO_NODE.layer_pool else 1
                            
                            if layer_ids:  # Guard against empty layer list
                                CURRENT_QUORUM = QUORUM_FORMATION.form_quorum(
                                    initiator_node_id=NEURO_NODE.node_id,
                                    initiator_endpoint=grpc_endpoint,
                                    initiator_layers=(min(layer_ids), max(layer_ids) + 1),
                                    initiator_speed_tier=STATE.get("speed_tier", "tier5"),
                                    total_layers=total_layers,
                                )
                                if CURRENT_QUORUM:
                                    logger.info(f"[QUORUM] Joined quorum {CURRENT_QUORUM.quorum_id[:8]}...")
                
                # No quorum yet - try to form one
                elif enable_training and QUORUM_FORMATION:
                    logger.debug("[QUORUM] No quorum, attempting formation...")
                    forming_retry_count = 0  # Reset backoff on fresh formation
                    last_forming_retry = now
                    grpc_port = port + 1000
                    grpc_endpoint = f"{ip_addr}:{grpc_port}"
                    layer_ids = NEURO_NODE.my_layer_ids
                    total_layers = NEURO_NODE.layer_pool.current_num_layers if NEURO_NODE.layer_pool else 1
                    
                    if layer_ids:  # Guard against empty layer list
                        CURRENT_QUORUM = QUORUM_FORMATION.form_quorum(
                            initiator_node_id=NEURO_NODE.node_id,
                            initiator_endpoint=grpc_endpoint,
                            initiator_layers=(min(layer_ids), max(layer_ids) + 1),
                            initiator_speed_tier=STATE.get("speed_tier", "tier5"),
                            total_layers=total_layers,
                        )
                        if CURRENT_QUORUM:
                            logger.info(f"[QUORUM] Formed quorum {CURRENT_QUORUM.quorum_id[:8]}...")
                
                # Update training stats from QuorumTrainer
                if QUORUM_TRAINER and QUORUM_TRAINER.running:
                    current_quorum_batches = QUORUM_TRAINER.total_batches
                    STATE["quorum_batches"] = current_quorum_batches
                    STATE["quorum_loss"] = QUORUM_TRAINER.current_loss
                    STATE["quorum_sync_round"] = QUORUM_TRAINER.sync_round
                    # Add DELTA batches (not cumulative) - P2P resets training_batches after each proof
                    batch_delta = current_quorum_batches - last_quorum_batches
                    if batch_delta > 0:
                        STATE["training_batches"] = STATE.get("training_batches", 0) + batch_delta
                        last_quorum_batches = current_quorum_batches
                    STATE["last_loss"] = QUORUM_TRAINER.current_loss
                    STATE["current_loss"] = QUORUM_TRAINER.current_loss
                    
                    # Detect stalled pipeline (processor not receiving activations)
                    if getattr(QUORUM_TRAINER, '_pipeline_stalled', False):
                        logger.warning("[QUORUM] Pipeline stalled — processor not receiving "
                                      "activations. Switching to ASYNC training mode.")
                        QUORUM_TRAINER.stop()
                        QUORUM_TRAINER = None
                        # Disconnect inbound buffer from gRPC handler
                        from neuroshard.grpc_server import GRPC_SERVICER as _gs_stall
                        if _gs_stall:
                            _gs_stall.quorum_inbound_buffer = None
                            _gs_stall._active_quorum_id = None
                        if CURRENT_QUORUM:
                            CURRENT_QUORUM.lifecycle = QuorumLifecycle.DISSOLVED
                        CURRENT_QUORUM = None
                        STATE["training_status"] = "switching_to_async"
                        # Force ASYNC mode re-evaluation on next cycle
                        CURRENT_CONTRIBUTION_MODE = ContributionMode.ASYNC
                        STATE["contribution_mode"] = "async"
                
                # =============================================================
                # CONTRIBUTION MODE UPDATE (per whitepaper)
                # =============================================================
                # Re-evaluate contribution mode based on current state
                has_layers = bool(NEURO_NODE.my_layer_ids)
                has_quorum = CURRENT_QUORUM is not None and CURRENT_QUORUM.lifecycle == QuorumLifecycle.ACTIVE
                has_genesis_data = hasattr(NEURO_NODE, 'genesis_loader') and NEURO_NODE.genesis_loader is not None
                has_stake = P2P.ledger.get_account_info().get("stake", 0.0) >= 10.0 if P2P and P2P.ledger else False
                current_speed_tier = STATE.get("speed_tier_enum", SpeedTier.T5)
                
                new_mode = select_contribution_mode(
                    speed_tier=current_speed_tier,
                    has_layers=has_layers,
                    has_quorum=has_quorum,
                    has_genesis_data=has_genesis_data,
                    has_stake=has_stake,
                    network_needs_verifiers=False,
                    network_needs_data_providers=False,
                )
                
                # Log mode changes
                if new_mode != CURRENT_CONTRIBUTION_MODE:
                    old_mode = CURRENT_CONTRIBUTION_MODE
                    logger.info(f"[MODE] Contribution mode changed: {old_mode.value} -> {new_mode.value}")
                    CURRENT_CONTRIBUTION_MODE = new_mode
                    STATE["contribution_mode"] = new_mode.value
                    
                    # Handle mode-specific transitions
                    if new_mode == ContributionMode.PIPELINE and has_quorum:
                        # Stop async trainer if running
                        if ASYNC_TRAINER and ASYNC_TRAINER.running:
                            logger.info("[ASYNC] Stopping AsyncTrainer (switching to PIPELINE)")
                            ASYNC_TRAINER.stop()
                        
                        # Create and start QuorumTrainer for pipeline mode
                        if not QUORUM_TRAINER and CURRENT_QUORUM:
                            _base = getattr(NEURO_NODE, 'base_node', NEURO_NODE)
                            if not hasattr(_base, 'genesis_loader') or _base.genesis_loader is None:
                                try:
                                    from neuroshard.core.training.distributed import GenesisDataLoader
                                    from neuroshard.core.model.tokenizer import get_neuro_tokenizer
                                    _base.genesis_loader = GenesisDataLoader(
                                        NEURO_NODE.node_id, get_neuro_tokenizer(),
                                        max_storage_mb=max_storage_mb)
                                except Exception as e:
                                    logger.warning(f"[GENESIS] Init failed: {e}")
                            QUORUM_TRAINER = QuorumTrainer(
                                quorum=CURRENT_QUORUM,
                                node_id=NEURO_NODE.node_id,
                                model=NEURO_NODE.model,
                                optimizer=NEURO_NODE.optimizer,
                                genesis_loader=getattr(_base, 'genesis_loader', None),
                                dht_protocol=DHT_PROTOCOL,
                            )
                            # Wire inbound buffer to gRPC handler
                            from neuroshard.grpc_server import GRPC_SERVICER as _gs2
                            if _gs2 and QUORUM_TRAINER.inbound_buffer:
                                _gs2.quorum_inbound_buffer = QUORUM_TRAINER.inbound_buffer
                                _gs2._active_quorum_id = CURRENT_QUORUM.quorum_id
                            QUORUM_TRAINER.start()
                            logger.info("[QUORUM] Started QuorumTrainer (PIPELINE mode)")
                        STATE["training_mode"] = "quorum"
                        STATE["training_status"] = "active"
                        
                    elif new_mode == ContributionMode.ASYNC:
                        # Stop quorum trainer if running
                        if QUORUM_TRAINER and QUORUM_TRAINER.running:
                            logger.info("[QUORUM] Stopping QuorumTrainer (switching to ASYNC)")
                            QUORUM_TRAINER.stop()
                        
                        # Ensure genesis loader is initialized
                        _base = getattr(NEURO_NODE, 'base_node', NEURO_NODE)
                        if not hasattr(_base, 'genesis_loader') or _base.genesis_loader is None:
                            try:
                                from neuroshard.core.training.distributed import GenesisDataLoader
                                from neuroshard.core.model.tokenizer import get_neuro_tokenizer
                                _base.genesis_loader = GenesisDataLoader(
                                    NEURO_NODE.node_id, get_neuro_tokenizer(),
                                    max_storage_mb=max_storage_mb)
                            except Exception as e:
                                logger.warning(f"[GENESIS] Init failed: {e}")
                        
                        # Start async trainer if not running
                        if not ASYNC_TRAINER:
                            logger.info("[ASYNC] Creating AsyncTrainer...")
                            ASYNC_TRAINER = AsyncTrainer(
                                node_id=NEURO_NODE.node_id,
                                model=NEURO_NODE.model,
                                optimizer=NEURO_NODE.optimizer,
                                genesis_loader=getattr(_base, 'genesis_loader', None),
                                dht_protocol=DHT_PROTOCOL,
                            )
                            
                            # Set up callback for P2P to get fresh training stats
                            if P2P and not P2P.get_fresh_training_stats:
                                def _get_fresh_stats():
                                    if ASYNC_TRAINER:
                                        return ASYNC_TRAINER.get_stats()
                                    return None
                                P2P.get_fresh_training_stats = _get_fresh_stats
                            
                            # Set up callback for proof acceptance notification
                            if P2P and not P2P.on_proof_accepted:
                                def _on_proof_accepted():
                                    if ASYNC_TRAINER:
                                        ASYNC_TRAINER.reset_proof_period()
                                P2P.on_proof_accepted = _on_proof_accepted
                        
                        if not ASYNC_TRAINER.running:
                            logger.info("[ASYNC] Starting AsyncTrainer...")
                            ASYNC_TRAINER.start()
                        STATE["training_mode"] = "async"
                        STATE["training_status"] = "async_training"
                        
                    elif new_mode == ContributionMode.IDLE:
                        # Stop both trainers
                        if ASYNC_TRAINER and ASYNC_TRAINER.running:
                            ASYNC_TRAINER.stop()
                        if QUORUM_TRAINER and QUORUM_TRAINER.running:
                            QUORUM_TRAINER.stop()
                        STATE["training_mode"] = "idle"
                        STATE["training_status"] = "waiting"
                
                # Update async trainer stats if running
                if ASYNC_TRAINER and ASYNC_TRAINER.running:
                    async_stats = ASYNC_TRAINER.get_stats()
                    current_async_batches = async_stats["total_batches"]
                    STATE["async_batches"] = current_async_batches
                    STATE["async_syncs"] = async_stats["total_syncs"]
                    STATE["current_loss"] = async_stats["current_loss"]
                    
                    # Chained PoNW: Get model hashes from async trainer
                    # Note: After proof submission, p2p.py resets STATE["model_hash_start"]
                    # We need to sync this back to ASYNC_TRAINER
                    state_start = STATE.get("model_hash_start", "")
                    trainer_start = async_stats.get("model_hash_start", "")
                    
                    # If state was reset by p2p.py (to model_hash_end), sync to trainer
                    if state_start and state_start != trainer_start:
                        # p2p.py reset the start hash - update trainer
                        ASYNC_TRAINER.model_hash_start = state_start
                    elif trainer_start:
                        # Use trainer's start hash
                        STATE["model_hash_start"] = trainer_start
                    
                    if async_stats.get("model_hash_end"):
                        STATE["model_hash_end"] = async_stats["model_hash_end"]
                        STATE["model_hash"] = async_stats["model_hash_end"]
                    
                    # Add DELTA batches (not cumulative) - P2P resets training_batches after each proof
                    batch_delta = current_async_batches - last_async_batches
                    if batch_delta > 0:
                        STATE["training_batches"] = STATE.get("training_batches", 0) + batch_delta
                        last_async_batches = current_async_batches
            
            # =================================================================
            # STATE UPDATES
            # =================================================================
            # Update token/training counts from node
            current_tokens = NEURO_NODE.total_tokens_processed
            current_training = NEURO_NODE.total_training_rounds
            
            STATE["token_count"] = STATE.get("token_count", 0) + (current_tokens - last_tokens)
            STATE["total_tokens_processed"] = current_tokens
            STATE["total_training_rounds"] = current_training
            
            last_tokens = current_tokens
            last_training_rounds = current_training
            
            # Update model hash for chained PoNW
            # SKIP if ASYNC_TRAINER already provided hashes (more accurate)
            if NEURO_NODE.model and hasattr(NEURO_NODE, '_get_model_hash'):
                if not (ASYNC_TRAINER and ASYNC_TRAINER.running and STATE.get("model_hash_end")):
                    # Only compute hash if ASYNC_TRAINER didn't provide one
                    current_hash = NEURO_NODE._get_model_hash()
                    
                    # Track model_hash_start at beginning of each proof period
                    if "model_hash_start" not in STATE or not STATE.get("model_hash_start"):
                        STATE["model_hash_start"] = current_hash
                    
                    # model_hash_end is always the current hash
                    STATE["model_hash"] = current_hash
                    STATE["model_hash_end"] = current_hash
            
            # =================================================================
            # HEARTBEAT (every 30 seconds)
            # =================================================================
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                last_heartbeat = now
                
                # Layer pool heartbeat
                if NEURO_NODE.layer_pool:
                    NEURO_NODE.layer_pool.heartbeat(NEURO_NODE.node_id, NEURO_NODE.my_layer_ids)
                
                # Log training status
                if QUORUM_TRAINER and QUORUM_TRAINER.running:
                    logger.info(f"[QUORUM] Training: batches={QUORUM_TRAINER.total_batches}, "
                               f"loss={QUORUM_TRAINER.current_loss or 'N/A'}, "
                               f"sync_round={QUORUM_TRAINER.sync_round}")
                elif enable_training:
                    logger.info(f"[QUORUM] Waiting for quorum formation...")
            
            # =================================================================
            # LAYER GROWTH CHECK (every 60 seconds)
            # =================================================================
            if now - last_layer_growth_check >= LAYER_GROWTH_CHECK_INTERVAL:
                last_layer_growth_check = now
                
                try:
                    # Get network stats for growth check
                    current_layers = NEURO_NODE.layer_pool.current_num_layers if NEURO_NODE.layer_pool else 32
                    current_hidden_dim = getattr(NEURO_NODE.model.config, 'n_embd', 1024) if hasattr(NEURO_NODE, 'model') else 1024
                    
                    # Get layer coverage from layer pool
                    layer_coverage = {}
                    if NEURO_NODE.layer_pool:
                        for layer_id in range(current_layers):
                            layer_info = NEURO_NODE.layer_pool.get_layer_info(layer_id) if hasattr(NEURO_NODE.layer_pool, 'get_layer_info') else None
                            layer_coverage[layer_id] = layer_info.replica_count if layer_info else 1
                    
                    # Estimate total network memory (rough estimate based on peer count)
                    network_size = len(P2P.known_peers) + 1
                    total_network_memory_mb = network_size * available_memory_mb
                    
                    # Get steps since last growth
                    steps_since_last_growth = STATE.get("total_training_rounds", 0) - STATE.get("last_growth_step", 0)
                    
                    # Check if layer growth is needed
                    upgrade = check_layer_growth(
                        current_layers=current_layers,
                        current_hidden_dim=current_hidden_dim,
                        total_network_memory_mb=total_network_memory_mb,
                        steps_since_last_growth=steps_since_last_growth,
                        layer_coverage=layer_coverage,
                        network_size=network_size,
                    )
                    
                    # If upgrade is needed, start it
                    if upgrade and LAYER_GROWTH_MANAGER:
                        logger.info(f"[GROWTH] Layer growth triggered: {upgrade.target_layers} layers")
                        if LAYER_GROWTH_MANAGER.start_upgrade(upgrade):
                            STATE["last_growth_step"] = STATE.get("total_training_rounds", 0)
                            STATE["growth_phase"] = "announcement"
                            STATE["growth_target_layers"] = upgrade.target_layers
                    
                    # Advance growth phase if in progress
                    if LAYER_GROWTH_MANAGER and LAYER_GROWTH_MANAGER.current_upgrade:
                        new_phase = LAYER_GROWTH_MANAGER.advance_phase()
                        if new_phase:
                            logger.info(f"[GROWTH] Advanced to phase: {new_phase.value}")
                            STATE["growth_phase"] = new_phase.value
                        else:
                            STATE["growth_phase"] = LAYER_GROWTH_MANAGER.current_upgrade.phase.value if LAYER_GROWTH_MANAGER.current_upgrade else "none"
                    else:
                        STATE["growth_phase"] = "none"
                        
                except Exception as e:
                    logger.debug(f"[GROWTH] Layer growth check error: {e}")
            
            # =================================================================
            # MEMORY REPORT (every 60 seconds)
            # =================================================================
            if now - last_memory_report >= MEMORY_REPORT_INTERVAL:
                last_memory_report = now
                try:
                    import os
                    process = psutil.Process(os.getpid())
                    process_mem_mb = process.memory_info().rss / (1024 * 1024)
                    system_mem = psutil.virtual_memory()
                    
                    logger.info(f"[NODE] Memory: {process_mem_mb:.0f}MB, "
                               f"System: {system_mem.percent:.0f}%")
                    
                    # Quorum status
                    if CURRENT_QUORUM:
                        q_type = getattr(CURRENT_QUORUM, 'quorum_type', QuorumType.PIPELINE).value
                        logger.info(f"[QUORUM] Quorum: {CURRENT_QUORUM.quorum_id[:8]}... "
                                   f"({q_type}, {CURRENT_QUORUM.lifecycle.value}, "
                                   f"{len(CURRENT_QUORUM.members)} members)")
                except Exception:
                    pass
            
            # =================================================================
            # PERIODIC CHECKPOINT (every 5 minutes - crash protection)
            # =================================================================
            if now - last_checkpoint_save >= CHECKPOINT_SAVE_INTERVAL:
                last_checkpoint_save = now
                try:
                    base = getattr(NEURO_NODE, 'base_node', NEURO_NODE)
                    if hasattr(base, 'save_checkpoint') and hasattr(base, 'total_training_rounds'):
                        if base.total_training_rounds > 0:
                            base.save_checkpoint()
                except Exception as e:
                    logger.debug(f"[CHECKPOINT] Periodic save error: {e}")
            
            # =================================================================
            # HOUSEKEEPING (every 60 seconds)
            # =================================================================
            if int(now) % 60 == 0:
                # Session cleanup
                to_remove = [sid for sid, ts in SESSION_TIMESTAMPS.items() if now - ts > 300]
                for sid in to_remove:
                    del SESSION_TIMESTAMPS[sid]
                
                # Marketplace cleanup
                if P2P and P2P.ledger:
                    market = P2P.ledger.inference_market
                    stale = market.cleanup_stale_claims()
                    if stale > 0:
                        logger.info(f"[MARKET] Cleaned up {stale} stale claims")
                    market.cleanup_old_results()
                
                # Validator eligibility check
                if NEURO_NODE and NEURO_NODE.layer_pool:
                    def get_node_stake(node_id: str) -> float:
                        if node_id == NEURO_NODE.node_id:
                            return P2P.ledger.get_account_info().get("stake", 0.0)
                        return float('inf')
                    
                    demoted = NEURO_NODE.layer_pool.validate_all_validators(get_node_stake)
                    if NEURO_NODE.node_id in demoted and NEURO_NODE.model:
                        NEURO_NODE.model.disable_lm_head()
                        logger.warning("[NODE] Demoted from Validator")
                
                # Layer pool cleanup
                if NEURO_NODE.layer_pool:
                    removed = NEURO_NODE.layer_pool.cleanup_stale_assignments()
                    if removed:
                        logger.info(f"[LAYER_POOL] Cleaned up {len(removed)} stale assignments")
            
            # =================================================================
            # TOKENIZER REFRESH (every 10 minutes)
            # =================================================================
            if int(now) % 600 == 0:
                try:
                    if hasattr(NEURO_NODE, '_load_learned_tokenizer'):
                        old_vocab = NEURO_NODE.tokenizer.current_vocab_size if NEURO_NODE.tokenizer else 0
                        NEURO_NODE._load_learned_tokenizer()
                        new_vocab = NEURO_NODE.tokenizer.current_vocab_size if NEURO_NODE.tokenizer else 0
                        if new_vocab > old_vocab:
                            logger.info(f"[TOKENIZER] Vocab: {old_vocab:,} → {new_vocab:,}")
                except Exception:
                    pass
            
            # Sleep - training happens in QuorumTrainer thread
            time.sleep(1)
    
    threading.Thread(target=background_tasks, daemon=True).start()
    
    # DRIVER WORKER LOOP: Poll marketplace AND process requests
    def driver_worker_loop():
        """
        PRODUCTION-READY Driver Worker Loop
        
        1. Polls marketplace for pending requests
        2. Claims requests assigned to this driver
        3. Waits for encrypted prompt from user
        4. Processes inference through distributed pipeline
        5. Submits PoNW proof for rewards
        """
        import time
        
        # Check if this node is an initiator (has embedding layer)
        is_initiator = NEURO_NODE and NEURO_NODE.model.has_embedding
        
        if not is_initiator:
            logger.info("[INITIATOR] Not an initiator node - skipping marketplace worker loop")
            return
        
        logger.info("[INITIATOR] Starting PRODUCTION marketplace worker loop...")
        logger.info(f"[INITIATOR] Will poll for requests assigned to: {NEURO_NODE.node_id[:16]}...")
        
        # Import encrypted prompt handling
        from neuroshard.core.network.encrypted_channel import PromptEncryption, PromptQueue
        
        prompt_queue = PromptQueue()
        
        # Store in node for API access
        NEURO_NODE.prompt_queue = prompt_queue
        
        last_claim_attempt = 0
        processing_requests = {}  # request_id -> asyncio.Task
        
        def process_request(request_id: str):
            """Process a single inference request using existing distributed inference."""
            try:
                # Get marketplace request for parameters
                market = P2P.ledger.inference_market
                
                market_request = market.get_request(request_id)
                if not market_request:
                    logger.warning(f"[INITIATOR] ✗ Request {request_id[:8]}... not found in marketplace")
                    return
                
                # Get encrypted prompt
                encrypted_prompt = prompt_queue.get_prompt(request_id)
                
                if not encrypted_prompt:
                    logger.warning(f"[INITIATOR] ✗ No prompt found for {request_id[:8]}...")
                    return
                
                # Decrypt prompt
                try:
                    prompt_text = PromptEncryption.decrypt_prompt(
                        encrypted_prompt.encrypted_data,
                        request_id
                    )
                    logger.info(f"[INITIATOR] ✓ Decrypted prompt: '{prompt_text[:50]}...'")
                except Exception as e:
                    logger.error(f"[INITIATOR] ✗ Failed to decrypt prompt: {e}")
                    return
                
                # Process using EXISTING distributed inference
                try:
                    output = NEURO_NODE.generate(
                        prompt=prompt_text,
                        max_tokens=market_request.tokens_requested,
                        temperature=0.8
                    )
                    
                    logger.info(f"[INITIATOR] ✓ Generated: '{output[:100]}...'")
                    logger.info(f"[INITIATOR] ✓ Request {request_id[:8]}... completed")
                    processing_requests[request_id] = "completed"
                    
                    # Store result in marketplace
                    market.store_result(request_id, output)
                    
                except Exception as e:
                    logger.error(f"[INITIATOR] ✗ Generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    processing_requests[request_id] = "failed"
                    
            except Exception as e:
                logger.error(f"[INITIATOR] ✗ Error processing {request_id[:8]}...: {e}")
                import traceback
                logger.error(traceback.format_exc())
                processing_requests[request_id] = "failed"
        
        while not _SHUTDOWN_REQUESTED.is_set():
            now = time.time()
            
            # STEP 1: Poll marketplace for new requests (every 5 seconds)
            if now - last_claim_attempt >= 5:
                try:
                    market = P2P.ledger.inference_market
                    
                    # Try to claim a request
                    request = market.claim_request(NEURO_NODE.node_id)
                    
                    if request:
                        logger.info(f"[INITIATOR] ✓ Claimed request {request.request_id[:8]}... "
                              f"({request.tokens_requested} tokens @ {request.locked_price:.6f} NEURO/1M)")
                        
                        # Start pipeline session
                        market.start_pipeline_session(
                            request_id=request.request_id,
                            session_id=request.pipeline_session_id,
                            driver_node_id=NEURO_NODE.node_id
                        )
                        
                        # Check if we already have the prompt
                        if prompt_queue.has_prompt(request.request_id):
                            logger.info(f"[INITIATOR] ✓ Prompt already received, processing immediately")
                            # Process immediately
                            process_request(request.request_id)
                        else:
                            logger.info(f"[INITIATOR] Waiting for encrypted prompt from user...")
                            logger.info(f"[INITIATOR] User should POST to /api/driver/prompt/{request.request_id[:8]}...")
                            processing_requests[request.request_id] = None  # Mark as waiting
                                    
                except Exception as e:
                    if "not found" not in str(e).lower():
                        logger.error(f"[INITIATOR] Marketplace poll error: {e}")
                
                last_claim_attempt = now
            
            # STEP 2: Check for prompts that arrived for waiting requests
            for request_id in list(processing_requests.keys()):
                if processing_requests[request_id] is None:  # Waiting for prompt
                    if prompt_queue.has_prompt(request_id):
                        logger.info(f"[INITIATOR] ✓ Prompt received for {request_id[:8]}..., starting processing")
                        # Process (uses existing distributed inference)
                        process_request(request_id)
                        processing_requests[request_id] = "processing"  # Mark as processing
            
            # STEP 3: Cleanup finished requests
            for request_id in list(processing_requests.keys()):
                if processing_requests[request_id] == "completed":
                    del processing_requests[request_id]
            
            # STEP 4: Cleanup old prompts
            prompt_queue.cleanup_old_prompts()
            
            time.sleep(1)  # Fast loop for responsiveness
    
    # Start driver worker loop if this is a driver node
    if NEURO_NODE and NEURO_NODE.model.has_embedding:
        threading.Thread(target=driver_worker_loop, daemon=True).start()
    
    # Custom log config: disable access logs and customize startup messages
    # Handle Windows GUI mode where stdout may be None
    if sys.stdout is not None and hasattr(sys.stdout, 'write'):
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "[NODE] %(message)s"},
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                # Suppress uvicorn's default startup messages (including "Press CTRL+C")
                "uvicorn": {"handlers": ["default"], "level": "WARNING", "propagate": False},
                "uvicorn.error": {"handlers": ["default"], "level": "WARNING", "propagate": False},
                "uvicorn.access": {"handlers": [], "level": "CRITICAL", "propagate": False},
            },
        }
    else:
        # Fallback to file logging when stdout is unavailable (Windows frozen GUI)
        log_dir = os.path.join(os.path.expanduser("~"), ".neuroshard")
        log_file = os.path.join(log_dir, "uvicorn.log")
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": "[NODE] %(message)s"},
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file,
                    "maxBytes": 5*1024*1024,
                    "backupCount": 2,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                # Suppress uvicorn's default startup messages
                "uvicorn": {"handlers": ["default"], "level": "WARNING", "propagate": False},
                "uvicorn.error": {"handlers": ["default"], "level": "WARNING", "propagate": False},
                "uvicorn.access": {"handlers": [], "level": "CRITICAL", "propagate": False},
            },
        }
    
    # Use Server object so we can stop it from outside (GUI shutdown)
    global _UVICORN_SERVER
    config = uvicorn.Config(node_app, host="0.0.0.0", port=port, log_config=log_config)
    _UVICORN_SERVER = uvicorn.Server(config)
    
    # Print our own clean startup message (without "Press CTRL+C")
    logger.info(f"[NODE] HTTP server started on port {port}")
    
    try:
        _UVICORN_SERVER.run()
    except KeyboardInterrupt:
        # On Windows, Ctrl+C raises KeyboardInterrupt instead of triggering
        # the signal handler. Ensure checkpoint is saved before exit.
        logger.info("[NODE] KeyboardInterrupt received, saving checkpoint and shutting down...")
        request_shutdown()


def main():
    import signal
    import atexit
    
    # Register signal handlers for graceful shutdown
    def _signal_handler(signum, frame):
        logger.info(f"[NODE] Received signal {signum}, initiating graceful shutdown...")
        request_shutdown()
        sys.exit(0)
    
    # Handle Ctrl+C (SIGINT) and SIGTERM
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    # Also register atexit handler as backup
    atexit.register(lambda: request_shutdown() if NEURO_NODE else None)
    
    parser = argparse.ArgumentParser(description="NeuroShard Node Runner - Truly Decentralized LLM")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tracker", type=str, default="https://neuroshard.com/api/tracker")
    parser.add_argument("--token", type=str, default=None, 
                       help="Node Token OR 12-word mnemonic phrase for wallet access")
    parser.add_argument("--announce-ip", type=str, default=None, help="Force IP address to announce")
    parser.add_argument("--announce-port", type=int, default=None, help="Force port to announce")
    parser.add_argument("--no-training", action="store_true", help="Disable training (inference only)")
    parser.add_argument("--observer", action="store_true", 
                       help="Observer mode: sync ledger from network but don't generate proofs (for explorer)")
    parser.add_argument("--memory", type=int, default=None, 
                       help="Override detected memory (MB) - for testing")
    parser.add_argument("--max-storage", type=int, default=100,
                       help="Max disk space for training data (MB)")
    parser.add_argument("--cpu-threads", type=int, default=None,
                       help="Max CPU threads to use")
    parser.add_argument("--diloco-steps", type=int, default=500,
                       help="DiLoCo inner steps before gradient sync (default: 500)")
    
    args = parser.parse_args()
    
    # Handle mnemonic input: If token is 12 words, convert to token
    node_token = args.token
    if node_token:
        words = node_token.strip().split()
        if len(words) == 12:
            # It's a BIP39 mnemonic - derive token from it
            try:
                from mnemonic import Mnemonic
                mnemo = Mnemonic("english")
                if mnemo.check(node_token):
                    # Convert mnemonic to deterministic token
                    seed = mnemo.to_seed(node_token, passphrase="")
                    node_token = seed[:32].hex()  # Use first 32 bytes as token
                    logger.info("✅ Wallet recovered from mnemonic")
                else:
                    logger.warning("⚠️  Invalid mnemonic phrase - treating as raw token")
            except ImportError:
                logger.warning("⚠️  'mnemonic' package not installed - treating as raw token")
            except Exception as e:
                logger.warning(f"⚠️  Mnemonic error: {e} - treating as raw token")
    
    run_node(
        port=args.port,
        tracker=args.tracker,
        node_token=node_token,
        announce_ip=args.announce_ip,
        announce_port=args.announce_port,
        enable_training=not args.no_training,
        observer_mode=args.observer,
        available_memory_mb=args.memory,
        max_storage_mb=args.max_storage,
        max_cpu_threads=args.cpu_threads,
        diloco_inner_steps=args.diloco_steps,
    )


if __name__ == "__main__":
    main()
