"""
Distributed Training System for NeuroLLM

Implements decentralized training where:
1. Nodes contribute compute for forward/backward passes
2. Gradients are aggregated via gossip protocol
3. Training rewards are distributed in NEURO tokens
4. Model checkpoints are shared across the network

Key Components:
- GradientContribution: Data class for gradient contributions from nodes
- GradientCompressor: Compresses gradients for efficient network transmission
- GenesisDataLoader: Loads training data from the verified Genesis Dataset

Note: Reward calculation is handled by economics.ledger via PoNW verification.
"""

import torch
import threading
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging
import json
import io
import zlib
import base64
import os
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict


logger = logging.getLogger(__name__)


@dataclass
class GradientContribution:
    """A gradient contribution from a node."""
    node_id: str
    round_id: int
    layer_gradients: Dict[str, bytes]  # layer_name -> compressed gradient
    batch_size: int
    loss: float
    timestamp: float
    signature: str  # Proof of work


class GradientCompressor:
    """
    Compresses gradients for efficient network transmission.
    
    Uses a combination of:
    1. Top-K sparsification (keep only largest gradients)
    2. Quantization (reduce precision)
    3. zlib compression
    """
    
    def __init__(self, top_k_ratio: float = 0.1, bits: int = 8):
        self.top_k_ratio = top_k_ratio
        self.bits = bits
    
    def compress(self, gradient: torch.Tensor) -> bytes:
        """Compress a gradient tensor."""
        # CRITICAL: Move to CPU first for MPS/CUDA compatibility
        gradient = gradient.detach().cpu()
        
        # Flatten
        flat = gradient.flatten()
        
        # Top-K sparsification
        k = max(1, int(len(flat) * self.top_k_ratio))
        values, indices = torch.topk(flat.abs(), k)
        
        # Get actual values (with signs)
        sparse_values = flat[indices]
        
        # Quantize to specified bits
        max_val = sparse_values.abs().max()
        if max_val > 0:
            scale = (2 ** (self.bits - 1) - 1) / max_val
            quantized = (sparse_values * scale).round().to(torch.int8)
        else:
            quantized = torch.zeros(k, dtype=torch.int8)
            scale = 1.0
        
        # Pack into bytes (tensors already on CPU)
        data = {
            "shape": list(gradient.shape),
            "k": k,
            "indices": base64.b64encode(indices.numpy().tobytes()).decode('ascii'),
            "values": base64.b64encode(quantized.numpy().tobytes()).decode('ascii'),
            "scale": float(scale),
            "dtype": str(gradient.dtype),
        }
        
        # Serialize and compress
        json_data = json.dumps(data).encode()
        return zlib.compress(json_data)
    
    def decompress(self, data: bytes, device: str = "cpu") -> torch.Tensor:
        """Decompress a gradient tensor."""
        # Decompress and deserialize
        json_data = zlib.decompress(data)
        packed = json.loads(json_data)
        
        # Unpack
        shape = packed["shape"]
        k = packed["k"]
        indices = torch.frombuffer(
            bytearray(base64.b64decode(packed["indices"])), 
            dtype=torch.int64
        ).clone().to(device)
        values = torch.frombuffer(
            bytearray(base64.b64decode(packed["values"])), 
            dtype=torch.int8
        ).float().clone().to(device)
        scale = packed["scale"]
        
        # Dequantize
        values = values / scale
        
        # Reconstruct sparse tensor
        flat = torch.zeros(torch.prod(torch.tensor(shape)), device=device)
        flat[indices] = values
        
        return flat.view(*shape)


class GenesisDataLoader:
    """
    Loads training data from the verified Genesis Dataset.
    
    Features:
    - Dynamic shard count (reads from manifest)
    - User-configurable storage limit (max_storage_mb)
    - Shard rotation (cycles through dataset over time)
    - Multi-shard support (downloads multiple shards up to storage limit)
    - ASYNC PREFETCHING: Pre-downloads next shard while training on current
    
    Active only for nodes holding Layer 0 (Embedding Layer).
    
    Data Source: CloudFront CDN (backed by S3)
    """
    # CloudFront CDN URL - single source of truth (cached, DDoS protected)
    GENESIS_CDN_URL = "https://dwquwt9gkkeil.cloudfront.net"
    # Size per shard in MB (must match populate_genesis_s3.py)
    SHARD_SIZE_MB = 10
    
    def __init__(
        self, 
        node_id: str, 
        tokenizer, 
        cache_dir: str = None,  # Default to ~/.neuroshard/data_cache
        max_storage_mb: float = 100.0,  # User-configurable limit
        manifest_version: int = 1
    ):
        self.node_id = node_id
        self.tokenizer = tokenizer
        
        # Default cache_dir to ~/.neuroshard/data_cache for consistent storage
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".neuroshard", "data_cache")
        self.cache_dir = cache_dir
        self.max_storage_mb = max_storage_mb
        self.manifest_version = manifest_version
        
        # CloudFront CDN manifest URL - single source of truth
        self.manifest_url = f"{self.GENESIS_CDN_URL}/manifest.json"
        
        # Manifest data (cached, refreshed periodically)
        self.manifest = None
        self.total_shards = 0
        self.manifest_last_fetch = 0
        self.MANIFEST_REFRESH_INTERVAL = 600  # Refresh manifest every 10 minutes (auto-update tokenizer)
        
        # Shard management
        self.max_shards = max(1, int(max_storage_mb / self.SHARD_SIZE_MB))
        self.assigned_shard_ids = []  # List of shard IDs this node is responsible for
        self.loaded_shards = {}  # shard_id -> tensor data
        self.current_shard_idx = 0  # Index into assigned_shard_ids for rotation
        self.shard_rotation_count = 0  # How many times we've rotated through
        self.loading_shards = set()  # Track shards currently being downloaded
        self._shard_lock = threading.Lock()  # Lock for shard loading
        self._download_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="shard-download")
        
        # ASYNC PREFETCHING: Keep next shard(s) ready in background
        self._prefetch_in_progress = set()  # Shard IDs being prefetched
        self._prefetch_ready = {}  # shard_id -> tensor data (ready to use)
        self._prefetch_ahead = 2  # Number of shards to prefetch ahead (was 1)
        
        # LOSS PLATEAU DETECTION: Track loss to detect when to rotate shards early
        self._loss_history = []  # Recent loss values
        self._loss_history_max = 50  # Number of loss values to track
        self._loss_plateau_threshold = 0.02  # If loss variance < this, plateau detected
        self._min_steps_per_shard = 100  # Minimum steps before considering early rotation
        self._steps_on_current_shard = 0  # Steps taken on current shard
        
        # Initialize Data Swarm for P2P downloading
        self.swarm = None 
        
        self.current_dataset = None
        self.dataset_iterator = 0
        
        # Fetch manifest and assign initial shards
        self._refresh_manifest()
        self._assign_shards()
        
        # Try to load learned tokenizer from CDN (for proper vocab)
        self._load_learned_tokenizer()
        
        # THUNDERING HERD PREVENTION: Add random jitter before first download
        # This spreads load across the CDN when many nodes start simultaneously
        # Jitter: 0-5 seconds based on node_id hash
        import random
        jitter_seed = int(hashlib.sha256(self.node_id.encode()).hexdigest()[:8], 16)
        jitter_seconds = (jitter_seed % 5000) / 1000.0  # 0-5 seconds
        
        def delayed_prefetch():
            time.sleep(jitter_seconds)
            self._start_prefetch_next()
        
        # Start prefetching first shard with jitter (non-blocking)
        threading.Thread(target=delayed_prefetch, daemon=True).start()
        
        logger.info(f"GenesisDataLoader initialized: "
                   f"total_shards={self.total_shards}, "
                   f"max_storage={max_storage_mb}MB ({self.max_shards} shards), "
                   f"assigned={self.assigned_shard_ids[:5]}{'...' if len(self.assigned_shard_ids) > 5 else ''}, "
                   f"prefetch_jitter={jitter_seconds:.2f}s")

    def _load_learned_tokenizer(self):
        """
        Download and load the learned tokenizer from Genesis CDN.
        Checks if the network has learned more tokens and updates locally.
        """
        try:
            tokenizer_url = f"{self.GENESIS_CDN_URL}/tokenizer.json"
            tokenizer_cache_path = os.path.join(self.cache_dir, "tokenizer.json")
            
            # Always try to fetch latest from CDN
            try:
                logger.debug(f"[GENESIS] Checking for tokenizer updates from {tokenizer_url}...")
                resp = requests.get(tokenizer_url, timeout=10)
                
                if resp.status_code == 200:
                    remote_tokenizer_data = resp.json()
                    remote_vocab_size = remote_tokenizer_data.get("next_merge_id", 0)
                    
                    # Always cache the downloaded tokenizer (for offline use)
                    os.makedirs(self.cache_dir, exist_ok=True)
                    with open(tokenizer_cache_path, 'w') as f:
                        f.write(resp.text)
                    
                    # Update our tokenizer if remote has more tokens
                    if remote_vocab_size > self.tokenizer.next_merge_id:
                        logger.info(f"[GENESIS] Found improved tokenizer! ({self.tokenizer.next_merge_id} -> {remote_vocab_size} tokens)")
                        
                        from neuroshard.core.model.tokenizer import NeuroTokenizer
                        learned_tokenizer = NeuroTokenizer.load(tokenizer_cache_path)
                        
                        self.tokenizer.merges = learned_tokenizer.merges
                        self.tokenizer.merge_to_tokens = learned_tokenizer.merge_to_tokens
                        self.tokenizer.next_merge_id = learned_tokenizer.next_merge_id
                        # Invalidate word cache — stale tokenizations from old merges
                        if hasattr(self.tokenizer, '_invalidate_cache'):
                            self.tokenizer._invalidate_cache()
                        
                        logger.info(f"[GENESIS] Tokenizer updated: {self.tokenizer.next_merge_id} tokens")
                    else:
                        logger.info(f"[GENESIS] Tokenizer cached: {remote_vocab_size} tokens (current: {self.tokenizer.next_merge_id})")
                    return
            except Exception as e:
                logger.debug(f"[GENESIS] Failed to check for tokenizer updates: {e}")
            
            # Fallback to cached version if download failed
            if os.path.exists(tokenizer_cache_path) and self.tokenizer.next_merge_id <= 266:
                logger.info(f"[GENESIS] Loading cached tokenizer from {tokenizer_cache_path}")
                try:
                    from neuroshard.core.model.tokenizer import NeuroTokenizer
                    learned_tokenizer = NeuroTokenizer.load(tokenizer_cache_path)
                    
                    if learned_tokenizer.next_merge_id > self.tokenizer.next_merge_id:
                        self.tokenizer.merges = learned_tokenizer.merges
                        self.tokenizer.merge_to_tokens = learned_tokenizer.merge_to_tokens
                        self.tokenizer.next_merge_id = learned_tokenizer.next_merge_id
                        # Invalidate word cache after merge update
                        if hasattr(self.tokenizer, '_invalidate_cache'):
                            self.tokenizer._invalidate_cache()
                        logger.info(f"[GENESIS] Loaded cached tokenizer: {self.tokenizer.next_merge_id} tokens")
                except Exception as e:
                    logger.warning(f"[GENESIS] Failed to load cached tokenizer: {e}")

        except Exception as e:
            logger.warning(f"[GENESIS] Error managing tokenizer: {e}")

    def _refresh_manifest_sync(self):
        """Synchronous manifest fetch (runs in background thread)."""
        try:
            logger.info(f"[GENESIS] Fetching manifest from {self.manifest_url}...")
            resp = requests.get(self.manifest_url, timeout=15)
            if resp.status_code == 200:
                manifest_data = resp.json()
                total_shards = manifest_data.get("total_shards", 0)
                
                # Update state atomically
                with self._shard_lock:
                    self.manifest = manifest_data
                    self.total_shards = total_shards
                    self.manifest_last_fetch = time.time()
                
                logger.info(f"[GENESIS] Manifest loaded: {self.total_shards} shards available")
                
                # Also check if tokenizer has improved (in background)
                self._load_learned_tokenizer()
            else:
                logger.error(f"[GENESIS] Failed to fetch manifest: HTTP {resp.status_code}")
                logger.error(f"[GENESIS] Response: {resp.text[:200]}")
        except Exception as e:
            logger.error(f"[GENESIS] Failed to fetch manifest from {self.manifest_url}: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[GENESIS] Traceback: {traceback.format_exc()}")
    
    def _refresh_manifest(self):
        """Fetch latest manifest from S3 (non-blocking after first load)."""
        now = time.time()
        
        # First time initialization - must be synchronous
        if self.manifest is None:
            self._refresh_manifest_sync()
            if self.total_shards == 0:
                raise RuntimeError(f"Cannot fetch manifest from {self.manifest_url}. Check S3 bucket.")
            return
        
        # Subsequent refreshes - use cached if recent
        if (now - self.manifest_last_fetch) < self.MANIFEST_REFRESH_INTERVAL:
            return  # Use cached manifest
        
        # Refresh in background (non-blocking)
        self._download_executor.submit(self._refresh_manifest_sync)

    def _assign_shards(self):
        """
        Assign shards to this node based on:
        1. Manifest's explicit shard list (only valid shards that exist)
        2. Node's deterministic hash (ensures different nodes get different shards)
        3. User's storage limit (max_shards)
        4. Rotation offset (allows cycling through entire dataset over time)
        """
        # Get list of valid shard IDs from manifest
        valid_shard_ids = self._get_valid_shard_ids()
        
        if not valid_shard_ids:
            logger.warning("[GENESIS] No valid shards in manifest, using shard 0")
            self.assigned_shard_ids = [0]
            return
        
        # Sort for deterministic assignment across nodes
        valid_shard_ids = sorted(valid_shard_ids)
        num_valid = len(valid_shard_ids)
        
        # Base offset from node ID (deterministic)
        node_hash = int(hashlib.sha256(self.node_id.encode()).hexdigest(), 16)
        base_offset = node_hash % num_valid
        
        # Rotation offset (changes over time to cover more data)
        rotation_offset = (self.shard_rotation_count * self.max_shards) % num_valid
        
        # Assign shards from VALID list only (no gaps!)
        self.assigned_shard_ids = []
        for i in range(min(self.max_shards, num_valid)):
            idx = (base_offset + rotation_offset + i) % num_valid
            self.assigned_shard_ids.append(valid_shard_ids[idx])
        
        logger.info(f"Assigned {len(self.assigned_shard_ids)} shards from {num_valid} valid: "
                   f"{self.assigned_shard_ids[:5]}{'...' if len(self.assigned_shard_ids) > 5 else ''}")
    
    def _get_valid_shard_ids(self) -> List[int]:
        """
        Get list of valid shard IDs from manifest.
        
        The manifest contains a 'shards' array listing every shard that actually exists.
        This prevents 403 errors from trying to download non-existent shards.
        """
        if not self.manifest:
            return []
        
        # Check for explicit shard list in manifest
        shards_list = self.manifest.get("shards", [])
        if shards_list:
            return [s["shard_id"] for s in shards_list if "shard_id" in s]
        
        # Fallback: assume contiguous (legacy manifests)
        return list(range(self.total_shards))

    def rotate_shards(self):
        """
        Rotate to next set of shards.
        Call this periodically to train on different parts of the dataset.
        """
        # Clear old loaded shards to free memory
        old_shards = list(self.loaded_shards.keys())
        self.loaded_shards.clear()
        self.current_dataset = None
        self.dataset_iterator = 0
        self.current_shard_idx = 0  # Reset to start of new assignment
        
        # Increment rotation counter
        self.shard_rotation_count += 1
        
        # Refresh manifest (in case new shards were added)
        self._refresh_manifest()
        
        # Reassign shards with new rotation offset
        self._assign_shards()
        
        # Clean up old shard files from disk
        self._cleanup_old_shards(old_shards)
        
        logger.info(f"Rotated to new shards (rotation #{self.shard_rotation_count})")

    def _cleanup_old_shards(self, old_shard_ids: list):
        """Remove old shard files from disk to stay within storage limit."""
        for shard_id in old_shard_ids:
            if shard_id not in self.assigned_shard_ids:
                shard_path = os.path.join(self.cache_dir, f"genesis_shard_{shard_id}.pt")
                try:
                    if os.path.exists(shard_path):
                        os.remove(shard_path)
                        logger.debug(f"Cleaned up old shard: {shard_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup shard {shard_id}: {e}")

    def set_swarm(self, swarm):
        """Set the DataSwarm instance."""
        self.swarm = swarm
    
    def record_loss(self, loss: float):
        """
        Record a training loss for plateau detection.
        
        Call this from the training loop to enable adaptive shard rotation.
        When loss plateaus, the loader will rotate to fresh data.
        """
        self._loss_history.append(loss)
        if len(self._loss_history) > self._loss_history_max:
            self._loss_history.pop(0)
        self._steps_on_current_shard += 1
    
    def _should_rotate_early(self) -> bool:
        """
        Check if we should rotate to a new shard early due to loss plateau.
        
        Conditions for early rotation:
        1. Have enough loss samples (at least 20)
        2. Minimum steps on current shard (100) 
        3. Loss has plateaued (low variance)
        4. Loss is low enough that we're not still actively learning
        """
        if len(self._loss_history) < 20:
            return False
        
        if self._steps_on_current_shard < self._min_steps_per_shard:
            return False
        
        # Calculate loss statistics
        recent_losses = self._loss_history[-20:]
        avg_loss = sum(recent_losses) / len(recent_losses)
        variance = sum((l - avg_loss) ** 2 for l in recent_losses) / len(recent_losses)
        
        # Also check if loss is decreasing (don't rotate if still improving)
        if len(self._loss_history) >= 40:
            older_avg = sum(self._loss_history[-40:-20]) / 20
            improvement = older_avg - avg_loss
            
            # Still improving significantly - don't rotate
            if improvement > 0.005:
                return False
        
        # Plateau detected: low variance AND low absolute loss
        if variance < self._loss_plateau_threshold and avg_loss < 0.05:
            logger.info(f"[GENESIS] Loss plateau detected: avg={avg_loss:.4f}, variance={variance:.6f}")
            logger.info(f"[GENESIS] Rotating to fresh data for continued learning")
            return True
        
        return False
    
    def force_shard_rotation(self, reason: str = "manual"):
        """
        Force rotation to a new shard.
        
        Call this when you want to move to fresh data (e.g., loss plateau).
        """
        logger.info(f"[GENESIS] Forcing shard rotation: {reason}")
        self._loss_history.clear()
        self._steps_on_current_shard = 0
        
        # Move to next shard
        self.current_shard_idx += 1
        
        if self.current_shard_idx >= len(self.assigned_shard_ids):
            # We've gone through all assigned shards - rotate to new set
            logger.info(f"[GENESIS] Exhausted all {len(self.assigned_shard_ids)} assigned shards. Getting new set...")
            self.rotate_shards()
        
        # Reset dataset iterator to start fresh
        self.current_dataset = None
        self.dataset_iterator = 0
        
        # Start prefetching the new shard
        self._start_prefetch_next()
    
    def _start_prefetch_next(self):
        """Start prefetching the next shard(s) in background."""
        if not self.assigned_shard_ids:
            return
        
        # Prefetch current and multiple next shards for faster data access
        shards_to_prefetch = []
        for offset in range(self._prefetch_ahead + 1):  # Current + prefetch_ahead (default: 0, 1, 2)
            idx = (self.current_shard_idx + offset) % len(self.assigned_shard_ids)
            shard_id = self.assigned_shard_ids[idx]
            
            with self._shard_lock:
                # Skip if already loaded, prefetching, or ready
                if (shard_id in self.loaded_shards or 
                    shard_id in self._prefetch_in_progress or
                    shard_id in self._prefetch_ready or
                    shard_id in self.loading_shards):
                    continue
                
                # Limit total prefetch in progress to avoid overwhelming the system
                if len(self._prefetch_in_progress) >= 3:
                    break
                
                shards_to_prefetch.append(shard_id)
                self._prefetch_in_progress.add(shard_id)
        
        # Start downloads in background
        for shard_id in shards_to_prefetch:
            target_url = self.get_shard_url(shard_id)
            logger.debug(f"Prefetching shard {shard_id} in background...")
            self._download_executor.submit(self._prefetch_shard_sync, shard_id, target_url)
    
    def _prefetch_shard_sync(self, shard_id: int, target_url: str):
        """Synchronous shard prefetch (runs in background thread)."""
        try:
            logger.info(f"[GENESIS] Downloading shard {shard_id}...")
            # Download the Shard
            shard_path = None
            
            if self.swarm:
                try:
                    shard_path = self.swarm.download_shard(shard_id, manifest_url=target_url)
                    logger.info(f"[GENESIS] Swarm download succeeded for shard {shard_id}")
                except Exception as e:
                    logger.warning(f"[GENESIS] Swarm prefetch failed: {e}")
            
            if not shard_path:
                logger.info(f"[GENESIS] Using HTTP fallback for shard {shard_id}")
                shard_path = self._http_fallback_download(shard_id, target_url)
                logger.info(f"[GENESIS] HTTP download completed for shard {shard_id}")
            
            # Load tensor into prefetch buffer
            tensor_data = torch.load(shard_path, weights_only=True)
            
            with self._shard_lock:
                # DYNAMIC MEMORY LIMIT: Based on user's max_storage_mb setting
                # Each shard is ~10MB compressed on disk, ~100-200MB uncompressed in RAM
                # Calculate max shards we can keep in memory
                shard_size_mb = 150  # Conservative estimate per shard in RAM
                max_cached_shards = max(3, int(self.max_storage_mb / shard_size_mb))
                
                total_loaded = len(self.loaded_shards) + len(self._prefetch_ready)
                if total_loaded >= max_cached_shards:
                    # Clear oldest loaded shard (not the current one)
                    current_shard = self.assigned_shard_ids[self.current_shard_idx % len(self.assigned_shard_ids)] if self.assigned_shard_ids else None
                    for old_shard_id in list(self.loaded_shards.keys()):
                        if old_shard_id != current_shard:
                            del self.loaded_shards[old_shard_id]
                            logger.debug(f"Evicted shard {old_shard_id} from cache (limit: {max_cached_shards} shards)")
                            break
                
                self._prefetch_ready[shard_id] = tensor_data
                self._prefetch_in_progress.discard(shard_id)
            
            logger.info(f"[GENESIS] Shard {shard_id} ready: {len(tensor_data):,} tokens")
            
        except Exception as e:
            logger.error(f"[GENESIS] Download FAILED for shard {shard_id}: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[GENESIS] Traceback: {traceback.format_exc()}")
            with self._shard_lock:
                self._prefetch_in_progress.discard(shard_id)
    
    def is_data_ready(self) -> bool:
        """Check if data is ready for training (non-blocking check)."""
        # Try to acquire lock with timeout to prevent blocking training loop
        acquired = self._shard_lock.acquire(timeout=0.5)
        if not acquired:
            # Lock held by download thread - assume data might be ready soon
            logger.debug("[GENESIS] Lock contention in is_data_ready - skipping check")
            return False
        
        try:
            # Data ready if we have current dataset OR prefetched shard is ready
            if self.current_dataset is not None and len(self.current_dataset) > 0:
                return True
            
            # Check if ANY assigned shard is ready (not just current)
            # This handles the case where prefetch completes before is_data_ready is called
            if self._prefetch_ready:
                # A prefetched shard is ready - we can use it
                return True
            
            # Also check loaded_shards
            if self.loaded_shards:
                return True
            
            # Check if current shard is specifically ready
            if self.assigned_shard_ids:
                shard_id = self.assigned_shard_ids[self.current_shard_idx % len(self.assigned_shard_ids)]
                if shard_id in self._prefetch_ready:
                    return True
                if shard_id in self.loaded_shards:
                    return True
            
            return False
        finally:
            self._shard_lock.release()

    def get_shard_url(self, shard_id: int) -> str:
        """Get download URL for a specific shard (always use CDN)."""
        # Always use CDN URL regardless of what manifest says
        # This ensures we go through CloudFront for caching/security
        return f"{self.GENESIS_CDN_URL}/shard_{shard_id}.pt"

    def _load_shard_sync(self, shard_id: int, target_url: str):
        """Synchronous shard loading (runs in background thread)."""
        # Download the Shard (Swarm or HTTP)
        shard_path = None
        
        if self.swarm:
            try:
                shard_path = self.swarm.download_shard(shard_id, manifest_url=target_url)
            except Exception as e:
                logger.error(f"Swarm download failed: {e}")
        
        if not shard_path:
            shard_path = self._http_fallback_download(shard_id, target_url)
                   
        # Load tensor
        try:
            tensor_data = torch.load(shard_path, weights_only=True)
            with self._shard_lock:
                self.loaded_shards[shard_id] = tensor_data
                self.current_dataset = tensor_data
                self.dataset_iterator = 0
                self.loading_shards.discard(shard_id)
            logger.info(f"Loaded Shard {shard_id}: {len(tensor_data)} tokens")
        except Exception as e:
            logger.error(f"Failed to load shard {shard_path}: {e}")
            with self._shard_lock:
                self.loading_shards.discard(shard_id)
            # Do NOT fall back to dummy data — training on random tokens corrupts
            # the model silently. Let the caller retry with a different shard.
            logger.error(f"[INTEGRITY] Shard {shard_id} is unusable. Node will retry "
                        f"with next shard in rotation.")

    def ensure_shard_loaded(self, shard_id: int = None):
        """
        Download and load a shard if not present.
        Opportunistically switches to ANY ready shard if the target isn't ready.
        """
        target_shard_id = shard_id
        
        if target_shard_id is None:
            # Default: try current shard in rotation
            if not self.assigned_shard_ids:
                return
            target_shard_id = self.assigned_shard_ids[self.current_shard_idx % len(self.assigned_shard_ids)]
        
        with self._shard_lock:
            # 1. Check if target is ready (Fastest)
            if target_shard_id in self.loaded_shards:
                self.current_dataset = self.loaded_shards[target_shard_id]
                return

            # 2. Check if target is in prefetch buffer
            if target_shard_id in self._prefetch_ready:
                self.current_dataset = self._prefetch_ready.pop(target_shard_id)
                self.loaded_shards[target_shard_id] = self.current_dataset
                self.dataset_iterator = 0
                logger.info(f"Using prefetched shard {target_shard_id}: {len(self.current_dataset)} tokens")
                self._start_prefetch_next_unlocked()
                return
            
            # 3. OPPORTUNISTIC: If target isn't ready, check if ANY assigned shard is ready in prefetch
            # This prevents blocking on shard A when shard B is already downloaded
            if shard_id is None:  # Only if caller didn't request specific shard
                for ready_id in list(self._prefetch_ready.keys()):
                    if ready_id in self.assigned_shard_ids:
                        # Switch to this ready shard!
                        logger.info(f"Opportunistically switching to ready shard {ready_id} (was waiting for {target_shard_id})")
                        
                        # Update index to match
                        try:
                            new_idx = self.assigned_shard_ids.index(ready_id)
                            self.current_shard_idx = new_idx
                        except ValueError:
                            pass
                            
                        self.current_dataset = self._prefetch_ready.pop(ready_id)
                        self.loaded_shards[ready_id] = self.current_dataset
                        self.dataset_iterator = 0
                        self._start_prefetch_next_unlocked()
                        return

            # 4. If still nothing, trigger download for target
            if target_shard_id in self.loading_shards or target_shard_id in self._prefetch_in_progress:
                logger.debug(f"Shard {target_shard_id} is already being downloaded, waiting...")
                return  # Don't block
            
            # Mark as loading and start download in background
            self.loading_shards.add(target_shard_id)
        
        target_url = self.get_shard_url(target_shard_id)
        logger.info(f"Loading Shard {target_shard_id} from {target_url}")
        
        # Submit to thread pool (non-blocking)
        self._download_executor.submit(self._load_shard_sync, target_shard_id, target_url)
    
    def _start_prefetch_next_unlocked(self):
        """Start prefetching next shard (call only when holding _shard_lock)."""
        # Schedule prefetch in background (don't hold lock during download)
        self._download_executor.submit(self._start_prefetch_next)

    def _verify_shard_integrity(self, shard_path: str, shard_id: int,
                                strict: bool = False) -> bool:
        """Verify SHA-256 hash of a downloaded shard against the manifest.
        
        Args:
            shard_path: Path to the shard file on disk
            shard_id: Shard ID for manifest lookup
            strict: If True, reject on hash mismatch (for cached files).
                    If False, warn but accept (for fresh downloads where
                    manifest hashes may be stale).
        
        Returns True if the shard is verified, no hash exists in manifest,
        or the file is a valid torch tensor (non-strict mode).
        """
        expected_hash = self._get_shard_hash(shard_id)
        if not expected_hash:
            # No hash in manifest — skip verification (backward compatible)
            return True
        
        try:
            sha256 = hashlib.sha256()
            with open(shard_path, 'rb') as f:
                for block in iter(lambda: f.read(65536), b''):
                    sha256.update(block)
            actual_hash = sha256.hexdigest()
            
            if actual_hash == expected_hash:
                logger.debug(f"[INTEGRITY] Shard {shard_id} verified: {actual_hash[:16]}...")
                return True
            else:
                if strict:
                    # Cached file mismatch — could be corrupted, reject
                    logger.warning(f"[INTEGRITY] Shard {shard_id} hash mismatch "
                                  f"(expected {expected_hash[:16]}..., got {actual_hash[:16]}...) "
                                  f"— removing cached file")
                    try:
                        os.remove(shard_path)
                    except OSError:
                        pass
                    return False
                else:
                    # Fresh download mismatch — manifest hash may be stale
                    # (CDN shards can be regenerated without updating manifest)
                    # Accept the file but log warning for monitoring
                    logger.info(f"[INTEGRITY] Shard {shard_id} hash differs from manifest "
                               f"(manifest may be stale) — accepting fresh download")
                    return True
        except Exception as e:
            logger.warning(f"[INTEGRITY] Shard {shard_id} verification error: {e}")
            # If we can't verify, accept the file — torch.load will catch real corruption
            return True
    
    def _get_shard_hash(self, shard_id: int) -> Optional[str]:
        """Get expected SHA-256 hash for a shard from the manifest."""
        if not self.manifest:
            return None
        shards_list = self.manifest.get("shards", [])
        for s in shards_list:
            if s.get("shard_id") == shard_id:
                return s.get("hash") or s.get("sha256")
        return None
    
    def _http_fallback_download(self, shard_id: int, target_url: str = None,
                                max_retries: int = 4) -> str:
        """Download shard from CloudFront CDN with retry + integrity verification.
        
        Uses exponential backoff with jitter on failure. Verifies SHA-256 hash
        against the manifest after download (if hash is available).
        """
        import random
        
        os.makedirs(self.cache_dir, exist_ok=True)
        shard_path = os.path.join(self.cache_dir, f"genesis_shard_{shard_id}.pt")
        
        # Check local cache — strict mode for cached files (could be corrupted on disk)
        if os.path.exists(shard_path):
            if self._verify_shard_integrity(shard_path, shard_id, strict=True):
                return shard_path
            # Integrity failed — re-download
            logger.warning(f"[GENESIS] Cached shard {shard_id} failed integrity check, re-downloading")
            
        # Use target URL from manifest, or construct CDN URL
        url = target_url or f"{self.GENESIS_CDN_URL}/shard_{shard_id}.pt"
        
        last_error = None
        for attempt in range(max_retries):
            try:
                # Download with streaming
                with requests.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    # Write to temp file first, then rename (atomic)
                    tmp_path = shard_path + f".tmp.{os.getpid()}"
                    with open(tmp_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): 
                            f.write(chunk)
                    os.replace(tmp_path, shard_path)
                
                # Verify integrity after fresh download — non-strict mode because
                # manifest hashes may be stale (CDN shards can be regenerated)
                self._verify_shard_integrity(shard_path, shard_id, strict=False)
                
                logger.info(f"Downloaded shard {shard_id}: {os.path.getsize(shard_path)/1e6:.1f}MB "
                           f"(attempt {attempt + 1})")
                return shard_path
                
            except Exception as e:
                last_error = e
                # Exponential backoff with jitter: 1s, 2s, 4s, 8s (+ 0-1s random)
                wait = min(2 ** attempt + random.random(), 30)
                logger.warning(f"[GENESIS] Shard {shard_id} download attempt {attempt + 1}/{max_retries} "
                              f"failed: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
        
        raise RuntimeError(f"Failed to download shard {shard_id} after {max_retries} attempts: {last_error}")

    def get_batch(self, batch_size: int = 4, seq_len: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch from the current shard.
        
        NON-BLOCKING VERSION: Returns quickly if data not ready.
        Uses prefetch buffer for instant shard switches.
        
        Automatically rotates to next shard when current one is exhausted.
        Returns (input_ids, labels).
        
        Raises RuntimeError if data not ready (caller should retry later).
        """
        # Try to load from prefetch buffer first
        self.ensure_shard_loaded()
        
        # NON-BLOCKING: Check if data is actually ready
        # Don't wait/block - let the caller handle the retry
        if self.current_dataset is None:
            # Check if anything is in progress
            with self._shard_lock:
                loading_any = bool(self.loading_shards or self._prefetch_in_progress)
                prefetch_ready = bool(self._prefetch_ready)
            
            if prefetch_ready:
                # There's a prefetched shard - try to use it
                self.ensure_shard_loaded()
            elif not loading_any:
                # Nothing loading - kick off a new load
                self._start_prefetch_next()
            
            # Return early - data not ready yet
            raise RuntimeError("Data not ready - shard still loading")
        
        data_len = len(self.current_dataset)
        req_len = (batch_size * seq_len) + 1 
        
        # Check for early rotation due to loss plateau
        if self._should_rotate_early():
            current_shard = self.assigned_shard_ids[self.current_shard_idx % len(self.assigned_shard_ids)]
            logger.info(f"[GENESIS] Early rotation from shard {current_shard} due to loss plateau")
            self.force_shard_rotation("loss_plateau")
            # Ensure new shard is loaded
            self.ensure_shard_loaded()
            if self.current_dataset is None:
                raise RuntimeError("Data not ready - loading fresh shard after plateau")
            data_len = len(self.current_dataset)
        
        # Check if we've exhausted current shard
        if self.dataset_iterator + req_len > data_len:
            # Log completion of current shard
            completed_shard = self.assigned_shard_ids[self.current_shard_idx % len(self.assigned_shard_ids)]
            steps_done = data_len // req_len
            logger.info(f"✓ Completed shard {completed_shard} ({steps_done} steps, {data_len:,} tokens)")
            
            # Reset loss tracking for new shard
            self._loss_history.clear()
            self._steps_on_current_shard = 0
            
            # Move to next shard in our assigned list
            self.current_shard_idx += 1
            
            if self.current_shard_idx >= len(self.assigned_shard_ids):
                # We've gone through all assigned shards - rotate to new set
                logger.info(f"Exhausted all {len(self.assigned_shard_ids)} assigned shards. Rotating to new set...")
                self.rotate_shards()
            
            # Try to use prefetched shard (FAST PATH)
            next_shard_id = self.assigned_shard_ids[self.current_shard_idx % len(self.assigned_shard_ids)]
            
            with self._shard_lock:
                if next_shard_id in self._prefetch_ready:
                    # Instant switch to prefetched shard
                    self.current_dataset = self._prefetch_ready.pop(next_shard_id)
                    self.loaded_shards[next_shard_id] = self.current_dataset
                    logger.info(f"Switched to prefetched shard {next_shard_id}: {len(self.current_dataset)} tokens")
                elif next_shard_id in self.loaded_shards:
                    self.current_dataset = self.loaded_shards[next_shard_id]
                else:
                    # Need to wait for next shard - trigger load
                    self.ensure_shard_loaded(next_shard_id)
                    raise RuntimeError("Data not ready - loading next shard")
            
            # Start prefetching the shard after next
            self._start_prefetch_next()
            
            self.dataset_iterator = 0
            data_len = len(self.current_dataset)
        
        start_idx = self.dataset_iterator
        end_idx = start_idx + req_len
        
        chunk = self.current_dataset[start_idx:end_idx]
        self.dataset_iterator += req_len
        
        # Log shard progress periodically (every 100 steps within shard)
        steps_in_shard = self.dataset_iterator // req_len
        total_steps_in_shard = data_len // req_len
        if steps_in_shard % 100 == 0:
            progress_pct = (self.dataset_iterator / data_len) * 100
            current_shard = self.assigned_shard_ids[self.current_shard_idx % len(self.assigned_shard_ids)]
            logger.info(f"Shard {current_shard} progress: {progress_pct:.1f}% "
                       f"({steps_in_shard}/{total_steps_in_shard} steps)")
        
        # Prepare batch
        exact_len = batch_size * seq_len
        
        # Check if chunk is large enough (might be end of shard)
        if len(chunk) < exact_len + 1:
            # Not enough data for a full batch + label shift
            # Force rotation to the next shard so we don't get stuck here
            logger.info(f"[GENESIS] Reached end of shard (partial batch: {len(chunk)} < {exact_len + 1}), rotating...")
            
            # Reset loss tracking
            self._loss_history.clear()
            self._steps_on_current_shard = 0
            
            # Move to next shard
            self.current_shard_idx += 1
            if self.current_shard_idx >= len(self.assigned_shard_ids):
                self.rotate_shards()
            
            # Switch to next shard immediately
            next_shard_id = self.assigned_shard_ids[self.current_shard_idx % len(self.assigned_shard_ids)]
            with self._shard_lock:
                if next_shard_id in self._prefetch_ready:
                    self.current_dataset = self._prefetch_ready.pop(next_shard_id)
                    self.loaded_shards[next_shard_id] = self.current_dataset
                elif next_shard_id in self.loaded_shards:
                    self.current_dataset = self.loaded_shards[next_shard_id]
                else:
                    self.current_dataset = None  # Force wait for new data
                    self.ensure_shard_loaded(next_shard_id) # Triggers background load
            
            self.dataset_iterator = 0
            self._start_prefetch_next()
            
            # Return None to skip this step (will pick up new shard next time)
            return None, None

        inputs = chunk[:exact_len].view(batch_size, seq_len)
        labels = chunk[1:exact_len+1].view(batch_size, seq_len)
        
        return inputs, labels
    
    def get_stats(self) -> dict:
        """Get loader statistics."""
        # Calculate progress within current shard
        shard_progress = 0.0
        steps_in_shard = 0
        total_steps_in_shard = 0
        current_shard_id = None
        
        if self.current_dataset is not None and len(self.current_dataset) > 0:
            data_len = len(self.current_dataset)
            req_len = 1025  # Approximate: batch_size * seq_len + 1
            shard_progress = (self.dataset_iterator / data_len) * 100
            steps_in_shard = self.dataset_iterator // req_len
            total_steps_in_shard = data_len // req_len
            if self.assigned_shard_ids:
                current_shard_id = self.assigned_shard_ids[self.current_shard_idx % len(self.assigned_shard_ids)]
        
        # Compute loss plateau stats
        loss_avg = 0.0
        loss_variance = 0.0
        if self._loss_history:
            loss_avg = sum(self._loss_history) / len(self._loss_history)
            if len(self._loss_history) >= 2:
                loss_variance = sum((l - loss_avg) ** 2 for l in self._loss_history) / len(self._loss_history)
        
        return {
            "total_shards_available": self.total_shards,
            "max_shards_configured": self.max_shards,
            "max_storage_mb": self.max_storage_mb,
            "assigned_shards": len(self.assigned_shard_ids),
            "loaded_shards": len(self.loaded_shards),
            "prefetch_in_progress": len(self._prefetch_in_progress),
            "prefetch_ready": len(self._prefetch_ready),
            "current_shard_idx": self.current_shard_idx,
            "current_shard_id": current_shard_id,
            "shard_progress_pct": round(shard_progress, 1),
            "steps_in_shard": steps_in_shard,
            "total_steps_in_shard": total_steps_in_shard,
            "rotation_count": self.shard_rotation_count,
            "storage_used_mb": len(self.loaded_shards) * self.SHARD_SIZE_MB,
            # Loss plateau detection stats
            "steps_on_current_shard": self._steps_on_current_shard,
            "loss_history_size": len(self._loss_history),
            "loss_avg": round(loss_avg, 6),
            "loss_variance": round(loss_variance, 8),
            "plateau_threshold": self._loss_plateau_threshold,
        }
