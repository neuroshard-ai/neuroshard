#!/usr/bin/env python3
"""
NeuroShard Genesis Parallel Populator v5.0

High-performance, multi-source parallel shard creation for S3.
Designed for c5.2xlarge (8 vCPU, 16GB RAM) and similar instances.

NEW in v5.0:
- Parquet file-based processing for O(1) instant resume
- No more slow skip() operations
- Can resume from any point instantly (~0.5s vs 30+ minutes)

Key Features:
- TRUE parallel processing: multiple sources simultaneously
- Intelligent resource allocation based on available RAM/CPU
- Shared tokenizer pool for maximum efficiency
- Vocabulary grows immediately from ALL sources
- Unified progress tracking and graceful shutdown
- Backward compatible with v3.0 checkpoints

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    GenesisOrchestrator                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ SourceWorker │  │ SourceWorker │  │ SourceWorker │      │
│  │ (fineweb-edu)│  │  (fineweb)   │  │ (redpajama)  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └────────────┬────┴─────────────────┘               │
│                      ▼                                      │
│         ┌────────────────────────┐                         │
│         │  SharedTokenizerPool   │ (BPE, all sources)      │
│         └────────────────────────┘                         │
│                      │                                      │
│                      ▼                                      │
│         ┌────────────────────────┐                         │
│         │   SharedS3Uploader     │ (parallel uploads)      │
│         └────────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
"""

import os
import sys
import json
import torch
import boto3
import logging
import argparse
import tempfile
import hashlib
import signal
import queue
import threading
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count, Manager, Value, Lock
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("GenesisParallel")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Standard shard size - MUST be consistent across all training data
SHARD_SIZE_MB = 10.0
TOKENS_PER_SHARD = int(SHARD_SIZE_MB * 1e6 / 4)  # ~2.5M tokens per 10MB shard

# Resource limits (will be auto-tuned based on available resources)
DEFAULT_CONFIG = {
    "max_memory_gb": 14.0,          # Leave 2GB for system
    "memory_per_source_gb": 3.0,    # Each source needs ~3GB
    "max_concurrent_sources": 4,    # Maximum parallel sources
    "tokenizer_workers_per_source": 2,  # Tokenizer workers per source
    "upload_workers": 8,            # Total S3 upload workers (shared)
    "batch_size": 4,                # Shards to batch before upload
    "checkpoint_interval": 32,      # Save checkpoint every N shards
    "manifest_update_interval": 64, # Update manifest every N shards
}

# Data sources configuration
DATA_SOURCES = {
    "fineweb-edu": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "split": "train",
        "text_field": "text",
        "priority": 1,
        "description": "High-quality educational web content"
    },
    "fineweb": {
        "hf_path": "HuggingFaceFW/fineweb",
        "split": "train",
        "text_field": "text",
        "priority": 2,
        "description": "Large-scale web content"
    },
    "redpajama": {
        "hf_path": "togethercomputer/RedPajama-Data-1T",
        "split": "train",
        "text_field": "text",
        "priority": 3,
        "description": "RedPajama 1T token dataset"
    },
    "slimpajama": {
        "hf_path": "cerebras/SlimPajama-627B",
        "split": "train",
        "text_field": "text",
        "priority": 4,
        "description": "SlimPajama 627B tokens"
    },
    "c4": {
        "hf_path": "allenai/c4",
        "hf_name": "en",  # Required: specify config name for c4
        "split": "train",
        "text_field": "text",
        "priority": 5,
        "description": "Colossal Clean Crawled Corpus"
    },
}


# ============================================================================
# ENVIRONMENT & S3
# ============================================================================

def load_env():
    """Load environment variables from .env files."""
    possible_paths = [
        Path(__file__).parent.parent / '.env',
        Path(__file__).parent.parent / 'website' / '.env',
        Path.home() / '.env',
    ]
    
    for p in possible_paths:
        if p.exists():
            logger.info(f"Loading environment from {p}")
            with open(p, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip("'").strip('"')
                        os.environ[key] = value
            return True
    
    logger.warning("No .env file found")
    return False


def get_s3_client():
    """Create optimized S3 client."""
    from botocore.config import Config
    
    config = Config(
        max_pool_connections=100,
        retries={'max_attempts': 3, 'mode': 'adaptive'}
    )
    
    return boto3.client('s3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'),
        config=config
    )


# ============================================================================
# RESOURCE MANAGER
# ============================================================================

class ResourceManager:
    """Manages system resources and determines optimal parallelism."""
    
    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG.copy()
        self._analyze_system()
    
    def _analyze_system(self):
        """Analyze available system resources."""
        # Memory
        mem = psutil.virtual_memory()
        self.total_memory_gb = mem.total / (1024**3)
        self.available_memory_gb = mem.available / (1024**3)
        
        # CPU
        self.cpu_count = cpu_count()
        self.cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Calculate optimal settings
        usable_memory = min(
            self.available_memory_gb * 0.8,  # Use 80% of available
            self.config["max_memory_gb"]
        )
        
        self.max_sources = min(
            int(usable_memory / self.config["memory_per_source_gb"]),
            self.config["max_concurrent_sources"],
            max(1, self.cpu_count // 2)  # At least 1 source even on small machines
        )
        
        # CRITICAL: Always allow at least 1 source — even small instances can
        # process data, just slower. Without this, the populator exits immediately
        # on t3.micro/small instances with < 3GB available RAM.
        if self.max_sources == 0 and self.available_memory_gb >= 0.5:
            self.max_sources = 1
            logger.warning(f"Low memory ({self.available_memory_gb:.1f}GB available) — "
                          f"running 1 source in low-memory mode")
        
        # Workers per source (share CPUs fairly)
        self.tokenizer_workers = max(1, self.cpu_count // max(1, self.max_sources))
        
        logger.info(f"System: {self.total_memory_gb:.1f}GB RAM, {self.cpu_count} CPUs")
        logger.info(f"Available: {self.available_memory_gb:.1f}GB RAM, {100-self.cpu_percent:.0f}% CPU idle")
        logger.info(f"Config: max {self.max_sources} parallel sources, {self.tokenizer_workers} workers each")
    
    def get_source_config(self, num_active_sources: int) -> dict:
        """Get configuration for a source based on current load."""
        workers = max(1, self.cpu_count // max(1, num_active_sources))
        return {
            "tokenizer_workers": workers,
            "batch_size": self.config["batch_size"],
        }


# ============================================================================
# CHECKPOINT & MANIFEST
# ============================================================================

@dataclass
class SourceCheckpoint:
    """
    Checkpoint for a single data source.
    
    NEW in v5.0: Uses parquet file indices for O(1) resume instead of
    document counts which require O(n) skip().
    """
    source: str
    # NEW: Track completed parquet file indices for instant resume
    completed_file_indices: List[int] = field(default_factory=list)
    current_file_idx: int = 0  # Currently processing this file
    docs_in_current_file: int = 0  # Progress within current file
    # Legacy fields (kept for compatibility, derived from files)
    documents_processed: int = 0
    tokens_processed: int = 0
    shards_created: int = 0
    last_shard_id: int = -1
    leftover_tokens: List[int] = field(default_factory=list)
    started_at: str = ""
    updated_at: str = ""
    # Cache: total files in dataset (set on first load)
    total_files: int = 0
    
    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.utcnow().isoformat()
        self.updated_at = datetime.utcnow().isoformat()
    
    def get_next_file_idx(self) -> int:
        """Get the next file to process (first incomplete)."""
        if not self.completed_file_indices:
            return 0
        # Find first gap in completed indices
        completed_set = set(self.completed_file_indices)
        for i in range(self.total_files):
            if i not in completed_set:
                return i
        return self.total_files  # All done
    
    def mark_file_complete(self, file_idx: int):
        """Mark a parquet file as fully processed."""
        if file_idx not in self.completed_file_indices:
            self.completed_file_indices.append(file_idx)
        self.current_file_idx = self.get_next_file_idx()
        self.docs_in_current_file = 0
    
    def to_dict(self) -> dict:
        d = asdict(self)
        # Truncate leftover tokens for JSON size
        d['leftover_tokens'] = d['leftover_tokens'][:10000]
        # Only keep last 1000 completed indices to limit size
        d['completed_file_indices'] = d['completed_file_indices'][-10000:]
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'SourceCheckpoint':
        # Handle legacy checkpoints that don't have new fields
        d.setdefault('completed_file_indices', [])
        d.setdefault('current_file_idx', 0)
        d.setdefault('docs_in_current_file', 0)
        d.setdefault('total_files', 0)
        return cls(**d)


class ManifestManager:
    """Thread-safe manifest and checkpoint management."""
    
    CLOUDFRONT_DISTRIBUTION_ID = "E3P008LWABLO43"
    
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3 = get_s3_client()
        self.cf = boto3.client('cloudfront',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name='us-east-1'
        )
        
        self._lock = threading.RLock()
        self._manifest = self._load_manifest()
        self._checkpoints: Dict[str, SourceCheckpoint] = self._load_checkpoints()
        self._pending_shards: List[dict] = []
        self._last_cdn_invalidation = 0
        self._shards_since_manifest_save = 0
    
    def _load_manifest(self) -> dict:
        """Load or create manifest."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key="manifest.json")
            manifest = json.loads(obj['Body'].read().decode('utf-8'))
            logger.info(f"Loaded manifest: {manifest['total_shards']:,} shards, {manifest.get('total_tokens', 0)/1e9:.2f}B tokens")
            
            # Ensure all required fields
            manifest.setdefault('sources', {})
            manifest.setdefault('total_tokens', 0)
            manifest.setdefault('shard_size_mb', SHARD_SIZE_MB)
            manifest.setdefault('tokens_per_shard', TOKENS_PER_SHARD)
            manifest['version'] = 4
            
            return manifest
        except Exception as e:
            logger.info(f"Creating new manifest (reason: {e})")
            return {
                "version": 4,
                "shard_size_mb": SHARD_SIZE_MB,
                "tokens_per_shard": TOKENS_PER_SHARD,
                "total_shards": 0,
                "total_tokens": 0,
                "sources": {},
                "shards": [],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
    
    def _load_checkpoints(self) -> Dict[str, SourceCheckpoint]:
        """Load checkpoints from S3."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key="checkpoints.json")
            data = json.loads(obj['Body'].read().decode('utf-8'))
            checkpoints = {}
            for k, v in data.items():
                try:
                    checkpoints[k] = SourceCheckpoint.from_dict(v)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint for {k}: {e}")
            logger.info(f"Loaded {len(checkpoints)} source checkpoints")
            return checkpoints
        except Exception as e:
            logger.info(f"No existing checkpoints (reason: {e})")
            return {}
    
    def get_checkpoint(self, source: str) -> SourceCheckpoint:
        """Get or create checkpoint for a source."""
        with self._lock:
            if source not in self._checkpoints:
                self._checkpoints[source] = SourceCheckpoint(source=source)
            return self._checkpoints[source]
    
    def get_next_shard_id(self) -> int:
        """Get next available shard ID (thread-safe)."""
        with self._lock:
            next_id = self._manifest['total_shards'] + len(self._pending_shards)
            return next_id
    
    def add_shard(self, shard_id: int, source: str, file_hash: str, 
                  size_tokens: int, size_bytes: int):
        """Add a shard to pending list (thread-safe)."""
        with self._lock:
            shard_meta = {
                "shard_id": shard_id,
                "source": source,
                "hash": file_hash,
                "size_tokens": size_tokens,
                "size_bytes": size_bytes,
                "created_at": datetime.utcnow().isoformat()
            }
            self._pending_shards.append(shard_meta)
            self._shards_since_manifest_save += 1
    
    def flush_to_manifest(self, force: bool = False):
        """Flush pending shards to manifest and save."""
        with self._lock:
            if not self._pending_shards and not force:
                return
            
            # Add pending shards
            for shard in self._pending_shards:
                self._manifest['shards'].append(shard)
                source = shard['source']
                if source not in self._manifest['sources']:
                    self._manifest['sources'][source] = {'shards': 0, 'tokens': 0}
                self._manifest['sources'][source]['shards'] += 1
                self._manifest['sources'][source]['tokens'] += shard['size_tokens']
            
            self._manifest['total_shards'] = len(self._manifest['shards'])
            self._manifest['total_tokens'] = sum(s['size_tokens'] for s in self._manifest['shards'])
            self._manifest['updated_at'] = datetime.utcnow().isoformat()
            
            self._pending_shards = []
            self._shards_since_manifest_save = 0
            
            # Save to S3
            self.s3.put_object(
                Bucket=self.bucket,
                Key="manifest.json",
                Body=json.dumps(self._manifest, indent=2),
                ContentType='application/json'
            )
            
            # CDN invalidation (rate limited)
            now = time.time()
            if now - self._last_cdn_invalidation > 300:
                self._invalidate_cdn()
                self._last_cdn_invalidation = now
    
    def save_checkpoints(self):
        """Save all checkpoints to S3."""
        with self._lock:
            data = {k: v.to_dict() for k, v in self._checkpoints.items()}
            self.s3.put_object(
                Bucket=self.bucket,
                Key="checkpoints.json",
                Body=json.dumps(data, indent=2),
                ContentType='application/json'
            )
    
    def _invalidate_cdn(self):
        """Invalidate CloudFront cache."""
        try:
            self.cf.create_invalidation(
                DistributionId=self.CLOUDFRONT_DISTRIBUTION_ID,
                InvalidationBatch={
                    'Paths': {'Quantity': 2, 'Items': ['/manifest.json', '/tokenizer.json']},
                    'CallerReference': f'genesis-parallel-{int(time.time())}'
                }
            )
            logger.debug("CDN cache invalidated")
        except Exception as e:
            logger.warning(f"CDN invalidation failed (non-fatal): {e}")
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        with self._lock:
            return {
                "total_shards": self._manifest['total_shards'] + len(self._pending_shards),
                "total_tokens": self._manifest['total_tokens'],
                "pending_shards": len(self._pending_shards),
                "sources": dict(self._manifest['sources']),
            }


# ============================================================================
# TOKENIZER (Shared across sources)
# ============================================================================

class SharedTokenizer:
    """
    Shared tokenizer that learns from all sources.
    
    BPE vocabulary grows as each source contributes its allocated merges.
    """
    
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.s3 = get_s3_client()
        self._lock = threading.Lock()
        self._tokenizer = None
        self._tokenizer_path = Path(__file__).parent / "learned_tokenizer.json"
        self._sources_config = self._load_sources_config()
        
    def _load_sources_config(self) -> dict:
        """Load sources configuration for fair vocab allocation."""
        config_path = Path(__file__).parent / "genesis_sources.json"
        try:
            with open(config_path) as f:
                return json.load(f)
        except:
            return {"sources": [], "total_target_shards": 8400000}
    
    def initialize(self):
        """Initialize or load tokenizer."""
        from neuroshard.core.model.tokenizer import NeuroTokenizer
        
        with self._lock:
            if self._tokenizer_path.exists():
                logger.info(f"Loading existing tokenizer from {self._tokenizer_path}")
                self._tokenizer = NeuroTokenizer.load(str(self._tokenizer_path))
                logger.info(f"Tokenizer: {self._tokenizer.current_vocab_size:,} tokens, "
                           f"{len(self._tokenizer.merges):,} BPE merges")
            else:
                logger.info("Creating new tokenizer")
                self._tokenizer = NeuroTokenizer()
    
    def ensure_source_contributed(self, source: str) -> bool:
        """
        Ensure a source has contributed its full BPE merge allocation.
        
        With unlimited vocab growth, sources can contribute MORE merges
        if they haven't reached their full allocation yet.
        
        Returns True if new merges were learned.
        """
        with self._lock:
            # Check if vocabulary is saturated (no more pairs to merge)
            # This happens when previous learning attempts returned 0 merges
            # Detect saturation: if total contributions == current merges, we can't grow more
            total_contributions = sum(self._tokenizer.sources_contributed.values())
            current_merges = len(self._tokenizer.merges)
            if total_contributions > 0 and total_contributions >= current_merges * 0.95:
                # Vocab is 95%+ utilized - likely saturated
                logger.info(f"[{source}] Vocabulary saturated ({current_merges:,} merges, "
                           f"{total_contributions:,} contributed), skipping BPE learning")
                return False
            
            # Check if vocabulary is full (very unlikely with 10M capacity)
            if self._tokenizer.current_vocab_size >= self._tokenizer.vocab_size:
                logger.info(f"[{source}] Vocabulary full ({self._tokenizer.current_vocab_size:,})")
                return False
            
            # Calculate how many merges this source SHOULD have
            target_allocation = self._calculate_allocation(source)
            
            # Check how many this source has ALREADY contributed
            already_contributed = self._tokenizer.sources_contributed.get(source, 0)
            
            # Calculate additional merges needed to reach allocation
            additional_needed = target_allocation - already_contributed
            
            if additional_needed <= 0:
                logger.info(f"[{source}] Already at full allocation ({already_contributed:,} merges)")
                return False
            
            # Don't exceed remaining vocab capacity
            remaining_capacity = self._tokenizer.vocab_size - self._tokenizer.next_merge_id
            merges_to_learn = min(additional_needed, remaining_capacity)
            
            if merges_to_learn <= 0:
                return False
            
            logger.info(f"[{source}] Learning {merges_to_learn:,} additional BPE merges "
                       f"(already: {already_contributed:,}, target: {target_allocation:,})...")
            self._learn_merges(source, merges_to_learn)
            return True
    
    def _calculate_allocation(self, source: str) -> int:
        """
        Calculate this source's fair share of vocabulary.
        
        With unlimited vocab (10M), each source gets a generous allocation
        to learn its unique vocabulary. The vocab grows continuously as
        more sources contribute - this is a key feature of NeuroShard.
        """
        # With 10M vocab capacity, each source can learn 100K+ merges
        # This ensures rich vocabulary from each data domain
        BASE_ALLOCATION_PER_SOURCE = 100000  # 100K merges per source
        
        total_shards = 0
        source_shards = 0
        
        for s in self._sources_config.get("sources", []):
            if s.get("enabled", True):
                shards = s.get("target_shards", 0)
                total_shards += shards
                if s["name"] == source:
                    source_shards = shards
        
        if total_shards == 0 or source_shards == 0:
            return BASE_ALLOCATION_PER_SOURCE
        
        # Proportional allocation with generous base
        proportion = source_shards / total_shards
        allocated = max(BASE_ALLOCATION_PER_SOURCE, int(1000000 * proportion))  # At least 100K
        
        # Don't exceed remaining slots
        remaining = self._tokenizer.vocab_size - self._tokenizer.next_merge_id
        return min(allocated, remaining)
    
    def _learn_merges(self, source: str, num_merges: int):
        """Learn BPE merges from source data."""
        from datasets import load_dataset
        
        config = DATA_SOURCES[source]
        sample_size = 10000
        
        logger.info(f"[{source}] Sampling {sample_size:,} documents for BPE learning...")
        
        # Some datasets (like c4) require a config name
        hf_name = config.get('hf_name')
        
        dataset = load_dataset(
            config['hf_path'],
            name=hf_name,  # None is fine for datasets that don't need it
            split=config['split'],
            streaming=True
        )
        
        sample_texts = []
        for i, doc in enumerate(dataset):
            if i >= sample_size:
                break
            text = doc.get(config['text_field'], "")
            if text:
                sample_texts.append(text)
        
        logger.info(f"[{source}] Learning {num_merges:,} merges from {len(sample_texts):,} docs...")
        
        vocab_before = self._tokenizer.current_vocab_size
        self._tokenizer.learn_merges(sample_texts, num_merges=num_merges, min_frequency=2)
        vocab_after = self._tokenizer.current_vocab_size
        
        actual_learned = vocab_after - vocab_before
        
        # If we learned 0 merges, mark vocabulary as saturated
        # This prevents future sources from wasting time trying to learn
        if actual_learned == 0:
            self._tokenizer._vocab_saturated = True
            logger.info(f"[{source}] Vocabulary saturated - no more frequent pairs to merge")
        
        # Add to existing contribution (for incremental learning)
        prev_contribution = self._tokenizer.sources_contributed.get(source, 0)
        self._tokenizer.sources_contributed[source] = prev_contribution + actual_learned
        logger.info(f"[{source}] Total contribution: {self._tokenizer.sources_contributed[source]:,} merges")
        
        logger.info(f"[{source}] ✓ Learned {actual_learned:,} merges (vocab: {vocab_before:,} → {vocab_after:,})")
        
        # Save locally
        self._tokenizer.save(str(self._tokenizer_path))
        
        # Upload to S3
        self._upload_tokenizer()
    
    def _upload_tokenizer(self):
        """Upload tokenizer to S3."""
        try:
            with open(self._tokenizer_path, 'rb') as f:
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key='tokenizer.json',
                    Body=f.read(),
                    ContentType='application/json'
                )
            logger.info("✓ Tokenizer uploaded to S3")
        except Exception as e:
            logger.error(f"Failed to upload tokenizer: {e}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to tokens (thread-safe read)."""
        return self._tokenizer.encode(text, add_special_tokens=False)
    
    def get_stats(self) -> dict:
        """Get tokenizer statistics."""
        with self._lock:
            return {
                "vocab_size": self._tokenizer.current_vocab_size,
                "max_vocab_size": self._tokenizer.vocab_size,
                "num_merges": len(self._tokenizer.merges),
                "sources_contributed": dict(self._tokenizer.sources_contributed),
            }


# ============================================================================
# SOURCE WORKER
# ============================================================================

class SourceWorker:
    """
    Worker that processes a single data source.
    
    Each worker:
    - Streams data from HuggingFace
    - Tokenizes using shared tokenizer
    - Creates shards
    - Submits to shared uploader
    """
    
    def __init__(self, source: str, target_shards: int,
                 manifest: ManifestManager, tokenizer: SharedTokenizer,
                 upload_queue: queue.Queue, config: dict):
        self.source = source
        self.target_shards = target_shards
        self.manifest = manifest
        self.tokenizer = tokenizer
        self.upload_queue = upload_queue
        self.config = config
        
        self.checkpoint = manifest.get_checkpoint(source)
        self.source_config = DATA_SOURCES[source]
        
        self._running = False
        self._thread = None
        self._stats = {
            "shards_created": 0,
            "tokens_processed": 0,
            "docs_processed": 0,
            "rate_shards_per_min": 0.0,
        }
        self._stats_lock = threading.Lock()
        self._start_time = None
        
        # NEW v5.0: Parquet file-based processing
        self._parquet_files = []
        self._current_file_idx = -1
        self._docs_in_file = 0
        self._docs_seen = 0
    
    def start(self):
        """Start the worker thread."""
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, name=f"Worker-{self.source}")
        self._thread.start()
        logger.info(f"[{self.source}] Worker started")
    
    def stop(self):
        """Stop the worker gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=30)
        logger.info(f"[{self.source}] Worker stopped")
    
    def is_complete(self) -> bool:
        """Check if source has reached target or processed all files."""
        # Check shard target
        current = self.manifest._manifest['sources'].get(self.source, {}).get('shards', 0)
        if current >= self.target_shards:
            return True
        
        # Also check if all files are processed
        if hasattr(self, '_parquet_files') and self._parquet_files:
            if len(self.checkpoint.completed_file_indices) >= len(self._parquet_files):
                return True
        
        return False
    
    def get_stats(self) -> dict:
        """Get worker statistics."""
        with self._stats_lock:
            return dict(self._stats)
    
    def _run(self):
        """Main worker loop."""
        try:
            # Ensure tokenizer has this source's BPE merges
            self.tokenizer.ensure_source_contributed(self.source)
            
            # Setup data stream
            self._setup_stream()
            
            # Process loop
            batch_tokens = []
            while self._running and not self.is_complete():
                # Fetch and tokenize documents
                tokens = self._fetch_and_tokenize_batch()
                if not tokens:
                    logger.warning(f"[{self.source}] End of data stream")
                    break
                
                batch_tokens.extend(tokens)
                
                # Create shards when we have enough tokens
                while len(batch_tokens) >= TOKENS_PER_SHARD:
                    shard_tokens = batch_tokens[:TOKENS_PER_SHARD]
                    batch_tokens = batch_tokens[TOKENS_PER_SHARD:]
                    
                    shard_id = self.manifest.get_next_shard_id()
                    
                    # Submit to upload queue
                    self.upload_queue.put({
                        "shard_id": shard_id,
                        "source": self.source,
                        "tokens": shard_tokens,
                    })
                    
                    with self._stats_lock:
                        self._stats["shards_created"] += 1
                        self._stats["tokens_processed"] += len(shard_tokens)
                        
                        elapsed = time.time() - self._start_time
                        if elapsed > 0:
                            self._stats["rate_shards_per_min"] = self._stats["shards_created"] / elapsed * 60
                    
                    # Update checkpoint periodically
                    if self._stats["shards_created"] % self.config.get("checkpoint_interval", 32) == 0:
                        self._save_checkpoint(batch_tokens)
            
            # Save final checkpoint
            self._save_checkpoint(batch_tokens)
            
        except Exception as e:
            logger.error(f"[{self.source}] Worker error: {e}", exc_info=True)
        
        logger.info(f"[{self.source}] Worker finished: {self._stats['shards_created']:,} shards")
    
    def _setup_stream(self):
        """
        Setup HuggingFace data stream with INSTANT resume.
        
        NEW in v5.0: Uses parquet file-based access for O(1) resume.
        Instead of skip(N_million_docs), we just load the next unprocessed file.
        """
        from datasets import load_dataset
        from huggingface_hub import HfApi
        
        logger.info(f"[{self.source}] Loading dataset with parquet-file-based access...")
        
        hf_path = self.source_config['hf_path']
        hf_name = self.source_config.get('hf_name')
        
        # Get list of parquet files for this dataset
        if not hasattr(self, '_parquet_files') or not self._parquet_files:
            self._parquet_files = self._get_parquet_files(hf_path)
            self.checkpoint.total_files = len(self._parquet_files)
            logger.info(f"[{self.source}] Found {len(self._parquet_files)} parquet files")
        
        # Find next file to process
        next_file_idx = self.checkpoint.get_next_file_idx()
        
        if next_file_idx >= len(self._parquet_files):
            logger.info(f"[{self.source}] All files processed!")
            self._iterator = iter([])  # Empty iterator
            self._current_file_idx = -1
            return
        
        self._current_file_idx = next_file_idx
        current_file = self._parquet_files[next_file_idx]
        
        logger.info(f"[{self.source}] Loading file {next_file_idx}/{len(self._parquet_files)}: {current_file}")
        
        # Load specific parquet file - INSTANT, no skip() needed!
        try:
            dataset = load_dataset(
                hf_path,
                name=hf_name,
                data_files=current_file,
                split="train",
                streaming=True
            )
            self._iterator = iter(dataset)
            self._docs_seen = self.checkpoint.documents_processed
            self._docs_in_file = 0
            
        except Exception as e:
            logger.error(f"[{self.source}] Failed to load file {current_file}: {e}")
            # Mark as complete and try next
            self.checkpoint.mark_file_complete(next_file_idx)
            self._setup_stream()  # Recursively try next file
    
    def _get_parquet_files(self, hf_path: str) -> List[str]:
        """
        Get list of data files for a dataset.
        
        Supports multiple formats:
        - .parquet files (fineweb, fineweb-edu)
        - .json.gz files (c4)
        """
        from huggingface_hub import HfApi
        
        try:
            api = HfApi()
            all_files = api.list_repo_files(hf_path, repo_type="dataset")
            
            # Filter for data files (parquet or json.gz)
            data_files = []
            
            # Try parquet first (most common for HF datasets)
            parquet_files = [
                f for f in all_files 
                if f.endswith('.parquet') and ('data/' in f or 'train' in f.lower())
            ]
            
            if parquet_files:
                data_files = parquet_files
            else:
                # Try json.gz (used by c4)
                json_files = [
                    f for f in all_files
                    if f.endswith('.json.gz') and 'train' in f.lower()
                ]
                data_files = json_files
            
            # Sort for consistent ordering
            data_files.sort()
            
            return data_files
            
        except Exception as e:
            logger.error(f"[{self.source}] Failed to list data files: {e}")
            return []
    
    def _advance_to_next_file(self):
        """Move to the next parquet file after completing current one."""
        if self._current_file_idx >= 0:
            self.checkpoint.mark_file_complete(self._current_file_idx)
            logger.info(f"[{self.source}] Completed file {self._current_file_idx}, "
                       f"total files done: {len(self.checkpoint.completed_file_indices)}/{self.checkpoint.total_files}")
        self._setup_stream()
    
    def _fetch_and_tokenize_batch(self, batch_size: int = 64) -> List[int]:
        """
        Fetch and tokenize a batch of documents.
        
        NEW in v5.0: Handles end-of-file by automatically advancing
        to the next parquet file.
        
        NEW in v5.1: Parallel tokenization for better CPU utilization.
        """
        from concurrent.futures import ThreadPoolExecutor
        
        texts = []
        text_field = self.source_config['text_field']
        
        # First, fetch a batch of documents
        for _ in range(batch_size):
            try:
                doc = next(self._iterator)
                text = doc.get(text_field, "")
                if text:
                    texts.append(text)
                self._docs_seen += 1
                self._docs_in_file += 1
                    
            except StopIteration:
                # End of current file - advance to next
                logger.info(f"[{self.source}] Finished file {self._current_file_idx} "
                           f"({self._docs_in_file:,} docs)")
                self._advance_to_next_file()
                
                # Check if we've exhausted all files
                if self._current_file_idx < 0:
                    break  # All files processed
                    
                # Continue fetching from new file
                continue
        
        if not texts:
            return []
        
        # Tokenize documents - use larger batches for efficiency
        # NOTE: Python GIL prevents true parallel tokenization in threads.
        # For maximum speed, we process more documents per batch.
        tokens = []
        for text in texts:
            doc_tokens = self.tokenizer.encode(text)
            tokens.extend(doc_tokens)
        
        with self._stats_lock:
            self._stats["docs_processed"] = self._docs_seen
        
        return tokens
    
    def _save_checkpoint(self, leftover_tokens: List[int]):
        """Save checkpoint to manifest manager."""
        self.checkpoint.documents_processed = self._docs_seen
        self.checkpoint.shards_created = self._stats["shards_created"]
        self.checkpoint.tokens_processed = self._stats["tokens_processed"]
        self.checkpoint.leftover_tokens = leftover_tokens[:10000]
        self.checkpoint.updated_at = datetime.utcnow().isoformat()


# ============================================================================
# UPLOAD WORKER
# ============================================================================

class SharedUploader:
    """
    Shared uploader that handles S3 uploads for all sources.
    
    Uses a thread pool to upload shards in parallel.
    """
    
    def __init__(self, bucket: str, manifest: ManifestManager, num_workers: int = 8):
        self.bucket = bucket
        self.manifest = manifest
        self.num_workers = num_workers
        
        self._queue = queue.Queue(maxsize=100)
        self._running = False
        self._workers = []
        self._s3_pool = None
        
        self._stats = {
            "uploaded": 0,
            "failed": 0,
            "bytes_uploaded": 0,
        }
        self._stats_lock = threading.Lock()
    
    def get_queue(self) -> queue.Queue:
        """Get the upload queue for workers to submit to."""
        return self._queue
    
    def start(self):
        """Start upload workers."""
        self._running = True
        self._s3_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        
        for i in range(self.num_workers):
            t = threading.Thread(target=self._upload_loop, name=f"Uploader-{i}")
            t.start()
            self._workers.append(t)
        
        logger.info(f"Started {self.num_workers} upload workers")
    
    def stop(self):
        """Stop upload workers gracefully."""
        self._running = False
        
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except:
                break
        
        # Signal workers to stop
        for _ in self._workers:
            self._queue.put(None)
        
        # Wait for workers
        for t in self._workers:
            t.join(timeout=10)
        
        if self._s3_pool:
            self._s3_pool.shutdown(wait=True)
        
        logger.info(f"Upload workers stopped. Uploaded: {self._stats['uploaded']:,}")
    
    def _upload_loop(self):
        """Upload worker loop."""
        s3 = get_s3_client()
        
        while self._running:
            try:
                item = self._queue.get(timeout=1)
                if item is None:
                    break
                
                self._upload_shard(s3, item)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Upload error: {e}")
    
    def _upload_shard(self, s3, item: dict):
        """Upload a single shard."""
        shard_id = item["shard_id"]
        source = item["source"]
        tokens = item["tokens"]
        
        try:
            # Create tensor and save to temp file
            tensor = torch.tensor(tokens, dtype=torch.long)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                torch.save(tensor, tmp.name)
                tmp_path = tmp.name
            
            try:
                # Calculate hash
                with open(tmp_path, 'rb') as f:
                    file_bytes = f.read()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()
                
                # Upload
                filename = f"shard_{shard_id}.pt"
                s3.upload_file(tmp_path, self.bucket, filename)
                
                # Update manifest
                self.manifest.add_shard(
                    shard_id=shard_id,
                    source=source,
                    file_hash=file_hash,
                    size_tokens=len(tokens),
                    size_bytes=len(file_bytes)
                )
                
                with self._stats_lock:
                    self._stats["uploaded"] += 1
                    self._stats["bytes_uploaded"] += len(file_bytes)
                
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"Failed to upload shard {shard_id}: {e}")
            with self._stats_lock:
                self._stats["failed"] += 1
    
    def get_stats(self) -> dict:
        """Get upload statistics."""
        with self._stats_lock:
            return dict(self._stats)


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class GenesisOrchestrator:
    """
    Main orchestrator that coordinates all components.
    
    Manages:
    - Multiple source workers running in parallel
    - Shared tokenizer with growing vocabulary
    - Shared upload pool
    - Progress tracking and graceful shutdown
    """
    
    def __init__(self, bucket: str, sources: List[Tuple[str, int]], config: dict = None):
        """
        Initialize orchestrator.
        
        Args:
            bucket: S3 bucket name
            sources: List of (source_name, target_shards) tuples
            config: Optional configuration overrides
        """
        self.bucket = bucket
        self.sources = sources
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        self.resource_mgr = ResourceManager(self.config)
        self.manifest = ManifestManager(bucket)
        self.tokenizer = SharedTokenizer(bucket)
        self.uploader = None
        self.workers: Dict[str, SourceWorker] = {}
        
        self._running = False
        self._interrupted = False
        self._start_time = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"\n⚠️  Signal {signum} received, shutting down gracefully...")
        self._interrupted = True
        self._running = False
    
    def run(self):
        """Run the orchestrator."""
        self._start_time = time.time()
        self._running = True
        
        logger.info("=" * 70)
        logger.info("  NeuroShard Genesis Parallel Populator v4.0")
        logger.info("=" * 70)
        
        # Initialize tokenizer
        self.tokenizer.initialize()
        
        # Start uploader
        self.uploader = SharedUploader(
            self.bucket,
            self.manifest,
            num_workers=self.config["upload_workers"]
        )
        self.uploader.start()
        
        # Filter sources that need work
        active_sources = []
        for source, target in self.sources:
            current = self.manifest._manifest['sources'].get(source, {}).get('shards', 0)
            if current < target:
                active_sources.append((source, target))
                logger.info(f"  • {source}: {current:,}/{target:,} shards (need {target-current:,})")
            else:
                logger.info(f"  ✓ {source}: Complete ({current:,}/{target:,})")
        
        if not active_sources:
            logger.info("All sources complete!")
            self.uploader.stop()
            return
        
        # Limit concurrent sources based on resources
        max_concurrent = min(len(active_sources), self.resource_mgr.max_sources)
        logger.info(f"\nRunning {max_concurrent} sources in parallel")
        logger.info("=" * 70)
        
        # Start workers for active sources
        for source, target in active_sources[:max_concurrent]:
            worker_config = self.resource_mgr.get_source_config(max_concurrent)
            worker = SourceWorker(
                source=source,
                target_shards=target,
                manifest=self.manifest,
                tokenizer=self.tokenizer,
                upload_queue=self.uploader.get_queue(),
                config=worker_config
            )
            self.workers[source] = worker
            worker.start()
        
        # Monitor loop
        last_stats_time = time.time()
        last_manifest_save = time.time()
        
        try:
            while self._running:
                time.sleep(5)
                
                # Check if all workers done
                all_complete = all(w.is_complete() or not w._running 
                                   for w in self.workers.values())
                if all_complete:
                    logger.info("All active sources complete!")
                    break
                
                # Periodic stats
                now = time.time()
                if now - last_stats_time > 30:
                    self._log_stats()
                    last_stats_time = now
                
                # Periodic manifest save
                if now - last_manifest_save > 60:
                    self.manifest.flush_to_manifest()
                    self.manifest.save_checkpoints()
                    last_manifest_save = now
                
        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
        
        finally:
            self._shutdown()
    
    def _log_stats(self):
        """Log current statistics."""
        elapsed = time.time() - self._start_time
        
        manifest_stats = self.manifest.get_stats()
        upload_stats = self.uploader.get_stats()
        tokenizer_stats = self.tokenizer.get_stats()
        
        total_rate = upload_stats["uploaded"] / elapsed * 60 if elapsed > 0 else 0
        
        logger.info("-" * 50)
        logger.info(f"📊 Progress Report (elapsed: {elapsed/60:.1f}min)")
        logger.info(f"   Total shards: {manifest_stats['total_shards']:,} ({total_rate:.1f}/min)")
        logger.info(f"   Total tokens: {manifest_stats['total_tokens']/1e9:.2f}B")
        logger.info(f"   Vocabulary:   {tokenizer_stats['vocab_size']:,}/{tokenizer_stats['max_vocab_size']:,}")
        logger.info(f"   Upload queue: {self.uploader._queue.qsize()}")
        
        for source, worker in self.workers.items():
            stats = worker.get_stats()
            logger.info(f"   [{source}] {stats['shards_created']:,} shards, "
                       f"{stats['rate_shards_per_min']:.1f}/min")
        
        logger.info("-" * 50)
    
    def _shutdown(self):
        """Graceful shutdown."""
        logger.info("\n🛑 Shutting down...")
        
        # Stop workers
        for worker in self.workers.values():
            worker.stop()
        
        # Stop uploader
        if self.uploader:
            self.uploader.stop()
        
        # Final save
        logger.info("Saving final checkpoint and manifest...")
        self.manifest.flush_to_manifest(force=True)
        self.manifest.save_checkpoints()
        
        # Final stats
        self._log_stats()
        
        if self._interrupted:
            logger.info("\n✓ Checkpoint saved. Run again to continue from exact position.")


# ============================================================================
# CLI
# ============================================================================

def load_sources_from_config() -> List[Tuple[str, int]]:
    """Load sources from genesis_sources.json."""
    config_path = Path(__file__).parent / "genesis_sources.json"
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        sources = []
        for s in sorted(config.get("sources", []), key=lambda x: x.get("priority", 99)):
            if s.get("enabled", True):
                sources.append((s["name"], s.get("target_shards", 500000)))
        
        return sources
    except Exception as e:
        logger.warning(f"Could not load sources config: {e}")
        return [("fineweb-edu", 600000)]


def main():
    load_env()
    
    parser = argparse.ArgumentParser(
        description="NeuroShard Genesis Parallel Populator v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration (all enabled sources in parallel)
  python genesis_parallel.py --bucket neuroshard-training-data
  
  # Run specific sources only
  python genesis_parallel.py --bucket neuroshard-training-data --sources fineweb-edu,fineweb
  
  # Limit concurrent sources
  python genesis_parallel.py --bucket neuroshard-training-data --max-sources 2
  
  # Show current status
  python genesis_parallel.py --bucket neuroshard-training-data --status

Performance on c5.2xlarge (8 vCPU, 16GB):
  - 3-4 sources in parallel
  - ~40-60 shards/minute total
  - Vocabulary grows immediately from all sources
        """
    )
    
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--sources", help="Comma-separated list of sources to run")
    parser.add_argument("--max-sources", type=int, default=4, 
                        help="Maximum concurrent sources (default: 4)")
    parser.add_argument("--upload-workers", type=int, default=8,
                        help="Number of upload workers (default: 8)")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    
    args = parser.parse_args()
    
    if args.status:
        manifest = ManifestManager(args.bucket)
        tokenizer = SharedTokenizer(args.bucket)
        tokenizer.initialize()
        
        stats = manifest.get_stats()
        tok_stats = tokenizer.get_stats()
        
        print(f"\n{'='*60}")
        print(f"NeuroShard Genesis Status")
        print(f"{'='*60}")
        print(f"Total Shards:  {stats['total_shards']:,}")
        print(f"Total Tokens:  {stats['total_tokens']/1e9:.2f}B")
        print(f"Vocabulary:    {tok_stats['vocab_size']:,}/{tok_stats['max_vocab_size']:,} tokens")
        print(f"\nSources:")
        for src, data in stats['sources'].items():
            print(f"  {src}: {data['shards']:,} shards, {data['tokens']/1e9:.2f}B tokens")
        print(f"\nVocab Contributions:")
        for src, merges in tok_stats['sources_contributed'].items():
            print(f"  {src}: {merges:,} BPE merges")
        print(f"{'='*60}\n")
        return
    
    # Load sources
    if args.sources:
        source_names = [s.strip() for s in args.sources.split(",")]
        all_sources = dict(load_sources_from_config())
        sources = [(name, all_sources.get(name, 500000)) for name in source_names]
    else:
        sources = load_sources_from_config()
    
    # Create config
    config = {
        **DEFAULT_CONFIG,
        "max_concurrent_sources": args.max_sources,
        "upload_workers": args.upload_workers,
    }
    
    # Run
    orchestrator = GenesisOrchestrator(args.bucket, sources, config)
    orchestrator.run()


if __name__ == "__main__":
    main()
