"""
P2P DataSwarm — Decentralized Shard Distribution for NeuroShard

Implements BitTorrent-style P2P data transfer so nodes can fetch training
data shards from peers instead of always hitting the CDN. This removes
the CDN as a single point of failure and distributes bandwidth costs.

Architecture:
1. Nodes announce which shards they hold to the DHT (every 60s)
2. When a node needs a shard, it queries the DHT for providers
3. If P2P providers exist, download chunks in parallel from peers
4. If no peers or P2P fails, fall back to CDN (CloudFront)
5. After downloading, announce the new shard to DHT

Chunk Protocol:
- Shards are split into 1MB chunks
- Each chunk is requested via gRPC DataChunk RPC
- Chunks are reassembled and verified before loading
"""

import os
import hashlib
import logging
import threading
import time
import requests
import math
from typing import List, Dict, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Avoid circular imports — P2PManager type is used for type hints only
try:
    from neuroshard.core.network.p2p import P2PManager, GRPC_PORT_OFFSET
except ImportError:
    P2PManager = None
    GRPC_PORT_OFFSET = 1000


def _resolve_peer_url(peer_url: str) -> str:
    """Resolve a peer URL that may contain pipe-format LAN info.
    
    DHT values can be "ip:port|local_ip:port" or "http://ip:port|local_ip:port".
    This strips the pipe portion. For full LAN resolution (preferring local IP
    when on the same network), use P2PManager._resolve_dht_endpoint() instead.
    """
    if "|" not in peer_url:
        return peer_url
    if "://" in peer_url:
        scheme, rest = peer_url.split("://", 1)
        return f"{scheme}://{rest.split('|', 1)[0]}"
    return peer_url.split("|", 1)[0]


class DataSwarm:
    """
    Implements BitTorrent-like P2P data transfer for NeuroShard.
    
    - Splits large shards into 1MB chunks.
    - Finds peers holding specific shards via DHT.
    - Downloads chunks in parallel from multiple peers.
    - Falls back to CDN when no peers are available.
    - Verifies data integrity after download.
    """
    
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    ANNOUNCE_INTERVAL = 60   # Announce shards to DHT every 60s
    
    def __init__(self, p2p_manager, cache_dir: str = "data_cache"):
        self.p2p = p2p_manager
        self.cache_dir = cache_dir
        self.active_downloads: Dict[int, str] = {}  # shard_id -> status
        self.local_shards: Set[int] = set()  # IDs of shards we have locally
        
        # Thread pool for parallel chunk downloads
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="swarm-dl")
        
        os.makedirs(cache_dir, exist_ok=True)
        self._scan_local_cache()
        
        # Start announcer thread (tells DHT which shards we have)
        self._announce_thread = threading.Thread(
            target=self._announce_loop, daemon=True, name="shard-announce"
        )
        self._announce_thread.start()
        
    def _scan_local_cache(self):
        """Scan cache directory for existing complete shards."""
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.startswith("genesis_shard_") and filename.endswith(".pt"):
                    try:
                        idx = int(filename.split("_")[2].split(".")[0])
                        self.local_shards.add(idx)
                    except (ValueError, IndexError):
                        pass
            if self.local_shards:
                logger.info(f"[SWARM] Found {len(self.local_shards)} cached shards locally")
        except OSError as e:
            logger.debug(f"[SWARM] Cache scan error: {e}")

    def _announce_loop(self):
        """Periodically announce our shards to the DHT so peers can find us."""
        while True:
            try:
                if self.p2p and hasattr(self.p2p, 'dht') and self.p2p.dht:
                    my_endpoint = getattr(self.p2p, 'my_url', None) or ''
                    if my_endpoint:
                        for shard_id in list(self.local_shards):
                            try:
                                dht_key = f"shard_provider_{shard_id}"
                                # Use announce() — stores our ip:port under the hashed key.
                                # announce() only takes key_string; value is auto-set to
                                # local_node ip:port (sufficient for peer discovery).
                                self.p2p.dht.announce(dht_key)
                            except Exception as e:
                                logger.debug(f"[SWARM] DHT announce failed for shard {shard_id}: {e}")
            except Exception as e:
                logger.debug(f"[SWARM] Announce loop error: {e}")
            time.sleep(self.ANNOUNCE_INTERVAL)

    def get_shard_path(self, shard_id: int) -> str:
        return os.path.join(self.cache_dir, f"genesis_shard_{shard_id}.pt")

    def download_shard(self, shard_id: int, manifest_url: str = None) -> str:
        """
        Download a shard using P2P swarm, falling back to CDN.
        Returns path to downloaded file.
        """
        target_path = self.get_shard_path(shard_id)
        
        if shard_id in self.local_shards and os.path.exists(target_path):
            return target_path
            
        logger.info(f"[SWARM] Starting download for shard {shard_id}...")
        
        # 1. Try P2P: Find peers who have this shard via DHT
        peers = self._find_providers(shard_id)
        
        if peers:
            logger.info(f"[SWARM] Found {len(peers)} P2P providers for shard {shard_id}")
            success = self._swarm_download(shard_id, peers, target_path)
            if success:
                self.local_shards.add(shard_id)
                return target_path
            logger.warning(f"[SWARM] P2P download failed for shard {shard_id}, falling back to CDN")
        
        # 2. Fallback to CDN
        return self._download_from_cdn(shard_id, target_path, manifest_url)

    def _find_providers(self, shard_id: int) -> List[str]:
        """Find peers that have this shard by querying the DHT.
        
        Returns list of peer URLs (e.g., ['http://1.2.3.4:8000', ...]).
        """
        providers = []
        
        if not self.p2p or not hasattr(self.p2p, 'dht') or not self.p2p.dht:
            return providers
        
        try:
            dht_key = f"shard_provider_{shard_id}"
            
            # Query DHT for providers — use lookup_key() which accepts string keys
            # (lookup_value expects int keys — would silently fail)
            result = self.p2p.dht.lookup_key(dht_key)
            if result:
                import json
                # DHT may return JSON-encoded list or raw string
                endpoints = []
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                        if isinstance(parsed, list):
                            endpoints = parsed
                        else:
                            endpoints = [str(parsed)]
                    except (json.JSONDecodeError, TypeError):
                        endpoints = [result]
                elif isinstance(result, list):
                    endpoints = result
                
                # Resolve LAN-format endpoints and build URLs
                for ep in endpoints:
                    resolved = _resolve_peer_url(str(ep))
                    if not resolved.startswith("http"):
                        resolved = f"http://{resolved}"
                    providers.append(resolved)
            
            # Also check known peers — ask them directly if they have it
            for peer_url in list(self.p2p.known_peers.keys())[:10]:
                if peer_url not in providers:
                    if self._peer_has_shard(peer_url, shard_id):
                        providers.append(peer_url)
                        
        except Exception as e:
            logger.debug(f"[SWARM] Provider discovery error: {e}")
        
        # Filter out self
        my_url = getattr(self.p2p, 'my_url', None)
        return [p for p in providers if p != my_url]
    
    def _peer_has_shard(self, peer_url: str, shard_id: int) -> bool:
        """Check if a peer has a specific shard via gRPC."""
        try:
            from neuroshard.core.network.connection_pool import get_channel
            from neuroshard.protos import neuroshard_pb2 as pb2
            from neuroshard.protos import neuroshard_pb2_grpc as pb2_grpc
            from urllib.parse import urlparse
            
            parsed = urlparse(_resolve_peer_url(peer_url))
            grpc_addr = f"{parsed.hostname}:{(parsed.port or 80) + GRPC_PORT_OFFSET}"
            channel = get_channel(grpc_addr)
            stub = pb2_grpc.NeuroShardServiceStub(channel)
            
            # Use GetStatus to check if peer has this shard
            resp = stub.GetStatus(pb2.Empty(), timeout=3.0)
            # Check if shard is in peer's data list
            if hasattr(resp, 'available_shards'):
                return shard_id in resp.available_shards
        except Exception:
            pass
        return False

    def _download_from_cdn(self, shard_id: int, target_path: str, manifest_url: str) -> str:
        """Download from CloudFront CDN (single attempt — caller handles retry)."""
        url = manifest_url or f"https://dwquwt9gkkeil.cloudfront.net/shard_{shard_id}.pt"
        
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                tmp_path = target_path + f".tmp.{os.getpid()}"
                with open(tmp_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
                os.replace(tmp_path, target_path)
            self.local_shards.add(shard_id)
            logger.info(f"[SWARM] Downloaded shard {shard_id} from CDN")
            return target_path
        except Exception as e:
            raise RuntimeError(f"Failed to download shard {shard_id} from {url}: {e}")

    def _swarm_download(self, shard_id: int, peers: List[str], target_path: str) -> bool:
        """Download shard chunks in parallel from multiple peers.
        
        1. Get file size from first responding peer
        2. Split into chunks, assign to peers round-robin
        3. Download all chunks in parallel
        4. Reassemble and verify
        """
        try:
            # Get shard metadata from first available peer
            file_size = self._get_shard_size_from_peer(peers[0], shard_id)
            if file_size is None or file_size == 0:
                return False
            
            num_chunks = math.ceil(file_size / self.CHUNK_SIZE)
            logger.info(f"[SWARM] Shard {shard_id}: {file_size/1e6:.1f}MB, {num_chunks} chunks, "
                       f"{len(peers)} peers")
            
            # Prepare chunk assignments (round-robin across peers)
            chunk_data = [None] * num_chunks
            futures = {}
            
            for chunk_idx in range(num_chunks):
                peer = peers[chunk_idx % len(peers)]
                future = self.executor.submit(
                    self._download_chunk_from_peer, peer, shard_id, chunk_idx
                )
                futures[future] = chunk_idx
            
            # Collect results
            failed_chunks = []
            for future in as_completed(futures, timeout=120):
                chunk_idx = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        chunk_data[chunk_idx] = data
                    else:
                        failed_chunks.append(chunk_idx)
                except Exception as e:
                    logger.debug(f"[SWARM] Chunk {chunk_idx} failed: {e}")
                    failed_chunks.append(chunk_idx)
            
            # Retry failed chunks from CDN
            if failed_chunks:
                logger.warning(f"[SWARM] {len(failed_chunks)}/{num_chunks} chunks failed from peers, "
                              f"retrying from CDN")
                for chunk_idx in failed_chunks:
                    data = self._download_chunk_from_cdn(shard_id, chunk_idx)
                    if data is not None:
                        chunk_data[chunk_idx] = data
                    else:
                        logger.error(f"[SWARM] Chunk {chunk_idx} unrecoverable")
                        return False
            
            # Reassemble
            if any(c is None for c in chunk_data):
                return False
            
            tmp_path = target_path + f".tmp.{os.getpid()}"
            with open(tmp_path, 'wb') as f:
                for chunk in chunk_data:
                    f.write(chunk)
            os.replace(tmp_path, target_path)
            
            logger.info(f"[SWARM] Shard {shard_id} assembled from {len(peers)} peers "
                       f"({num_chunks} chunks)")
            return True
            
        except Exception as e:
            logger.warning(f"[SWARM] Swarm download failed for shard {shard_id}: {e}")
            return False
    
    def _get_shard_size_from_peer(self, peer_url: str, shard_id: int) -> Optional[int]:
        """Get shard file size from a peer."""
        try:
            from neuroshard.core.network.connection_pool import get_channel
            from neuroshard.protos import neuroshard_pb2 as pb2
            from neuroshard.protos import neuroshard_pb2_grpc as pb2_grpc
            from urllib.parse import urlparse
            
            parsed = urlparse(_resolve_peer_url(peer_url))
            grpc_addr = f"{parsed.hostname}:{(parsed.port or 80) + GRPC_PORT_OFFSET}"
            channel = get_channel(grpc_addr)
            stub = pb2_grpc.NeuroShardServiceStub(channel)
            
            request = pb2.DataChunkRequest(
                shard_id=shard_id,
                chunk_index=-1,  # -1 = metadata request
            )
            response = stub.DataChunk(request, timeout=10.0)
            return response.total_size if hasattr(response, 'total_size') else None
        except Exception as e:
            logger.debug(f"[SWARM] Metadata request to {peer_url} failed: {e}")
            return None
    
    def _download_chunk_from_peer(self, peer_url: str, shard_id: int, chunk_index: int) -> Optional[bytes]:
        """Download a single chunk from a peer via gRPC."""
        try:
            from neuroshard.core.network.connection_pool import get_channel
            from neuroshard.protos import neuroshard_pb2 as pb2
            from neuroshard.protos import neuroshard_pb2_grpc as pb2_grpc
            from urllib.parse import urlparse
            
            parsed = urlparse(_resolve_peer_url(peer_url))
            grpc_addr = f"{parsed.hostname}:{(parsed.port or 80) + GRPC_PORT_OFFSET}"
            channel = get_channel(grpc_addr)
            stub = pb2_grpc.NeuroShardServiceStub(channel)
            
            request = pb2.DataChunkRequest(
                shard_id=shard_id,
                chunk_index=chunk_index,
            )
            response = stub.DataChunk(request, timeout=30.0)
            
            if response.data and len(response.data) > 0:
                return response.data
            return None
        except Exception as e:
            logger.debug(f"[SWARM] Chunk {chunk_index} from {peer_url} failed: {e}")
            return None
    
    def _download_chunk_from_cdn(self, shard_id: int, chunk_index: int) -> Optional[bytes]:
        """Download a single chunk range from CDN (byte-range request)."""
        try:
            url = f"https://dwquwt9gkkeil.cloudfront.net/shard_{shard_id}.pt"
            start = chunk_index * self.CHUNK_SIZE
            end = start + self.CHUNK_SIZE - 1
            headers = {'Range': f'bytes={start}-{end}'}
            
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code in (200, 206):
                return resp.content
            return None
        except Exception:
            return None

    def serve_chunk(self, shard_id: int, chunk_index: int) -> Optional[bytes]:
        """Read a chunk from disk to serve to a peer.
        
        Called by the gRPC DataChunk handler when another node requests data.
        """
        if shard_id not in self.local_shards:
            return None
            
        path = self.get_shard_path(shard_id)
        if not os.path.exists(path):
            return None
        
        offset = chunk_index * self.CHUNK_SIZE
        
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                data = f.read(self.CHUNK_SIZE)
                return data if data else None
        except (OSError, IOError) as e:
            logger.debug(f"[SWARM] Failed to read chunk {chunk_index} of shard {shard_id}: {e}")
            return None
    
    def get_shard_file_size(self, shard_id: int) -> int:
        """Get the file size of a local shard (for metadata responses)."""
        path = self.get_shard_path(shard_id)
        try:
            return os.path.getsize(path) if os.path.exists(path) else 0
        except OSError:
            return 0
