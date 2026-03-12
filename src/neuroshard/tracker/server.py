
from fastapi import FastAPI, HTTPException, Body, Request, Query
from pydantic import BaseModel
from typing import Dict, List, Optional
import time
import uvicorn
import os
import asyncio
import logging

# Database Abstraction
class DatabaseManager:
    def __init__(self):
        self.pool = None
        self.db_url = os.getenv("DATABASE_URL", "sqlite://tracker.db")
        self.is_postgres = self.db_url.startswith("postgres")
        self._sqlite_path = self.db_url.replace("sqlite://", "")

    async def connect(self):
        if self.is_postgres:
            import asyncpg
            # Wait for DB to be ready
            for i in range(5):
                try:
                    self.pool = await asyncpg.create_pool(self.db_url)
                    break
                except Exception as e:
                    print(f"Waiting for DB... {e}")
                    await asyncio.sleep(2)
            if not self.pool:
                raise Exception("Could not connect to Postgres")
            
            await self.init_postgres()
        else:
            self.init_sqlite()

    async def close(self):
        if self.pool:
            await self.pool.close()

    def init_sqlite(self):
        import sqlite3
        with sqlite3.connect(self._sqlite_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS peers (
                    url TEXT PRIMARY KEY,
                    ip TEXT,
                    port INTEGER,
                    shard_range TEXT,
                    shard_start INTEGER,
                    shard_end INTEGER,
                    is_entry BOOLEAN,
                    is_exit BOOLEAN,
                    last_seen REAL,
                    tps REAL,
                    latency REAL,
                    node_token TEXT,
                    training_enabled BOOLEAN DEFAULT 1
                )
            """)
            # Migration: add training_enabled column if missing
            try:
                conn.execute("ALTER TABLE peers ADD COLUMN training_enabled BOOLEAN DEFAULT 1")
            except:
                pass  # Column already exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stakes (
                    url TEXT PRIMARY KEY,
                    amount REAL,
                    slashed BOOLEAN DEFAULT 0
                )
            """)
            # Phase 4: Tensor shard tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tensor_shards (
                    id TEXT PRIMARY KEY,
                    model_id TEXT,
                    layer_id INTEGER,
                    shard_id INTEGER,
                    total_shards INTEGER,
                    node_url TEXT,
                    grpc_addr TEXT,
                    available_memory_mb REAL,
                    current_load REAL,
                    last_seen REAL,
                    node_token TEXT
                )
            """)
            # Phase 4: Model registry
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT,
                    family TEXT,
                    num_layers INTEGER,
                    hidden_dim INTEGER,
                    total_size_mb REAL,
                    required_stake REAL,
                    approved BOOLEAN DEFAULT 1,
                    proposer_token TEXT,
                    created_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shard_start ON peers(shard_start)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shard_range ON peers(shard_range)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_seen ON peers(last_seen)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tensor_model_layer ON tensor_shards(model_id, layer_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tensor_last_seen ON tensor_shards(last_seen)")

    async def init_postgres(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS peers (
                    url TEXT PRIMARY KEY,
                    ip TEXT,
                    port INTEGER,
                    shard_range TEXT,
                    shard_start INTEGER,
                    shard_end INTEGER,
                    is_entry BOOLEAN,
                    is_exit BOOLEAN,
                    last_seen DOUBLE PRECISION,
                    tps DOUBLE PRECISION,
                    latency DOUBLE PRECISION,
                    node_token TEXT,
                    training_enabled BOOLEAN DEFAULT TRUE
                )
            """)
            # Migration: add training_enabled column if missing
            try:
                await conn.execute("ALTER TABLE peers ADD COLUMN training_enabled BOOLEAN DEFAULT TRUE")
            except:
                pass  # Column already exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS stakes (
                    url TEXT PRIMARY KEY,
                    amount DOUBLE PRECISION,
                    slashed BOOLEAN DEFAULT FALSE
                )
            """)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_shard_start ON peers(shard_start)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_shard_range ON peers(shard_range)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_last_seen ON peers(last_seen)")

    async def upsert_peer(self, url, ip, port, shard_range, start, end, is_entry, is_exit, now, tps, latency, node_token, training_enabled=True):
        if self.is_postgres:
            query = """
                INSERT INTO peers 
                (url, ip, port, shard_range, shard_start, shard_end, is_entry, is_exit, last_seen, tps, latency, node_token, training_enabled)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (url) DO UPDATE SET
                    ip = EXCLUDED.ip,
                    port = EXCLUDED.port,
                    shard_range = EXCLUDED.shard_range,
                    shard_start = EXCLUDED.shard_start,
                    shard_end = EXCLUDED.shard_end,
                    is_entry = EXCLUDED.is_entry,
                    is_exit = EXCLUDED.is_exit,
                    last_seen = EXCLUDED.last_seen,
                    tps = EXCLUDED.tps,
                    latency = EXCLUDED.latency,
                    node_token = EXCLUDED.node_token,
                    training_enabled = EXCLUDED.training_enabled
            """
            await self.pool.execute(query, url, ip, port, shard_range, start, end, is_entry, is_exit, now, tps, latency, node_token, training_enabled)
        else:
            import sqlite3
            with sqlite3.connect(self._sqlite_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO peers 
                    (url, ip, port, shard_range, shard_start, shard_end, is_entry, is_exit, last_seen, tps, latency, node_token, training_enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (url, ip, port, shard_range, start, end, is_entry, is_exit, now, tps, latency, node_token, training_enabled))

    async def get_slashed_status(self, url):
        if self.is_postgres:
            row = await self.pool.fetchrow("SELECT slashed FROM stakes WHERE url = $1", url)
            return row['slashed'] if row else False
        else:
            import sqlite3
            with sqlite3.connect(self._sqlite_path) as conn:
                cursor = conn.execute("SELECT slashed FROM stakes WHERE url = ?", (url,))
                row = cursor.fetchone()
                return row[0] if row else False

    async def grant_initial_stake(self, url):
        if self.is_postgres:
            await self.pool.execute("INSERT INTO stakes (url, amount) VALUES ($1, 1000.0) ON CONFLICT DO NOTHING", url)
        else:
            import sqlite3
            with sqlite3.connect(self._sqlite_path) as conn:
                conn.execute("INSERT OR IGNORE INTO stakes (url, amount) VALUES (?, 1000.0)", (url,))

    async def get_stats(self, now):
        if self.is_postgres:
            count = await self.pool.fetchval("SELECT COUNT(*) FROM peers WHERE last_seen > $1", now - 60)
            row = await self.pool.fetchrow("SELECT SUM(tps) as tps, AVG(latency) as lat FROM peers WHERE last_seen > $1", now - 60)
            return count, row['tps'] or 0, row['lat'] or 0
        else:
            import sqlite3
            with sqlite3.connect(self._sqlite_path) as conn:
                count = conn.execute("SELECT COUNT(*) FROM peers WHERE last_seen > ?", (now - 60,)).fetchone()[0]
                stats = conn.execute("SELECT SUM(tps), AVG(latency) FROM peers WHERE last_seen > ?", (now - 60,)).fetchone()
                return count, stats[0] or 0, stats[1] or 0

    async def get_stake(self, url):
        if self.is_postgres:
            return await self.pool.fetchval("SELECT amount FROM stakes WHERE url = $1", url)
        else:
            import sqlite3
            with sqlite3.connect(self._sqlite_path) as conn:
                res = conn.execute("SELECT amount FROM stakes WHERE url = ?", (url,)).fetchone()
                return res[0] if res else 0.0
    
    async def get_all_stakes(self):
        if self.is_postgres:
            rows = await self.pool.fetch("SELECT url, amount, slashed FROM stakes")
            return [dict(r) for r in rows]
        else:
            import sqlite3
            with sqlite3.connect(self._sqlite_path) as conn:
                cursor = conn.execute("SELECT url, amount, slashed FROM stakes")
                return [{"url": r[0], "amount": r[1], "slashed": bool(r[2])} for r in cursor]

    async def check_active(self, token, now):
        if self.is_postgres:
            return await self.pool.fetchval("SELECT 1 FROM peers WHERE node_token = $1 AND last_seen > $2", token, now - 60)
        else:
            import sqlite3
            with sqlite3.connect(self._sqlite_path) as conn:
                return conn.execute("SELECT 1 FROM peers WHERE node_token = ? AND last_seen > ?", (token, now - 60)).fetchone()

    async def get_peers(self, layer_needed, shard_range, limit, now, training_only=False):
        if self.is_postgres:
            query = "SELECT url, shard_range, last_seen, tps, latency, node_token, training_enabled FROM peers WHERE last_seen > $1"
            params = [now - 60]

            idx = 2
            if layer_needed is not None:
                query += f" AND shard_start <= ${idx} AND shard_end > ${idx+1}"
                params.extend([layer_needed, layer_needed])
                idx += 2

            if shard_range is not None:
                query += f" AND shard_range = ${idx}"
                params.append(shard_range)
                idx += 1

            if training_only:
                query += f" AND training_enabled = TRUE"

            query += f" ORDER BY RANDOM() LIMIT ${idx}"
            params.append(limit)

            rows = await self.pool.fetch(query, *params)
            return [dict(row) for row in rows]
        else:
            import sqlite3
            query = "SELECT url, shard_range, last_seen, tps, latency, node_token, training_enabled FROM peers WHERE last_seen > ?"
            params = [now - 60]

            if layer_needed is not None:
                query += " AND shard_start <= ? AND shard_end > ?"
                params.extend([layer_needed, layer_needed])

            if shard_range is not None:
                query += " AND shard_range = ?"
                params.append(shard_range)

            if training_only:
                query += " AND training_enabled = 1"

            query += " ORDER BY RANDOM() LIMIT ?"
            params.append(limit)

            with sqlite3.connect(self._sqlite_path) as conn:
                cursor = conn.execute(query, params)
                return [
                    {"url": r[0], "shard_range": r[1], "last_seen": r[2], "tps": r[3], "latency": r[4], "node_token": r[5], "training_enabled": bool(r[6]) if r[6] is not None else True}
                    for r in cursor
                ]

    async def get_active_tokens(self, limit, offset, now):
        if self.is_postgres:
            tokens = await self.pool.fetch(
                "SELECT node_token FROM peers WHERE last_seen > $1 AND node_token IS NOT NULL ORDER BY last_seen DESC LIMIT $2 OFFSET $3",
                now - 60, limit, offset
            )
            total = await self.pool.fetchval(
                "SELECT COUNT(*) FROM peers WHERE last_seen > $1 AND node_token IS NOT NULL",
                now - 60
            )
            return [r['node_token'] for r in tokens], total
        else:
            import sqlite3
            with sqlite3.connect(self._sqlite_path) as conn:
                cursor = conn.execute(
                    "SELECT node_token FROM peers WHERE last_seen > ? AND node_token IS NOT NULL ORDER BY last_seen DESC LIMIT ? OFFSET ?", 
                    (now - 60, limit, offset)
                )
                tokens = [row[0] for row in cursor.fetchall()]
                total = conn.execute(
                    "SELECT COUNT(*) FROM peers WHERE last_seen > ? AND node_token IS NOT NULL", 
                    (now - 60,)
                ).fetchone()[0]
                return tokens, total

    async def add_stake(self, url, amount):
        if self.is_postgres:
            await self.pool.execute("""
                INSERT INTO stakes (url, amount) VALUES ($1, $2) 
                ON CONFLICT(url) DO UPDATE SET amount = stakes.amount + $3
            """, url, amount, amount)
            return await self.pool.fetchval("SELECT amount FROM stakes WHERE url = $1", url)
        else:
            import sqlite3
            with sqlite3.connect(self._sqlite_path) as conn:
                conn.execute("INSERT INTO stakes (url, amount) VALUES (?, ?) ON CONFLICT(url) DO UPDATE SET amount = amount + ?", (url, amount, amount))
                return conn.execute("SELECT amount FROM stakes WHERE url = ?", (url,)).fetchone()[0]

    async def slash_node(self, url):
        if self.is_postgres:
            async with self.pool.acquire() as conn:
                await conn.execute("UPDATE stakes SET amount = 0, slashed = TRUE WHERE url = $1", url)
                await conn.execute("DELETE FROM peers WHERE url = $1", url)
        else:
            import sqlite3
            with sqlite3.connect(self._sqlite_path) as conn:
                conn.execute("UPDATE stakes SET amount = 0, slashed = 1 WHERE url = ?", (url,))
                conn.execute("DELETE FROM peers WHERE url = ?", (url,))

app = FastAPI(title="NeuroShard Tracker")
db = DatabaseManager()

@app.on_event("startup")
async def startup():
    await db.connect()

@app.on_event("shutdown")
async def shutdown():
    await db.close()

class NodeAnnouncement(BaseModel):
    ip: Optional[str] = None  # Can be None, will use request client IP as fallback
    port: int
    shard_range: str = "0-0"  # e.g. "0-4", "observer", "unassigned"
    is_entry: bool = False
    is_exit: bool = False
    tps: float = 0.0
    latency: float = 0.0
    node_token: Optional[str] = None
    training_enabled: bool = True  # Whether node can participate in training pipeline

class PeerInfo(BaseModel):
    url: str
    shard_range: str
    last_seen: float
    tps: float
    latency: float
    training_enabled: bool = True  # Whether node can participate in training pipeline
    node_token: Optional[str] = None

@app.post("/announce")
async def announce(node: NodeAnnouncement, request: Request):
    # Use node.ip if provided, otherwise fall back to request client IP
    # This handles cases where the client sends None or empty IP
    client_ip = node.ip
    if not client_ip or client_ip in ("None", "null", ""):
        client_ip = request.client.host if request.client else "unknown"
    
    url = f"http://{client_ip}:{node.port}"
    
    # Parse shard range - supports multiple formats:
    # - "0-11" (traditional layer range)
    # - "dynamic-57-layers" (dynamic mode)
    # - "observer" (observer mode - no layers)
    # - "unassigned" (node not yet assigned layers)
    shard_range = node.shard_range or "0-0"
    
    # Handle special shard_range values
    if shard_range in ("observer", "unassigned"):
        # Observer nodes and unassigned nodes don't have layers yet
        # Set to -1,-1 so they won't match layer queries
        start, end = -1, -1
    elif shard_range.startswith("dynamic-"):
        # Dynamic mode: "dynamic-57-layers" means node has layers 0 to 56 (57 layers total)
        try:
            parts = shard_range.split("-")
            if len(parts) >= 2:
                num_layers = int(parts[1])
                start = 0  # Dynamic nodes always start at layer 0 (embedding)
                end = num_layers  # End is exclusive, so 57 layers = 0-56
            else:
                start, end = 0, 1
        except Exception as e:
            logging.debug(f"Failed to parse dynamic shard_range '{shard_range}': {e}")
            start, end = 0, 1
    else:
        # Traditional format: "0-11" means layers 0 to 11
        try:
            start, end = map(int, shard_range.split("-"))
            end = end + 1  # Make end exclusive for consistent querying
        except Exception as e:
            logging.debug(f"Failed to parse shard_range '{shard_range}': {e}")
            start, end = 0, 1  # Default to at least layer 0

    now = time.time()

    # Check if slashed
    if await db.get_slashed_status(url):
         raise HTTPException(status_code=403, detail="Node is banned (slashed).")

    # UPSERT Peer (include training_enabled for pipeline routing)
    await db.upsert_peer(url, client_ip, node.port, node.shard_range, start, end, node.is_entry, node.is_exit, now, node.tps, node.latency, node.node_token, node.training_enabled)
    
    # REMOVED: Grant initial free stake for PoC if new
    # This was a development feature - nodes should earn or stake their own NEURO
    # await db.grant_initial_stake(url)
    
    # Get stats
    count, tps, latency = await db.get_stats(now)
    stake = await db.get_stake(url)

    return {"status": "registered", "peer_count": count, "stake": stake}

@app.get("/check_active")
async def check_active(token: str):
    now = time.time()
    if await db.check_active(token, now):
        return {"active": True}
    raise HTTPException(status_code=404, detail="No active node found with this token")

@app.get("/peers")
async def get_peers(
    layer_needed: Optional[int] = None,
    shard_range: Optional[str] = None,
    limit: int = 50,
    training_only: bool = False
):
    """
    Get list of active peers.
    
    Args:
        layer_needed: Filter to peers that have this layer
        shard_range: Filter to peers with matching shard range
        limit: Max number of peers to return
        training_only: If True, only return peers with training enabled (for pipeline routing)
    """
    now = time.time()
    peers = await db.get_peers(layer_needed, shard_range, limit, now, training_only)
    return peers

@app.get("/active_tokens")
async def get_active_tokens(limit: int = 100, offset: int = 0):
    """Return a paginated list of active node tokens for reward distribution."""
    now = time.time()
    tokens, total = await db.get_active_tokens(limit, offset, now)
    
    return {
        "tokens": tokens,
        "total": total,
        "page_size": limit,
        "offset": offset
    }

@app.get("/stats")
async def get_stats_endpoint():
    now = time.time()
    count, tps, lat = await db.get_stats(now)
    
    return {
        "active_nodes": count,
        "model_size": "142B",
        "total_tps": int(tps),
        "avg_latency": f"{int(lat)}ms"
    }

# --- Staking & Slashing Endpoints ---

@app.get("/stakes")
async def get_all_stakes():
    """Get list of all stakes."""
    return await db.get_all_stakes()

@app.post("/stake")
async def add_stake_endpoint(url: str = Body(...), amount: float = Body(...)):
    # Check slashed
    if await db.get_slashed_status(url):
        return {"error": "Node slashed"}
        
    new_stake = await db.add_stake(url, amount)
    return {"new_stake": new_stake}

@app.post("/slash")
async def slash_node_endpoint(url: str = Body(...), reason: str = Body(...)):
    print(f"SLASHING NODE {url} for {reason}")
    await db.slash_node(url)
    return {"status": "slashed", "url": url}

# --- Phase 4: Tensor Shard Endpoints ---

class TensorShardAnnouncement(BaseModel):
    model_id: str
    layer_id: int
    shard_id: int
    total_shards: int
    grpc_addr: str
    available_memory_mb: float = 0.0
    current_load: float = 0.0
    node_token: Optional[str] = None

@app.post("/tensor_shards/announce")
async def announce_tensor_shard(shard: TensorShardAnnouncement, request: Request):
    """Announce a tensor shard availability."""
    now = time.time()
    
    # Generate unique ID
    shard_id_key = f"{shard.model_id}:{shard.layer_id}:{shard.shard_id}:{shard.total_shards}"
    
    # Get client URL from request or grpc_addr
    client_ip = request.client.host if request.client else "unknown"
    node_url = f"http://{client_ip}:8000"  # Assume standard port
    
    import sqlite3
    with sqlite3.connect(db._sqlite_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO tensor_shards 
            (id, model_id, layer_id, shard_id, total_shards, node_url, grpc_addr, 
             available_memory_mb, current_load, last_seen, node_token)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (shard_id_key, shard.model_id, shard.layer_id, shard.shard_id, 
              shard.total_shards, node_url, shard.grpc_addr, 
              shard.available_memory_mb, shard.current_load, now, shard.node_token))
    
    return {"status": "registered", "shard_key": shard_id_key}

@app.get("/tensor_shards")
async def get_tensor_shards(
    model_id: str,
    layer_id: int,
    total_shards: int
):
    """Get all tensor shards for a specific layer."""
    now = time.time()
    
    import sqlite3
    with sqlite3.connect(db._sqlite_path) as conn:
        cursor = conn.execute("""
            SELECT shard_id, grpc_addr, available_memory_mb, current_load, last_seen, node_url
            FROM tensor_shards 
            WHERE model_id = ? AND layer_id = ? AND total_shards = ? AND last_seen > ?
            ORDER BY shard_id
        """, (model_id, layer_id, total_shards, now - 120))  # 2 minute timeout
        
        shards = []
        for row in cursor:
            shards.append({
                "shard_id": row[0],
                "grpc_addr": row[1],
                "available_memory_mb": row[2],
                "current_load": row[3],
                "last_seen": row[4],
                "node_url": row[5]
            })
    
    return {"shards": shards, "total_found": len(shards)}

@app.get("/tensor_shards/coverage")
async def get_tensor_shard_coverage(model_id: str):
    """Get tensor shard coverage for a model."""
    now = time.time()
    
    import sqlite3
    with sqlite3.connect(db._sqlite_path) as conn:
        # Get all active shards for this model
        cursor = conn.execute("""
            SELECT layer_id, shard_id, total_shards, COUNT(*) as node_count
            FROM tensor_shards 
            WHERE model_id = ? AND last_seen > ?
            GROUP BY layer_id, shard_id, total_shards
        """, (model_id, now - 120))
        
        coverage = {}
        for row in cursor:
            layer_id, shard_id, total_shards, node_count = row
            key = f"layer_{layer_id}"
            if key not in coverage:
                coverage[key] = {"layer_id": layer_id, "shards": {}}
            coverage[key]["shards"][shard_id] = {
                "shard_id": shard_id,
                "total_shards": total_shards,
                "node_count": node_count
            }
    
    # Check completeness
    complete_layers = []
    incomplete_layers = []
    
    for layer_key, layer_info in coverage.items():
        shards = layer_info["shards"]
        if shards:
            total = list(shards.values())[0]["total_shards"]
            if len(shards) == total and all(s["node_count"] > 0 for s in shards.values()):
                complete_layers.append(layer_info["layer_id"])
            else:
                incomplete_layers.append(layer_info["layer_id"])
    
    return {
        "model_id": model_id,
        "coverage": coverage,
        "complete_layers": complete_layers,
        "incomplete_layers": incomplete_layers,
        "is_inference_ready": len(incomplete_layers) == 0 and len(complete_layers) > 0
    }

# --- Phase 4: Model Registry Endpoints ---

class ModelRegistration(BaseModel):
    model_id: str
    name: str
    family: str
    num_layers: int
    hidden_dim: int
    total_size_mb: float
    required_stake: float = 0.0
    node_token: Optional[str] = None

@app.post("/models/register")
async def register_model(model: ModelRegistration):
    """Register a new model (requires stake for custom models)."""
    now = time.time()
    
    import sqlite3
    with sqlite3.connect(db._sqlite_path) as conn:
        # Check if already exists
        existing = conn.execute(
            "SELECT 1 FROM models WHERE model_id = ?", 
            (model.model_id,)
        ).fetchone()
        
        if existing:
            return {"status": "exists", "model_id": model.model_id}
        
        # Only NeuroLLM is supported - this is a decentralized network for our model
        approved = model.family == "neurollm"
        
        conn.execute("""
            INSERT INTO models 
            (model_id, name, family, num_layers, hidden_dim, total_size_mb, 
             required_stake, approved, proposer_token, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (model.model_id, model.name, model.family, model.num_layers,
              model.hidden_dim, model.total_size_mb, model.required_stake,
              approved, model.node_token, now))
    
    return {"status": "registered", "model_id": model.model_id, "approved": approved}

@app.get("/models")
async def list_models(approved_only: bool = True, family: Optional[str] = None):
    """List available models."""
    import sqlite3
    with sqlite3.connect(db._sqlite_path) as conn:
        query = "SELECT * FROM models WHERE 1=1"
        params = []
        
        if approved_only:
            query += " AND approved = 1"
        
        if family:
            query += " AND family = ?"
            params.append(family)
        
        query += " ORDER BY total_size_mb"
        
        cursor = conn.execute(query, params)
        models = []
        for row in cursor:
            models.append({
                "model_id": row[0],
                "name": row[1],
                "family": row[2],
                "num_layers": row[3],
                "hidden_dim": row[4],
                "total_size_mb": row[5],
                "required_stake": row[6],
                "approved": bool(row[7])
            })
    
    return {"models": models}

@app.get("/models/{model_id}/status")
async def get_model_status(model_id: str):
    """Get status of a specific model in the network."""
    now = time.time()
    
    import sqlite3
    with sqlite3.connect(db._sqlite_path) as conn:
        # Get model info
        model_row = conn.execute(
            "SELECT * FROM models WHERE model_id = ?", 
            (model_id,)
        ).fetchone()
        
        if not model_row:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Get pipeline coverage (from peers)
        peers = await db.get_peers(layer_needed=None, shard_range=None, limit=1000, now=now)
        
        # Filter peers serving this model (for now, assume all peers serve neurollm)
        # In production, peers would announce which model they serve
        
        layer_coverage = {}
        for peer in peers:
            shard_range = peer.get("shard_range", "0-0")
            try:
                start, end = map(int, shard_range.split("-"))
                for layer in range(start, end + 1):
                    if layer not in layer_coverage:
                        layer_coverage[layer] = 0
                    layer_coverage[layer] += 1
            except:
                continue
        
        # Get tensor shard coverage
        tensor_cursor = conn.execute("""
            SELECT layer_id, COUNT(DISTINCT shard_id) as shard_count, 
                   MAX(total_shards) as total_shards
            FROM tensor_shards 
            WHERE model_id = ? AND last_seen > ?
            GROUP BY layer_id
        """, (model_id, now - 120))
        
        tensor_coverage = {}
        for row in tensor_cursor:
            tensor_coverage[row[0]] = {
                "shards_available": row[1],
                "total_shards": row[2],
                "complete": row[1] == row[2]
            }
    
    return {
        "model_id": model_id,
        "name": model_row[1],
        "family": model_row[2],
        "num_layers": model_row[3],
        "pipeline_coverage": layer_coverage,
        "tensor_coverage": tensor_coverage,
        "total_nodes": len(peers),
        "is_fully_covered": all(
            layer_coverage.get(i, 0) > 0 
            for i in range(model_row[3])
        )
    }

@app.get("/layer_coverage")
async def get_layer_coverage():
    """
    Get the current distribution of nodes across transformer layers.
    Used by DynamicShardManager to determine optimal layer allocation.
    """
    now = time.time()
    
    # Get all active peers
    peers = await db.get_peers(layer_needed=None, shard_range=None, limit=1000, now=now)
    
    # Count nodes per layer (GPT-2 has 12 layers: 0-11)
    TOTAL_LAYERS = 12
    layer_coverage = {i: {"layer_id": i, "node_count": 0, "nodes": []} for i in range(TOTAL_LAYERS)}
    
    for peer in peers:
        shard_range = peer.get("shard_range", "0-0")
        try:
            start, end = map(int, shard_range.split("-"))
            # Each node covers layers from start to end (inclusive based on convention)
            for layer in range(start, min(end + 1, TOTAL_LAYERS)):
                layer_coverage[layer]["node_count"] += 1
                layer_coverage[layer]["nodes"].append(peer.get("url", ""))
        except:
            continue
    
    # Calculate statistics
    node_counts = [lc["node_count"] for lc in layer_coverage.values()]
    total_nodes = len(peers)
    avg_coverage = sum(node_counts) / TOTAL_LAYERS if TOTAL_LAYERS > 0 else 0
    min_coverage = min(node_counts) if node_counts else 0
    max_coverage = max(node_counts) if node_counts else 0
    
    # Find underserved layers (below average)
    underserved = [
        layer_id for layer_id, lc in layer_coverage.items() 
        if lc["node_count"] < avg_coverage
    ]
    
    # Find critical layers (entry=0, exit=11)
    entry_coverage = layer_coverage[0]["node_count"]
    exit_coverage = layer_coverage[TOTAL_LAYERS - 1]["node_count"]
    
    return {
        "total_layers": TOTAL_LAYERS,
        "total_active_nodes": total_nodes,
        "layer_coverage": [
            {
                "layer_id": lc["layer_id"],
                "node_count": lc["node_count"],
                # Don't include full node list in response for privacy/size
            }
            for lc in layer_coverage.values()
        ],
        "statistics": {
            "avg_nodes_per_layer": round(avg_coverage, 2),
            "min_coverage": min_coverage,
            "max_coverage": max_coverage,
            "underserved_layers": underserved,
            "entry_layer_nodes": entry_coverage,
            "exit_layer_nodes": exit_coverage
        },
        "health": {
            "has_full_coverage": min_coverage > 0,
            "is_balanced": (max_coverage - min_coverage) <= avg_coverage if avg_coverage > 0 else True,
            "entry_healthy": entry_coverage >= 1,
            "exit_healthy": exit_coverage >= 1
        }
    }


@app.get("/network_architecture")
async def get_network_architecture():
    """
    Get the current network-wide architecture.
    
    This endpoint helps nodes rejoin the network with the correct architecture.
    The architecture is determined by querying active nodes and finding consensus.
    
    Returns the architecture used by the majority of active nodes.
    """
    import requests
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    now = time.time()
    
    # Get active peers
    peers = await db.get_peers(layer_needed=None, shard_range=None, limit=50, now=now)
    
    if not peers:
        return {
            "status": "no_peers",
            "message": "No active peers - network is bootstrapping",
            "hidden_dim": None,
        }
    
    # Query architecture from active peers (using requests in thread pool)
    def query_peer_arch(peer_url: str):
        """Query a single peer for its architecture (blocking)."""
        try:
            response = requests.get(f"{peer_url}/api/node/architecture", timeout=3.0)
            if response.status_code == 200:
                arch_data = response.json()
                if arch_data.get("hidden_dim"):
                    return arch_data
        except Exception:
            pass
        return None
    
    # Run queries in thread pool to not block the event loop
    architectures = []
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=5) as executor:
        tasks = []
        for peer in peers[:10]:  # Sample up to 10 peers
            peer_url = peer.get("url", "")
            if peer_url:
                tasks.append(loop.run_in_executor(executor, query_peer_arch, peer_url))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            architectures = [r for r in results if r and not isinstance(r, Exception)]
    
    if not architectures:
        return {
            "status": "unavailable",
            "message": "Could not query peer architectures",
            "hidden_dim": None,
        }
    
    # Find consensus (majority architecture)
    # Group by (hidden_dim, num_heads, num_kv_heads) tuple
    arch_counts = {}
    for arch in architectures:
        key = (arch["hidden_dim"], arch["num_heads"], arch["num_kv_heads"])
        if key not in arch_counts:
            arch_counts[key] = {"count": 0, "data": arch}
        arch_counts[key]["count"] += 1
    
    # Find the most common architecture
    consensus = max(arch_counts.values(), key=lambda x: x["count"])
    consensus_arch = consensus["data"]
    
    return {
        "status": "ok",
        "hidden_dim": consensus_arch["hidden_dim"],
        "intermediate_dim": consensus_arch.get("intermediate_dim"),
        "num_layers": consensus_arch.get("num_layers"),
        "num_heads": consensus_arch["num_heads"],
        "num_kv_heads": consensus_arch["num_kv_heads"],
        "estimated_params": consensus_arch.get("estimated_params"),
        "consensus_peers": consensus["count"],
        "total_peers_sampled": len(architectures),
    }


# =============================================================================
# Quorum Support for Native NeuroShard Architecture
# =============================================================================

class QuorumAnnouncement(BaseModel):
    """Announcement of a quorum for network discovery."""
    quorum_id: str
    speed_tier: str  # tier1-tier5
    initiator_endpoint: str
    members: List[Dict]  # List of {node_id, endpoint, layer_range, role}
    lifecycle: str  # forming, active, renewing, dissolving
    total_batches: int = 0
    session_start: float = 0.0
    session_end: float = 0.0


class QuorumInfo(BaseModel):
    """Information about a quorum."""
    quorum_id: str
    speed_tier: str
    initiator_endpoint: str
    member_count: int
    lifecycle: str
    last_seen: float
    total_batches: int = 0


@app.post("/quorums/announce")
async def announce_quorum(quorum: QuorumAnnouncement):
    """
    Announce a quorum to the tracker for discovery.
    
    Nodes use this to register their quorum so other nodes can:
    - Find quorums to join
    - Route inference requests
    - Monitor network health
    """
    now = time.time()
    
    import sqlite3
    import json
    
    # Ensure quorums table exists
    with sqlite3.connect(db._sqlite_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quorums (
                quorum_id TEXT PRIMARY KEY,
                speed_tier TEXT,
                initiator_endpoint TEXT,
                members TEXT,
                lifecycle TEXT,
                total_batches INTEGER,
                session_start REAL,
                session_end REAL,
                last_seen REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quorum_tier ON quorums(speed_tier)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_quorum_lifecycle ON quorums(lifecycle)")
        
        # Upsert quorum
        conn.execute("""
            INSERT OR REPLACE INTO quorums
            (quorum_id, speed_tier, initiator_endpoint, members, lifecycle, 
             total_batches, session_start, session_end, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            quorum.quorum_id,
            quorum.speed_tier,
            quorum.initiator_endpoint,
            json.dumps(quorum.members),
            quorum.lifecycle,
            quorum.total_batches,
            quorum.session_start,
            quorum.session_end,
            now
        ))
    
    return {"status": "registered", "quorum_id": quorum.quorum_id}


@app.get("/quorums")
async def get_quorums(
    speed_tier: Optional[str] = None,
    lifecycle: Optional[str] = None,
    limit: int = 50
) -> Dict:
    """
    Get list of active quorums.
    
    Args:
        speed_tier: Filter by speed tier (tier1-tier5)
        lifecycle: Filter by lifecycle state (forming, active)
        limit: Max number of quorums to return
    
    Returns:
        List of quorums sorted by activity
    """
    now = time.time()
    
    import sqlite3
    import json
    
    try:
        with sqlite3.connect(db._sqlite_path) as conn:
            # Check if table exists
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='quorums'"
            ).fetchone()
            
            if not table_check:
                return {"quorums": [], "total": 0}
            
            query = "SELECT * FROM quorums WHERE last_seen > ?"
            params = [now - 120]  # 2 minute timeout
            
            if speed_tier:
                query += " AND speed_tier = ?"
                params.append(speed_tier)
            
            if lifecycle:
                query += " AND lifecycle = ?"
                params.append(lifecycle)
            
            query += " ORDER BY total_batches DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            quorums = []
            for row in cursor:
                quorums.append({
                    "quorum_id": row[0],
                    "speed_tier": row[1],
                    "initiator_endpoint": row[2],
                    "members": json.loads(row[3]) if row[3] else [],
                    "lifecycle": row[4],
                    "total_batches": row[5],
                    "session_start": row[6],
                    "session_end": row[7],
                    "last_seen": row[8]
                })
        
        return {"quorums": quorums, "total": len(quorums)}
    
    except Exception as e:
        logging.error(f"Error getting quorums: {e}")
        return {"quorums": [], "total": 0, "error": str(e)}


@app.get("/quorums/{quorum_id}")
async def get_quorum(quorum_id: str) -> Dict:
    """Get details of a specific quorum."""
    import sqlite3
    import json
    
    try:
        with sqlite3.connect(db._sqlite_path) as conn:
            row = conn.execute(
                "SELECT * FROM quorums WHERE quorum_id = ?",
                (quorum_id,)
            ).fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Quorum not found")
            
            return {
                "quorum_id": row[0],
                "speed_tier": row[1],
                "initiator_endpoint": row[2],
                "members": json.loads(row[3]) if row[3] else [],
                "lifecycle": row[4],
                "total_batches": row[5],
                "session_start": row[6],
                "session_end": row[7],
                "last_seen": row[8]
            }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/quorums/{quorum_id}")
async def remove_quorum(quorum_id: str):
    """Remove a quorum (called when dissolved)."""
    import sqlite3
    
    with sqlite3.connect(db._sqlite_path) as conn:
        conn.execute("DELETE FROM quorums WHERE quorum_id = ?", (quorum_id,))
    
    return {"status": "removed", "quorum_id": quorum_id}


@app.get("/network/status")
async def get_network_status():
    """
    Get overall network status for the native NeuroShard architecture.
    
    Returns comprehensive network health and training progress.
    """
    now = time.time()
    
    import sqlite3
    import json
    
    # Get peer stats
    count, tps, lat = await db.get_stats(now)
    
    # Get quorum stats
    quorum_stats = {
        "total": 0,
        "active": 0,
        "forming": 0,
        "by_tier": {}
    }
    
    try:
        with sqlite3.connect(db._sqlite_path) as conn:
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='quorums'"
            ).fetchone()
            
            if table_check:
                # Total quorums
                quorum_stats["total"] = conn.execute(
                    "SELECT COUNT(*) FROM quorums WHERE last_seen > ?",
                    (now - 120,)
                ).fetchone()[0]
                
                # Active quorums
                quorum_stats["active"] = conn.execute(
                    "SELECT COUNT(*) FROM quorums WHERE lifecycle = 'active' AND last_seen > ?",
                    (now - 120,)
                ).fetchone()[0]
                
                # Forming quorums
                quorum_stats["forming"] = conn.execute(
                    "SELECT COUNT(*) FROM quorums WHERE lifecycle = 'forming' AND last_seen > ?",
                    (now - 120,)
                ).fetchone()[0]
                
                # By tier
                tier_cursor = conn.execute("""
                    SELECT speed_tier, COUNT(*) 
                    FROM quorums 
                    WHERE last_seen > ? 
                    GROUP BY speed_tier
                """, (now - 120,))
                
                for row in tier_cursor:
                    quorum_stats["by_tier"][row[0]] = row[1]
                
                # Total batches across all quorums
                total_batches = conn.execute(
                    "SELECT SUM(total_batches) FROM quorums WHERE last_seen > ?",
                    (now - 120,)
                ).fetchone()[0] or 0
                quorum_stats["total_batches"] = total_batches
    
    except Exception as e:
        logging.warning(f"Could not get quorum stats: {e}")
    
    return {
        "status": "healthy" if count > 0 else "no_nodes",
        "timestamp": now,
        "nodes": {
            "active": count,
            "total_tps": int(tps),
            "avg_latency_ms": int(lat)
        },
        "quorums": quorum_stats,
        "network": {
            "architecture": "NeuroShard",
            "training_mode": "quorum-based",
            "sync_protocol": "DiLoCo"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
