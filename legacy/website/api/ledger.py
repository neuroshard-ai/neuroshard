"""
Ledger Explorer API Endpoints
Provides blockchain-explorer-like functionality for NeuroShard's distributed ledger
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict
from datetime import datetime
import sqlite3
import os
import json
import requests
import hashlib

from pydantic import BaseModel
from neuroshard.core.economics import (
    calculate_stake_multiplier,
    is_valid_stake_amount,
    is_valid_stake_duration,
    MIN_STAKE_AMOUNT,
    MAX_STAKE_AMOUNT,
    MIN_STAKE_DURATION_DAYS,
    MAX_STAKE_DURATION_DAYS,
)

router = APIRouter(prefix="/api/ledger", tags=["ledger"])


def derive_ecdsa_node_id(token: str) -> str:
    """
    Derive ECDSA node_id from token.
    
    This matches the derivation in neuroshard/core/crypto.py:
    1. private_key = SHA256(token)
    2. public_key = ECDSA_derive(private_key) on secp256k1
    3. node_id = SHA256(public_key)[:32]
    """
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
    
    # Derive private key from token
    private_key_bytes = hashlib.sha256(token.encode()).digest()
    
    # Create ECDSA private key
    private_key = ec.derive_private_key(
        int.from_bytes(private_key_bytes, 'big'),
        ec.SECP256K1(),
        default_backend()
    )
    
    # Get compressed public key
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.CompressedPoint
    )
    
    # node_id = SHA256(public_key)[:32]
    return hashlib.sha256(public_key_bytes).hexdigest()[:32]

# Path to ledger database (nodes store their local ledger here)
LEDGER_DB_PATH = os.getenv("LEDGER_DB_PATH", "node_ledger.db")
LEDGER_DATA_DIR = os.getenv("LEDGER_DATA_DIR", "/data")
TRACKER_URL = os.getenv("TRACKER_URL", "http://tracker:3000")
OBSERVER_URL = os.getenv("OBSERVER_URL", "http://observer:8001")

def get_ledger_connection():
    """
    Get connection to the ledger database.
    
    PRIORITY ORDER:
    1. LEDGER_DB_PATH env var (explicit override)
    2. observer_ledger.db (PREFERRED - has global view of all nodes)
    3. node_ledger.db (fallback - only local node's view)
    
    For website/backend, observer_ledger.db is the correct choice because
    it maintains the authoritative global state from all PoNW proofs.
    """
    # 1. Try the specific file path from ENV first (explicit override)
    if os.path.exists(LEDGER_DB_PATH) and LEDGER_DB_PATH != "node_ledger.db":
        return sqlite3.connect(LEDGER_DB_PATH, check_same_thread=False)
    
    # 2. PREFER observer_ledger.db (global view from observer node)
    observer_ledger = os.path.join(LEDGER_DATA_DIR, "observer_ledger.db")
    if os.path.exists(observer_ledger):
        print(f"Ledger Explorer: Using observer ledger (authoritative): {observer_ledger}")
        return sqlite3.connect(observer_ledger, check_same_thread=False)
    
    # 3. Fallback to node_ledger.db (local view only)
    node_ledger = os.path.join(LEDGER_DATA_DIR, "node_ledger.db")
    if os.path.exists(node_ledger):
        print(f"Ledger Explorer: Using node ledger (local only): {node_ledger}")
        return sqlite3.connect(node_ledger, check_same_thread=False)
    
    # 4. Last resort - check current working directory
    if os.path.exists("node_ledger.db"):
        return sqlite3.connect("node_ledger.db", check_same_thread=False)
        
    return None


def query_observer_dht(endpoint: str, params: dict = None) -> Optional[dict]:
    """
    Query the observer's DHT-based ledger API.
    
    The observer maintains a truly decentralized view of the network via DHT.
    """
    try:
        url = f"{OBSERVER_URL}/api/ledger/{endpoint}"
        resp = requests.get(url, params=params, timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        print(f"Observer DHT query failed: {e}")
        return None

def get_stakes_from_tracker() -> Dict[str, float]:
    """Fetch all stakes from the tracker."""
    try:
        resp = requests.get(f"{TRACKER_URL}/stakes", timeout=2)
        if resp.status_code == 200:
            stakes = resp.json()
            # Map url or node_token to amount? The tracker stores stakes by URL.
            # But ledger stores by node_id.
            # We need a mapping. For now, we can't easily map URL -> Node ID without more info.
            # However, usually the node ID is derived from the token, and the tracker knows tokens.
            # The /stakes endpoint returns {url, amount, slashed}.
            # We need to cross-reference with /peers to get token/node_id.
            
            # Let's fetch peers too to map URL -> Token -> NodeID
            peers_resp = requests.get(f"{TRACKER_URL}/peers?limit=1000", timeout=2)
            if peers_resp.status_code == 200:
                peers = peers_resp.json()
                url_to_token = {p['url']: p.get('node_token') for p in peers}
                
                import hashlib
                token_to_id = {}
                for url, token in url_to_token.items():
                    if token:
                        # New format: first 32 hex chars of SHA256(token)
                        node_id = hashlib.sha256(token.encode()).hexdigest()[:32]
                        token_to_id[url] = node_id
                
                stake_map = {}
                for s in stakes:
                    node_id = token_to_id.get(s['url'])
                    if node_id:
                        stake_map[node_id] = s['amount']
                return stake_map
        return {}
    except Exception as e:
        print(f"Error fetching stakes: {e}")
        return {}

@router.get("/epochs")
async def get_epochs(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    Get recent epochs (time-based groupings of PoNW proofs).
    Each epoch represents a 60-second window where nodes submit their PoNW proofs.
    Similar to 'blocks' in blockchain, but time-based rather than hash-based.
    """
    conn = get_ledger_connection()
    if not conn:
        return {"epochs": [], "total": 0, "genesis_epoch": None}
    
    try:
        cursor = conn.cursor()
        
        # Get genesis epoch (first epoch) for relative numbering
        cursor.execute("""
            SELECT MIN(CAST(timestamp / 60 AS INTEGER)) 
            FROM proof_history 
            WHERE signature != 'GENESIS_BLOCK'
        """)
        genesis_epoch_id = cursor.fetchone()[0] or 0
        
        # Group proofs by epoch (60-second windows)
        cursor.execute("""
            SELECT 
                CAST(timestamp / 60 AS INTEGER) as epoch_id,
                MIN(timestamp) as epoch_start,
                MAX(timestamp) as epoch_end,
                COUNT(*) as proof_count,
                COUNT(DISTINCT node_id) as unique_nodes,
                SUM(CASE WHEN tokens_processed > 0 THEN 1 ELSE 0 END) as inference_proofs,
                SUM(CASE WHEN training_batches > 0 THEN 1 ELSE 0 END) as training_proofs,
                SUM(tokens_processed) as total_tokens_processed,
                SUM(training_batches) as total_training_batches,
                SUM(uptime_seconds) as total_uptime,
                SUM(reward_amount) as total_rewards
            FROM proof_history
            WHERE signature != 'GENESIS_BLOCK'
            GROUP BY epoch_id
            ORDER BY epoch_id DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))
        
        epochs = []
        for row in cursor.fetchall():
            epoch_id, start, end, proof_count, unique_nodes, inference_proofs, training_proofs, total_tokens, total_batches, total_uptime, total_rewards = row
            
            # Calculate relative epoch number (1-based from genesis)
            relative_number = epoch_id - genesis_epoch_id + 1
            
            epochs.append({
                "epoch_id": epoch_id,
                "epoch_number": relative_number,  # Human-readable (1, 2, 3...)
                "absolute_epoch": epoch_id,  # Unix minutes since 1970
                "timestamp_start": start,
                "timestamp_end": end,
                "datetime_start": datetime.fromtimestamp(start).isoformat(),
                "datetime_end": datetime.fromtimestamp(end).isoformat(),
                "proof_count": proof_count,
                "unique_nodes": unique_nodes,
                "inference_proofs": inference_proofs or 0,
                "training_proofs": training_proofs or 0,
                "total_tokens_processed": total_tokens or 0,
                "total_training_batches": total_batches or 0,
                "total_uptime_seconds": total_uptime or 0,
                "total_neuro_minted": round(total_rewards or 0, 6),
                "hash": f"epoch_{epoch_id}"
            })
        
        # Get total count (excluding genesis)
        cursor.execute("SELECT COUNT(DISTINCT CAST(timestamp / 60 AS INTEGER)) FROM proof_history WHERE signature != 'GENESIS_BLOCK'")
        total = cursor.fetchone()[0]
        
        return {
            "epochs": epochs,
            "total": total,
            "limit": limit,
            "offset": offset,
            "genesis_epoch": genesis_epoch_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching epochs: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.get("/epoch/{epoch_id}")
async def get_epoch_details(epoch_id: int):
    """
    Get detailed information about a specific epoch including all proofs.
    """
    conn = get_ledger_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Ledger not available")
    
    try:
        cursor = conn.cursor()
        
        # Get genesis epoch for relative numbering
        cursor.execute("""
            SELECT MIN(CAST(timestamp / 60 AS INTEGER)) 
            FROM proof_history 
            WHERE signature != 'GENESIS_BLOCK'
        """)
        genesis_epoch_id = cursor.fetchone()[0] or 0
        relative_number = epoch_id - genesis_epoch_id + 1
        
        # Get epoch summary
        cursor.execute("""
            SELECT 
                MIN(timestamp) as epoch_start,
                MAX(timestamp) as epoch_end,
                COUNT(*) as proof_count,
                COUNT(DISTINCT node_id) as unique_nodes,
                SUM(tokens_processed) as total_tokens,
                SUM(training_batches) as total_batches,
                SUM(uptime_seconds) as total_uptime,
                SUM(reward_amount) as total_rewards
            FROM proof_history
            WHERE CAST(timestamp / 60 AS INTEGER) = ? AND signature != 'GENESIS_BLOCK'
        """, (epoch_id,))
        
        row = cursor.fetchone()
        if not row or row[0] is None:
            raise HTTPException(status_code=404, detail=f"Epoch {epoch_id} not found")
        
        start, end, proof_count, unique_nodes, total_tokens, total_batches, total_uptime, total_rewards = row
        
        # Get all proofs in this epoch
        cursor.execute("""
            SELECT 
                signature, node_id, proof_type, timestamp, 
                uptime_seconds, tokens_processed, training_batches, 
                data_samples, reward_amount
            FROM proof_history
            WHERE CAST(timestamp / 60 AS INTEGER) = ? AND signature != 'GENESIS_BLOCK'
            ORDER BY timestamp ASC
        """, (epoch_id,))
        
        proofs = []
        for p in cursor.fetchall():
            sig, node, ptype, ts, uptime, tokens, batches, samples, reward = p
            proofs.append({
                "signature": sig,
                "node_id": node,
                "proof_type": ptype or "uptime",
                "timestamp": ts,
                "datetime": datetime.fromtimestamp(ts).isoformat(),
                "uptime_seconds": uptime or 0,
                "tokens_processed": tokens or 0,
                "training_batches": batches or 0,
                "data_samples": samples or 0,
                "reward_neuro": round(reward or 0, 6)
            })
        
        # Get reward breakdown by type
        cursor.execute("""
            SELECT 
                proof_type,
                COUNT(*) as count,
                SUM(reward_amount) as total_reward
            FROM proof_history
            WHERE CAST(timestamp / 60 AS INTEGER) = ? AND signature != 'GENESIS_BLOCK'
            GROUP BY proof_type
        """, (epoch_id,))
        
        reward_breakdown = {}
        for ptype, count, reward in cursor.fetchall():
            reward_breakdown[ptype or "uptime"] = {
                "count": count,
                "total_reward": round(reward or 0, 6)
            }
        
        # Get participating nodes with their rewards
        cursor.execute("""
            SELECT 
                node_id,
                COUNT(*) as proof_count,
                SUM(reward_amount) as total_reward
            FROM proof_history
            WHERE CAST(timestamp / 60 AS INTEGER) = ? AND signature != 'GENESIS_BLOCK'
            GROUP BY node_id
            ORDER BY total_reward DESC
        """, (epoch_id,))
        
        participants = []
        for node, pcount, reward in cursor.fetchall():
            participants.append({
                "node_id": node,
                "proof_count": pcount,
                "reward_neuro": round(reward or 0, 6)
            })
        
        return {
            "epoch_id": epoch_id,
            "epoch_number": relative_number,
            "absolute_epoch": epoch_id,
            "timestamp_start": start,
            "timestamp_end": end,
            "datetime_start": datetime.fromtimestamp(start).isoformat(),
            "datetime_end": datetime.fromtimestamp(end).isoformat(),
            "duration_seconds": round(end - start, 2),
            "summary": {
                "proof_count": proof_count,
                "unique_nodes": unique_nodes,
                "total_tokens_processed": total_tokens or 0,
                "total_training_batches": total_batches or 0,
                "total_uptime_seconds": total_uptime or 0,
                "total_neuro_minted": round(total_rewards or 0, 6)
            },
            "reward_breakdown": reward_breakdown,
            "participants": participants,
            "proofs": proofs
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching epoch: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.get("/wallet/{wallet_id}")
async def get_wallet_balance(
    wallet_id: str,
    include_proofs: bool = Query(False, description="Include recent proof history")
):
    """
    Get balance and account info for a specific wallet.
    
    This is used by new nodes to bootstrap their balance when starting
    with an existing wallet on a new machine.
    
    The wallet_id can be:
    - The first 16 chars of the node_id (wallet identifier)
    - The full node_id (ECDSA-derived from token)
    
    Security: This only returns PUBLIC information. Anyone can query
    any wallet's balance (like Etherscan for Ethereum).
    """
    conn = get_ledger_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Ledger database not available")
    
    try:
        cursor = conn.cursor()
        
        # Search by wallet_id prefix OR full node_id
        # wallet_id is typically the first 16 chars of node_id
        cursor.execute("""
            SELECT node_id, balance, total_earned, total_spent, proof_count, last_proof_time
            FROM balances
            WHERE node_id LIKE ? OR node_id = ?
            ORDER BY balance DESC
            LIMIT 10
        """, (f"{wallet_id}%", wallet_id))
        
        rows = cursor.fetchall()
        
        if not rows:
            # Wallet not found - this is OK for new wallets
            return {
                "found": False,
                "wallet_id": wallet_id,
                "balance": 0.0,
                "total_earned": 0.0,
                "total_spent": 0.0,
                "proof_count": 0,
                "message": "Wallet not found. This is normal for new wallets."
            }
        
        # Aggregate across all node_ids for this wallet
        # (same wallet can have multiple instance IDs on different machines)
        total_balance = 0.0
        total_earned = 0.0
        total_spent = 0.0
        total_proofs = 0
        last_activity = 0
        instances = []
        
        for row in rows:
            node_id, balance, earned, spent, proofs, last_time = row
            total_balance += balance or 0
            total_earned += earned or 0
            total_spent += spent or 0
            total_proofs += proofs or 0
            last_activity = max(last_activity, last_time or 0)
            
            # Get stake for this instance
            cursor.execute("SELECT amount, locked_until FROM stakes WHERE node_id = ?", (node_id,))
            stake_row = cursor.fetchone()
            stake = stake_row[0] if stake_row and stake_row[1] > datetime.utcnow().timestamp() else 0
            
            instances.append({
                "node_id": node_id,
                "balance": round(balance or 0, 6),
                "earned": round(earned or 0, 6),
                "proofs": proofs or 0,
                "stake": round(stake, 2)
            })
        
        result = {
            "found": True,
            "wallet_id": wallet_id,
            "balance": round(total_balance, 6),
            "total_earned": round(total_earned, 6),
            "total_spent": round(total_spent, 6),
            "proof_count": total_proofs,
            "last_activity": datetime.fromtimestamp(last_activity).isoformat() if last_activity else None,
            "instances": instances,  # All machines using this wallet
            "instance_count": len(instances)
        }
        
        # Optionally include recent proofs for verification
        if include_proofs and rows:
            node_ids = [r[0] for r in rows]
            placeholders = ",".join("?" * len(node_ids))
            cursor.execute(f"""
                SELECT node_id, timestamp, reward, signature
                FROM proof_history
                WHERE node_id IN ({placeholders})
                ORDER BY timestamp DESC
                LIMIT 50
            """, node_ids)
            
            result["recent_proofs"] = [
                {
                    "node_id": r[0],
                    "timestamp": r[1],
                    "reward": round(r[2], 6),
                    "signature": r[3][:16] + "..." if r[3] else None
                }
                for r in cursor.fetchall()
            ]
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching wallet: {str(e)}")
    finally:
        conn.close()


@router.get("/proofs")
async def get_proofs(
    node_id: Optional[str] = Query(None, description="Filter by node ID"),
    epoch_id: Optional[int] = Query(None, description="Filter by epoch ID"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    Get PoNW proofs (like transactions in a blockchain explorer).
    Each proof represents a node's claim of work done in a 60-second period.
    """
    conn = get_ledger_connection()
    if not conn:
        return {"proofs": [], "total": 0}
    
    stake_map = get_stakes_from_tracker()

    try:
        cursor = conn.cursor()
        
        # New NEUROLedger format uses uptime_seconds, tokens_processed, training_batches
        # Also includes proof_type and reward_amount (pre-calculated)
        query = """
            SELECT signature, node_id, proof_type, timestamp, uptime_seconds, 
                   tokens_processed, training_batches, reward_amount, received_at 
            FROM proof_history 
            WHERE signature != 'GENESIS_BLOCK'
        """
        params = []
        
        if node_id:
            query += " AND node_id = ?"
            params.append(node_id)
        
        if epoch_id:
            query += " AND CAST(timestamp / 60 AS INTEGER) = ?"
            params.append(epoch_id)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        
        proofs = []
        for row in cursor.fetchall():
            sig, n_id, proof_type, timestamp, uptime_seconds, tokens_processed, training_batches, reward_amount, received_at = row
            
            # Use pre-calculated reward from ledger (already includes all multipliers)
            total_reward = reward_amount or 0.0
            
            # Get stake for display
            stake = stake_map.get(n_id, 0.0)
            multiplier = calculate_stake_multiplier(stake)
            
            proofs.append({
                "signature": sig,
                "node_id": n_id,
                "proof_type": proof_type or "UPTIME",
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "uptime_seconds": uptime_seconds or 0,
                "tokens_processed": tokens_processed or 0,
                "training_batches": training_batches or 0,
                "reward_neuro": round(total_reward, 6),
                "epoch_id": int(timestamp / 60),
                "received_at": received_at,
                "stake_multiplier": round(multiplier, 2)
            })
        
        # Get total count (excluding genesis block)
        count_query = "SELECT COUNT(*) FROM proof_history WHERE signature != 'GENESIS_BLOCK'"
        count_params = []
        if node_id:
            count_query += " AND node_id = ?"
            count_params.append(node_id)
        if epoch_id:
            count_query += " AND CAST(timestamp / 60 AS INTEGER) = ?"
            count_params.append(epoch_id)
        
        cursor.execute(count_query, count_params)
        total = cursor.fetchone()[0]
        
        return {
            "proofs": proofs,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching proofs: {str(e)}")
    finally:
        if conn:
            conn.close()

@router.get("/balances")
async def get_balances(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    include_burn: bool = Query(False, description="Include burn address in results")
):
    """
    Get NEURO token balances for all nodes (like account balances in Etherscan).
    Supports both new NEUROLedger format and legacy format.
    """
    conn = get_ledger_connection()
    if not conn:
        return {"balances": [], "total": 0}
    
    stake_map = get_stakes_from_tracker()

    try:
        cursor = conn.cursor()
        
        # Check which format we're using
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='balances'")
        use_new_format = cursor.fetchone() is not None
        
        if use_new_format:
            # New NEUROLedger format
            where_clause = "" if include_burn else "WHERE node_id NOT LIKE 'BURN_%'"
            
            cursor.execute(f"""
                SELECT node_id, balance, total_earned, total_spent, proof_count, last_proof_time
                FROM balances
                {where_clause}
                ORDER BY balance DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            balances = []
            for row in cursor.fetchall():
                node_id, balance, total_earned, total_spent, proof_count, last_proof_time = row
                
                # Get stake from stakes table
                cursor.execute("SELECT amount, locked_until FROM stakes WHERE node_id = ?", (node_id,))
                stake_row = cursor.fetchone()
                stake = stake_row[0] if stake_row and stake_row[1] > datetime.utcnow().timestamp() else 0
                
                # Calculate multiplier (diminishing returns) - from economics module
                multiplier = calculate_stake_multiplier(stake)
                
                balances.append({
                    "node_id": node_id,
                    "address": node_id[:20] + "..." if len(node_id) > 20 else node_id,
                    "balance_neuro": round(balance, 6),
                    "total_earned": round(total_earned or 0, 6),
                    "total_spent": round(total_spent or 0, 6),
                    "staked_neuro": round(stake, 2),
                    "stake_multiplier": round(multiplier, 2),
                    "proof_count": proof_count or 0,
                    "last_activity": datetime.fromtimestamp(last_proof_time).isoformat() if last_proof_time else None,
                    "last_activity_timestamp": last_proof_time,
                    "is_burn_address": node_id.startswith("BURN_")
                })
            
            count_clause = "" if include_burn else "WHERE node_id NOT LIKE 'BURN_%'"
            cursor.execute(f"SELECT COUNT(*) FROM balances {count_clause}")
            total = cursor.fetchone()[0]
        else:
            # Legacy format
            cursor.execute("""
                SELECT node_id, balance, last_proof_time
                FROM credits
                ORDER BY balance DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            balances = []
            for row in cursor.fetchall():
                node_id, balance, last_proof_time = row
                
                balances.append({
                    "node_id": node_id,
                    "address": node_id[:20] + "..." if len(node_id) > 20 else node_id,
                    "balance_neuro": round(balance, 6),
                    "total_earned": 0,
                    "total_spent": 0,
                    "staked_neuro": stake_map.get(node_id, 0.0),
                    "stake_multiplier": 1.0,
                    "proof_count": 0,
                    "last_activity": datetime.fromtimestamp(last_proof_time).isoformat() if last_proof_time else None,
                    "last_activity_timestamp": last_proof_time,
                    "is_burn_address": False
                })
            
            cursor.execute("SELECT COUNT(*) FROM credits")
            total = cursor.fetchone()[0]
        
        return {
            "balances": balances,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching balances: {str(e)}")
    finally:
        if conn:
            conn.close()

@router.get("/node/{node_id}")
async def get_node_details(node_id: str):
    """
    Get detailed information about a specific node (like an address page in Etherscan).
    """
    conn = get_ledger_connection()
    if not conn:
        raise HTTPException(status_code=404, detail="Ledger database not found")
    
    stake_map = get_stakes_from_tracker()

    try:
        cursor = conn.cursor()
        
        # Get balance from new format first
        cursor.execute("""
            SELECT balance, total_earned, total_spent, proof_count, last_proof_time 
            FROM balances WHERE node_id = ?
        """, (node_id,))
        balance_row = cursor.fetchone()
        
        if not balance_row:
            # Instead of 404, return empty account if not found but valid ID format
            balance = 0.0
            total_earned = 0.0
            total_spent = 0.0
            proof_count_db = 0
            last_proof_time = None
        else:
            balance, total_earned, total_spent, proof_count_db, last_proof_time = balance_row
        
        # Get proof count from history
        cursor.execute("SELECT COUNT(*) FROM proof_history WHERE node_id = ? AND signature != 'GENESIS_BLOCK'", (node_id,))
        proof_count = cursor.fetchone()[0]
        
        # Get total tokens processed
        cursor.execute("SELECT SUM(tokens_processed) FROM proof_history WHERE node_id = ?", (node_id,))
        total_tokens = cursor.fetchone()[0] or 0
        
        # Get total uptime
        cursor.execute("SELECT SUM(uptime_seconds) FROM proof_history WHERE node_id = ?", (node_id,))
        total_uptime = cursor.fetchone()[0] or 0
        
        # Get total training batches
        cursor.execute("SELECT SUM(training_batches) FROM proof_history WHERE node_id = ?", (node_id,))
        total_training_batches = cursor.fetchone()[0] or 0
        
        # Get recent proofs with new format
        cursor.execute("""
            SELECT signature, proof_type, timestamp, uptime_seconds, tokens_processed, training_batches, reward_amount
            FROM proof_history
            WHERE node_id = ? AND signature != 'GENESIS_BLOCK'
            ORDER BY timestamp DESC
            LIMIT 10
        """, (node_id,))
        
        recent_proofs = []
        stake = stake_map.get(node_id, 0.0)
        multiplier = calculate_stake_multiplier(stake)
        
        for row in cursor.fetchall():
            sig, proof_type, timestamp, uptime_seconds, tokens_processed, training_batches, reward_amount = row
            
            recent_proofs.append({
                "signature": sig,
                "proof_type": proof_type or "UPTIME",
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "uptime_seconds": uptime_seconds or 0,
                "tokens_processed": tokens_processed or 0,
                "training_batches": training_batches or 0,
                "reward_neuro": round(reward_amount or 0, 6),
                "epoch_id": int(timestamp / 60)
            })
        
        return {
            "node_id": node_id,
            "address": node_id,
            "balance_neuro": round(balance, 6),
            "total_earned_neuro": round(total_earned or 0, 6),
            "total_spent_neuro": round(total_spent or 0, 6),
            "staked_neuro": stake,
            "stake_multiplier": round(multiplier, 2),
            "last_activity": datetime.fromtimestamp(last_proof_time).isoformat() if last_proof_time else None,
            "proof_count": proof_count,
            "total_tokens_processed": total_tokens,
            "total_training_batches": total_training_batches,
            "total_uptime_hours": round(total_uptime / 3600, 2),
            "recent_proofs": recent_proofs
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching node details: {str(e)}")
    finally:
        if conn:
            conn.close()

@router.get("/proof/{signature}")
async def get_proof_details(signature: str):
    """
    Get detailed information about a specific PoNW proof (like a transaction detail page).
    """
    conn = get_ledger_connection()
    if not conn:
        raise HTTPException(status_code=404, detail="Ledger database not found")
    
    stake_map = get_stakes_from_tracker()

    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT node_id, proof_type, timestamp, uptime_seconds, tokens_processed, 
                   training_batches, data_samples, reward_amount, received_at
            FROM proof_history
            WHERE signature = ?
        """, (signature,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Proof not found")
        
        node_id, proof_type, timestamp, uptime_seconds, tokens_processed, training_batches, data_samples, reward_amount, received_at = row
        
        stake = stake_map.get(node_id, 0.0)
        multiplier = calculate_stake_multiplier(stake)
        
        return {
            "signature": signature,
            "node_id": node_id,
            "proof_type": proof_type or "UPTIME",
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "epoch_id": int(timestamp / 60),
            "uptime_seconds": uptime_seconds or 0,
            "tokens_processed": tokens_processed or 0,
            "training_batches": training_batches or 0,
            "data_samples": data_samples or 0,
            "reward_neuro": round(reward_amount or 0, 6),
            "stake_multiplier": round(multiplier, 2),
            "received_at": received_at,
            "status": "verified"  # All processed proofs passed validation
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching proof details: {str(e)}")
    finally:
        if conn:
            conn.close()

class TransferRequest(BaseModel):
    sender_token: str
    recipient_address: str
    amount: float

@router.post("/transfer")
async def transfer_neuro(req: TransferRequest):
    """Transfer NEURO from one node wallet to another using ECDSA node_id."""
    import time
    
    # 1. Derive ECDSA Sender ID from Token
    sender_id = derive_ecdsa_node_id(req.sender_token)
    
    # 2. Get Ledger Connection
    conn = get_ledger_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Ledger database not available")
    
    try:
        # 3. Atomic Transfer Transaction
        cursor = conn.cursor()
        
        # Check which format we're using
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='balances'")
        use_new_format = cursor.fetchone() is not None
        
        if not use_new_format:
            raise HTTPException(status_code=500, detail="Legacy ledger format not supported for transfers")
        
        # Start Transaction
        conn.execute("BEGIN TRANSACTION")
        
        # Calculate fee and burn (5% fee burn)
        fee = req.amount * 0.05
        burn_amount = fee
        net_amount = req.amount - fee
        
        # Check Balance
        cursor.execute("SELECT balance FROM balances WHERE node_id = ?", (sender_id,))
        row = cursor.fetchone()
        if not row or row[0] < req.amount:
            conn.rollback()
            raise HTTPException(status_code=400, detail=f"Insufficient balance. Have: {row[0] if row else 0}, Need: {req.amount}")
            
        # Deduct from Sender (full amount including fee)
        cursor.execute("""
            UPDATE balances SET 
                balance = balance - ?,
                total_spent = total_spent + ?
            WHERE node_id = ?
        """, (req.amount, req.amount, sender_id))
        
        # Add to Recipient (net amount after fee)
        now = time.time()
        cursor.execute("""
            INSERT INTO balances (node_id, balance, total_earned, created_at) 
            VALUES (?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                balance = balance + ?,
                total_earned = total_earned + ?
        """, (req.recipient_address, net_amount, net_amount, now, net_amount, net_amount))
        
        # Credit burn address
        burn_address = "BURN_0x0000000000000000000000000000000000000000"
        cursor.execute("""
            INSERT INTO balances (node_id, balance, total_earned, created_at) 
            VALUES (?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                balance = balance + ?,
                total_earned = total_earned + ?
        """, (burn_address, burn_amount, burn_amount, now, burn_amount, burn_amount))
        
        # Generate transaction hash
        tx_hash = hashlib.sha256(f"{sender_id}:{req.recipient_address}:{req.amount}:{now}".encode()).hexdigest()[:32]
        
        # Record transaction
        cursor.execute("""
            INSERT INTO transactions 
            (tx_id, from_id, to_id, amount, fee, burn_amount, timestamp, memo, signature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (tx_hash, sender_id, req.recipient_address, net_amount, fee, burn_amount, now, "API Transfer", "api_transfer"))
        
        # Update global stats
        cursor.execute("""
            UPDATE global_stats SET
                total_burned = total_burned + ?,
                total_transferred = total_transferred + ?,
                total_transactions = total_transactions + 1,
                updated_at = ?
            WHERE id = 1
        """, (burn_amount, net_amount, now))
        
        conn.commit()
        
        return {
            "status": "success",
            "sender": sender_id,
            "recipient": req.recipient_address,
            "amount_sent": req.amount,
            "fee": round(fee, 6),
            "burned": round(burn_amount, 6),
            "net_received": round(net_amount, 6),
            "tx_hash": tx_hash
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Transfer failed: {str(e)}")
    finally:
        if conn:
            conn.close()


class StakeRequest(BaseModel):
    sender_token: str
    amount: float
    duration_days: int = 30


class UnstakeRequest(BaseModel):
    sender_token: str


@router.post("/stake")
async def stake_neuro(req: StakeRequest):
    """
    Stake NEURO tokens for a reward multiplier.
    
    Staking provides:
    - 10% bonus per 1000 NEURO staked
    - Tokens locked for specified duration
    - Higher multiplier = more rewards from PoNW
    """
    import hashlib
    import time
    
    # Validate input using centralized economics
    is_valid, error = is_valid_stake_amount(req.amount)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    is_valid, error = is_valid_stake_duration(req.duration_days)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    # Derive ECDSA sender ID
    sender_id = derive_ecdsa_node_id(req.sender_token)
    
    conn = get_ledger_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Ledger database not available")
    
    try:
        cursor = conn.cursor()
        
        # Check for new format
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='balances'")
        if not cursor.fetchone():
            raise HTTPException(status_code=500, detail="Legacy ledger format not supported for staking")
        
        conn.execute("BEGIN TRANSACTION")
        
        # Check balance
        cursor.execute("SELECT balance FROM balances WHERE node_id = ?", (sender_id,))
        row = cursor.fetchone()
        if not row or row[0] < req.amount:
            conn.rollback()
            raise HTTPException(status_code=400, detail=f"Insufficient balance. Have: {row[0] if row else 0}, Need: {req.amount}")
        
        # Deduct from balance
        cursor.execute("""
            UPDATE balances SET 
                balance = balance - ?,
                total_spent = total_spent + ?
            WHERE node_id = ?
        """, (req.amount, req.amount, sender_id))
        
        # Calculate lock time
        lock_until = time.time() + (req.duration_days * 24 * 3600)
        now = time.time()
        
        # Get current stake (if any)
        cursor.execute("SELECT amount FROM stakes WHERE node_id = ?", (sender_id,))
        existing = cursor.fetchone()
        
        if existing:
            # Add to existing stake
            new_amount = existing[0] + req.amount
            cursor.execute("""
                UPDATE stakes SET 
                    amount = ?,
                    locked_until = ?,
                    updated_at = ?
                WHERE node_id = ?
            """, (new_amount, lock_until, now, sender_id))
        else:
            # Create new stake
            cursor.execute("""
                INSERT INTO stakes (node_id, amount, locked_until, updated_at)
                VALUES (?, ?, ?, ?)
            """, (sender_id, req.amount, lock_until, now))
        
        conn.commit()
        
        # Calculate new multiplier (diminishing returns)
        final_stake = (existing[0] if existing else 0) + req.amount
        new_multiplier = calculate_stake_multiplier(final_stake)
        
        return {
            "status": "success",
            "staked_amount": req.amount,
            "total_staked": final_stake,
            "new_multiplier": round(new_multiplier, 2),
            "locked_until": lock_until,
            "duration_days": req.duration_days
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Staking failed: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.post("/unstake")
async def unstake_neuro(req: UnstakeRequest):
    """
    Unstake NEURO tokens and return to balance.
    
    Only works if the stake lock period has expired.
    """
    import time
    
    # Derive ECDSA sender ID
    sender_id = derive_ecdsa_node_id(req.sender_token)
    
    conn = get_ledger_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Ledger database not available")
    
    try:
        cursor = conn.cursor()
        
        # Check for new format
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stakes'")
        if not cursor.fetchone():
            raise HTTPException(status_code=500, detail="Legacy ledger format not supported")
        
        conn.execute("BEGIN TRANSACTION")
        
        # Get current stake
        cursor.execute("SELECT amount, locked_until FROM stakes WHERE node_id = ?", (sender_id,))
        row = cursor.fetchone()
        
        if not row or row[0] <= 0:
            conn.rollback()
            raise HTTPException(status_code=400, detail="No stake found")
        
        stake_amount, locked_until = row
        
        # Check if locked
        if locked_until and locked_until > time.time():
            conn.rollback()
            lock_date = time.strftime('%Y-%m-%d %H:%M', time.localtime(locked_until))
            raise HTTPException(status_code=400, detail=f"Stake locked until {lock_date}")
        
        # Return to balance
        now = time.time()
        cursor.execute("""
            UPDATE balances SET 
                balance = balance + ?,
                total_earned = total_earned + ?
            WHERE node_id = ?
        """, (stake_amount, stake_amount, sender_id))
        
        # Clear stake
        cursor.execute("""
            UPDATE stakes SET 
                amount = 0,
                locked_until = 0,
                updated_at = ?
            WHERE node_id = ?
        """, (now, sender_id))
        
        conn.commit()
        
        return {
            "status": "success",
            "unstaked_amount": stake_amount,
            "message": f"Returned {stake_amount:.4f} NEURO to balance"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Unstaking failed: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.get("/genesis")
async def get_genesis_info():
    """
    Get information about the Genesis Block - proves no pre-mine.
    
    TRANSPARENCY GUARANTEE:
    =======================
    The Genesis Block proves that the ledger started with ZERO supply.
    All NEURO tokens must be earned through verified Proof of Neural Work.
    There is NO pre-allocation, NO founder tokens, NO ICO.
    """
    conn = get_ledger_connection()
    if not conn:
        return {
            "genesis_exists": False,
            "message": "Ledger not initialized"
        }
    
    try:
        cursor = conn.cursor()
        
        # Get genesis block
        cursor.execute("""
            SELECT timestamp, reward_amount, received_at 
            FROM proof_history 
            WHERE signature = 'GENESIS_BLOCK'
        """)
        genesis = cursor.fetchone()
        
        if not genesis:
            return {
                "genesis_exists": False,
                "message": "No genesis block found (older ledger format)"
            }
        
        genesis_time, genesis_reward, received_at = genesis
        
        # Get global stats to show current state
        cursor.execute("""
            SELECT total_minted, total_burned, total_transferred, total_proofs, total_transactions
            FROM global_stats WHERE id = 1
        """)
        stats = cursor.fetchone()
        
        if stats:
            total_minted, total_burned, total_transferred, total_proofs, total_transactions = stats
        else:
            total_minted = total_burned = total_transferred = total_proofs = total_transactions = 0
        
        return {
            "genesis_exists": True,
            "genesis_timestamp": genesis_time,
            "genesis_datetime": datetime.fromtimestamp(genesis_time).isoformat(),
            "genesis_reward": genesis_reward,  # Should be 0.0
            "message": "Ledger initialized with ZERO supply. All tokens earned through PoNW.",
            "transparency_guarantee": {
                "pre_mine": 0.0,
                "founder_allocation": 0.0,
                "ico_tokens": 0.0,
                "total_minted_since_genesis": round(total_minted, 6),
                "total_burned": round(total_burned, 6),
                "circulating_supply": round(total_minted - total_burned, 6),
                "total_proofs_verified": total_proofs,
                "total_transactions": total_transactions
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching genesis info: {str(e)}")
    finally:
        if conn:
            conn.close()

@router.get("/stats")
async def get_ledger_stats():
    """
    Get aggregate statistics about the ledger (like network stats in Etherscan).
    Supports both new NEUROLedger format and legacy format.
    """
    conn = get_ledger_connection()
    if not conn:
        return {
            "total_nodes": 0,
            "total_neuro_supply": 0,
            "total_proofs": 0,
            "total_tokens_processed": 0,
            "total_minted": 0,
            "total_burned": 0,
            "circulating_supply": 0,
            "burn_rate": "5%"
        }
    
    try:
        cursor = conn.cursor()
        
        # Check which format we're using
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='balances'")
        use_new_format = cursor.fetchone() is not None
        
        if use_new_format:
            # New NEUROLedger format
            # Total nodes (excluding burn address)
            cursor.execute("SELECT COUNT(*) FROM balances WHERE node_id NOT LIKE 'BURN_%'")
            total_nodes = cursor.fetchone()[0]
            
            # Get global stats
            cursor.execute("""
                SELECT total_minted, total_burned, total_proofs, total_transactions 
                FROM global_stats WHERE id = 1
            """)
            stats_row = cursor.fetchone()
            
            if stats_row:
                total_minted, total_burned, total_proofs_global, total_transactions = stats_row
                total_minted = total_minted or 0
                total_burned = total_burned or 0
                total_proofs_global = total_proofs_global or 0
            else:
                total_minted = total_burned = total_proofs_global = total_transactions = 0
            
            circulating_supply = total_minted - total_burned
            
            # Total proofs from history
            cursor.execute("SELECT COUNT(*) FROM proof_history")
            total_proofs = cursor.fetchone()[0]
            
            # Total tokens processed
            cursor.execute("SELECT SUM(tokens_processed) FROM proof_history")
            total_tokens = cursor.fetchone()[0] or 0
            
            # Latest epoch
            cursor.execute("SELECT MAX(CAST(timestamp / 60 AS INTEGER)) FROM proof_history")
            latest_epoch = cursor.fetchone()[0]
            
            # Total staked
            cursor.execute("SELECT SUM(amount) FROM stakes WHERE locked_until > ?", (datetime.utcnow().timestamp(),))
            total_staked = cursor.fetchone()[0] or 0
            
            return {
                "total_nodes": total_nodes,
                "total_neuro_supply": round(circulating_supply, 6),
                "total_minted": round(total_minted, 6),
                "total_burned": round(total_burned, 6),
                "circulating_supply": round(circulating_supply, 6),
                "total_staked": round(total_staked, 6),
                "total_proofs": total_proofs,
                "total_transactions": total_transactions or 0,
                "total_tokens_processed": total_tokens,
                "latest_epoch": latest_epoch,
                "burn_rate": "5%",
                "ledger_type": "neuro_ledger",
                "consensus": "proof_of_neural_work"
            }
        else:
            # Legacy format
            cursor.execute("SELECT COUNT(*) FROM credits")
            total_nodes = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(balance) FROM credits")
            total_supply = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM proof_history")
            total_proofs = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(token_count) FROM proof_history")
            total_tokens = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT MAX(CAST(timestamp / 60 AS INTEGER)) FROM proof_history")
            latest_epoch = cursor.fetchone()[0]
            
            return {
                "total_nodes": total_nodes,
                "total_neuro_supply": round(total_supply, 6),
                "total_minted": round(total_supply, 6),
                "total_burned": 0,
                "circulating_supply": round(total_supply, 6),
                "total_staked": 0,
                "total_proofs": total_proofs,
                "total_transactions": 0,
                "total_tokens_processed": total_tokens,
                "latest_epoch": latest_epoch,
                "burn_rate": "5%",
                "ledger_type": "legacy_ledger",
                "consensus": "proof_of_neural_work"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.get("/burn")
async def get_burn_stats():
    """
    Get detailed burn statistics for the deflationary mechanism.
    """
    conn = get_ledger_connection()
    if not conn:
        return {
            "total_burned": 0,
            "burn_rate": "5%",
            "burn_sources": {}
        }
    
    try:
        cursor = conn.cursor()
        
        # Check for new format
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='global_stats'")
        if not cursor.fetchone():
            return {
                "total_burned": 0,
                "burn_rate": "5%",
                "burn_sources": {},
                "note": "Legacy ledger format - no burn tracking"
            }
        
        # Get total burned from global stats
        cursor.execute("SELECT total_minted, total_burned FROM global_stats WHERE id = 1")
        row = cursor.fetchone()
        total_minted = row[0] if row else 0
        total_burned = row[1] if row else 0
        
        # Get burn address balance (should match total_burned)
        cursor.execute("SELECT balance FROM balances WHERE node_id LIKE 'BURN_%'")
        burn_balance = cursor.fetchone()
        burn_balance = burn_balance[0] if burn_balance else 0
        
        # Calculate burn percentage
        burn_percentage = (total_burned / total_minted * 100) if total_minted > 0 else 0
        
        return {
            "total_burned": round(total_burned, 6),
            "total_minted": round(total_minted, 6),
            "circulating_supply": round(total_minted - total_burned, 6),
            "burn_percentage": round(burn_percentage, 4),
            "burn_rate": "5%",
            "burn_address_balance": round(burn_balance, 6),
            "deflationary_effect": f"{burn_percentage:.2f}% of supply burned"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching burn stats: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.get("/transactions")
async def get_transactions(
    node_id: Optional[str] = Query(None, description="Filter by sender or recipient"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Get NEURO transfer transactions (new NEUROLedger format only).
    """
    conn = get_ledger_connection()
    if not conn:
        return {"transactions": [], "total": 0}
    
    try:
        cursor = conn.cursor()
        
        # Check for new format
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions'")
        if not cursor.fetchone():
            return {"transactions": [], "total": 0, "note": "Legacy format - no transaction history"}
        
        query = """
            SELECT tx_id, from_id, to_id, amount, fee, burn_amount, timestamp, memo
            FROM transactions WHERE 1=1
        """
        params = []
        
        if node_id:
            query += " AND (from_id = ? OR to_id = ?)"
            params.extend([node_id, node_id])
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        
        transactions = []
        for row in cursor.fetchall():
            tx_id, from_id, to_id, amount, fee, burn_amount, timestamp, memo = row
            transactions.append({
                "tx_id": tx_id,
                "from_id": from_id,
                "to_id": to_id,
                "amount": round(amount, 6),
                "fee": round(fee, 6),
                "burn_amount": round(burn_amount, 6),
                "timestamp": timestamp,
                "datetime": datetime.fromtimestamp(timestamp).isoformat(),
                "memo": memo or "",
                "type": "burn" if to_id.startswith("BURN_") else "transfer"
            })
        
        # Get total
        count_query = "SELECT COUNT(*) FROM transactions WHERE 1=1"
        count_params = []
        if node_id:
            count_query += " AND (from_id = ? OR to_id = ?)"
            count_params.extend([node_id, node_id])
        
        cursor.execute(count_query, count_params)
        total = cursor.fetchone()[0]
        
        return {
            "transactions": transactions,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transactions: {str(e)}")
    finally:
        if conn:
            conn.close()
