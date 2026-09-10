"""Rebuildable explorer index of accepted training receipts; never consensus state."""
import json
import math
from pathlib import Path
import sqlite3
import threading


class TrainingHistory:
    def __init__(self, path, genesis_sha256):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(Path(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (id INTEGER PRIMARY KEY, genesis TEXT, height INTEGER, hash TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS training (height INTEGER PRIMARY KEY, round INTEGER UNIQUE, record TEXT)")
        self.db.execute("INSERT OR IGNORE INTO meta VALUES (1, ?, 0, '')", (genesis_sha256,))
        if self.db.execute("SELECT genesis FROM meta WHERE id=1").fetchone()[0] != genesis_sha256:
            self.db.close()
            raise ValueError("Explorer index belongs to a different genesis")
        self.db.commit()
        self.error = None

    def cursor(self):
        with self.lock:
            return self.db.execute("SELECT height, hash FROM meta WHERE id=1").fetchone()

    def append(self, block):
        with self.lock, self.db:
            height, previous_hash = self.cursor()
            if block["height"] != height + 1 or (height and block["previous_hash"] != previous_hash):
                raise ValueError("Explorer history is not contiguous; rebuild the local index")
            round_number = self.db.execute("SELECT COALESCE(MAX(round), 0) FROM training").fetchone()[0]
            for tx in block["transactions"]:
                body = tx["body"]
                receipts = body.get("receipts", [])
                if tx["code"] != 0 or tx["kind"] != "submit" or len(receipts) != 2:
                    continue
                stages = [r["body"] for r in receipts]
                if [r.get("stage") for r in stages] != [0, 1]:
                    continue
                loss_hex = stages[1]["loss_hex"]
                loss = float.fromhex(loss_hex)
                if not math.isfinite(loss) or stages[0].get("loss_hex", loss_hex) != loss_hex:
                    raise ValueError("Inconsistent accepted training receipt")
                round_number += 1
                record = {"height": block["height"], "block_hash": block["hash"], "time": block["time"],
                          "round": round_number, "transaction_hash": tx["hash"], "task_id": body["task_id"],
                          "model_root": body["result_root"], "loss": loss, "loss_hex": loss_hex,
                          "workers": [r["public_key"] for r in receipts]}
                self.db.execute("INSERT INTO training VALUES (?, ?, ?)",
                                (block["height"], round_number, json.dumps(record)))
            self.db.execute("UPDATE meta SET height=?, hash=? WHERE id=1", (block["height"], block["hash"]))
            self.error = None

    def page(self, limit=50, before=2 ** 63 - 1):
        if not 1 <= limit <= 100 or not 1 <= before < 2 ** 63:
            raise ValueError("History limit must be 1–100 and cursor a positive block height")
        with self.lock:
            rows = self.db.execute("SELECT record FROM training WHERE height < ? ORDER BY height DESC LIMIT ?",
                                   (before, limit)).fetchall()
            records = [json.loads(row[0]) for row in rows]
            return {"records": records, "indexed_height": self.cursor()[0],
                    "indexed_rounds": self.db.execute("SELECT COUNT(*) FROM training").fetchone()[0],
                    "next_before": records[-1]["height"] if len(records) == limit else None,
                    "error": self.error, "metric": "Training minibatch cross-entropy before each accepted update; not held-out quality"}

    def follow(self, gateway, stop):
        while not stop.is_set():
            try:
                target = gateway.summary()["height"]
                height, block_hash = self.cursor()
                if height > target:
                    raise ValueError("Full node is behind the explorer index")
                if height and gateway.block(height)["hash"] != block_hash:
                    raise ValueError("Indexed history differs from the full node")
                # Limit each batch; catch-up never runs in a public request thread.
                for next_height in range(height + 1, min(target, height + 100) + 1):
                    if stop.is_set():
                        return
                    self.append(gateway.block(next_height))
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
                self.error = "Index paused: waiting for consistent full-node history"
            stop.wait(1)

    def close(self):
        with self.lock:
            self.db.close()
