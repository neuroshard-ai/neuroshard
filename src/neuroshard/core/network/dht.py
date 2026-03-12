import threading
import time
import bisect
from typing import List, Tuple, Dict

K_BUCKET_SIZE = 20
ID_BITS = 160

class Node:
    def __init__(self, node_id: int, ip: str, port: int):
        self.id = node_id
        self.ip = ip
        self.port = port
        self.last_seen = time.time()

    def __eq__(self, other):
        return self.id == other.id

    def __repr__(self):
        return f"<Node {hex(self.id)[:6]}... {self.ip}:{self.port}>"

class KBucket:
    def __init__(self, lower: int, upper: int):
        self.range = (lower, upper)
        self.nodes: List[Node] = []
        self.lock = threading.Lock()

    def add_node(self, node: Node) -> bool:
        """
        Add node to bucket. 
        Returns True if added/updated, False if bucket full and needs splitting (not handled here).
        """
        with self.lock:
            # Check if node exists
            for i, n in enumerate(self.nodes):
                if n.id == node.id:
                    # Move to tail (most recently seen)
                    self.nodes.pop(i)
                    self.nodes.append(node)
                    return True
            
            if len(self.nodes) < K_BUCKET_SIZE:
                self.nodes.append(node)
                return True
            else:
                # Bucket full
                return False

    def get_nodes(self) -> List[Node]:
        with self.lock:
            return list(self.nodes)

class RoutingTable:
    def __init__(self, local_node: Node):
        self.local_node = local_node
        self.buckets: List[KBucket] = [KBucket(0, 2**ID_BITS)]
        self.lock = threading.Lock()

    def _bucket_index(self, node_id: int) -> int:
        # Linear scan for simplicity, can be optimized with Trie or bit logs
        for i, b in enumerate(self.buckets):
            if b.range[0] <= node_id < b.range[1]:
                return i
        return -1

    def add_contact(self, node: Node):
        if node.id == self.local_node.id:
            return

        with self.lock:
            # Iterative attempt to add node, handling splits
            while True:
                index = self._bucket_index(node.id)
                if index == -1: # Should not happen
                    return
                    
                bucket = self.buckets[index]
                
                if bucket.add_node(node):
                    return # Success
                
                # Bucket is full, try to split
                # Only split if local node falls in this bucket range OR (relaxed) depth is small
                can_split = (bucket.range[0] <= self.local_node.id < bucket.range[1]) or (len(self.buckets) < ID_BITS)
                
                if can_split:
                    self._split_bucket(index)
                    # Loop continues to try adding again in the new bucket structure
                else:
                    # Cannot split, drop node (or ping oldest in full impl)
                    return

    def _split_bucket(self, index: int):
        # Assumes lock is held by caller
        bucket = self.buckets[index]
        lower, upper = bucket.range
        mid = lower + (upper - lower) // 2
        
        left = KBucket(lower, mid)
        right = KBucket(mid, upper)
        
        # Redistribute nodes
        for n in bucket.nodes:
            if n.id < mid:
                left.add_node(n)
            else:
                right.add_node(n)
                
        # Replace old bucket with the two new ones
        self.buckets[index] = left
        self.buckets.insert(index + 1, right)

    def get_all_nodes(self) -> List[Node]:
        """Return all known nodes in the routing table."""
        with self.lock:
            all_nodes = []
            for b in self.buckets:
                all_nodes.extend(b.nodes)
            return all_nodes

    def find_closest(self, target_id: int, k: int = K_BUCKET_SIZE) -> List[Node]:
        """Find k nodes closest to target_id (XOR metric)"""
        with self.lock:
            all_nodes = []
            for b in self.buckets:
                all_nodes.extend(b.nodes)
        
        # Sort by XOR distance
        all_nodes.sort(key=lambda n: n.id ^ target_id)
        return all_nodes[:k]
