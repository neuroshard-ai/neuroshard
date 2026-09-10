"""
NeuroShard Client

A client for interacting with the NeuroShard network.
Connects to a node and uses the collective NeuroLLM for text generation.

Usage:
    python client.py --prompt "Hello world" --tokens 20
"""

import requests
import argparse
import sys
import uuid
import random
import time


def run_client(prompt_text: str, max_new_tokens: int = 20, node_url: str = None):
    """
    Connect to a NeuroShard node and generate text.
    
    Args:
        prompt_text: The prompt to generate from
        max_new_tokens: Maximum number of tokens to generate
        node_url: URL of the node to connect to (auto-discover if None)
    """
    # Find a node to connect to
    if node_url is None:
        print("🔍 Finding nodes from NeuroShard Network...")
        try:
            tracker_url = "https://neuroshard.com/api/tracker"
            peers_resp = requests.get(f"{tracker_url}/peers", params={"limit": 10}, timeout=5)
            if peers_resp.status_code == 200:
                peers = peers_resp.json()
                if peers:
                    node_url = random.choice([p['url'] for p in peers if 'url' in p])
                    print(f"✅ Found {len(peers)} nodes. Connecting to: {node_url}")
                else:
                    print("⚠️ No nodes found in network. Using localhost.")
                    node_url = "http://localhost:8000"
            else:
                print("⚠️ Failed to contact tracker. Using localhost.")
                node_url = "http://localhost:8000"
        except Exception as e:
            print(f"⚠️ Tracker error ({e}). Using localhost.")
            node_url = "http://localhost:8000"
    
    print(f"🌍 Connecting to: {node_url}")
    print(f"📝 Prompt: '{prompt_text}'")
    print(f"🎯 Max tokens: {max_new_tokens}")
    
    # Generate text
    print("\n🚀 Generating...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{node_url}/generate_text",
            json={
                "prompt": prompt_text,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.8
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if "error" in data:
                print(f"❌ Error: {data['error']}")
                return
            
            generated_text = data.get("text", "")
            elapsed = time.time() - start_time
            
            print(f"\n{'='*60}")
            print(f"OUTPUT: {generated_text}")
            print(f"{'='*60}")
            print(f"\n✅ Generated in {elapsed:.2f}s")
            
            # Show node info
            my_layers = data.get("my_layers", [])
            training_rounds = data.get("total_training_rounds", 0)
            note = data.get("note", "")
            
            print(f"\n📊 Node Info:")
            print(f"   Layers held: {len(my_layers)}")
            print(f"   Training rounds: {training_rounds}")
            if note:
                print(f"   Note: {note}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {node_url}")
        print("   Make sure a NeuroShard node is running.")
    except Exception as e:
        print(f"❌ Error: {e}")


def get_network_stats(tracker_url: str = "https://neuroshard.com/api/tracker"):
    """Get overall network statistics."""
    print("📊 NeuroShard Network Stats")
    print("="*60)
    
    try:
        # Get peers
        resp = requests.get(f"{tracker_url}/peers", params={"limit": 1000}, timeout=5)
        if resp.status_code == 200:
            peers = resp.json()
            print(f"Total nodes: {len(peers)}")
            
            # Calculate total capacity
            total_layers = 0
            for peer in peers:
                shard_range = peer.get("shard_range", "0-0")
                try:
                    start, end = map(int, shard_range.split("-"))
                    total_layers += (end - start + 1)
                except:
                    pass
            
            print(f"Total layer coverage: {total_layers}")
            
        # Get network stats
        resp = requests.get(f"{tracker_url}/stats", timeout=5)
        if resp.status_code == 200:
            stats = resp.json()
            print(f"Total inferences: {stats.get('total_inferences', 0)}")
            print(f"Total training rounds: {stats.get('total_training_rounds', 0)}")
            
    except Exception as e:
        print(f"Error getting stats: {e}")


def contribute_data(node_url: str, text: str):
    """Contribute training data to the network."""
    print(f"📤 Contributing training data to {node_url}")
    
    try:
        response = requests.post(
            f"{node_url}/contribute_data",
            json={"text": text},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Contributed {data.get('tokens_added', 0)} tokens")
            print(f"   Buffer size: {data.get('buffer_size', 0)} samples")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroShard Client")
    parser.add_argument("--prompt", type=str, default="The future of AI is",
                        help="Prompt text for generation")
    parser.add_argument("--tokens", type=int, default=20,
                        help="Maximum tokens to generate")
    parser.add_argument("--node", type=str, default=None,
                        help="Node URL (auto-discover if not specified)")
    parser.add_argument("--stats", action="store_true",
                        help="Show network statistics")
    parser.add_argument("--contribute", type=str, default=None,
                        help="Contribute training data")
    
    args = parser.parse_args()
    
    if args.stats:
        get_network_stats()
    elif args.contribute:
        node_url = args.node or "http://localhost:8000"
        contribute_data(node_url, args.contribute)
    else:
        run_client(args.prompt, args.tokens, args.node)
