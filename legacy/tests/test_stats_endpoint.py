#!/usr/bin/env python3
"""
Test the /api/stats endpoint to diagnose slow response times.
Run this on the machine where the node is running.

Usage:
    python test_stats_endpoint.py [port]
"""

import sys
import time
import requests

def test_stats_endpoint(port=8000):
    """Test the /api/stats endpoint and measure response time."""
    url = f"http://localhost:{port}/api/stats"
    
    print(f"Testing {url}...")
    print("-" * 60)
    
    # Test connection
    print("1. Testing connection...")
    try:
        start = time.time()
        resp = requests.get(url, timeout=10)
        elapsed = time.time() - start
        
        print(f"   ✓ Connected in {elapsed:.2f}s")
        print(f"   Status: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"\n2. Response data:")
            for key, value in data.items():
                if isinstance(value, (int, float, str, bool)):
                    print(f"   {key}: {value}")
                elif isinstance(value, list):
                    print(f"   {key}: {len(value)} items")
                else:
                    print(f"   {key}: {type(value).__name__}")
            
            print(f"\n3. Full JSON response:")
            import json
            print(json.dumps(data, indent=2))
        else:
            print(f"   ✗ Error: {resp.text}")
            
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Connection refused - is the node running on port {port}?")
    except requests.exceptions.Timeout:
        print(f"   ✗ Request timed out after 10 seconds")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("-" * 60)
    
    # Multiple rapid requests to check consistency
    print("\n4. Testing 5 rapid requests...")
    times = []
    for i in range(5):
        try:
            start = time.time()
            resp = requests.get(url, timeout=10)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"   Request {i+1}: {elapsed:.2f}s - Status {resp.status_code}")
        except Exception as e:
            print(f"   Request {i+1}: FAILED - {e}")
        time.sleep(0.5)
    
    if times:
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        print(f"\n   Average: {avg_time:.2f}s")
        print(f"   Min: {min_time:.2f}s, Max: {max_time:.2f}s")
        
        if avg_time > 2:
            print(f"\n   ⚠️  WARNING: Average response time ({avg_time:.2f}s) exceeds 2s timeout!")
            print(f"   The GUI was timing out because the default was 2s.")
            print(f"   This has been increased to 5s in the latest version.")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    test_stats_endpoint(port)

