#!/usr/bin/env python3
"""Test the new wallet API endpoints"""

import requests
import json

BASE_URL = "http://localhost:8090"

def test_signup():
    print("1. Testing signup...")
    import time
    email = f"user{int(time.time())}@example.com"
    resp = requests.post(f"{BASE_URL}/signup", json={
        "email": email,
        "password": "testpass123"
    })
    print(f"   Email: {email}")
    print(f"   Status: {resp.status_code}")
    data = resp.json()
    print(f"   Response: {json.dumps(data, indent=2)}")
    return data.get("email") if resp.status_code == 200 else False

def test_login(email):
    print("\n2. Testing login...")
    resp = requests.post(f"{BASE_URL}/token", data={
        "username": email,
        "password": "testpass123"
    })
    print(f"   Status: {resp.status_code}")
    data = resp.json()
    print(f"   Token received: {data.get('access_token', 'NONE')[:50]}...")
    return data.get("access_token")

def test_wallet_create(token):
    print("\n3. Testing wallet creation...")
    resp = requests.post(f"{BASE_URL}/wallet/create", headers={
        "Authorization": f"Bearer {token}"
    })
    print(f"   Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"   Mnemonic: {data.get('mnemonic', 'NONE')}")
        print(f"   Node ID: {data.get('node_id', 'NONE')}")
        print(f"   Wallet ID: {data.get('wallet_id', 'NONE')}")
        return data
    else:
        print(f"   Error: {resp.text}")
        return None

def test_wallet_info(token):
    print("\n4. Testing wallet info...")
    resp = requests.get(f"{BASE_URL}/users/me/wallet", headers={
        "Authorization": f"Bearer {token}"
    })
    print(f"   Status: {resp.status_code}")
    print(f"   Response: {json.dumps(resp.json(), indent=2)}")

def test_wallet_connect(token, mnemonic):
    print("\n5. Testing wallet connect (import)...")
    # First, create a new user
    resp = requests.post(f"{BASE_URL}/signup", json={
        "email": "import@example.com",
        "password": "testpass123"
    })
    
    # Login as new user
    resp = requests.post(f"{BASE_URL}/token", data={
        "username": "import@example.com",
        "password": "testpass123"
    })
    new_token = resp.json()["access_token"]
    
    # Import wallet with mnemonic
    resp = requests.post(f"{BASE_URL}/wallet/connect", 
        headers={"Authorization": f"Bearer {new_token}"},
        json={"secret": mnemonic}
    )
    print(f"   Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"   Response: {json.dumps(resp.json(), indent=2)}")
    else:
        print(f"   Error: {resp.text}")

if __name__ == "__main__":
    print("=== Testing NeuroShard Wallet API ===\n")
    
    # Test signup
    result = test_signup()
    if not result:
        print("❌ Signup failed!")
        exit(1)
    email = result
    
    # Test login
    token = test_login(email)
    if not token:
        print("❌ Login failed!")
        exit(1)
    
    # Test wallet creation
    wallet = test_wallet_create(token)
    if not wallet:
        print("❌ Wallet creation failed!")
        exit(1)
    
    # Test wallet info
    test_wallet_info(token)
    
    # Test wallet import
    if wallet and 'mnemonic' in wallet:
        test_wallet_connect(token, wallet['mnemonic'])
    
    print("\n✅ All tests completed!")

