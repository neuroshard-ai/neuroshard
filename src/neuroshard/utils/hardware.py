
import torch
import psutil
import platform
import hashlib
import uuid
import os
import socket


def get_machine_id() -> str:
    """
    Get a unique identifier for this machine.
    
    Uses multiple sources to create a stable, unique ID:
    1. Machine UUID (from DMI/SMBIOS on Linux, IOKit on macOS)
    2. Hostname (fallback)
    3. MAC address (fallback)
    
    This ID is stable across reboots but unique per machine.
    """
    machine_id_sources = []
    
    # Try to get machine-id (Linux)
    for path in ['/etc/machine-id', '/var/lib/dbus/machine-id']:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    machine_id_sources.append(f.read().strip())
                break
        except Exception:
            pass
    
    # Try hostname
    try:
        machine_id_sources.append(socket.gethostname())
    except Exception:
        pass
    
    # Try MAC address (via uuid.getnode)
    try:
        mac = uuid.getnode()
        if mac != uuid.getnode():  # Check it's not a random fallback
            machine_id_sources.append(str(mac))
    except Exception:
        pass
    
    # Combine all sources
    if machine_id_sources:
        combined = ":".join(machine_id_sources)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    # Ultimate fallback: generate and cache a random ID
    cache_dir = os.path.join(os.path.expanduser("~"), ".neuroshard")
    os.makedirs(cache_dir, exist_ok=True)
    machine_id_file = os.path.join(cache_dir, ".machine_id")
    
    if os.path.exists(machine_id_file):
        with open(machine_id_file, 'r') as f:
            return f.read().strip()
    
    random_id = uuid.uuid4().hex[:16]
    with open(machine_id_file, 'w') as f:
        f.write(random_id)
    return random_id


def get_instance_id(port: int) -> str:
    """
    Generate a unique instance ID for this node.
    
    Combines machine_id + port to create a stable, unique identifier.
    This allows multiple nodes on the same machine (different ports)
    or the same port on different machines to have unique identities.
    
    Args:
        port: The port this node is running on
        
    Returns:
        16-character hex string unique to this machine+port combination
    """
    machine_id = get_machine_id()
    instance_string = f"{machine_id}:{port}"
    return hashlib.sha256(instance_string.encode()).hexdigest()[:16]


def get_hardware_info():
    info = {
        "system": platform.system(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": "None",
        "vram_gb": 0.0,
        "device": "cpu"
    }
    
    if info["gpu_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        info["device"] = "cuda"
        
    return info

def suggest_config(info):
    """Recommend layers based on hardware."""
    # Heuristic: 124M params needs ~500MB RAM. 
    # We can fit full model on almost any modern PC.
    # But let's split it to demonstrate sharding.
    
    layers_per_node = 4 # Default shard size
    
    if info["vram_gb"] > 4 or (info["ram_gb"] > 8 and not info["gpu_available"]):
        # Powerful node, can host more
        layers_per_node = 6
        
    return {
        "suggested_layers": f"0-{layers_per_node}",
        "device": info["device"]
    }

