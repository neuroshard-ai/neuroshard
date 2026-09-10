# Installation

Complete guide to installing NeuroShard on different platforms.

## System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 2GB | 8GB+ |
| **Storage** | 1GB | 10GB+ |
| **CPU** | 2 cores | 4+ cores |
| **Python** | 3.9 | 3.11 |
| **Network** | 10 Mbps | 100 Mbps+ |

### GPU Support

NeuroShard automatically detects and uses available GPUs:

| GPU | Support Level |
|-----|--------------|
| NVIDIA CUDA | ✅ Full support (recommended) |
| NVIDIA Jetson (ARM64) | ✅ Full support |
| Apple Metal (M1/M2/M3) | ✅ Full support |
| AMD ROCm | ⚠️ Experimental |
| CPU Only | ✅ Supported (slower) |

## Installation Methods

### Method 1: pip (Recommended)

NeuroShard is published to PyPI as `neuroshard-ai`.

#### Basic Install (CPU)

```bash
# Create a virtual environment (recommended)
python -m venv neuroshard-env
source neuroshard-env/bin/activate  # On Windows: neuroshard-env\Scripts\activate

# Install NeuroShard (without PyTorch)
pip install neuroshard-ai
```

#### With GPU Support (x86 Linux/Windows)

```bash
# Install with GPU support - automatically installs PyTorch from PyPI
pip install neuroshard-ai[gpu]
```

#### With GUI Support (Desktop App)

```bash
# Install with GUI libraries
pip install neuroshard-ai[gui]

# Or install everything (GPU + GUI)
pip install neuroshard-ai[full]
```

### Platform-Specific PyTorch

PyTorch is an **optional dependency** because different platforms need different builds:

#### NVIDIA CUDA (x86 Linux/Windows)

```bash
# Option A: Use the [gpu] extra (simplest)
pip install neuroshard-ai[gpu]

# Option B: Install specific CUDA version first
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install neuroshard-ai
```

#### Apple Silicon (M1/M2/M3/M4)

```bash
# PyTorch from PyPI includes Metal (MPS) support
pip install neuroshard-ai[gpu]
```

#### NVIDIA Jetson (ARM64)

For Jetson Orin, AGX, or other Jetson devices, **install PyTorch from NVIDIA first**:

```bash
# Step 1: Install NVIDIA's PyTorch (JetPack 6.x)
pip install torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60

# Step 2: Install NeuroShard (no [gpu] needed - torch already installed!)
pip install neuroshard-ai

# Step 3: Run with CUDA
neuroshard --token YOUR_TOKEN --device cuda
```

::: tip Why separate steps for Jetson?
Jetson uses ARM64 architecture with a custom CUDA build. NVIDIA provides pre-built PyTorch wheels optimized for Jetson that aren't available on PyPI. By pre-installing torch from NVIDIA, pip sees it's already satisfied and won't try to download an incompatible version.
:::

### Method 2: Docker

Run NeuroShard in a Docker container:

```bash
# Pull the official image
docker pull neuroshard/node:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 -p 9000:9000 \
  -e NEUROSHARD_TOKEN=YOUR_TOKEN \
  neuroshard/node:latest
```

#### Docker Compose

```yaml
version: '3.8'
services:
  neuroshard-node:
    image: neuroshard/node:latest
    ports:
      - "8000:8000"
      - "9000:9000"
    environment:
      - NEUROSHARD_TOKEN=${NEUROSHARD_TOKEN}
    volumes:
      - neuroshard_data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

volumes:
  neuroshard_data:
```

## Verify Installation

After installation, verify everything works:

```bash
# Check version
neuroshard --version

# Check available options
neuroshard --help

# Test GPU detection
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')"
```

### Device Selection

NeuroShard auto-detects the best device, but you can override it:

```bash
# Auto-detect (default)
neuroshard --token YOUR_TOKEN --device auto

# Force CUDA
neuroshard --token YOUR_TOKEN --device cuda

# Force Apple Metal
neuroshard --token YOUR_TOKEN --device mps

# Force CPU
neuroshard --token YOUR_TOKEN --device cpu
```

## Platform-Specific Notes

### Windows

1. **Install Python**: Download from [python.org](https://python.org)
2. **Enable Long Paths**: Run as admin:
   ```powershell
   Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1
   ```
3. **Install CUDA Toolkit** (if using NVIDIA GPU): Download from [NVIDIA](https://developer.nvidia.com/cuda-downloads)

### macOS

1. **Install Homebrew**: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. **Install Python**: `brew install python@3.11`
3. **For M1/M2/M3**: PyTorch automatically uses Metal Performance Shaders

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv

# For NVIDIA GPU
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install python3 python3-pip

# For NVIDIA GPU
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda
```

### Jetson (JetPack)

```bash
# Ensure JetPack is installed (includes CUDA, cuDNN)
# Then install PyTorch from NVIDIA
pip install torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60

# Install NeuroShard
pip install neuroshard-ai
```

## Updating

### pip

```bash
pip install --upgrade neuroshard-ai
```

## Uninstalling

### pip

```bash
pip uninstall neuroshard-ai
```

### Remove Data

```bash
# Remove checkpoints and cache
rm -rf ~/.neuroshard

# On Windows
rd /s /q %USERPROFILE%\.neuroshard
```

## Troubleshooting

### PyTorch Not Found

If you get `ModuleNotFoundError: No module named 'torch'`:

```bash
# Install with GPU support
pip install neuroshard-ai[gpu]

# Or install torch manually first
pip install torch
pip install neuroshard-ai
```

### CUDA Not Detected

If GPU isn't detected on a system with NVIDIA GPU:

```bash
# Check if torch sees CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall torch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Jetson: Wrong PyTorch Version

If you accidentally installed x86 torch on Jetson:

```bash
# Remove wrong version
pip uninstall torch torchvision

# Install NVIDIA's version
pip install torch torchvision --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60
```

## Next Steps

- [Running a Node](/guide/running-a-node) — Configure and start your node
- [Quick Start](/guide/quick-start) — 5-minute setup guide
- [CLI Reference](/guide/cli-reference) — All command options
