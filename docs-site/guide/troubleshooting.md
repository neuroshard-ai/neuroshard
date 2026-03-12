# Troubleshooting

Solutions to common issues when running a NeuroShard node.

## Startup Issues

### "Wallet token required!"

```
[ERROR] Wallet token required!
```

**Cause**: No token provided.

**Solution**: 
```bash
neuroshard --token YOUR_WALLET_TOKEN
```

Get your token at [neuroshard.com/dashboard](https://neuroshard.com/dashboard).

### "Invalid mnemonic"

```
[WARNING] Invalid mnemonic - treating as raw token
```

**Cause**: Mnemonic phrase is incorrect.

**Solution**: 
- Verify all 12 words are correct and in order
- Ensure words are separated by single spaces
- Use quotes around the mnemonic:
  ```bash
  neuroshard --token "word1 word2 word3 ..."
  ```

### "No GPU detected"

```
[NODE] No GPU detected, using CPU
```

**For NVIDIA GPUs**:
```bash
# Check if NVIDIA driver is installed
nvidia-smi

# Install CUDA-enabled PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**For Apple Silicon**:
```bash
# Ensure PyTorch is installed correctly
pip install torch
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Memory Issues

### "Out of memory"

```
RuntimeError: MPS backend out of memory
RuntimeError: CUDA out of memory
```

**Solutions**:

1. **Limit memory usage**:
   ```bash
   neuroshard --token YOUR_TOKEN --memory 4096
   ```

2. **Reduce batch size** (automatic on OOM, but you can force lower):
   ```bash
   # If you see "Reduced batch size to X"
   # The node auto-recovers, no action needed
   ```

3. **Close other applications** using GPU memory

4. **Use CPU** if GPU memory is too limited:
   ```bash
   CUDA_VISIBLE_DEVICES="" neuroshard --token YOUR_TOKEN
   ```

### "System memory at X%, skipping training"

**Cause**: System RAM is critically low.

**Solution**:
```bash
# Free up memory
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

# Or limit neuroshard memory
neuroshard --token YOUR_TOKEN --memory 2048
```

## Network Issues

### "Connection refused"

```
Failed to forward to peer http://...: Connection refused
```

**Causes & Solutions**:

1. **Firewall blocking ports**:
   ```bash
   sudo ufw allow 8000/tcp
   sudo ufw allow 9000/tcp
   ```

2. **Peer is offline**: Normal, the network will route around it

3. **Port already in use**:
   ```bash
   neuroshard --token YOUR_TOKEN --port 8001
   ```

### "Tracker connection failed"

```
Failed to connect to tracker
```

**Solutions**:

1. Check internet connection
2. Verify tracker URL:
   ```bash
   curl https://neuroshard.com/api/tracker/peers
   ```
3. Use a different tracker if available

### "No peers found"

**Cause**: Network is bootstrapping or you're the first node.

**Solution**: Wait a few minutes. The tracker will provide peers as they join.

## Training Issues

### "Data not ready"

```
RuntimeError: Data not ready - shard still loading
```

**Cause**: Genesis data loader is still downloading.

**Solution**: Wait 30-60 seconds. The node will retry automatically.

### "Genesis loader init failed"

```
[GENESIS] ERROR: Failed to initialize loader
```

**Solutions**:

1. Check disk space:
   ```bash
   df -h
   ```

2. Increase storage limit:
   ```bash
   neuroshard --token YOUR_TOKEN --max-storage 200
   ```

3. Check write permissions:
   ```bash
   ls -la ~/.neuroshard/
   ```

### Training Loss Not Decreasing

**Causes**:

1. **Early network stage**: Expected behavior when model is small
2. **Gradient poisoning**: Rare, network defenses should handle it
3. **Learning rate issues**: Currently fixed, no user action needed

**Monitor**:
```bash
curl http://localhost:8000/api/stats | jq '.current_loss'
```

## Dashboard Issues

### Dashboard Not Opening

```
[NODE] Could not open browser
```

**Solution**: Open manually at `http://localhost:8000/`

### Dashboard Shows Stale Data

**Solution**: Refresh the page. The dashboard auto-refreshes every 5 seconds.

### API Returns 404

```
curl: (52) Empty reply from server
```

**Solution**: Wait for the node to fully initialize (10-30 seconds after startup).

## Checkpoint Issues

### "No checkpoint found, starting fresh"

```
[NODE] No checkpoint found at dynamic_node_XXXX.pt, starting fresh
```

**Cause**: No previous checkpoint exists for this wallet.

**Solution**: This is normal on first run. Checkpoints are saved every 10 steps.

### "Architecture mismatch!"

```
[NODE] Architecture mismatch! Checkpoint is incompatible.
[NODE]   Saved: 15L × 704H, heads=11/1
[NODE]   Current: 17L × 770H, heads=11/1
[NODE]   Starting fresh (architecture was upgraded)
```

**Cause**: The model architecture changed due to:
- Network capacity increased (more nodes joined)
- Memory fluctuation caused different architecture calculation

**Solutions**:

1. **Normal behavior** - when network genuinely upgrades, old checkpoints are incompatible
2. **If happening frequently on restarts**, update to latest version with memory tier rounding fix:
   ```bash
   pip install --upgrade neuroshard
   ```
   This rounds memory to 500MB tiers, preventing small fluctuations from changing architecture.

3. **Clear old checkpoints and start fresh**:
   ```bash
   rm -rf ~/.neuroshard/checkpoints/*
   rm -rf ~/.neuroshard/training_logs/*
   ```

### "Checkpoint layer mismatch"

```
[WARNING] Checkpoint layer mismatch, starting fresh
```

**Cause**: Your assigned layers changed (network rebalanced).

**Solution**: Normal behavior. Common layers will be loaded if possible.

### Checkpoint Not Loading After Restart

**Symptoms**: Training restarts from step 1 instead of resuming.

**Causes & Solutions**:

1. **Architecture changed** - Check logs for "Architecture mismatch"
2. **Different node_id** - Old versions used machine-specific IDs. Update to v0.0.20+ which uses wallet_id
3. **Checkpoint corrupted** - Delete and restart:
   ```bash
   rm ~/.neuroshard/checkpoints/dynamic_node_*.pt
   ```

### Checkpoint Corrupted

```
RuntimeError: Failed to load checkpoint
```

**Solution**:
```bash
# Delete corrupted checkpoint
rm ~/.neuroshard/checkpoints/dynamic_node_*.pt

# Clear tracker state too (optional, but recommended)
rm ~/.neuroshard/training_logs/*.json

# Restart node
neuroshard --token YOUR_TOKEN
```

### GlobalTrainingTracker State Preserved But Model Lost

If you see logs like:
```
[NODE] Restored tracker state: 120 steps, avg_loss=0.4872
[NODE] No checkpoint found, starting fresh
```

**Cause**: The tracker state (loss history) persisted, but model weights didn't (architecture changed).

**Solution**: This is actually fine! The tracker history helps you see long-term trends even when the model architecture changes.

## Performance Issues

### Low GPU Utilization

**Causes & Solutions**:

1. **Small batch size**: Expected with limited memory
2. **Network bottleneck**: Increase `--diloco-steps`
3. **Data loading**: Genesis loader might be slow initially

### High CPU Usage

**Normal** during training. To limit:
```bash
neuroshard --token YOUR_TOKEN --cpu-threads 4
```

### Node Seems Slow

1. Check GPU utilization:
   ```bash
   nvidia-smi  # For NVIDIA
   # Or Activity Monitor on macOS
   ```

2. Check if training is active:
   ```bash
   curl http://localhost:8000/api/stats | jq '.total_training_rounds'
   ```

## Getting Help

### Check Logs

```bash
# Live logs (if running in foreground)
# Check console output

# Systemd logs
journalctl -u neuroshard -f

# Docker logs
docker logs -f neuroshard
```

### Report Issues

1. Check our [Discord](https://discord.gg/4R49xpj7vn) for community support
2. Include:
   - NeuroShard version (`neuroshard --version`)
   - OS and Python version
   - Full error message
   - Steps to reproduce

### Community Support

- Website Community
- [Discord Community](https://discord.gg/4R49xpj7vn)

## See Also

- [Configuration](/guide/configuration)
- [Running a Node](/guide/running-a-node)
- [FAQ](/guide/faq)

