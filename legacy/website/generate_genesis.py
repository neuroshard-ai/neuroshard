import os
import json
import torch
import requests
import logging
import gzip
import io
import pandas as pd

# Configure
DATA_DIR = "/data/genesis"
os.makedirs(DATA_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Hugging Face FineWeb-Edu URL (Parquet file)
# This is a single shard from the dataset
FINEWEB_URL = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10k_sample.parquet"

def generate_genesis_data():
    """
    Generates the initial Genesis Dataset for NeuroShard using real FineWeb-Edu data.
    """
    logger.info("Generating Genesis Data from FineWeb-Edu...")
    
    from neuroshard.core.model.tokenizer import get_neuro_tokenizer
    tokenizer = get_neuro_tokenizer()
    
    # 1. Download Real Data
    logger.info(f"Downloading sample from {FINEWEB_URL}...")
    try:
        response = requests.get(FINEWEB_URL, timeout=30)
        response.raise_for_status()
        
        # Read parquet
        buffer = io.BytesIO(response.content)
        df = pd.read_parquet(buffer)
        
        texts = df['text'].tolist()
        logger.info(f"Downloaded {len(texts)} documents.")
        
    except Exception as e:
        logger.error(f"Failed to download real data: {e}")
        logger.warning("Falling back to synthetic data for offline dev...")
        texts = [f"NeuroShard Decentralized AI. Sample document {i}. " * 100 for i in range(1000)]

    # 2. Create Shards
    # We will split the documents into 1000 shards for "Large Scale" architecture
    TOTAL_SHARDS = 1000
    shards_metadata = []
    total_docs = len(texts)
    docs_per_shard = max(1, total_docs // TOTAL_SHARDS)
    
    for i in range(TOTAL_SHARDS):
        start_idx = i * docs_per_shard
        end_idx = min((i + 1) * docs_per_shard, total_docs)
        
        # Handle last shard taking remainder
        if i == TOTAL_SHARDS - 1:
            end_idx = total_docs
        
        if start_idx >= total_docs:
             # If we ran out of real data, fill with synthetic for remaining shards
             shard_texts = [f"Synthetic filler data for shard {i}. NeuroShard Decentralized AI." for _ in range(10)]
        else:
        shard_texts = texts[start_idx:end_idx]
             
        if not shard_texts:
             shard_texts = [f"Synthetic filler data for shard {i}."]

        full_text = "\n\n<|endoftext|>\n\n".join(shard_texts)
        
        filename = f"shard_{i}.pt"
        filepath = os.path.join(DATA_DIR, filename)
        
        # Tokenize
        tokens = tokenizer.encode(full_text)
        tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Save
        torch.save(tensor, filepath)
        
        # Calculate Hash (SHA256)
        import hashlib
        file_hash = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        
        # Add to metadata
        shards_metadata.append({
            "shard_id": i,
            "url": f"https://neuroshard.com/api/genesis/{filename}", # Public URL (Future)
            "fallback_url": f"http://genesis-host:8080/{filename}",   # Internal Docker URL
            "hash": file_hash,
            "size_tokens": len(tokens),
            "size_bytes": os.path.getsize(filepath)
        })
        
        logger.info(f"Created {filename}: {len(tokens)} tokens, {len(shard_texts)} docs")

    # 3. Create Manifest
    manifest = {
        "version": 1,
        "timestamp": 1732982400,
        "dataset": "FineWeb-Edu Sample",
        "total_shards": TOTAL_SHARDS,
        "shards": shards_metadata
    }
    
    manifest_path = os.path.join(DATA_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    logger.info(f"Manifest saved to {manifest_path}")

if __name__ == "__main__":
    generate_genesis_data()
