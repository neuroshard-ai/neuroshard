#!/usr/bin/env python3
"""
Generate SHA256 checksums for release builds.

This script should be run after builds are uploaded to S3.
It creates a checksums.json file that can be used by the auto-updater
to verify downloaded files.

Usage:
    python scripts/generate_checksums.py v0.4.2
    python scripts/generate_checksums.py latest
"""

import sys
import hashlib
import requests
import json
from pathlib import Path


def compute_sha256_from_url(url: str) -> str:
    """
    Download file from URL and compute SHA256 hash.
    
    Args:
        url: URL of file to hash
        
    Returns:
        Hex string of SHA256 hash
    """
    print(f"Downloading {url}...")
    
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    sha256 = hashlib.sha256()
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            sha256.update(chunk)
            downloaded += len(chunk)
            
            # Progress indicator
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                if downloaded % (10 * 1024 * 1024) < 8192:  # Every 10MB
                    print(f"  {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")
    
    result = sha256.hexdigest()
    print(f"  ✅ SHA256: {result}")
    
    return result


def generate_checksums_for_version(version: str):
    """
    Generate checksums for all builds of a specific version.
    
    Args:
        version: Version tag (e.g., "v0.4.2" or "latest")
    """
    base_url = f"https://d1qsvy9420pqcs.cloudfront.net/releases/{version}"
    
    builds = {
        "windows_cpu": f"{base_url}/NeuroShardNode-CPU.exe",
        "windows_gpu": f"{base_url}/NeuroShardNode-GPU.exe",
        "macos": f"{base_url}/NeuroShardNode_Mac.zip",
    }
    
    checksums = {
        "version": version,
        "generated_at": None,  # Will be filled by S3 upload time
        "builds": {}
    }
    
    for build_name, url in builds.items():
        print(f"\nProcessing {build_name}...")
        
        try:
            sha256 = compute_sha256_from_url(url)
            
            checksums["builds"][build_name] = {
                "url": url,
                "sha256": sha256,
                "filename": url.split('/')[-1]
            }
        
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Failed: {e}")
            checksums["builds"][build_name] = {
                "url": url,
                "sha256": None,
                "error": str(e)
            }
    
    return checksums


def save_checksums_json(checksums: dict, output_path: str = None):
    """
    Save checksums to JSON file.
    
    Args:
        checksums: Checksums dictionary
        output_path: Path to save JSON (default: checksums.json)
    """
    if output_path is None:
        output_path = "checksums.json"
    
    with open(output_path, 'w') as f:
        json.dump(checksums, f, indent=2)
    
    print(f"\n✅ Checksums saved to: {output_path}")


def upload_to_s3(checksums_file: str, version: str):
    """
    Upload checksums.json to S3.
    
    Args:
        checksums_file: Path to checksums.json
        version: Version tag
    """
    import boto3
    
    s3 = boto3.client('s3')
    bucket = "neuroshard"
    key = f"releases/{version}/checksums.json"
    
    print(f"\nUploading to s3://{bucket}/{key}...")
    
    s3.upload_file(
        checksums_file,
        bucket,
        key,
        ExtraArgs={
            'ACL': 'public-read',
            'ContentType': 'application/json',
            'CacheControl': 'max-age=3600'
        }
    )
    
    print(f"✅ Uploaded to: https://d1qsvy9420pqcs.cloudfront.net/{key}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python generate_checksums.py <version>")
        print("Example: python generate_checksums.py v0.4.2")
        print("Example: python generate_checksums.py latest")
        sys.exit(1)
    
    version = sys.argv[1]
    
    print("=" * 60)
    print(f"Generating checksums for version: {version}")
    print("=" * 60)
    
    checksums = generate_checksums_for_version(version)
    
    # Save to file
    output_file = f"checksums_{version.replace('/', '_')}.json"
    save_checksums_json(checksums, output_file)
    
    # Display summary
    print("\n" + "=" * 60)
    print("Checksum Summary")
    print("=" * 60)
    
    for build_name, build_info in checksums["builds"].items():
        sha = build_info.get("sha256")
        if sha:
            print(f"{build_name:20} {sha}")
        else:
            print(f"{build_name:20} ERROR: {build_info.get('error', 'Unknown')}")
    
    # Ask if user wants to upload to S3
    if checksums["builds"]:
        print("\n" + "=" * 60)
        upload = input("Upload checksums.json to S3? (y/N): ").strip().lower()
        
        if upload == 'y':
            try:
                upload_to_s3(output_file, version)
            except Exception as e:
                print(f"❌ Upload failed: {e}")
                print("\nYou can upload manually with:")
                print(f"aws s3 cp {output_file} s3://neuroshard/releases/{version}/checksums.json --acl public-read")


if __name__ == "__main__":
    main()

