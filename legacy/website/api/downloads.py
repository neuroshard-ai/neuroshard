from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import requests
from pydantic import BaseModel
from typing import List, Optional
import os
import time
import re

router = APIRouter()

class Asset(BaseModel):
    name: str
    id: Optional[int] = None
    browser_download_url: str
    size: int

class Release(BaseModel):
    name: str
    tag_name: str
    published_at: str
    html_url: Optional[str] = None
    assets: List[Asset]

GITHUB_REPO = "LinirZamir/neuroshard"

# Simple in-memory cache: (timestamp, data)
CACHE_TTL = 60  # 1 minute - reduced for faster updates
_release_cache = {"timestamp": 0, "data": None}

def parse_version(version_str: str) -> tuple:
    """Parse version string into tuple for comparison (handles v0.1.1 and 0.1.1)"""
    if not version_str:
        return (0, 0, 0)
    # Remove 'v' prefix if present
    version_str = version_str.lstrip('v')
    # Split by dots and convert to integers
    parts = version_str.split('.')
    try:
        # Pad with zeros if needed (e.g., "0.1" -> (0, 1, 0))
        while len(parts) < 3:
            parts.append('0')
        return tuple(int(p) for p in parts[:3])
    except ValueError:
        # If parsing fails, return (0, 0, 0) as fallback
        return (0, 0, 0)

def compare_versions(v1: str, v2: str) -> int:
    """Compare two version strings. Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal"""
    v1_tuple = parse_version(v1)
    v2_tuple = parse_version(v2)
    if v1_tuple > v2_tuple:
        return 1
    elif v1_tuple < v2_tuple:
        return -1
    return 0

def get_github_headers():
    """Get headers for GitHub API requests, including auth token if available."""
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        # Use Bearer format for modern GitHub PATs (github_pat_*)
        # Both "token" and "Bearer" work, but Bearer is preferred for PATs
        headers["Authorization"] = f"Bearer {github_token}"
    return headers

@router.get("/latest", response_model=Release)
async def get_latest_release():
    global _release_cache
    now = time.time()
    
    # Return cached data if valid
    if _release_cache["data"] and (now - _release_cache["timestamp"] < CACHE_TTL):
        return _release_cache["data"]

    try:
        headers = get_github_headers()
        
        # Strategy 1: Fetch tags first to get the absolute latest version
        url_tags = f"https://api.github.com/repos/{GITHUB_REPO}/tags"
        resp_tags = requests.get(url_tags, headers=headers, timeout=5)
        
        sorted_tags = []
        if resp_tags.status_code == 200:
            tags = resp_tags.json()
            if tags and isinstance(tags, list) and len(tags) > 0:
                # Sort tags by semantic version (newest first)
                sorted_tags = sorted(tags, key=lambda t: parse_version(t.get('name', '')), reverse=True)
                
        # Strategy 2: Fetch releases to get assets
        url_list = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
        resp_list = requests.get(url_list, headers=headers, timeout=5)
        
        releases_list = []
        if resp_list.status_code == 200:
            releases_list = resp_list.json()

        # Strategy 3: Iterate tags to find the latest one WITH A VALID RELEASE
        # This avoids the issue where a new tag exists (e.g. 0.1.30) but the release is still building (404),
        # which previously caused a fallback to v0.0.1. Now we check 0.1.29, 0.1.28, etc.
        
        for tag in sorted_tags:
            tag_name = tag.get('name', '')
            print(f"Checking release for tag: {tag_name}")
            
            # 1. Check if this tag is already in the releases list
            matching_release = next((r for r in releases_list if r.get('tag_name') == tag_name), None)
            
            if matching_release:
                 print(f"Found matching release in list for {tag_name}")
                 release = parse_github_release(matching_release)
                 if release["tag_name"].startswith('v'):
                     release["tag_name"] = release["tag_name"][1:]
                 update_cache(release)
                 return release
            
            # 2. If not in list, try fetching specific release (in case it's not in the top list)
            try:
                url_specific = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{tag_name}"
                resp_specific = requests.get(url_specific, headers=headers, timeout=2)
                
                if resp_specific.status_code == 200:
                    print(f"Found specific release for {tag_name}")
                    specific_release = resp_specific.json()
                    release = parse_github_release(specific_release)
                    if release["tag_name"].startswith('v'):
                        release["tag_name"] = release["tag_name"][1:]
                    update_cache(release)
                    return release
                else:
                    print(f"Tag {tag_name} has no release yet (status {resp_specific.status_code}). Skipping...")
            except Exception as e:
                 print(f"Error checking release for {tag_name}: {e}")
                 continue
        
        # Fallback if loop fails (shouldn't happen if tags exist)
        if releases_list:
            print("No tags matched, falling back to latest release in list.")
            release = parse_github_release(releases_list[0])
            update_cache(release)
            return release

        # If all else fails
        raise HTTPException(status_code=404, detail="No releases or tags found")

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error fetching GitHub data: {type(e).__name__}: {str(e)}")
        print(f"Full traceback:\n{error_trace}")
        if _release_cache["data"]:
             print("Returning stale cache due to error.")
             return _release_cache["data"]
        raise HTTPException(status_code=503, detail=f"GitHub API unavailable: {type(e).__name__}: {str(e)}")

@router.get("/asset/{asset_id}")
async def download_asset(asset_id: int):
    """Proxy download of a private GitHub release asset"""
    try:
        headers = get_github_headers()
        headers["Accept"] = "application/octet-stream"
        
        url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/assets/{asset_id}"
        
        # GitHub API returns 302 to S3. Requests follows it by default.
        # We stream the content to the client.
        # Increased timeout for large files (Mac .zip can be 100MB+)
        r = requests.get(url, headers=headers, stream=True, timeout=300)
        
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Asset not found")
        r.raise_for_status()
        
        # Get filename from Content-Disposition if possible, or just use generic
        cd = r.headers.get("Content-Disposition")
        filename = "download"
        if cd:
            # Simple regex to extract filename
            fname = re.findall('filename="?([^"]+)"?', cd)
            if len(fname) > 0:
                filename = fname[0]
        
        # Build response headers - CRITICAL: pass through Content-Length
        response_headers = {
            "Content-Disposition": f"attachment; filename={filename}"
        }
        
        # Pass through Content-Length so client knows expected size
        content_length = r.headers.get("Content-Length")
        if content_length:
            response_headers["Content-Length"] = content_length
            print(f"Proxying asset {asset_id}: {filename}, size={content_length}")
        
        def generate():
            """Generator that yields chunks and handles errors gracefully."""
            try:
                for chunk in r.iter_content(chunk_size=65536):  # 64KB chunks for better throughput
                    if chunk:
                        yield chunk
            except Exception as e:
                print(f"Stream error for asset {asset_id}: {e}")
                # Can't raise here, just stop yielding
            finally:
                r.close()
        
        return StreamingResponse(
            generate(),
            media_type=r.headers.get("content-type", "application/octet-stream"),
            headers=response_headers
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Download error for asset {asset_id}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=503, detail=f"Download failed: {str(e)}")

def parse_github_release(data):
    """Helper to transform GitHub JSON to our schema"""
    return {
        "name": data.get("name") or data.get("tag_name", "Latest Release"),
        "tag_name": data.get("tag_name", ""),
        "published_at": data.get("published_at") or data.get("created_at", ""),
        "html_url": data.get("html_url", ""),
        "assets": [
            {
                "name": asset.get("name", ""),
                "id": asset.get("id"),
                "browser_download_url": asset.get("browser_download_url", ""),
                "size": asset.get("size", 0)
            }
            for asset in data.get("assets", [])
            if not asset.get("name", "").startswith("Source code")
        ]
    }

def update_cache(data):
    global _release_cache
    _release_cache = {
        "timestamp": time.time(),
        "data": data
    }

# Checksums cache
_checksums_cache = {"timestamp": 0, "data": None}

@router.get("/checksums")
async def get_checksums():
    """Get SHA256 checksums for latest release builds."""
    global _checksums_cache, _release_cache
    now = time.time()
    
    # Return cached data if valid (5 min TTL)
    if _checksums_cache["data"] and (now - _checksums_cache["timestamp"] < CACHE_TTL):
        return _checksums_cache["data"]
    
    try:
        # Get latest version first if not cached
        if not _release_cache["data"]:
            try:
                release_data = await get_latest_release()
                version = release_data["tag_name"]
            except:
                # Fallback: try "latest" path
                version = "latest"
        else:
            version = _release_cache["data"]["tag_name"]
        
        # Fetch checksums from CloudFront
        checksums_url = f"https://d1qsvy9420pqcs.cloudfront.net/releases/{version}/checksums.json"
        print(f"Fetching checksums from: {checksums_url}")
        resp = requests.get(checksums_url, timeout=5)
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"Checksums loaded successfully for version {version}")
            
            # Update cache
            _checksums_cache = {
                "timestamp": now,
                "data": data
            }
            
            return data
        else:
            print(f"Checksums request failed with status {resp.status_code}")
            raise HTTPException(status_code=404, detail="Checksums not available for this version yet")
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error fetching checksums: {e}")
        print(traceback.format_exc())
        # Return stale cache if available
        if _checksums_cache["data"]:
            return _checksums_cache["data"]
        raise HTTPException(status_code=503, detail=f"Could not fetch checksums: {str(e)}")
