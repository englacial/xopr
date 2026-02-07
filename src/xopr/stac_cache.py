"""
STAC catalog caching utilities for xopr.

This module provides functions to cache STAC GeoParquet catalogs locally,
reducing network latency for repeated queries.
"""

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests
from platformdirs import user_cache_dir

# OPR catalog constants
OPR_CATALOG_S3_PREFIX = "englacial/xopr/catalog/"
OPR_CATALOG_S3_GLOB = (
    "s3://us-west-2.opendata.source.coop/englacial/xopr/catalog/**/*.parquet"
)
S3_LIST_URL = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop"
OPR_CATALOG_HTTPS_BASE = "https://data.source.coop/"

# Cloud URLs for bedmap catalogs
BEDMAP_CATALOG_BASE_URL = "https://data.source.coop/englacial/bedmap"
BEDMAP_CATALOG_FILES = ["bedmap1.parquet", "bedmap2.parquet", "bedmap3.parquet"]


def get_cache_dir() -> Path:
    """
    Get the xopr cache directory.

    Checks $XOPR_CACHE_DIR environment variable first, otherwise uses
    platform-specific user cache directory.

    Returns
    -------
    Path
        Path to xopr cache directory
    """
    env_cache = os.environ.get("XOPR_CACHE_DIR")
    if env_cache:
        cache_path = Path(env_cache)
    else:
        cache_path = Path(user_cache_dir("xopr", "englacial"))

    return cache_path


def get_bedmap_catalog_dir() -> Path:
    """
    Get the bedmap catalog cache directory.

    Returns
    -------
    Path
        Path to bedmap catalog directory within cache
    """
    return get_cache_dir() / "catalogs" / "bedmap"


def _download_file(url: str, dest: Path) -> bool:
    """
    Download a file from URL to destination path.

    Parameters
    ----------
    url : str
        URL to download from
    dest : Path
        Destination file path

    Returns
    -------
    bool
        True if download succeeded, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        dest.parent.mkdir(parents=True, exist_ok=True)

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"Warning: Failed to download {url}: {e}")
        return False


def ensure_bedmap_catalogs(force_download: bool = False) -> Optional[Path]:
    """
    Ensure bedmap catalogs are cached locally, downloading if needed.

    Parameters
    ----------
    force_download : bool, default False
        If True, re-download catalogs even if they exist

    Returns
    -------
    Path or None
        Path to catalog directory if successful, None if download failed
    """
    catalog_dir = get_bedmap_catalog_dir()

    # Check if all catalogs exist
    all_exist = all((catalog_dir / f).exists() for f in BEDMAP_CATALOG_FILES)

    if all_exist and not force_download:
        return catalog_dir

    # Download missing catalogs
    print(f"Downloading bedmap catalogs to {catalog_dir}...")
    catalog_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for filename in BEDMAP_CATALOG_FILES:
        dest = catalog_dir / filename
        if dest.exists() and not force_download:
            continue

        url = f"{BEDMAP_CATALOG_BASE_URL}/{filename}"
        if not _download_file(url, dest):
            success = False

    if success:
        print(f"Bedmap catalogs cached successfully")
        return catalog_dir
    else:
        print("Warning: Some catalogs failed to download")
        # Return catalog_dir anyway - partial cache may still be useful
        return catalog_dir if any((catalog_dir / f).exists() for f in BEDMAP_CATALOG_FILES) else None


def get_bedmap_catalog_path() -> str:
    """
    Get the path pattern for bedmap catalogs, downloading if needed.

    This is the main entry point for query functions. It ensures catalogs
    are cached locally and returns the glob pattern for querying.

    Returns
    -------
    str
        Glob pattern to local bedmap catalog files, or cloud URL as fallback
    """
    catalog_dir = ensure_bedmap_catalogs()

    if catalog_dir and any((catalog_dir / f).exists() for f in BEDMAP_CATALOG_FILES):
        return str(catalog_dir / "bedmap*.parquet")
    else:
        # Fallback to cloud URL if local cache failed
        print("Warning: Using cloud catalogs (local cache unavailable)")
        return f"{BEDMAP_CATALOG_BASE_URL}/bedmap*.parquet"


def clear_bedmap_cache() -> None:
    """
    Clear cached bedmap catalogs.

    Useful for forcing a fresh download of catalogs.
    """
    catalog_dir = get_bedmap_catalog_dir()
    if catalog_dir.exists():
        for f in BEDMAP_CATALOG_FILES:
            path = catalog_dir / f
            if path.exists():
                path.unlink()
        print(f"Cleared bedmap catalog cache at {catalog_dir}")


# ---------------------------------------------------------------------------
# OPR catalog caching
# ---------------------------------------------------------------------------


def get_opr_catalog_dir() -> Path:
    """
    Get the OPR catalog cache directory.

    Returns
    -------
    Path
        Path to OPR catalog directory within cache
    """
    return get_cache_dir() / "catalogs" / "opr"


def _list_remote_opr_catalogs() -> list[dict]:
    """
    List remote OPR STAC catalog files via the S3 ListBucketV2 API.

    Returns
    -------
    list[dict]
        Each dict has keys ``key``, ``etag``, and ``size``.
    """
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    results: list[dict] = []
    continuation_token = None

    while True:
        params = {"list-type": "2", "prefix": OPR_CATALOG_S3_PREFIX}
        if continuation_token:
            params["continuation-token"] = continuation_token

        resp = requests.get(S3_LIST_URL, params=params, timeout=30)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        for content in root.findall(f"{ns}Contents"):
            key = content.findtext(f"{ns}Key", "")
            if key.endswith(".parquet"):
                results.append({
                    "key": key,
                    "etag": content.findtext(f"{ns}ETag", "").strip('"'),
                    "size": int(content.findtext(f"{ns}Size", "0")),
                })

        if root.findtext(f"{ns}IsTruncated", "false") == "true":
            continuation_token = root.findtext(f"{ns}NextContinuationToken")
        else:
            break

    return results


def _load_opr_manifest() -> dict:
    """
    Load the local OPR manifest (maps relative path -> etag/size).

    Returns
    -------
    dict
        Manifest dictionary, or empty dict if missing.
    """
    manifest_path = get_opr_catalog_dir() / "_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def _save_opr_manifest(manifest: dict) -> None:
    """
    Persist the OPR manifest to disk.

    Parameters
    ----------
    manifest : dict
        Manifest mapping relative paths to ``{"etag", "size"}`` dicts.
    """
    catalog_dir = get_opr_catalog_dir()
    catalog_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = catalog_dir / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)


def sync_opr_catalogs() -> None:
    """
    Sync OPR STAC catalog parquet files to local cache.

    Compares remote ETags against a local manifest and only downloads
    new or changed files.  Designed to run in a background thread;
    silently returns on network errors.
    """
    try:
        remote_files = _list_remote_opr_catalogs()
    except Exception:
        return  # silent failure on network errors

    catalog_dir = get_opr_catalog_dir()
    manifest = _load_opr_manifest()
    changed = False

    for entry in remote_files:
        key = entry["key"]
        # Relative path under the catalog dir (strip the S3 prefix)
        rel = key[len(OPR_CATALOG_S3_PREFIX):]
        cached = manifest.get(rel)
        if cached and cached.get("etag") == entry["etag"]:
            continue  # unchanged

        # Download via HTTPS with atomic write
        url = f"{OPR_CATALOG_HTTPS_BASE}{key}"
        dest = catalog_dir / rel
        tmp = dest.with_suffix(".tmp")
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            os.replace(tmp, dest)
            manifest[rel] = {"etag": entry["etag"], "size": entry["size"]}
            changed = True
        except Exception:
            if tmp.exists():
                tmp.unlink()
            continue  # skip this file, keep going

    if changed:
        _save_opr_manifest(manifest)


def get_opr_catalog_path() -> str:
    """
    Get a path/glob for OPR STAC catalogs, preferring local cache.

    Returns
    -------
    str
        Local glob pattern if cached files exist, otherwise the S3 glob.
    """
    catalog_dir = get_opr_catalog_dir()
    if catalog_dir.exists() and any(catalog_dir.rglob("*.parquet")):
        return str(catalog_dir / "**" / "*.parquet")
    return OPR_CATALOG_S3_GLOB


def clear_opr_cache() -> None:
    """
    Remove the entire OPR catalog cache directory.
    """
    catalog_dir = get_opr_catalog_dir()
    if catalog_dir.exists():
        shutil.rmtree(catalog_dir)
        print(f"Cleared OPR catalog cache at {catalog_dir}")
