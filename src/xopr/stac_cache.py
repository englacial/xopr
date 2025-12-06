"""
STAC catalog caching utilities for xopr.

This module provides functions to cache STAC GeoParquet catalogs locally,
reducing network latency for repeated queries.
"""

import os
from pathlib import Path
from typing import List, Optional

import requests
from platformdirs import user_cache_dir


# Cloud URLs for bedmap catalogs
BEDMAP_CATALOG_BASE_URL = "https://storage.googleapis.com/opr_stac/bedmap"
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
