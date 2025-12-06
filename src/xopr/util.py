"""
General utility functions for xOPR

This module provides helper functions for data comparison, dictionary merging,
and metadata retrieval that are used throughout xOPR.
"""

import itertools
import json
from typing import Any, Dict, List, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd
import requests

T = TypeVar("T")

def dict_equiv(first: dict, second: dict) -> bool:
    """Compare two dictionaries for equivalence (identity or equality).

    Parameters
    ----------
    first : dict
        First dictionary to compare.
    second : dict
        Second dictionary to compare.

    Returns
    -------
    bool
        True if dictionaries are identical or have the same keys with
        equivalent values (as determined by the equivalent function), False otherwise.
    """
    if first is second:
        return True
    if len(first) != len(second):
        return False
    for key in first:
        if key not in second:
            return False
        if not equivalent(first[key], second[key]):
            return False
    return True

def list_equiv(first: Sequence[T], second: Sequence[T]) -> bool:
    """Compare two sequences for element-wise equivalence.

    Parameters
    ----------
    first : Sequence[T]
        First sequence to compare.
    second : Sequence[T]
        Second sequence to compare.

    Returns
    -------
    bool
        True if sequences have equal length and all corresponding elements
        are equivalent (as determined by the equivalent function), False otherwise.
    """
    if len(first) != len(second):
        return False
    return all(itertools.starmap(equivalent, zip(first, second, strict=True)))

def equivalent(first, second) -> bool:
    """Compare two objects for equivalence (identity or equality).

    Handles different data types:

    * Arrays: Uses numpy.array_equal for comparison
    * Lists/sequences: Recursively compares all elements
    * Dictionaries: Compares keys and values recursively
    * Other types: Uses equality operator or pandas null checking

    Parameters
    ----------
    first : Any
        First object to compare.
    second : Any
        Second object to compare.

    Returns
    -------
    bool
        True if objects are identical, equal, or both are null/NaN, False otherwise.
    """

    if first is second:
        return True
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        try:
            return np.array_equal(first, second)
        except Exception:
            return False
    if isinstance(first, list) or isinstance(second, list):
        return list_equiv(first, second)  # type: ignore[arg-type]
    if isinstance(first, dict) or isinstance(second, dict): # Added: Also supports dictionaries
        return dict_equiv(first, second)
    return (first == second) or (pd.isnull(first) and pd.isnull(second))  # type: ignore[call-overload]

def merge_dicts_no_conflicts(dicts: List[Dict[str, Any]], context=None) -> Dict[str, Any]:
    """Merge a list of dictionaries, dropping conflicting keys.

    This function is designed to be passed to xarray's combine_attrs parameter.
    It merges dictionaries by keeping only keys where all values are equivalent.
    For nested dictionaries, merging is applied recursively.

    Parameters
    ----------
    dicts : List[Dict[str, Any]]
        List of dictionaries to merge.
    context : Any, optional
        Optional context parameter (unused but included for xarray compatibility).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing only non-conflicting key-value pairs from input dictionaries.
        Keys with conflicting values across dictionaries are dropped.

    Examples
    --------
    >>> dicts = [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]
    >>> merge_dicts_no_conflicts(dicts)
    {'a': 1}  # 'b' dropped due to conflict
    """
    merged = {}
    # Create set of all keys across dictionaries
    all_keys = set().union(*(d.keys() for d in dicts))
    for key in all_keys:
        # Collect values for the current key from all dictionaries
        values = [d.get(key) for d in dicts if key in d]
        if len(values) == 1:
            merged[key] = values[0]  # Only one value, no conflict
        else:
            # Check if all values have the same type
            types = set(type(v) for v in values if v is not None)
            if len(types) > 1:
                continue  # Skip conflicting keys
            if isinstance(values[0], dict):
                # If values are dictionaries, merge them recursively
                merged_dict = merge_dicts_no_conflicts(values)
                if len(merged_dict) > 0:
                    merged[key] = merged_dict
            else:
                all_equiv = True
                for idx in range(1, len(values)):
                    if not equivalent(values[0], values[idx]):
                        all_equiv = False
                        break
                if all_equiv:
                    merged[key] = values[0]
    return merged

def get_ror_display_name(ror_id: str) -> Optional[str]:
    """
    Fetch institution display name from ROR API.

    Parameters
    ----------
    ror_id : str
        ROR identifier (full URL or just ID, e.g., "02jx3x895").

    Returns
    -------
    str or None
        Institution display name, or None if request fails.
    """
    # Clean the ROR ID - extract just the identifier part if full URL is provided
    if ror_id.startswith('https://ror.org/'):
        ror_id = ror_id.replace('https://ror.org/', '')

    try:
        # Make request to ROR API
        url = f"https://api.ror.org/organizations/{ror_id}"
        response = requests.get(url)
        response.raise_for_status()

        # Parse JSON response
        data = response.json()

        # Extract for_display name
        names = data.get('names', [])
        for name_entry in names:
            if name_entry.get('types') and 'ror_display' in name_entry['types']:
                return name_entry.get('value')

        # Fallback to primary name if no for_display found
        return data.get('name')

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from ROR API: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing ROR API response: {e}")
        return None
