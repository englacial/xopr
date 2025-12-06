"""
Utilities and fixes for loading MATLAB format files into Xarray.

This module provides various utilities for correctly reading MATLAB files and
converting them to formats compatible with xarray datasets. It handles both
modern HDF5-format MATLAB files (.mat v7.3+) and legacy MATLAB file formats.

The module addresses several common issues when loading MATLAB data:

- Dereferencing HDF5 object references in MATLAB files
- Decoding MATLAB char arrays (both uint16 Unicode and uint8 ASCII)
- Converting MATLAB cell arrays to Python lists
- Handling empty MATLAB arrays
- Stripping sensitive data (API keys) from attributes
- Converting object ndarrays to JSON-serializable lists

HDF5-Format MATLAB Files (v7.3+)
--------------------------------
- dereference_h5value: Recursively dereference HDF5 object references
- decode_hdf5_matlab_variable: Decode MATLAB variables from HDF5 storage

Legacy MATLAB Files (v4-v7.2)
------------------------------
- extract_legacy_mat_attributes: Extract attributes from legacy .mat files
- strip_api_key: Remove API keys from attribute dictionaries
- convert_object_ndarrays_to_lists: Convert object arrays to lists

Notes
-----
@private This module is not intended for external use.

"""

from collections.abc import Iterable

import h5py
import numpy as np
import scipy.io

#
# HDF5-format MATLAB files
#

def dereference_h5value(value, h5file, make_array=True):
    if isinstance(value, h5py.Reference):
        return dereference_h5value(h5file[value], h5file=h5file)
    elif isinstance(value, h5py.Group):
        # Pass back to decode_hdf5_matlab_variable to handle groups
        return decode_hdf5_matlab_variable(value, h5file=h5file)
    elif isinstance(value, Iterable):
        v = [dereference_h5value(v, h5file=h5file) for v in value]
        if make_array:
            try:
                return np.squeeze(np.array(v))
            except:
                return v
        else:
            return v
    elif isinstance(value, np.number):
        return value.item()
    else:
        return value

def decode_hdf5_matlab_variable(h5var, skip_variables=False, debug_path="", skip_errors=True, h5file=None):
    """
    Decode a MATLAB variable stored in an HDF5 file.
    This function assumes the variable is stored as a byte string.
    """
    if h5file is None:
        h5file = h5var.file
    matlab_class = h5var.attrs.get('MATLAB_class', None)

    # Handle MATLAB_class as either bytes or string
    if matlab_class and (matlab_class == b'cell' or matlab_class == 'cell'):
        return dereference_h5value(h5var[:], h5file=h5file, make_array=False)
    elif matlab_class and (matlab_class == b'char' or matlab_class == 'char'):
        # Check if this is an empty MATLAB char array
        if h5var.attrs.get('MATLAB_empty', 0):
            return ''

        # MATLAB stores char arrays as uint16 (Unicode code points)
        # or sometimes uint8 (ASCII). Handle both cases properly.
        data = h5var[:]

        if data.dtype == np.dtype('uint16'):
            # Each uint16 value is a Unicode code point (UCS-2/UTF-16)
            # Convert to string by treating each value as a character code
            chars = [chr(c) for c in data.flatten() if c != 0]
            return ''.join(chars).rstrip()
        elif data.dtype == np.dtype('uint8'):
            # uint8 data can be decoded directly as UTF-8
            return data.tobytes().decode('utf-8').rstrip('\x00')
        else:
            # Fallback for unexpected dtypes (including uint64 for empty arrays)
            # First check if it's all zeros (empty string)
            if np.all(data == 0):
                return ''
            # Try the old method that may work for some cases
            try:
                return data.astype(dtype=np.uint8).tobytes().decode('utf-8').rstrip('\x00')
            except UnicodeDecodeError:
                # If that fails, try to convert assuming Unicode code points
                chars = [chr(min(c, 0x10FFFF)) for c in data.flatten() if c != 0]
                return ''.join(chars).rstrip()
    elif isinstance(h5var, (h5py.Group, h5py.File)):
        attrs = {}
        for k in h5var:
            if k.startswith('#'):
                continue
            if 'api_key' in k:
                attrs[k] = "API_KEY_REMOVED"
                continue
            if isinstance(h5var[k], h5py.Dataset):
                if not skip_variables:
                    try:
                        attrs[k] = decode_hdf5_matlab_variable(h5var[k], debug_path=debug_path + "/" + k, skip_errors=skip_errors, h5file=h5file)
                    except Exception as e:
                        print(f"Failed to decode variable {k} at {debug_path}: {e}")
                        if not skip_errors:
                            raise e
            else:
                attrs[k] = decode_hdf5_matlab_variable(h5var[k], debug_path=debug_path + "/" + k, skip_errors=skip_errors, h5file=h5file)
        return attrs
    elif isinstance(h5var, h5py.Dataset):
        if h5var.dtype == 'O':
            return dereference_h5value(h5var[:], h5file=h5file)
        else:
            return np.squeeze(h5var[:])
    else:
        return h5var[:]

#
# Legacy MATLAB files (non-HDF5)
#

def extract_legacy_mat_attributes(file, skip_keys=[], skip_errors=True):
    m = scipy.io.loadmat(file, mat_dtype=False, simplify_cells=True, squeeze_me=True)

    attrs = {key: value for key, value in m.items()
             if not key.startswith('__') and key not in skip_keys}

    attrs = strip_api_key(attrs)
    attrs = convert_object_ndarrays_to_lists(attrs)
    return attrs

def strip_api_key(attrs):
    attrs_clean = {}
    for key, value in attrs.items():
        if 'api_key' in key:
            attrs_clean[key] = "API_KEY_REMOVED"
        elif isinstance(value, dict):
            attrs_clean[key] = strip_api_key(value)
        else:
            attrs_clean[key] = value
    return attrs_clean

def convert_object_ndarrays_to_lists(attrs):
    """
    Convert any object ndarray attributes to lists.
    """
    for key, value in attrs.items():
        if isinstance(value, np.ndarray) and value.dtype == 'object':
            attrs[key] = value.tolist()
        elif isinstance(value, dict):
            convert_object_ndarrays_to_lists(value)
        else:
            attrs[key] = value
    return attrs
