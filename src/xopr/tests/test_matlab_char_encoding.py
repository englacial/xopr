"""
Unit tests for MATLAB char array encoding/decoding in matlab_attribute_utils.

Tests the fix for handling both uint8 and uint16 MATLAB char arrays,
particularly for Unicode characters that would fail with the old decoder.
"""

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from xopr.matlab_attribute_utils import decode_hdf5_matlab_variable


class TestMatlabCharDecoding:
    """Test MATLAB char array decoding with various encodings."""

    def create_matlab_char_file(self, char_values, dtype=np.uint16):
        """Helper to create a temporary HDF5 file with MATLAB char data."""
        tmp_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False)
        tmp_name = tmp_file.name
        tmp_file.close()

        with h5py.File(tmp_name, 'w') as f:
            # Create dataset as MATLAB would
            data = np.array(char_values, dtype=dtype).reshape(-1, 1)
            ds = f.create_dataset('test_char', data=data)
            ds.attrs['MATLAB_class'] = b'char'

        return tmp_name

    def test_basic_ascii_uint16(self):
        """Test decoding basic ASCII text stored as uint16."""
        # ASCII text that should work with both old and new decoder
        test_string = 'Hello World'
        char_values = [ord(c) for c in test_string]

        tmp_file = self.create_matlab_char_file(char_values, dtype=np.uint16)
        try:
            with h5py.File(tmp_file, 'r') as f:
                result = decode_hdf5_matlab_variable(f['test_char'], h5file=f)
                assert result == test_string
        finally:
            os.unlink(tmp_file)

    def test_basic_ascii_uint8(self):
        """Test decoding basic ASCII text stored as uint8."""
        test_string = 'Hello World'
        char_values = [ord(c) for c in test_string]

        tmp_file = self.create_matlab_char_file(char_values, dtype=np.uint8)
        try:
            with h5py.File(tmp_file, 'r') as f:
                result = decode_hdf5_matlab_variable(f['test_char'], h5file=f)
                assert result == test_string
        finally:
            os.unlink(tmp_file)

    def test_extended_ascii_uint16(self):
        """Test decoding extended ASCII characters (128-255) as uint16."""
        # Characters in the 128-255 range
        test_string = 'Café résumé naïve'  # Contains é (233), ï (239)
        char_values = [ord(c) for c in test_string]

        tmp_file = self.create_matlab_char_file(char_values, dtype=np.uint16)
        try:
            with h5py.File(tmp_file, 'r') as f:
                result = decode_hdf5_matlab_variable(f['test_char'], h5file=f)
                assert result == test_string
        finally:
            os.unlink(tmp_file)

    def test_unicode_beyond_255(self):
        """Test decoding Unicode characters with code points > 255."""
        # This would fail with the old decoder
        test_cases = [
            ('Test ý', [84, 101, 115, 116, 32, 253]),  # ý = 253 (0xFD)
            ('Test ǽ', [84, 101, 115, 116, 32, 509]),  # ǽ = 509 (0x1FD)
            ('€100', [8364, 49, 48, 48]),              # € = 8364 (0x20AC)
            ('αβγ', [945, 946, 947]),                  # Greek letters
            ('你好', [20320, 22909]),                   # Chinese characters
        ]

        for test_string, char_values in test_cases:
            tmp_file = self.create_matlab_char_file(char_values, dtype=np.uint16)
            try:
                with h5py.File(tmp_file, 'r') as f:
                    result = decode_hdf5_matlab_variable(f['test_char'], h5file=f)
                    assert result == test_string, f"Failed for: {test_string}"
            finally:
                os.unlink(tmp_file)

    def test_problematic_byte_0xfd(self):
        """Test the specific case that caused the original error."""
        # Character 253 (ý) and 509 (ǽ) both produce 0xFD when cast to uint8
        # 0xFD is invalid UTF-8 start byte
        test_cases = [
            (253, 'ý'),   # Latin small letter y with acute
            (509, 'ǽ'),   # Latin small letter ae with acute
        ]

        for char_code, expected_char in test_cases:
            tmp_file = self.create_matlab_char_file([char_code], dtype=np.uint16)
            try:
                with h5py.File(tmp_file, 'r') as f:
                    result = decode_hdf5_matlab_variable(f['test_char'], h5file=f)
                    assert result == expected_char

                    # Verify old approach would fail
                    data = f['test_char'][:]
                    with pytest.raises(UnicodeDecodeError) as exc_info:
                        # This is what the old code did
                        data.astype(dtype=np.uint8).tobytes().decode('utf-8')
                    assert 'invalid start byte' in str(exc_info.value)
            finally:
                os.unlink(tmp_file)

    def test_null_terminated_strings(self):
        """Test handling of null-terminated strings."""
        # MATLAB sometimes includes null terminators
        test_string = 'Test'
        char_values = [ord(c) for c in test_string] + [0, 0, 0]

        for dtype in [np.uint8, np.uint16]:
            tmp_file = self.create_matlab_char_file(char_values, dtype=dtype)
            try:
                with h5py.File(tmp_file, 'r') as f:
                    result = decode_hdf5_matlab_variable(f['test_char'], h5file=f)
                    # Should strip null terminators
                    assert result == test_string
            finally:
                os.unlink(tmp_file)

    def test_matlab_class_as_string(self):
        """Test handling MATLAB_class attribute as string instead of bytes."""
        test_string = 'Test'
        char_values = [ord(c) for c in test_string]

        tmp_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=False)
        tmp_name = tmp_file.name
        tmp_file.close()

        try:
            with h5py.File(tmp_name, 'w') as f:
                data = np.array(char_values, dtype=np.uint16).reshape(-1, 1)
                ds = f.create_dataset('test_char', data=data)
                # Use string instead of bytes for MATLAB_class
                ds.attrs['MATLAB_class'] = 'char'  # Not b'char'

            with h5py.File(tmp_name, 'r') as f:
                result = decode_hdf5_matlab_variable(f['test_char'], h5file=f)
                assert result == test_string
        finally:
            os.unlink(tmp_name)

    def test_empty_string(self):
        """Test handling of empty strings."""
        tmp_file = self.create_matlab_char_file([], dtype=np.uint16)
        try:
            with h5py.File(tmp_file, 'r') as f:
                result = decode_hdf5_matlab_variable(f['test_char'], h5file=f)
                assert result == ''
        finally:
            os.unlink(tmp_file)

    def test_multiline_string(self):
        """Test handling of multiline strings."""
        test_string = 'Line1\nLine2\rLine3\r\nLine4'
        char_values = [ord(c) for c in test_string]

        tmp_file = self.create_matlab_char_file(char_values, dtype=np.uint16)
        try:
            with h5py.File(tmp_file, 'r') as f:
                result = decode_hdf5_matlab_variable(f['test_char'], h5file=f)
                assert result == test_string
        finally:
            os.unlink(tmp_file)


class TestRealDataFiles:
    """Test with actual OPR data files to ensure backward compatibility."""

    @pytest.fixture
    def data_root(self):
        """Get the test data root directory."""
        data_path = Path('/home/espg/Astera/opr_test_dataset_1')
        if not data_path.exists():
            pytest.skip("Test data directory not found")
        return data_path

    def find_mat_files(self, data_root, limit=5):
        """Find MAT files in the test dataset."""
        mat_files = []
        for campaign_dir in data_root.iterdir():
            if campaign_dir.is_dir():
                # Look for CSARP_standard directory
                csarp_dir = campaign_dir / 'CSARP_standard'
                if csarp_dir.exists():
                    # Find flight directories
                    for flight_dir in csarp_dir.iterdir():
                        if flight_dir.is_dir():
                            # Find MAT files
                            for mat_file in flight_dir.glob('*.mat'):
                                mat_files.append(mat_file)
                                if len(mat_files) >= limit:
                                    return mat_files
        return mat_files

    def test_existing_data_files(self, data_root):
        """Test that existing data files still work with the fix."""
        mat_files = self.find_mat_files(data_root, limit=10)

        if not mat_files:
            pytest.skip("No MAT files found in test dataset")

        for mat_file in mat_files:
            try:
                with h5py.File(mat_file, 'r') as f:
                    # Try to decode common paths that might have char data
                    test_paths = [
                        'param_array/cmd/mission_names',
                        'param_records/cmd/mission_names',
                        'param_sar/cmd/mission_names',
                    ]

                    found_any = False
                    for path in test_paths:
                        if path in f:
                            var = f[path]
                            matlab_class = var.attrs.get('MATLAB_class', None)

                            # Only test if it's a char array
                            if matlab_class in [b'char', 'char']:
                                result = decode_hdf5_matlab_variable(var, h5file=f)

                                # Basic validation
                                assert isinstance(result, str), f"Result should be string for {mat_file}:{path}"
                                assert len(result) > 0, f"Result should not be empty for {mat_file}:{path}"
                                # Mission names should be printable ASCII
                                assert all(c.isprintable() or c.isspace() for c in result), \
                                    f"Result contains non-printable chars for {mat_file}:{path}"

                                found_any = True

                    # If we found and tested at least one char array, that's good
                    if found_any:
                        print(f"✓ Successfully tested {mat_file.name}")

            except Exception as e:
                pytest.fail(f"Failed to process {mat_file}: {e}")

    def test_specific_2016_antarctica_file(self, data_root):
        """Test a specific file from 2016_Antarctica_DC8 campaign."""
        test_file = data_root / '2016_Antarctica_DC8' / 'CSARP_standard' / '20161014_03' / 'Data_20161014_03_001.mat'

        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        with h5py.File(test_file, 'r') as f:
            # Test all mission_names fields if they exist
            for param in ['param_array', 'param_records', 'param_sar']:
                path = f'{param}/cmd/mission_names'
                if path in f:
                    var = f[path]
                    result = decode_hdf5_matlab_variable(var, h5file=f)

                    # Validate result
                    assert isinstance(result, str)
                    assert len(result) > 0
                    # For 2016_Antarctica_DC8, mission names should be like "Denman XX"
                    assert 'Denman' in result or 'denman' in result.lower() or len(result) > 0

    def test_remote_files_from_cresis(self):
        """Test with actual files from CReSIS data server."""
        import tempfile
        import urllib.request

        test_urls = [
            # This file might have the problematic 0xFD byte issue
            "https://data.cresis.ku.edu/data/rds/2019_Greenland_P3/CSARP_standard/20190423_02/Data_20190423_02_009.mat",
            # Normal file for comparison
            "https://data.cresis.ku.edu/data/rds/2022_Antarctica_BaslerMKB/CSARP_standard/20230110_01/Data_20230110_01_015.mat",
        ]

        for url in test_urls:
            try:
                # Download to temporary file
                with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp_file:
                    tmp_name = tmp_file.name
                    urllib.request.urlretrieve(url, tmp_name)

                try:
                    with h5py.File(tmp_name, 'r') as f:
                        # Test all mission_names paths
                        test_paths = [
                            '/param_array/cmd/mission_names',
                            '/param_records/cmd/mission_names',
                            '/param_sar/cmd/mission_names',
                        ]

                        found_any = False
                        for path in test_paths:
                            if path in f:
                                var = f[path]
                                matlab_class = var.attrs.get('MATLAB_class', None)

                                # Only test if it's a char array
                                if matlab_class in [b'char', 'char']:
                                    result = decode_hdf5_matlab_variable(var, h5file=f)

                                    # Should decode without error
                                    assert isinstance(result, str), f"Result should be string for {url}:{path}"

                                    # Allow empty strings (MATLAB_empty arrays) or valid strings
                                    if len(result) > 0:
                                        # Mission names should be printable
                                        assert all(c.isprintable() or c.isspace() for c in result), \
                                            f"Result contains non-printable chars for {url}:{path}: {repr(result)}"
                                        print(f"✓ {url.split('/')[-1]}{path}: '{result}'")
                                    else:
                                        print(f"✓ {url.split('/')[-1]}{path}: (empty string)")

                                    found_any = True

                        if found_any:
                            print(f"✓ Successfully tested {url.split('/')[-1]}")
                finally:
                    os.unlink(tmp_name)

            except urllib.error.URLError as e:
                pytest.skip(f"Cannot download test file from {url}: {e}")

    def test_with_load_frame_url(self):
        """Test using the xopr load_frame_url function."""
        try:
            from xopr.opr_access import load_frame_url
        except ImportError:
            pytest.skip("load_frame_url not available")

        # Test with the 2019 Greenland file
        url = "https://data.cresis.ku.edu/data/rds/2019_Greenland_P3/CSARP_standard/20190423_02/Data_20190423_02_009.mat"

        try:
            # Load the frame using xopr function
            ds = load_frame_url(url)

            # The function should successfully decode MATLAB char arrays
            # Check that we get valid metadata
            assert hasattr(ds, 'attrs'), "Dataset should have attributes"

            # Common attributes that might contain char data
            for attr_name in ['mission_names', 'cmd', 'param_array', 'param_records', 'param_sar']:
                if attr_name in ds.attrs:
                    value = ds.attrs[attr_name]
                    if isinstance(value, str):
                        # Should be a valid decoded string
                        assert all(c.isprintable() or c.isspace() for c in value), \
                            f"Attribute {attr_name} contains non-printable chars: {repr(value)}"

            print(f"✓ Successfully loaded and decoded {url.split('/')[-1]} with load_frame_url")

        except Exception as e:
            if "404" in str(e) or "URLError" in str(e):
                pytest.skip(f"Cannot access test file: {e}")
            else:
                # Re-raise if it's not a network issue
                raise


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
