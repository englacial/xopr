# MIT License
#
# Copyright (c) 2025 Thomas Teisberg, Shane Grigsby
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
xopr is a Python library designed to make accessing Open Polar Radar's data
archives easy, scalable, and reproducible.

See https://github.com/englacial/xopr for details.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from . import geometry as geometry
from .opr_access import OPRConnection as OPRConnection
from .opr_tools import find_intersections as find_intersections
from .opr_tools import merge_frames as merge_frames
from .radar_util import interpolate_to_vertical_grid as interpolate_to_vertical_grid
from .radar_util import layer_twtt_to_range as layer_twtt_to_range

# Import Xarray Dataset accessor
from .xarray_accessor.xopr_accessor import XoprAccessor as XoprAccessor
