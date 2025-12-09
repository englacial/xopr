import xarray as xr

from xopr.opr_tools import generate_citation
from xopr.xarray_accessor.formatting_html import dataset_repr


@xr.register_dataset_accessor("xopr")
class XoprAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _repr_html_(self):
        return dataset_repr(self._obj)

    @property
    def citation(self) -> str:
        """
        Generate a citation string for the dataset based on its attributes.
        """

        return generate_citation(self._obj)
