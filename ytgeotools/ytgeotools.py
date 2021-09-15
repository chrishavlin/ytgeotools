"""Main module."""
import numpy as np
from yt import load_uniform_grid as lug
from typing import Type

# abstract class
class Dataset:

    geometry = "cartesian"

    def __init__(self, data: dict, coords: dict):
        """
        coords : dict
            {0: {"values": xvals, "name":x},
             1: {"values": yvals, "name":y},
             2: {"values": zvals, "name":z},
            }
        """

        self.fields = list(data.keys())

        for fld, vals in data.items():
            if len(vals.shape) != 3:
                raise ValueError(f"data arrays must be 3d, {fld} is {len(vals.shape)}d")
            setattr(self, fld, vals)

        if all([dim in coords.keys() for dim in [0, 1, 2]]) is False:
            raise ValueError("coords dictionary must contain values for 0, 1, 2")

        self.coords = [coords[dim]["values"] for dim in range(3)]

        self.bbox = np.array(
            [[self.coords[dim].min(), self.coords[dim].max()] for dim in range(3)]
        )
        self._coord_order = [coords[dim]["name"] for dim in range(3)]
        self._coord_hash = {dim: coords[dim]["name"] for dim in range(3)}
        self._coord_hash_r = {coords[dim]["name"]: dim for dim in range(3)}

    def data_dict(self, field_subset: list = None) -> dict:
        if field_subset is None:
            field_subset = self.fields
        return {f: getattr(self, f) for f in field_subset}

    def load_uniform_grid(self):
        return load_uniform_grid(self)


def load_uniform_grid(ds: Type[Dataset], *args, **kwargs):

    data = ds.data_dict()
    sizes = data[list(data.keys())[0]].shape
    dims = tuple(ds._coord_order)
    geometry = (ds.geometry, (dims))
    return lug(data, sizes, 1.0, *args, bbox=ds.bbox, geometry=geometry, **kwargs)
