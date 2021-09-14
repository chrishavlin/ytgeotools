"""Main module."""
import numpy as np
from yt import load_uniform_grid

# abstract class
class Dataset:

    geometry = None

    def __init__(self, data: dict, coords: dict):

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
