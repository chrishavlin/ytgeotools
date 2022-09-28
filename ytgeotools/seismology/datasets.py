from abc import ABC, abstractmethod
from typing import Any, List, Type

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ytgeotools.data_manager import data_manager as _dm
from ytgeotools.typing import all_numbers


def _calculate_perturbation(
    ref_data: np.ndarray, field_data: np.ndarray, perturbation_type: str
) -> np.ndarray:
    return_data = field_data - ref_data
    if perturbation_type in ["percent", "fractional"]:
        return_data = return_data / ref_data
        if perturbation_type == "percent":
            return_data = return_data * 100
    return return_data


def _calculate_absolute(
    ref_data: np.ndarray, field_data: np.ndarray, perturbation_type: str
) -> np.ndarray:
    # field_data is a perturbation, ref frame value
    if perturbation_type == "absolute":
        return_data = ref_data + field_data
    elif perturbation_type == "fractional":
        return_data = ref_data * (1 + field_data)
    elif perturbation_type == "percent":
        return_data = ref_data * (1 + field_data / 100)
    return return_data


class ReferenceModel(ABC):
    @abstractmethod
    def interpolate_func(self):
        pass

    @abstractmethod
    def evaluate(self):
        # return model values at a point
        pass

    def _validate_array(self, vals: np.typing.ArrayLike) -> np.ndarray:
        if type(vals) == np.ndarray:
            return vals
        return np.asarray(vals)


def _sanitize_ndarray(input_array: all_numbers) -> all_numbers:
    if type(input_array) == np.ndarray:
        if input_array.shape == ():
            return input_array.item()
    return input_array


class ReferenceModel1D(ReferenceModel):
    """
    A one-dimensional reference model

    Parameters
    ----------
    fieldname : str
        the name of the reference fild
    depth : ArrayLike
        array-like depth values for the reference model
    vals : Arraylike
        array-like model values
    disc_correction : bool
        if True (the default), will apply a discontinuity correction before
        creating the interpolating function. This looks for points at the same
        depth and offsets them by a small value.
    """

    def __init__(
        self,
        fieldname: str,
        depth: np.typing.ArrayLike,
        vals: np.typing.ArrayLike,
        disc_correction: bool = True,
    ):
        self.fieldname = fieldname
        self.depth = self._validate_array(depth)
        self.depth_range = (np.min(self.depth), np.max(self.depth))
        self.vals = self._validate_array(vals)
        self.disc_correction = disc_correction
        self.disc_off_eps = np.finfo(float).eps

    _interpolate_func = None

    @property
    def interpolate_func(self):
        if self._interpolate_func is None:

            depth = self.depth
            vals = self.vals

            if self.disc_correction:
                # deal with discontinuities
                # offset disc depths by a small number
                # disc_vals=[]
                eps_off = self.disc_off_eps
                d_diffs = depth[1:] - depth[0:-1]  # will be 1 element smaller
                disc_i = np.where(d_diffs == 0)[0]  # indices of discontinuties
                depth[disc_i + 1] = depth[disc_i + 1] + eps_off

            # build and return the interpolation function
            self._interpolate_func = interp1d(depth, vals)
        return self._interpolate_func

    def evaluate(self, depths: np.typing.ArrayLike, method: str = "interp") -> Any:
        if method == "interp":
            return _sanitize_ndarray(self.interpolate_func(depths))
        elif method == "nearest":
            raise NotImplementedError

    def perturbation(
        self,
        depths: np.typing.ArrayLike,
        abs_vals: np.typing.ArrayLike,
        method: str = "interp",
        perturbation_type: str = "percent",
    ) -> np.ndarray:

        ref_vals = self.evaluate(depths, method=method)
        pert = _calculate_perturbation(ref_vals, abs_vals, perturbation_type)
        return _sanitize_ndarray(pert)

    def absolute(
        self,
        depths: np.typing.ArrayLike,
        pert_vals: np.typing.ArrayLike,
        method: str = "interp",
        perturbation_type: str = "percent",
    ) -> np.ndarray:

        ref_vals = self.evaluate(depths, method=method)
        abs_vals = _calculate_absolute(ref_vals, pert_vals, perturbation_type)
        return _sanitize_ndarray(abs_vals)


class ReferenceCollection:
    def __init__(self, ref_models: List[ReferenceModel1D]):
        self.reference_fields = []
        for ref_mod in ref_models:
            setattr(self, ref_mod.fieldname, ref_mod)
            self.reference_fields.append(ref_mod.fieldname)


def load_1d_csv_ref(
    filename: str, depth_column: str, value_column: str, **kwargs: Any
) -> Type[ReferenceModel1D]:
    """

    loads a 1D reference model from a CSV file

    Parameters
    ----------
    filename : str
        filename
    depth_column : str
        the name of the depth column
    value_columns :str
        the column of the reference values
    **kwargs : Any
        all kwargs forwarded to pandas.read_csv()

    Returns
    -------
    ReferenceModel1D

    Examples
    --------
    from ytgeotools.seismology.datasets import load_1d_csv_ref
    import numpy as np
    ref = load_1d_csv_ref("IRIS/refModels/AK135F_AVG.csv", 'depth_km', 'Vs_kms')
    ref.evaluate([100., 150.])
    depth_new = np.linspace(ref.depth_range[0], ref.depth_range[1], 400)
    vs = ref.evaluate(depth_new)
    """
    filename = _dm.validate_file(filename)
    df = pd.read_csv(filename, **kwargs)
    d = df[depth_column].to_numpy()
    v = df[value_column].to_numpy()
    return ReferenceModel1D(value_column, d, v, disc_correction=True)


def load_1d_csv_ref_collection(
    filename: str, depth_column: str, value_columns: List[str] = None, **kwargs: Any
) -> Type[ReferenceCollection]:
    """

    loads a 1D reference model collection from a CSV file

    Parameters
    ----------
    filename : str
        filename
    depth_column : str
        the name of the depth column
    value_columns : List[str]
        list of columns to load as reference curves.
    **kwargs : Any
        all kwargs forwarded to pandas.read_csv()

    Returns
    -------
    ReferenceCollection

    Examples
    --------
    from ytgeotools.seismology.datasets import load_1d_csv_ref_collection
    import matplotlib.pyplot as plt
    import numpy as np

    refs = load_1d_csv_ref_collection("IRIS/refModels/AK135F_AVG.csv", 'depth_km')
    print(refs.reference_fields)

    depth_new = np.linspace(0, 500, 50000)
    vs = refs.Vs_kms.evaluate(depth_new)
    vp = refs.Vp_kms.evaluate(depth_new)
    rho = refs.rho_kgm3.evaluate(depth_new)

    f, ax = plt.subplots(1)
    ax.plot(vs, depth_new, label='V_s')
    ax.plot(refs.Vs_kms.vals, refs.Vs_kms.depth,'.k', label='V_s')
    ax.plot(vp, depth_new, label='V_p')
    ax.plot(refs.Vp_kms.vals, refs.Vp_kms.depth,'.k', label='V_p')
    ax.set_ylim(0, 500)
    ax.invert_yaxis()

    """
    filename = _dm.validate_file(filename)
    df = pd.read_csv(filename, **kwargs)
    d = df[depth_column].to_numpy()
    if value_columns is None:
        value_columns = [c for c in df.columns if c != depth_column]

    ref_mods = []
    for vcol in value_columns:
        vals = df[vcol].to_numpy()
        ref_mods.append(ReferenceModel1D(vcol, d, vals))

    return ReferenceCollection(ref_mods)
