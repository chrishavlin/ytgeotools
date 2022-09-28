import os

import xarray as xr

from ytgeotools._testing import create_fake_ds, save_fake_ds


def test_creak_fake_ds():
    flds = ["dvs", "Q"]
    ds = create_fake_ds(fields=flds)
    assert isinstance(ds, xr.Dataset)
    for fld in flds:
        assert hasattr(ds, fld)


def test_save_fake_ds(tmp_path):
    flds = ["dvs", "Q"]

    dir_to_use = tmp_path / "data"
    os.mkdir(dir_to_use)
    fname = dir_to_use / "hello.nc"
    save_fake_ds(fname, fields=flds)
    assert os.path.isfile(fname)

    ds = xr.open_dataset(fname)
    for fld in flds:
        assert hasattr(ds, fld)
