#!/usr/bin/env python

"""Tests for `ytgeotools` package."""

import os

import geopandas as gpd
import pytest
import xarray as xr

import ytgeotools
from ytgeotools._testing import save_fake_ds


@pytest.fixture
def on_disk_nc_file(tmp_path):
    savedir = tmp_path / "data"
    os.mkdir(savedir)
    fname = savedir / "test_nc.nc"
    save_fake_ds(fname, fields=["dvs", "Q"])
    return fname


def test_open_dataset(on_disk_nc_file):
    ds = ytgeotools.open_dataset(on_disk_nc_file)
    ds2 = xr.open_dataset(on_disk_nc_file)

    for c in ds.coords:
        assert c in ds2.coords


def test_profiler(on_disk_nc_file):
    ds = ytgeotools.open_dataset(on_disk_nc_file)
    assert hasattr(ds, "profiler")


def test_surface_gpd(on_disk_nc_file):
    ds = ytgeotools.open_dataset(on_disk_nc_file)

    surf_gpd = ds.profiler.surface_gpd
    assert isinstance(surf_gpd, gpd.GeoDataFrame)
    surf_grid_size = ds.coords["longitude"].size * ds.coords["latitude"].size
    assert len(surf_gpd) == surf_grid_size


def test_profile_extraction(on_disk_nc_file):
    ds = ytgeotools.open_dataset(on_disk_nc_file)
    profiles = ds.profiler.get_profiles("Q")
    gridsize = ds.coords["longitude"].size * ds.coords["latitude"].size
    assert profiles.profiles.shape[0] == gridsize
    assert profiles.profiles.shape[1] == ds.coords["depth"].size
