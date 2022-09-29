#!/usr/bin/env python

"""Tests for `ytgeotools` package."""

import os

import geopandas as gpd
import pytest
import xarray as xr

import ytgeotools
from ytgeotools._testing import geo_df_for_testing, save_fake_ds


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

    # get profiles inside, outside some bounds
    df = geo_df_for_testing()
    dfl = [
        df,
    ]
    profiles_in = ds.profiler.get_profiles("Q", df_gpds=dfl, drop_null=True)
    profiles_out = ds.profiler.get_profiles("Q", df_gpds=dfl, drop_inside=True)
    n_out = profiles_out.profiles.shape[0]
    n_in = profiles_in.profiles.shape[0]
    assert n_in > 0
    assert n_out > 0
    assert profiles_out.profiles.shape[1] == ds.coords["depth"].size
    assert n_out < gridsize
    assert n_in < gridsize
    assert n_in + n_out == gridsize


def test_filter_surface_gpd(on_disk_nc_file):
    ds = ytgeotools.open_dataset(on_disk_nc_file)
    df = geo_df_for_testing()
    df_inside = ds.profiler.filter_surface_gpd(df, drop_null=True)
    df_outside = ds.profiler.filter_surface_gpd(df, drop_inside=True)
    df_all = ds.profiler.filter_surface_gpd(df)

    assert len(df_inside) > 0
    assert len(df_outside) > 0
    assert len(df_all) > 0
    assert len(df_outside) < len(df_all)
    assert len(df_inside) < len(df_all)
    assert len(df_outside) + len(df_inside) == len(df_all)

    # should error:
    with pytest.raises(
        ValueError, match="Only one of drop_na and drop_inside can be True"
    ):
        _ = ds.profiler.filter_surface_gpd(df, drop_null=True, drop_inside=True,)
