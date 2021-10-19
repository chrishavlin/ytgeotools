#!/usr/bin/env python

"""Tests for `ytgeotools` package."""

import numpy as np
import yt

from ytgeotools import ytgeotools


def get_dataset(shp=(2, 3, 4), fields=("density",), cnms=("x", "y", "z")):
    shp = (2, 3, 4)

    data = {f: np.random.random(shp) for f in fields}
    coords = {}
    for cid, c in enumerate(cnms):
        coords[cid] = {"values": np.linspace(0, 1, shp[cid]), "name": c}

    return ytgeotools.Dataset(data, coords)


def check_field(ds, field, fshape):
    assert field in ds.fields
    assert hasattr(ds, field)
    fvar = getattr(ds, field)
    assert fvar.shape == fshape


def test_dataset():
    # simple tests of instantiation and functionality of base dataset
    test_fields = ("test1", "test2")
    fshp = (2, 3, 4)
    cnms = ("x", "y", "z")
    ds = get_dataset(shp=fshp, fields=test_fields, cnms=cnms)

    for fld in test_fields:
        check_field(ds, fld, fshp)

    assert ds._coord_order == [c for c in cnms]
    for cid, c in enumerate(cnms):
        ds.get_coord(c).shape == (fshp[cid],)


def test_to_yt():
    # simple tests of instantiation and functionality of base dataset
    test_fields = ("test1", "test2")
    fshp = (2, 3, 4)
    cnms = ("x", "y", "z")
    ds = get_dataset(shp=fshp, fields=test_fields, cnms=cnms)

    yt_ds = ds.load_uniform_grid()
    for f in test_fields:
        assert ("stream", f) in yt_ds.field_list
        assert type(yt_ds) == yt.frontends.stream.StreamDataset
