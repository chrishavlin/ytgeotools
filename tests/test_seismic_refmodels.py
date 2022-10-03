import numpy as np
import pandas as pd
import pytest

from ytgeotools._testing import create_fake_ds
from ytgeotools.seismology import datasets as ysds


@pytest.fixture
def ref1dmodelvals():
    vs = np.array(
        [
            4.4,
            4.35,
            4.32,
            4.29,
            4.29,
            4.32,
            4.35,
            4.39,
            4.43,
            4.47,
            4.51,
            4.57,
            4.63,
            4.68,
            4.73,
            4.78,
            5.0,
            5.05,
            5.09,
            5.14,
            5.19,
            5.24,
            5.29,
            5.345,
            5.395,
            5.445,
            5.5,
            5.91,
            5.98,
            6.05,
            6.13,
            6.2,
            6.22,
            6.24,
            6.26,
            6.275,
            6.29,
            6.305,
            6.32,
            6.335,
            6.35,
            6.365,
            6.385,
            6.405,
        ]
    )
    return vs


@pytest.fixture
def ref1dmodeldepths():
    depth = np.array(
        [
            38,
            50,
            75,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            275,
            300,
            325,
            350,
            375,
            405,
            406,
            425,
            450,
            475,
            500,
            525,
            550,
            575,
            600,
            625,
            659,
            660,
            675,
            700,
            725,
            750,
            775,
            800,
            825,
            850,
            875,
            900,
            925,
            950,
            975,
            1000,
            1025,
            1050,
        ]
    )
    return depth


def refmodel_tests_1d(ref_model: ysds.ReferenceModel1D):
    # some checks on an instantiated 1d reference model
    assert ref_model.depth.shape == ref_model.vals.shape

    # check that we get back the exact values if evaluating at the reference
    # depth values
    depths = ref_model.depth
    assert np.all(ref_model.evaluate(depths) == ref_model.vals)

    test_dep = np.linspace(ref_model.depth_range[0], ref_model.depth_range[1], 10)
    results = ref_model.evaluate(test_dep)
    assert np.all(np.isreal(results))
    assert results.shape == test_dep.shape


def refmodel_tests_1d_reversed(ref_model: ysds.ReferenceModel1D):
    d_r = ref_model.depth[::-1]
    v_r = ref_model.vals[::-1]
    ref_model_r = ysds.ReferenceModel1D("refmodel.csv", d_r, v_r)

    refmodel_tests_1d(ref_model_r)
    test_deps = np.linspace(ref_model.depth_range[0], ref_model.depth_range[1], 10)
    assert np.all(ref_model_r.evaluate(test_deps) == ref_model.evaluate(test_deps))


def test_1d_eval(tmpdir, ref1dmodeldepths, ref1dmodelvals):

    # check direct instantiation
    ref_model = ysds.ReferenceModel1D("refmodel.csv", ref1dmodeldepths, ref1dmodelvals)
    refmodel_tests_1d(ref_model)

    # test the load from disk. save a model to disc first
    df = pd.DataFrame({"depth": ref1dmodeldepths, "vs": ref1dmodelvals})
    fn = tmpdir.mkdir("modeldir").join("refmodel.csv")
    df.to_csv(str(fn), index=False)
    ref_model = ysds.load_1d_csv_ref(fn, "depth", "vs")
    refmodel_tests_1d(ref_model)

    # check the kwargs
    df.to_csv(str(fn), index=False, sep="|")
    ref_model = ysds.load_1d_csv_ref(fn, "depth", "vs", sep="|")
    refmodel_tests_1d(ref_model)

    # check that reversed values also works and return the same values
    refmodel_tests_1d_reversed(ref_model)


def test_1d_eval_with_disc(ref1dmodeldepths, ref1dmodelvals):
    # build a new model with a discontinuity
    depth = np.concatenate([np.array([10, 20, 20, 21]), ref1dmodeldepths])
    vs = np.concatenate([np.array([2.0, 2.0, 4.0, 4.0]), ref1dmodelvals])
    ref_model = ysds.ReferenceModel1D("refmodel.csv", depth, vs, disc_offset=0.00001)
    vals = ref_model.evaluate([19, 19.9999, 20.0, 20.000001])
    assert np.all(vals[:-1] == 2.0)
    assert np.all(vals[-1] == 4.0)


def test_1d_pert_abs(ref1dmodeldepths, ref1dmodelvals):
    ref_model = ysds.ReferenceModel1D("refmodel.csv", ref1dmodeldepths, ref1dmodelvals)

    test_dep = np.mean(ref_model.depth_range)
    value = ref_model.evaluate(test_dep)
    assert ref_model.perturbation(test_dep, value) == 0.0
    assert ref_model.perturbation(test_dep, value, perturbation_type="percent") == 0.0
    assert (
        ref_model.perturbation(test_dep, value, perturbation_type="fractional") == 0.0
    )

    newvalue = value * 1.1
    dv = newvalue - value
    assert ref_model.perturbation(test_dep, newvalue, perturbation_type=None) == dv
    frac = ref_model.perturbation(test_dep, newvalue, perturbation_type="fractional")
    expected_frac = dv / value
    assert frac == expected_frac
    perc = ref_model.perturbation(test_dep, newvalue, perturbation_type="percent")
    assert perc == dv / value * 100

    # and now back
    assert ref_model.absolute(test_dep, dv, perturbation_type="absolute") == newvalue
    assert (
        ref_model.absolute(test_dep, expected_frac, perturbation_type="fractional")
        == newvalue
    )
    assert ref_model.absolute(test_dep, perc, perturbation_type="percent") == newvalue


def test_ref_1d_collection(tmpdir, ref1dmodeldepths, ref1dmodelvals):

    ref1 = ysds.ReferenceModel1D("vs", ref1dmodeldepths, ref1dmodelvals)
    ref2 = ysds.ReferenceModel1D("vp", ref1dmodeldepths, ref1dmodelvals * 2.0)
    refCollection = ysds.ReferenceCollection([ref1, ref2])

    for mod in ["vs", "vp"]:
        assert hasattr(refCollection, mod)
        assert type(getattr(refCollection, mod)) == ysds.ReferenceModel1D
        refmodel_tests_1d(getattr(refCollection, mod))

    df = pd.DataFrame(
        {"depth": ref1dmodeldepths, "vs": ref1dmodelvals, "vp": ref1dmodelvals * 2.0}
    )

    fn = tmpdir.mkdir("modeldir").join("refmodelvsvp.csv")
    df.to_csv(str(fn), index=False)
    refCollection = ysds.load_1d_csv_ref_collection(fn, "depth")
    for mod in ["vs", "vp"]:
        assert hasattr(refCollection, mod)
        assert type(getattr(refCollection, mod)) == ysds.ReferenceModel1D
        refmodel_tests_1d(getattr(refCollection, mod))

    refCollection = ysds.load_1d_csv_ref_collection(fn, "depth", value_columns=["vp"])
    assert hasattr(refCollection, "vs") is False
    assert hasattr(refCollection, "vp")


def test_with_profiler(tmpdir, ref1dmodeldepths, ref1dmodelvals):
    # just checks that it runs
    ref1 = ysds.ReferenceModel1D("vs", ref1dmodeldepths, ref1dmodelvals)
    ref2 = ysds.ReferenceModel1D("vp", ref1dmodeldepths, ref1dmodelvals * 2.0)
    refCollection = ysds.ReferenceCollection([ref1, ref2])

    ds = create_fake_ds(fields=["dvs", "dvp"])
    _ = ds.profiler.get_absolute(refCollection, field="dvs", ref_model_field="vs")
    _ = ds.profiler.get_perturbation(refCollection, field="dvs", ref_model_field="vs")
    _ = ds.profiler.get_absolute(refCollection, field="dvp", ref_model_field="vp")
    _ = ds.profiler.get_perturbation(refCollection, field="dvp", ref_model_field="vp")
