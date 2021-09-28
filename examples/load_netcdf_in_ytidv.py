import numpy as np
import yt_idv
from ytgeotools.seismology.datasets import XarrayGeoSpherical


def refill(vals):
    vals[np.isnan(vals)] = 0.0
    vals[vals > 0] = 0.0
    return vals


filename = "IRIS/NWUS11-S_percent.nc"
ds = XarrayGeoSpherical(filename)
ds_yt = ds.interpolate_to_uniform_cartesian(
    ["dvs"],
    N=50,
    max_dist=50,
    return_yt=True,
    rescale_coords=True,
    apply_functions=[refill, np.abs],
)

rc = yt_idv.render_context(height=800, width=800, gui=True)
sg = rc.add_scene(ds_yt, "dvs", no_ghost=True)
rc.run()
