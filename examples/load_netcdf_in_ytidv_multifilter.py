import numpy as np
import yt_idv
from ytgeotools.seismology.datasets import XarrayGeoSpherical
from yt_idv.scene_components.blocks import BlockRendering
from yt_idv.scene_data.block_collection import BlockCollection


def refill(vals):
    vals[np.isnan(vals)] = 0.0
    vals[vals > 0] = 0.0
    return vals


def refill2(vals):
    vals[np.isnan(vals)] = 0.0
    vals[vals < 0] = 0.0
    return vals


filename = "IRIS/NWUS11-S_percent.nc"
ds = XarrayGeoSpherical(filename)
ds_yt = ds.interpolate_to_uniform_cartesian(
    ["dvs"],
    N=100,
    return_yt=True,
    rescale_coords=True,
    apply_functions={"dvs": [refill, np.abs]},
)

ds_yt_2 = ds.interpolate_to_uniform_cartesian(
    ["dvs"],
    N=100,
    return_yt=True,
    rescale_coords=True,
    apply_functions={"dvs": [refill2, np.abs]},
)


rc = yt_idv.render_context(height=800, width=800, gui=True)
sg = rc.add_scene(ds_yt, "dvs", no_ghost=True)


dd = ds_yt_2.all_data()
br = BlockCollection(data_source=dd)
br.add_data("dvs")
br_render = BlockRendering(data=br)

sg.data_objects.append(br)
sg.components.append(br_render)
rc.run()
