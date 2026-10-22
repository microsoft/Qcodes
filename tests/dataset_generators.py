from typing import TYPE_CHECKING

import numpy as np

from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.parameters import ParamSpecBase

if TYPE_CHECKING:
    from qcodes.dataset.data_set import DataSet


def dataset_with_outliers_generator(
    ds: "DataSet",
    data_offset: float = 5,
    low_outlier: float = -3,
    high_outlier: float = 1,
    background_noise: bool = True,
) -> "DataSet":
    x = ParamSpecBase("x", "numeric", label="Flux", unit="e^2/hbar")
    t = ParamSpecBase("t", "numeric", label="Time", unit="s")
    z = ParamSpecBase("z", "numeric", label="Majorana number", unit="Anyon")

    idps = InterDependencies_(dependencies={z: (x, t)})
    ds.set_interdependencies(idps)
    ds.mark_started()

    npoints = 50
    xvals = np.linspace(0, 1, npoints)
    tvals = np.linspace(0, 1, npoints)

    rng = np.random.default_rng()

    for counter, xv in enumerate(xvals):
        if background_noise and (
            counter < round(npoints / 2.3) or counter > round(npoints / 1.8)
        ):
            data = rng.random(npoints) - data_offset
        else:
            data = xv * np.linspace(0, 1, npoints)
        if counter == round(npoints / 1.9):
            data[round(npoints / 1.9)] = high_outlier
        if counter == round(npoints / 2.1):
            data[round(npoints / 2.5)] = low_outlier
        ds.add_results([{"x": xv, "t": tv, "z": z} for z, tv in zip(data, tvals)])
    ds.mark_completed()
    return ds
