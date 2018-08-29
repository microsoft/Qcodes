"""
This file is supposed to hold all the plotting utility functions that are
independent of the plotting backend or the dataset from which to be plotted.
For the current dataset see qcodes.dataset.plotting
For the legacy dataset see qcodes.plots
"""

import logging
from typing import Tuple
import numpy as np

log = logging.getLogger(__name__)

def auto_range_iqr(data_array: np.ndarray) -> Tuple[float, float]:
    """
    Get the min and max range of the provided array that excludes outliers
    following the IQR rule.

    This function computes the inter-quartile-range (IQR), defined by Q3-Q1,
    i.e. the percentiles for 75% and 25% of the destribution. The region
    without outliers is defined by [Q1-1.5*IQR, Q3+1.5*IQR].
    Args:
        data_array: numpy array of arbitrary dimension containing the
            statistical data
    returns:
        vmin, vmax: region limits [vmin, vmax]
    """
    z = data_array.flatten()
    zmax = z.max()
    zmin = z.min()
    zrange = zmax-zmin
    q3, q1 = np.percentile(z, [75, 25])
    IQR = q3-q1
    # handle corner case of all data zero, such that IQR is zero
    # to counter numerical artifacts do not test IQR == 0, but IQR on its
    # natural scale (zrange) to be smaller than some very small number.
    # also test for zrange to be 0.0 to avoid division by 0.
    # all This is possibly to careful...
    if zrange == 0.0 or IQR/zrange < 1e-8:
        vmin = zmin
        vmax = zmax
    else:
        vmin = max(q1 - 1.5*IQR, zmin)
        vmax = min(q3 + 1.5*IQR, zmax)
    return vmin, vmax
