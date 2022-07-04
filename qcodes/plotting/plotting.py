"""
This file holds plotting utils that are independent of the backend
"""
import logging
from collections import OrderedDict
from typing import Dict, Set, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# turn off limiting percentiles by default
DEFAULT_PERCENTILE = (50, 50)


def auto_range_iqr(
    data_array: np.ndarray,
    cutoff_percentile: Union[Tuple[float, float], float] = DEFAULT_PERCENTILE,
) -> Tuple[float, float]:
    """
    Get the min and max range of the provided array that excludes outliers
    following the IQR rule.

    This function computes the inter-quartile-range (IQR), defined by Q3-Q1,
    i.e. the percentiles for 75% and 25% of the distribution. The region
    without outliers is defined by [Q1-1.5*IQR, Q3+1.5*IQR].

    Args:
        data_array: Numpy array of arbitrary dimension containing the
            statistical data.
        cutoff_percentile: Percentile of data that may maximally be clipped
            on both sides of the distribution.
            If given a tuple (a,b) the percentile limits will be a and 100-b.

    Returns:
        region limits [vmin, vmax]
    """
    if isinstance(cutoff_percentile, tuple):
        t = cutoff_percentile[0]
        b = cutoff_percentile[1]
    else:
        t = cutoff_percentile
        b = cutoff_percentile
    z = data_array.flatten()
    zmax = np.nanmax(z)
    zmin = np.nanmin(z)
    zrange = zmax - zmin
    pmin, q3, q1, pmax = np.nanpercentile(z, [b, 75, 25, 100 - t])
    IQR = q3 - q1
    # handle corner case of all data zero, such that IQR is zero
    # to counter numerical artifacts do not test IQR == 0, but IQR on its
    # natural scale (zrange) to be smaller than some very small number.
    # also test for zrange to be 0.0 to avoid division by 0.
    # all This is possibly to careful...
    if zrange == 0.0 or IQR / zrange < 1e-8:
        vmin = zmin
        vmax = zmax
    else:
        vmin = max(q1 - 1.5 * IQR, zmin)
        vmax = min(q3 + 1.5 * IQR, zmax)
        # do not clip more than cutoff_percentile:
        vmin = min(vmin, pmin)
        vmax = max(vmax, pmax)
    return vmin, vmax


_UNITS_FOR_RESCALING: Set[str] = {
    # SI units (without some irrelevant ones like candela)
    # 'kg' is not included because it is 'kilo' and rarely used
    "m",
    "s",
    "A",
    "K",
    "mol",
    "rad",
    "Hz",
    "N",
    "Pa",
    "J",
    "W",
    "C",
    "V",
    "F",
    "ohm",
    "Ohm",
    "Ω",
    "\N{GREEK CAPITAL LETTER OMEGA}",
    "S",
    "Wb",
    "T",
    "H",
    # non-SI units as well, for convenience
    "eV",
    "g",
}

_ENGINEERING_PREFIXES: Dict[int, str] = OrderedDict(
    {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "\N{GREEK SMALL LETTER MU}",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
    }
)

_THRESHOLDS: Dict[float, int] = OrderedDict(
    {10 ** (scale + 3): scale for scale in _ENGINEERING_PREFIXES.keys()}
)


def find_scale_and_prefix(data: np.ndarray, unit: str) -> Tuple[str, int]:
    """
    Given a numpy array of data and a unit find the best engineering prefix
    and matching scale that best describes the data.

    The units for which unit prefixes are added can be found in
    ``_UNITS_FOR_RESCALING``. For all other units
    the prefix is an exponential scaling factor e.g. ``10^3``.
    If no unit is given (i.e. an empty string) no scaling is performed.

    Args:
        data: A numpy array of data.
        unit: The unit the data is measured in.
            Should not contain a prefix already.

    Returns:
        tuple of prefix and exponential scale

    """
    maxval = np.nanmax(np.abs(data))
    if unit in _UNITS_FOR_RESCALING:
        for threshold, scale in _THRESHOLDS.items():
            if maxval == 0.0:
                # handle all zero data
                selected_scale = 0
                prefix = _ENGINEERING_PREFIXES[selected_scale]
                break
            if maxval < threshold:
                selected_scale = scale
                prefix = _ENGINEERING_PREFIXES[scale]
                break
        else:
            # here, maxval is larger than the largest threshold
            largest_scale = max(list(_ENGINEERING_PREFIXES.keys()))
            selected_scale = largest_scale
            prefix = _ENGINEERING_PREFIXES[largest_scale]
    elif unit != "":
        if maxval > 0:
            selected_scale = 3 * (np.floor(np.floor(np.log10(maxval)) / 3))
        else:
            selected_scale = 0
        if selected_scale != 0:
            prefix = f"$10^{{{selected_scale:.0f}}}$ "
        else:
            prefix = ""
    else:
        prefix = ""
        selected_scale = 0
    return prefix, selected_scale
