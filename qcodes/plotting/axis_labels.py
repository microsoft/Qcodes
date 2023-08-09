"""
This file holds scaling logic for axis that is independent of plotting backend
"""
from collections import OrderedDict

import numpy as np

_UNITS_FOR_RESCALING: set[str] = {
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

_ENGINEERING_PREFIXES: dict[int, str] = OrderedDict(
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

_THRESHOLDS: dict[float, int] = OrderedDict(
    {10 ** (scale + 3): scale for scale in _ENGINEERING_PREFIXES.keys()}
)


def find_scale_and_prefix(data: np.ndarray, unit: str) -> tuple[str, int]:
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
