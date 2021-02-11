"""
This file holds plotting utility functions that are
independent of the dataset from which to be plotted.
For the current dataset see :class:`qcodes.dataset.plotting`
For the legacy dataset see :class:`qcodes.plots`
"""
import copy
import logging
from typing import Tuple, Union, Optional, Any, cast, Set, Dict
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.colorbar
import matplotlib.collections
import qcodes

log = logging.getLogger(__name__)


"""
General functions (independent of plotting framework)
"""

# turn off limiting percentiles by default
DEFAULT_PERCENTILE = (50, 50)


def auto_range_iqr(data_array: np.ndarray,
                   cutoff_percentile: Union[
                       Tuple[float, float], float] = DEFAULT_PERCENTILE
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
    zrange = zmax-zmin
    pmin, q3, q1, pmax = np.nanpercentile(z, [b, 75, 25, 100-t])
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
        # do not clip more than cutoff_percentile:
        vmin = min(vmin, pmin)
        vmax = max(vmax, pmax)
    return vmin, vmax


# Matplotlib functions

DEFAULT_COLOR_OVER = 'Magenta'
DEFAULT_COLOR_UNDER = 'Cyan'


def _set_colorbar_extend(colorbar: matplotlib.colorbar.Colorbar,
                         extend: str) -> None:
    """
    Workaround for a missing setter for the extend property of a matplotlib
    colorbar.

    The colorbar object in matplotlib has no setter method and setting the
    colorbar extend does not take any effect.
    Calling a subsequent update will cause a runtime
    error because of the internal implementation of the rendering of the
    colorbar. To circumvent this we need to manually specify the property
    `_inside`, which is a slice that describes which of the colorbar levels
    lie inside of the box and it is thereby dependent on the extend.

    Args:
        colorbar: the colorbar for which to set the extend
        extend: the desired extend ('neither', 'both', 'min' or 'max')
    """
    colorbar.extend = extend
    _slice_dict = {'neither': slice(0, None),
                   'both': slice(1, -1),
                   'min': slice(1, None),
                   'max': slice(0, -1)}
    colorbar._inside = _slice_dict[extend]


def apply_color_scale_limits(colorbar: matplotlib.colorbar.Colorbar,
                             new_lim: Tuple[Optional[float], Optional[float]],
                             data_lim: Optional[Tuple[float, float]] = None,
                             data_array: Optional[np.ndarray] = None,
                             color_over: Optional[Any] = DEFAULT_COLOR_OVER,
                             color_under: Optional[Any] = DEFAULT_COLOR_UNDER
                             ) -> None:
    """
    Applies limits to colorscale and updates extend.

    This function applies the limits `new_lim` to the heatmap plot associated
    with the provided `colorbar`, updates the colorbar limits, and also adds
    the colorbar clipping indicators in form of small triangles on the top and
    bottom of the colorbar, according to where the limits are exceeded.

    Args:
        colorbar: The actual colorbar to be updated.
        new_lim: 2-tuple of the desired minimum and maximum value of the color
            scale. If any is `None` it will be left unchanged.
        data_lim: 2-tuple of the actual minimum and maximum value of the data.
            If left out the minimum and maximum are deduced from the provided
            data, or the data associated with the colorbar.
        data_array: Numpy array containing the data to be considered for
            scaling. Must be left out if `data_lim` is provided. If neither is
            provided the data associated with the colorbar is used.
        color_over: Matplotlib color representing the datapoints clipped by the
            upper limit.
        color_under: Matplotlib color representing the datapoints clipped by
            the lower limit.

    Raise:
        RuntimeError: If not received mesh data. Or if you specified both
        `data_lim` and `data_array`.
    """
    # browse the input data and make sure that `data_lim` and `new_lim` are
    # available
    if not isinstance(colorbar.mappable, matplotlib.collections.QuadMesh):
        raise RuntimeError('Can only scale mesh data, but received '
                           f'"{type(colorbar.mappable)}" instead')
    if data_lim is None:
        if data_array is None:
            data_array = colorbar.mappable.get_array()
        data_lim = np.nanmin(data_array), np.nanmax(data_array)
    else:
        if data_array is not None:
            raise RuntimeError('You may not specify `data_lim` and `data_array` '
                               'at the same time. Please refer to the docstring of '
                               '`apply_color_scale_limits for details:\n\n`'
                               + str(apply_color_scale_limits.__doc__))
        else:
            data_lim = cast(Tuple[float, float], tuple(sorted(data_lim)))
    # if `None` is provided in the new limits don't change this limit
    vlim = [new or old for new, old in zip(new_lim, colorbar.mappable.get_clim())]
    # sort limits in case they were given in a wrong order
    vlim = sorted(vlim)
    # detect exceeding colorscale and apply new limits
    exceeds_min, exceeds_max = (data_lim[0] < vlim[0],
                                data_lim[1] > vlim[1])
    if exceeds_min and exceeds_max:
        extend = 'both'
    elif exceeds_min:
        extend = 'min'
    elif exceeds_max:
        extend = 'max'
    else:
        extend = 'neither'
    _set_colorbar_extend(colorbar, extend)
    cmap = copy.copy(colorbar.mappable.get_cmap())
    cmap.set_over(color_over)
    cmap.set_under(color_under)
    colorbar.mappable.set_cmap(cmap)
    colorbar.mappable.set_clim(vlim)


def apply_auto_color_scale(colorbar: matplotlib.colorbar.Colorbar,
                           data_array: Optional[np.ndarray] = None,
                           cutoff_percentile: Union[Tuple[
                               float, float], float] = DEFAULT_PERCENTILE,
                           color_over: Optional[Any] = DEFAULT_COLOR_OVER,
                           color_under: Optional[Any] = DEFAULT_COLOR_UNDER
                           ) -> None:
    """
    Sets the color limits such that outliers are disregarded.

    This method combines the automatic color scaling from
    :meth:`auto_range_iqr` with the color bar setting from
    :meth:`apply_color_scale_limits`.
    If you want to adjust the color scale based on the configuration file
    `qcodesrc.json`, use :meth:`auto_color_scale_from_config`, which is used
    In :func:`qcodes.dataset.plotting.plot_by_id`.

    Args:
        colorbar: The matplotlib colorbar to which to apply.
        data_array: The data on which the statistical analysis is based. If
            left out, the data associated with the `colorbar` is used
        cutoff_percentile: Percentile of data that may maximally be clipped
            on both sides of the distribution.
            If given a tuple (a,b) the percentile limits will be a and 100-b.
        color_over: Matplotlib color representing the datapoints clipped by the
            upper limit.
        color_under: Matplotlib color representing the datapoints clipped by
            the lower limit.

    Raises:
        RuntimeError: If not mesh data.
    """
    if data_array is None:
        if not isinstance(colorbar.mappable, matplotlib.collections.QuadMesh):
            raise RuntimeError('Can only scale mesh data.')
        data_array = colorbar.mappable.get_array()
    new_lim = auto_range_iqr(data_array, cutoff_percentile)
    apply_color_scale_limits(colorbar, new_lim=new_lim, data_array=data_array,
                             color_over=color_over, color_under=color_under)


def auto_color_scale_from_config(colorbar: matplotlib.colorbar.Colorbar,
                                 auto_color_scale: Optional[bool] = None,
                                 data_array: Optional[np.ndarray] = None,
                                 cutoff_percentile: Optional[Union[Tuple[
                                     float, float], float]] = DEFAULT_PERCENTILE,
                                 color_over: Optional[Any] = None,
                                 color_under: Optional[Any] = None,
                                 ) -> None:
    """
    Sets the color limits such that outliers are disregarded, depending on
    the configuration file `qcodesrc.json`. If optional arguments are
    passed the config values are overridden.

    Args:
        colorbar: The colorbar to scale.
        auto_color_scale: Enable smart colorscale. If `False` nothing happens.
            Default value is read from
            ``config.plotting.auto_color_scale.enabled``.
        data_array: Numpy array containing the data to be considered for
            scaling.
        cutoff_percentile: The maxiumum percentile that is cut from the data.
            Default value is read from
            ``config.plotting.auto_color_scale.cutoff_percentile``.
        color_over: Matplotlib color representing the datapoints clipped
            by the upper limit. Default value is read from
            ``config.plotting.auto_color_scale.color_over``.
        color_under: Matplotlib color representing the datapoints clipped
            by the lower limit. Default value is read from
            ``config.plotting.auto_color_scale.color_under``.
    """
    if colorbar is None:
        log.warning('"auto_color_scale_from_config" did not receive a colorbar '
                    'for scaling. Are you trying to scale a plot without '
                    'colorbar?')
        return
    if auto_color_scale is None:
        auto_color_scale = qcodes.config.plotting.auto_color_scale.enabled
    if not auto_color_scale:
        return
    if color_over is None:
        color_over = qcodes.config.plotting.auto_color_scale.color_over
    if color_under is None:
        color_under = qcodes.config.plotting.auto_color_scale.color_under
    if cutoff_percentile is None:
        cutoff_percentile = cast(
            Tuple[float, float],
            tuple(qcodes.config.plotting.auto_color_scale.cutoff_percentile))

    apply_auto_color_scale(colorbar, data_array, cutoff_percentile,
                           color_over, color_under)


_UNITS_FOR_RESCALING: Set[str] = {
    # SI units (without some irrelevant ones like candela)
    # 'kg' is not included because it is 'kilo' and rarely used
    'm', 's', 'A', 'K', 'mol', 'rad', 'Hz', 'N', 'Pa', 'J',
    'W', 'C', 'V', 'F', 'ohm', 'Ohm', 'Ω',
    '\N{GREEK CAPITAL LETTER OMEGA}', 'S', 'Wb', 'T', 'H',
    # non-SI units as well, for convenience
    'eV', 'g'
}

_ENGINEERING_PREFIXES: Dict[int, str] = OrderedDict({
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
     24: "Y"
})

_THRESHOLDS: Dict[float, int] = OrderedDict(
    {10**(scale + 3): scale for scale in _ENGINEERING_PREFIXES.keys()})


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
            prefix = f'$10^{{{selected_scale:.0f}}}$ '
        else:
            prefix = ''
    else:
        prefix = ""
        selected_scale = 0
    return prefix, selected_scale
