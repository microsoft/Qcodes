from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.colorbar

import numpy as np

from .auto_range import DEFAULT_PERCENTILE, auto_range_iqr

DEFAULT_COLOR_OVER = "Magenta"
DEFAULT_COLOR_UNDER = "Cyan"

_LOG = logging.getLogger(__name__)

_EXTEND_TYPE = Literal["neither", "both", "min", "max"]


def _set_colorbar_extend(
    colorbar: matplotlib.colorbar.Colorbar,
    extend: _EXTEND_TYPE,
) -> None:
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
    _slice_dict = {
        "neither": slice(0, None),
        "both": slice(1, -1),
        "min": slice(1, None),
        "max": slice(0, -1),
    }
    colorbar._inside = _slice_dict[extend]  # type: ignore[attr-defined]


def apply_color_scale_limits(
    colorbar: matplotlib.colorbar.Colorbar,
    new_lim: tuple[float | None, float | None],
    data_lim: tuple[float, float] | None = None,
    data_array: np.ndarray | None = None,
    color_over: Any = DEFAULT_COLOR_OVER,
    color_under: Any = DEFAULT_COLOR_UNDER,
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
    import matplotlib.collections

    # browse the input data and make sure that `data_lim` and `new_lim` are
    # available
    if not isinstance(colorbar.mappable, matplotlib.collections.QuadMesh):
        raise RuntimeError(
            "Can only scale mesh data, but received "
            f'"{type(colorbar.mappable)}" instead'
        )
    if data_lim is None:
        if data_array is None:
            data_array = cast(np.ndarray, colorbar.mappable.get_array())
        data_lim = np.nanmin(data_array), np.nanmax(data_array)
    elif data_array is not None:
        raise RuntimeError(
            "You may not specify `data_lim` and `data_array` "
            "at the same time. Please refer to the docstring of "
            "`apply_color_scale_limits for details:\n\n`"
            f"{apply_color_scale_limits.__doc__!s}"
        )
    else:
        data_lim = cast(tuple[float, float], tuple(sorted(data_lim)))
    # if `None` is provided in the new limits don't change this limit
    vlim = [new or old for new, old in zip(new_lim, colorbar.mappable.get_clim())]
    # sort limits in case they were given in a wrong order
    vlim = sorted(vlim)
    # detect exceeding colorscale and apply new limits
    exceeds_min, exceeds_max = (data_lim[0] < vlim[0], data_lim[1] > vlim[1])
    if exceeds_min and exceeds_max:
        extend: _EXTEND_TYPE = "both"
    elif exceeds_min:
        extend = "min"
    elif exceeds_max:
        extend = "max"
    else:
        extend = "neither"
    _set_colorbar_extend(colorbar, extend)
    cmap = copy.copy(colorbar.mappable.get_cmap())
    cmap.set_over(color_over)
    cmap.set_under(color_under)
    colorbar.mappable.set_cmap(cmap)
    colorbar.mappable.set_clim(*vlim)


def apply_auto_color_scale(
    colorbar: matplotlib.colorbar.Colorbar,
    data_array: np.ndarray | None = None,
    cutoff_percentile: tuple[float, float] | float = DEFAULT_PERCENTILE,
    color_over: Any | None = DEFAULT_COLOR_OVER,
    color_under: Any | None = DEFAULT_COLOR_UNDER,
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
    import matplotlib.collections

    if data_array is None:
        if not isinstance(colorbar.mappable, matplotlib.collections.QuadMesh):
            raise RuntimeError("Can only scale mesh data.")
        data_array = cast(np.ndarray, colorbar.mappable.get_array())
    assert data_array is not None
    new_lim = auto_range_iqr(data_array, cutoff_percentile)
    apply_color_scale_limits(
        colorbar,
        new_lim=new_lim,
        data_array=data_array,
        color_over=color_over,
        color_under=color_under,
    )


def auto_color_scale_from_config(
    colorbar: matplotlib.colorbar.Colorbar,
    auto_color_scale: bool | None = None,
    data_array: np.ndarray | None = None,
    cutoff_percentile: tuple[float, float] | float | None = DEFAULT_PERCENTILE,
    color_over: Any | None = None,
    color_under: Any | None = None,
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
    import qcodes

    if colorbar is None:
        _LOG.warning(
            '"auto_color_scale_from_config" did not receive a colorbar '
            "for scaling. Are you trying to scale a plot without "
            "colorbar?"
        )
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
            tuple[float, float],
            tuple(qcodes.config.plotting.auto_color_scale.cutoff_percentile),
        )

    apply_auto_color_scale(
        colorbar, data_array, cutoff_percentile, color_over, color_under
    )
