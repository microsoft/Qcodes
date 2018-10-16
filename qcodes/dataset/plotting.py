import logging
from collections import OrderedDict
from functools import partial
from typing import (Optional, List, Sequence, Union, Tuple, Dict,
                    Any, Set)
import inspect
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import qcodes as qc
from qcodes.dataset.data_set import load_by_id
from qcodes.utils.plotting import auto_color_scale_from_config

from .data_export import get_data_by_id, flatten_1D_data_for_plot
from .data_export import (plottype_for_2d_data,
                          plottype_for_3d_data, reshape_2D_data)

log = logging.getLogger(__name__)
DB = qc.config["core"]["db_location"]

AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]
AxesTupleList = Tuple[List[matplotlib.axes.Axes],
                      List[Optional[matplotlib.colorbar.Colorbar]]]
Number = Union[float, int]

# list of kwargs for plotting function, so that kwargs can be passed to
# :meth:`plot_by_id` and will be distributed to the respective plotting func.
# subplots passes on the kwargs called `fig_kw` to the underlying `figure` call
# First find the kwargs that belong to subplots and than add those that are
# redirected to the `figure`-call.
SUBPLOTS_OWN_KWARGS = set(inspect.signature(plt.subplots).parameters.keys())
SUBPLOTS_OWN_KWARGS.remove('fig_kw')
FIGURE_KWARGS = set(inspect.signature(plt.figure).parameters.keys())
FIGURE_KWARGS.remove('kwargs')
SUBPLOTS_KWARGS = SUBPLOTS_OWN_KWARGS.union(FIGURE_KWARGS)

def plot_by_id(run_id: int,
               axes: Optional[Union[matplotlib.axes.Axes,
                              Sequence[matplotlib.axes.Axes]]]=None,
               colorbars: Optional[Union[matplotlib.colorbar.Colorbar,
                                   Sequence[
                                       matplotlib.colorbar.Colorbar]]]=None,
               rescale_axes: bool=True,
               auto_color_scale: Optional[bool]=None,
               cutoff_percentile: Optional[Union[Tuple[Number, Number], Number]]=None,
               **kwargs) -> AxesTupleList:
    """
    Construct all plots for a given run

    Implemented so far:
       * 1D line and scatter plots
       * 2D plots on filled out rectangular grids
       * 2D scatterplots (fallback)

    The function can optionally be supplied with a matplotlib axes or a list
    of axes that will be used for plotting. The user should ensure that the
    number of axes matches the number of datasets to plot. To plot several (1D)
    dataset in the same axes supply it several times. Colorbar axes are
    created dynamically. If colorbar axes are supplied, they will be reused,
    yet new colorbar axes will be returned.

    The plot has a title that comprises run id, experiment name, and sample
    name.

    ``**kwargs`` are passed to matplotlib's relevant plotting functions
    By default the data in any vector plot will be rasterized
    for scatter plots and heatmaps if more that 5000 points are supplied.
    This can be overridden by supplying the `rasterized` kwarg.

    Args:
        run_id:
            ID of the run to plot
        axes:
            Optional Matplotlib axes to plot on. If not provided, new axes
            will be created
        colorbars:
            Optional Matplotlib Colorbars to use for 2D plots. If not
            provided, new ones will be created
        rescale_axes: if True, tick labels and units for axes of parameters
            with standard SI units will be rescaled so that, for example,
            '0.00000005' tick label on 'V' axis are transformed to '50' on 'nV'
            axis ('n' is 'nano')
        auto_color_scale: if True, the colorscale of heatmap plots will be
            automatically adjusted to disregard outliers.
        cutoff_percentile: percentile of data that may maximally be clipped
            on both sides of the distribution.
            If given a tuple (a,b) the percentile limits will be a and 100-b.
            See also the plotting tuorial notebook.

    Returns:
        a list of axes and a list of colorbars of the same length. The
        colorbar axes may be None if no colorbar is created (e.g. for
        1D plots)

    Config dependencies: (qcodesrc.json)
    """
    # handle arguments and defaults
    subplots_kwargs = {k:kwargs.pop(k)
                       for k in set(kwargs).intersection(SUBPLOTS_KWARGS)}

    # Retrieve info about the run for the title
    dataset = load_by_id(run_id)
    experiment_name = dataset.exp_name
    sample_name = dataset.sample_name
    title = f"Run #{run_id}, Experiment {experiment_name} ({sample_name})"

    alldata = get_data_by_id(run_id)
    nplots = len(alldata)

    if isinstance(axes, matplotlib.axes.Axes):
        axes = [axes]
    if isinstance(colorbars, matplotlib.colorbar.Colorbar):
        colorbars = [colorbars]

    if axes is None:
        axes = []
        for i in range(nplots):
            fig, ax = plt.subplots(1, 1, **subplots_kwargs)
            axes.append(ax)
    else:
        if len(subplots_kwargs) != 0:
            raise RuntimeError(f"Error: You cannot provide arguments for the "
                               f"axes/figure creation if you supply your own "
                               f"axes. "
                               f"Provided arguments: {subplots_kwargs}")
        if len(axes) != nplots:
            raise RuntimeError(f"Trying to make {nplots} plots, but"
                               f"received {len(axes)} axes objects.")

    if colorbars is None:
        colorbars = len(axes)*[None]
    new_colorbars: List[matplotlib.colorbar.Colorbar] = []

    for data, ax, colorbar in zip(alldata, axes, colorbars):

        if len(data) == 2:  # 1D PLOTTING
            log.debug('Plotting by id, doing a 1D plot')

            xpoints = data[0]['data']
            ypoints = data[1]['data']

            plottype = plottype_for_2d_data(xpoints, ypoints)

            if plottype == 'line':
                # sort for plotting
                order = xpoints.argsort()
                xpoints = xpoints[order]
                ypoints = ypoints[order]

                ax.plot(xpoints, ypoints, **kwargs)
            elif plottype == 'point':
                ax.scatter(xpoints, ypoints, **kwargs)
            else:
                raise ValueError('Unknown plottype. Something is way wrong.')

            _set_data_axes_labels(ax, data)

            if rescale_axes:
                _rescale_ticks_and_units(ax, data, colorbar)

            new_colorbars.append(None)

            ax.set_title(title)

        elif len(data) == 3:  # 2D PLOTTING
            log.debug('Plotting by id, doing a 2D plot')

            # From the setpoints, figure out which 2D plotter to use
            # TODO: The "decision tree" for what gets plotted how and how
            # we check for that is still unfinished/not optimised

            how_to_plot = {'grid': plot_on_a_plain_grid,
                           'equidistant': plot_on_a_plain_grid,
                           'point': plot_2d_scatterplot,
                           'unknown': plot_2d_scatterplot}

            xpoints = flatten_1D_data_for_plot(data[0]['data'])
            ypoints = flatten_1D_data_for_plot(data[1]['data'])
            zpoints = flatten_1D_data_for_plot(data[2]['data'])

            plottype = plottype_for_3d_data(xpoints, ypoints, zpoints)

            plot_func = how_to_plot[plottype]

            if colorbar is None and 'cmap' not in kwargs:
                kwargs['cmap'] = qc.config.plotting.default_color_map

            ax, colorbar = plot_func(xpoints, ypoints, zpoints, ax, colorbar,
                                     **kwargs)

            _set_data_axes_labels(ax, data, colorbar)

            if rescale_axes:
                _rescale_ticks_and_units(ax, data, colorbar)

            auto_color_scale_from_config(colorbar, auto_color_scale,
                                         zpoints, cutoff_percentile)

            new_colorbars.append(colorbar)

            ax.set_title(title)

        else:
            log.warning('Multi-dimensional data encountered. '
                        f'parameter {data[-1]["name"]} depends on '
                        f'{len(data)-1} parameters, cannot plot '
                        f'that.')
            new_colorbars.append(None)

    if len(axes) != len(new_colorbars):
        raise RuntimeError("Non equal number of axes. Perhaps colorbar is "
                           "missing from one of the cases above")
    return axes, new_colorbars


def _get_label_of_data(data_dict: Dict[str, Any]) -> str:
    return data_dict['label'] if data_dict['label'] != '' else data_dict['name']


def _make_axis_label(label: str, unit: str) -> str:
    label = f'{label}'
    if unit != '' and unit is not None:
        label += f' ({unit})'
    return label


def _make_label_for_data_axis(data: List[Dict[str, Any]], axis_index: int
                              ) -> str:
    label = _get_label_of_data(data[axis_index])
    unit = data[axis_index]['unit']
    return _make_axis_label(label, unit)


def _set_data_axes_labels(ax: matplotlib.axes.Axes,
                          data: List[Dict[str, Any]],
                          cax: Optional[matplotlib.colorbar.Colorbar]=None
                          ) -> None:
    ax.set_xlabel(_make_label_for_data_axis(data, 0))
    ax.set_ylabel(_make_label_for_data_axis(data, 1))

    if cax is not None and len(data) > 2:
        cax.set_label(_make_label_for_data_axis(data, 2))


def plot_2d_scatterplot(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        ax: matplotlib.axes.Axes,
                        colorbar: matplotlib.colorbar.Colorbar=None,
                        **kwargs) -> AxesTuple:
    """
    Make a 2D scatterplot of the data. ``**kwargs`` are passed to matplotlib's
    scatter used for the plotting. By default the data will be rasterized
    in any vector plot if more that 5000 points are supplied. This can be
    overridden by supplying the `rasterized` kwarg.

    Args:
        x: The x values
        y: The y values
        z: The z values
        ax: The axis to plot onto
        colorbar: The colorbar to plot into

    Returns:
        The matplotlib axis handles for plot and colorbar
    """
    if 'rasterized' in kwargs.keys():
        rasterized = kwargs.pop('rasterized')
    else:
        rasterized = len(z) > qc.config.plotting.rasterize_threshold

    z_is_string_valued = isinstance(z[0], str)

    if z_is_string_valued:
        z_int = list(range(len(z)))
        mappable = ax.scatter(x=x, y=y, c=z_int,
                              rasterized=rasterized,**kwargs)
    else:
        mappable = ax.scatter(x=x, y=y, c=z,
                              rasterized=rasterized, **kwargs)

    if colorbar is not None:
        colorbar = ax.figure.colorbar(mappable, ax=ax, cax=colorbar.ax)
    else:
        colorbar = ax.figure.colorbar(mappable, ax=ax)

    if z_is_string_valued:
        colorbar.ax.set_yticklabels(z)

    return ax, colorbar


def plot_on_a_plain_grid(x: np.ndarray,
                         y: np.ndarray,
                         z: np.ndarray,
                         ax: matplotlib.axes.Axes,
                         colorbar: matplotlib.colorbar.Colorbar=None,
                         **kwargs
                         ) -> AxesTuple:
    """
    Plot a heatmap of z using x and y as axes. Assumes that the data
    are rectangular, i.e. that x and y together describe a rectangular
    grid. The arrays of x and y need not be sorted in any particular
    way, but data must belong together such that z[n] has x[n] and
    y[n] as setpoints.  The setpoints need not be equidistantly
    spaced, but linear interpolation is used to find the edges of the
    plotted squares. ``**kwargs`` are passed to matplotlib's pcolormesh used
    for the plotting. By default the data in any vector plot will be rasterized
    if more that 5000 points are supplied. This can be overridden
    by supplying the `rasterized` kwarg.

    Args:
        x: The x values
        y: The y values
        z: The z values
        ax: The axis to plot onto
        colorbar: a colorbar to reuse the axis for

    Returns:
        The matplotlib axes handle for plot and colorbar
    """

    xrow, yrow, z_to_plot = reshape_2D_data(x, y, z)

    # we use a general edge calculator,
    # in the case of non-equidistantly spaced data
    # TODO: is this appropriate for a log ax?
    dxs = np.diff(xrow)/2
    dys = np.diff(yrow)/2
    x_edges = np.concatenate((np.array([xrow[0] - dxs[0]]),
                              xrow[:-1] + dxs,
                              np.array([xrow[-1] + dxs[-1]])))
    y_edges = np.concatenate((np.array([yrow[0] - dys[0]]),
                              yrow[:-1] + dys,
                              np.array([yrow[-1] + dys[-1]])))
    if 'rasterized' in kwargs.keys():
        rasterized = kwargs.pop('rasterized')
    else:
        rasterized = len(x_edges) * len(y_edges) \
                      > qc.config.plotting.rasterize_threshold

    colormesh = ax.pcolormesh(x_edges, y_edges,
                              np.ma.masked_invalid(z_to_plot),
                              rasterized=rasterized,
                              **kwargs)
    if colorbar is not None:
        colorbar = ax.figure.colorbar(colormesh, ax=ax, cax=colorbar.ax)
    else:
        colorbar = ax.figure.colorbar(colormesh, ax=ax)
    return ax, colorbar


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


def _scale_formatter(tick_value: float, pos: int, factor: float) -> str:
    """
    Function for matplotlib.ticker.FuncFormatter that scales the tick values
    according to the given `scale` value.
    """
    return "{0:g}".format(tick_value*factor)


def _make_rescaled_ticks_and_units(data_dict: Dict[str, Any]) \
        -> Tuple[matplotlib.ticker.FuncFormatter, str]:
    """
    Create a ticks formatter and a new label for the data that is to be used
    on the axes where the data is plotted.

    For example, if values of data are all "nano" in units of volts "V",
    then the plot might be more readable if the tick formatter would show
    values like "1" instead of "0.000000001" while the units in the axis label
    are changed from "V" to "nV" ('n' is for 'nano').

    The units for which unit prefixes are added can be found in
    `_UNITS_FOR_RESCALING`. For all other units an exponential scaling factor
    is added to the label i.e. `(10^3 x e^2/hbar)`.

    Args:
        data_dict: a dictionary of the following structure
            {
                'data': <1D numpy array of points>,
                'name': <name of the parameter>,
                'label': <label of the parameter or ''>,
                'unit': <unit of the parameter or ''>
            }

    Returns:
        a tuple with the ticks formatter (matlplotlib.ticker.FuncFormatter) and
        the new label.
    """
    unit = data_dict['unit']

    maxval = np.nanmax(np.abs(data_dict['data']))
    if unit in _UNITS_FOR_RESCALING:
        for threshold, scale in _THRESHOLDS.items():
            if maxval < threshold:
                selected_scale = scale
                prefix = _ENGINEERING_PREFIXES[scale]
                break
        else:
            # here, maxval is larger than the largest threshold
            largest_scale = max(list(_ENGINEERING_PREFIXES.keys()))
            selected_scale = largest_scale
            prefix = _ENGINEERING_PREFIXES[largest_scale]
    else:
        if maxval > 0:
            selected_scale = 3*(np.floor(np.floor(np.log10(maxval))/3))
        else:
            selected_scale = 0
        if selected_scale != 0:
            prefix = f'$10^{{{selected_scale:.0f}}}$ '
        else:
            prefix = ''

    new_unit = prefix + unit
    label = _get_label_of_data(data_dict)
    new_label = _make_axis_label(label, new_unit)

    scale_factor = 10**(-selected_scale)
    ticks_formatter = FuncFormatter(
        partial(_scale_formatter, factor=scale_factor))

    return ticks_formatter, new_label


def _rescale_ticks_and_units(ax: matplotlib.axes.Axes,
                             data: List[Dict[str, Any]],
                             cax: matplotlib.colorbar.Colorbar=None):
    """
    Rescale ticks and units for the provided axes as described in
    :meth:`~_make_rescaled_ticks_and_units`
    """
    # for x axis
    if not _is_string_valued_array(data[0]['data']):
        x_ticks_formatter, new_x_label = _make_rescaled_ticks_and_units(data[0])
        ax.xaxis.set_major_formatter(x_ticks_formatter)
        ax.set_xlabel(new_x_label)

    # for y axis
    if not _is_string_valued_array(data[1]['data']):
        y_ticks_formatter, new_y_label = _make_rescaled_ticks_and_units(data[1])
        ax.yaxis.set_major_formatter(y_ticks_formatter)
        ax.set_ylabel(new_y_label)

    # for z aka colorbar axis
    if cax is not None and len(data) > 2:
        if not _is_string_valued_array(data[2]['data']):
            z_ticks_formatter, new_z_label = \
                _make_rescaled_ticks_and_units(data[2])
            cax.set_label(new_z_label)
            cax.formatter = z_ticks_formatter
            cax.update_ticks()


def _is_string_valued_array(values: np.ndarray) -> bool:
    """
    Check if the given 1D numpy array contains categorical data, or, in other
    words, if it is string-valued.

    Args:
        values:
            a 1D numpy array of values

    Returns:
        True, if the array contains string; False otherwise
    """
    return isinstance(values[0], str)
