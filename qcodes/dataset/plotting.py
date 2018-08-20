import logging
from collections import OrderedDict
from functools import partial
from typing import Optional, List, Sequence, Union, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import qcodes as qc

from .data_export import get_data_by_id, flatten_1D_data_for_plot
from .data_export import (datatype_from_setpoints_1d,
                          datatype_from_setpoints_2d, reshape_2D_data)

log = logging.getLogger(__name__)
DB = qc.config["core"]["db_location"]

AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]
AxesTupleList = Tuple[List[matplotlib.axes.Axes], List[Optional[matplotlib.colorbar.Colorbar]]]


def plot_by_id(run_id: int,
               axes: Optional[Union[matplotlib.axes.Axes,
                              Sequence[matplotlib.axes.Axes]]]=None,
               colorbars: Optional[Union[matplotlib.colorbar.Colorbar,
                                   Sequence[
                                       matplotlib.colorbar.Colorbar]]]=None,
               rescale_axes: bool=True) -> AxesTupleList:
    """
    Construct all plots for a given run

    Implemented so far:
       * 1D plots
       * 2D plots on filled out rectangular grids
       * 2D scatterplots (fallback)

    The function can optionally be supplied with a matplotlib axes
    or a list of axes that will be used for plotting. The user should ensure
    that the number of axes matches the number of datasets to plot. To plot
    several (1D) dataset in the same axes supply it several times. Colorbar
    axes are created dynamically and cannot be supplied.

    Args:
        run_id: ID of the dataset to plot
        axes: Optional Matplotlib axes to plot on. If non provided new axes will be created
        colorbars: Optional Matplotlib Colorbars to use for 2D plots. If non provided new ones will be createds
        rescale_axes: if True, tick labels and units for axes of parameters
            with standard SI units will be rescaled so that, for example,
            '0.00000005' tick label on 'V' axis are transformed to '50' on 'nV'
            axis ('n' is 'nano')

    Returns:
        a list of axes and a list of colorbars of the same length.
        The colorbar axes may be None if no colorbar is created (e.g. for
        1D plots)
    """
    alldata = get_data_by_id(run_id)
    nplots = len(alldata)

    if isinstance(axes, matplotlib.axes.Axes):
        axes = [axes]
    if isinstance(colorbars, matplotlib.colorbar.Colorbar):
        colorbars = [colorbars]

    if axes is None:
        axes = []
        for i in range(nplots):
            fig, ax = plt.subplots(1, 1)
            axes.append(ax)
    else:
        if len(axes) != nplots:
            raise RuntimeError(f"Trying to make {nplots} plots, but"
                               f"received {len(axes)} axes objects.")

    if colorbars is None:
        colorbars = len(axes)*[None]
    new_colorbars: List[matplotlib.colorbar.Colorbar] = []

    for data, ax, colorbar in zip(alldata, axes, colorbars):

        if len(data) == 2:  # 1D PLOTTING
            log.debug('Plotting by id, doing a 1D plot')

            # sort for plotting
            order = data[0]['data'].argsort()
            xpoints = data[0]['data'][order]
            ypoints = data[1]['data'][order]

            plottype = datatype_from_setpoints_1d(xpoints)

            if plottype == 'line':
                ax.plot(xpoints, ypoints)
            elif plottype == 'point':
                ax.scatter(xpoints, ypoints)
            else:
                raise ValueError('Unknown plottype. Something is way wrong.')

            _set_data_axes_labels(ax, data)

            if rescale_axes:
                _rescale_ticks_and_units(ax, data, colorbar)

            new_colorbars.append(None)

        elif len(data) == 3:  # 2D PLOTTING
            log.debug('Plotting by id, doing a 2D plot')

            # From the setpoints, figure out which 2D plotter to use
            # TODO: The "decision tree" for what gets plotted how and how
            # we check for that is still unfinished/not optimised

            how_to_plot = {'grid': plot_on_a_plain_grid,
                           'equidistant': plot_on_a_plain_grid,
                           'point': plot_2d_scatterplot,
                           'unknown': plot_2d_scatterplot}

            log.debug('Determining plottype')
            plottype = datatype_from_setpoints_2d([data[0]['data'],
                                                   data[1]['data']])
            log.debug(f'Plottype is: "f{plottype}".')
            log.debug('Now doing the actual plot')
            xpoints = flatten_1D_data_for_plot(data[0]['data'])
            ypoints = flatten_1D_data_for_plot(data[1]['data'])
            zpoints = flatten_1D_data_for_plot(data[2]['data'])
            plot_func = how_to_plot[plottype]
            ax, colorbar = plot_func(xpoints, ypoints, zpoints, ax, colorbar)

            _set_data_axes_labels(ax, data, colorbar)

            if rescale_axes:
                _rescale_ticks_and_units(ax, data, colorbar)

            new_colorbars.append(colorbar)

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


def _make_label_for_data_axis(data, axis_index):
    if data[axis_index]['label'] == '':
        label = data[axis_index]['name']
    else:
        label = data[axis_index]['label']

    if data[axis_index]['unit'] == '':
        unit = ''
    else:
        unit = data[axis_index]['unit']

    return _make_axis_label(label, unit)


def _make_axis_label(label, unit):
    return f'{label} ({unit})'


def _set_data_axes_labels(ax, data, cax=None):
    ax.set_xlabel(_make_label_for_data_axis(data, 0))
    ax.set_ylabel(_make_label_for_data_axis(data, 1))

    if cax is not None and len(data) > 2:
        cax.set_label(_make_label_for_data_axis(data, 2))


def plot_2d_scatterplot(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        ax: matplotlib.axes.Axes,
                        colorbar: matplotlib.colorbar.Colorbar=None) -> AxesTuple:
    """
    Make a 2D scatterplot of the data

    Args:
        x: The x values
        y: The y values
        z: The z values
        ax: The axis to plot onto
        colorbar: The colorbar to plot into

    Returns:
        The matplotlib axis handles for plot and colorbar
    """
    mappable = ax.scatter(x=x, y=y, c=z)
    if colorbar is not None:
        colorbar = ax.figure.colorbar(mappable, ax=ax, cax=colorbar.ax)
    else:
        colorbar = ax.figure.colorbar(mappable, ax=ax)
    return ax, colorbar


def plot_on_a_plain_grid(x: np.ndarray, y: np.ndarray,
                         z: np.ndarray,
                         ax: matplotlib.axes.Axes,
                         colorbar: matplotlib.colorbar.Colorbar=None) -> AxesTuple:
    """
    Plot a heatmap of z using x and y as axes. Assumes that the data
    are rectangular, i.e. that x and y together describe a rectangular
    grid. The arrays of x and y need not be sorted in any particular
    way, but data must belong together such that z[n] has x[n] and
    y[n] as setpoints.  The setpoints need not be equidistantly
    spaced, but linear interpolation is used to find the edges of the
    plotted squares.

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

    colormesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(z_to_plot))
    if colorbar is not None:
        colorbar = ax.figure.colorbar(colormesh, ax=ax, cax=colorbar.ax)
    else:
        colorbar = ax.figure.colorbar(colormesh, ax=ax)
    return ax, colorbar


_SI_UNITS = {'m', 's', 'A', 'K', 'mol', 'rad', 'Hz', 'N', 'Pa', 'J',
             'W', 'C', 'V', 'F', 'ohm', 'Ohm',
             '\N{GREEK CAPITAL LETTER OMEGA}', 'S', 'Wb', 'T', 'H'}

_ENGINEERING_PREFIXES = OrderedDict({
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

_THRESHOLDS = OrderedDict(
    {10**(scale + 3): scale for scale in _ENGINEERING_PREFIXES.keys()})


def _scale_formatter(tick_value, pos, factor):
    """
    Function for matplotlib.ticker.FuncFormatter that scales the tick values
    according to the given `scale` value.
    """
    return "{0:g}".format(tick_value*factor)


def _make_rescaled_ticks_and_units(data_dict):
    """
    Create a ticks formatter and a new label for the data that is to be used
    on the axes where the data is plotted.

    ...

    Args:
        data_dict: a dictionary of the following structure
            {
                'data': <1D numpy array of points>,
                'name': <name of the parameter>,
                'label': <label of the parameter or ''>,
                'unit': <unit of the parameter or ''>
            }

    Returns:
        a tuple with the ticks formatter and the new label; in case it is not
        possible to scale, the returned values are None's
    """
    ticks_formatter = None
    new_label = None

    unit = data_dict['unit']

    if unit in _SI_UNITS:
        maxval = np.nanmax(data_dict['data'])

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

        new_unit = prefix + unit
        label = data_dict['label']
        new_label = _make_axis_label(label, new_unit)

        scale_factor = 10**(-selected_scale)
        ticks_formatter = FuncFormatter(
            partial(_scale_formatter, factor=scale_factor))

    return ticks_formatter, new_label


def _rescale_ticks_and_units(ax, data, cax=None):
    """
    Rescale ticks and units for axes that are in standard SI units (i.e. V,
    s, J) to milli (m), kilo (k), etc. Refer to the `_SI_UNITS` for the list
    of units that are rescaled.

    Note that combined or non-standard SI units do not get rescaled.
    """
    # for x axis
    x_ticks_formatter, new_x_label = _make_rescaled_ticks_and_units(data[0])
    if x_ticks_formatter is not None and new_x_label is not None:
        ax.xaxis.set_major_formatter(x_ticks_formatter)
        ax.set_xlabel(new_x_label)

    # for y axis
    y_ticks_formatter, new_y_label = _make_rescaled_ticks_and_units(data[1])
    if y_ticks_formatter is not None and new_y_label is not None:
        ax.yaxis.set_major_formatter(y_ticks_formatter)
        ax.set_ylabel(new_y_label)

    # for z aka colorbar axis
    if cax is not None and len(data) > 2:
        z_ticks_formatter, new_z_label = _make_rescaled_ticks_and_units(data[2])
        if z_ticks_formatter is not None and new_z_label is not None:
            cax.set_label(new_z_label)
            cax.formatter = z_ticks_formatter
            cax.update_ticks()
