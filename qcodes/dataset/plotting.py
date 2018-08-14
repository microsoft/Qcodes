import logging
from typing import Optional, List, Sequence, Union, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
                                   Sequence[matplotlib.colorbar.Colorbar]]]=None) -> AxesTupleList:
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

    Returns:
        a list of axes and a list of colorbars of the same length.
        The colorbar axes may be None if no colorbar is created (e.g. for
        1D plots)
    """

    def set_axis_labels(ax, data, cax=None):
        if data[0]['label'] == '':
            lbl = data[0]['name']
        else:
            lbl = data[0]['label']
        if data[0]['unit'] == '':
            unit = ''
        else:
            unit = data[0]['unit']
            unit = f"({unit})"
        ax.set_xlabel(f'{lbl} {unit}')

        if data[1]['label'] == '':
            lbl = data[1]['name']
        else:
            lbl = data[1]['label']
        if data[1]['unit'] == '':
            unit = ''
        else:
            unit = data[1]['unit']
            unit = f'({unit})'
        ax.set_ylabel(f'{lbl} {unit}')
        if cax is not None and len(data) > 2:
            if data[2]['label'] == '':
                lbl = data[2]['name']
            else:
                lbl = data[2]['label']
            if data[2]['unit'] == '':
                unit = ''
            else:
                unit = data[2]['unit']
                unit = f'({unit})'
            cax.set_label(f'{lbl} {unit}')

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

            set_axis_labels(ax, data)
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
            set_axis_labels(ax, data, colorbar)
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
