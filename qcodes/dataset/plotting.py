from typing import List, Any, Sequence
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import qcodes as qc

from .data_export import get_data_by_id, flatten_1D_data_for_plot
from .data_export import datatype_from_setpoints_2d, reshape_2D_data

log = logging.getLogger(__name__)
DB = qc.config["core"]["db_location"]


def plot_by_id(run_id: int) -> Figure:
    def set_axis_labels(ax, data):
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

    """
    Construct all plots for a given run

    Implemented so far:
       * 1D plots
       * 2D plots on filled out rectangular grids
    """

    alldata = get_data_by_id(run_id)

    for data in alldata:

        if len(data) == 2:  # 1D PLOTTING
            log.debug('Plotting by id, doing a 1D plot')

            figure, ax = plt.subplots()

            # sort for plotting
            order = data[0]['data'].argsort()

            ax.plot(data[0]['data'][order], data[1]['data'][order])
            set_axis_labels(ax, data)
            return figure

        elif len(data) == 3:  # 2D PLOTTING
            log.debug('Plotting by id, doing a 2D plot')

            # From the setpoints, figure out which 2D plotter to use
            # TODO: The "decision tree" for what gets plotted how and how
            # we check for that is still unfinished/not optimised
            how_to_plot = {'grid': plot_on_a_plain_grid,
                           'equidistant': plot_on_a_plain_grid}

            log.debug('Plotting by id, determining plottype')
            plottype = datatype_from_setpoints_2d([data[0]['data'],
                                                        data[1]['data']])

            if plottype in how_to_plot.keys():
                log.debug('Plotting by id, doing the actual plot')
                xpoints = flatten_1D_data_for_plot(data[0]['data'])
                ypoints = flatten_1D_data_for_plot(data[1]['data'])
                zpoints = flatten_1D_data_for_plot(data[2]['data'])
                figure = how_to_plot[plottype](xpoints, ypoints, zpoints)

                ax = figure.axes[0]
                set_axis_labels(ax, data)
                # TODO: get a colorbar

                return figure

            else:
                log.warning('2D data does not seem to be on a '
                            'grid. Falling back to scatter plot')
                fig, ax = plt.subplots(1,1)
                xpoints = flatten_1D_data_for_plot(data[0]['data'])
                ypoints = flatten_1D_data_for_plot(data[1]['data'])
                zpoints = flatten_1D_data_for_plot(data[2]['data'])
                ax.scatter(x=xpoints, y=ypoints, c=zpoints)
                set_axis_labels(ax, data)

        else:
            raise ValueError('Multi-dimensional data encountered. '
                             f'parameter {data[-1].name} depends on '
                             f'{len(data-1)} parameters, cannot plot '
                             f'that.')


def plot_on_a_plain_grid(x: np.ndarray, y: np.ndarray,
                         z: np.ndarray) -> Figure:
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

    Returns:
        The matplotlib figure handle
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

    fig, ax = plt.subplots()
    ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(z_to_plot))

    return fig
