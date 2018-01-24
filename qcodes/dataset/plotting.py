from typing import List, Any, Sequence
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import qcodes as qc
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.sqlite_base import (get_dependencies, get_dependents,
                                        get_layout)

log = logging.getLogger(__name__)
DB = qc.config["core"]["db_location"]


def flatten_1D_data_for_plot(rawdata: Sequence[Sequence[Any]]) -> np.ndarray:
    """
    Cast the return value of the database query to
    a numpy array

    Args:
        rawdata: The return of the get_values function

    Returns:
        A one-dimensional numpy array
    """
    dataarray = np.array(rawdata)
    shape = np.shape(dataarray)
    dataarray = dataarray.reshape(np.product(shape))

    return dataarray


def get_data_by_id(run_id: int) -> List:
    """
    Load data from database and reshapes into 1D arrays with minimal
    name, unit and label metadata.
    """

    data = qc.load_by_id(run_id)
    conn = data.conn
    deps = get_dependents(conn, run_id)
    output = []
    for dep in deps:

        dependencies = get_dependencies(conn, dep)
        data_axis = get_layout(conn, dep)
        rawdata = data.get_values(data_axis['name'])
        data_axis['data'] = flatten_1D_data_for_plot(rawdata)
        raw_setpoint_data = data.get_setpoints(data_axis['name'])
        my_output = []

        for i, dependency in enumerate(dependencies):
            axis = get_layout(conn, dependency[0])
            axis['data'] = flatten_1D_data_for_plot(raw_setpoint_data[i])
            my_output.append(axis)

        my_output.append(data_axis)
        output.append(my_output)
    return output


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
            try:
                plottype = _plottype_from_setpoints([data[0]['data'],
                                                     data[1]['data']])
            except RuntimeError:
                plottype = "unknown"
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
                             f'parameter {dep} depends on {len(recipe)} '
                             'parameters, cannot plot that.')


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

    xrow = np.array(_rows_from_datapoints(x)[0])
    yrow = np.array(_rows_from_datapoints(y)[0])
    nx = len(xrow)
    ny = len(yrow)

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

    # potentially slow method of filling in the data, should be optimised
    log.debug('Sorting data onto grid for plotting')
    z_to_plot = np.full((ny, nx), np.nan)
    x_index = np.zeros_like(x, dtype=np.int)
    y_index = np.zeros_like(y, dtype=np.int)
    for i, xval in enumerate(xrow):
        x_index[np.where(x==xval)[0]] = i
    for i, yval in enumerate(yrow):
        y_index[np.where(y==yval)[0]] = i

    z_to_plot[y_index, x_index] = z
    fig, ax = plt.subplots()
    ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(z_to_plot))

    return fig


def _rows_from_datapoints(inputsetpoints: np.ndarray) -> np.ndarray:
    """
    Cast the (potentially) unordered setpoints into rows
    of sorted, unique setpoint values. Because of the way they are ordered,
    these rows do not necessarily correspond to actual rows of the scan,
    but they can nonetheless be used to identify certain scan types

    Args:
        setpoints: The raw setpoints as a one-dimensional array

    Returns:
        A ndarray of the rows
    """

    rows = []
    setpoints = inputsetpoints.copy()

    # first check if there is only one unique array in which case we can avoid the
    # potentially slow loop below
    temp, inds, count = np.unique(setpoints, return_index=True,
                                  return_counts=True)
    num_repeats_array = np.unique(count)
    if len(num_repeats_array) == 1 and count.sum() == len(inputsetpoints):
        return np.tile(temp, (num_repeats_array[0], 1))
    else:
        rows.append(temp)
        setpoints = np.delete(setpoints, inds)

    while len(setpoints) > 0:
        temp, inds = np.unique(setpoints, return_index=True)
        rows.append(temp)
        setpoints = np.delete(setpoints, inds)

    return np.array(rows)


def _all_in_group_or_subgroup(rows: np.ndarray) -> bool:
    """
    Detects whether the setpoints correspond to two groups of
    of identical rows, one being contained in the other.

    This is the test for whether the setpoints correspond to a
    rectangular sweep. It allows for a single rectangular hole
    in the setpoint grid, thus allowing for an interrupted sweep.
    Note that each axis needs NOT be equidistantly spaced.

    Args:
        rows: The output from _rows_from_datapoints

    Returns:
        A boolean indicating whether the setpoints meet the
            criterion
    """

    groups = 1
    comp_to = rows[0]

    aigos = True
    switchindex = 0

    for rowind, row in enumerate(rows[1:]):
        if np.array_equal(row, comp_to):
            continue
        else:
            groups += 1
            comp_to = row
            switchindex = rowind
            if groups > 2:
                aigos = False
                break

    # if there are two groups, check that the rows of one group
    # are all contained in the rows of the other
    if aigos and switchindex > 0:
        for row in rows[1+switchindex:]:
            if sum([r in rows[0] for r in row]) != len(row):
                aigos = False
                break

    return aigos


def _all_steps_multiples_of_min_step(rows: Sequence[np.ndarray]) -> bool:
    """
    Are all steps integer multiples of the smallest step?
    This is used in determining whether the setpoints correspond
    to a regular grid

    Args:
        rows: the output of _rows_from_datapoints

    Returns:
        The answer to the question
    """

    steps: List[np.ndarray] = []
    for row in rows:
        # TODO: What is an appropriate precision?
        steps += list(np.unique(np.diff(row).round(decimals=15)))

    steps = np.unique((steps))
    remainders = np.mod(steps[1:]/steps[0], 1)

    # TODO: What are reasonable tolerances for allclose?
    asmoms = bool(np.allclose(remainders, np.zeros_like(remainders)))

    return asmoms


def _plottype_from_setpoints(setpoints: Sequence[Sequence[Sequence[Any]]]) -> str:
    """
    For a 2D plot, figure out what kind of visualisation we can use
    to display the data.

    Args:
        setpoints: The raw response of the DataSet's get_setpoints

    Returns:
        A string with the name of a plot routine, e.g. 'grid' or 'voronoi'
    """
    # We first check for being on a "simple" grid, which means that the data
    # FILLS a (possibly non-equidistant) grid with at most a single rectangular
    # hole in it
    #
    # Next we check whether the data can be put on an equidistant grid,
    # but loosen the requirement that anything is filled
    #
    # Finally we just scatter (I think?)

    xpoints = flatten_1D_data_for_plot(setpoints[0])
    ypoints = flatten_1D_data_for_plot(setpoints[1])

    # Now check if this is a simple rectangular sweep,
    # possibly interrupted in the middle of one row

    xrows = _rows_from_datapoints(xpoints)
    yrows = _rows_from_datapoints(ypoints)

    x_check = _all_in_group_or_subgroup(xrows)
    y_check = _all_in_group_or_subgroup(yrows)

    x_check = x_check and (len(xrows[0]) == len(yrows))
    y_check = y_check and (len(yrows[0]) == len(xrows))

    # this is the check that we are on a "simple" grid
    if y_check and x_check:
        return 'grid'

    x_check = _all_steps_multiples_of_min_step(xrows)
    y_check = _all_steps_multiples_of_min_step(yrows)

    # this is the check that we are on an equidistant grid
    if y_check and x_check:
        return 'equidistant'

    raise RuntimeError("Could not find plottype")
