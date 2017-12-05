from typing import List, Any, Tuple, Sequence
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


def plot_by_id(run_id: int) -> None:
    """
    Construct all plots for a given run

    Implemented so far:
       * 1D plots
       * 2D plots on filled out rectangular grids
    """

    conn = DataSet(DB).conn

    data = qc.load_by_id(run_id)
    deps = get_dependents(conn, run_id)

    for dep in deps:
        recipe = get_dependencies(conn, dep)

        if len(recipe) == 1:  # 1D PLOTTING
            log.debug('Plotting by id, doing a 1D plot')
            # get plotting info

            # THE MEASURED AXIS
            second_axis_layout = get_layout(conn, dep)
            second_axis_name = second_axis_layout['name']
            second_axis_label = second_axis_layout['label']
            second_axis_unit = second_axis_layout['unit']
            rawdata = data.get_values(second_axis_name)
            second_axis_data = flatten_1D_data_for_plot(rawdata)

            # THE SETPOINT AXIS
            first_axis_layout = get_layout(conn, recipe[0][0])
            first_axis_name = first_axis_layout['name']
            first_axis_label = first_axis_layout['label']
            first_axis_unit = first_axis_layout['unit']
            rawdata = data.get_setpoints(second_axis_name)
            first_axis_data = flatten_1D_data_for_plot(rawdata)

            # perform plotting
            figure, ax = plt.subplots()

            # sort for plotting
            order = first_axis_data.argsort()

            ax.plot(first_axis_data[order], second_axis_data[order])

            if first_axis_label == '':
                lbl = first_axis_name
            else:
                lbl = first_axis_label
            if first_axis_unit == '':
                unit = ''
            else:
                unit = f'({first_axis_unit})'
            ax.set_xlabel(f'{lbl} {unit}')

            if second_axis_label == '':
                lbl = second_axis_name
            else:
                lbl = second_axis_label
            if second_axis_unit == '':
                unit = ''
            else:
                unit = f'({second_axis_unit})'
            ax.set_ylabel(f'{lbl} {unit}')

            return figure

        elif len(recipe) == 2:  # 2D PLOTTING
            log.debug('Plotting by id, doing a 2D plot')

            heatmap_layout = get_layout(conn, dep)
            heatmap_name = heatmap_layout['name']
            heatmap_label = heatmap_layout['label']
            heatmap_unit = heatmap_layout['unit']

            setpoints = data.get_setpoints(heatmap_name)
            heatmap_data = data.get_values(heatmap_name)

            # FIRST SETPOINT AXIS
            first_axis_layout = get_layout(conn, recipe[0][0])
            first_axis_name = first_axis_layout['name']
            first_axis_label = first_axis_layout['label']
            first_axis_unit = first_axis_layout['unit']

            # SECOND SETPOINT AXIS
            second_axis_layout = get_layout(conn, recipe[1][0])
            second_axis_name = second_axis_layout['name']
            second_axis_label = second_axis_layout['label']
            second_axis_unit = second_axis_layout['unit']

            # From the setpoints, figure out which 2D plotter to use
            # TODO: The "decision tree" for what gets plotted how and how
            # we check for that is still unfinished/not optimised
            how_to_plot = {'grid': plot_on_a_plain_grid,
                           'equidistant': plot_on_a_plain_grid}

            log.debug('Plotting by id, determining plottype')
            plottype = _plottype_from_setpoints(setpoints)

            if plottype in how_to_plot.keys():
                log.debug('Plotting by id, doing the actual plot')
                xpoints = flatten_1D_data_for_plot(setpoints[0])
                ypoints = flatten_1D_data_for_plot(setpoints[1])
                zpoints = flatten_1D_data_for_plot(heatmap_data)
                figure = how_to_plot[plottype](xpoints, ypoints, zpoints)

                ax = figure.axes[0]

                if first_axis_label == '':
                    lbl = first_axis_name
                else:
                    lbl = first_axis_label
                if first_axis_unit == '':
                    unit = ''
                else:
                    unit = f'({first_axis_unit})'
                ax.set_xlabel(f'{lbl} {unit}')

                if second_axis_label == '':
                    lbl = second_axis_name
                else:
                    lbl = second_axis_label
                if second_axis_unit == '':
                    unit = ''
                else:
                    unit = f'({second_axis_unit})'
                ax.set_ylabel(f'{lbl} {unit}')

                # TODO: get a colorbar

                return figure

            else:
                raise NotImplementedError('2D data does not seem to be on a '
                                          'grid. A plotting function for this'
                                          ' does not exists yet.')

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
    for (xp, yp, zp) in zip(x, y, z):
        xind = list(xrow).index(xp)
        yind = list(yrow).index(yp)
        z_to_plot[yind, xind] = zp

    fig, ax = plt.subplots()
    ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(z_to_plot))

    return fig


def _rows_from_datapoints(setpoints: np.ndarray) -> List[np.ndarray]:
    """
    Cast the (potentially) unordered setpoints into rows
    of sorted, unique setpoint values. Because of the way they are ordered,
    these rows do not necessarily correspond to actual rows of the scan,
    but they can nonetheless be used to identify certain scan types

    Args:
        setpoints: The raw setpoints as a one-dimensional array

    Returns:
        A list of the rowsg
    """

    rows = []

    while len(setpoints) > 0:
        temp, inds = np.unique(setpoints, return_index=True)
        rows.append(temp)
        setpoints = np.delete(setpoints, inds)

    return rows


def _all_in_group_or_subgroup(setpoints: np.ndarray) -> bool:
    """
    Detects whether the setpoints correspond to two groups of
    of identical rows, one being contained in the other.

    This is the test for whether the setpoints correspond to a
    rectangular sweep. It allows for a single rectangular hole
    in the setpoint grid, thus allowing for an interrupted sweep.
    Note that each axis needs NOT be equidistantly spaced.

    Args:
        setpoints: The setpoints for one dimension as a
            potentially unordered one-dimensional array

    Returns:
        A boolean indicating whether the setpoints meet the
            criterion
    """

    groups = 1
    rows = _rows_from_datapoints(setpoints)
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
    x_check = _all_in_group_or_subgroup(xpoints)
    y_check = _all_in_group_or_subgroup(ypoints)

    xrows = _rows_from_datapoints(xpoints)
    yrows = _rows_from_datapoints(ypoints)

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
