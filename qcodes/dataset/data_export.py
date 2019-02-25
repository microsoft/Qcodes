from typing import List, Any, Sequence, Tuple, Dict, Union
import logging

import numpy as np

from qcodes.dataset.sqlite_base import (get_dependencies, get_dependents,
                                        get_layout)
from qcodes.dataset.data_set import load_by_id

log = logging.getLogger(__name__)


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
    name, unit and label metadata (see `get_layout` function).

    Args:
        run_id: run ID from the database

    Returns:
        a list of lists of dictionaries like this:

    ::

        [
          # each element in this list refers
          # to one dependent (aka measured) parameter
            [
              # each element in this list refers
              # to one independent (aka setpoint) parameter
              # that the dependent parameter depends on;
              # a dictionary with the data and metadata of the dependent
              # parameter is in the *last* element in this list
                ...
                {
                    'data': <1D numpy array of points>,
                    'name': <name of the parameter>,
                    'label': <label of the parameter or ''>,
                    'unit': <unit of the parameter or ''>
                },
                ...
            ],
            ...
        ]

    """

    data = load_by_id(run_id)

    conn = data.conn
    deps = get_dependents(conn, run_id)

    output = []
    for dep in deps:

        dependencies = get_dependencies(conn, dep)

        data_axis: Dict[str, Union[str, np.ndarray]] = get_layout(conn, dep)

        rawdata = data.get_values(data_axis['name'])
        data_axis['data'] = flatten_1D_data_for_plot(rawdata)

        raw_setpoint_data = data.get_setpoints(data_axis['name'])

        output_axes = []

        max_size = 0
        for dependency in dependencies:
            axis: Dict[str, Union[str, np.ndarray]] = get_layout(conn,
                                                                 dependency[0])

            mydata = flatten_1D_data_for_plot(raw_setpoint_data[axis['name']])
            axis['data'] = mydata

            size = mydata.size
            if size > max_size:
                max_size = size

            output_axes.append(axis)

        for axis in output_axes:
            size = axis['data'].size  # type: ignore
            if size < max_size:
                if max_size % size != 0:
                    raise RuntimeError("Inconsistent shapes of data. Got "
                                       f"{size} which is not a whole fraction"
                                       f"of {max_size}")
                axis['data'] = np.repeat(axis['data'], max_size//size)

        output_axes.append(data_axis)

        output.append(output_axes)
    return output


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


def _strings_as_ints(inputarray: np.ndarray) -> np.ndarray:
    """
    Return an integer-valued array version of a string-valued array. Maps, e.g.
    array(['a', 'b', 'c', 'a', 'c']) to array([0, 1, 2, 0, 2]). Useful for
    numerical setpoint analysis

    Args:
        inputarray: A 1D array of strings
    """
    newdata = np.zeros(len(inputarray))
    for n, word in enumerate(np.unique(inputarray)):
        newdata += ((inputarray == word).astype(int)*n)
    return newdata


def get_1D_plottype(xpoints: np.ndarray, ypoints: np.ndarray) -> str:
    """
    Determine plot type for a 1D plot by inspecting the data

    Possible plot types are:
    * '1D_bar' - bar plot
    * '1D_point' - scatter plot
    * '1D_line' - line plot

    Args:
        xpoints: The x-axis values
        ypoints: The y-axis values

    Returns:
        Determined plot type as a string
    """

    if isinstance(xpoints[0], str) and not isinstance(ypoints[0], str):
        if len(xpoints) == len(np.unique(xpoints)):
            return '1D_bar'
        else:
            return '1D_point'
    if isinstance(xpoints[0], str) or isinstance(ypoints[0], str):
        return '1D_point'
    else:
        return datatype_from_setpoints_1d(xpoints)


def datatype_from_setpoints_1d(setpoints: np.ndarray) -> str:
    """
    Figure out what type of visualisation is proper for the
    provided setpoints.

    The type is:
        * '1D_point' (scatter plot) when all setpoints are identical
        * '1D_line' otherwise

    Args:
        setpoints: The x-axis values

    Returns:
        A string representing the plot type as described above
    """
    if np.allclose(setpoints, setpoints[0]):
        return '1D_point'
    else:
        return '1D_line'


def get_2D_plottype(xpoints: np.ndarray,
                    ypoints: np.ndarray,
                    zpoints: np.ndarray) -> str:
    """
    Determine plot type for a 2D plot by inspecting the data

    Plot types are:
    * '2D_grid' - colormap plot for data that is on a grid
    * '2D_equidistant' - colormap plot for data that is on equidistant grid
    * '2D_scatter' - scatter plot
    * '2D_unknown' - returned in case the data did not match any criteria of the
    other plot types

    Args:
        xpoints: The x-axis values
        ypoints: The y-axis values
        zpoints: The z-axis (colorbar) values

    Returns:
        Determined plot type as a string
    """

    plottype = datatype_from_setpoints_2d(xpoints, ypoints)
    return plottype


def datatype_from_setpoints_2d(xpoints: np.ndarray,
                               ypoints: np.ndarray
                               ) -> str:
    """
    For a 2D plot, figure out what kind of visualisation we can use
    to display the data.

    Plot types are:
    * '2D_point' - all setpoint are the same in each direction; one point
    * '2D_grid' - colormap plot for data that is on a grid
    * '2D_equidistant' - colormap plot for data that is on equidistant grid
    * '2D_scatter' - scatter plot
    * '2D_unknown' - returned in case the data did not match any criteria of the
    other plot types

    Args:
        xpoints: The x-axis values
        ypoints: The y-axis values

    Returns:
        A string with the name of the determined plot type
    """
    # We represent categorical data as integer-valued data
    if isinstance(xpoints[0], str):
        xpoints = _strings_as_ints(xpoints)
    if isinstance(ypoints[0], str):
        ypoints = _strings_as_ints(ypoints)

    # First check whether all setpoints are identical along
    # any dimension
    x_all_the_same = np.allclose(xpoints, xpoints[0])
    y_all_the_same = np.allclose(ypoints, ypoints[0])

    if x_all_the_same or y_all_the_same:
        return '2D_point'

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
        return '2D_grid'

    x_check = _all_steps_multiples_of_min_step(xrows)
    y_check = _all_steps_multiples_of_min_step(yrows)

    # this is the check that we are on an equidistant grid
    if y_check and x_check:
        return '2D_equidistant'

    return '2D_unknown'


def reshape_2D_data(x: np.ndarray, y: np.ndarray, z: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xrow = np.array(_rows_from_datapoints(x)[0])
    yrow = np.array(_rows_from_datapoints(y)[0])
    nx = len(xrow)
    ny = len(yrow)

    # potentially slow method of filling in the data, should be optimised
    log.debug('Sorting 2D data onto grid')

    if isinstance(z[0], str):
        z_to_plot = np.full((ny, nx), '', dtype=z.dtype)
    else:
        z_to_plot = np.full((ny, nx), np.nan)
    x_index = np.zeros_like(x, dtype=np.int)
    y_index = np.zeros_like(y, dtype=np.int)
    for i, xval in enumerate(xrow):
        x_index[np.where(x == xval)[0]] = i
    for i, yval in enumerate(yrow):
        y_index[np.where(y == yval)[0]] = i

    z_to_plot[y_index, x_index] = z

    return xrow, yrow, z_to_plot


def get_shaped_data_by_runid(run_id: int) -> List:
    """
    Get data for a given run ID, but shaped according to its nature

    The data might get flattened, and additionally reshaped if it falls on a
    grid (equidistant or not).

    Args:
        run_id: The ID of the run for which to get data

    Returns:
        List of lists of dictionaries, the same as for `get_data_by_id`
    """
    mydata = get_data_by_id(run_id)

    for independet in mydata:
        data_length_long_enough = len(independet) == 3 \
                                  and len(independet[0]['data']) > 0 \
                                  and len(independet[1]['data']) > 0

        if data_length_long_enough:
            independet[0]['data'] = flatten_1D_data_for_plot(
                independet[0]['data'])
            independet[1]['data'] = flatten_1D_data_for_plot(
                independet[1]['data'])

            datatype = datatype_from_setpoints_2d(independet[0]['data'],
                                                  independet[1]['data'])
            if datatype in ('grid', 'equidistant'):
                independet[0]['data'], \
                independet[1]['data'], \
                independet[2]['data'] = reshape_2D_data(independet[0]['data'],
                                                        independet[1]['data'],
                                                        independet[2]['data'])

    return mydata
