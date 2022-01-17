"""
This plotting module provides various functions to plot the data measured
using QCoDeS.
"""

import inspect
import logging
from contextlib import contextmanager
from functools import partial
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

import qcodes as qc
from qcodes.dataset.data_set import load_by_run_spec
from qcodes.dataset.data_set_protocol import DataSetProtocol
from qcodes.utils.plotting import auto_color_scale_from_config, find_scale_and_prefix

from .data_export import (
    DSPlotData,
    _get_data_from_ds,
    _strings_as_ints,
    get_1D_plottype,
    get_2D_plottype,
    reshape_2D_data,
)

log = logging.getLogger(__name__)
DB = qc.config["core"]["db_location"]

AxesTuple = Tuple[matplotlib.axes.Axes, matplotlib.colorbar.Colorbar]
AxesTupleList = Tuple[List[matplotlib.axes.Axes],
                      List[Optional[matplotlib.colorbar.Colorbar]]]
# NamedData is the structure _get_data_from_ds returns and that plot_by_id
# uses internally
NamedData = List[List[DSPlotData]]

# list of kwargs for plotting function, so that kwargs can be passed to
# :func:`plot_dataset` and will be distributed to the respective plotting func.
# subplots passes on the kwargs called `fig_kw` to the underlying `figure` call
# First find the kwargs that belong to subplots and than add those that are
# redirected to the `figure`-call.
SUBPLOTS_OWN_KWARGS = set(inspect.signature(plt.subplots).parameters.keys())
SUBPLOTS_OWN_KWARGS.remove('fig_kw')
FIGURE_KWARGS = set(inspect.signature(plt.figure).parameters.keys())
FIGURE_KWARGS.remove('kwargs')
SUBPLOTS_KWARGS = SUBPLOTS_OWN_KWARGS.union(FIGURE_KWARGS)


@contextmanager
def _appropriate_kwargs(plottype: str,
                        colorbar_present: bool,
                        **kwargs: Any) -> Any:
    """
    NB: Only to be used inside :func"`plot_dataset`.

    Context manager to temporarily mutate the plotting kwargs to be appropriate
    for a specific plottype. This is helpful since :func:`plot_dataset` may have
    to generate different kinds of plots (e.g. heatmaps and line plots) and
    the user may want to specify kwargs only relevant to some of them
    (e.g. 'cmap', that line plots cannot consume). Those kwargs should then not
    be passed to all plots, which is what this contextmanager handles.

    Args:
        plottype: The plot type for which the kwargs should be adjusted
        colorbar_present: Is there a non-None colorbar in this plot iteration?
    """

    def linehandler(**kwargs: Any) -> Any:
        kwargs.pop('cmap', None)
        return kwargs

    def heatmaphandler(**kwargs: Any) -> Any:
        if not(colorbar_present) and 'cmap' not in kwargs:
            kwargs['cmap'] = qc.config.plotting.default_color_map
        return kwargs

    plot_handler_mapping = {'1D_line': linehandler,
                            '1D_point': linehandler,
                            '1D_bar': linehandler,
                            '2D_point': heatmaphandler,
                            '2D_grid': heatmaphandler,
                            '2D_scatter': heatmaphandler,
                            '2D_equidistant': heatmaphandler,
                            '2D_unknown': heatmaphandler}

    yield plot_handler_mapping[plottype](**kwargs.copy())


def plot_dataset(
    dataset: DataSetProtocol,
    axes: Optional[Union[matplotlib.axes.Axes, Sequence[matplotlib.axes.Axes]]] = None,
    colorbars: Optional[
        Union[matplotlib.colorbar.Colorbar, Sequence[matplotlib.colorbar.Colorbar]]
    ] = None,
    rescale_axes: bool = True,
    auto_color_scale: Optional[bool] = None,
    cutoff_percentile: Optional[Union[Tuple[float, float], float]] = None,
    complex_plot_type: str = "real_and_imag",
    complex_plot_phase: str = "radians",
    **kwargs: Any,
) -> AxesTupleList:
    """
    Construct all plots for a given dataset

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
    for scatter plots and heatmaps if more than 5000 points are supplied.
    This can be overridden by supplying the `rasterized` kwarg.

    Args:
        dataset: The dataset to plot
        axes: Optional Matplotlib axes to plot on. If not provided, new axes
            will be created
        colorbars: Optional Matplotlib Colorbars to use for 2D plots. If not
            provided, new ones will be created
        rescale_axes: If True, tick labels and units for axes of parameters
            with standard SI units will be rescaled so that, for example,
            '0.00000005' tick label on 'V' axis are transformed to '50' on 'nV'
            axis ('n' is 'nano')
        auto_color_scale: If True, the colorscale of heatmap plots will be
            automatically adjusted to disregard outliers.
        cutoff_percentile: Percentile of data that may maximally be clipped
            on both sides of the distribution.
            If given a tuple (a,b) the percentile limits will be a and 100-b.
            See also the plotting tuorial notebook.
        complex_plot_type: Method for converting complex-valued parameters
            into two real-valued parameters, either ``"real_and_imag"`` or
            ``"mag_and_phase"``. Applicable only for the cases where the
            dataset contains complex numbers
        complex_plot_phase: Format of phase for plotting complex-valued data,
            either ``"radians"`` or ``"degrees"``. Applicable only for the
            cases where the dataset contains complex numbers

    Returns:
        A list of axes and a list of colorbars of the same length. The
        colorbar axes may be `None` if no colorbar is created (e.g. for
        1D plots)

    Config dependencies: (qcodesrc.json)
    """

    # handle arguments and defaults
    subplots_kwargs = {k: kwargs.pop(k)
                       for k in set(kwargs).intersection(SUBPLOTS_KWARGS)}

    # sanitize the complex plotting kwargs
    if complex_plot_type not in ['real_and_imag', 'mag_and_phase']:
        raise ValueError(
            f'Invalid complex plot type given. Received {complex_plot_type} '
            'but can only accept "real_and_imag" or "mag_and_phase".')
    if complex_plot_phase not in ['radians', 'degrees']:
        raise ValueError(
            f'Invalid complex plot phase given. Received {complex_plot_phase} '
            'but can only accept "degrees" or "radians".')
    degrees = complex_plot_phase == "degrees"

    # Retrieve info about the run for the title

    experiment_name = dataset.exp_name
    sample_name = dataset.sample_name
    title = f"Run #{dataset.captured_run_id}, " \
            f"Experiment {experiment_name} ({sample_name})"

    alldata: NamedData = _get_data_from_ds(dataset)
    alldata = _complex_to_real_preparser(alldata,
                                         conversion=complex_plot_type,
                                         degrees=degrees)

    nplots = len(alldata)

    if isinstance(axes, matplotlib.axes.Axes):
        axeslist = [axes]
    else:
        axeslist = cast(List[matplotlib.axes.Axes], axes)
    if isinstance(colorbars, matplotlib.colorbar.Colorbar):
        colorbars = [colorbars]

    if axeslist is None:
        axeslist = []
        for i in range(nplots):
            fig, ax = plt.subplots(1, 1, **subplots_kwargs)
            axeslist.append(ax)
    else:
        if len(subplots_kwargs) != 0:
            raise RuntimeError(f"Error: You cannot provide arguments for the "
                               f"axes/figure creation if you supply your own "
                               f"axes. "
                               f"Provided arguments: {subplots_kwargs}")
        if len(axeslist) != nplots:
            raise RuntimeError(f"Trying to make {nplots} plots, but"
                               f"received {len(axeslist)} axes objects.")

    if colorbars is None:
        colorbars = len(axeslist)*[None]
    new_colorbars: List[matplotlib.colorbar.Colorbar] = []

    for data, ax, colorbar in zip(alldata, axeslist, colorbars):

        if len(data) == 2:  # 1D PLOTTING
            log.debug(f'Doing a 1D plot with kwargs: {kwargs}')

            xpoints = data[0]["data"]
            ypoints = data[1]["data"]

            plottype = get_1D_plottype(xpoints, ypoints)
            log.debug(f'Determined plottype: {plottype}')

            if plottype == '1D_line':
                # sort for plotting
                order = xpoints.argsort()
                xpoints = xpoints[order]
                ypoints = ypoints[order]

                with _appropriate_kwargs(plottype,
                                         colorbar is not None, **kwargs) as k:
                    ax.plot(xpoints, ypoints, **k)
            elif plottype == '1D_point':
                with _appropriate_kwargs(plottype,
                                         colorbar is not None, **kwargs) as k:
                    ax.scatter(xpoints, ypoints, **k)
            elif plottype == '1D_bar':
                with _appropriate_kwargs(plottype,
                                         colorbar is not None, **kwargs) as k:
                    ax.bar(xpoints, ypoints, **k)
            else:
                raise ValueError('Unknown plottype. Something is way wrong.')

            _set_data_axes_labels(ax, data)

            if rescale_axes:
                _rescale_ticks_and_units(ax, data, colorbar)

            new_colorbars.append(None)

            ax.set_title(title)

        elif len(data) == 3:  # 2D PLOTTING
            log.debug(f'Doing a 2D plot with kwargs: {kwargs}')

            if data[2]["shape"] is None:
                xpoints = data[0]["data"].flatten()
                ypoints = data[1]["data"].flatten()
                zpoints = data[2]["data"].flatten()
                plottype = get_2D_plottype(xpoints, ypoints, zpoints)
                log.debug(f"Determined plottype: {plottype}")
            else:
                xpoints = data[0]["data"]
                ypoints = data[1]["data"]
                zpoints = data[2]["data"]
                plottype = "2D_grid"

            how_to_plot = {'2D_grid': plot_on_a_plain_grid,
                           '2D_equidistant': plot_on_a_plain_grid,
                           '2D_point': plot_2d_scatterplot,
                           '2D_unknown': plot_2d_scatterplot}
            plot_func = how_to_plot[plottype]

            with _appropriate_kwargs(plottype,
                                     colorbar is not None, **kwargs) as k:
                ax, colorbar = plot_func(xpoints, ypoints, zpoints,
                                         ax, colorbar,
                                         **k)

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

    if len(axeslist) != len(new_colorbars):
        raise RuntimeError("Non equal number of axes. Perhaps colorbar is "
                           "missing from one of the cases above")
    return axeslist, new_colorbars


def plot_by_id(
    run_id: int,
    axes: Optional[Union[matplotlib.axes.Axes, Sequence[matplotlib.axes.Axes]]] = None,
    colorbars: Optional[
        Union[matplotlib.colorbar.Colorbar, Sequence[matplotlib.colorbar.Colorbar]]
    ] = None,
    rescale_axes: bool = True,
    auto_color_scale: Optional[bool] = None,
    cutoff_percentile: Optional[Union[Tuple[float, float], float]] = None,
    complex_plot_type: str = "real_and_imag",
    complex_plot_phase: str = "radians",
    **kwargs: Any,
) -> AxesTupleList:
    """
    Construct all plots for a given `run_id`. Here `run_id` is an
    alias for `captured_run_id` for historical reasons. See the docs
    of :func:`.load_by_run_spec` for details of loading runs.
    All other arguments are forwarded
    to :func:`.plot_dataset`, see this for more details.
    """

    dataset = load_by_run_spec(captured_run_id=run_id)
    return plot_dataset(dataset,
                        axes,
                        colorbars,
                        rescale_axes,
                        auto_color_scale,
                        cutoff_percentile,
                        complex_plot_type,
                        complex_plot_phase,
                        **kwargs)


def _complex_to_real_preparser(
    alldata: Sequence[Sequence[DSPlotData]], conversion: str, degrees: bool = False
) -> NamedData:
    """
    Convert complex-valued parameters to two real-valued parameters, either
    real and imaginary part or phase and magnitude part

    Args:
        alldata: The data to convert, should be the output of `_get_data_from_ds`
        conversion: the conversion method, either "real_and_imag" or
            "mag_and_phase"
        degrees: Whether to return the phase in degrees. The default is to
            return the phase in radians
    """

    if conversion not in ['real_and_imag', 'mag_and_phase']:
        raise ValueError(f'Invalid conversion given. Received {conversion}, '
                         'but can only accept "real_and_imag" or '
                         '"mag_and_phase".')

    newdata: NamedData = []

    # we build a new NamedData object from the given `alldata` input.
    # Note that the length of `newdata` will be larger than that of `alldata`
    # in the case of complex top-level parameters, because a single complex
    # top-level parameter will be split into two real top-level parameters
    # (that have the same setpoints). This is the reason why we
    # use two variables below, new_group and new_groups.

    for group in alldata:
        new_group = []
        new_groups: NamedData = [[], []]
        for index, parameter in enumerate(group):
            data = parameter["data"]
            if data.dtype.kind == "c":
                p1, p2 = _convert_complex_to_real(
                    parameter, conversion=conversion, degrees=degrees
                )
                if index < len(group) - 1:
                    # if the above condition is met, we are dealing with
                    # complex setpoints
                    new_group.append(p1)
                    new_group.append(p2)
                else:
                    # in this case, we are dealing with a complex top-level
                    # parameter. Also, all the setpoints will have been handled
                    # by now. We split the group into two groups, one for each
                    # new (real) top-level parameter
                    new_groups[0] = new_group.copy()
                    new_groups[1] = new_group.copy()
                    new_groups[0].append(p1)
                    new_groups[1].append(p2)
            else:
                new_group.append(parameter)
        if new_groups == [[], []]:
            # if the above condition is met, the group did not contain a
            # complex top-level parameter and has thus not been split into two
            # new groups
            newdata.append(new_group)
        else:
            newdata.append(new_groups[0])
            newdata.append(new_groups[1])

    return newdata


def _convert_complex_to_real(
    parameter: DSPlotData, conversion: str, degrees: bool
) -> Tuple[DSPlotData, DSPlotData]:
    """
    Do the actual conversion and turn one parameter into two.
    Should only be called from within _complex_to_real_preparser.
    """

    phase_unit = 'deg' if degrees else 'rad'

    converters = {
        'data': {'real_and_imag': lambda x: (np.real(x), np.imag(x)),
                 'mag_and_phase': lambda x: (np.abs(x),
                                             np.angle(x, deg=degrees))},
        'labels': {'real_and_imag': lambda l: (l + ' [real]', l + ' [imag]'),
                   'mag_and_phase': lambda l: (l + ' [mag]', l + ' [phase]')},
        'units': {'real_and_imag': lambda u: (u, u),
                  'mag_and_phase': lambda u: (u, phase_unit)},
        'names': {'real_and_imag': lambda n: (n + '_real', n + '_imag'),
                  'mag_and_phase': lambda n: (n + '_mag', n + '_phase')}}

    new_data = converters['data'][conversion](parameter['data'])
    new_labels = converters['labels'][conversion](parameter['label'])
    new_units = converters['units'][conversion](parameter['unit'])
    new_names = converters['names'][conversion](parameter['name'])

    parameter1: DSPlotData = {
        "name": new_names[0],
        "label": new_labels[0],
        "unit": new_units[0],
        "data": new_data[0],
        "shape": parameter["shape"],
    }

    parameter2: DSPlotData = {
        "name": new_names[1],
        "label": new_labels[1],
        "unit": new_units[1],
        "data": new_data[1],
        "shape": parameter["shape"],
    }

    return parameter1, parameter2


def _get_label_of_data(data_dict: DSPlotData) -> str:
    return data_dict['label'] if data_dict['label'] != '' \
        else data_dict['name']


def _make_axis_label(label: str, unit: str) -> str:
    label = f'{label}'
    if unit != '' and unit is not None:
        label += f' ({unit})'
    return label


def _make_label_for_data_axis(data: Sequence[DSPlotData], axis_index: int) -> str:
    label = _get_label_of_data(data[axis_index])
    unit = data[axis_index]['unit']
    return _make_axis_label(label, unit)


def _set_data_axes_labels(
    ax: matplotlib.axes.Axes,
    data: Sequence[DSPlotData],
    cax: Optional[matplotlib.colorbar.Colorbar] = None,
) -> None:
    ax.set_xlabel(_make_label_for_data_axis(data, 0))
    ax.set_ylabel(_make_label_for_data_axis(data, 1))

    if cax is not None and len(data) > 2:
        cax.set_label(_make_label_for_data_axis(data, 2))


def plot_2d_scatterplot(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                        ax: matplotlib.axes.Axes,
                        colorbar: matplotlib.colorbar.Colorbar = None,
                        **kwargs: Any) -> AxesTuple:
    """
    Make a 2D scatterplot of the data. ``**kwargs`` are passed to matplotlib's
    scatter used for the plotting. By default the data will be rasterized
    in any vector plot if more than 5000 points are supplied. This can be
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

    z_is_stringy = isinstance(z[0], str)

    if z_is_stringy:
        z_strings = np.unique(z)
        z = _strings_as_ints(z)

    cmap = kwargs.pop('cmap') if 'cmap' in kwargs else None

    if z_is_stringy:
        name = cmap.name if hasattr(cmap, 'name') else 'viridis'
        cmap = matplotlib.cm.get_cmap(name, len(z_strings))

    mappable = ax.scatter(x=x, y=y, c=z,
                          rasterized=rasterized, cmap=cmap, **kwargs)

    if colorbar is not None:
        colorbar = ax.figure.colorbar(mappable, ax=ax, cax=colorbar.ax)
    else:
        colorbar = ax.figure.colorbar(mappable, ax=ax)

    if z_is_stringy:
        N = len(z_strings)
        f = (N-1)/N
        colorbar.set_ticks([(n+0.5)*f for n in range(N)])
        colorbar.set_ticklabels(z_strings)

    return ax, colorbar


def plot_on_a_plain_grid(x: np.ndarray,
                         y: np.ndarray,
                         z: np.ndarray,
                         ax: matplotlib.axes.Axes,
                         colorbar: matplotlib.colorbar.Colorbar = None,
                         **kwargs: Any
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
        colorbar: A colorbar to reuse the axis for

    Returns:
        The matplotlib axes handle for plot and colorbar
    """

    log.debug(f'Got kwargs: {kwargs}')

    x_is_stringy = isinstance(x[0], str)
    y_is_stringy = isinstance(y[0], str)
    z_is_stringy = isinstance(z[0], str)

    if x_is_stringy:
        x_strings = np.unique(x)
        x = _strings_as_ints(x)

    if y_is_stringy:
        y_strings = np.unique(y)
        y = _strings_as_ints(y)

    if z_is_stringy:
        z_strings = np.unique(z)
        z = _strings_as_ints(z)

    if x.ndim == 2 and y.ndim == 2 and z.ndim == 2:
        if not np.logical_or(np.any(np.isnan(x)), np.any(np.isnan(y))):
            # data is on a grid that may or may not be
            # rectilinear. Rely on matplotlib to plot
            # this directly
            x_to_plot, y_to_plot, z_to_plot = x, y, z
            num_points = x_to_plot.size
        else:
            x_to_plot, y_to_plot, z_to_plot = _clip_nan_from_shaped_data(x, y, z)
            num_points = x_to_plot.size * y_to_plot.size
    else:
        x_to_plot, y_to_plot, z_to_plot = reshape_2D_data(x, y, z)
        num_points = x_to_plot.size * y_to_plot.size

    if 'rasterized' in kwargs.keys():
        rasterized = kwargs.pop('rasterized')
    else:
        rasterized = num_points > qc.config.plotting.rasterize_threshold

    cmap = kwargs.pop('cmap') if 'cmap' in kwargs else None

    if z_is_stringy:
        name = cmap.name if hasattr(cmap, 'name') else 'viridis'
        cmap = matplotlib.cm.get_cmap(name, len(z_strings))

    colormesh = ax.pcolormesh(
        x_to_plot,
        y_to_plot,
        np.ma.masked_invalid(z_to_plot),
        rasterized=rasterized,
        cmap=cmap,
        shading="nearest",
        **kwargs,
    )

    if x_is_stringy:
        ax.set_xticks(np.arange(len(np.unique(x_strings))))
        ax.set_xticklabels(x_strings)

    if y_is_stringy:
        ax.set_yticks(np.arange(len(np.unique(y_strings))))
        ax.set_yticklabels(y_strings)

    if colorbar is not None:
        colorbar = ax.figure.colorbar(colormesh, ax=ax, cax=colorbar.ax)
    else:
        colorbar = ax.figure.colorbar(colormesh, ax=ax)

    if z_is_stringy:
        N = len(z_strings)
        f = (N-1)/N
        colorbar.set_ticks([(n+0.5)*f for n in range(N)])
        colorbar.set_ticklabels(z_strings)

    return ax, colorbar


def _clip_nan_from_shaped_data(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _on_rectilinear_grid_except_nan(x_data: np.ndarray, y_data: np.ndarray) -> bool:
        """
        check that data is on a rectilinear grid. e.g. all points are the same as the  first
        row and column with the exception of nans. Those represent points not yet measured.
        """
        x_row = x_data[:, 0:1]
        y_row = y_data[0:1, :]
        return (
            np.nanmax(np.abs(x_data - x_row)) == 0
            and np.nanmax(np.abs(y_data - y_row)) == 0
        )

    if _on_rectilinear_grid_except_nan(x, y):
        # clip any row or column where there are nans in the first row
        # or column. Since we fill from here we assume that means that no data
        # has been measured for this row
        x_to_plot, y_to_plot = x[:, 0], y[0, :]

        filter_x = ~np.isnan(x_to_plot)
        filter_y = ~np.isnan(y_to_plot)

        x_to_plot = x_to_plot[filter_x]
        y_to_plot = y_to_plot[filter_y]
        z_to_plot = z[filter_x, :]
        z_to_plot = z_to_plot[:, filter_y].transpose()
    else:
        # fallback to flattening the data and use the same path as
        # non shaped data after filtering the nans.
        # this is not ideal as we loose the shape data but
        # not clear how to do this better. Either return a ragged
        # array or clip all inner dims that have nans completely
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        filter_nans = np.logical_and(~np.isnan(x), ~np.isnan(y))
        x_to_plot, y_to_plot, z_to_plot = reshape_2D_data(
            x[filter_nans], y[filter_nans], z[filter_nans]
        )
    return x_to_plot, y_to_plot, z_to_plot


def _scale_formatter(tick_value: float, pos: int, factor: float) -> str:
    """
    Function for matplotlib.ticker.FuncFormatter that scales the tick values
    according to the given `scale` value.
    """
    return f"{tick_value*factor:g}"


def _make_rescaled_ticks_and_units(
    data_dict: DSPlotData,
) -> Tuple[matplotlib.ticker.FuncFormatter, str]:
    """
    Create a ticks formatter and a new label for the data that is to be used
    on the axes where the data is plotted.

    For example, if values of data are all "nano" in units of volts "V",
    then the plot might be more readable if the tick formatter would show
    values like "1" instead of "0.000000001" while the units in the axis label
    are changed from "V" to "nV" ('n' is for 'nano').

    The units for which unit prefixes are added can be found in
    `qcodes.utils.plotting._UNITS_FOR_RESCALING`. For all other units
    an exponential scaling factor is added to the label i.e.
    `(10^3 x e^2/hbar)`.

    Args:
        data_dict: A dictionary of the following structure
            {
                'data': <1D numpy array of points>,
                'name': <name of the parameter>,
                'label': <label of the parameter or ''>,
                'unit': <unit of the parameter or ''>
            }

    Returns:
        A tuple with the ticks formatter (matlplotlib.ticker.FuncFormatter) and
        the new label.
    """
    unit = data_dict['unit']

    maxval = np.nanmax(np.abs(data_dict['data']))
    prefix, selected_scale = find_scale_and_prefix(maxval, unit)

    new_unit = prefix + unit
    label = _get_label_of_data(data_dict)
    new_label = _make_axis_label(label, new_unit)

    scale_factor = 10**(-selected_scale)
    ticks_formatter = FuncFormatter(
        partial(_scale_formatter, factor=scale_factor))

    return ticks_formatter, new_label


def _rescale_ticks_and_units(
    ax: matplotlib.axes.Axes,
    data: Sequence[DSPlotData],
    cax: matplotlib.colorbar.Colorbar = None,
) -> None:
    """
    Rescale ticks and units for the provided axes as described in
    :func:`~_make_rescaled_ticks_and_units`
    """
    # for x axis
    if not _is_string_valued_array(data[0]['data']):
        x_ticks_formatter, new_x_label = \
            _make_rescaled_ticks_and_units(data[0])
        ax.xaxis.set_major_formatter(x_ticks_formatter)
        ax.set_xlabel(new_x_label)

    # for y axis
    if not _is_string_valued_array(data[1]['data']):
        y_ticks_formatter, new_y_label = \
            _make_rescaled_ticks_and_units(data[1])
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
        values: A 1D numpy array of values

    Returns:
        True, if the array contains string; False otherwise
    """
    return isinstance(values[0], str)
