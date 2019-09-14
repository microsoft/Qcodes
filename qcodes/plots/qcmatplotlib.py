"""
Live plotting in Jupyter notebooks
using the nbagg backend and matplotlib
"""
from collections.abc import Mapping
from collections.abc import Sequence
from functools import partial
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from matplotlib.transforms import Bbox
from numpy.ma import masked_invalid, getmask

from .base import BasePlot
import qcodes.config
from qcodes.data.data_array import DataArray


class MatPlot(BasePlot):
    """
    Plot x/y lines or x/y/z heatmap data. The first trace may be included
    in the constructor, other traces can be added with MatPlot.add()

    Args:
        *args: Sequence of data to plot. Each element will have its own subplot.
            An element can be a single array, or a sequence of arrays. In the
            latter case, all arrays will be plotted in the same subplot.

        figsize (Tuple[float, float]): (width, height) tuple in inches to pass
            to plt.figure. If not provided, figsize is determined from
            subplots shape

        interval: period in seconds between update checks

        subplots: either a sequence (args) or mapping (kwargs) to pass to
            plt.subplots. default is a single simple subplot (1, 1)
            you can use this to pass kwargs to the plt.figure constructor

        num: integer or None
            specifies the index of the matplotlib figure window to use. If None
            then open a new window

        **kwargs: passed along to MatPlot.add() to add the first data trace
    """

    # Maximum default number of subplot columns. Used to determine shape of
    # subplots when not explicitly provided
    max_subplot_columns = 3

    def __init__(self, *args, figsize=None, interval=1, subplots=None, num=None,
                 **kwargs):
        super().__init__(interval)

        if subplots is None:
            # Subplots is equal to number of args, or 1 if no args provided
            subplots = max(len(args), 1)

        self._init_plot(subplots, figsize, num=num)

        # Add data to plot if passed in args, kwargs are passed to all subplots
        for k, arg in enumerate(args):
            if isinstance(arg, Sequence):
                # Arg consists of multiple elements, add all to same subplot
                for subarg in arg:
                    self[k].add(subarg, **kwargs)
            else:
                # Arg is single element, add to subplot
                self[k].add(arg, **kwargs)

        self.tight_layout()

    def __getitem__(self, key):
        """
        Subplots can be accessed via indices.
        Args:
            key: subplot idx

        Returns:
            Subplot with idx key
        """
        return self.subplots[key]

    def _init_plot(self, subplots=None, figsize=None, num=None):
        if isinstance(subplots, Mapping):
            if figsize is None:
                figsize = (6, 4)
            self.fig, self.subplots = plt.subplots(figsize=figsize, num=num, squeeze=False, **subplots)
        else:
            # Format subplots as tuple (nrows, ncols)
            if isinstance(subplots, int):
                # self.max_subplot_columns defines the limit on how many
                # subplots can be in one row. Adjust subplot rows and columns
                #  accordingly
                nrows = int(np.ceil(subplots / self.max_subplot_columns))
                ncols = min(subplots, self.max_subplot_columns)
                subplots = (nrows, ncols)
            if subplots is None:
                subplots = (1,1)
            if figsize is None:
                # Adjust figsize depending on rows and columns in subplots
                figsize = self.default_figsize(subplots)

            self.fig, self.subplots = plt.subplots(*subplots, num=num,
                                                   figsize=figsize,
                                                   squeeze=False)

        # squeeze=False ensures that subplots is always a 2D array independent
        # of the number of subplots.
        # However the qcodes api assumes that subplots is always a 1D array
        # so flatten here
        self.subplots = self.subplots.flatten()

        for k, subplot in enumerate(self.subplots):
            # Include `add` method to subplots, making it easier to add data to
            # subplots. Note that subplot kwarg is 1-based, to adhere to
            # Matplotlib standards
            subplot.add = partial(self.add, subplot=k+1)

        self.title = self.fig.suptitle('')

    def clear(self, subplots=None, figsize=None):
        """
        Clears the plot window and removes all subplots and traces
        so that the window can be reused.
        """
        self.traces = []
        self.fig.clf()
        self._init_plot(subplots, figsize, num=self.fig.number)

    def add_to_plot(self, use_offset: bool=False, **kwargs):
        """
        adds one trace to this MatPlot.

        Args:
            use_offset: Whether or not ticks can have an offset
            **kwargs: with the exceptions given in the notes below
                (mostly the data!), these are passed directly to
                the matplotlib plotting routine.

        Returns:
            Plot handle for trace

        Notes:
            The following special cases apply for kwargs that are
            not passed directly to the plotting routine.

            * `subplot`: the 1-based axes number to append to (default 1)
            * if kwargs include `z`, we will draw a heatmap (ax.pcolormesh)
              `x`, `y`, and `z` are passed as positional args to pcolormesh
            * without `z` we draw a scatter/lines plot (ax.plot)
              `x`, `y`, and `fmt` (if present) are passed as positional
              args
        """
        # TODO some way to specify overlaid axes?
        # Note that there is a conversion from subplot kwarg, which is
        # 1-based, to subplot idx, which is 0-based.
        ax = self[kwargs.get('subplot', 1) - 1]
        if 'z' in kwargs:
            plot_object = self._draw_pcolormesh(ax, **kwargs)
        else:
            plot_object = self._draw_plot(ax, **kwargs)

        # Specify if axes ticks can have offset or not
        ax.ticklabel_format(useOffset=use_offset)

        self._update_labels(ax, kwargs)
        prev_default_title = self.get_default_title()

        self.traces.append({
            'config': kwargs,
            'plot_object': plot_object
        })

        if prev_default_title == self.title.get_text():
            # in case the user has updated title, don't change it anymore
            self.title.set_text(self.get_default_title())

        return plot_object

    def _update_labels(self, ax, config):
        for axletter in ("x", "y"):
            if axletter+'label' in config:
                label = config[axletter+'label']
            else:
                label = None

            # find if any kwarg from plot.add in the base class
            # matches xunit or yunit, signaling a custom unit
            if axletter+'unit' in config:
                unit = config[axletter+'unit']
            else:
                unit = None

            #  find ( more hope to) unit and label from
            # the data array inside the config
            getter = getattr(ax, "get_{}label".format(axletter))
            if axletter in config and not getter():
                # now if we did not have any kwarg for label or unit
                # fallback to the data_array
                if unit is None:
                    _, unit = self.get_label(config[axletter])
                if label is None:
                    label, _ = self.get_label(config[axletter])
            elif getter():
                # The axis already has label. Assume that is correct
                # We should probably check consistent units and error or warn
                # if not consistent. It's also not at all clear how to handle
                # labels/names as these will in general not be consistent on
                # at least one axis
                return
            axsetter = getattr(ax, "set_{}label".format(axletter))
            axsetter("{} ({})".format(label, unit))

    @staticmethod
    def default_figsize(subplots):
        """
        Provides default figsize for given subplots.
        Args:
            subplots (Tuple[Int, Int]): shape (nrows, ncols) of subplots

        Returns:
            Tuple[float, float]: (width, height) of default figsize
              for given subplot shape
        """
        if not isinstance(subplots, tuple):
            raise TypeError('Subplots {} must be a tuple'.format(subplots))
        return (3 + 3 * subplots[1], 1 + 3 * subplots[0])

    def update_plot(self):
        """
        update the plot. The DataSets themselves have already been updated
        in update, here we just push the changes to the plot.
        """
        # matplotlib doesn't know how to autoscale to a pcolormesh after the
        # first draw (relim ignores it...) so we have to do this ourselves
        bboxes = dict(zip(self.subplots, [[] for p in self.subplots]))

        for trace in self.traces:
            config = trace['config']
            plot_object = trace['plot_object']
            if 'z' in config:
                # pcolormesh doesn't seem to allow editing x and y data, only z
                # so instead, we'll remove and re-add the data.
                if plot_object:
                    plot_object.remove()

                ax = self[config.get('subplot', 1) - 1]
                kwargs = deepcopy(config)
                # figsize may be passed in as part of config.
                # pcolormesh will raise an error if this is passed to it
                # so strip it here.
                if 'figsize' in kwargs:
                    kwargs.pop('figsize')
                plot_object = self._draw_pcolormesh(ax, **kwargs)
                trace['plot_object'] = plot_object

                if plot_object:
                    bboxes[plot_object.axes].append(
                        plot_object.get_datalim(plot_object.axes.transData))
            else:
                for axletter in 'xy':
                    setter = 'set_' + axletter + 'data'
                    if axletter in config:
                        getattr(plot_object, setter)(config[axletter])

        for ax in self.subplots:
            if ax.get_autoscale_on():
                ax.relim()
                if bboxes[ax]:
                    bbox = Bbox.union(bboxes[ax])
                    if np.all(np.isfinite(ax.dataLim)):
                        # should take care of the case of lines + heatmaps
                        # where there's already a finite dataLim from relim
                        ax.dataLim.set(Bbox.union(ax.dataLim, bbox))
                    else:
                        # when there's only a heatmap, relim gives inf bounds
                        # so just completely overwrite it
                        ax.dataLim = bbox
                ax.autoscale()

        self.fig.canvas.draw()

    def _draw_plot(self, ax, y, x=None, fmt=None, subplot=1,
                   xlabel=None,
                   ylabel=None,
                   zlabel=None,
                   xunit=None,
                   yunit=None,
                   zunit=None,
                   **kwargs):
        # NOTE(alexj)stripping out subplot because which subplot we're in is
        # already described by ax, and it's not a kwarg to matplotlib's ax.plot.
        # But I didn't want to strip it out of kwargs earlier because it should
        # stay part of trace['config'].
        args = [arg for arg in [x, y, fmt] if arg is not None]

        line, = ax.plot(*args, **kwargs)
        return line

    @staticmethod
    def _make_args_for_pcolormesh(args_masked, x, y):
        """
        Make args for pcolormesh.
        pcolormesh accepts as args either
        C - a (potentially) masked array
        or
        x, y, C where x and y are the colour box edge arrays and
        are NOT allowed to be masked
        """

        if x is not None and y is not None:
            # If x and y are provided, modify the arrays such that they
            # correspond to grid corners instead of grid centers.
            # This is to ensure that pcolormesh centers correctly and
            # does not ignore edge points.
            args = []
            for k, arr in enumerate(args_masked[:-1]):
                # If a two-dimensional array is provided, only consider the
                # first row/column, depending on the axis
                if arr.ndim > 1:
                    arr = arr[0] if k == 0 else arr[:, 0]
                # first extrapolate to fill any empty values Matplotlib 2.2 no
                # longer support nans in x and y for pcolormesh.
                if np.ma.is_masked(arr[1]):
                    step_size = 1.
                    # Only the first element is not nan. We have to guess the
                    # step size
                else:
                    # the average stepsize is our best guess
                    step_size = np.ma.average(np.ma.diff(arr))

                last_good_value = arr[np.logical_not(arr.mask)][-1]
                extrapolation_start = last_good_value+step_size
                n_invalid = np.sum(arr.mask)
                extrapolation_stop = extrapolation_start+step_size*(n_invalid-1)
                # numpy (1.14) has a deprecation warning related to shared
                # masks. Let's silence this by making sure that this is
                # not shared before modifying the mask
                arr.unshare_mask()
                arr[arr.mask] = np.linspace(extrapolation_start,
                                            extrapolation_stop,
                                            num=n_invalid)
                # Now shift array to get edge coordinates rather than
                # centre coordinates
                arr_shift = np.insert(arr, 0, arr[0])
                arr_shift[0] -= step_size/2
                arr_shift[1:] += step_size/2
                args.append(arr_shift)
            args.append(args_masked[-1])
        else:
            # Only the masked value of z is used as a mask
            args = args_masked[-1:]

        return args

    def _draw_pcolormesh(self, ax, z, x=None, y=None, subplot=1,
                         xlabel=None,
                         ylabel=None,
                         zlabel=None,
                         xunit=None,
                         yunit=None,
                         zunit=None,
                         nticks=None,
                         **kwargs):
        # NOTE(alexj)stripping out subplot because which subplot we're in is already
        # described by ax, and it's not a kwarg to matplotlib's ax.plot. But I
        # didn't want to strip it out of kwargs earlier because it should stay
        # part of trace['config'].

        args_masked = [masked_invalid(arg) for arg in [x, y, z]
                       if arg is not None]

        if np.any([np.all(getmask(arg)) for arg in args_masked]):
            # if the z array is masked, don't draw at all
            # there's nothing to draw, and anyway it throws a warning
            # pcolormesh does not accept masked x and y axes, so we do not need
            # to check for them.
            return False
        if 'cmap' not in kwargs:
            kwargs['cmap'] = qcodes.config['gui']['defaultcolormap']

        args = self._make_args_for_pcolormesh(args_masked, x, y)

        pc = ax.pcolormesh(*args, **kwargs)

        # Set x and y limits if arrays are provided
        if x is not None and y is not None:
            ax.set_xlim(np.nanmin(args[0]), np.nanmax(args[0]))
            ax.set_ylim(np.nanmin(args[1]), np.nanmax(args[1]))

        # Specify preferred number of ticks with labels
        if nticks and ax.get_xscale() != 'log' and ax.get_yscale != 'log':
            ax.locator_params(nbins=nticks)

        if getattr(ax, 'qcodes_colorbar', None):
            # update_normal doesn't seem to work...
            ax.qcodes_colorbar.update_bruteforce(pc)
        else:
            # TODO: what if there are several colormeshes on this subplot,
            # do they get the same colorscale?
            # We should make sure they do, and have it include
            # the full range of both.
            ax.qcodes_colorbar = self.fig.colorbar(pc, ax=ax)

            # ideally this should have been in _update_labels, but
            # the colorbar doesn't necessarily exist there.
            # I guess we could create the colorbar no matter what,
            # and just give it a dummy mappable to start, so we could
            # put this where it belongs.
            if zunit is None:
                _, zunit = self.get_label(z)
            if zlabel is None:
                zlabel, _ = self.get_label(z)

            label = "{} ({})".format(zlabel, zunit)
            ax.qcodes_colorbar.set_label(label)

        # Scale colors if z has elements
        cmin = np.nanmin(args_masked[-1])
        cmax = np.nanmax(args_masked[-1])
        ax.qcodes_colorbar.set_clim(cmin, cmax)

        return pc

    def save(self, filename=None):
        """
        Save current plot to filename, by default
        to the location corresponding to the default
        title.

        Args:
            filename (Optional[str]): Location of the file
        """
        default = "{}.png".format(self.get_default_title())
        filename = filename or default
        self.fig.savefig(filename)

    def tight_layout(self):
        """
        Perform a tight layout on the figure. A bit of additional spacing at
        the top is also added for the title.
        """
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

    def rescale_axis(self):
        """
        Rescale axis and units for axis that are in standard units
        i.e. V, s J ... to m μ, m
        This scales units defined in BasePlot.standardunits only
        to avoid prefixes on combined or non standard units
        """
        def scale_formatter(i, pos, scale):
            return "{0:g}".format(i * scale)

        for i, subplot in enumerate(self.subplots):
            traces = [trace for trace in self.traces if trace['config'].get('subplot', None) == i+1]
            if not traces:
                continue
            else:
                # TODO: include all traces when calculating maxval etc.
                trace = traces[0]
            for axis in 'x', 'y', 'z':
                if axis in trace['config'] and isinstance(trace['config'][axis], DataArray):
                    unit = trace['config'][axis].unit
                    label = trace['config'][axis].label
                    maxval = np.nanmax(abs(trace['config'][axis].ndarray))
                    units_to_scale = self.standardunits

                    # allow values up to a <1000. i.e. nV is used up to 1000 nV
                    prefixes = ['n', 'μ', 'm', '', 'k', 'M', 'G']
                    thresholds = [10**(-6 + 3*n) for n in range(len(prefixes))]
                    scales = [10**(9 - 3*n) for n in range(len(prefixes))]

                    if unit in units_to_scale:
                        scale = 1
                        new_unit = unit
                        for prefix, threshold, trialscale in zip(prefixes,
                                                                 thresholds,
                                                                 scales):
                            if maxval < threshold:
                                scale = trialscale
                                new_unit = prefix + unit
                                break
                        # special case the largest
                        if maxval > thresholds[-1]:
                            scale = scales[-1]
                            new_unit = prefixes[-1] + unit

                        tx = ticker.FuncFormatter(
                            partial(scale_formatter, scale=scale))
                        new_label = "{} ({})".format(label, new_unit)
                        if axis in ('x', 'y'):
                            getattr(subplot,
                                    "{}axis".format(axis)).set_major_formatter(
                                tx)
                            getattr(subplot, "set_{}label".format(axis))(
                                new_label)
                        else:
                            subplot.qcodes_colorbar.formatter = tx
                            subplot.qcodes_colorbar.set_label(new_label)
                            subplot.qcodes_colorbar.update_ticks()
