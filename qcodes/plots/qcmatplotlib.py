"""
Live plotting in Jupyter notebooks
using the nbagg backend and matplotlib
"""
from collections import Mapping

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
from numpy.ma import masked_invalid, getmask

from .base import BasePlot


class MatPlot(BasePlot):
    plot_1D_kwargs = {}
    plot_2D_kwargs = {}
    """
    Plot x/y lines or x/y/z heatmap data. The first trace may be included
    in the constructor, other traces can be added with MatPlot.add()

    Args:
        *args: shortcut to provide the x/y/z data. See BasePlot.add

        figsize (Tuple[Float, Float]): (width, height) tuple in inches to pass to plt.figure
            default (8, 5)

        interval: period in seconds between update checks

        subplots: either a sequence (args) or mapping (kwargs) to pass to
            plt.subplots. default is a single simple subplot (1, 1)
            you can use this to pass kwargs to the plt.figure constructor

        num: integer or None
            specifies the index of the matplotlib figure window to use. If None
            then open a new window

        **kwargs: passed along to MatPlot.add() to add the first data trace
    """
    def __init__(self, *args, figsize=None, interval=1, subplots=None, num=None,
                 **kwargs):

        super().__init__(interval)

        self._init_plot(subplots, figsize, num=num)
        if args or kwargs:
            self.add(*args, **kwargs)

    def _init_plot(self, subplots=None, figsize=None, num=None):
        if figsize is None:
            figsize = (8, 5)

        if subplots is None:
            subplots = (1, 1)

        if isinstance(subplots, Mapping):
            self.fig, self.subplots = plt.subplots(figsize=figsize, num=num,
                                                   **subplots)
        else:
            self.fig, self.subplots = plt.subplots(*subplots, num=num,
                                                   figsize=figsize)
        if not hasattr(self.subplots, '__len__'):
            self.subplots = (self.subplots,)

        self.title = self.fig.suptitle('')

    def clear(self, subplots=None, figsize=None):
        """
        Clears the plot window and removes all subplots and traces
        so that the window can be reused.
        """
        self.traces = []
        self.fig.clf()
        self._init_plot(subplots, figsize, num=self.fig.number)

    def add_to_plot(self, **kwargs):
        """
        adds one trace to this MatPlot.

        kwargs: with the following exceptions (mostly the data!), these are
            passed directly to the matplotlib plotting routine.

            `subplot`: the 1-based axes number to append to (default 1)

            if kwargs include `z`, we will draw a heatmap (ax.pcolormesh):
                `x`, `y`, and `z` are passed as positional args to pcolormesh

            without `z` we draw a scatter/lines plot (ax.plot):
                `x`, `y`, and `fmt` (if present) are passed as positional args
        """
        # TODO some way to specify overlaid axes?
        ax = self._get_axes(kwargs)
        if 'z' in kwargs:
            plot_object = self._draw_pcolormesh(ax, **kwargs)
        else:
            plot_object = self._draw_plot(ax, **kwargs)

        self._update_labels(ax, kwargs)
        prev_default_title = self.get_default_title()

        self.traces.append({
            'config': kwargs,
            'plot_object': plot_object
        })

        if prev_default_title == self.title.get_text():
            # in case the user has updated title, don't change it anymore
            self.title.set_text(self.get_default_title())

    def _get_axes(self, config):
        return self.subplots[config.get('subplot', 1) - 1]

    def _update_labels(self, ax, config):
        if 'x' in config and not ax.get_xlabel():
            ax.set_xlabel(self.get_label(config['x']))
        if 'y' in config and not ax.get_ylabel():
            ax.set_ylabel(self.get_label(config['y']))

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

                ax = self._get_axes(config)
                plot_object = self._draw_pcolormesh(ax, **config)
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

    def _draw_plot(self, ax, y, x=None, fmt=None, subplot=1, **kwargs):
        # NOTE(alexj)stripping out subplot because which subplot we're in is already
        # described by ax, and it's not a kwarg to matplotlib's ax.plot. But I
        # didn't want to strip it out of kwargs earlier because it should stay
        # part of trace['config'].
        args = [arg for arg in [x, y, fmt] if arg is not None]

        full_kwargs = {**self.plot_1D_kwargs, **kwargs}
        line, = ax.plot(*args, **full_kwargs)
        return line

    def _draw_pcolormesh(self, ax, z, x=None, y=None, subplot=1,
                         nticks=None, use_offset=False, **kwargs):
        """
        Draws a 2D color plot

        Args:
            ax (Axis): Matplotlib axis object to plot in
            z: 2D array of data values
            x (Array, Optional): Array of values along x-axis. Dimensions should
                be either same as z, or equal to length along x-axis.
            y (Array, Optional): Array of values along y-axis. Dimensions should
                be either same as z, or equal to length along y-axis.
            subplot (int, Optional): Deprecated, see alexj notes below
            nticks (int, Optional): preferred number of ticks along axes
            use_offset (bool, Optional): Whether or not axes can have an offset
            **kwargs: Optional list of kwargs to be passed on to pcolormesh.
                These will overwrite any of the default kwargs in plot_kwargs.
        """

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
                    arr = arr[0] if k == 0 else arr[:,0]

                if np.ma.is_masked(arr[1]):
                    # Only the first element is not nan, in this case pad with
                    # a value, and separate their values by 1
                    arr_pad = np.pad(arr, (1, 0), mode='symmetric')
                    arr_pad[:2] += [-0.5, 0.5]
                else:
                    # Add padding on both sides equal to endpoints
                    arr_pad = np.pad(arr, (1, 1), mode='symmetric')
                    # Add differences to edgepoints (may be nan)
                    arr_pad[0] += arr_pad[1] - arr_pad[2]
                    arr_pad[-1] += arr_pad[-2] - arr_pad[-3]

                    diff = np.ma.diff(arr_pad) / 2
                    # Insert value at beginning and end of diff to ensure same
                    # length
                    diff = np.insert(diff, 0, diff[0])

                    arr_pad += diff
                    # Ignore final value
                    arr_pad = arr_pad[:-1]
                args.append(masked_invalid(arr_pad))
            args.append(args_masked[-1])
        else:
            # Only the masked value of z is used as a mask
            args = args_masked[-1:]

        # Include default plotting kwargs, which can be overwritten by given
        # kwargs
        full_kwargs = {**self.plot_2D_kwargs, **kwargs}
        pc = ax.pcolormesh(*args, **full_kwargs)

        # Set x and y limits if arrays are provided
        if x is not None and y is not None:
            ax.set_xlim(np.nanmin(args[0]), np.nanmax(args[0]))
            ax.set_ylim(np.nanmin(args[1]), np.nanmax(args[1]))

        # Specify preferred number of ticks with labels
        if nticks:
            ax.locator_params(nbins=nticks)

        # Specify if axes can have offset or not
        ax.ticklabel_format(useOffset=use_offset)

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
            ax.qcodes_colorbar.set_label(self.get_label(z))

        # Scale colors if z has elements
        cmin = np.nanmin(args_masked[-1])
        cmax = np.nanmax(args_masked[-1])
        ax.qcodes_colorbar.set_clim(cmin, cmax)

        return pc
