'''
Live plotting in Jupyter notebooks
using the nbagg backend
'''
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
from numpy.ma import masked_invalid, getmask
from IPython.display import display
from collections import Mapping

from qcodes.widgets.widgets import HiddenUpdateWidget


class Plot(object):
    '''
    Plot x/y lines or x/y/z heatmap data. The first trace may be included
    in the constructor, other traces can be added with Plot.add()

    args: shortcut to provide the x/y/z data. See Plot.add

    figsize: (width, height) tuple to pass to plt.figure
        default (12, 5)
    interval: period in seconds between update checks

    subplots: either a sequence (args) or mapping (kwargs) to pass to
        plt.subplots. default is a single simple subplot (1, 1)
        you can use this to pass kwargs to the plt.figure constructor

    kwargs: passed along to Plot.add() to add the first data trace
    '''
    def __init__(self, *args, figsize=(12, 5), interval=1, subplots=(1, 1),
                 **kwargs):

        if isinstance(subplots, Mapping):
            self.fig, self.subplots = plt.subplots(**subplots)
        else:
            self.fig, self.subplots = plt.subplots(*subplots)
        if not hasattr(self.subplots, '__len__'):
            self.subplots = (self.subplots,)

        self.traces = []
        self.data_updaters = set()

        self.title = self.fig.suptitle('')

        if args or kwargs:
            self.add(*args, **kwargs)

        self.update_widget = HiddenUpdateWidget(self.update, interval)
        display(self.update_widget)

    def add(self, *args, updater=None, **kwargs):
        '''
        adds one trace to this Plot.

        args: a way to provide x/y/z data without keywords
            The last one is the dependent data, and we look at its
            dimensionality to determine if it's `y` or `z`.
            If it's `y`, it may optionally be preceded by `x`.
            If it's `z`, it may optionally be preceded by `x` and `y`.

        updater: a callable (with no args) that updates the data in this trace
            if omitted, we will look for DataSets referenced in this data, and
            call their sync methods.

        kwargs: with the following exceptions (mostly the data!), these are
            passed directly to the matplotlib plotting routine.

            `subplot`: the 1-based axes number to append to (default 1)

            if kwargs include `z`, we will draw a heatmap (ax.pcolormesh):
                `x`, `y`, and `z` are passed as positional args to pcolormesh

            without `z` we draw a scatter/lines plot (ax.plot):
                `x`, `y`, and `fmt` (if present) are passed as positional args
        '''
        # TODO some way to specify overlaid axes?

        if args:
            if hasattr(args[-1][0], '__len__'):
                # 2D (or higher... but ignore this for now)
                self._args_to_kwargs(args, kwargs, 'xyz', 'pcolormesh')
            else:
                # 1D
                self._args_to_kwargs(args, kwargs, 'xy', 'plot')

        self._find_data_in_set_arrays(kwargs)

        ax = self._get_axes(kwargs)
        if 'z' in kwargs:
            plot_object = self._draw_pcolormesh(ax, **kwargs)
        else:
            plot_object = self._draw_plot(ax, **kwargs)

        self._update_labels(ax, kwargs)
        prev_default_title = self._get_default_title()

        self.traces.append({
            'config': kwargs,
            'plot_object': plot_object
        })

        if prev_default_title == self.title.get_text():
            # in case the user has updated title, don't change it anymore
            self.title.set_text(self._get_default_title())

        if updater is not None:
            self.data_updaters.add(updater)
        else:
            for part in 'xyz':
                data_array = kwargs.get(part, '')
                if hasattr(data_array, 'data_set'):
                    self.data_updaters.add(data_array.data_set.sync)

    def _get_axes(self, config):
        return self.subplots[config.get('subplot', 1) - 1]

    def _get_default_title(self):
        title_parts = []
        for trace in self.traces:
            config = trace['config']
            for part in 'xyz':
                data_array = config.get(part, '')
                if hasattr(data_array, 'data_set'):
                    location = data_array.data_set.location
                    if location not in title_parts:
                        title_parts.append(location)
        return ', '.join(title_parts)

    def set_title(self, title):
        self.title.set_text(title)

    def _update_labels(self, ax, config):
        if 'x' in config and not ax.get_xlabel():
            ax.set_xlabel(self._get_label(config['x']))
        if 'y' in config and not ax.get_ylabel():
            ax.set_ylabel(self._get_label(config['y']))

    def _get_label(self, data_array):
        return (getattr(data_array, 'label', '') or
                getattr(data_array, 'name', ''))

    def _args_to_kwargs(self, args, kwargs, axletters, plot_func_name):
        if len(args) not in (1, len(axletters)):
            raise ValueError('{} needs either 1 or {} unnamed args'.format(
                plot_func_name, len(axletters)))

        arg_axletters = axletters[-len(args):]

        for arg, arg_axletters in zip(args, arg_axletters):
            if arg_axletters in kwargs:
                raise ValueError(arg_axletters + ' data provided twice')
            kwargs[arg_axletters] = arg

    def _find_data_in_set_arrays(self, kwargs):
        axletters = 'xyz' if 'z' in kwargs else 'xy'
        main_data = kwargs[axletters[-1]]
        if hasattr(main_data, 'set_arrays'):
            num_axes = len(axletters) - 1
            # things will probably fail if we try to plot arrays of the
            # wrong dimension... but we'll give it a shot anyway.
            set_arrays = main_data.set_arrays[-num_axes:]
            # for 2D: y is outer loop, which is earlier in set_arrays,
            # and x is the inner loop... is this the right convention?
            set_axletters = reversed(axletters[:-1])
            for axletter, set_array in zip(set_axletters, set_arrays):
                if axletter not in kwargs:
                    kwargs[axletter] = set_array

    def update(self):
        '''
        update the data in this plot, using the updaters given with
        Plot.add() or in the included DataSets, then include this in
        the plot
        '''
        any_updates = False
        for updater in self.data_updaters:
            updates = updater()
            if updates is not False:
                any_updates = True

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

        # once all updaters report they're finished (by returning exactly
        # False) we stop updating the plot.
        if any_updates is False and hasattr(self, 'update_widget'):
            self.halt()

    def halt(self):
        '''
        stop automatic updates to this plot, by canceling its update widget
        '''
        self.update_widget.halt()

    def _draw_plot(self, ax, y, x=None, fmt=None, subplot=1, **kwargs):
        # subplot=1 is just there to strip this out of kwargs
        args = [arg for arg in [x, y, fmt] if arg is not None]
        return ax.plot(*args, **kwargs)[0]

    def _draw_pcolormesh(self, ax, z, x=None, y=None, subplot=1, **kwargs):
        args = [masked_invalid(arg) for arg in [x, y, z]
                if arg is not None]

        for arg in args:
            if np.all(getmask(arg)):
                # if any entire array is masked, don't draw at all
                # there's nothing to draw, and anyway it throws a warning
                return False

        pc = ax.pcolormesh(*args, **kwargs)

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
            ax.qcodes_colorbar.set_label(self._get_label(z))

        return pc
