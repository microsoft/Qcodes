'''
Live plotting in Jupyter notebooks
using the nbagg backend

Ironically, this also works best with interactive mode OFF
(or maybe doesn't care?)
'''
import matplotlib.pyplot as plt
from numpy.ma import masked_invalid as masked
from IPython.display import display
from collections import Mapping
import warnings

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

        self.traces.append(kwargs)

        if updater is not None:
            self.data_updaters.add(updater)
        else:
            for part in ('x', 'y', 'z'):
                data_array = kwargs.get(part, '')
                if hasattr(data_array, 'data_set'):
                    self.data_updaters.add(data_array.data_set.sync)

        self.draw()

    def _args_to_kwargs(self, args, kwargs, axletters, plot_func_name):
        if len(args) not in (1, len(axletters)):
            raise ValueError('{} needs either 1 or {} unnamed args'.format(
                plot_func_name, len(axletters)))

        axletters = axletters[-len(args):]

        for arg, axletter in zip(args, axletters):
            if axletter in kwargs:
                raise ValueError(axletter + ' data provided twice')
            kwargs[axletter] = arg

    def update(self):
        any_updates = False
        for updater in self.data_updaters:
            updates = updater()
            if updates is not False:
                any_updates = True

        self.draw()

        # once all updaters report they're finished (by returning exactly
        # FALSE) we stop updating the plot.
        if any_updates is False and hasattr(self, 'update_widget'):
            self.update_widget.halt()

    def draw(self):
        for ax in self.subplots:
            ax.clear()

        for trace in self.traces:
            if 'z' in trace:
                self._draw_pcolormesh(**trace)
            else:
                self._draw_plot(**trace)

        self.fig.show()

    def _draw_plot(self, y, x=None, fmt=None, subplot=1, **kwargs):
        ax = self.subplots[subplot - 1]
        if x is None and hasattr(y, 'set_arrays'):
            x = y.set_arrays[-1]
        args = [arg for arg in [x, y, fmt] if arg is not None]
        ax.plot(*args, **kwargs)

    def _draw_pcolormesh(self, z, x=None, y=None, subplot=1, **kwargs):
        ax = self.subplots[subplot - 1]
        set_arrays = getattr(z, 'set_arrays', None)
        if x is None and set_arrays:
            x = set_arrays[-1]
        if y is None and set_arrays:
            y = set_arrays[-2]

        with warnings.catch_warnings():
            # we get a warning about masking NaNs right at the beginning here.
            # doesn't seem to be an issue so I'll ignore it.
            warnings.simplefilter('ignore', UserWarning)
            args = [masked(arg) for arg in [x, y, z] if arg is not None]

            ax.pcolormesh(*args, **kwargs)
