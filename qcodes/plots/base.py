'''
Live plotting in Jupyter notebooks
'''
from IPython.display import display

from qcodes.widgets.widgets import HiddenUpdateWidget


class BasePlot:

    '''
    create an auto-updating plot connected to a Jupyter notebook

    interval: period in seconds between update checks
        default 1

    data_keys: which keys in trace config can contain data
        that we should look for updates in.
        default 'xyz' (treated as a sequence) but add more if
        for example marker size or color can contain data
    '''

    def __init__(self, interval=1, data_keys='xyz'):
        self.data_keys = data_keys
        self.traces = []
        self.data_updaters = set()

        self.interval = interval
        if interval:
            self.update_widget = HiddenUpdateWidget(self.update, interval)
            display(self.update_widget)

    def clear(self):
        '''
        Clears the plot window and removes all subplots and traces
        so that the window can be reused.
        '''
        # any derived class should implement this
        raise NotImplementedError
        # typically traces and subplots should be cleared as well as the
        # figure window for the particular backend
        self.traces = []
        self.subplots = []

    def replace(self, *args, updater=None, **kwargs):
        self.clear()
        self.add(*args, updater=updater, **kwargs)

    def add(self, *args, updater=None, **kwargs):
        '''
        add one trace to this plot

        args: optional way to provide x/y/z data without keywords
            If the last one is 1D, may be `y` or `x`, `y`
            If the last one is 2D, may be `z` or `x`, `y`, `z`

        updater: a callable (with no args) that updates the data in this trace
            if omitted, we will look for DataSets referenced in this data, and
            call their sync methods.

        kwargs: after inserting info found in args and possibly in set_arrays
            into `x`, `y`, and optionally `z`, these are passed along to
            self.add_to_plot

        Array shapes for 2D plots:
            x:(1D-length m), y:(1D-length n), z: (2D- n*m array)
        '''
        self.expand_trace(args, kwargs)
        self.add_to_plot(**kwargs)
        self.add_updater(updater, kwargs)

    def add_to_plot(self, **kwargs):
        '''
        add a trace the plot itself (typically called by self.add,
        which incorporates args into kwargs, so the subclass doesn't
        need to worry about this). Data will be in `x`, `y`, and optionally
        `z`

        should be implemented by a subclass, and each call should append
        a dictionary to self.traces, containing at least {'config': kwargs}
        '''
        raise NotImplementedError

    def add_updater(self, updater, plot_config):
        if updater is not None:
            self.data_updaters.add(updater)
        else:
            for key in self.data_keys:
                data_array = plot_config.get(key, '')
                if hasattr(data_array, 'data_set'):
                    if data_array.data_set is not None:
                        self.data_updaters.add(data_array.data_set.sync)

        if self.data_updaters:
            self.update_widget.interval = self.interval

    def get_default_title(self):
        '''
        the default title for a plot is just a list of DataSet locations
        a custom title can be passed using **kwargs.
        '''
        title_parts = []
        for trace in self.traces:
            config = trace['config']
            if 'title' in config:  # can be passed using **kw
                return config['title']
            for part in self.data_keys:
                data_array = config.get(part, '')
                if hasattr(data_array, 'data_set'):
                    if data_array.data_set is not None:
                        location = data_array.data_set.location
                        if location and location not in title_parts:
                            title_parts.append(location)
        return ', '.join(title_parts)

    def get_label(self, data_array):
        '''
        look for a label, falling back on name.
        '''
        return (getattr(data_array, 'label', '') or
                getattr(data_array, 'name', ''))

    def expand_trace(self, args, kwargs):
        '''
        the x, y (and potentially z) data for a trace may be provided
        as positional rather than keyword args. The allowed forms are
        [y] or [x, y] if the last arg is 1D, and
        [z] or [x, y, z] if the last arg is 2D

        also, look in the main data array (`z` if it exists, else `y`)
        for set_arrays that can provide the `x` (and potentially `y`) data

        even if we allow data in other attributes (like marker size/color)
        by providing a different self.data_keys, set_arrays will only
        contribute x from y, or x & y from z, so we don't use data_keys here
        '''
        if args:
            if hasattr(args[-1][0], '__len__'):
                # 2D (or higher... but ignore this for now)
                # this test works for both numpy arrays and bare sequences
                axletters = 'xyz'
                ndim = 2
            else:
                axletters = 'xy'
                ndim = 1

            if len(args) not in (1, len(axletters)):
                raise ValueError('{}D data needs 1 or {} unnamed args'.format(
                    ndim, len(axletters)))

            arg_axletters = axletters[-len(args):]

            for arg, arg_axletters in zip(args, arg_axletters):
                if arg_axletters in kwargs:
                    raise ValueError(arg_axletters + ' data provided twice')
                kwargs[arg_axletters] = arg

        # reset axletters, we may or may not have found them above
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
        MatPlot.add() or in the included DataSets, then include this in
        the plot

        this is a wrapper routine that the update widget calls,
        inside this we call self.update() which should be subclassed
        '''
        any_updates = False
        for updater in self.data_updaters:
            updates = updater()
            if updates is not False:
                any_updates = True

        self.update_plot()

        # once all updaters report they're finished (by returning exactly
        # False) we stop updating the plot.
        if any_updates is False:
            self.halt()

    def update_plot(self):
        '''
        update the plot itself (typically called by self.update).
        should be implemented by a subclass
        '''
        raise NotImplementedError

    def halt(self):
        '''
        stop automatic updates to this plot, by canceling its update widget
        '''
        if hasattr(self, 'update_widget'):
            self.update_widget.halt()
