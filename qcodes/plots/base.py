"""
Live plotting in Jupyter notebooks
"""
from IPython.display import display

from qcodes import config

class BasePlot:

    """
    Auto-updating plot connected to a Jupyter notebook

    Args:
        interval (Int): period in seconds between update checks
         default 1

        data_keys(String): sequence of keys in trace config can contain data
            that we should look for updates in.
            default 'xyz' (treated as a sequence) but add more if
            for example marker size or color can contain data
    """

    def __init__(self, interval=1, data_keys='xyz'):
        self.data_keys = data_keys
        self.traces = []
        self.data_updaters = set()
        # only import in name space if the gui is set to noebook
        # and there is multiprocessing
        self.interval = interval
        if config['gui']['notebook'] and config['core']['legacy_mp']:
            from qcodes.widgets.widgets import HiddenUpdateWidget
            self.update_widget = HiddenUpdateWidget(self.update, interval)
            display(self.update_widget)

    def clear(self):
        """
        Clears the plot window and removes all subplots and traces
        so that the window can be reused.
        """
        # any derived class should implement this
        raise NotImplementedError
        # typically traces and subplots should be cleared as well as the
        # figure window for the particular backend
        # TODO(giulioungaretti) the following unreachable lines should really
        # be documentation.
        self.traces = []
        self.subplots = []

    def replace(self, *args, updater=None, **kwargs):
        """
        Clear all content and add new trace.

        Args:
            args (): optional way to provide x/y/z data without keywords
                If the last one is 1D, may be `y` or `x`, `y`
                If the last one is 2D, may be `z` or `x`, `y`, `z`

            updater: a callable (with no args) that updates the data in this trace
                if omitted, we will look for DataSets referenced in this data, and
                call their sync methods.

            **kwargs: passed on to self.add()
        """
        self.clear()
        self.add(*args, updater=updater, **kwargs)

    def add(self, *args, updater=None, **kwargs):
        """
        Add one trace to this plot.

        Args:
            args: optional way to provide x/y/z data without keywords
                If the last one is 1D, may be `y` or `x`, `y`
                If the last one is 2D, may be `z` or `x`, `y`, `z`

            updater: a callable (with no args) that updates the data in this trace
                if omitted, we will look for DataSets referenced in this data, and
                call their sync methods.

            kwargs: after inserting info found in args and possibly in set_arrays
                into `x`, `y`, and optionally `z`, these are passed along to
                self.add_to_plot.
                To use custom labels and units pass for example:
                    plot.add(x=set, y=amplitude, 
                             xlabel="set"
                             xunit="V",
                             ylabel= "Amplitude",
                             yunit ="V")

        Array shapes for 2D plots:
            x:(1D-length m), y:(1D-length n), z: (2D- n*m array)
        """
        # TODO(giulioungaretti): replace with an explicit version, see expand trace
        self.expand_trace(args, kwargs)
        self.add_to_plot(**kwargs)
        self.add_updater(updater, kwargs)

    def add_to_plot(self, **kwargs):
        """
        Add a trace the plot itself (typically called by self.add,
        which incorporates args into kwargs, so the subclass doesn't
        need to worry about this). Data will be in `x`, `y`, and optionally
        `z`.

        Should be implemented by a subclass, and each call should append
        a dictionary to self.traces, containing at least {'config': kwargs}
        """
        raise NotImplementedError

    def add_updater(self, updater, plot_config):
        """
        Add an updater to the plot.

        Args:
            updater (callable): callable (with no args) that updates the data in this trace
                if omitted, we will look for DataSets referenced in this data, and
                call their sync methods.
            plot_config (dict): this is a dictionary that gets populated inside
                add() via expand_trace().
                The reason this is here is to fetch from the data_set the sync method
                to use it as an updater.
        """
        if updater is not None:
            self.data_updaters.add(updater)
        else:
            for key in self.data_keys:
                data_array = plot_config.get(key, '')
                if hasattr(data_array, 'data_set'):
                    if data_array.data_set is not None:
                        self.data_updaters.add(data_array.data_set.sync)

        # If previous data on this plot became static, perhaps because
        # its measurement loop finished, the updater may have been halted.
        # If we have new update functions, re-activate the updater
        # by reinstating its update interval
        if self.data_updaters:
            if hasattr(self, 'update_widget'):
                self.update_widget.interval = self.interval

    def get_default_title(self):
        """
        Get the default title, which for a plot is just a list of DataSet locations.
        A custom title can be set when adding any trace (via either __init__ or add.
        these kwargs all eventually end up in self.traces[i]['config']) and it looks
        like we will take the first title we find from any trace... otherwise, if no
        trace specifies a title, then we combine whatever dataset locations we find.

        Note: (alexj): yeah, that's awkward, isn't it, and it looks like a weird
        implementation, feel free to change it 👼

        Returns:
            string: the title of the figure
        """
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
        """
        Look for a label in data_array falling back on name.

        Args:
            data_array (DataArray): data array to get label from

        Returns:
            string: label or name of the data_array

        """
        # TODO this should really be a static method
        name = (getattr(data_array, 'label', '') or
                getattr(data_array, 'name', ''))
        unit = getattr(data_array, 'unit', '')
        return  name, unit

    def expand_trace(self, args, kwargs):
        """
        Complete the x, y (and possibly z) data definition for a trace.

        Also modifies kwargs in place so that all the data needed to fully specify the
        trace is present (ie either x and y or x and y and z)

        Both ``__init__`` (for the first trace) and the ``add`` method support multiple
        ways to specify the data in the trace:

        As \*args:
            - ``add(y)`` or ``add(z)`` specify just the main 1D or 2D data, with the setpoint
              axis or axes implied.
            - ``add(x, y)`` or ``add(x, y, z)`` specify all axes of the data.
        And as \*\*kwargs:
            - ``add(x=x, y=y, z=z)`` you specify exactly the data you want on each axis.
              Any but the last (y or z) can be omitted, which allows for all of the same
              forms as with \*args, plus x and z or y and z, with just one axis implied from
              the setpoints of the z data.

        This method takes any of those forms and converts them into a complete set of
        kwargs, containing all of the explicit or implied data to be used in plotting this trace.

        Args:
            args (Tuple[DataArray]): positional args, as passed to either ``__init__`` or ``add``
            kwargs (Dict(DataArray]): keyword args, as passed to either ``__init__`` or ``add``.
                kwargs may contain non-data items in keys other than x, y, and z.

        Raises:
           ValueError: if the shape of the data does not match that of args
           ValueError: if the data is provided twice
        """
        # TODO(giulioungaretti): replace with an explicit version:
        # return the new kwargs  instead of modifying in place
        # TODO this should really be a static method
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
        """
        Update the data in this plot, using the updaters given with
        MatPlot.add() or in the included DataSets, then include this in
        the plot.

        This is a wrapper routine that the update widget calls,
        inside this we call self.update() which should be subclassed
        """
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
        """
        Update the plot itself (typically called by self.update).
        Should be implemented by a subclass
        """
        raise NotImplementedError

    def halt(self):
        """
        Stop automatic updates to this plot, by canceling its update widget
        """
        if hasattr(self, 'update_widget'):
            self.update_widget.halt()

    def save(self, filename=None):
        """
        Save current plot to filename

        Args:
            filename (Optional[str]): Location of the file
        """
        raise NotImplementedError
