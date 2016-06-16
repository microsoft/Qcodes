"""Qcodes-specific widgets for jupyter notebook."""
from IPython.display import display
from ipywidgets import widgets
from multiprocessing import active_children
from traitlets import Unicode, Float, Enum

from qcodes.process.stream_queue import get_stream_queue
from .display import display_auto
from qcodes.loops import MP_NAME, halt_bg

display_auto('widgets/widgets.js')
display_auto('widgets/widgets.css')


class UpdateWidget(widgets.DOMWidget):

    """
    Execute a callable periodically, and display its return in the output area.

    The Javascript portion of this is in widgets.js with the same name.

    Args:
        fn (callable): To be called (with no parameters) periodically.

        interval (number): The call period, in seconds. Can be changed later
            by setting the ``interval`` attribute. ``interval=0`` or the
            ``halt()`` method disables updates.

        first_call (bool): Whether to call the update function immediately
            or only after the first interval. Default True.
    """

    _view_name = Unicode('UpdateView', sync=True)  # see widgets.js
    _message = Unicode(sync=True)
    interval = Float(sync=True)

    def __init__(self, fn, interval, first_call=True):
        super().__init__()

        self._fn = fn
        self.interval = interval
        self.previous_interval = interval

        # callbacks send the widget (self) as the first arg
        # so bind to __func__ and we can leave the duplicate out
        # of the method signature
        self.on_msg(self.do_update.__func__)

        if first_call:
            self.do_update({}, [])

    def do_update(self, content=None, buffers=None):
        """
        Execute the callback and send its return value to the notebook.

        Args:
            content: required by DOMWidget, unused
            buffers: required by DOMWidget, unused
        """
        self._message = str(self._fn())

    def halt(self):
        """
        Stop future updates.

        Keeps a record of the interval so we can ``restart()`` later.
        You can also restart by explicitly setting ``self.interval`` to a
        positive value.
        """
        if self.interval:
            self.previous_interval = self.interval
        self.interval = 0

    def restart(self, **kwargs):
        """
        Reinstate updates with the most recent interval.

        TODO: why did I include kwargs?
        """
        if not hasattr(self, 'previous_interval'):
            self.previous_interval = 1

        if self.interval != self.previous_interval:
            self.interval = self.previous_interval


class HiddenUpdateWidget(UpdateWidget):

    """
    A variant on UpdateWidget that hides its section of the output area.

    The Javascript portion of this is in widgets.js with the same name.

    Just lets the front end periodically execute code that takes care of its
    own display. By default, first_call is False here, unlike UpdateWidget,
    because it is assumed this widget is created to update something that
    has been displayed by other means.

    Args:
        fn (callable): To be called (with no parameters) periodically.

        interval (number): The call period, in seconds. Can be changed later
            by setting the ``interval`` attribute. ``interval=0`` or the
            ``halt()`` method disables updates.

        first_call (bool): Whether to call the update function immediately
            or only after the first interval. Default False.
    """

    _view_name = Unicode('HiddenUpdateView', sync=True)  # see widgets.js

    def __init__(self, *args, first_call=False, **kwargs):
        super().__init__(*args, first_call=first_call, **kwargs)


def get_subprocess_widget(**kwargs):
    """
    Convenience function to get a singleton SubprocessWidget.

    Restarts widget updates if it has been halted.

    Args:
        **kwargs: passed to SubprocessWidget constructor

    Returns:
        SubprocessWidget
    """
    if SubprocessWidget.instance is None:
        w = SubprocessWidget(**kwargs)
    else:
        w = SubprocessWidget.instance

    w.restart()

    return w


def show_subprocess_widget(**kwargs):
    """
    Display the subprocess widget, creating it if needed.

    Args:
        **kwargs: passed to SubprocessWidget constructor
    """
    display(get_subprocess_widget(**kwargs))


class SubprocessWidget(UpdateWidget):

    """
    Display subprocess output in a box in the jupyter notebook window.

    Output is collected from each process's stdout and stderr by the
    ``StreamQueue`` and read periodically from the main process, triggered
    by Javascript.

    The Javascript portion of this is in widgets.js with the same name.

    Args:
        interval (number): The call period, in seconds. Can be changed later
            by setting the ``interval`` attribute. ``interval=0`` or the
            ``halt()`` method disables updates. Default 0.5.
        state (str): starting window state of the widget. Options are
            'docked' (default), 'minimized', 'floated'
    """

    _view_name = Unicode('SubprocessView', sync=True)  # see widgets.js
    _processes = Unicode(sync=True)
    _state = Enum(('minimized', 'docked', 'floated'), sync=True)

    instance = None

    # max seconds to wait for a measurement to abort
    abort_timeout = 30

    def __init__(self, interval=0.5, state='docked'):
        if self.instance is not None:
            raise RuntimeError(
                'Only one instance of SubprocessWidget should exist at '
                'a time. Use the function get_subprocess_output to find or '
                'create it.')

        self.stream_queue = get_stream_queue()
        self._state = state
        super().__init__(fn=None, interval=interval)

        self.__class__.instance = self

    def do_update(self, content=None, buffers=None):
        """
        Update the information to be displayed in the widget.

        Send any new messages to the notebook, and update the list of
        active processes.

        Args:
            content: required by DOMWidget, unused
            buffers: required by DOMWidget, unused
        """
        self._message = self.stream_queue.get()

        loops = []
        others = []

        for p in active_children():
            if getattr(p, 'name', '') == MP_NAME:
                # take off the <> on the ends, just to shorten the names
                loops.append(str(p)[1:-1])
            else:
                others.append(str(p)[1:-1])

        self._processes = '\n'.join(loops + others)

        if content.get('abort'):
            halt_bg(timeout=self.abort_timeout, traceback=False)
