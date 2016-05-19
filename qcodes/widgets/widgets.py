from IPython.display import display
from ipywidgets import widgets
from multiprocessing import active_children
from traitlets import Unicode, Float

from qcodes.process.stream_queue import get_stream_queue
from .display import display_auto
from qcodes.loops import MP_NAME, halt_bg


display_auto('widgets/widgets.js')
display_auto('widgets/widgets.css')


class UpdateWidget(widgets.DOMWidget):
    '''
    Execute a callable periodically, and display its return in the output area

    fn - the callable (with no parameters) to execute
    interval - the period, in seconds
        can be changed later by setting the interval attribute
        interval=0 or the halt() method disables updates.
    first_call - do we call the update function immediately (default, True),
        or only after the first interval?
    '''
    _view_name = Unicode('UpdateView', sync=True)  # see widgets.js
    _message = Unicode(sync=True)
    interval = Float(sync=True)

    def __init__(self, fn, interval, first_call=True, **kwargs):
        super().__init__(**kwargs)

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
        self._message = str(self._fn())

    def halt(self):
        if self.interval:
            self.previous_interval = self.interval
        self.interval = 0

    def restart(self, **kwargs):
        if self.interval != self.previous_interval:
            self.interval = self.previous_interval


class HiddenUpdateWidget(UpdateWidget):
    '''
    A variant on UpdateWidget that hides its section of the output area
    Just lets the front end periodically execute code
    that takes care of its own display.
    by default, first_call is False here, unlike UpdateWidget
    '''
    _view_name = Unicode('HiddenUpdateView', sync=True)  # see widgets.js

    def __init__(self, *args, first_call=False, **kwargs):
        super().__init__(*args, first_call=first_call, **kwargs)


def get_subprocess_widget():
    '''
    convenience function to get a singleton SubprocessWidget
    and restart it if it has been halted
    '''
    if SubprocessWidget.instance is None:
        return SubprocessWidget()

    return SubprocessWidget.instance


def show_subprocess_widget():
    display(get_subprocess_widget())


class SubprocessWidget(UpdateWidget):
    '''
    Display the subprocess outputs collected by the StreamQueue
    in a box in the notebook window
    '''
    _view_name = Unicode('SubprocessView', sync=True)  # see widgets.js
    _processes = Unicode(sync=True)

    instance = None

    # max seconds to wait for a measurement to abort
    abort_timeout = 30

    def __init__(self, interval=0.5):
        if self.instance is not None:
            raise RuntimeError(
                'Only one instance of SubprocessWidget should exist at '
                'a time. Use the function get_subprocess_output to find or '
                'create it.')

        self.__class__.instance = self

        self.stream_queue = get_stream_queue()
        super().__init__(fn=None, interval=interval)

    def do_update(self, content=None, buffers=None):
        self._message = self.stream_queue.get()

        loops = []
        others = []

        for p in active_children():
            if getattr(p, 'name', '') == MP_NAME:
                loops.append(str(p))
            else:
                others.append(str(p))

        self._processes = ', '.join(others + loops)

        if content.get('abort'):
            halt_bg(timeout=self.abort_timeout, traceback=False)
