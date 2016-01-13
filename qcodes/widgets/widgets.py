import os

from ipywidgets import widgets
from traitlets import Unicode, Float
from IPython.display import Javascript, display

# load and display the javascript from this directory
# some people use pkg_resources.resource_string for this, but that seems to
# require an absolute path within the package, this way gives a relative path
with open(os.path.join(os.path.split(__file__)[0], 'widgets.js')) as jsfile:
    display(Javascript(jsfile.read()))


class UpdateWidget(widgets.DOMWidget):
    '''
    Execute a callable periodically, and display its return in the output area

    fn - the callable (with no parameters) to execute
    interval - the period, in seconds
        can be changed later by setting the interval attribute
        interval=0 or the halt() method disables updates.
    '''
    _view_name = Unicode('UpdateView', sync=True)  # see widgets.js
    _message = Unicode(sync=True)
    interval = Float(sync=True)

    def __init__(self, fn, interval, **kwargs):
        super().__init__(**kwargs)

        self._fn = fn
        self.interval = interval

        self.on_msg(self._handle_msg)

        self._handle_msg({'init': True})

    def _handle_msg(self, message=None):
        self._message = str(self._fn())

    def halt(self):
        self.interval = 0


class HiddenUpdateWidget(UpdateWidget):
    '''
    A variant on UpdateWidget that hides its section of the output area
    Just lets the front end periodically execute code
    that takes care of its own display.
    '''
    _view_name = Unicode('HiddenUpdateView', sync=True)  # see widgets.js
