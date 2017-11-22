import atexit
from IPython.display import display
from ipywidgets.widgets import *
from traitlets import Unicode, List, Bool


class TOCWidget(DOMWidget):
    _view_name = Unicode('TOCView').tag(sync=True)
    _view_module = Unicode('toc').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    _widget_name = Unicode('none').tag(sync=True)