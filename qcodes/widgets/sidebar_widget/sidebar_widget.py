import atexit
from IPython.display import display
from ipywidgets.widgets import *
from traitlets import Unicode, Bool

from qcodes.widgets import display_auto


display_auto('widgets/sidebar_widget/sidebar_widget.js')


class SidebarWidget(DOMWidget):
    _view_name = Unicode('SidebarView').tag(sync=True)
    _view_module = Unicode('sidebar').tag(sync=True)
    _view_module_version = Unicode('0.1.0').tag(sync=True)
    _widget_name = Unicode('none').tag(sync=True)
    _add_widget = Bool(False).tag(sync=True)
    _remove_widget = Bool(False).tag(sync=True)
    _playing = Bool(False).tag(sync=True)
    _clear_all_widgets = Bool(False).tag(sync=True)

    def __init__(self):
        super().__init__()
        display(self)
        self.clear_all_widgets()

    def add_widget(self, widget_name, remove_on_exit=True):
        self._widget_name = widget_name + '.display()'
        self._add_widget = not self._add_widget

        if remove_on_exit:
            atexit.register(self.remove_widget, widget_name)

    def remove_widget(self, widget_name):
        self._widget_name = widget_name + '.display()'
        self._remove_widget = not self._remove_widget

    def clear_all_widgets(self):
        self._clear_all_widgets = not self._clear_all_widgets