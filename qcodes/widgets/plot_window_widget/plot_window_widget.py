from IPython.display import display
from ipywidgets.widgets import *
from traitlets import Unicode, Bool

from qcodes.widgets import display_auto


display_auto('widgets/plot_window_widget/plot_window_widget.css')
display_auto('widgets/plot_window_widget/plot_window_widget.js')


class PlotWindowWidget(DOMWidget):
    _view_name = Unicode('PlotWindowView').tag(sync=True)
    _view_module = Unicode('plot_window').tag(sync=True)

    cell_text = Unicode('').tag(sync=True)
    _execute_code = Bool().tag(sync=True)
    _execute_cell = Bool().tag(sync=True)
    collapsed = Bool(True).tag(sync=True)

    get_size = Bool().tag(sync=True)

    def __init__(self):
        super().__init__()
        display(self)

    def execute_code(self, code, expand=True):
        self.cell_text = str(code)
        self._execute_code = not self._execute_code
        if expand:
            self.expand()

    def execute_cell(self):
        self._execute_cell = not self._execute_cell

    def expand(self):
        self.collapsed = False

    def collapse(self):
        self.collapsed = True