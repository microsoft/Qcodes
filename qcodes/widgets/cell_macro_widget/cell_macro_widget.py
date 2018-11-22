from IPython.display import display
from ipywidgets.widgets import *
from traitlets import Unicode, Bool

from qcodes.widgets import display_auto

class CellMacroWidget(DOMWidget):
    _view_name = Unicode('CellMacroView').tag(sync=True)
    _view_module = Unicode('cell_macro').tag(sync=True)

    sidebar_position = Unicode().tag(sync=True)

    def __init__(self):
        super().__init__()

        display_auto('widgets/cell_macro_widget/cell_macro_widget.css')
        display_auto('widgets/cell_macro_widget/cell_macro_widget.js')

    def display(self):
        display(self)