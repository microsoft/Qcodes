# set up the qcodes namespace
# flake8: noqa (we don't need the "<...> imported but unused" error)

# just for convenience in debugging, so we don't have to
# separately import multiprocessing
from multiprocessing import active_children

from qcodes.version import __version__
from qcodes.utils.multiprocessing import set_mp_method
from qcodes.utils.helpers import in_notebook

# code that should only be imported into the main (notebook) thread
# in particular, importing matplotlib in the side processes takes a long
# time and spins up other processes in order to try and get a front end
if in_notebook():  # pragma: no cover
    try:
        from qcodes.plots.matplotlib import MatPlot
    except Exception:
        print('matplotlib plotting not supported, '
              'try "from qcodes.plots.matplotlib import MatPlot" '
              'to see the full error')

    try:
        from qcodes.plots.pyqtgraph import QtPlot
    except Exception:
        print('pyqtgraph plotting not supported, '
              'try "from qcodes.plots.pyqtgraph import QtPlot" '
              'to see the full error')

    from qcodes.widgets.widgets import show_subprocess_widget

from qcodes.station import Station
from qcodes.loops import get_bg, halt_bg, Loop, Task, Wait

from qcodes.data.manager import get_data_manager
from qcodes.data.data_set import DataMode, DataSet, new_data, load_data
from qcodes.data.data_array import DataArray
from qcodes.data.format import Formatter, GNUPlotFormat
from qcodes.data.io import DiskIO

from qcodes.instrument.base import Instrument
from qcodes.instrument.ip import IPInstrument
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.mock import MockInstrument, MockModel

from qcodes.instrument.function import Function
from qcodes.instrument.parameter import Parameter, StandardParameter
from qcodes.instrument.sweep_values import SweepFixedValues, SweepValues

from qcodes.utils import validators

from qcodes.instrument_drivers.test import test_instruments, test_instrument
from qcodes.test import test_core
