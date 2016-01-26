# set up the qcodes namespace
# flake8: noqa (we don't need the "<...> imported but unused" error)

from qcodes.station import Station
from qcodes.loops import get_bg, halt_bg, Loop, Task, Wait

# hack for code that should only be imported into the main (notebook) thread
# see: http://stackoverflow.com/questions/15411967
# in particular, importing matplotlib in the side processes takes a long
# time and spins up other processes in order to try and get a front end
import sys as _sys
if 'ipy' in repr(_sys.stdout):
    from qcodes.plots.matplotlib import MatPlot
    from qcodes.plots.pyqtgraph import QtPlot
    from qcodes.widgets.widgets import show_subprocess_widget

from qcodes.data.manager import get_data_manager
from qcodes.data.data_set import DataMode, DataSet
from qcodes.data.data_array import DataArray
from qcodes.data.format import Formatter, GNUPlotFormat
from qcodes.data.io import DiskIO

from qcodes.instrument.base import Instrument
from qcodes.instrument.ip import IPInstrument
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.mock import MockInstrument

from qcodes.instrument.function import Function
from qcodes.instrument.parameter import Parameter, InstrumentParameter
from qcodes.instrument.sweep_values import SweepFixedValues, AdaptiveSweep

from qcodes.utils.helpers import reload_code
from qcodes.utils.multiprocessing import set_mp_method

# just for convenience in debugging, so we don't have to
# separately import multiprocessing
from multiprocessing import active_children
