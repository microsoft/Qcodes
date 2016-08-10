"""Set up the main qcodes namespace."""

# flake8: noqa (we don't need the "<...> imported but unused" error)

# just for convenience in debugging, so we don't have to
# separately import multiprocessing
from multiprocessing import active_children

from qcodes.version import __version__

# load config system
import qcodes.config
from qcodes.config import (is_int, is_bool, is_text, is_float,
                           is_instance_factory, is_one_of_factory,
                           get_default_val)
from qcodes.config import (get_option, set_option, reset_option,
                           describe_option, option_context, options)

# create various options
usezmq_doc = """
: bool
    If set to True the framework will install a hook to send logging data
    to a ZMQ socket.
"""

with config.config_prefix('display'):
    config.register_option(
        'frontend', 1, 'frontend that is used (1: notebook, 0: unknown, 2: spyder', validator=is_int)
with config.config_prefix('logging'):
    config.register_option('usezmq', 1, usezmq_doc, validator=is_int)


# load config file
path = qcodes.config.qcodes_fname()
if path is not None:
    qcodes.config.from_file(path)


from qcodes.process.helpers import set_mp_method
from qcodes.utils.helpers import in_notebook

# code that should only be imported into the main (notebook) thread
# in particular, importing matplotlib in the side processes takes a long
# time and spins up other processes in order to try and get a front end
if in_notebook():  # pragma: no cover
    try:
        from qcodes.plots.qcmatplotlib import MatPlot
    except Exception:
        print('matplotlib plotting not supported, '
              'try "from qcodes.plots.qcmatplotlib import MatPlot" '
              'to see the full error')

    try:
        from qcodes.plots.pyqtgraph import QtPlot
    except Exception:
        print('pyqtgraph plotting not supported, '
              'try "from qcodes.plots.pyqtgraph import QtPlot" '
              'to see the full error')

    from qcodes.widgets.widgets import show_subprocess_widget

from qcodes.station import Station
from qcodes.loops import get_bg, halt_bg, Loop
from qcodes.actions import Task, Wait, BreakIf

from qcodes.data.manager import get_data_manager
from qcodes.data.data_set import DataMode, DataSet, new_data, load_data
from qcodes.data.location import FormatLocation
from qcodes.data.data_array import DataArray
from qcodes.data.format import Formatter
from qcodes.data.gnuplot_format import GNUPlotFormat
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
from qcodes.test import test_core, test_part
