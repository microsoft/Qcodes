"""Set up the main qcodes namespace."""

# flake8: noqa (we don't need the "<...> imported but unused" error)

# config

from qcodes.config import Config
from qcodes.utils.helpers import add_to_spyder_UMR_excludelist

# we dont want spyder to reload qcodes as this will overwrite the default station
# instrument list and running monitor
add_to_spyder_UMR_excludelist('qcodes')
config = Config() # type: Config

from qcodes.version import __version__

plotlib = config.gui.plotlib
if plotlib in {'QT', 'all'}:
    try:
        from qcodes.plots.pyqtgraph import QtPlot
    except Exception:
        print('pyqtgraph plotting not supported, '
              'try "from qcodes.plots.pyqtgraph import QtPlot" '
              'to see the full error')

if plotlib in {'matplotlib', 'all'}:
    try:
        from qcodes.plots.qcmatplotlib import MatPlot
    except Exception:
        print('matplotlib plotting not supported, '
              'try "from qcodes.plots.qcmatplotlib import MatPlot" '
              'to see the full error')


from qcodes.station import Station
from qcodes.loops import Loop, active_loop, active_data_set
from qcodes.measure import Measure
from qcodes.actions import Task, Wait, BreakIf
haswebsockets = True
try:
    import websockets
except ImportError:
    haswebsockets = False
if haswebsockets:
    from qcodes.monitor.monitor import Monitor

from qcodes.data.data_set import DataSet, new_data, load_data
from qcodes.data.location import FormatLocation
from qcodes.data.data_array import DataArray
from qcodes.data.format import Formatter
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.hdf5_format import HDF5Format
from qcodes.data.io import DiskIO

from qcodes.instrument.base import Instrument
from qcodes.instrument.ip import IPInstrument
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList

from qcodes.instrument.function import Function
from qcodes.instrument.parameter import (
    Parameter,
    ArrayParameter,
    MultiParameter,
    StandardParameter,
    ManualParameter,
    combine,
    CombinedParameter)
from qcodes.instrument.sweep_values import SweepFixedValues, SweepValues

from qcodes.utils import validators
from qcodes.utils.zmq_helpers import Publisher
from qcodes.instrument_drivers.test import test_instruments, test_instrument

from qcodes.dataset.data_set import new_data_set, load_by_counter, load_by_id
from qcodes.dataset.experiment_container import new_experiment, load_experiment, load_experiment_by_name, \
    load_last_experiment, experiments
from qcodes.dataset.sqlite_settings import SQLiteSettings
from qcodes.dataset.param_spec import ParamSpec
# TODO: do we want this?
from qcodes.dataset.sqlite_base import connect as _connect
from qcodes.dataset.sqlite_base import init_db as _init_db

_c = _connect(config["core"]["db_location"], config["core"]["db_debug"])
# init is actually idempotent so it's safe to always call!
_init_db(_c)
_c.close()
del _c

try:
    get_ipython() # type: ignore # Check if we are in iPython
    from qcodes.utils.magic import register_magic_class
    _register_magic = config.core.get('register_magic', False)
    if _register_magic is not False:
        register_magic_class(magic_commands=_register_magic)
except NameError:
    pass
except RuntimeError as e:
    print(e)

# ensure to close all isntruments when interpreter is closed
import atexit
atexit.register(Instrument.close_all)
