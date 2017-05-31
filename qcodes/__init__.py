"""Set up the main qcodes namespace."""
# flake8: noqa (we don't need the "<...> imported but unused" error)
# config and logging
import logging

from qcodes.config import Config
config = Config()

from qcodes.utils.zmq_helpers import check_broker
haz_broker = check_broker()
if haz_broker:
    from qcodes.utils.zmq_helpers import QPUBHandler
    import logging.config
    import pkg_resources as pkgr
    logger_config = pkgr.resource_filename(__name__, "./config/logging.conf")
    logging.config.fileConfig(logger_config)
else:
    logging.warning("Can't publish logs, did you star the server?")


# name space
from qcodes.version import __version__
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

from qcodes.station import Station
from qcodes.loops import Loop, active_loop, active_data_set
from qcodes.measure import Measure
from qcodes.actions import Task, Wait, BreakIf

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

from qcodes.instrument_drivers.test import test_instruments, test_instrument
