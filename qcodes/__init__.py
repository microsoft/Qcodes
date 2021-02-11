"""Set up the main qcodes namespace."""

# flake8: noqa (we don't need the "<...> imported but unused" error)

# config
from typing import Any
import qcodes.configuration as qcconfig
from qcodes.logger.logger import conditionally_start_all_logging
from qcodes.utils.helpers import add_to_spyder_UMR_excludelist
from .version import __version__

config: qcconfig.Config = qcconfig.Config()

conditionally_start_all_logging()

# we dont want spyder to reload qcodes as this will overwrite the default station
# instrument list and running monitor
add_to_spyder_UMR_excludelist('qcodes')

if config.core.import_legacy_api:
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
    from qcodes.loops import Loop, active_loop, active_data_set
    from qcodes.measure import Measure
    from qcodes.data.data_set import DataSet, new_data, load_data
    from qcodes.actions import Task, Wait, BreakIf
    from qcodes.data.location import FormatLocation
    from qcodes.data.data_array import DataArray
    from qcodes.data.format import Formatter
    from qcodes.data.gnuplot_format import GNUPlotFormat
    from qcodes.data.hdf5_format import HDF5Format
    from qcodes.data.io import DiskIO


from qcodes.station import Station
haswebsockets = True
try:
    import websockets
except ImportError:
    haswebsockets = False
if haswebsockets:
    from qcodes.monitor.monitor import Monitor




from qcodes.instrument.base import Instrument, find_or_create_instrument
from qcodes.instrument.ip import IPInstrument
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.instrument.function import Function
from qcodes.instrument.parameter import (
    Parameter,
    ArrayParameter,
    MultiParameter,
    ParameterWithSetpoints,
    DelegateParameter,
    ManualParameter,
    ScaledParameter,
    combine,
    CombinedParameter)
from qcodes.instrument.sweep_values import SweepFixedValues, SweepValues

from qcodes.utils import validators

from qcodes.instrument_drivers.test import test_instruments, test_instrument

from qcodes.dataset.measurements import Measurement
from qcodes.dataset.data_set import new_data_set, load_by_counter, load_by_id, load_by_run_spec, load_by_guid
from qcodes.dataset.experiment_container import new_experiment, load_experiment, load_experiment_by_name, \
    load_last_experiment, experiments, load_or_create_experiment
from qcodes.dataset.sqlite.settings import SQLiteSettings
from qcodes.dataset.descriptions.param_spec import ParamSpec
from qcodes.dataset.sqlite.database import initialise_database, \
    initialise_or_create_database_at

try:
    # Check if we are in iPython
    get_ipython()  # type: ignore[name-defined]
    from qcodes.utils.magic import register_magic_class
    _register_magic = config.core.get('register_magic', False)
    if _register_magic is not False:
        register_magic_class(magic_commands=_register_magic)
except NameError:
    pass
except RuntimeError as e:
    print(e)

import logging

# ensure to close all instruments when interpreter is closed
import atexit
atexit.register(Instrument.close_all)


def test(**kwargs: Any) -> int:
    """
    Run QCoDeS tests. This requires the test requirements given
    in test_requirements.txt to be installed.
    All arguments are forwarded to pytest.main
    """
    try:
        import pytest
        from hypothesis import settings
        settings(deadline=1000)
    except ImportError:
        print("Need pytest and hypothesis to run tests")
        return 1
    args = ['--pyargs', 'qcodes.tests']
    retcode = pytest.main(args, **kwargs)
    return retcode


test.__test__ = False  # type: ignore[attr-defined] # Don't try to run this method as a test
