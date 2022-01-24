"""Set up the main qcodes namespace."""

# flake8: noqa (we don't need the "<...> imported but unused" error)

# config
import warnings
from typing import Any

import qcodes.configuration as qcconfig
from qcodes.logger.logger import conditionally_start_all_logging
from qcodes.utils.helpers import add_to_spyder_UMR_excludelist
from qcodes.utils.installation_info import get_qcodes_version

__version__ = get_qcodes_version()

config: qcconfig.Config = qcconfig.Config()

conditionally_start_all_logging()

# we dont want spyder to reload qcodes as this will overwrite the default station
# instrument list and running monitor
add_to_spyder_UMR_excludelist('qcodes')


import atexit

from qcodes.dataset.data_set import (
    load_by_counter,
    load_by_guid,
    load_by_id,
    load_by_run_spec,
    new_data_set,
)
from qcodes.dataset.descriptions.param_spec import ParamSpec
from qcodes.dataset.experiment_container import (
    experiments,
    load_experiment,
    load_experiment_by_name,
    load_last_experiment,
    load_or_create_experiment,
    new_experiment,
)
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.sqlite.database import (
    initialise_database,
    initialise_or_create_database_at,
)
from qcodes.dataset.sqlite.settings import SQLiteSettings
from qcodes.instrument.base import Instrument, find_or_create_instrument
from qcodes.instrument.channel import ChannelList, ChannelTuple, InstrumentChannel
from qcodes.instrument.function import Function
from qcodes.instrument.ip import IPInstrument
from qcodes.instrument.parameter import (
    ArrayParameter,
    CombinedParameter,
    DelegateParameter,
    ManualParameter,
    MultiParameter,
    Parameter,
    ParameterWithSetpoints,
    ScaledParameter,
    combine,
)
from qcodes.instrument.sweep_values import SweepFixedValues, SweepValues
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument_drivers.test import test_instrument, test_instruments
from qcodes.monitor.monitor import Monitor
from qcodes.station import Station
from qcodes.utils import validators

# ensure to close all instruments when interpreter is closed
atexit.register(Instrument.close_all)

if config.core.import_legacy_api:
    from qcodes.utils.deprecate import QCoDeSDeprecationWarning

    warnings.warn(
        "import_legacy_api config option is deprecated and will be removed "
        "in a future release please update your imports to import these "
        "modules directly.",
        QCoDeSDeprecationWarning,
    )

    plotlib = config.gui.plotlib
    if plotlib in {"QT", "all"}:
        try:
            from qcodes.plots.pyqtgraph import QtPlot
        except Exception:
            print(
                "pyqtgraph plotting not supported, "
                'try "from qcodes.plots.pyqtgraph import QtPlot" '
                "to see the full error"
            )

    if plotlib in {"matplotlib", "all"}:
        try:
            from qcodes.plots.qcmatplotlib import MatPlot
        except Exception:
            print(
                "matplotlib plotting not supported, "
                'try "from qcodes.plots.qcmatplotlib import MatPlot" '
                "to see the full error"
            )
    from qcodes.actions import BreakIf, Task, Wait
    from qcodes.data.data_array import DataArray
    from qcodes.data.data_set import DataSet, load_data, new_data
    from qcodes.data.format import Formatter
    from qcodes.data.gnuplot_format import GNUPlotFormat
    from qcodes.data.hdf5_format import HDF5Format
    from qcodes.data.io import DiskIO
    from qcodes.data.location import FormatLocation
    from qcodes.loops import Loop, active_data_set, active_loop
    from qcodes.measure import Measure


try:
    _register_magic = config.core.get('register_magic', False)
    if _register_magic is not False:
        from IPython import get_ipython

        # Check if we are in IPython
        ip = get_ipython()
        if ip is not None:
            from qcodes.utils.magic import register_magic_class

            register_magic_class(magic_commands=_register_magic)
except ImportError:
    pass
except RuntimeError as e:
    print(e)


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
