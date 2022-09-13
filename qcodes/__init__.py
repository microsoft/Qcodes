"""Set up the main qcodes namespace."""

# flake8: noqa (we don't need the "<...> imported but unused" error)

# config
import warnings
from typing import Any

import qcodes.configuration as qcconfig
from qcodes.logger.logger import conditionally_start_all_logging
from qcodes.utils import add_to_spyder_UMR_excludelist

from ._version import __version__

config: qcconfig.Config = qcconfig.Config()

conditionally_start_all_logging()

# we dont want spyder to reload qcodes as this will overwrite the default station
# instrument list and running monitor
add_to_spyder_UMR_excludelist('qcodes')


import atexit

import qcodes.validators
from qcodes.dataset import (
    Measurement,
    ParamSpec,
    SQLiteSettings,
    experiments,
    get_guids_by_run_spec,
    initialise_database,
    initialise_or_create_database_at,
    initialised_database_at,
    load_by_counter,
    load_by_guid,
    load_by_id,
    load_by_run_spec,
    load_experiment,
    load_experiment_by_name,
    load_last_experiment,
    load_or_create_experiment,
    new_data_set,
    new_experiment,
)
from qcodes.instrument import (
    ChannelList,
    ChannelTuple,
    Instrument,
    InstrumentChannel,
    IPInstrument,
    VisaInstrument,
    find_or_create_instrument,
)
from qcodes.instrument_drivers.test import test_instrument, test_instruments
from qcodes.monitor import Monitor
from qcodes.parameters import (
    ArrayParameter,
    CombinedParameter,
    DelegateParameter,
    Function,
    ManualParameter,
    MultiParameter,
    Parameter,
    ParameterWithSetpoints,
    ScaledParameter,
    SweepFixedValues,
    SweepValues,
    combine,
)
from qcodes.station import Station

# ensure to close all instruments when interpreter is closed
atexit.register(Instrument.close_all)

if config.core.import_legacy_api:
    from qcodes.utils import QCoDeSDeprecationWarning

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
