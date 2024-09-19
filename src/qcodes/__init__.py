"""Set up the main qcodes namespace."""

# ruff: noqa: F401, E402
# This module still contains a lot of short hand imports
# since these imports are discouraged and they are officially
# added elsewhere under their respective submodules we cannot add
# them to __all__ here so silence the warning.

# config
import warnings
from typing import Any

from typing_extensions import deprecated

import qcodes._version
import qcodes.configuration as qcconfig
from qcodes.logger.logger import conditionally_start_all_logging
from qcodes.utils import QCoDeSDeprecationWarning
from qcodes.utils.spyder_utils import add_to_spyder_UMR_excludelist

__version__ = qcodes._version.__version__


config: qcconfig.Config = qcconfig.Config()

conditionally_start_all_logging()

# we dont want spyder to reload qcodes as this will overwrite the default station
# instrument list and running monitor
add_to_spyder_UMR_excludelist("qcodes")


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
from qcodes.utils import deprecate

# ensure to close all instruments when interpreter is closed
atexit.register(Instrument.close_all)

if config.core.import_legacy_api:
    warnings.warn(
        "`core.import_legacy_api` and `gui.plotlib` config option has no effect "
        "and will be removed in the future. "
        "Please avoid setting this in your `qcodesrc.json` config file.",
        QCoDeSDeprecationWarning,
    )


@deprecated(
    "tests are no longer shipped as part of QCoDeS. Clone git repo to matching tag and run `pytest tests` from the root of the repo.",
    category=QCoDeSDeprecationWarning,
)
def test(**kwargs: Any) -> int:
    """
    Deprecated
    """
    return 0


del deprecated

test.__test__ = False  # type: ignore[attr-defined] # Don't try to run this method as a test
