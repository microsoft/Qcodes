"""Set up the main qcodes namespace."""

import importlib
import warnings
from typing import TYPE_CHECKING, Any

import qcodes.configuration as qcconfig
from qcodes.logger.logger import conditionally_start_all_logging
from qcodes.utils import QCoDeSDeprecationWarning

config: qcconfig.Config = qcconfig.Config()

conditionally_start_all_logging()

if config.core.import_legacy_api:
    warnings.warn(
        "`core.import_legacy_api` and `gui.plotlib` config option has no effect "
        "and will be removed in the future. "
        "Please avoid setting this in your `qcodesrc.json` config file.",
        QCoDeSDeprecationWarning,
    )

# The following names are re-exported for backwards compatibility as short hand
# for the objects in their respective submodules. Importing them from the top
# level ``qcodes`` namespace is deprecated (import them from the submodule
# instead); they are provided lazily via a module level ``__getattr__`` that
# emits a ``QCoDeSDeprecationWarning``. Importing them eagerly here would also
# pull ``qcodes.dataset``, ``qcodes.instrument``, ``qcodes.parameters``,
# ``qcodes.monitor`` and ``qcodes.station`` into a single large import cycle at
# type-check time (which triggers an internal error in mypy >= 2.2).
_LAZY_NAME_TO_MODULE = (
    {
        name: "qcodes.dataset"
        for name in (
            "Measurement",
            "ParamSpec",
            "SQLiteSettings",
            "experiments",
            "get_guids_by_run_spec",
            "initialise_database",
            "initialise_or_create_database_at",
            "initialised_database_at",
            "load_by_counter",
            "load_by_guid",
            "load_by_id",
            "load_by_run_spec",
            "load_experiment",
            "load_experiment_by_name",
            "load_last_experiment",
            "load_or_create_experiment",
            "new_data_set",
            "new_experiment",
        )
    }
    | {
        name: "qcodes.instrument"
        for name in (
            "ChannelList",
            "ChannelTuple",
            "Instrument",
            "InstrumentChannel",
            "IPInstrument",
            "VisaInstrument",
            "find_or_create_instrument",
        )
    }
    | {
        name: "qcodes.parameters"
        for name in (
            "ArrayParameter",
            "CombinedParameter",
            "DelegateParameter",
            "Function",
            "ManualParameter",
            "MultiParameter",
            "Parameter",
            "ParameterWithSetpoints",
            "ScaledParameter",
            "SweepFixedValues",
            "SweepValues",
            "combine",
        )
    }
    | {
        "Monitor": "qcodes.monitor",
        "Station": "qcodes.station",
    }
)


# These public submodules are exposed lazily so that ``import qcodes`` stays
# cheap and does not force the submodules (and their dependencies) to be
# imported. Accessing them is not deprecated. The submodules that the deprecated
# short hands above live in are included here so that ``qcodes.dataset`` and
# friends keep working as submodule accessors once the deprecated short hands are
# eventually removed.
_LAZY_SUBMODULES = frozenset(
    {
        "validators",
        "dataset",
        "instrument",
        "parameters",
        "monitor",
        "station",
    }
)


# The lazy ``__getattr__`` is intentionally hidden from static type checkers via
# ``if not TYPE_CHECKING`` so that these discouraged short hands are reported as
# unknown attributes (rather than typed as ``Any``); import the names from their
# respective submodules to get proper type information.
#
# ``__version__`` is also resolved lazily: computing it is slow in an editable
# install since we have to shell out to run ``git describe``, so we only do it if
# it is actually requested. Under ``TYPE_CHECKING`` it is defined below with a
# placeholder value so static type checkers still know it exists.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        if name == "__version__":
            import qcodes._version  # noqa: PLC0415  # lazy import since getting version is slow in editable install.

            return qcodes._version.__version__
        if name in _LAZY_SUBMODULES:
            return importlib.import_module(f"{__name__}.{name}")
        module_name = _LAZY_NAME_TO_MODULE.get(name)
        if module_name is not None:
            warnings.warn(
                f"Importing {name!r} from the top level {__name__!r} namespace "
                f"is deprecated. Import it from {module_name!r} instead.",
                QCoDeSDeprecationWarning,
                stacklevel=2,
            )
            return getattr(importlib.import_module(module_name), name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
else:
    __version__ = "not_set"

__all__ = [
    "__version__",
    "config",
]
