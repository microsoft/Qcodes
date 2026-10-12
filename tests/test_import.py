"""Tests for the lazily provided attributes on the top level ``qcodes``
namespace and on ``qcodes.instrument``.

The short hand aliases on ``qcodes`` (such as ``qcodes.Parameter``) and the
parameter classes re-exported from ``qcodes.instrument`` are provided lazily via
module level ``__getattr__`` hooks that are hidden from static type checkers.
Accessing them still works at runtime for backwards compatibility but is
deprecated: it emits a ``QCoDeSDeprecationWarning`` at runtime *and* is reported
as an unknown attribute by static type checkers. The ``# type: ignore`` pragmas
below document (and, via the pre-commit type checkers, assert) that these lines
are indeed type errors. Importing the names from their respective submodules is
the supported, statically typed alternative.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

import qcodes
import qcodes.dataset
import qcodes.instrument
import qcodes.monitor
import qcodes.parameters
import qcodes.station
import qcodes.validators
from qcodes.utils import QCoDeSDeprecationWarning


@pytest.mark.skipif(
    not hasattr(sys, "set_lazy_imports"), reason="requires Python 3.15 lazy imports"
)
def test_public_reexports_with_lazy_qcodes_imports() -> None:
    lazy_imports_path = Path(__file__).parents[1] / ".github" / "lazy-imports"
    code = textwrap.dedent(
        """
        import sys

        import qcodes.dataset as dataset
        import qcodes.dataset._dond.do_nd
        import qcodes.utils as utils
        import qcodes.utils._full_class
        from qcodes.instrument_drivers import basel, stanford_research

        assert callable(dataset.dond)
        assert callable(utils.full_class)
        assert callable(stanford_research.SR830)
        assert callable(basel.BaselSP983)

        lazy_filter = sys.get_lazy_imports_filter()
        assert lazy_filter is not None
        assert not lazy_filter(None, "qcodes.dataset.dond", None)
        assert not lazy_filter(None, "qcodes.utils.full_class", None)
        """
    )

    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={
            **os.environ,
            "PYTHON_LAZY_IMPORTS": "all",
            "PYTHONPATH": os.pathsep.join(
                (str(lazy_imports_path), os.environ.get("PYTHONPATH", ""))
            ),
        },
    )


@pytest.mark.parametrize(
    ("module_name", "parent_module_name", "attribute_name"),
    [
        (
            "qcodes.dataset.dond.do_nd",
            "qcodes.dataset",
            "dond",
        ),
        (
            "qcodes.instrument_drivers.basel.BaselSP983",
            "qcodes.instrument_drivers.basel",
            "BaselSP983",
        ),
        (
            "qcodes.instrument_drivers.stanford_research.SR830",
            "qcodes.instrument_drivers.stanford_research",
            "SR830",
        ),
        ("qcodes.utils.full_class", "qcodes.utils", "full_class"),
    ],
)
def test_deprecated_module_aliases(
    module_name: str, parent_module_name: str, attribute_name: str
) -> None:
    code = textwrap.dedent(
        f"""
        import importlib
        import warnings

        from qcodes.utils.deprecate import QCoDeSDeprecationWarning

        parent_module = importlib.import_module({parent_module_name!r})
        expected = getattr(parent_module, {attribute_name!r})
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always", QCoDeSDeprecationWarning)
            deprecated_module = importlib.import_module({module_name!r})

        assert getattr(deprecated_module, {attribute_name!r}) is expected
        assert len(warning_records) == 1
        assert issubclass(warning_records[0].category, QCoDeSDeprecationWarning)
        assert "module" in str(warning_records[0].message)
        assert "deprecated" in str(warning_records[0].message)
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_top_level_parameter_shorthand_deprecated() -> None:
    with pytest.warns(
        QCoDeSDeprecationWarning, match=r"top level.*'qcodes\.parameters'"
    ):
        obj = qcodes.Parameter  # type: ignore[attr-defined]
    assert obj is qcodes.parameters.Parameter


def test_top_level_combine_shorthand_deprecated() -> None:
    with pytest.warns(
        QCoDeSDeprecationWarning, match=r"top level.*'qcodes\.parameters'"
    ):
        obj = qcodes.combine  # type: ignore[attr-defined]
    assert obj is qcodes.parameters.combine


def test_top_level_measurement_shorthand_deprecated() -> None:
    with pytest.warns(QCoDeSDeprecationWarning, match=r"top level.*'qcodes\.dataset'"):
        obj = qcodes.Measurement  # type: ignore[attr-defined]
    assert obj is qcodes.dataset.Measurement


def test_top_level_instrument_shorthand_deprecated() -> None:
    with pytest.warns(
        QCoDeSDeprecationWarning, match=r"top level.*'qcodes\.instrument'"
    ):
        obj = qcodes.Instrument  # type: ignore[attr-defined]
    assert obj is qcodes.instrument.Instrument


def test_top_level_monitor_shorthand_deprecated() -> None:
    with pytest.warns(QCoDeSDeprecationWarning, match=r"top level.*'qcodes\.monitor'"):
        obj = qcodes.Monitor  # type: ignore[attr-defined]
    assert obj is qcodes.monitor.Monitor


def test_top_level_station_shorthand_deprecated() -> None:
    with pytest.warns(QCoDeSDeprecationWarning, match=r"top level.*'qcodes\.station'"):
        obj = qcodes.Station  # type: ignore[attr-defined]
    assert obj is qcodes.station.Station


def test_top_level_submodule_access_not_deprecated() -> None:
    """Accessing the (imported) submodules from ``qcodes`` is statically typed
    and does not emit a deprecation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", QCoDeSDeprecationWarning)
        assert qcodes.dataset is not None
        assert qcodes.instrument is not None
        assert qcodes.parameters is not None
        assert qcodes.monitor is not None
        assert qcodes.station is not None
        assert qcodes.validators is not None


def test_top_level_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="definitely_not_a_qcodes_attribute"):
        _ = qcodes.definitely_not_a_qcodes_attribute  # type: ignore[attr-defined]


def test_instrument_parameter_reexport_deprecated() -> None:
    """Document that a parameter class accessed from ``qcodes.instrument`` is
    both a runtime deprecation warning and a static type error."""
    with pytest.warns(
        QCoDeSDeprecationWarning, match=r"'ManualParameter'.*'qcodes\.parameters'"
    ):
        obj = qcodes.instrument.ManualParameter  # type: ignore[attr-defined]
    assert obj is qcodes.parameters.ManualParameter


@pytest.mark.parametrize(
    "name",
    [
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
    ],
)
def test_instrument_parameter_reexport_deprecated_all(name: str) -> None:
    """Every parameter class re-exported from ``qcodes.instrument`` still
    resolves to the object in ``qcodes.parameters`` and emits a warning."""
    with pytest.warns(
        QCoDeSDeprecationWarning, match=rf"{name!r}.*'qcodes\.parameters'"
    ):
        obj = getattr(qcodes.instrument, name)
    assert obj is getattr(qcodes.parameters, name)


def test_instrument_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="definitely_not_a_qcodes_attribute"):
        _ = qcodes.instrument.definitely_not_a_qcodes_attribute  # type: ignore[attr-defined]
