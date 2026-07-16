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

import warnings

import pytest

import qcodes
import qcodes.dataset
import qcodes.instrument
import qcodes.monitor
import qcodes.parameters
import qcodes.station
import qcodes.validators
from qcodes.utils import QCoDeSDeprecationWarning


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
        qcodes.definitely_not_a_qcodes_attribute  # type: ignore[attr-defined]


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
        qcodes.instrument.definitely_not_a_qcodes_attribute  # type: ignore[attr-defined]
