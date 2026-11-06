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

import importlib
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import qcodes
import qcodes.dataset
import qcodes.instrument
import qcodes.monitor
import qcodes.parameters
import qcodes.station
import qcodes.validators
from qcodes.utils import QCoDeSDeprecationWarning

if TYPE_CHECKING:
    from pytest import MonkeyPatch


@pytest.mark.skipif(
    not hasattr(sys, "set_lazy_imports"), reason="requires Python 3.15 lazy imports"
)
def test_public_reexports_with_lazy_qcodes_imports() -> None:
    lazy_imports_path = Path(__file__).parents[1] / ".github" / "lazy-imports"
    code = textwrap.dedent(
        """
        import sys
        import warnings

        warnings.simplefilter("ignore")

        import qcodes.dataset as dataset
        import qcodes.utils as utils
        from qcodes.dataset._dond.do_nd import dond
        from qcodes.dataset.dond.do_nd import dond as deprecated_dond
        from qcodes.instrument_drivers import basel, stanford_research
        from qcodes.instrument_drivers.basel.BaselSP983 import BaselSP983
        from qcodes.instrument_drivers.stanford_research.SR830 import SR830
        from qcodes.utils._full_class import full_class
        from qcodes.utils.full_class import full_class as deprecated_full_class

        assert sys.get_lazy_imports_filter() is not None
        assert "qcodes.instrument_drivers.stanford_research.SR830" not in sys.modules

        # reify the lazy imports of the deprecated module aliases
        assert deprecated_dond is dond
        assert deprecated_full_class is full_class
        assert isinstance(SR830, type)
        assert isinstance(BaselSP983, type)

        assert dataset.dond is dond
        assert utils.full_class is full_class
        assert stanford_research.SR830 is SR830
        assert basel.BaselSP983 is BaselSP983
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


_DATASET = "qcodes.dataset"
_BASEL = "qcodes.instrument_drivers.basel"
_SRS = "qcodes.instrument_drivers.stanford_research"


@pytest.mark.parametrize(
    ("module_name", "parent_module_name", "attribute_name"),
    [
        ("qcodes.dataset.dond", _DATASET, "dond"),
        ("qcodes.dataset.dond.do_0d", _DATASET, "do0d"),
        ("qcodes.dataset.dond.do_1d", _DATASET, "do1d"),
        ("qcodes.dataset.dond.do_2d", _DATASET, "do2d"),
        ("qcodes.dataset.dond.do_nd", _DATASET, "dond"),
        ("qcodes.dataset.dond.do_nd_utils", _DATASET, "BreakConditionInterrupt"),
        ("qcodes.dataset.dond.sweeps", _DATASET, "LinSweep"),
        (f"{_BASEL}.BaselSP983", _BASEL, "BaselSP983"),
        (f"{_BASEL}.BaselSP983a", _BASEL, "BaselSP983a"),
        (f"{_BASEL}.BaselSP983c", _BASEL, "BaselSP983c"),
        (f"{_SRS}.SG384", _SRS, "SG384"),
        (f"{_SRS}.SR560", _SRS, "SR560"),
        (f"{_SRS}.SR830", _SRS, "SR830"),
        (f"{_SRS}.SR860", _SRS, "SR860"),
        (f"{_SRS}.SR865", _SRS, "SR865"),
        (f"{_SRS}.SR865A", _SRS, "SR865A"),
        (f"{_SRS}.SR86x", _SRS, "SR86x"),
        ("qcodes.utils.full_class", "qcodes.utils", "full_class"),
    ],
)
def test_deprecated_module_aliases(
    module_name: str,
    parent_module_name: str,
    attribute_name: str,
    monkeypatch: MonkeyPatch,
) -> None:
    parent_module = importlib.import_module(parent_module_name)
    expected = getattr(parent_module, attribute_name)
    # Import the alias (and for the dond subpackage, the package that warns)
    # afresh so that the deprecation warning is emitted in this test.
    for name in list(sys.modules):
        if name == module_name or name.startswith("qcodes.dataset.dond"):
            monkeypatch.delitem(sys.modules, name)

    with pytest.warns(
        QCoDeSDeprecationWarning, match=r"The module .* is deprecated"
    ) as warning_records:
        deprecated_module = importlib.import_module(module_name)

    assert len(warning_records) == 1
    if module_name != "qcodes.dataset.dond":
        assert getattr(deprecated_module, attribute_name) is expected
        for name in deprecated_module.__all__:
            assert getattr(deprecated_module, name) is not None
    # importing a module sets it as an attribute on its parent package.
    # Make sure that this does not replace the public object of the same name.
    assert getattr(parent_module, attribute_name) is expected


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
