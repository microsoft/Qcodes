"""Tests for _add_inferred_data_vars with multiple inferred-from parents.

These tests verify that the inferred parameter's data size is checked
against the xr_dataset dimensions rather than individual parent sizes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import xarray as xr

from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.exporters.export_to_xarray import _add_inferred_data_vars
from qcodes.parameters import ParamSpecBase

if TYPE_CHECKING:
    import pytest


def _make_mock_dataset(
    interdeps: InterDependencies_,
    run_id: int = 1,
) -> MagicMock:
    """Create a minimal mock DataSetProtocol with the given interdeps."""
    ds = MagicMock()
    ds.description.interdeps = interdeps
    ds.run_id = run_id
    return ds


def _make_interdeps(
    *,
    deps: dict[ParamSpecBase, tuple[ParamSpecBase, ...]],
    inferences: dict[ParamSpecBase, tuple[ParamSpecBase, ...]],
) -> InterDependencies_:
    return InterDependencies_(dependencies=deps, inferences=inferences)


# ---------------------------------------------------------------------------
# Scenario: inferred param has ONE parent, sizes match → included
# (baseline sanity check)
# ---------------------------------------------------------------------------
class TestSingleParentBaseline:
    def test_single_parent_matching_size_is_included(self) -> None:
        """An inferred param whose data matches its single parent is added."""
        sp = ParamSpecBase("sp", "numeric")
        meas = ParamSpecBase("meas", "numeric")
        inf = ParamSpecBase("inf_param", "numeric")

        interdeps = _make_interdeps(
            deps={meas: (sp,)},
            inferences={inf: (meas,)},
        )
        ds = _make_mock_dataset(interdeps)

        n = 10
        sub_dict: dict[str, np.ndarray] = {
            "sp": np.arange(n, dtype=float),
            "meas": np.random.randn(n),
            "inf_param": np.linspace(0, 1, n),
        }

        xr_ds = xr.Dataset(
            {"meas": (("sp",), sub_dict["meas"])},
            coords={"sp": sub_dict["sp"]},
        )

        result = _add_inferred_data_vars(ds, "meas", sub_dict, xr_ds)

        assert "inf_param" in result.data_vars
        npt.assert_array_almost_equal(result["inf_param"].values, sub_dict["inf_param"])


# ---------------------------------------------------------------------------
# Scenario: inferred param has TWO parents, data matches BOTH
# → should always be included regardless of strategy
# ---------------------------------------------------------------------------
class TestMultipleParentsAllMatch:
    def test_inferred_matches_all_parents_is_included(self) -> None:
        """When data size matches all parents, the inferred param is added."""
        sp = ParamSpecBase("sp", "numeric")
        parent1 = ParamSpecBase("parent1", "numeric")
        parent2 = ParamSpecBase("parent2", "numeric")
        inf = ParamSpecBase("inf_param", "numeric")

        interdeps = _make_interdeps(
            deps={parent1: (sp,), parent2: (sp,)},
            inferences={inf: (parent1, parent2)},
        )
        ds = _make_mock_dataset(interdeps)

        n = 10
        sub_dict: dict[str, np.ndarray] = {
            "sp": np.arange(n, dtype=float),
            "parent1": np.random.randn(n),
            "parent2": np.random.randn(n),
            "inf_param": np.linspace(0, 1, n),
        }

        xr_ds = xr.Dataset(
            {
                "parent1": (("sp",), sub_dict["parent1"]),
                "parent2": (("sp",), sub_dict["parent2"]),
            },
            coords={"sp": sub_dict["sp"]},
        )

        result = _add_inferred_data_vars(ds, "parent1", sub_dict, xr_ds)

        assert "inf_param" in result.data_vars
        npt.assert_array_almost_equal(result["inf_param"].values, sub_dict["inf_param"])


# ---------------------------------------------------------------------------
# Scenario: inferred param has TWO parents with DIFFERENT sizes,
# data matches only the FIRST parent
# → current "match any" includes it; "match all" would reject it
# ---------------------------------------------------------------------------
class TestMultipleParentsOnlyFirstMatches:
    def test_inferred_matches_first_parent_only(self) -> None:
        """Data matches parent1 (size 10) but not parent2 (size 5).

        Current behavior: included (matches any parent).
        If "match all" were required, this would NOT be included.
        """
        sp1 = ParamSpecBase("sp1", "numeric")
        sp2 = ParamSpecBase("sp2", "numeric")
        parent1 = ParamSpecBase("parent1", "numeric")
        parent2 = ParamSpecBase("parent2", "numeric")
        inf = ParamSpecBase("inf_param", "numeric")

        interdeps = _make_interdeps(
            deps={parent1: (sp1,), parent2: (sp2,)},
            inferences={inf: (parent1, parent2)},
        )
        ds = _make_mock_dataset(interdeps)

        n1, n2 = 10, 5
        sub_dict: dict[str, np.ndarray] = {
            "sp1": np.arange(n1, dtype=float),
            "sp2": np.arange(n2, dtype=float),
            "parent1": np.random.randn(n1),
            "parent2": np.random.randn(n2),
            # inf_param has the same size as parent1
            "inf_param": np.linspace(0, 1, n1),
        }

        xr_ds = xr.Dataset(
            {"parent1": (("sp1",), sub_dict["parent1"])},
            coords={"sp1": sub_dict["sp1"]},
        )

        result = _add_inferred_data_vars(ds, "parent1", sub_dict, xr_ds)

        # Current behavior: included because it matches parent1
        assert "inf_param" in result.data_vars
        npt.assert_array_almost_equal(result["inf_param"].values, sub_dict["inf_param"])


# ---------------------------------------------------------------------------
# Scenario: inferred param has TWO parents with DIFFERENT sizes,
# data matches only the SECOND parent
# → current "match any" includes it; "match all" would reject it
# ---------------------------------------------------------------------------
class TestMultipleParentsOnlySecondMatches:
    def test_inferred_matches_second_parent_not_dataset_dims_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Data matches parent2 (size 5) but not the dataset dims (size 10).

        The inferred param should NOT be included because its data cannot
        be reshaped to the xr_dataset dimensions. A warning is emitted.
        """
        sp1 = ParamSpecBase("sp1", "numeric")
        sp2 = ParamSpecBase("sp2", "numeric")
        parent1 = ParamSpecBase("parent1", "numeric")
        parent2 = ParamSpecBase("parent2", "numeric")
        inf = ParamSpecBase("inf_param", "numeric")

        interdeps = _make_interdeps(
            deps={parent1: (sp1,), parent2: (sp2,)},
            inferences={inf: (parent1, parent2)},
        )
        ds = _make_mock_dataset(interdeps)

        n1, n2 = 10, 5
        sub_dict: dict[str, np.ndarray] = {
            "sp1": np.arange(n1, dtype=float),
            "sp2": np.arange(n2, dtype=float),
            "parent1": np.random.randn(n1),
            "parent2": np.random.randn(n2),
            # inf_param has the same size as parent2 but NOT the dataset dims
            "inf_param": np.linspace(0, 1, n2),
        }

        xr_ds = xr.Dataset(
            {"parent1": (("sp1",), sub_dict["parent1"])},
            coords={"sp1": sub_dict["sp1"]},
        )

        with caplog.at_level(
            logging.WARNING, logger="qcodes.dataset.exporters.export_to_xarray"
        ):
            result = _add_inferred_data_vars(ds, "parent1", sub_dict, xr_ds)

        assert "inf_param" not in result.data_vars
        assert any(
            "Cannot add inferred parameter 'inf_param'" in msg
            for msg in caplog.messages
        )


# ---------------------------------------------------------------------------
# Scenario: inferred param has TWO parents, data matches NEITHER
# → should be excluded and emit a warning
# ---------------------------------------------------------------------------
class TestMultipleParentsNoneMatch:
    def test_inferred_matches_no_parent_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Data size doesn't match any parent → warning, not included."""
        sp = ParamSpecBase("sp", "numeric")
        parent1 = ParamSpecBase("parent1", "numeric")
        parent2 = ParamSpecBase("parent2", "numeric")
        inf = ParamSpecBase("inf_param", "numeric")

        interdeps = _make_interdeps(
            deps={parent1: (sp,), parent2: (sp,)},
            inferences={inf: (parent1, parent2)},
        )
        ds = _make_mock_dataset(interdeps)

        n = 10
        sub_dict: dict[str, np.ndarray] = {
            "sp": np.arange(n, dtype=float),
            "parent1": np.random.randn(n),
            "parent2": np.random.randn(n),
            # inf_param has a completely different size
            "inf_param": np.linspace(0, 1, 7),
        }

        xr_ds = xr.Dataset(
            {"parent1": (("sp",), sub_dict["parent1"])},
            coords={"sp": sub_dict["sp"]},
        )

        with caplog.at_level(
            logging.WARNING, logger="qcodes.dataset.exporters.export_to_xarray"
        ):
            result = _add_inferred_data_vars(ds, "parent1", sub_dict, xr_ds)

        assert "inf_param" not in result.data_vars
        assert any(
            "Cannot add inferred parameter 'inf_param'" in msg
            for msg in caplog.messages
        )


# ---------------------------------------------------------------------------
# Scenario: inferred param has TWO parents, only one is in sub_dict
# → should match against the available parent
# ---------------------------------------------------------------------------
class TestMultipleParentsOneUnavailable:
    def test_matches_available_parent_ignores_missing(self) -> None:
        """When one parent is not in sub_dict, the other is still checked."""
        sp = ParamSpecBase("sp", "numeric")
        parent1 = ParamSpecBase("parent1", "numeric")
        parent2 = ParamSpecBase("parent2", "numeric")
        inf = ParamSpecBase("inf_param", "numeric")

        interdeps = _make_interdeps(
            deps={parent1: (sp,), parent2: (sp,)},
            inferences={inf: (parent1, parent2)},
        )
        ds = _make_mock_dataset(interdeps)

        n = 10
        sub_dict: dict[str, np.ndarray] = {
            "sp": np.arange(n, dtype=float),
            "parent1": np.random.randn(n),
            # parent2 is NOT in sub_dict
            "inf_param": np.linspace(0, 1, n),
        }

        xr_ds = xr.Dataset(
            {"parent1": (("sp",), sub_dict["parent1"])},
            coords={"sp": sub_dict["sp"]},
        )

        result = _add_inferred_data_vars(ds, "parent1", sub_dict, xr_ds)

        assert "inf_param" in result.data_vars
        npt.assert_array_almost_equal(result["inf_param"].values, sub_dict["inf_param"])

    def test_warns_when_only_available_parent_mismatches(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One parent missing, the other has wrong size → warning."""
        sp = ParamSpecBase("sp", "numeric")
        parent1 = ParamSpecBase("parent1", "numeric")
        parent2 = ParamSpecBase("parent2", "numeric")
        inf = ParamSpecBase("inf_param", "numeric")

        interdeps = _make_interdeps(
            deps={parent1: (sp,), parent2: (sp,)},
            inferences={inf: (parent1, parent2)},
        )
        ds = _make_mock_dataset(interdeps)

        n = 10
        sub_dict: dict[str, np.ndarray] = {
            "sp": np.arange(n, dtype=float),
            "parent1": np.random.randn(n),
            # parent2 missing, inf_param has wrong size
            "inf_param": np.linspace(0, 1, 7),
        }

        xr_ds = xr.Dataset(
            {"parent1": (("sp",), sub_dict["parent1"])},
            coords={"sp": sub_dict["sp"]},
        )

        with caplog.at_level(
            logging.WARNING, logger="qcodes.dataset.exporters.export_to_xarray"
        ):
            result = _add_inferred_data_vars(ds, "parent1", sub_dict, xr_ds)

        assert "inf_param" not in result.data_vars
        assert any(
            "Cannot add inferred parameter 'inf_param'" in msg
            for msg in caplog.messages
        )


# ---------------------------------------------------------------------------
# Scenario: inferred param has TWO parents of SAME size, both match
# → should be included; the "match any" and "match all" give same result
# ---------------------------------------------------------------------------
class TestMultipleParentsSameSizeAllMatch:
    def test_both_parents_same_size_included(self) -> None:
        """Both parents have same size and match → included either way."""
        sp = ParamSpecBase("sp", "numeric")
        parent1 = ParamSpecBase("parent1", "numeric")
        parent2 = ParamSpecBase("parent2", "numeric")
        inf = ParamSpecBase("inf_param", "numeric")

        interdeps = _make_interdeps(
            deps={parent1: (sp,), parent2: (sp,)},
            inferences={inf: (parent1, parent2)},
        )
        ds = _make_mock_dataset(interdeps)

        n = 10
        sub_dict: dict[str, np.ndarray] = {
            "sp": np.arange(n, dtype=float),
            "parent1": np.random.randn(n),
            "parent2": np.random.randn(n),
            "inf_param": np.linspace(0, 1, n),
        }

        xr_ds = xr.Dataset(
            {"parent1": (("sp",), sub_dict["parent1"])},
            coords={"sp": sub_dict["sp"]},
        )

        result = _add_inferred_data_vars(ds, "parent1", sub_dict, xr_ds)

        assert "inf_param" in result.data_vars
        npt.assert_array_almost_equal(result["inf_param"].values, sub_dict["inf_param"])
