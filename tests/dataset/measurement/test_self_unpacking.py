from typing import TYPE_CHECKING

import numpy as np
import pytest

from qcodes.dataset import Measurement
from qcodes.dataset.measurements import (
    _non_numeric_values_are_equal,
    _numeric_values_are_equal,
    _values_are_equal,
)
from qcodes.parameters import (
    ManualParameter,
    Parameter,
    ParameterWithSetpoints,
    ParamRawDataType,
)
from qcodes.validators import Arrays

if TYPE_CHECKING:
    from collections.abc import Generator

    from qcodes.dataset.data_set_protocol import ValuesType
    from qcodes.parameters.parameter_base import ParameterBase


class ControllingParameter(Parameter):
    def __init__(
        self, name: str, components: dict[Parameter, tuple[float, float]]
    ) -> None:
        super().__init__(name=name, get_cmd=False)
        # dict of Parameter to (slope, offset) of components
        self._components_dict: dict[Parameter, tuple[float, float]] = components
        for param in self._components_dict.keys():
            self._has_control_of.add(param)
            param.is_controlled_by.add(self)

    def set_raw(self, value: ParamRawDataType) -> None:
        # Set all dependent parameters based on their mapping functions
        for param, slope_offset in self._components_dict.items():
            param(value * slope_offset[0] + slope_offset[1])

    def unpack_self(
        self, value: "ValuesType"
    ) -> list[tuple["ParameterBase", "ValuesType"]]:
        assert isinstance(value, float)
        unpacked_results = super().unpack_self(value)
        for param, slope_offset in self._components_dict.items():
            unpacked_results.append((param, value * slope_offset[0] + slope_offset[1]))
        return unpacked_results


@pytest.fixture
def controlling_parameters() -> (
    "Generator[tuple[ControllingParameter, ManualParameter, ManualParameter], None, None]"
):
    comp1 = ManualParameter("comp1")
    comp2 = ManualParameter("comp2")
    control1 = ControllingParameter(
        "control1", components={comp1: (1, 0), comp2: (-1, 10)}
    )
    yield control1, comp1, comp2


def test_add_result_self_unpack(controlling_parameters, experiment):
    control1, comp1, comp2 = controlling_parameters
    meas1 = ManualParameter("meas1")

    meas = Measurement(experiment)
    meas.register_parameter(meas1, setpoints=[control1])

    assert all(
        param in meas._registered_parameters
        for param in (comp1, comp2, control1, meas1)
    )

    with meas.run() as datasaver:
        for val in np.linspace(0, 1, 11):
            control1(val)
            datasaver.add_result((meas1, val + 1), (control1, val))
        ds = datasaver.dataset

    dataset_data = ds.get_parameter_data()
    meas1_data = dataset_data.get("meas1", None)
    assert meas1_data is not None
    assert all(
        param_name in meas1_data.keys()
        for param_name in ("meas1", "comp1", "comp2", "control1")
    )
    assert meas1_data["meas1"] == pytest.approx(np.linspace(1, 2, 11))
    assert meas1_data["control1"] == pytest.approx(np.linspace(0, 1, 11))
    assert meas1_data["comp1"] == pytest.approx(np.linspace(0, 1, 11))
    assert meas1_data["comp2"] == pytest.approx(np.linspace(10, 9, 11))


def test_add_result_self_unpack_with_PWS(controlling_parameters, experiment):
    control1, comp1, comp2 = controlling_parameters
    pws_setpoints = Parameter(
        "pws_setpoints",
        get_cmd=lambda: np.linspace(-1, 1, 11),
        vals=Arrays(shape=(11,)),
    )
    pws = ParameterWithSetpoints(
        "pws",
        setpoints=(pws_setpoints,),
        vals=Arrays(shape=(11,)),
        get_cmd=lambda: np.linspace(-2, 2, 11) + comp1(),
    )

    meas = Measurement(experiment)
    meas.register_parameter(pws, setpoints=[control1])

    assert all(
        param in meas._registered_parameters
        for param in (comp1, comp2, control1, pws, pws_setpoints)
    )

    with meas.run() as datasaver:
        for val in np.linspace(0, 1, 11):
            control1(val)
            datasaver.add_result((pws, pws()), (control1, val))
        ds = datasaver.dataset

    dataset_data = ds.get_parameter_data()
    pws_data = dataset_data.get("pws", None)
    assert (pws_data) is not None
    assert all(
        param_name in pws_data.keys()
        for param_name in ("pws", "comp1", "comp2", "control1", "pws_setpoints")
    )
    expected_setpoints, expected_control = np.meshgrid(
        np.linspace(-1, 1, 11), np.linspace(0, 1, 11)
    )
    assert pws_data["control1"] == pytest.approx(expected_control)
    assert pws_data["comp1"] == pytest.approx(expected_control)
    assert pws_data["comp2"] == pytest.approx(10 - expected_control)
    assert pws_data["pws_setpoints"] == pytest.approx(expected_setpoints)

    assert pws_data["control1"].shape == (11, 11)
    assert pws_data["comp1"].shape == (11, 11)
    assert pws_data["comp2"].shape == (11, 11)
    assert pws_data["pws"].shape == (11, 11)
    assert pws_data["pws_setpoints"].shape == (11, 11)


# Testing equality methods for deduplication
def test_non_numeric_values_are_equal() -> None:
    # test str
    assert _non_numeric_values_are_equal(np.array("string_val"), np.array("string_val"))
    assert not _non_numeric_values_are_equal(
        np.array("string_val"), np.array("different_string")
    )

    # test Sequence[str]
    seq_value1 = ["a", "b", "c", "d"]
    seq_value2 = ["a1", "b", "c", "d"]
    assert _non_numeric_values_are_equal(np.array(seq_value1), np.array(seq_value1))
    assert not _non_numeric_values_are_equal(np.array(seq_value1), np.array(seq_value2))

    # test NDArray[str]
    arr_value1 = np.array(seq_value1)
    arr_value2 = np.array(seq_value2)
    assert _non_numeric_values_are_equal(np.array(arr_value1), np.array(arr_value1))
    assert not _non_numeric_values_are_equal(np.array(arr_value1), np.array(arr_value2))


def test_numeric_values_are_equal() -> None:
    # test complex
    val1 = 1.0 + 3.0 * 1.0j
    val2 = 2.0 + 3.0 * 1.0j
    assert _numeric_values_are_equal(np.array(val1), np.array(val1))
    assert not _numeric_values_are_equal(np.array(val1), np.array(val2))

    # test complex w/ nans
    val1 = 1.0 + np.nan * 1.0j
    val2 = np.nan + 3.0 * 1.0j
    assert not _numeric_values_are_equal(np.array(val1), np.array(val2))

    # test ndarray[complex]
    real_1 = np.linspace(0, 1, 11)
    imag_1 = np.linspace(0, -1, 11)
    val1 = real_1 + 1.0j * imag_1

    real_2 = np.linspace(0, -1, 11)
    imag_2 = np.linspace(0, 1, 11)
    val2 = real_2 + 1.0j * imag_2
    assert _numeric_values_are_equal(np.array(val1), np.array(val1))
    assert not _numeric_values_are_equal(np.array(val1), np.array(val2))

    # test float
    val1 = 1.0
    val2 = 2.0
    assert _numeric_values_are_equal(np.array(val1), np.array(val1))
    assert not _numeric_values_are_equal(np.array(val1), np.array(val2))

    # test ndarray[float]
    val1 = np.linspace(0, 1, 11)
    val2 = np.linspace(0, -1, 11)
    assert _numeric_values_are_equal(np.array(val1), np.array(val1))
    assert not _numeric_values_are_equal(np.array(val1), np.array(val2))


def test_values_are_equal() -> None:
    # test Sequence[str]
    seq_value1 = ["a", "b", "c", "d"]
    seq_value2 = ["a1", "b", "c", "d"]
    assert _values_are_equal(np.array(seq_value1), np.array(seq_value1))
    assert not _values_are_equal(np.array(seq_value1), np.array(seq_value2))

    # test ndarray[complex]
    real_1 = np.linspace(0, 1, 11)
    imag_1 = np.linspace(0, -1, 11)
    val1 = real_1 + 1.0j * imag_1

    real_2 = np.linspace(0, -1, 11)
    imag_2 = np.linspace(0, 1, 11)
    val2 = real_2 + 1.0j * imag_2
    assert _values_are_equal(np.array(val1), np.array(val1))
    assert not _values_are_equal(np.array(val1), np.array(val2))

    # test float
    val1 = 1.0
    val2 = 2.0
    assert _values_are_equal(np.array(val1), np.array(val1))
    assert not _values_are_equal(np.array(val1), np.array(val2))
