"""Tests for StructParameter."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from qcodes.dataset import Measurement
from qcodes.parameters import ManualParameter
from qcodes.parameters.struct_parameter import (
    StructParameter,
    _extract_field_value,
    _FieldParameter,
    _get_struct_fields,
    _infer_paramtype_from_annotation,
    _is_pydantic_model_class,
)

if TYPE_CHECKING:
    from qcodes.parameters.parameter_base import ParamRawDataType


# --- Test struct types ---


@dataclasses.dataclass
class SimpleResult:
    voltage: float
    current: float


@dataclasses.dataclass
class MixedResult:
    name: str
    value: float
    count: int
    flag: bool


@dataclasses.dataclass
class ComplexFieldResult:
    impedance: complex
    signal: float


@dataclasses.dataclass
class ArrayFieldResult:
    trace: np.ndarray
    amplitude: float


@dataclasses.dataclass
class EmptyStruct:
    pass


# --- Concrete StructParameter subclasses for testing ---


class SimpleStructParam(StructParameter[SimpleResult, None]):
    def __init__(self, name: str, result: SimpleResult, **kwargs: Any) -> None:
        self._result = result
        super().__init__(name, struct_type=SimpleResult, **kwargs)

    def get_raw(self) -> ParamRawDataType:
        return self._result


class MixedStructParam(StructParameter[MixedResult, None]):
    def __init__(self, name: str, result: MixedResult, **kwargs: Any) -> None:
        self._result = result
        super().__init__(name, struct_type=MixedResult, **kwargs)

    def get_raw(self) -> ParamRawDataType:
        return self._result


class ComplexStructParam(StructParameter[ComplexFieldResult, None]):
    def __init__(self, name: str, result: ComplexFieldResult, **kwargs: Any) -> None:
        self._result = result
        super().__init__(name, struct_type=ComplexFieldResult, **kwargs)

    def get_raw(self) -> ParamRawDataType:
        return self._result


# --- Helper function tests ---


class TestIsPydanticModelClass:
    def test_dataclass_is_not_pydantic(self) -> None:
        assert not _is_pydantic_model_class(SimpleResult)

    def test_regular_class_is_not_pydantic(self) -> None:
        assert not _is_pydantic_model_class(int)

    def test_non_type_is_not_pydantic(self) -> None:
        assert not _is_pydantic_model_class(42)  # type: ignore[arg-type]


class TestGetStructFields:
    def test_dataclass_fields(self) -> None:
        fields = _get_struct_fields(SimpleResult)
        assert len(fields) == 2
        assert fields[0] == ("voltage", float)
        assert fields[1] == ("current", float)

    def test_mixed_dataclass_fields(self) -> None:
        fields = _get_struct_fields(MixedResult)
        assert len(fields) == 4
        names = [f[0] for f in fields]
        assert names == ["name", "value", "count", "flag"]

    def test_non_struct_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a dataclass"):
            _get_struct_fields(int)

    def test_regular_class_raises(self) -> None:
        class NotAStruct:
            x: int = 5

        with pytest.raises(TypeError, match="must be a dataclass"):
            _get_struct_fields(NotAStruct)


class TestInferParamtype:
    def test_float(self) -> None:
        assert _infer_paramtype_from_annotation(float) == "numeric"

    def test_int(self) -> None:
        assert _infer_paramtype_from_annotation(int) == "numeric"

    def test_bool(self) -> None:
        assert _infer_paramtype_from_annotation(bool) == "numeric"

    def test_str(self) -> None:
        assert _infer_paramtype_from_annotation(str) == "text"

    def test_complex(self) -> None:
        assert _infer_paramtype_from_annotation(complex) == "complex"

    def test_ndarray(self) -> None:
        assert _infer_paramtype_from_annotation(np.ndarray) == "array"

    def test_nested_dataclass_raises(self) -> None:
        with pytest.raises(TypeError, match="Nested structured types"):
            _infer_paramtype_from_annotation(SimpleResult)

    def test_unknown_defaults_to_numeric(self) -> None:
        assert _infer_paramtype_from_annotation(bytes) == "numeric"


class TestExtractFieldValue:
    def test_dataclass(self) -> None:
        result = SimpleResult(voltage=1.5, current=0.3)
        assert _extract_field_value(result, "voltage") == 1.5
        assert _extract_field_value(result, "current") == 0.3


# --- FieldParameter tests ---


class TestFieldParameter:
    def test_basic_creation(self) -> None:
        fp = _FieldParameter("test_field", label="Test", unit="V")
        assert fp.name == "test_field"
        assert fp.label == "Test"
        assert fp.unit == "V"
        assert not fp.gettable
        assert not fp.settable

    def test_default_label_and_unit(self) -> None:
        fp = _FieldParameter("my_field")
        assert fp.label == "my_field"
        assert fp.unit == ""

    def test_paramtype(self) -> None:
        fp = _FieldParameter("f", paramtype="text")
        assert fp.paramtype == "text"

    def test_not_bound_to_instrument(self) -> None:
        fp = _FieldParameter("f")
        assert fp.instrument is None

    def test_snapshot_excluded(self) -> None:
        fp = _FieldParameter("f")
        assert fp.snapshot_exclude is False
        assert fp.snapshot_value is False


# --- StructParameter tests ---


class TestStructParameterInit:
    def test_basic_creation(self) -> None:
        result = SimpleResult(voltage=1.0, current=0.5)
        param = SimpleStructParam("iv", result=result)
        assert param.name == "iv"
        assert param.struct_type is SimpleResult
        assert param.struct_type_name == "SimpleResult"
        assert param.names == ("voltage", "current")
        assert param.labels == ("voltage", "current")
        assert param.units == ("", "")
        assert param.gettable

    def test_custom_labels_and_units(self) -> None:
        result = SimpleResult(voltage=1.0, current=0.5)
        param = SimpleStructParam(
            "iv",
            result=result,
            field_labels={"voltage": "Voltage", "current": "Current"},
            field_units={"voltage": "V", "current": "A"},
        )
        assert param.labels == ("Voltage", "Current")
        assert param.units == ("V", "A")

    def test_custom_paramtypes(self) -> None:
        result = SimpleResult(voltage=1.0, current=0.5)
        param = SimpleStructParam(
            "iv",
            result=result,
            field_paramtypes={"voltage": "text"},
        )
        field_params = param.field_parameters
        assert field_params["voltage"].paramtype == "text"
        assert field_params["current"].paramtype == "numeric"

    def test_snapshot_value_defaults_to_false(self) -> None:
        result = SimpleResult(voltage=1.0, current=0.5)
        param = SimpleStructParam("iv", result=result)
        assert param.snapshot_value is False

    def test_snapshot_value_can_be_overridden(self) -> None:
        result = SimpleResult(voltage=1.0, current=0.5)
        param = SimpleStructParam("iv", result=result, snapshot_value=True)
        assert param.snapshot_value is True

    def test_empty_struct_raises(self) -> None:
        with pytest.raises(TypeError, match="has no fields"):

            class EmptyStructParam(StructParameter[EmptyStruct, None]):
                def get_raw(self) -> ParamRawDataType:
                    return EmptyStruct()

            EmptyStructParam("empty", struct_type=EmptyStruct)

    def test_non_struct_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a dataclass"):

            class BadStructParam(StructParameter[int, None]):
                def get_raw(self) -> ParamRawDataType:
                    return 42

            BadStructParam("bad", struct_type=int)

    def test_unknown_field_label_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown field names"):
            SimpleStructParam(
                "iv",
                result=SimpleResult(1.0, 0.5),
                field_labels={"nonexistent": "Nope"},
            )

    def test_unknown_field_unit_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown field names"):
            SimpleStructParam(
                "iv",
                result=SimpleResult(1.0, 0.5),
                field_units={"nonexistent": "X"},
            )

    def test_invalid_paramtype_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid paramtype"):
            SimpleStructParam(
                "iv",
                result=SimpleResult(1.0, 0.5),
                field_paramtypes={"voltage": "invalid"},
            )

    def test_nested_dataclass_field_raises(self) -> None:
        @dataclasses.dataclass
        class Outer:
            inner: SimpleResult

        with pytest.raises(TypeError, match="Nested structured types"):

            class NestedStructParam(StructParameter[Outer, None]):
                def get_raw(self) -> ParamRawDataType:
                    return Outer(inner=SimpleResult(1.0, 0.5))

            NestedStructParam("nested", struct_type=Outer)


class TestStructParameterFieldParameters:
    def test_field_parameters_dict(self) -> None:
        param = SimpleStructParam("iv", result=SimpleResult(1.0, 0.5))
        fps = param.field_parameters
        assert set(fps.keys()) == {"voltage", "current"}
        assert isinstance(fps["voltage"], _FieldParameter)
        assert isinstance(fps["current"], _FieldParameter)

    def test_field_param_names(self) -> None:
        param = SimpleStructParam("iv", result=SimpleResult(1.0, 0.5))
        fps = param.field_parameters
        assert fps["voltage"].name == "iv_voltage"
        assert fps["current"].name == "iv_current"

    def test_field_param_returns_copy(self) -> None:
        param = SimpleStructParam("iv", result=SimpleResult(1.0, 0.5))
        fps1 = param.field_parameters
        fps2 = param.field_parameters
        assert fps1 is not fps2
        assert fps1.keys() == fps2.keys()


class TestStructParameterNames:
    def test_short_names(self) -> None:
        param = SimpleStructParam("iv", result=SimpleResult(1.0, 0.5))
        assert param.short_names == ("voltage", "current")

    def test_full_names_no_instrument(self) -> None:
        param = SimpleStructParam("iv", result=SimpleResult(1.0, 0.5))
        assert param.full_names == ("iv_voltage", "iv_current")


class TestStructParameterGet:
    def test_get_returns_struct(self) -> None:
        result = SimpleResult(voltage=1.5, current=0.3)
        param = SimpleStructParam("iv", result=result)
        got = param.get()
        assert got == result

    def test_call_returns_struct(self) -> None:
        result = SimpleResult(voltage=1.5, current=0.3)
        param = SimpleStructParam("iv", result=result)
        got = param()
        assert got == result

    def test_mixed_types(self) -> None:
        result = MixedResult(name="test", value=1.5, count=42, flag=True)
        param = MixedStructParam("mixed", result=result)
        got = param.get()
        assert got.name == "test"
        assert got.value == 1.5
        assert got.count == 42
        assert got.flag is True

    def test_get_cmd_callable(self) -> None:
        expected = SimpleResult(voltage=2.0, current=0.5)
        param = StructParameter(
            "iv",
            struct_type=SimpleResult,
            get_cmd=lambda: expected,
        )
        got = param.get()
        assert got == expected

    def test_get_cmd_with_labels_and_units(self) -> None:
        expected = SimpleResult(voltage=3.0, current=1.0)
        param = StructParameter(
            "iv",
            struct_type=SimpleResult,
            get_cmd=lambda: expected,
            field_labels={"voltage": "V_out"},
            field_units={"current": "mA"},
        )
        assert param.get() == expected
        assert param.field_parameters["voltage"].label == "V_out"
        assert param.field_parameters["current"].unit == "mA"

    def test_get_cmd_with_subclass_get_raw_raises(self) -> None:
        with pytest.raises(
            TypeError,
            match="Supplying get_cmd to a StructParameter that already implements get_raw",
        ):
            SimpleStructParam(
                "iv",
                result=SimpleResult(voltage=1.0, current=0.1),
                get_cmd=lambda: SimpleResult(voltage=2.0, current=0.2),  # type: ignore[call-arg]
            )


class TestStructParameterUnpackSelf:
    def test_unpack_simple(self) -> None:
        result = SimpleResult(voltage=1.5, current=0.3)
        param = SimpleStructParam("iv", result=result)
        unpacked = param.unpack_self(result)  # type: ignore[arg-type]
        assert len(unpacked) == 2
        # Check that we get the field parameters with the right values
        param_names = [p.name for p, _ in unpacked]
        values = [v for _, v in unpacked]
        assert "iv_voltage" in param_names
        assert "iv_current" in param_names
        assert 1.5 in values
        assert 0.3 in values

    def test_unpack_does_not_include_self(self) -> None:
        result = SimpleResult(voltage=1.5, current=0.3)
        param = SimpleStructParam("iv", result=result)
        unpacked = param.unpack_self(result)  # type: ignore[arg-type]
        # None of the unpacked parameters should be the struct parameter itself
        for p, _ in unpacked:
            assert p is not param

    def test_unpack_mixed_types(self) -> None:
        result = MixedResult(name="hello", value=3.14, count=7, flag=False)
        param = MixedStructParam("data", result=result)
        unpacked = param.unpack_self(result)  # type: ignore[arg-type]
        assert len(unpacked) == 4
        values_dict = {p.name: v for p, v in unpacked}
        assert values_dict["data_name"] == "hello"
        assert values_dict["data_value"] == 3.14
        assert values_dict["data_count"] == 7
        assert values_dict["data_flag"] is False


class TestStructParameterSnapshot:
    def test_snapshot_no_value_by_default(self) -> None:
        result = SimpleResult(voltage=1.0, current=0.5)
        param = SimpleStructParam("iv", result=result)
        snap = param.snapshot()
        assert "value" not in snap
        assert "raw_value" not in snap

    def test_snapshot_includes_struct_metadata(self) -> None:
        result = SimpleResult(voltage=1.0, current=0.5)
        param = SimpleStructParam("iv", result=result)
        snap = param.snapshot()
        assert snap["struct_type_name"] == "SimpleResult"
        assert snap["names"] == ("voltage", "current")

    def test_snapshot_with_value(self) -> None:
        result = SimpleResult(voltage=1.0, current=0.5)
        param = SimpleStructParam("iv", result=result, snapshot_value=True)
        snap = param.snapshot(update=True)
        assert "value" in snap


class TestStructParameterDocstring:
    def test_docstring_generated(self) -> None:
        result = SimpleResult(voltage=1.0, current=0.5)
        param = SimpleStructParam("iv", result=result)
        assert param.__doc__ is not None
        assert "iv" in param.__doc__
        assert "SimpleResult" in param.__doc__

    def test_custom_docstring(self) -> None:
        result = SimpleResult(voltage=1.0, current=0.5)
        param = SimpleStructParam("iv", result=result, docstring="Custom docs")
        assert param.__doc__ is not None
        assert param.__doc__.startswith("Custom docs")


# --- Measurement integration tests ---


class TestStructParameterMeasurement:
    def test_register_struct_parameter(self, experiment: Any) -> None:

        setpoint = ManualParameter("x")
        result = SimpleResult(voltage=1.0, current=0.5)
        param = SimpleStructParam("iv", result=result)

        meas = Measurement(experiment)
        meas.register_parameter(setpoint)
        meas.register_parameter(param, setpoints=[setpoint])

        # Field parameters should be registered
        interdeps = meas._interdeps
        param_names = {ps.name for ps in interdeps.dependencies.keys()}
        assert "iv_voltage" in param_names
        assert "iv_current" in param_names

    def test_add_result_with_struct(self, experiment: Any) -> None:

        setpoint = ManualParameter("x")
        result = SimpleResult(voltage=1.5, current=0.3)
        param = SimpleStructParam("iv", result=result)

        meas = Measurement(experiment)
        meas.register_parameter(setpoint)
        meas.register_parameter(param, setpoints=[setpoint])

        with meas.run() as datasaver:
            for x_val in [0.0, 1.0, 2.0]:
                setpoint(x_val)
                struct_val = SimpleResult(voltage=x_val * 2, current=x_val * 0.1)
                datasaver.add_result(
                    (setpoint, x_val),
                    (param, struct_val),  # type: ignore[arg-type]
                )
            ds = datasaver.dataset

        data = ds.get_parameter_data()
        # Check that both field columns exist
        assert "iv_voltage" in data
        assert "iv_current" in data

        # Check the data values
        voltage_data = data["iv_voltage"]
        assert "iv_voltage" in voltage_data
        assert "x" in voltage_data
        np.testing.assert_array_almost_equal(
            voltage_data["iv_voltage"], [0.0, 2.0, 4.0]
        )
        np.testing.assert_array_almost_equal(voltage_data["x"], [0.0, 1.0, 2.0])

        current_data = data["iv_current"]
        assert "iv_current" in current_data
        np.testing.assert_array_almost_equal(
            current_data["iv_current"], [0.0, 0.1, 0.2]
        )

    def test_add_result_with_get(self, experiment: Any) -> None:
        """Test using param.get() and passing the struct value."""

        setpoint = ManualParameter("x")
        result = SimpleResult(voltage=3.0, current=1.5)
        param = SimpleStructParam("iv", result=result)

        meas = Measurement(experiment)
        meas.register_parameter(setpoint)
        meas.register_parameter(param, setpoints=[setpoint])

        with meas.run() as datasaver:
            setpoint(0.0)
            val = param.get()
            datasaver.add_result(
                (setpoint, 0.0),
                (param, val),  # type: ignore[arg-type]
            )
            ds = datasaver.dataset

        data = ds.get_parameter_data()
        assert "iv_voltage" in data
        voltage_data = data["iv_voltage"]
        np.testing.assert_array_almost_equal(voltage_data["iv_voltage"], [3.0])

    def test_mixed_types_measurement(self, experiment: Any) -> None:
        """Test struct with mixed field types in a measurement."""

        setpoint = ManualParameter("x")

        @dataclasses.dataclass
        class TextNumResult:
            label: str
            value: float

        class TextNumParam(StructParameter[TextNumResult, None]):
            def get_raw(self) -> ParamRawDataType:
                return TextNumResult(label="test", value=42.0)

        param = TextNumParam(
            "tn",
            struct_type=TextNumResult,
            field_paramtypes={"label": "text"},
        )

        meas = Measurement(experiment)
        meas.register_parameter(setpoint)
        meas.register_parameter(param, setpoints=[setpoint])

        with meas.run() as datasaver:
            setpoint(1.0)
            datasaver.add_result(
                (setpoint, 1.0),
                (param, TextNumResult(label="hello", value=99.0)),  # type: ignore[arg-type]
            )
            ds = datasaver.dataset

        data = ds.get_parameter_data()
        assert "tn_label" in data
        assert "tn_value" in data
