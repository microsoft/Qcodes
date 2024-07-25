from enum import StrEnum

from typing_extensions import assert_type

from qcodes.metadatable.metadatable_base import EmptyMetaDataModel
from qcodes.parameters import Parameter
from qcodes.parameters.parameter import ParameterSnapshot


def test_parameter_typed_metadata_basic():

    class ParamType(StrEnum):

        voltage = "voltage"
        current = "current"

    class MyParameterMetadata(EmptyMetaDataModel):
        param_type: ParamType

    a = Parameter(
        name="myparam",
        set_cmd=None,
        get_cmd=None,
        model=ParameterSnapshot,
        metadata_model=MyParameterMetadata,
    )
    a.metadata["param_type"] = (
        ParamType.voltage
    )  # TODO setting metadata should validate against the model
    value = 123
    a.set(value)

    assert isinstance(a.typed_snapshot(), ParameterSnapshot)
    assert isinstance(a.typed_metadata(), MyParameterMetadata)

    assert a.typed_metadata().param_type == ParamType.voltage
    assert a.typed_snapshot().value == value
    assert a.typed_snapshot().name == "myparam"

    assert_type(
        a.typed_metadata(), MyParameterMetadata
    )  # TODO this is only checked if the type checker runs agains the test
    assert_type(a.typed_snapshot(), ParameterSnapshot)
