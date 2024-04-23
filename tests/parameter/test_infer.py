from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qcodes.instrument import Instrument, InstrumentModule
from qcodes.parameters import DelegateParameter, ManualParameter, Parameter
from qcodes.parameters.infer import (
    InferAttrs,
    InferError,
    get_parameter_chain,
    get_root_parameter,
    infer_channel,
    infer_instrument,
)


class DummyModule(InstrumentModule):
    def __init__(self, name: str, parent: Instrument):
        super().__init__(name=name, parent=parent)
        self.good_chan_parameter = ManualParameter(
            "good_chan_parameter", instrument=self
        )
        self.bad_chan_parameter = ManualParameter("bad_chan_parameter")


class DummyInstrument(Instrument):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.good_inst_parameter = ManualParameter(
            "good_inst_parameter", instrument=self
        )
        self.bad_inst_parameter = ManualParameter("bad_inst_parameter")
        self.module = DummyModule(name="module", parent=self)


class DummyDelegateInstrument(Instrument):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.inst_delegate = DelegateParameter(
            name="inst_delegate", source=None, instrument=self, bind_to_instrument=True
        )
        self.module = DummyDelegateModule(name="dummy_delegate_module", parent=self)


class DummyDelegateModule(InstrumentModule):
    def __init__(self, name: str, parent: Instrument):
        super().__init__(name=name, parent=parent)
        self.chan_delegate = DelegateParameter(
            name="chan_delegate", source=None, instrument=self, bind_to_instrument=True
        )


class UserLinkingParameter(Parameter):
    def __init__(
        self, name: str, linked_parameter: Parameter | None = None, **kwargs: Any
    ):
        super().__init__(name=name, **kwargs)
        self.linked_parameter: Parameter | None = linked_parameter


@pytest.fixture(name="instrument_fixture")
def make_instrument_fixture():
    inst = DummyInstrument("dummy_instrument")
    InferAttrs.clear_attrs()
    try:
        yield inst
    finally:
        inst.close()


@pytest.fixture(name="good_inst_delegates")
def make_good_delegate_parameters(instrument_fixture):
    inst = instrument_fixture
    good_inst_del_1 = DelegateParameter(
        "good_inst_del_1", source=inst.good_inst_parameter
    )
    good_inst_del_2 = DelegateParameter("good_inst_del_2", source=good_inst_del_1)
    good_inst_del_3 = UserLinkingParameter(
        "good_inst_del_3", linked_parameter=good_inst_del_2
    )
    return good_inst_del_1, good_inst_del_2, good_inst_del_3


def test_get_root_parameter_valid(instrument_fixture, good_inst_delegates):
    inst = instrument_fixture
    good_inst_del_1, good_inst_del_2, good_inst_del_3 = good_inst_delegates

    assert get_root_parameter(good_inst_del_1) is inst.good_inst_parameter
    assert get_root_parameter(good_inst_del_2) is inst.good_inst_parameter

    assert (
        get_root_parameter(good_inst_del_3, "linked_parameter")
        is inst.good_inst_parameter
    )

    InferAttrs.clear_attrs()
    assert get_root_parameter(good_inst_del_3) is good_inst_del_3

    InferAttrs.add_attr("linked_parameter")
    assert get_root_parameter(good_inst_del_3) is inst.good_inst_parameter


def test_get_root_parameter_no_source(good_inst_delegates):
    good_inst_del_1, good_inst_del_2, _ = good_inst_delegates

    good_inst_del_1.source = None

    with pytest.raises(InferError):
        get_root_parameter(good_inst_del_2)


def test_get_root_parameter_no_user_attr(good_inst_delegates):
    _, _, good_inst_del_3 = good_inst_delegates
    InferAttrs.clear_attrs()
    assert get_root_parameter(good_inst_del_3, "external_parameter") is good_inst_del_3


def test_get_root_parameter_none_user_attr(good_inst_delegates):
    _, _, good_inst_del_3 = good_inst_delegates
    good_inst_del_3.linked_parameter = None
    with pytest.raises(InferError):
        get_root_parameter(good_inst_del_3, "linked_parameter")


def test_infer_instrument_valid(instrument_fixture, good_inst_delegates):
    inst = instrument_fixture
    _, _, good_inst_del_3 = good_inst_delegates
    InferAttrs.add_attr("linked_parameter")
    assert infer_instrument(good_inst_del_3) is inst


def test_infer_instrument_no_instrument(instrument_fixture):
    inst = instrument_fixture
    no_inst_delegate = DelegateParameter(
        "no_inst_delegate", source=inst.bad_inst_parameter
    )
    with pytest.raises(InferError):
        infer_instrument(no_inst_delegate)


def test_infer_channel_valid(instrument_fixture):
    inst = instrument_fixture
    chan_delegate = DelegateParameter(
        "chan_delegate", source=inst.module.good_chan_parameter
    )
    assert infer_channel(chan_delegate) is inst.module


def test_infer_channel_no_channel(instrument_fixture):
    inst = instrument_fixture
    no_chan_delegate = DelegateParameter(
        "no_chan_delegate", source=inst.module.bad_chan_parameter
    )
    with pytest.raises(InferError):
        infer_instrument(no_chan_delegate)


def test_get_parameter_chain(instrument_fixture, good_inst_delegates):
    inst = instrument_fixture
    good_inst_del_1, good_inst_del_2, good_inst_del_3 = good_inst_delegates
    parameter_chain = get_parameter_chain(good_inst_del_3, "linked_parameter")
    assert np.all(
        [
            param in parameter_chain
            for param in (
                inst.good_inst_parameter,
                good_inst_del_1,
                good_inst_del_2,
                good_inst_del_3,
            )
        ]
    )

    # This is a broken chain. get_root_parameter would throw an InferError, but
    # get_parameter_chain should run successfully
    good_inst_del_1.source = None
    parameter_chain = get_parameter_chain(good_inst_del_3, "linked_parameter")
    assert np.all(
        [
            param in parameter_chain
            for param in (
                good_inst_del_1,
                good_inst_del_2,
                good_inst_del_3,
            )
        ]
    )


def test_parameters_on_delegate_instruments(instrument_fixture, good_inst_delegates):
    inst = instrument_fixture
    _, good_inst_del_2, _ = good_inst_delegates

    delegate_inst = DummyDelegateInstrument("dummy_delegate_instrument")
    delegate_inst.inst_delegate.source = good_inst_del_2
    delegate_inst.module.chan_delegate.source = inst.module.good_chan_parameter

    assert infer_channel(delegate_inst.module.chan_delegate) is inst.module
    assert infer_instrument(delegate_inst.module.chan_delegate) is inst
    assert infer_instrument(delegate_inst.inst_delegate) is inst
