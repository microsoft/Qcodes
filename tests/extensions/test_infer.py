from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qcodes.extensions.infer import (
    InferAttrs,
    InferError,
    _merge_user_and_class_attrs,
    get_parameter_chain,
    get_root_parameter,
    infer_channel,
    infer_instrument,
)
from qcodes.instrument import Instrument, InstrumentBase, InstrumentModule
from qcodes.parameters import DelegateParameter, ManualParameter, Parameter


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


class DummyDelegateInstrument(InstrumentBase):
    def __init__(self, name: str):
        super().__init__(name=name)
        self.inst_delegate = DelegateParameter(
            name="inst_delegate", source=None, instrument=self, bind_to_instrument=True
        )
        self.module = DummyDelegateModule(name="dummy_delegate_module", parent=self)
        self.inst_base_parameter = ManualParameter(
            "inst_base_parameter", instrument=self
        )


class DummyDelegateModule(InstrumentModule):
    def __init__(self, name: str, parent: InstrumentBase):
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
    InferAttrs.clear()
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

    InferAttrs.clear()
    assert get_root_parameter(good_inst_del_3) is good_inst_del_3

    InferAttrs.add("linked_parameter")
    assert get_root_parameter(good_inst_del_3) is inst.good_inst_parameter


def test_get_root_parameter_no_source(good_inst_delegates):
    good_inst_del_1, good_inst_del_2, _ = good_inst_delegates

    good_inst_del_1.source = None

    with pytest.raises(InferError) as exc_info:
        get_root_parameter(good_inst_del_2)
    assert "is not attached to a source" in str(exc_info.value)


def test_get_root_parameter_no_user_attr(good_inst_delegates):
    _, _, good_inst_del_3 = good_inst_delegates
    InferAttrs.clear()
    assert get_root_parameter(good_inst_del_3, "external_parameter") is good_inst_del_3


def test_get_root_parameter_none_user_attr(good_inst_delegates):
    _, _, good_inst_del_3 = good_inst_delegates
    good_inst_del_3.linked_parameter = None
    with pytest.raises(InferError) as exc_info:
        get_root_parameter(good_inst_del_3, "linked_parameter")
    assert "is not attached to a source on attribute" in str(exc_info.value)


def test_infer_instrument_valid(instrument_fixture, good_inst_delegates):
    inst = instrument_fixture
    _, _, good_inst_del_3 = good_inst_delegates
    InferAttrs.add("linked_parameter")
    assert infer_instrument(good_inst_del_3) is inst


def test_infer_instrument_no_instrument(instrument_fixture):
    inst = instrument_fixture
    no_inst_delegate = DelegateParameter(
        "no_inst_delegate", source=inst.bad_inst_parameter
    )
    with pytest.raises(InferError) as exc_info:
        infer_instrument(no_inst_delegate)
    assert "has no instrument" in str(exc_info.value)


def test_infer_instrument_root_instrument_base():
    delegate_inst = DummyDelegateInstrument("dummy_delegate_instrument")

    with pytest.raises(InferError) as exc_info:
        infer_instrument(delegate_inst.inst_base_parameter)
    assert "Could not determine source instrument for parameter" in str(exc_info.value)


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
    with pytest.raises(InferError) as exc_info:
        infer_channel(no_chan_delegate)
    assert "has no instrument" in str(exc_info.value)

    inst_but_not_chan_delegate = DelegateParameter(
        "inst_but_not_chan_delegate", source=inst.good_inst_parameter
    )
    with pytest.raises(InferError) as exc_info:
        infer_channel(inst_but_not_chan_delegate)
    assert "Could not determine a root instrument channel" in str(exc_info.value)


def test_get_parameter_chain(instrument_fixture, good_inst_delegates):
    inst = instrument_fixture
    good_inst_del_1, good_inst_del_2, good_inst_del_3 = good_inst_delegates
    parameter_chain = get_parameter_chain(good_inst_del_3, "linked_parameter")
    expected_chain = (
        good_inst_del_3,
        good_inst_del_2,
        good_inst_del_1,
        inst.good_inst_parameter,
    )
    assert np.all(
        [parameter_chain[i] is param for i, param in enumerate(expected_chain)]
    )

    # This is a broken chain. get_root_parameter would throw an InferError, but
    # get_parameter_chain should run successfully
    good_inst_del_1.source = None
    parameter_chain = get_parameter_chain(good_inst_del_3, "linked_parameter")
    expected_chain = (
        good_inst_del_3,
        good_inst_del_2,
        good_inst_del_1,
    )
    assert np.all(
        [parameter_chain[i] is param for i, param in enumerate(expected_chain)]
    )

    # Make the linked_parameter at the end of the chain
    good_inst_del_3.linked_parameter = None
    good_inst_del_1.source = good_inst_del_3
    parameter_chain = get_parameter_chain(good_inst_del_2, "linked_parameter")
    expected_chain = (
        good_inst_del_2,
        good_inst_del_1,
        good_inst_del_3,
    )
    assert np.all(
        [parameter_chain[i] is param for i, param in enumerate(expected_chain)]
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


def test_merge_user_and_class_attrs():
    InferAttrs.add("attr1")
    attr_set = _merge_user_and_class_attrs("attr2")
    assert set(("attr1", "attr2")) == attr_set

    attr_set_list = _merge_user_and_class_attrs(("attr2", "attr3"))
    assert set(("attr1", "attr2", "attr3")) == attr_set_list


def test_infer_attrs():
    InferAttrs.clear()
    assert InferAttrs.known_attrs() == ()

    InferAttrs.add("attr1")
    assert set(InferAttrs.known_attrs()) == set(("attr1",))

    InferAttrs.add("attr2")
    InferAttrs.discard("attr1")
    assert set(InferAttrs.known_attrs()) == set(("attr2",))

    InferAttrs.add(("attr1", "attr3"))
    assert set(InferAttrs.known_attrs()) == set(("attr1", "attr2", "attr3"))


def test_get_parameter_chain_with_loops(good_inst_delegates):
    good_inst_del_1, good_inst_del_2, good_inst_del_3 = good_inst_delegates
    good_inst_del_1.source = good_inst_del_3
    parameter_chain = get_parameter_chain(good_inst_del_3, "linked_parameter")
    expected_chain = (
        good_inst_del_3,
        good_inst_del_2,
        good_inst_del_1,
        good_inst_del_3,
    )
    assert np.all(
        [parameter_chain[i] is param for i, param in enumerate(expected_chain)]
    )


def test_get_root_parameter_with_loops(good_inst_delegates):
    good_inst_del_1, good_inst_del_2, good_inst_del_3 = good_inst_delegates
    good_inst_del_1.source = good_inst_del_3
    with pytest.raises(InferError) as exc_info:
        get_root_parameter(good_inst_del_2, "linked_parameter")
    assert "generated a loop of linking parameters" in str(exc_info.value)
