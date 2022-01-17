import pytest

from qcodes.instrument.parameter import Parameter
import qcodes.utils.validators as vals
from qcodes.utils.helpers import create_on_off_val_mapping
from qcodes.tests.instrument_mocks import DummyInstrument
from .conftest import ParameterMemory


@pytest.fixture(name="dummyinst")
def _make_dummy_inst():
    inst = DummyInstrument('dummy_holder')
    try:
        yield inst
    finally:
        inst.close()


def test_val_mapping_basic():
    # We store value external to cache
    # to allow testing of interaction with cache
    mem = ParameterMemory()

    p = Parameter('p', set_cmd=mem.set, get_cmd=mem.get,
                  val_mapping={'off': 0, 'on': 1},
                  vals=vals.Enum('off', 'on'))

    p('off')
    assert p.cache.raw_value == 0
    assert mem.get() == 0
    assert p() == 'off'

    mem.set(1)
    assert p() == 'on'

    # implicit mapping to ints
    mem.set('0')
    assert p() == 'off'

    # unrecognized response
    mem.set(2)
    with pytest.raises(KeyError):
        p()

    mem.set(1)

    p.cache.set('off')
    assert p.get_latest() == 'off'
    # Nothing has been passed to the "instrument" at ``cache.set``
    # call, hence the following assertions should hold
    assert mem.get() == 1
    assert p() == 'on'
    assert p.get_latest() == 'on'


def test_val_mapping_with_parsers():
    # We store value external to cache
    # to allow testing of interaction with cache
    mem = ParameterMemory()

    # # set_parser with val_mapping
    # Parameter('p', set_cmd=mem.set, get_cmd=mem.get,
    #           val_mapping={'off': 0, 'on': 1},
    #           set_parser=mem.parse_set_p)

    # get_parser with val_mapping
    p = Parameter('p', set_cmd=mem.set_p_prefixed,
                  get_cmd=mem.get, get_parser=mem.strip_prefix,
                  val_mapping={'off': 0, 'on': 1},
                  vals=vals.Enum('off', 'on'))

    p('off')
    assert mem.get() == 'PVAL: 0'
    # this is slight strange. Since it uses a custom set_cmd
    # rather than a set_parser the raw_value does not match
    # what is actually sent to the instrument
    assert p.cache.raw_value == 0
    assert p() == 'off'

    mem.set('PVAL: 1')
    assert p() == 'on'

    p.cache.set('off')
    assert p.get_latest() == 'off'
    # Nothing has been passed to the "instrument" at ``cache.set``
    # call, hence the following assertions should hold
    assert mem.get() == 'PVAL: 1'
    assert p() == 'on'
    assert p.get_latest() == 'on'
    assert p.cache.get() == 'on'


def test_on_off_val_mapping():
    instrument_value_for_on = 'on_'
    instrument_value_for_off = 'off_'

    parameter_return_value_for_on = True
    parameter_return_value_for_off = False

    mem = ParameterMemory()

    p = Parameter('p', set_cmd=mem.set, get_cmd=mem.get,
                  val_mapping=create_on_off_val_mapping(
                      on_val=instrument_value_for_on,
                      off_val=instrument_value_for_off))

    test_data = [(instrument_value_for_on,
                  parameter_return_value_for_on,
                  ('On', 'on', 'ON', 1, True)),
                 (instrument_value_for_off,
                  parameter_return_value_for_off,
                  ('Off', 'off', 'OFF', 0, False))]

    for instr_value, parameter_return_value, inputs in test_data:
        for inp in inputs:
            # Setting parameter with any of the `inputs` is allowed
            p(inp)
            # For any value from the `inputs`, what gets send to the
            # instrument is on_val/off_val which are specified in
            # `create_on_off_val_mapping`
            assert mem.get() == instr_value
            # When getting a value of the parameter, only specific
            # values are returned instead of `inputs`
            assert p() == parameter_return_value


def test_val_mapping_on_instrument(dummyinst):

    dummyinst.add_parameter('myparameter', set_cmd=None, get_cmd=None,
                            val_mapping={'A': 0, 'B': 1})
    dummyinst.myparameter('A')
    assert dummyinst.myparameter() == 'A'
    assert dummyinst.myparameter() == 'A'
    assert dummyinst.myparameter.raw_value == 0
