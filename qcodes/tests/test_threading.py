"""
Test suite for utils.threading.*
"""
import pytest
from typing import Any
import threading

from qcodes.instrument.parameter import Parameter, ParamRawDataType
from qcodes.utils.threading import call_params_threaded

from .instrument_mocks import DummyInstrument


class ParameterWithThreadKnowledge(Parameter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def get_raw(self) -> ParamRawDataType:
        return threading.currentThread().ident


@pytest.fixture(name='dummy_1', scope='function')
def _dummy_dac1():
    instrument = DummyInstrument(
        name='dummy_1', gates=['ch1'])

    instrument.add_parameter(name='voltage_1',
                             parameter_class=ParameterWithThreadKnowledge)

    instrument.add_parameter(name='voltage_2',
                             parameter_class=ParameterWithThreadKnowledge)

    try:
        yield instrument
    finally:
        instrument.close()


@pytest.fixture(name='dummy_2', scope='function')
def _dummy_dac2():
    instrument = DummyInstrument(
        name='dummy_2', gates=['ch1'])

    instrument.add_parameter(name='voltage_1',
                             parameter_class=ParameterWithThreadKnowledge)

    instrument.add_parameter(name='voltage_2',
                             parameter_class=ParameterWithThreadKnowledge)

    try:
        yield instrument
    finally:
        instrument.close()


def test_call_params_threaded(dummy_1, dummy_2):

    params_output = call_params_threaded((dummy_1.voltage_1,
                                          dummy_1.voltage_2,
                                          dummy_2.voltage_1,
                                          dummy_2.voltage_2))

    param1 = params_output[0][0]
    thread_id1 = params_output[0][1]
    for i in range(1, 4):
        param2 = params_output[i][0]
        thread_id2 = params_output[i][1]

        if param1.instrument is param2.instrument:
            assert thread_id1 == thread_id2

        if param1.instrument is not param2.instrument:
            assert thread_id1 != thread_id2

        param1 = param2
        thread_id1 = thread_id2
