import time

from qcodes.instrument.base import Instrument
from qcodes.instrument.mock import MockInstrument, MockModel
from qcodes.utils.validators import Numbers
from qcodes.instrument.parameter import ManualParameter


class AMockModel(MockModel):

    def __init__(self):
        self._memory = {}
        self._reset()
        super().__init__()

    def _reset(self):
        self._gates = [0.0, 0.0, 0.0]
        self._excitation = 0.1

    @staticmethod
    def fmt(value):
        return '{:.3f}'.format(value)

    def gates_set(self, parameter, value):
        if parameter[0] == 'c':
            self._gates[int(parameter[1:])] = float(value)
        elif parameter == 'rst' and value is None:
            # resets gates AND excitation, so we can use gates.reset() to
            # reset the entire model
            self._reset()
        elif parameter[:3] == 'mem':
            slot = int(parameter[3:])
            self._memory[slot] = value
        else:
            raise ValueError

    def gates_get(self, parameter):
        if parameter[0] == 'c':
            return self.fmt(self._gates[int(parameter[1:])])
        elif parameter[:3] == 'mem':
            slot = int(parameter[3:])
            return self._memory[slot]
        else:
            raise ValueError

    def source_set(self, parameter, value):
        if parameter == 'ampl':
            try:
                self._excitation = float(value)
            # TODO(giulioungaretti)  fix bare-except
            except:
                # "Off" as in the MultiType sweep step test
                self._excitation = None
        else:
            raise ValueError(parameter, value)

    def source_get(self, parameter):
        if parameter == 'ampl':
            return self.fmt(self._excitation)
        # put mem here too, just so we can be 100% sure it's going through
        # the model
        elif parameter[:3] == 'mem':
            slot = int(parameter[3:])
            return self._memory[slot]
        else:
            raise ValueError

    def meter_get(self, parameter):
        if parameter == 'ampl':
            gates = self._gates
            # here's my super complex model output!
            return self.fmt(self._excitation *
                            (gates[0] + gates[1]**2 + gates[2]**3))
        elif parameter[:5] == 'echo ':
            return self.fmt(float(parameter[5:]))

    # alias because we need new names when we instantiate an instrument
    # locally at the same time as remotely
    def gateslocal_set(self, parameter, value):
        return self.gates_set(parameter, value)

    def gateslocal_get(self, parameter):
        return self.gates_get(parameter)

    def sourcelocal_set(self, parameter, value):
        return self.source_set(parameter, value)

    def sourcelocal_get(self, parameter):
        return self.source_get(parameter)

    def meterlocal_get(self, parameter):
        return self.meter_get(parameter)


class ParamNoDoc:

    def __init__(self, name, *args, **kwargs):
        self.name = name

    def get_attrs(self):
        return []


class MockInstTester(MockInstrument):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attach_adder()

    def attach_adder(self):
        """
        this function attaches a closure to the object, so can only be
        executed after creating the server because a closure is not
        picklable
        """
        a = 5

        def f(b):
            """
            not the same function as the original method
            """
            return a + b
        self.add5 = f

    def add5(self, b):
        """
        The class copy of this should not get run, because it should
        be overwritten on the server by the closure version.
        """
        raise RuntimeError('dont run this one!')


class MockGates(MockInstTester):

    def __init__(self, name='gates', model=None, **kwargs):
        super().__init__(name, model=model, delay=0.001, **kwargs)

        for i in range(3):
            cmdbase = 'c{}'.format(i)
            self.add_parameter('chan{}'.format(i), get_cmd=cmdbase + '?',
                               set_cmd=cmdbase + ':{:.4f}',
                               get_parser=float,
                               vals=Numbers(-10, 10))
            self.add_parameter('chan{}step'.format(i),
                               get_cmd=cmdbase + '?',
                               set_cmd=cmdbase + ':{:.4f}',
                               get_parser=float,
                               vals=Numbers(-10, 10),
                               step=0.1, delay=0.005)

        self.add_parameter('chan0slow', get_cmd='c0?',
                           set_cmd=self.slow_neg_set, get_parser=float,
                           vals=Numbers(-10, 10), step=0.2,
                           delay=0.02)
        self.add_parameter('chan0slow2', get_cmd='c0?',
                           set_cmd=self.slow_neg_set, get_parser=float,
                           vals=Numbers(-10, 10), step=0.2,
                           delay=0.01, max_delay=0.02)
        self.add_parameter('chan0slow3', get_cmd='c0?',
                           set_cmd=self.slow_neg_set, get_parser=float,
                           vals=Numbers(-10, 10), step=0.2,
                           delay=0.01, max_delay=0.08)
        self.add_parameter('chan0slow4', get_cmd='c0?',
                           set_cmd=self.slow_neg_set, get_parser=float,
                           vals=Numbers(-10, 10),
                           delay=0.01, max_delay=0.02)
        self.add_parameter('chan0slow5', get_cmd='c0?',
                           set_cmd=self.slow_neg_set, get_parser=float,
                           vals=Numbers(-10, 10),
                           delay=0.01, max_delay=0.08)

        self.add_function('reset', call_cmd='rst')

        self.add_parameter('foo', parameter_class=ParamNoDoc)

    def slow_neg_set(self, val):
        if val < 0:
            time.sleep(0.05)
        self.chan0.set(val)


class MockSource(MockInstTester):

    def __init__(self, name='source', model=None, **kwargs):
        super().__init__(name, model=model, delay=0.001, **kwargs)

        self.add_parameter('amplitude', get_cmd='ampl?',
                           set_cmd='ampl:{:.4f}', get_parser=float,
                           vals=Numbers(0, 1),
                           step=0.2, delay=0.005)


class MockMeter(MockInstTester):

    def __init__(self, name='meter', model=None, **kwargs):
        super().__init__(name, model=model, delay=0.001, **kwargs)

        self.add_parameter('amplitude', get_cmd='ampl?', get_parser=float)
        self.add_function('echo', call_cmd='echo {:.2f}?',
                          args=[Numbers(0, 1000)], return_parser=float)


class DummyInstrument(Instrument):

    def __init__(self, name='dummy', gates=['dac1', 'dac2', 'dac3'], **kwargs):
        """
        Create a dummy instrument that can be used for testing

        Args:
            name (string): name for the instrument
            gates (list): list of names that is used to create parameters for
                            the instrument
        """
        super().__init__(name, **kwargs)

        # make gates
        for _, g in enumerate(gates):
            self.add_parameter(g,
                               parameter_class=ManualParameter,
                               initial_value=0,
                               label='Gate {} (arb. units)'.format(g),
                               vals=Numbers(-800, 400))
