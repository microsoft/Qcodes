import time
import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.instrument.mock import MockInstrument, MockModel
from qcodes.utils.validators import Numbers
from qcodes.instrument.parameter import Parameter, ManualParameter


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
            except ValueError:
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


class MockParabola(Instrument):
    '''
    Holds dummy parameters which are get and set able as well as provides
    some basic functions that depends on these parameters for testing
    purposes.

    This instrument is intended to be simpler than the mock model in that it
    does not emulate communications.

    It has 3 main parameters (x, y, z) in order to allow for testing of 3D
    sweeps. The function (parabola with optional noise) is chosen to allow
    testing of numerical optimizations.
    '''

    def __init__(self, name, **kw):
        super().__init__(name, **kw)

        # Instrument parameters
        for parname in ['x', 'y', 'z']:
            self.add_parameter(parname, units='a.u.',
                               parameter_class=ManualParameter,
                               vals=Numbers(), initial_value=0)

        self.add_parameter('noise', units='a.u.',
                           label='white noise amplitude',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=0)

        self.add_parameter('parabola', units='a.u.',
                           get_cmd=self._measure_parabola)
        self.add_parameter('skewed_parabola', units='a.u.',
                           get_cmd=self._measure_skewed_parabola)

    def _measure_parabola(self):
        return (self.x.get()**2 + self.y.get()**2 + self.z.get()**2 +
                self.noise.get()*np.random.rand(1))

    def _measure_skewed_parabola(self):
        '''
        Adds an -x term to add a corelation between the parameters.
        '''
        return ((self.x.get()**2 + self.y.get()**2 +
                self.z.get()**2)*(1 + abs(self.y.get()-self.x.get())) +
                self.noise.get()*np.random.rand(1))


class MockMetaParabola(Instrument):
    '''
    Test for a meta instrument, has a tunable gain knob
    '''
    shared_kwargs = ['mock_parabola_inst']

    def __init__(self, name, mock_parabola_inst, **kw):
        super().__init__(name, **kw)
        self.mock_parabola_inst = mock_parabola_inst

        # Instrument parameters
        for parname in ['x', 'y', 'z']:
            self.add_parameter(parname, units='a.u.',
                               parameter_class=ManualParameter,
                               vals=Numbers(), initial_value=0)
        self.add_parameter('gain', parameter_class=ManualParameter,
                           initial_value=1)

        self.add_parameter('parabola', units='a.u.',
                           get_cmd=self._get_parabola)
        self.add_parameter('skewed_parabola', units='a.u.',
                           get_cmd=self._get_skew_parabola)

    def _get_parabola(self):
        val = self.mock_parabola_inst.parabola.get()
        return val*self.gain.get()

    def _get_skew_parabola(self):
        val = self.mock_parabola_inst.skewed_parabola.get()
        return val*self.gain.get()


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


class MultiGetter(Parameter):
    """
    Test parameters with complicated return values
    instantiate with kwargs:
        MultiGetter(name1=return_val1, name2=return_val2)
    to set the names and (constant) return values of the
    pieces returned. Each return_val can be any array-like
    object
    eg:
        MultiGetter(one=1, onetwo=(1, 2))
    """
    def __init__(self, **kwargs):
        if len(kwargs) == 1:
            name, self._return = list(kwargs.items())[0]
            super().__init__(name=name)
            self.shape = np.shape(self._return)
        else:
            names = tuple(sorted(kwargs.keys()))
            super().__init__(names=names)
            self._return = tuple(kwargs[k] for k in names)
            self.shapes = tuple(np.shape(v) for v in self._return)

    def get(self):
        return self._return
