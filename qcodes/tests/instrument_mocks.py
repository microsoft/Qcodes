import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Numbers
from qcodes.instrument.parameter import MultiParameter, ManualParameter


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
            self.add_parameter(parname, unit='a.u.',
                               parameter_class=ManualParameter,
                               vals=Numbers(), initial_value=0)

        self.add_parameter('noise', unit='a.u.',
                           label='white noise amplitude',
                           parameter_class=ManualParameter,
                           vals=Numbers(), initial_value=0)

        self.add_parameter('parabola', unit='a.u.',
                           get_cmd=self._measure_parabola)
        self.add_parameter('skewed_parabola', unit='a.u.',
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
            self.add_parameter(parname, unit='a.u.',
                               parameter_class=ManualParameter,
                               vals=Numbers(), initial_value=0)
        self.add_parameter('gain', parameter_class=ManualParameter,
                           initial_value=1)

        self.add_parameter('parabola', unit='a.u.',
                           get_cmd=self._get_parabola)
        self.add_parameter('skewed_parabola', unit='a.u.',
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
                               label='Gate {}'.format(g),
                               unit="V",
                               vals=Numbers(-800, 400))


class MultiGetter(MultiParameter):
    """
    Test parameters with complicated return values
    instantiate with kwargs::

        MultiGetter(name1=return_val1, name2=return_val2)

    to set the names and (constant) return values of the
    pieces returned. Each return_val can be any array-like
    object
    eg::

        MultiGetter(one=1, onetwo=(1, 2))

    """
    def __init__(self, **kwargs):
        names = tuple(sorted(kwargs.keys()))
        self._return = tuple(kwargs[k] for k in names)
        shapes = tuple(np.shape(v) for v in self._return)
        super().__init__(name='multigetter', names=names, shapes=shapes)

    def get(self):
        return self._return


class MultiSetPointParam(MultiParameter):
    """
    Multiparameter which only purpose it to test that units, setpoints
    and so on are copied correctly to the individual arrays in the datarray.
    """
    def __init__(self):
        name = 'testparameter'
        shapes = ((5,), (5,))
        names = ('this', 'that')
        labels = ('this label', 'that label')
        units = ('this unit', 'that unit')
        sp_base = tuple(np.linspace(5, 9, 5))
        setpoints = ((sp_base,), (sp_base,))
        setpoint_names = (('this_setpoint',), ('this_setpoint',))
        setpoint_labels = (('this setpoint',), ('this setpoint',))
        setpoint_units = (('this setpointunit',), ('this setpointunit',))
        super().__init__(name, names, shapes,
                         labels=labels,
                         units=units,
                         setpoints=setpoints,
                         setpoint_labels=setpoint_labels,
                         setpoint_names=setpoint_names,
                         setpoint_units=setpoint_units)

    def get(self):
        return np.zeros(5), np.ones(5)