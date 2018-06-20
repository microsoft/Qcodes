import numpy as np

from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Numbers
from qcodes.instrument.parameter import MultiParameter, Parameter, ArrayParameter
from qcodes.instrument.channel import InstrumentChannel, ChannelList

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
                               parameter_class=Parameter,
                               vals=Numbers(), initial_value=0,
                               get_cmd=None, set_cmd=None)

        self.add_parameter('noise', unit='a.u.',
                           label='white noise amplitude',
                           parameter_class=Parameter,
                           vals=Numbers(), initial_value=0,
                           get_cmd=None, set_cmd=None)

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

    def __init__(self, name, mock_parabola_inst, **kw):
        super().__init__(name, **kw)
        self.mock_parabola_inst = mock_parabola_inst

        # Instrument parameters
        for parname in ['x', 'y', 'z']:
            self.add_parameter(parname, unit='a.u.',
                               parameter_class=Parameter,
                               vals=Numbers(), initial_value=0,
                               get_cmd=None, set_cmd=None)
        self.add_parameter('gain', parameter_class=Parameter,
                           initial_value=1,
                           get_cmd=None, set_cmd=None)

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
                               parameter_class=Parameter,
                               initial_value=0,
                               label='Gate {}'.format(g),
                               unit="V",
                               vals=Numbers(-800, 400),
                               get_cmd=None, set_cmd=None)

class DummyChannel(InstrumentChannel):
    """
    A single dummy channel implementation
    """

    def __init__(self, parent, name, channel):
        super().__init__(parent, name)

        self._channel = channel

        # Add the various channel parameters
        self.add_parameter('temperature',
                           parameter_class=Parameter,
                           initial_value=0,
                           label="Temperature_{}".format(channel),
                           unit='K',
                           vals=Numbers(0, 300),
                           get_cmd=None, set_cmd=None)

        self.add_parameter(name='dummy_multi_parameter',
                           parameter_class=MultiSetPointParam)

        self.add_parameter(name='dummy_array_parameter',
                           parameter_class=ArraySetPointParam)

class DummyChannelInstrument(Instrument):
    """
    Dummy instrument with channels
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        channels = ChannelList(self, "TempSensors", DummyChannel, snapshotable=False)
        for chan_name in ('A', 'B', 'C', 'D', 'E', 'F'):
            channel = DummyChannel(self, 'Chan{}'.format(chan_name), chan_name)
            channels.append(channel)
            self.add_submodule(chan_name, channel)
        self.add_submodule("channels", channels)

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

    def get_raw(self):
        return self._return


class MultiSetPointParam(MultiParameter):
    """
    Multiparameter which only purpose it to test that units, setpoints
    and so on are copied correctly to the individual arrays in the datarray.
    """
    def __init__(self, instrument=None, name='testparameter'):
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
                         instrument=instrument,
                         labels=labels,
                         units=units,
                         setpoints=setpoints,
                         setpoint_labels=setpoint_labels,
                         setpoint_names=setpoint_names,
                         setpoint_units=setpoint_units)

    def get_raw(self):
        items = (np.zeros(5), np.ones(5))
        self._save_val(items)
        return items

class ArraySetPointParam(ArrayParameter):
    """
    Arrayparameter which only purpose it to test that units, setpoints
    and so on are copied correctly to the individual arrays in the datarray.
    """

    def __init__(self, instrument=None, name='testparameter'):
        shape = (5,)
        label = 'this label'
        unit = 'this unit'
        sp_base = tuple(np.linspace(5, 9, 5))
        setpoints = (sp_base,)
        setpoint_names = ('this_setpoint',)
        setpoint_labels = ('this setpoint',)
        setpoint_units = ('this setpointunit',)
        super().__init__(name,
                         shape,
                         instrument,
                         label=label,
                         unit=unit,
                         setpoints=setpoints,
                         setpoint_labels=setpoint_labels,
                         setpoint_names=setpoint_names,
                         setpoint_units=setpoint_units)

    def get_raw(self):
        item = np.ones(5) + 1
        self._save_val(item)
        return item
