
from functools import partial
import logging
from typing import Sequence, Dict, Optional

import numpy as np

from qcodes.instrument.base import Instrument, InstrumentBase
from qcodes.utils.validators import Numbers, Arrays, Strings, ComplexNumbers
from qcodes.instrument.parameter import MultiParameter, Parameter, \
    ArrayParameter, ParameterWithSetpoints
from qcodes.instrument.channel import InstrumentChannel, ChannelList
import random

log = logging.getLogger(__name__)


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


class MockMetaParabola(InstrumentBase):
    '''
    Test for a meta instrument, has a tunable gain knob

    Unlike a MockParabola, a MockMetaParabola does not have a connection, or
    access to ask_raw/write_raw, i.e. it would not be connected to a real instrument.

    It is also not tracked in the global _all_instruments list, but is still
    snapshottable in a station.
    '''

    def __init__(self, name, mock_parabola_inst, **kw):
        """
        Create a new MockMetaParabola, connected to an existing MockParabola instance.
        """
        super().__init__(name, **kw)
        self.mock_parabola_inst = mock_parabola_inst

        # Instrument parameters
        for parname in ['x', 'y', 'z']:
            self.parameters[parname] = getattr(mock_parabola_inst, parname)
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

    def __init__(self, name: str = 'dummy',
                 gates: Sequence = ('dac1', 'dac2', 'dac3'), **kwargs):

        """
        Create a dummy instrument that can be used for testing

        Args:
            name: name for the instrument
            gates: list of names that is used to create parameters for
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

        self.add_parameter(name='dummy_scalar_multi_parameter',
                           parameter_class=MultiScalarParam)

        self.add_parameter(name='dummy_2d_multi_parameter',
                           parameter_class=Multi2DSetPointParam)

        self.add_parameter(name='dummy_array_parameter',
                           parameter_class=ArraySetPointParam)

        self.add_parameter(name='dummy_complex_array_parameter',
                           parameter_class=ComplexArraySetPointParam)

        self.add_parameter('dummy_start',
                           initial_value=0,
                           unit='some unit',
                           label='f start',
                           vals=Numbers(0, 1e3),
                           get_cmd=None,
                           set_cmd=None)

        self.add_parameter('dummy_stop',
                           unit='some unit',
                           label='f stop',
                           vals=Numbers(1, 1e3),
                           get_cmd=None,
                           set_cmd=None)

        self.add_parameter('dummy_n_points',
                           unit='',
                           vals=Numbers(1, 1e3),
                           get_cmd=None,
                           set_cmd=None)

        self.add_parameter('dummy_sp_axis',
                           unit='some unit',
                           label='Dummy sp axis',
                           parameter_class=GeneratedSetPoints,
                           startparam=self.dummy_start,
                           stopparam=self.dummy_stop,
                           numpointsparam=self.dummy_n_points,
                           vals=Arrays(shape=(self.dummy_n_points,)))

        self.add_parameter(name='dummy_parameter_with_setpoints',
                           label='Dummy Parameter with Setpoints',
                           unit='some other unit',
                           setpoints=(self.dummy_sp_axis,),
                           vals=Arrays(shape=(self.dummy_n_points,)),
                           parameter_class=DummyParameterWithSetpoints1D)

        self.add_parameter(name='dummy_text',
                           label='Dummy text',
                           unit='text unit',
                           initial_value='thisisastring',
                           set_cmd=None,
                           vals=Strings())

        self.add_parameter(name='dummy_complex',
                           label='Dummy complex',
                           unit='complex unit',
                           initial_value=1+1j,
                           set_cmd=None,
                           vals=ComplexNumbers())

        self.add_parameter(name='dummy_parameter_with_setpoints_complex',
                           label='Dummy Parameter with Setpoints complex',
                           unit='some other unit',
                           setpoints=(self.dummy_sp_axis,),
                           vals=Arrays(shape=(self.dummy_n_points,),
                                       valid_types=(np.complexfloating,)),
                           parameter_class=DummyParameterWithSetpointsComplex)

        self.add_function(name='log_my_name',
                          call_cmd=partial(log.debug, f'{name}'))


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


class Multi2DSetPointParam(MultiParameter):
    """
    Multiparameter which only purpose it to test that units, setpoints
    and so on are copied correctly to the individual arrays in the datarray.
    """

    def __init__(self, instrument=None, name='testparameter'):
        shapes = ((5, 3), (5, 3))
        names = ('this', 'that')
        labels = ('this label', 'that label')
        units = ('this unit', 'that unit')
        sp_base_1 = tuple(np.linspace(5, 9, 5))
        sp_base_2 = tuple(np.linspace(9, 11, 3))
        array_setpoints = setpoint_generator(sp_base_1, sp_base_2)
        setpoints = (array_setpoints, array_setpoints)
        setpoint_names = (('this_setpoint', 'that_setpoint'),
                          ('this_setpoint', 'that_setpoint'))
        setpoint_labels = (('this setpoint', 'that setpoint'),
                           ('this setpoint', 'that setpoint'))
        setpoint_units = (('this setpointunit',
                           'that setpointunit'),
                          ('this setpointunit',
                           'that setpointunit'))
        super().__init__(name, names, shapes,
                         instrument=instrument,
                         labels=labels,
                         units=units,
                         setpoints=setpoints,
                         setpoint_labels=setpoint_labels,
                         setpoint_names=setpoint_names,
                         setpoint_units=setpoint_units)

    def get_raw(self):
        items = (np.zeros((5, 3)), np.ones((5, 3)))
        self._save_val(items)
        return items


class MultiScalarParam(MultiParameter):
    """
    Multiparameter whos elements are scalars i.e. similar to
    Parameter with no setpoints etc.
    """
    def __init__(self, instrument=None, name='multiscalarparameter'):
        shapes = ((), ())
        names = ('thisparam', 'thatparam')
        labels = ('thisparam label', 'thatparam label')
        units = ('thisparam unit', 'thatparam unit')
        setpoints = ((), ())
        super().__init__(name, names, shapes,
                         instrument=instrument,
                         labels=labels,
                         units=units,
                         setpoints=setpoints)

    def get_raw(self):
        items = (0, 1)
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



class ComplexArraySetPointParam(ArrayParameter):
    """
    Arrayparameter that returns complex numbers
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
        item = np.arange(5) - 1j*np.arange(5)
        self._save_val(item)
        return item


class GeneratedSetPoints(Parameter):
    """
    A parameter that generates a setpoint array from start, stop and num points
    parameters.
    """
    def __init__(self, startparam, stopparam, numpointsparam, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._startparam = startparam
        self._stopparam = stopparam
        self._numpointsparam = numpointsparam

    def get_raw(self):
        return np.linspace(self._startparam(), self._stopparam(),
                              self._numpointsparam())


class DummyParameterWithSetpoints1D(ParameterWithSetpoints):
    """
    Dummy parameter that returns data with a shape based on the
    `dummy_n_points` parameter in the instrument.
    """

    def get_raw(self):
        npoints = self.instrument.dummy_n_points()
        return np.random.rand(npoints)


class DummyParameterWithSetpointsComplex(ParameterWithSetpoints):
    """
    Dummy parameter that returns data with a shape based on the
    `dummy_n_points` parameter in the instrument. Returns Complex values
    """

    def get_raw(self):
        npoints = self.instrument.dummy_n_points()
        return np.random.rand(npoints) + 1j*np.random.rand(npoints)


def setpoint_generator(*sp_bases):
    """
    Helper function to generate setpoints in the format that ArrayParameter
    (and MultiParameter) expects

    Args:
        *sp_bases:

    Returns:

    """
    setpoints = []
    for i, sp_base in enumerate(sp_bases):
        if i == 0:
            setpoints.append(sp_base)
        else:
            repeats = [len(sp) for sp in sp_bases[:i]]
            repeats.append(1)
            setpoints.append(np.tile(sp_base, repeats))

    return tuple(setpoints)


class SnapShotTestInstrument(Instrument):
    """
    A highly specialized dummy instrument for testing the snapshot. Used by
    test_snapshot.py

    Args:
        name: name for the instrument
        params: parameter names. The instrument will have these as parameters
        params_to_skip: parameters to skip updating in the snapshot. Must be
            a subset of params
    """

    def __init__(self, name: str, params: Sequence[str] = ('v1', 'v2', 'v3'),
                 params_to_skip: Sequence[str] = ('v2')):

        super().__init__(name)

        if not(set(params_to_skip).issubset(params)):
            raise ValueError('Invalid input; params_to_skip must be a subset '
                             'of params')

        self._params_to_skip = params_to_skip
        self._params = params

        # dict to keep track of how many time 'get' has been called on each
        # parameter. Useful for testing params_to_skip_update in the snapshot
        self._get_calls = {p: 0 for p in params}

        for p_name in params:

            self.add_parameter(p_name, label=f'{name} Label', unit='V',
                               set_cmd=None,
                               get_cmd=partial(self._getter, p_name))

    def _getter(self, name: str):
        val = self.parameters[name]._latest['value']
        self._get_calls[name] += 1
        return val

    def snapshot_base(self, update: bool = True,
                      params_to_skip_update: Optional[Sequence[str]] = None
                      ) -> Dict:
        if params_to_skip_update is None:
            params_to_skip_update = self._params_to_skip
        snap = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update)
        return snap
