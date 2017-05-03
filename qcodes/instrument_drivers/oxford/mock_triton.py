import random

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.validators import Numbers, Enum, Ints

class MockNonSetableParameter(ManualParameter):

    def __init__(self, name, min, max, instrument=None, initial_value=None, **kwargs):
        super().__init__(name=name, instrument=instrument, initial_value=initial_value, **kwargs)
        self._min_val = min or 0.0
        self._max_val = max or 1.0
        self.has_set = False

    def set(self):
        raise NotImplementedError("This has no set")

    def get(self):
        """ Return latest value"""
        value = random.uniform(self._min_val, self._max_val)
        self.validate(value)
        self._save_val(value)
        return self._latest()['value']


class MockStatusParameter(ManualParameter):

    def __init__(self, name, values, instrument=None, initial_value=None, **kwargs):
        super().__init__(name=name, instrument=instrument, initial_value=initial_value, **kwargs)
        self._num_options = len(values)
        self._options = values
        self.has_set = False

    def set(self):
        raise NotImplementedError("This has no set")

    def get(self):
        """ Return latest value"""
        value = self._options[random.randint(0, self._num_options-1)]
        self.validate(value)
        self._save_val(value)
        return self._latest()['value']


class MockTriton(Instrument):

    def __init__(self, name='dummy', **kwargs):

        super().__init__(name, **kwargs)
        self.chan_temp_names = {}
        self._heater_range_curr = [0.316, 1, 3.16, 10, 31.6, 100]
        temperatures = ['T' + str(i) for i in range(1, 17)]
        pressures = ['P' + str(i) for i in range(1, 7)]

        for temperature in temperatures:
            self.add_parameter(temperature,
                               parameter_class=MockNonSetableParameter,
                               initial_value=0,
                               label=temperature,
                               unit="K",
                               min=0.0,
                               max=200,
                               vals=Numbers(0., 200.))

        for pressure in pressures:
            self.add_parameter(pressure,
                               parameter_class=MockNonSetableParameter,
                               initial_value=0,
                               label=pressure,
                               unit="Bar",
                               min=0.0,
                               max=1.0,
                               vals=Numbers(0., 1.))

        self._possible_actions = ('Precooling', 'Empty precool loop', 'Condensing',
                                  'Circulating', 'Idle', 'Collecting mixture', 'Unknown')

        self.add_parameter(name='action',
                           label='Current action',
                           parameter_class=MockStatusParameter,
                           values=self._possible_actions,
                           vals=Enum(*self._possible_actions))

        self._possible_status = ('Ok', 'Error')

        self.add_parameter(name='status',
                           label='Status',
                           parameter_class=MockStatusParameter,
                           values=self._possible_status,
                           vals=Enum(*self._possible_status))

        self.add_parameter(name='pid_control_channel',
                           label='PID control channel',
                           parameter_class=ManualParameter,
                           vals=Ints(1,16),
                           initial_value=1
                           )


        self.add_parameter(name='pid_mode',
                           parameter_class=ManualParameter,
                           label='PID Mode',
                           initial_value='off',
                           vals=Enum('on', 'off'))

        self.add_parameter(name='pid_ramp',
                           parameter_class=ManualParameter,
                           label='PID Mode',
                           initial_value='off',
                           vals=Enum('on', 'off'))

        self.add_parameter(name='pid_setpoint',
                           parameter_class=ManualParameter,
                           label='PID temperature setpoint',
                           unit='K',
                           initial_value=0.0,
                           vals=Numbers(0., 10.))

        self.add_parameter(name='pid_rate',
                           parameter_class=ManualParameter,
                           label='PID ramp rate',
                           unit='K/min',
                           initial_value=0.0,
                           vals=Numbers(0., 10.))

        self.add_parameter(name='pid_range',
                           label='PID heater range',
                           unit='mA',
                           parameter_class=ManualParameter,
                           initial_value=0.316,
                           vals=Enum(*self._heater_range_curr))