from http_instrument import HTTPInstrument
from qcodes.utils.validators import Numbers, Enum, Ints
from functools import partial


class RemoteTriton(HTTPInstrument):

    def __init__(self, name, address, port, timeout=5,
                 terminator='\n', persistent=True, write_confirmation=True,
                 **kwargs):
        super().__init__(name, address=address, port=port,
                         terminator=terminator, timeout=timeout, **kwargs)

        self._heater_range_curr = [0.316, 1, 3.16, 10, 31.6, 100]
        temperatures = ['T' + str(i) for i in range(1, 17)]
        pressures = ['P' + str(i) for i in range(1, 7)]

        self.add_parameter(name='time',
                           label='System Time',
                           get_cmd='time?attribute=value')

        self.add_parameter(name='action',
                           label='Current action',
                           get_cmd='action?attribute=value')

        self.add_parameter(name='status',
                           label='Status',
                           get_cmd='status?attribute=value')

        self.add_parameter(name='pid_control_channel',
                           label='PID control channel',
                           vals=Ints(1,16),
                           get_cmd='pid_control_channel?attribute=value',
                           set_cmd=partial(self.set_with_post, 'pid_control_channel')
                           )

        self.add_parameter(name='pid_mode',
                           label='PID mode',
                           vals=Enum('on', 'off'),
                           get_cmd='pid_mode?attribute=value',
                           set_cmd=partial(self.set_with_post, 'pid_mode')
                           )

        self.add_parameter(name='pid_ramp',
                           label='PID ramp enabled',
                           vals=Enum('on', 'off'),
                           get_cmd='pid_ramp?attribute=value',
                           set_cmd=partial(self.set_with_post, 'pid_mode')
                           )

        self.add_parameter(name='pid_setpoint',
                           label='PID temperature setpoint',
                           unit='K',
                           get_cmd='pid_setpoint?attribute=value',
                           set_cmd=partial(self.set_with_post, 'pid_setpoint'))

        self.add_parameter(name='pid_rate',
                           label='PID ramp rate',
                           unit='K/min',
                           get_cmd='pid_rate?attribute=value',
                           set_cmd=partial(self.set_with_post, 'pid_rate'))

        self.add_parameter(name='pid_range',
                           label='PID heater range',
                           unit='mA',
                           get_cmd='pid_range?attribute=value',
                           set_cmd=partial(self.set_with_post, 'pid_range'),
                           vals=Enum(*self._heater_range_curr))

        for temperature in temperatures:
            self.add_parameter(temperature,
                               label=temperature,
                               unit="K",
                               get_cmd='{}?attribute=value'.format(temperature))

        for pressure in pressures:
            self.add_parameter(pressure,
                               label=pressure,
                               unit="Bar",
                               get_cmd='{}?attribute=value'.format(pressure))

        # TODO Add alias parameters
