from qcodes import VisaInstrument
from qcodes.utils.validators import Numbers, Enum, Ints
from qcodes.utils.deprecate import deprecate_moved_to_qcd


@deprecate_moved_to_qcd(alternative="qcodes_contrib_drivers.drivers.Crycon.crycon_26.Cryocon_26")
class Cryocon_26(VisaInstrument):
    """
    Driver for the Cryo-con Model 26 temperature controller.
    """
    def __init__(self, name, address, terminator='\n', **kwargs):
        super().__init__(name, address, terminator=terminator, **kwargs)

        on_off_map = {True: 'ON', False: 'OFF'}

        for channel in ['A', 'B', 'C', 'D']:
            c = 'ch{}_'.format(channel)

            self.add_parameter(c + 'temperature',
                               get_cmd='input? {}'.format(channel),
                               get_parser=float)

            self.add_parameter(c + 'units',
                               get_cmd='input {}:units?'.format(channel),
                               get_parser=str.upper,
                               set_cmd='input {}:units {{}}'.format(channel),
                               vals=Enum('K', 'C', 'F', 'S'))

            self.add_parameter(c + 'sensor',
                               get_cmd='input {}:sensor?'.format(channel),
                               get_parser=int,
                               set_cmd='input {}:sensor {{}}'.format(channel),
                               vals=Ints(0))

            self.add_parameter(c + 'sensor_power',
                               get_cmd='input {}:sensp {{}}'.format(channel),
                               get_parser=float)

            self.add_parameter(c + 'min',
                               get_cmd='input? {}:min'.format(channel),
                               get_parser=float)

            self.add_parameter(c + 'max',
                               get_cmd='input? {}:max'.format(channel),
                               get_parser=float)

            self.add_parameter(c + 'variance',
                               get_cmd='input? {}:variance'.format(channel),
                               get_parser=float)

            self.add_parameter(c + 'slope',
                               get_cmd='input? {}:slope'.format(channel),
                               get_parser=float)

            self.add_parameter(c + 'offset',
                               get_cmd='input? {}:offset'.format(channel),
                               get_parser=float)

        self.add_function('stop_control_loops', call_cmd='stop')
        self.add_function('start_control_loops', call_cmd='control')

        # TODO: check case of returned strings
        self.add_parameter('control_enabled',
                           get_cmd='control?',
                           val_mapping=on_off_map)

        for loop in [1, 2, 3, 4]:
            l = 'loop{}_'.format(loop)

            self.add_parameter(l + 'source',
                               get_cmd='loop {}:source?'.format(loop),
                               get_parser=str.upper,
                               set_cmd='loop {}:source {{}}'.format(loop),
                               vals=Enum('A', 'B', 'C', 'D'))

            self.add_parameter(l + 'setpoint',
                               get_cmd='loop {}:setpt?'.format(loop),
                               get_parser=float,
                               set_cmd='loop {}:setpt {{}}'.format(loop),
                               vals=Numbers())

            self.add_parameter(l + 'type',
                               get_cmd='loop {}:type?',
                               set_cmd='loop {}:type {{}}'.format(loop),
                               val_mapping={'off': 'OFF',
                                            'manual': 'MAN',
                                            'PID': 'PID',
                                            'table': 'TABLE',
                                            'ramp': 'RAMPP'})

            mapping = {'high': 'HI', 'medium': 'MID', 'low': 'LOW'}

            if loop == 1:
                mapping['100W'] = '100W'

            self.add_parameter(l + 'range',
                               get_cmd='loop {}:range?',
                               set_cmd='loop {}:range {{}}'.format(loop),
                               val_mapping=mapping)

            self.add_parameter(l + 'is_ramping',
                               get_cmd='loop {}:ramp?',
                               val_mapping=on_off_map)

            self.add_parameter(l + 'ramp_rate',
                               get_cmd='loop {}:rate?'.format(loop),
                               get_parser=float,
                               set_cmd='loop {}:rate {{}}'.format(loop),
                               vals=Numbers(0, 100),
                               unit='units/min')

            self.add_parameter(l + 'P',
                               get_cmd='loop {}:pgain?'.format(loop),
                               get_parser=float,
                               set_cmd='loop {}:pgain {{}}'.format(loop),
                               vals=Numbers(0, 1000),
                               unit='-')

            self.add_parameter(l + 'I',
                               get_cmd='loop {}:igain?'.format(loop),
                               get_parser=float,
                               set_cmd='loop {}:igain {{}}'.format(loop),
                               vals=Numbers(0, 1000),
                               unit='s')

            self.add_parameter(l + 'D',
                               get_cmd='loop {}:dgain?'.format(loop),
                               get_parser=float,
                               set_cmd='loop {}:dgain {{}}'.format(loop),
                               vals=Numbers(0, 1000),
                               unit='1/s')

            self.add_parameter(l + 'manual_power',
                               get_cmd='loop {}:pman?',
                               get_parser=float,
                               set_cmd='loop {}:pman {{}}'.format(loop),
                               vals=Numbers(0, 100),
                               unit='%')

            self.add_parameter(l + 'output_power',
                               get_cmd='loop {}:outp?',
                               get_parser=float,
                               unit='%')

            self.add_parameter(l + 'read_heater',
                               get_cmd='loop {}:htrread?',
                               get_parser=float,
                               unit='%')

            self.add_parameter(l + 'max_power',
                               get_cmd='loop {}:maxp?',
                               get_parser=float,
                               set_cmd='loop {}:maxp {{}}'.format(loop),
                               vals=Numbers(0, 100),
                               unit='%')
