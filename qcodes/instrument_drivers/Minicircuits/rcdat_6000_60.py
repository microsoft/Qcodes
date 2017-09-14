
import telnetlib
import qcodes.instrument.base
from qcodes.utils.validators import Numbers, Ints, Enum, MultiType

class RCDAT_6000_60(qcodes.instrument.base.Instrument):

    def __init__(self, name, address, port=23, **kwargs):
        self.telnet = telnetlib.Telnet(address)
        super().__init__(name, **kwargs)
        # do a read until command here
        self.telnet.read_until(b"\n\r", timeout=1)

# General Reference commands

        self.add_parameter(name='attenuation',
                           label='attenuation',
                           unit='dB',
                           get_cmd='SETATT?',
                           set_cmd='SETATT={}',
                           get_parser=float,
                           vals=Numbers(min_value=0, max_value=60))

# Attenuation hopping commands

        self.add_parameter(name='hop_points',
                           label='number of points',
                           unit='#',
                           get_cmd=':HOP:POINTS?',
                           set_cmd=':HOP:POINTS:{}',
                           get_parser=int,
                           vals=Numbers(min_value=1, max_value=100))

        self.add_parameter(name='hop_direction',
                           get_cmd=':HOP:DIRECTION?',
                           set_cmd=':HOP:DIRECTION:{}',
                           val_mapping={
                               'forward': 0,
                               'backward': 1,
                               'bi': 2
                           })

        self.add_parameter(name='hop_index',
                           get_cmd=':HOP:POINT?',
                           set_cmd=':HOP:POINT:{}',
                           get_parser=int,
                           vals=Numbers(min_value=1, max_value=100))

        self.add_parameter(name='hop_time_unit',
                           set_cmd=':HOP:DWELL_UNIT:{}',
                           val_mapping={
                               'micros': 'U',
                               'millis': 'M',
                               's': 'S'
                           })

        self.add_parameter(name='hop_time',
                           unit='s',
                           label='hop time',
                           get_cmd=':HOP:DWELL?',
                           set_cmd=':HOP:DWELL:{}',
                           get_parser=self.string_to_float,
                           vals=Numbers(min_value=5e-3, max_value=1e10))

        self.add_parameter(name='attenuation_hop',
                           get_cmd=':HOP:ATT?',
                           set_cmd=':HOP:ATT:{}',
                           get_parser=float,
                           vals=Numbers(min_value=0, max_value=60))

        self.add_parameter(name='hop_mode',
                           set_cmd=':HOP:MODE:{}',
                           val_mapping={
                               'on': 'ON',
                               'off': 'OFF'
                           })

# Attenuation sweeping commands

        self.add_parameter(name='sweep_direction',
                           get_cmd=':SWEEP:DIRECTION?',
                           set_cmd=':SWEEP:DIRECTION:{}',
                           val_mapping={
                               'forward': 0,
                               'back': 1,
                               'bi': 2
                           })

        self.add_parameter(name='sweep_time_unit',
                           set_cmd=':SWEEP:DWELL_UNIT:{}',
                           val_mapping={
                               'micros': 'U',
                               'millis': 'M',
                               's': 'S'
                           })

        self.add_parameter(name='sweep_time',
                           get_cmd=':SWEEP:DWELL?',
                           set_cmd=':SWEEP:DWELL:{}',
                           unit='s',
                           label='sweep time',
                           get_parser=self.string_to_float,
                           vals=Numbers(min_value=5e-3, max_value=1e10))

        self.add_parameter(name='sweep_start',
                           get_cmd=':SWEEP:START?',
                           set_cmd=':SWEEP:START:{}',
                           get_parser=float,
                           vals=Numbers(min_value=0, max_value=60))

        self.add_parameter(name='sweep_stop',
                           get_cmd=':SWEEP:STOP?',
                           set_cmd=':SWEEP:STOP:{}',
                           get_parser=float,
                           vals=Numbers(min_value=0, max_value=60))

        self.add_parameter(name='sweep_stepsize',
                           get_cmd=':SWEEP:STEPSIZE?',
                           set_cmd=':SWEEP:STEPSIZE:{}',
                           get_parser=float,
                           vals=Numbers(min_value=0.25, max_value=60))

        self.add_parameter(name='sweep_mode',
                           set_cmd=':SWEEP:MODE:{}',
                           val_mapping={
                               'on': 'ON',
                               'off': 'OFF'
                           })
        self.connect_message()

    def string_to_float(self, time_string):
        value, prefix = time_string.split(maxsplit=1)
        if prefix == 'Sec':
            value = float(value)
            return value
        elif prefix == 'mSec':
            value = float(value)*1e-3
            return value
        elif prefix == 'uSec':
            value = float(value)*1e-6
            return value

    def get_idn(self):
        return {'vendor': 'Mini-Circuits',
                'model': self.ask('*MN?'),
                'serial': self.ask('*SN?'),
                'firmware': self.ask('*FIRMWARE?')}

    def ask_raw(self, command):
        command = command + '\n\r'
        self.telnet.write(command.encode('ASCII'))
        data = self.telnet.read_until(b"\n\r", timeout=1).decode('ASCII').strip()
        return data

    def write_raw(self, command):
        command = command + '\n\r'
        self.telnet.write(command.encode('ASCII'))
        data = self.telnet.read_until(b"\n\r", timeout=1).strip()
        if data in [b'1']:
            pass
        elif data in [b'0']:
            raise ValueError('Command failed')
