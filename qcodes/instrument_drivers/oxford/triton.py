import configparser
import re
from functools import partial
import logging
from traceback import format_exc

from qcodes import IPInstrument
from qcodes.utils.validators import Enum


class Triton(IPInstrument):
    """
    Triton Driver

    Args:
        tmpfile: Expects an exported windows registry file from the registry
            path:
            `[HKEY_CURRENT_USER\Software\Oxford Instruments\Triton System Control\Thermometry]`
            and is used to extract the available temperature channels.


    Status: beta-version.
        TODO:
        fetch registry directly from fridge-computer
    """

    def __init__(self, name, address=None, port=None, terminator='\r\n',
                 tmpfile=None, timeout=20, **kwargs):
        super().__init__(name, address=address, port=port,
                         terminator=terminator, timeout=timeout, **kwargs)

        self._heater_range_auto = False
        self._heater_range_temp = [0.03, 0.1, 0.3, 1, 12, 40]
        self._heater_range_curr = [0.316, 1, 3.16, 10, 31.6, 100]

        self._control_channel = 5

        self.add_parameter(name='time',
                           label='System Time',
                           get_cmd='READ:SYS:TIME',
                           get_parser=self._parse_time)

        self.add_parameter(name='action',
                           label='Current action',
                           get_cmd='READ:SYS:DR:ACTN',
                           get_parser=self._parse_action)

        self.add_parameter(name='status',
                           label='Status',
                           get_cmd='READ:SYS:DR:STATUS',
                           get_parser=self._parse_status)

        self.add_parameter(name='pid_control_channel',
                           label='PID control channel',
                           get_cmd=self._get_control_channel,
                           set_cmd=self._set_control_channel)

        self.add_parameter(name='pid_mode',
                           label='PID Mode',
                           get_cmd=partial(self._get_control_param, 'MODE'),
                           set_cmd=partial(self._set_control_param, 'MODE'),
                           val_mapping={'on':  'ON', 'off': 'OFF'})

        self.add_parameter(name='pid_ramp',
                           label='PID ramp enabled',
                           get_cmd=partial(self._get_control_param,
                                           'RAMP:ENAB'),
                           set_cmd=partial(self._set_control_param,
                                           'RAMP:ENAB'),
                           val_mapping={'on':  'ON', 'off': 'OFF'})

        self.add_parameter(name='pid_setpoint',
                           label='PID temperature setpoint',
                           units='K',
                           get_cmd=partial(self._get_control_param, 'TSET'),
                           set_cmd=partial(self._set_control_param, 'TSET'))

        self.add_parameter(name='pid_rate',
                           label='PID ramp rate',
                           units='K/min',
                           get_cmd=partial(self._get_control_param,
                                           'RAMP:RATE'),
                           set_cmd=partial(self._set_control_param,
                                           'RAMP:RATE'))

        self.add_parameter(name='pid_range',
                           label='PID heater range',
                           # TODO: The units in the software are mA, how to
                           # do this correctly?
                           units='mA',
                           get_cmd=partial(self._get_control_param, 'RANGE'),
                           set_cmd=partial(self._set_control_param, 'RANGE'),
                           vals=Enum(*self._heater_range_curr))

        self.chan_alias = {}
        self.chan_temps = {}
        if tmpfile is not None:
            self._get_temp_channels(tmpfile)
        self._get_pressure_channels()

        try:
            self._get_named_channels()
        except:
            logging.warn('Ignored an error in _get_named_channels\n' +
                         format_exc())

        self.connect_message()

    def _get_response(self, msg):
        return msg.split(':')[-1]

    def _get_response_value(self, msg):
        msg = self._get_response(msg)
        if msg.endswith('NOT_FOUND'):
            return None
        try:
            return float(re.findall("[-+]?\d*\.\d+|\d+", msg)[0])
        except Exception:
            return msg

    def get_idn(self):
        idstr = self.ask('*IDN?')
        idparts = [p.strip() for p in idstr.split(':', 4)][1:]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def _get_control_channel(self, force_get=False):
        if force_get or (not self._control_channel):
            for i in range(20):
                tempval = self.ask('READ:DEV:T%s:TEMP:LOOP:MODE' % (i))
                if not tempval.endswith('NOT_FOUND'):
                    self._control_channel = i

        return self._control_channel

    def _set_control_channel(self, channel):
        self._control_channel = channel
        self.write('SET:DEV:T%s:TEMP:LOOP:HTR:H1' %
                   self._get_control_channel())

    def _get_control_param(self, param):
        chan = self._get_control_channel()
        cmd = 'READ:DEV:T{}:TEMP:LOOP:{}'.format(chan, param)
        return self._get_response_value(self.ask(cmd))

    def _set_control_param(self, param, value):
        chan = self._get_control_channel()
        cmd = 'SET:DEV:T{}:TEMP:LOOP:{}:{}'.format(chan, param, value)
        self.write(cmd)

    def _get_named_channels(self):
        allchans = self.ask('READ:SYS:DR:CHAN')
        allchans = allchans.replace('STAT:SYS:DR:CHAN:', '', 1).split(':')
        for ch in allchans:
            msg = 'READ:SYS:DR:CHAN:%s' % ch
            rep = self.ask(msg)
            if 'INVALID' not in rep:
                alias, chan = rep.split(':')[-2:]
                self.chan_alias[alias] = chan
                self.add_parameter(name=alias,
                                   units='K',
                                   get_cmd='READ:DEV:%s:TEMP:SIG:TEMP' % chan,
                                   get_parser=self._parse_temp)

    def _get_pressure_channels(self):
        for i in range(1, 7):
            chan = 'P%d' % i
            self.add_parameter(name=chan,
                               units='bar',
                               get_cmd='READ:DEV:%s:PRES:SIG:PRES' % chan,
                               get_parser=self._parse_pres)

    def _get_temp_channels(self, file):
        config = configparser.ConfigParser()
        with open(file, 'r') as f:
            next(f)
            config.read_file(f)

        for section in config.sections():
            options = config.options(section)
            namestr = '"m_lpszname"'
            if namestr in options:
                chan = 'T'+section.split('\\')[-1].split('[')[-1]
                name = config.get(section, '"m_lpszname"').strip("\"")
                self.chan_temps[chan] = {'name': name, 'value': None}
                self.add_parameter(name=chan,
                                   units='K',
                                   get_cmd='READ:DEV:%s:TEMP:SIG:TEMP' % chan,
                                   get_parser=self._parse_temp)

    def _parse_action(self, msg):
        action = msg[17:]
        if action == 'PCL':
            action = 'Precooling'
        elif action == 'EPCL':
            action = 'Empty precool loop'
        elif action == 'COND':
            action = 'Condensing'
        elif action == 'NONE':
            if self.MC.get() < 2:
                action = 'Circulating'
            else:
                action = 'Idle'
        elif action == 'COLL':
            action = 'Collecting mixture'
        else:
            action = 'Unknown'
        return action

    def _parse_status(self, msg):
        return msg[19:]

    def _parse_time(self, msg):
        return msg[14:]

    def _parse_temp(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        return float(msg.split('SIG:TEMP:')[-1].strip('K'))

    def _parse_pres(self, msg):
        if 'NOT_FOUND' in msg:
            return None
        return float(msg.split('SIG:PRES:')[-1].strip('mB'))*1e3

    def _recv(self):
        return super()._recv().rstrip()
