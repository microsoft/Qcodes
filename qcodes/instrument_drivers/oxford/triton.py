import configparser
import re
from functools import partial
import logging
from traceback import format_exc
from typing import Optional, Any, Union, List, Dict

from qcodes import IPInstrument
from qcodes.utils.validators import Enum, Ints

from time import sleep


class Triton(IPInstrument):
    r"""
    Triton Driver

    Args:
        tmpfile: Optional: an exported windows registry file from the registry
            path:
            `[HKEY_CURRENT_USER\Software\Oxford Instruments\Triton System Control\Thermometry]`
            and is used to extract the available temperature channels.


    Status: beta-version.
        TODO:
        fetch registry directly from fridge-computer
    """

    def __init__(
            self,
            name: str,
            address: Optional[str] = None,
            port: Optional[int] = None,
            terminator: str = '\r\n',
            tmpfile: Optional[str] = None,
            timeout: float = 20,
            **kwargs: Any):
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
                           set_cmd=self._set_control_channel,
                           vals=Ints(1, 16))

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
                           unit='K',
                           get_cmd=partial(self._get_control_param, 'TSET'),
                           set_cmd=partial(self._set_control_param, 'TSET'))

        self.add_parameter(name='pid_rate',
                           label='PID ramp rate',
                           unit='K/min',
                           get_cmd=partial(self._get_control_param,
                                           'RAMP:RATE'),
                           set_cmd=partial(self._set_control_param,
                                           'RAMP:RATE'))

        self.add_parameter(name='pid_range',
                           label='PID heater range',
                           # TODO: The units in the software are mA, how to
                           # do this correctly?
                           unit='mA',
                           get_cmd=partial(self._get_control_param, 'RANGE'),
                           set_cmd=partial(self._set_control_param, 'RANGE'),
                           vals=Enum(*self._heater_range_curr))

        self.add_parameter(name='magnet_status',
                           label='Magnet status',
                           unit='',
                           get_cmd=partial(self._get_control_B_param, 'ACTN'))

        self.add_parameter(name='magnet_sweeprate',
                           label='Magnet sweep rate',
                           unit='T/min',
                           get_cmd=partial(
                               self._get_control_B_param, 'RVST:RATE'),
                           set_cmd=partial(self._set_control_magnet_sweeprate_param))

        self.add_parameter(name='magnet_sweeprate_insta',
                           label='Instantaneous magnet sweep rate',
                           unit='T/min',
                           get_cmd=partial(self._get_control_B_param, 'RFST'))

        self.add_parameter(name='B',
                           label='Magnetic field',
                           unit='T',
                           get_cmd=partial(self._get_control_B_param, 'VECT'))

        self.add_parameter(name='Bx',
                           label='Magnetic field x-component',
                           unit='T',
                           get_cmd=partial(
                               self._get_control_Bcomp_param, 'VECTBx'),
                           set_cmd=partial(self._set_control_Bx_param))

        self.add_parameter(name='By',
                           label='Magnetic field y-component',
                           unit='T',
                           get_cmd=partial(
                               self._get_control_Bcomp_param, 'VECTBy'),
                           set_cmd=partial(self._set_control_By_param))

        self.add_parameter(name='Bz',
                           label='Magnetic field z-component',
                           unit='T',
                           get_cmd=partial(
                               self._get_control_Bcomp_param, 'VECTBz'),
                           set_cmd=partial(self._set_control_Bz_param))

        self.add_parameter(name='magnet_sweep_time',
                           label='Magnet sweep time',
                           unit='T/min',
                           get_cmd=partial(self._get_control_B_param, 'RVST:TIME'))

        self.chan_alias: Dict[str, str] = {}
        self.chan_temp_names: Dict[str, Dict[str, Optional[str]]] = {}
        if tmpfile is not None:
            self._get_temp_channel_names(tmpfile)
        self._get_temp_channels()
        self._get_pressure_channels()

        try:
            self._get_named_channels()
        except:
            logging.warning('Ignored an error in _get_named_channels\n' +
                            format_exc())

        self.connect_message()

    def set_B(self, x: float, y: float, z: float, s: float) -> None:
        if 0 < s <= 0.2:
            self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
                       ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']\r\n')
            self.write('SET:SYS:VRM:ACTN:RTOS\r\n')
            t_wait = self.magnet_sweep_time() * 60 + 10
            print('Please wait ' + str(t_wait) +
                  ' seconds for the field sweep...')
            sleep(t_wait)
        else:
            print('Warning: set magnet sweep rate in range (0 , 0.2] T/min')

    def _get_control_B_param(
            self,
            param: str
    ) -> Optional[Union[float, str, List[float]]]:
        cmd = f'READ:SYS:VRM:{param}'
        return self._get_response_value(self.ask(cmd))

    def _get_control_Bcomp_param(
            self,
            param: str
    ) -> Optional[Union[float, str, List[float]]]:
        cmd = f'READ:SYS:VRM:{param}'
        return self._get_response_value(self.ask(cmd[:-2]) + cmd[-2:])

    def _get_response(self, msg: str) -> str:
        return msg.split(':')[-1]

    def _get_response_value(
            self,
            msg: str
    ) -> Optional[Union[float, str, List[float]]]:
        msg = self._get_response(msg)
        if msg.endswith('NOT_FOUND'):
            return None
        elif msg.endswith('IDLE'):
            return 'IDLE'
        elif msg.endswith('RTOS'):
            return 'RTOS'
        elif msg.endswith('Bx'):
            return float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[0])
        elif msg.endswith('By'):
            return float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[1])
        elif msg.endswith('Bz'):
            return float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[2])
        elif len(re.findall(r"[-+]?\d*\.\d+|\d+", msg)) > 1:
            return [float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[0]), float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[1]), float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[2])]
        try:
            return float(re.findall(r"[-+]?\d*\.\d+|\d+", msg)[0])
        except Exception:
            return msg

    def get_idn(self) -> Dict[str, Optional[str]]:
        """ Return the Instrument Identifier Message """
        idstr = self.ask('*IDN?')
        idparts = [p.strip() for p in idstr.split(':', 4)][1:]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def _get_control_channel(self, force_get: bool = False) -> int:

        # verify current channel
        if self._control_channel and not force_get:
            tempval = self.ask(
                f'READ:DEV:T{self._control_channel}:TEMP:LOOP:MODE')
            if not tempval.endswith('NOT_FOUND'):
                return self._control_channel

        # either _control_channel is not set or wrong
        for i in range(1, 17):
            tempval = self.ask(f'READ:DEV:T{i}:TEMP:LOOP:MODE')
            if not tempval.endswith('NOT_FOUND'):
                self._control_channel = i
                break
        return self._control_channel

    def _set_control_channel(self, channel: int) -> None:
        self._control_channel = channel
        self.write('SET:DEV:T{}:TEMP:LOOP:HTR:H1'.format(
            self._get_control_channel()))

    def _get_control_param(
            self,
            param: str
    ) -> Optional[Union[float, str, List[float]]]:
        chan = self._get_control_channel()
        cmd = f'READ:DEV:T{chan}:TEMP:LOOP:{param}'
        return self._get_response_value(self.ask(cmd))

    def _set_control_param(self, param: str, value: float) -> None:
        chan = self._get_control_channel()
        cmd = f'SET:DEV:T{chan}:TEMP:LOOP:{param}:{value}'
        self.write(cmd)

    def _set_control_magnet_sweeprate_param(self, s: float) -> None:
        if 0 < s <= 0.2:
            x = round(self.Bx(), 4)
            y = round(self.By(), 4)
            z = round(self.Bz(), 4)
            self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
                       ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']\r\n')
        else:
            print(
                'Warning: set sweeprate in range (0 , 0.2] T/min, not setting sweeprate')

    def _set_control_Bx_param(self, x: float) -> None:
        s = self.magnet_sweeprate()
        y = round(self.By(), 4)
        z = round(self.Bz(), 4)
        self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
                   ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']\r\n')
        self.write('SET:SYS:VRM:ACTN:RTOS\r\n')
        # just to give an time estimate, +10s for overhead
        t_wait = self.magnet_sweep_time() * 60 + 10
        print('Please wait ' + str(t_wait) + ' seconds for the field sweep...')
        while self.magnet_status() != 'IDLE':
            pass

    def _set_control_By_param(self, y: float) -> None:
        s = self.magnet_sweeprate()
        x = round(self.Bx(), 4)
        z = round(self.Bz(), 4)
        self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
                   ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']\r\n')
        self.write('SET:SYS:VRM:ACTN:RTOS\r\n')
        # just to give an time estimate, +10s for overhead
        t_wait = self.magnet_sweep_time() * 60 + 10
        print('Please wait ' + str(t_wait) + ' seconds for the field sweep...')
        while self.magnet_status() != 'IDLE':
            pass

    def _set_control_Bz_param(self, z: float) -> None:
        s = self.magnet_sweeprate()
        x = round(self.Bx(), 4)
        y = round(self.By(), 4)
        self.write('SET:SYS:VRM:COO:CART:RVST:MODE:RATE:RATE:' + str(s) +
                   ':VSET:[' + str(x) + ' ' + str(y) + ' ' + str(z) + ']\r\n')
        self.write('SET:SYS:VRM:ACTN:RTOS\r\n')
        # just to give an time estimate, +10s for overhead
        t_wait = self.magnet_sweep_time() * 60 + 10
        print('Please wait ' + str(t_wait) + ' seconds for the field sweep...')
        while self.magnet_status() != 'IDLE':
            pass

    def _get_named_channels(self) -> None:
        allchans_str = self.ask('READ:SYS:DR:CHAN')
        allchans = allchans_str.replace('STAT:SYS:DR:CHAN:', '', 1).split(':')
        for ch in allchans:
            msg = 'READ:SYS:DR:CHAN:%s' % ch
            rep = self.ask(msg)
            if 'INVALID' not in rep and 'NONE' not in rep:
                alias, chan = rep.split(':')[-2:]
                self.chan_alias[alias] = chan
                self.add_parameter(name=alias,
                                   unit='K',
                                   get_cmd='READ:DEV:%s:TEMP:SIG:TEMP' % chan,
                                   get_parser=self._parse_temp)

    def _get_pressure_channels(self) -> None:
        chan_pressure_list = []
        for i in range(1, 7):
            chan = 'P%d' % i
            chan_pressure_list.append(chan)
            self.add_parameter(name=chan,
                               unit='bar',
                               get_cmd='READ:DEV:%s:PRES:SIG:PRES' % chan,
                               get_parser=self._parse_pres)
        self.chan_pressure = set(chan_pressure_list)

    def _get_temp_channel_names(self, file: str) -> None:
        config = configparser.ConfigParser()
        with open(file, encoding='utf16') as f:
            next(f)
            config.read_file(f)

        for section in config.sections():
            options = config.options(section)
            namestr = '"m_lpszname"'
            if namestr in options:
                chan_number = int(section.split('\\')[-1].split('[')[-1]) + 1
                # the names used in the register file are base 0 but the api and the gui
                # uses base one names so add one
                chan = 'T' + str(chan_number)
                name = config.get(section, '"m_lpszname"').strip("\"")
                self.chan_temp_names[chan] = {'name': name, 'value': None}

    def _get_temp_channels(self) -> None:
        chan_temps_list = []
        for i in range(1, 17):
            chan = 'T%d' % i
            chan_temps_list.append(chan)
            self.add_parameter(name=chan,
                               unit='K',
                               get_cmd='READ:DEV:%s:TEMP:SIG:TEMP' % chan,
                               get_parser=self._parse_temp)
        self.chan_temps = set(chan_temps_list)

    def _parse_action(self, msg: str) -> str:
        """ Parse message and return action as a string

        Args:
            msg: message string
        Returns
            action: string describing the action
        """
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

    def _parse_status(self, msg: str) -> str:
        return msg[19:]

    def _parse_time(self, msg: str) -> str:
        return msg[14:]

    def _parse_temp(self, msg: str) -> Optional[float]:
        if 'NOT_FOUND' in msg:
            return None
        return float(msg.split('SIG:TEMP:')[-1].strip('K'))

    def _parse_pres(self, msg: str) -> Optional[float]:
        if 'NOT_FOUND' in msg:
            return None
        return float(msg.split('SIG:PRES:')[-1].strip('mB')) * 1e3

    def _recv(self) -> str:
        return super()._recv().rstrip()
