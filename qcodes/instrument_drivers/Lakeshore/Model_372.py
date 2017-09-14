from ctypes import LittleEndianStructure, Union, c_uint8
from functools import partial
from pyvisa.resources import SerialInstrument
import pyvisa.constants as vi_const

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Validator, Bool, Enum, Strings, Ints

class SensorStatusBits(LittleEndianStructure):
    """
    Sensor status bitfield
    """
    _fields_ = (
            ("CS_OVL", c_uint8, 1),
            ("VCM_OVL", c_uint8, 1),
            ("VMIX_OVL", c_uint8, 1),
            ("VDIF_OVL", c_uint8, 1),
            ("R_OVER", c_uint8, 1),
            ("R_UNDER", c_uint8, 1),
            ("T_OVER", c_uint8, 1),
            ("T_UNDER", c_uint8, 1)
        )
class SensorStatus(Union):
    """
    Convert returned byte to sensor status bitfield

    Individual error conditions can be checked in the bitfield. e.g.
    ```pyshell
    >>> stat = SensorStatus(1)
    >>> stat.b.CS_OVL
    1
    >>> stat.b.R_OVER
    0
    ```
    """
    _fields_ = (("b", SensorStatusBits),
                ("asbyte", c_uint8))
    def __init__(self, bitfield):
        self.asbyte = int(bitfield)
    def is_error(self):
        """
        Returns true if the sensor is showing an error
        """
        return (self.asbyte != 0)
    def __repr__(self):
        errors = []
        for i, field in enumerate(self.b._fields_):
            if (1 << i) & self.asbyte:
                errors.append(field[0])
        if len(errors) == 0:
            return "<SensorStatus: OK>"
        else:
            return "<SensorStatus: {}>".format("&&".join(errors))

class SensorSettingsValidator(Validator):
    _STAT_VAL = Bool()
    _DWELL_VAL = Ints(1, 200)
    _PAUSE_VAL = Ints(3, 200)
    _CURVE_VAL = Ints(0, 59)
    _TEMP_VAL = Enum('positive', 'negative')

    def __init__(self):
        pass

    def validate(self, val, context=''):
        if not isinstance(val, SensorSettings):
            raise TypeError("expecting a set of sensor settings; {}".format(context))
        """ Validate all parameters """
        self._STAT_VAL.validate(val.enabled)
        self._DWELL_VAL.validate(val.dwell)
        self._PAUSE_VAL.validate(val.pause)
        self._CURVE_VAL.validate(val.curve)
        self._TEMP_VAL.validate(val.tempco)

class SensorSettings(object):
    """
    Expand out the sensor status command `INSET` into it's constituent
    parts, which can then be set and queried
    """
    __slots__ = ('enabled', 'dwell', 'pause', 'curve', 'tempco')
    _TEMPCO_MAP = {'positive': 1, 'negative': 2}
    _INV_TEMPCO_MAP = {v: k for k, v in _TEMPCO_MAP.items()}

    def __init__(self, enabled, dwell, pause, curve, tempco):
        self.enabled = enabled
        self.dwell = dwell
        self.pause = pause
        # For now, this is just the curve ID, but it should be linked to the actual curve object
        # TODO: link to curve object
        self.curve = curve 
        self.tempco = tempco

    @classmethod
    def parse_input(cls, inp, field=None):
        """ Parse the output of the `INSET?` query """
        inset = inp.strip().split(',')
        if field is None:
            return SensorSettings(enabled=bool(int(inset[0])),
                dwell=int(inset[1]),
                pause=int(inset[2]),
                curve=int(inset[3]),
                tempco=cls._INV_TEMPCO_MAP[int(inset[4])])
        elif field == "enabled":
            return bool(inset[0])
        elif field == "dwell":
            return int(inset[1])
        elif field == "pause":
            return int(inset[2])
        elif field == "curve":
            return int(inset[3])
        else:
            return cls._INV_TEMPCO_MAP[int(inset[4])]

    @classmethod
    def parse_output(cls, val, parent=None, field=None):
        """ Parse the output of the `INSET?` query """
        inset = parent.sensor_statset.get()
        setattr(inset, field, val)
        return inset

    @property
    def set_format(self):
        return "{},{},{},{},{}".format(
            int(self.enabled),
            self.dwell,
            self.pause,
            self.curve,
            self._TEMPCO_MAP[self.tempco])

    def __repr__(self):
        return "SensorSettings(enabled={}, dwell={}, pause={}, curve={}, tempco={})".format(
            self.enabled, self.dwell, self.pause, self.curve, self.tempco)


class SensorChannel(InstrumentChannel):
    """
    A single sensor channel of a temperature controller
    """

    _CHANNEL_VAL = Ints(1, 16)

    def __init__(self, parent, name, channel):
        super().__init__(parent, name)


        # Validate the channel value
        self._CHANNEL_VAL.validate(channel)
        self._channel = channel  # Channel on the temperature controller. Can be 1-16

        # Add the various channel parameters
        self.add_parameter('temperature', get_cmd='KRDG? {}'.format(self._channel),
            get_parser=float, label='Temerature', unit='K')
        self.add_parameter('sensor_raw', get_cmd='SRDG? {}'.format(self._channel),
            get_parser=float, label='Raw Sensor Reading', unit='Ohms')
        self.add_parameter('sensor_status', get_cmd='RDGST? {}'.format(self._channel),
            label='Sensor Status', get_parser=SensorStatus)
        self.add_parameter('sensor_name', get_cmd='INNAME? {}'.format(self._channel),
            get_parser=str, set_cmd='INNAME {},\"{{}}\"'.format(self._channel), vals=Strings(15),
            label='Sensor Name')

        self.add_parameter('sensor_statset', get_cmd='INSET? {}'.format(self._channel),
            get_parser=SensorSettings.parse_input, set_cmd='INSET {},{{0.set_format}}'.format(self._channel),
            vals=SensorSettingsValidator(), label='Sensor Settings')
        status_parameters = ('enabled', 'dwell', 'pause', 'curve', 'tempco')
        for param in status_parameters:
            self.add_parameter('sensor_{}'.format(param), get_cmd='INSET? {}'.format(self._channel),
                get_parser=partial(SensorSettings.parse_input, field=param), 
                set_cmd='INSET {},{{0.set_format}}'.format(self._channel), 
                set_parser=partial(SensorSettings.parse_output, parent=self, field=param),
                vals=Bool(), label='Sensor {0.capitalize}'.format(param))



class Model_372(VisaInstrument):
    """
    Lakeshore Model 336 Temperature Controller Driver
    Controlled via sockets
    """

    _OFFON_MAP = {'off': 0, 'on': 1}
    _INV_OFFON_MAP = {v: k for k, v in _OFFON_MAP.items()}

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator="\r\n", **kwargs)
        if isinstance(self.visa_handle, SerialInstrument):
            # Set up serial connection parameters
            self.visa_handle.baud_rate = 57600
            self.visa_handle.data_bits = 7
            self.visa_handle.stop_bits = vi_const.StopBits.one
            self.visa_handle.flow_control = vi_const.VI_ASRL_FLOW_NONE
            self.visa_handle.parity = vi_const.Parity.odd

        # Allow access to channels either by referring to the channel name
        # or through a channel list.
        # i.e. Model_336.A.temperature() and Model_336.channels[0].temperature()
        # refer to the same parameter.
        channels = ChannelList(self, "TempSensors", SensorChannel, snapshotable=False)
        for chan_name in range(1, 17):
            channel = SensorChannel(self, 'Chan{}'.format(chan_name), chan_name)
            channels.append(channel)
            self.add_submodule('chan{}'.format(chan_name), channel)
        channels.lock()
        self.add_submodule("channels", channels)

        self.add_parameter('active_channel', get_cmd='SCAN?', set_cmd='SCAN {}',
            get_parser=partial(self._parse_scan, "CHAN"), 
            set_parser=partial(self._set_scan, "CHAN"),
            vals=self.channels.get_validator())
        self.add_parameter('scan', get_cmd='SCAN?', set_cmd='SCAN {}',
            get_parser=partial(self._parse_scan, "SCAN"),
            set_parser=partial(self._set_scan, "SCAN"),
            vals=Enum('off', 'on'))

        self.connect_message()

    def _parse_scan(self, param, inp):
        inp = inp.strip().split(',')
        channel = self.channels[int(inp[0])-1]
        scan = self._INV_OFFON_MAP[int(inp[1])]
        if param == "CHAN":
            return channel
        else:
            return scan

    def _set_scan(self, param, val):
        if param =="CHAN":
            channel = val._channel
            scan = self._OFFON_MAP[self.scan.get()]
        else:
            channel = self.active_channel.get()._channel
            scan = self._OFFON_MAP[val]
        return "{},{}".format(channel, scan)