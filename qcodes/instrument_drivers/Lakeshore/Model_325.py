import numpy as np

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Enum, Numbers
from qcodes.instrument.group_parameter import GroupParameter, Group


class Model_325_Channel(InstrumentChannel):
    """
    A single sensor channel of a temperature controller
    """

    sensor_status_codes = {
        0:  "OK",
        1:  "invalid reading",
        16:  "temp underrange",
        32:  "temp overrange",
        64:  "sensor units zero",
        128: "sensor units overrang"
    }

    def __init__(self, parent: 'Model_325', name: str, channel: str) ->None:
        super().__init__(parent, name)
        self._channel = channel

        self.add_parameter(
            'temperature',
            get_cmd='KRDG? {}'.format(self._channel),
            get_parser=float,
            label='Temerature',
            unit='K'
        )

        self.add_parameter(
            'status',
            get_cmd='RDGST? {}'.format(self._channel),
            get_parser=self.decode_sensor_status,
            label='Sensor_Status'
        )

        self.add_parameter(
            "type",
            val_mapping={
                "Silicon diode": 0,
                "GaAlAs diode": 1,
                "100 Ω platinum/250": 2,
                "100 Ω platinum/500": 3,
                "1000 Ω platinum": 4,
                "NTC RTD": 5,
                "Thermocouple 25mV": 6,
                "Thermocouple 50 mV": 7,
                "2.5 V, 1 mA": 8,
                "7.5 V, 1 mA": 9
            },
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "compensation",
            vals=Enum(0, 1),
            parameter_class=GroupParameter
        )

        Group(
            [self.type, self.compensation],
            set_cmd=f"INTYPE {self._channel}, {{type}}, {{compensation}}",
            get_cmd=f"INTYPE? {self._channel}"
        )

    def decode_sensor_status(self, sum_of_codes: int) ->str:
        """
        The sensor status is one of the status codes, or a sum thereof. Multiple
        status are possible as they are not necessarily mutually exclusive.

        args:
            sum_of_codes (int)
        """
        components = list(self.sensor_status_codes.keys())
        codes = self._get_sum_terms(components, int(sum_of_codes))
        return ", ".join([self.sensor_status_codes[k] for k in codes])

    @staticmethod
    def _get_sum_terms(components: list, number: int):
        """
        Example:
        >>> components = [0, 1, 16, 32, 64, 128]
        >>> _get_sum_terms(components, 96)
        >>> ...[64, 32]  # This is correct because 96=64+32
        """
        if number in components:
            terms = [number]
        else:
            terms = []
            comp = np.sort(components)[::-1]
            comp = comp[comp <= number]

            while len(comp):
                c = comp[0]
                number -= c
                terms.append(c)

                comp = comp[comp <= number]

        return terms


class Output_325(InstrumentChannel):
    def __init__(self, parent: 'Model_325', name: str, channel: str) -> None:
        super().__init__(parent, name)
        self._channel = channel

        self.add_parameter(
            "control_mode",
            get_cmd=f"CMODE? {self._channel}",
            set_cmd=f"CMODE {self._channel},{{}}",
            val_mapping={
                "Manual PID": "1",
                "Zone": "2",
                "Open Loop": "3",
                "AutoTune PID": "4",
                "AutoTune PI": "5",
                "AutoTune P": "6"
            }
        )

        self.add_parameter(
            "input_channel",
            vals=Enum("A", "B", "a", "b"),
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "unit",
            vals=Enum("Kelvin", "Celsius", "Sensor Units"),
            val_mapping={
                "Kelvin": "1",
                "Celsius": "2",
                "Sensor Units": "3"
            },
            parameter_class=GroupParameter
        )

        self.add_parameter(
            'powerup_enable',
            val_mapping={True: 1, False: 0},
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "output_metric",
            vals=Enum("current", "power"),
            val_mapping={
                "current": "1",
                "power": "2",
            },
            parameter_class=GroupParameter
        )

        Group(
            [self.input_channel, self.unit, self.powerup_enable,
             self.output_metric],
            set_cmd=f"CSET {self._channel}, {{input_channel}}, {{unit}}, "
                    f"{{powerup_enable}}, {{output_metric}}",
            get_cmd=f"CSET? {self._channel}"
        )

        self.add_parameter(
            'P',
            vals=Numbers(0, 1000),
            get_parser=float,
            parameter_class=GroupParameter
        )

        self.add_parameter(
            'I',
            vals=Numbers(0, 1000),
            get_parser=float,
            parameter_class=GroupParameter
        )

        self.add_parameter(
            'D',
            vals=Numbers(0, 1000),
            get_parser=float,
            parameter_class=GroupParameter
        )

        Group(
            [self.P, self.I, self.D],
            set_cmd=f'PID {self._channel}, {{P}}, {{I}}, {{D}}',
            get_cmd=f'PID? {self._channel}'
        )

        if self._channel == 0:
            valid_output_ranges = Enum(0, 1, 2)
        else:
            valid_output_ranges = Enum(0, 1)

        self.add_parameter(
            'output_range',
            vals=valid_output_ranges,
            set_cmd=f'RANGE {self._channel}, {{}}',
            get_cmd=f'RANGE? {self._channel}',
            val_mapping={
                "Off": '0',
                "Low (2.5W)": '1',
                "High (25W)": '2'
            }
        )

        self.add_parameter(
            'setpoint',
            vals=Numbers(0, 400),
            get_parser=float,
            set_cmd=f'SETP {self._channel}, {{}}',
            get_cmd=f'SETP? {self._channel}'
        )

        self.add_parameter(
            "ramp_state",
            vals=Enum(0, 1),
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "ramp_rate",
            vals=Numbers(0, 100 / 60 * 1E3),
            unit="mK/s",
            parameter_class=GroupParameter,
            get_parser=lambda v: float(v) / 60 * 1E3,  # We get values in K/min,
            set_parser=lambda v: v * 60 * 1E-3  # Convert to K/min
        )

        Group(
            [self.ramp_state, self.ramp_rate],
            set_cmd=f"RAMP {self._channel}, {{ramp_state}}, {{ramp_rate}}",
            get_cmd=f"RAMP? {self._channel}"
        )

        self.add_parameter(
            "is_ramping",
            get_cmd=f"RAMPST? {self._channel}"
        )


class Model_325(VisaInstrument):
    """
    Lakeshore Model 331 Temperature Controller Driver
    Controlled via sockets
    """

    def __init__(self, name: str, address: str, **kwargs) -> None:
        super().__init__(name, address, terminator="\r\n", **kwargs)

        input_channels = ChannelList(self, "thermometer", Model_325_Channel,
                                     snapshotable=False)

        for chan_name in ['a', 'b']:
            channel = Model_325_Channel(self, 'sensor_{}'.format(chan_name),
                                        chan_name)
            input_channels.append(channel)
            self.add_submodule('sensor_{}'.format(chan_name), channel)

        input_channels.lock()
        self.add_submodule("sensor", input_channels)

        output_channels = ChannelList(self, "heater", Output_325,
                                      snapshotable=False)

        for chan_name in ["1", "2"]:
            channel = Output_325(self, 'heater_{}'.format(chan_name),
                                 chan_name)
            output_channels.append(channel)
            self.add_submodule('heater_{}'.format(chan_name), channel)

        output_channels.lock()
        self.add_submodule("heater", output_channels)

        self.connect_message()
