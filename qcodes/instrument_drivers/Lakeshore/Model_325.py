import numpy as np
from typing import cast

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Enum, Numbers
from qcodes.instrument.group_parameter import GroupParameter, Group


class Model_325_Curve(InstrumentChannel):

    def __init__(self, parent: 'Model_325', index: int):

        self._index = index
        name = f"curve_{index}"
        super().__init__(parent, name)

        self.add_parameter(
            "serial_number",
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "format",
            vals=Enum(1, 2, 3, 4),
            val_mapping={
                "mV/K": 1,
                "V/K": 2,
                "Ohm/K": 3,
                "log Ohm/K": 4
            },
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "limit_value",
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "coefficient",
            vals=Enum(1, 2),
            val_mapping={
                "negative": 1,
                "positive": 2
            },
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "curve_name",
            parameter_class=GroupParameter
        )

        Group(
            [
                self.curve_name, self.serial_number, self.format,
                self.limit_value, self.coefficient
            ],
            set_cmd=f"CRVHDR {self._index}, {{curve_name}} "
                    f"{{serial_number}}, {{format}}, {{limit_value}}, "
                    f"{{coefficient}}",
            get_cmd=f"CRVHDR? {self._index}"
        )

    def get_data(self) -> dict:
        curve = [
            float(a) for point_index in range(1, 200)
            for a in self.ask(f"CRVPT? {self._index}, {point_index}").split(",")
        ]

        d = {"Temperature (K)": curve[1::2]}
        sensor_unit = self.format().split("/")[0]
        d[sensor_unit] = curve[::2]

        return d

    def set_data(self, curve: dict) ->None:

        temperature_values = curve["Temperature (K)"]
        sensor_unit = set(curve.keys()).difference({"Temperature (K)"})

        if len(sensor_unit) != 1:
            raise ValueError("Curve dictionary should have one key, other "
                             "then 'Temperature (K)'")

        sensor_unit = list(sensor_unit)[0]
        valid_sensor_units = [
            k.split("/")[0] for k in self.format.val_mapping.keys()]

        if sensor_unit not in valid_sensor_units:
            raise ValueError(f"Sensor unit {sensor_unit} invalid. This needs "
                             f"to be one of {','.join(valid_sensor_units)}")

        self.format(f"{sensor_unit}/K")
        sensor_values = curve[sensor_unit]

        for value_index, (temperature_value, sensor_value) in \
                enumerate(zip(temperature_values, sensor_values)):

            cmd_str = f"CRVHDR {self._index}, {value_index}, {sensor_value}, " \
                      f"{temperature_value}"

            self.write(cmd_str)


class Model_325_Sensor(InstrumentChannel):
    """
    A single sensor of a  Lakeshore 325.

    Args:
        parent (Model_325): The instrument this heater belongs to
        name (str)
        inp (str): Either "A" or "B"
    """

    sensor_status_codes = {
        0:  "OK",
        1:  "invalid reading",
        16:  "temp underrange",
        32:  "temp overrange",
        64:  "sensor units zero",
        128: "sensor units overrang"
    }

    def __init__(self, parent: 'Model_325', name: str, inp: str) ->None:

        if inp not in ["A", "B"]:
            raise ValueError("Please either specify input 'A' or 'B'")

        super().__init__(parent, name)
        self._input = inp

        self.add_parameter(
            'temperature',
            get_cmd='KRDG? {}'.format(self._input),
            get_parser=float,
            label='Temerature',
            unit='K'
        )

        self.add_parameter(
            'status',
            get_cmd='RDGST? {}'.format(self._input),
            get_parser=self.decode_sensor_status,
            label='Sensor_Status'
        )

        self.add_parameter(
            "type",
            val_mapping={
                "Silicon diode": 0,
                "GaAlAs diode": 1,
                "100 Ohm platinum/250": 2,
                "100 Ohm platinum/500": 3,
                "1000 Ohm platinum": 4,
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
            set_cmd=f"INTYPE {self._input}, {{type}}, {{compensation}}",
            get_cmd=f"INTYPE? {self._input}"
        )

        self.add_parameter(
            "curve_index",
            set_cmd=f"INCRV {self._input} {{}}",
            get_cmd=f"INCRV? {self._input}",
            get_parser=int
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

    @property
    def curve(self) ->Model_325_Curve:
        parent = cast(Model_325, self.parent)
        return Model_325_Curve(parent,  self.curve_index())


class Model_325_Heater(InstrumentChannel):
    """
    Heater control for the Lakeshore 325.

    Args:
        parent (Model_325): The instrument this heater belongs to
        name (str)
        loop (int): Either 1 or 2
    """
    def __init__(self, parent: 'Model_325', name: str, loop: int) -> None:

        if loop not in [1, 2]:
            raise ValueError("Please either specify loop 1 or 2")

        super().__init__(parent, name)
        self._loop = loop

        self.add_parameter(
            "control_mode",
            get_cmd=f"CMODE? {self._loop}",
            set_cmd=f"CMODE {self._loop},{{}}",
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
            vals=Enum("A", "B"),
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
            set_cmd=f"CSET {self._loop}, {{input_channel}}, {{unit}}, "
                    f"{{powerup_enable}}, {{output_metric}}",
            get_cmd=f"CSET? {self._loop}"
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
            set_cmd=f'PID {self._loop}, {{P}}, {{I}}, {{D}}',
            get_cmd=f'PID? {self._loop}'
        )

        if self._loop == 1:
            valid_output_ranges = Enum(0, 1, 2)
        else:
            valid_output_ranges = Enum(0, 1)

        self.add_parameter(
            'output_range',
            vals=valid_output_ranges,
            set_cmd=f'RANGE {self._loop}, {{}}',
            get_cmd=f'RANGE? {self._loop}',
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
            set_cmd=f'SETP {self._loop}, {{}}',
            get_cmd=f'SETP? {self._loop}'
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
            set_cmd=f"RAMP {self._loop}, {{ramp_state}}, {{ramp_rate}}",
            get_cmd=f"RAMP? {self._loop}"
        )

        self.add_parameter(
            "is_ramping",
            get_cmd=f"RAMPST? {self._loop}"
        )


class Model_325(VisaInstrument):
    """
    Lakeshore Model 325 Temperature Controller Driver
    """

    def __init__(self, name: str, address: str, **kwargs) -> None:
        super().__init__(name, address, terminator="\r\n", **kwargs)

        sensors = ChannelList(
            self, "sensor", Model_325_Sensor, snapshotable=False)

        for inp in ['A', 'B']:
            sensor = Model_325_Sensor(self, 'sensor_{}'.format(inp), inp)
            sensors.append(sensor)
            self.add_submodule('sensor_{}'.format(inp), sensor)

        sensors.lock()
        self.add_submodule("sensor", sensors)

        heaters = ChannelList(
            self, "heater", Model_325_Heater, snapshotable=False)

        for loop in [1, 2]:
            heater = Model_325_Heater(self, 'heater_{}'.format(loop), loop)
            heaters.append(heater)
            self.add_submodule('heater_{}'.format(loop), heater)

        heaters.lock()
        self.add_submodule("heater", heaters)

        self.connect_message()
