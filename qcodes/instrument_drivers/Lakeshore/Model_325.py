from enum import IntFlag
from typing import cast, List, Tuple, Iterable, TextIO, Any, Optional
from itertools import takewhile

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Enum, Numbers
from qcodes.instrument.group_parameter import GroupParameter, Group


def read_curve_file(curve_file: TextIO) -> dict:
    """
    Read a curve file with extension .330
    The file format of this file is shown in test_lakeshore_file_parser.py
    in the test module

    The output is a dictionary with keys: "metadata" and "data".
    The metadata dictionary contains the first n lines of the curve file which
    are in the format "item: value". The data dictionary contains the actual
    curve data.
    """

    def split_data_line(line: str, parser: type = str) -> List[str]:
        return [parser(i) for i in line.split("  ") if i != ""]

    def strip(strings: Iterable[str]) -> Tuple:
        return tuple(s.strip() for s in strings)

    lines = iter(curve_file.readlines())
    # Meta data lines contain a colon
    metadata_lines = takewhile(lambda s: ":" in s, lines)
    # Data from the file is collected in the following dict
    file_data = dict()
    # Capture meta data
    parsed_lines = [strip(line.split(":")) for line in metadata_lines]
    file_data["metadata"] = {key: value for key, value in parsed_lines}
    # After meta data we have a data header
    header_items = strip(split_data_line(next(lines)))
    # After that we have the curve data
    data = [
        split_data_line(line, parser=float)
        for line in lines if line.strip() != ""
    ]

    file_data["data"] = dict(
        zip(header_items, zip(*data))
    )

    return file_data


def get_sanitize_data(file_data: dict) -> dict:
    """
    Data as found in the curve files are slightly different then
    the dictionary as expected by the 'upload_curve' method of the
    driver
    """
    data_dict = dict(file_data["data"])
    # We do not need the index column
    del data_dict["No."]
    # Rename the 'Units' column to the appropriate name
    # Look up under the 'Data Format' entry to find what units we have
    data_format = file_data['metadata']['Data Format']
    # This is a string in the form '4      (Log Ohms/Kelvin)'
    data_format_int = int(data_format.split()[0])
    correct_name = Model_325_Curve.valid_sensor_units[data_format_int - 1]
    # Rename the column
    data_dict[correct_name] = data_dict["Units"]
    del data_dict["Units"]

    return data_dict


class Status(IntFlag):
    sensor_units_overrang = 128
    sensor_units_zero = 64
    temp_overrange = 32
    temp_underrange = 16
    invalid_reading = 1


class Model_325_Curve(InstrumentChannel):

    valid_sensor_units = ["mV", "V", "Ohm", "log Ohm"]
    temperature_key = "Temperature (K)"

    def __init__(self, parent: 'Model_325', index: int) -> None:

        self._index = index
        name = f"curve_{index}"
        super().__init__(parent, name)

        self.add_parameter(
            "serial_number",
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "format",
            val_mapping={
                f"{unt}/K": i+1 for i, unt in enumerate(self.valid_sensor_units)
            },
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "limit_value",
            parameter_class=GroupParameter
        )

        self.add_parameter(
            "coefficient",
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
            set_cmd=f"CRVHDR {self._index}, {{curve_name}}, "
                    f"{{serial_number}}, {{format}}, {{limit_value}}, "
                    f"{{coefficient}}",
            get_cmd=f"CRVHDR? {self._index}"
        )

    def get_data(self) -> dict:
        curve = [
            float(a) for point_index in range(1, 200)
            for a in self.ask(f"CRVPT? {self._index}, {point_index}").split(",")
        ]

        d = {self.temperature_key: curve[1::2]}
        sensor_unit = self.format().split("/")[0]
        d[sensor_unit] = curve[::2]

        return d

    @classmethod
    def validate_datadict(cls, data_dict: dict) -> str:
        """
        A data dict has two keys, one of which is 'Temperature (K)'. The other
        contains the units in which the curve is defined and must be one of:
        'mV', 'V', 'Ohm' or 'log Ohm'

        This method validates this and returns the sensor unit encountered in
        the data dict
        """
        if cls.temperature_key not in data_dict:
            raise ValueError(f"At least {cls.temperature_key} needed in the "
                             f"data dictionary")

        sensor_units = [i for i in data_dict.keys() if i != cls.temperature_key]

        if len(sensor_units) != 1:
            raise ValueError(
                "Data dictionary should have one other key, other then "
                "'Temperature (K)'"
            )

        sensor_unit = sensor_units[0]

        if sensor_unit not in cls.valid_sensor_units:
            raise ValueError(
                f"Sensor unit {sensor_unit} invalid. This needs to be one of "
                f"{', '.join(cls.valid_sensor_units)}"
            )

        data_size = len(data_dict[cls.temperature_key])
        if data_size != len(data_dict[sensor_unit]) or data_size > 200:
            raise ValueError("The length of the temperature axis should be "
                             "the same as the length of the sensor axis and "
                             "should not exceed 200 in size")

        return sensor_unit

    def set_data(self, data_dict: dict,
                 sensor_unit: Optional[str] = None) -> None:
        """
        Set the curve data according to the values found the the dictionary.

        Args:
            data_dict (dict): See `validate_datadict` to see the format of this
                                dictionary
            sensor_unit (str): If None, the data dict is validated and the
                                units are extracted.
        """
        if sensor_unit is None:
            sensor_unit = self.validate_datadict(data_dict)

        temperature_values = data_dict[self.temperature_key]
        sensor_values = data_dict[sensor_unit]

        for value_index, (temperature_value, sensor_value) in \
                enumerate(zip(temperature_values, sensor_values)):

            cmd_str = f"CRVPT {self._index}, {value_index + 1}, " \
                      f"{sensor_value:3.3f}, {temperature_value:3.3f}"

            self.write(cmd_str)


class Model_325_Sensor(InstrumentChannel):
    """
    A single sensor of a  Lakeshore 325.

    Args:
        parent (Model_325): The instrument this heater belongs to
        name (str)
        inp (str): Either "A" or "B"
    """

    def __init__(self, parent: 'Model_325', name: str, inp: str) -> None:

        if inp not in ["A", "B"]:
            raise ValueError("Please either specify input 'A' or 'B'")

        super().__init__(parent, name)
        self._input = inp

        self.add_parameter(
            'temperature',
            get_cmd=f'KRDG? {self._input}',
            get_parser=float,
            label='Temperature',
            unit='K'
        )

        self.add_parameter(
            'status',
            get_cmd=f'RDGST? {self._input}',
            get_parser=lambda status: self.decode_sensor_status(int(status)),
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
            set_cmd=f"INCRV {self._input}, {{}}",
            get_cmd=f"INCRV? {self._input}",
            get_parser=int,
            vals=Numbers(min_value=1, max_value=35)
        )

    @staticmethod
    def decode_sensor_status(sum_of_codes: int) -> str:
        total_status = Status(sum_of_codes)
        if sum_of_codes == 0:
            return 'OK'
        status_messages = [st.name.replace('_', ' ') for st in Status
                           if st in total_status]
        return ", ".join(status_messages)

    @property
    def curve(self) -> Model_325_Curve:
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

        self.add_parameter(
            "resistance",
            get_cmd=f"HTRRES? {self._loop}",
            set_cmd=f"HTRRES {self._loop}, {{}}",
            val_mapping={
                25: 1,
                50: 2,
            },
            label='Resistance',
            unit="Ohm"
        )

        self.add_parameter(
            "heater_output",
            get_cmd=f"HTR? {self._loop}",
            get_parser=float,
            label='Heater Output',
            unit="%"
        )


class Model_325(VisaInstrument):
    """
    Lakeshore Model 325 Temperature Controller Driver
    """

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, terminator="\r\n", **kwargs)

        sensors = ChannelList(
            self, "sensor", Model_325_Sensor, snapshotable=False)

        for inp in ['A', 'B']:
            sensor = Model_325_Sensor(self, f'sensor_{inp}', inp)
            sensors.append(sensor)
            self.add_submodule(f'sensor_{inp}', sensor)

        sensors.lock()
        self.add_submodule("sensor", sensors)

        heaters = ChannelList(
            self, "heater", Model_325_Heater, snapshotable=False)

        for loop in [1, 2]:
            heater = Model_325_Heater(self, f'heater_{loop}', loop)
            heaters.append(heater)
            self.add_submodule(f'heater_{loop}', heater)

        heaters.lock()
        self.add_submodule("heater", heaters)

        curves = ChannelList(
            self, "curve", Model_325_Curve, snapshotable=False
        )

        for curve_index in range(1, 35):
            curve = Model_325_Curve(self, curve_index)
            curves.append(curve)

        self.add_submodule("curve", curves)

        self.connect_message()

    def upload_curve(
            self, index: int, name: str, serial_number: str, data_dict: dict
    ) -> None:
        """
        Upload a curve to the given index

        Args:
             index (int): The index to upload the curve to. We can only use
                            indices reserved for user defined curves, 21-35
             name (str)
             serial_number (str)
             data_dict (dict): A dictionary containing the curve data
        """
        if index not in range(21, 36):
            raise ValueError("index value should be between 21 and 35")

        sensor_unit = Model_325_Curve.validate_datadict(data_dict)

        curve = self.curve[index - 1]
        curve.curve_name(name)
        curve.serial_number(serial_number)
        curve.format(f"{sensor_unit}/K")
        curve.set_data(data_dict, sensor_unit=sensor_unit)

    def upload_curve_from_file(self, index: int, file_path: str) -> None:
        """
        Upload a curve from a curve file. Note that we only support
        curve files with extension .330
        """
        if not file_path.endswith(".330"):
            raise ValueError("Only curve files with extension .330 are supported")

        with open(file_path) as curve_file:
            file_data = read_curve_file(curve_file)

        data_dict = get_sanitize_data(file_data)
        name = file_data["metadata"]["Sensor Model"]
        serial_number = file_data["metadata"]["Serial Number"]

        self.upload_curve(index, name, serial_number, data_dict)
