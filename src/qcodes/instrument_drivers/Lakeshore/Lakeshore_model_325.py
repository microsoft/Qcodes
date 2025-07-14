from enum import IntFlag
from itertools import takewhile
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    SupportsBytes,
    SupportsIndex,
    TextIO,
    cast,
)

from qcodes.instrument import (
    ChannelList,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import Group, GroupParameter, Parameter
from qcodes.validators import Enum, Numbers

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Self

    from typing_extensions import Buffer, Unpack

    from qcodes.instrument.channel import ChannelTuple


def _read_curve_file(curve_file: TextIO) -> dict[Any, Any]:
    """
    Read a curve file with extension .330
    The file format of this file is shown in test_lakeshore_file_parser.py
    in the test module

    The output is a dictionary with keys: "metadata" and "data".
    The metadata dictionary contains the first n lines of the curve file which
    are in the format "item: value". The data dictionary contains the actual
    curve data.
    """

    def split_data_line(line: str, parser: type = str) -> list[Any]:
        return [parser(i) for i in line.split("  ") if i != ""]

    def strip(strings: "Iterable[str]") -> tuple[str, ...]:
        return tuple(s.strip() for s in strings)

    lines = iter(curve_file.readlines())
    # Meta data lines contain a colon
    metadata_lines = takewhile(lambda s: ":" in s, lines)
    # Data from the file is collected in the following dict
    file_data: dict[str, dict[str, Any]] = dict()
    # Capture meta data
    parsed_lines = [strip(line.split(":")) for line in metadata_lines]
    file_data["metadata"] = {key: value for key, value in parsed_lines}
    # After meta data we have a data header
    header_items = strip(split_data_line(next(lines)))
    # After that we have the curve data
    data: list[list[float]] = [
        split_data_line(line, parser=float) for line in lines if line.strip() != ""
    ]
    file_data["data"] = dict(zip(header_items, zip(*data)))

    return file_data


def _get_sanitize_data(file_data: dict[Any, Any]) -> dict[Any, Any]:
    """
    Data as found in the curve files are slightly different from
    the dictionary as expected by the 'upload_curve' method of the
    driver
    """
    data_dict = dict(file_data["data"])
    # We do not need the index column
    del data_dict["No."]
    # Rename the 'Units' column to the appropriate name
    # Look up under the 'Data Format' entry to find what units we have
    data_format = file_data["metadata"]["Data Format"]
    # This is a string in the form '4      (Log Ohms/Kelvin)'
    data_format_int = int(data_format.split()[0])
    correct_name = LakeshoreModel325Curve.valid_sensor_units[data_format_int - 1]
    # Rename the column
    data_dict[correct_name] = data_dict["Units"]
    del data_dict["Units"]

    return data_dict


class LakeshoreModel325Status(IntFlag):
    """
    IntFlag that defines status codes for Lakeshore Model 325
    """

    sensor_units_overrang = 128
    sensor_units_zero = 64
    temp_overrange = 32
    temp_underrange = 16
    invalid_reading = 1

    # we reimplement from_bytes and to_bytes in order to fix docstrings that are incorrectly formatted
    # this in turn will enable us to build docs with warnings as errors.
    # This can be removed for python versions where https://github.com/python/cpython/pull/117847
    # is merged
    @classmethod
    def from_bytes(
        cls,
        bytes: "Iterable[SupportsIndex] | SupportsBytes | Buffer",
        byteorder: Literal["big", "little"] = "big",
        *,
        signed: bool = False,
    ) -> "Self":
        """
        Return the integer represented by the given array of bytes.

        Args:
            bytes: Holds the array of bytes to convert.  The argument must either
                support the buffer protocol or be an iterable object producing bytes.
                Bytes and bytearray are examples of built-in objects that support the
                buffer protocol.
            byteorder: The byte order used to represent the integer.  If byteorder is 'big',
                the most significant byte is at the beginning of the byte array.  If
                byteorder is 'little', the most significant byte is at the end of the
                byte array.  To request the native byte order of the host system, use
                `sys.byteorder` as the byte order value.  Default is to use 'big'.
            signed: Indicates whether two\'s complement is used to represent the integer.

        """
        return super().from_bytes(bytes, byteorder, signed=signed)

    def to_bytes(
        self,
        length: SupportsIndex = 1,
        byteorder: Literal["little", "big"] = "big",
        *,
        signed: bool = False,
    ) -> bytes:
        """
        Return an array of bytes representing an integer.

        Args:
            length: Length of bytes object to use.  An OverflowError is raised if the
                integer is not representable with the given number of bytes.  Default
                is length 1.
            byteorder: The byte order used to represent the integer.  If byteorder is \'big\',
                the most significant byte is at the beginning of the byte array.  If
                byteorder is \'little\', the most significant byte is at the end of the
                byte array. To request the native byte order of the host system, use
                `sys.byteorder` as the byte order value.  Default is to use \'big\'.
            signed: Determines whether two\'s complement is used to represent the integer.
                If signed is False and a negative integer is given, an OverflowError
                is raised.

        """
        return super().to_bytes(length, byteorder, signed=signed)


class LakeshoreModel325Curve(InstrumentChannel):
    """
    An InstrumentChannel representing a curve on a Lakeshore Model 325
    """

    valid_sensor_units = ("mV", "V", "Ohm", "log Ohm")
    temperature_key = "Temperature (K)"

    def __init__(
        self,
        parent: "LakeshoreModel325",
        index: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        self._index = index
        name = f"curve_{index}"
        super().__init__(parent, name, **kwargs)

        self.serial_number: GroupParameter = self.add_parameter(
            "serial_number", parameter_class=GroupParameter
        )
        """Parameter serial_number"""

        self.format: GroupParameter = self.add_parameter(
            "format",
            val_mapping={
                f"{unt}/K": i + 1 for i, unt in enumerate(self.valid_sensor_units)
            },
            parameter_class=GroupParameter,
        )
        """Parameter format"""

        self.limit_value: GroupParameter = self.add_parameter(
            "limit_value", parameter_class=GroupParameter
        )
        """Parameter limit_value"""

        self.coefficient: GroupParameter = self.add_parameter(
            "coefficient",
            val_mapping={"negative": 1, "positive": 2},
            parameter_class=GroupParameter,
        )
        """Parameter coefficient"""

        self.curve_name: GroupParameter = self.add_parameter(
            "curve_name", parameter_class=GroupParameter
        )
        """Parameter curve_name"""

        Group(
            [
                self.curve_name,
                self.serial_number,
                self.format,
                self.limit_value,
                self.coefficient,
            ],
            set_cmd=f"CRVHDR {self._index}, {{curve_name}}, "
            f"{{serial_number}}, {{format}}, {{limit_value}}, "
            f"{{coefficient}}",
            get_cmd=f"CRVHDR? {self._index}",
        )

    def get_data(self) -> dict[Any, Any]:
        curve = [
            float(a)
            for point_index in range(1, 200)
            for a in self.ask(f"CRVPT? {self._index}, {point_index}").split(",")
        ]

        d = {self.temperature_key: curve[1::2]}
        sensor_unit = self.format().split("/")[0]
        d[sensor_unit] = curve[::2]

        return d

    @classmethod
    def validate_datadict(cls, data_dict: dict[Any, Any]) -> str:
        """
        A data dict has two keys, one of which is 'Temperature (K)'. The other
        contains the units in which the curve is defined and must be one of:
        'mV', 'V', 'Ohm' or 'log Ohm'

        This method validates this and returns the sensor unit encountered in
        the data dict
        """
        if cls.temperature_key not in data_dict:
            raise ValueError(
                f"At least {cls.temperature_key} needed in the data dictionary"
            )

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
            raise ValueError(
                "The length of the temperature axis should be "
                "the same as the length of the sensor axis and "
                "should not exceed 200 in size"
            )

        return sensor_unit

    def set_data(
        self, data_dict: dict[Any, Any], sensor_unit: str | None = None
    ) -> None:
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

        for value_index, (temperature_value, sensor_value) in enumerate(
            zip(temperature_values, sensor_values)
        ):
            cmd_str = (
                f"CRVPT {self._index}, {value_index + 1}, "
                f"{sensor_value:3.3f}, {temperature_value:3.3f}"
            )

            self.write(cmd_str)


class LakeshoreModel325Sensor(InstrumentChannel):
    """
    InstrumentChannel for a single sensor of a Lakeshore Model 325.

    Args:
        parent (LakeshoreModel325): The instrument this heater belongs to
        name (str)
        inp (str): Either "A" or "B"

    """

    def __init__(
        self,
        parent: "LakeshoreModel325",
        name: str,
        inp: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        if inp not in ["A", "B"]:
            raise ValueError("Please either specify input 'A' or 'B'")

        super().__init__(parent, name)
        self._input = inp

        self.temperature: Parameter = self.add_parameter(
            "temperature",
            get_cmd=f"KRDG? {self._input}",
            get_parser=float,
            label="Temperature",
            unit="K",
        )
        """Parameter temperature"""

        self.status: Parameter = self.add_parameter(
            "status",
            get_cmd=f"RDGST? {self._input}",
            get_parser=lambda status: self.decode_sensor_status(int(status)),
            label="Sensor_Status",
        )
        """Parameter status"""

        self.type: GroupParameter = self.add_parameter(
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
                "7.5 V, 1 mA": 9,
            },
            parameter_class=GroupParameter,
        )
        """Parameter type"""

        self.compensation: GroupParameter = self.add_parameter(
            "compensation", vals=Enum(0, 1), parameter_class=GroupParameter
        )
        """Parameter compensation"""

        Group(
            [self.type, self.compensation],
            set_cmd=f"INTYPE {self._input}, {{type}}, {{compensation}}",
            get_cmd=f"INTYPE? {self._input}",
        )

        self.curve_index: Parameter = self.add_parameter(
            "curve_index",
            set_cmd=f"INCRV {self._input}, {{}}",
            get_cmd=f"INCRV? {self._input}",
            get_parser=int,
            vals=Numbers(min_value=1, max_value=35),
        )
        """Parameter curve_index"""

    @staticmethod
    def decode_sensor_status(sum_of_codes: int) -> str:
        total_status = LakeshoreModel325Status(sum_of_codes)
        if sum_of_codes == 0:
            return "OK"
        status_messages = [
            st.name.replace("_", " ")
            for st in LakeshoreModel325Status
            if st in total_status and st.name is not None
        ]
        return ", ".join(status_messages)

    @property
    def curve(self) -> LakeshoreModel325Curve:
        parent = cast("LakeshoreModel325", self.parent)
        return LakeshoreModel325Curve(parent, self.curve_index())


class LakeshoreModel325Heater(InstrumentChannel):
    def __init__(
        self,
        parent: "LakeshoreModel325",
        name: str,
        loop: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        InstrumentChannel for heater control on a Lakeshore Model 325.

        Args:
            parent: The instrument this heater belongs to
            name: Name of the Channel
            loop: Either 1 or 2
            **kwargs: Forwarded to baseclass.

        """

        if loop not in [1, 2]:
            raise ValueError("Please either specify loop 1 or 2")

        super().__init__(parent, name, **kwargs)
        self._loop = loop

        self.control_mode: Parameter = self.add_parameter(
            "control_mode",
            get_cmd=f"CMODE? {self._loop}",
            set_cmd=f"CMODE {self._loop},{{}}",
            val_mapping={
                "Manual PID": "1",
                "Zone": "2",
                "Open Loop": "3",
                "AutoTune PID": "4",
                "AutoTune PI": "5",
                "AutoTune P": "6",
            },
        )
        """Parameter control_mode"""

        self.input_channel: GroupParameter = self.add_parameter(
            "input_channel", vals=Enum("A", "B"), parameter_class=GroupParameter
        )
        """Parameter input_channel"""

        self.unit: GroupParameter = self.add_parameter(
            "unit",
            val_mapping={"Kelvin": "1", "Celsius": "2", "Sensor Units": "3"},
            parameter_class=GroupParameter,
        )
        """Parameter unit"""

        self.powerup_enable: GroupParameter = self.add_parameter(
            "powerup_enable",
            val_mapping={True: 1, False: 0},
            parameter_class=GroupParameter,
        )
        """Parameter powerup_enable"""

        self.output_metric: GroupParameter = self.add_parameter(
            "output_metric",
            val_mapping={
                "current": "1",
                "power": "2",
            },
            parameter_class=GroupParameter,
        )
        """Parameter output_metric"""

        Group(
            [self.input_channel, self.unit, self.powerup_enable, self.output_metric],
            set_cmd=f"CSET {self._loop}, {{input_channel}}, {{unit}}, "
            f"{{powerup_enable}}, {{output_metric}}",
            get_cmd=f"CSET? {self._loop}",
        )

        self.P: GroupParameter = self.add_parameter(
            "P", vals=Numbers(0, 1000), get_parser=float, parameter_class=GroupParameter
        )
        """Parameter P"""

        self.I: GroupParameter = self.add_parameter(
            "I", vals=Numbers(0, 1000), get_parser=float, parameter_class=GroupParameter
        )
        """Parameter I"""

        self.D: GroupParameter = self.add_parameter(
            "D", vals=Numbers(0, 1000), get_parser=float, parameter_class=GroupParameter
        )
        """Parameter D"""

        Group(
            [self.P, self.I, self.D],
            set_cmd=f"PID {self._loop}, {{P}}, {{I}}, {{D}}",
            get_cmd=f"PID? {self._loop}",
        )

        if self._loop == 1:
            valid_output_ranges = Enum(0, 1, 2)
        else:
            valid_output_ranges = Enum(0, 1)

        self.output_range: Parameter = self.add_parameter(
            "output_range",
            vals=valid_output_ranges,
            set_cmd=f"RANGE {self._loop}, {{}}",
            get_cmd=f"RANGE? {self._loop}",
            val_mapping={"Off": "0", "Low (2.5W)": "1", "High (25W)": "2"},
        )
        """Parameter output_range"""

        self.setpoint: Parameter = self.add_parameter(
            "setpoint",
            vals=Numbers(0, 400),
            get_parser=float,
            set_cmd=f"SETP {self._loop}, {{}}",
            get_cmd=f"SETP? {self._loop}",
        )
        """Parameter setpoint"""

        self.ramp_state: GroupParameter = self.add_parameter(
            "ramp_state", vals=Enum(0, 1), parameter_class=GroupParameter
        )
        """Parameter ramp_state"""

        self.ramp_rate: GroupParameter = self.add_parameter(
            "ramp_rate",
            vals=Numbers(0, 100 / 60 * 1e3),
            unit="mK/s",
            parameter_class=GroupParameter,
            get_parser=lambda v: float(v) / 60 * 1e3,  # We get values in K/min,
            set_parser=lambda v: v * 60 * 1e-3,  # Convert to K/min
        )
        """Parameter ramp_rate"""

        Group(
            [self.ramp_state, self.ramp_rate],
            set_cmd=f"RAMP {self._loop}, {{ramp_state}}, {{ramp_rate}}",
            get_cmd=f"RAMP? {self._loop}",
        )

        self.is_ramping: Parameter = self.add_parameter(
            "is_ramping", get_cmd=f"RAMPST? {self._loop}"
        )
        """Parameter is_ramping"""

        self.resistance: Parameter = self.add_parameter(
            "resistance",
            get_cmd=f"HTRRES? {self._loop}",
            set_cmd=f"HTRRES {self._loop}, {{}}",
            val_mapping={
                25: 1,
                50: 2,
            },
            label="Resistance",
            unit="Ohm",
        )
        """Parameter resistance"""

        self.heater_output: Parameter = self.add_parameter(
            "heater_output",
            get_cmd=f"HTR? {self._loop}",
            get_parser=float,
            label="Heater Output",
            unit="%",
        )
        """Parameter heater_output"""


class LakeshoreModel325(VisaInstrument):
    """
    QCoDeS driver for Lakeshore Model 325 Temperature Controller.
    """

    default_terminator = "\r\n"

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ) -> None:
        super().__init__(name, address, **kwargs)

        sensors = ChannelList(
            self, "sensor", LakeshoreModel325Sensor, snapshotable=False
        )

        self.sensor_A: LakeshoreModel325Sensor = self.add_submodule(
            "sensor_A", LakeshoreModel325Sensor(self, "sensor_A", "A")
        )
        """Sensor A"""
        sensors.append(self.sensor_A)
        self.sensor_B: LakeshoreModel325Sensor = self.add_submodule(
            "sensor_B", LakeshoreModel325Sensor(self, "sensor_B", "B")
        )
        """Sensor B"""
        sensors.append(self.sensor_B)

        self.sensor: ChannelTuple[LakeshoreModel325Sensor] = self.add_submodule(
            "sensor", sensors.to_channel_tuple()
        )
        """ChannelTuple of sensors"""

        heaters = ChannelList(
            self, "heater", LakeshoreModel325Heater, snapshotable=False
        )

        self.heater_1: LakeshoreModel325Heater = self.add_submodule(
            "heater_1", LakeshoreModel325Heater(self, "heater_1", 1)
        )
        """Heater 1"""
        heaters.append(self.heater_1)

        self.heater_2: LakeshoreModel325Heater = self.add_submodule(
            "heater_2", LakeshoreModel325Heater(self, "heater_2", 2)
        )
        """Heater 2"""
        heaters.append(self.heater_2)

        self.heater: ChannelTuple[LakeshoreModel325Heater] = self.add_submodule(
            "heater", heaters.to_channel_tuple()
        )
        """ChannelTuple of heaters"""

        curves = ChannelList(self, "curve", LakeshoreModel325Curve, snapshotable=False)

        for curve_index in range(1, 35):
            curve = LakeshoreModel325Curve(self, curve_index)
            curves.append(curve)

        self.curve: ChannelList[LakeshoreModel325Curve] = self.add_submodule(
            "curve", curves
        )
        """ChannelList of curves"""

        self.connect_message()

    def upload_curve(
        self, index: int, name: str, serial_number: str, data_dict: dict[Any, Any]
    ) -> None:
        """
        Upload a curve to the given index

        Args:
             index: The index to upload the curve to. We can only use
                indices reserved for user defined curves, 21-35
             name: Name of the curve
             serial_number: Serial number of the curve
             data_dict: A dictionary containing the curve data

        """
        if index not in range(21, 36):
            raise ValueError("index value should be between 21 and 35")

        sensor_unit = LakeshoreModel325Curve.validate_datadict(data_dict)

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
            file_data = _read_curve_file(curve_file)

        data_dict = _get_sanitize_data(file_data)
        name = file_data["metadata"]["Sensor Model"]
        serial_number = file_data["metadata"]["Serial Number"]

        self.upload_curve(index, name, serial_number, data_dict)
