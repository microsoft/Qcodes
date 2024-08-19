"""
This contains an alias of the Lakeshore Model 325 driver.
It will eventually be deprecated and removed
"""

from collections.abc import Iterable
from enum import IntFlag
from itertools import takewhile
from typing import TYPE_CHECKING, Any, Optional, TextIO, cast

from qcodes.instrument import (
    ChannelList,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import Group, GroupParameter
from qcodes.validators import Enum, Numbers

from .Lakeshore_model_325 import LakeshoreModel325Curve as Model_325_Curve
from .Lakeshore_model_325 import LakeshoreModel325Heater as Model_325_Heater
from .Lakeshore_model_325 import LakeshoreModel325Sensor as Model_325_Sensor
from .Lakeshore_model_325 import LakeshoreModel325Status as Status
from .Lakeshore_model_325 import _get_sanitize_data as get_sanitize_data
from .Lakeshore_model_325 import _read_curve_file as read_curve_file

if TYPE_CHECKING:
    from typing_extensions import Unpack


class Model_325(VisaInstrument):
    """
    Lakeshore Model 325 Temperature Controller Driver
    """

    default_terminator = "\r\n"

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ) -> None:
        super().__init__(name, address, **kwargs)

        sensors = ChannelList(self, "sensor", Model_325_Sensor, snapshotable=False)

        for inp in ["A", "B"]:
            sensor = Model_325_Sensor(self, f"sensor_{inp}", inp)  # type: ignore[arg-type]
            sensors.append(sensor)
            self.add_submodule(f"sensor_{inp}", sensor)

        self.add_submodule("sensor", sensors.to_channel_tuple())

        heaters = ChannelList(self, "heater", Model_325_Heater, snapshotable=False)

        for loop in [1, 2]:
            heater = Model_325_Heater(self, f"heater_{loop}", loop)  # type: ignore[arg-type]
            heaters.append(heater)
            self.add_submodule(f"heater_{loop}", heater)

        self.add_submodule("heater", heaters.to_channel_tuple())

        curves = ChannelList(self, "curve", Model_325_Curve, snapshotable=False)

        for curve_index in range(1, 35):
            curve = Model_325_Curve(self, curve_index)  # type: ignore[arg-type]
            curves.append(curve)

        self.add_submodule("curve", curves)

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
