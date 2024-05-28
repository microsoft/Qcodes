from typing import TYPE_CHECKING, ClassVar

import pyvisa.constants
import pyvisa.resources

import qcodes.validators as vals
from qcodes.parameters import Group, GroupParameter

from .lakeshore_base import (
    LakeshoreBase,
    LakeshoreBaseOutput,
    LakeshoreBaseSensorChannel,
)

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import InstrumentBaseKWArgs, VisaInstrumentKWArgs

# There are 2 sensors channels (a.k.a. measurement inputs) in Model 335.
# Unlike other Lakeshore models, Model 335 refers to the channels using
# letters, and not numbers
_channel_name_to_command_map: dict[str, str] = {"A": "A", "B": "B"}

# OUTMODE command of this model refers to the outputs via integer numbers,
# while everywhere else within this model letters are used. This map is
# created in order to preserve uniformity of referencing to sensor channels
# within this driver.
_channel_name_to_outmode_command_map: dict[str, int] = {
    ch_name: num_for_cmd + 1
    for num_for_cmd, ch_name in enumerate(_channel_name_to_command_map.keys())
}
_channel_name_to_outmode_command_map.update({"None": 0})


class LakeshoreModel335Channel(LakeshoreBaseSensorChannel):
    """
    An InstrumentChannel representing a single sensor on a Lakeshore Model 335.

    """

    SENSOR_STATUSES: ClassVar[dict[int, str]] = {
        0: "OK",
        1: "Invalid Reading",
        16: "Temp Underrange",
        32: "Temp Overrange",
        64: "Sensor Units Zero",
        128: "Sensor Units Overrange",
    }

    def __init__(
        self,
        parent: "LakeshoreModel335",
        name: str,
        channel: str,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        super().__init__(parent, name, channel, **kwargs)

        # Parameters related to Input Type Parameter Command (INTYPE)
        self.sensor_type: GroupParameter = self.add_parameter(
            "sensor_type",
            label="Input sensor type",
            docstring="Specifies input sensor type",
            val_mapping={
                "disabled": 0,
                "diode": 1,
                "platinum_rtd": 2,
                "ntc_rtd": 3,
                "thermocouple": 4,
            },
            parameter_class=GroupParameter,
        )
        """Specifies input sensor type"""
        self.auto_range_enabled: GroupParameter = self.add_parameter(
            "auto_range_enabled",
            label="Autoranging",
            docstring="Specifies if autoranging is enabled. "
            "Does not apply for diode sensor type",
            val_mapping={False: 0, True: 1},
            parameter_class=GroupParameter,
        )
        """Specifies if autoranging is enabled. Does not apply for diode sensor type"""
        self.range: GroupParameter = self.add_parameter(
            "range",
            label="Range",
            docstring="Specifies input range when autorange is "
            "not enabled. If autorange is on, the "
            "returned value corresponds to the "
            "currently auto-selected range. The list "
            "of available ranges depends on the "
            "chosen sensor type: diode 0-1, platinum "
            "RTD 0-6, NTC RTD 0-8. Refer to the page "
            "136 of the manual for the lookup table",
            vals=vals.Numbers(0, 8),
            parameter_class=GroupParameter,
        )
        """
        Specifies input range when autorange is not enabled. If autorange is on,
        the returned value corresponds to the currently auto-selected range.
        The list of available ranges depends on the chosen sensor type: diode 0-1,
        platinum RTD 0-6, NTC RTD 0-8. Refer to the page 136 of the manual for the lookup table
        """
        self.compensation_enabled: GroupParameter = self.add_parameter(
            "compensation_enabled",
            label="Compensation enabled",
            docstring="Specifies input compensation. Reversal "
            "for thermal EMF compensation if input "
            "is resistive, room compensation if "
            "input is thermocouple. Always 0 if input "
            "is a diode",
            val_mapping={False: 0, True: 1},
            parameter_class=GroupParameter,
        )
        """
        Specifies input compensation. Reversal for thermal EMF compensation if input
        is resistive, room compensation if input is thermocouple.
        Always 0 if input is a diode
        """
        self.units: GroupParameter = self.add_parameter(
            "units",
            label="Preferred units",
            docstring="Specifies the preferred units parameter "
            "for sensor readings and for the control "
            "setpoint (kelvin, celsius, or sensor)",
            val_mapping={"kelvin": 1, "celsius": 2, "sensor": 3},
            parameter_class=GroupParameter,
        )
        """
        Specifies the preferred units parameter for sensor readings and for the control
        setpoint (kelvin, celsius, or sensor)
        """
        self.output_group = Group(
            [
                self.sensor_type,
                self.auto_range_enabled,
                self.range,
                self.compensation_enabled,
                self.units,
            ],
            set_cmd=f"INTYPE {self._channel}, "
            f"{{sensor_type}}, "
            f"{{auto_range_enabled}}, {{range}}, "
            f"{{compensation_enabled}}, "
            f"{{units}}",
            get_cmd=f"INTYPE? {self._channel}",
        )


class LakeshoreModel335CurrentSource(LakeshoreBaseOutput):
    """
    InstrumentChannel for current sources on Lakeshore Model 335.

    Class for control outputs 1 and 2 of Lakeshore Model 335 that are variable DC current
    sources referenced to chassis ground.
    """

    MODES: ClassVar[dict[str, int]] = {
        "off": 0,
        "closed_loop": 1,
        "zone": 2,
        "open_loop": 3,
    }

    RANGES: ClassVar[dict[str, int]] = {"off": 0, "low": 1, "medium": 2, "high": 3}

    _input_channel_parameter_kwargs: ClassVar[dict[str, dict[str, int]]] = {
        "val_mapping": _channel_name_to_outmode_command_map
    }

    def __init__(
        self,
        parent: "LakeshoreModel335",
        output_name: str,
        output_index: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        super().__init__(parent, output_name, output_index, has_pid=True, **kwargs)

        self.P.vals = vals.Numbers(0.1, 1000)
        self.I.vals = vals.Numbers(0.1, 1000)
        self.D.vals = vals.Numbers(0, 200)


class LakeshoreModel335(LakeshoreBase):
    """
    Lakeshore Model 335 Temperature Controller Driver
    """

    channel_name_command: ClassVar[dict[str, str]] = _channel_name_to_command_map

    CHANNEL_CLASS = LakeshoreModel335Channel

    input_channel_parameter_values_to_channel_name_on_instrument = (
        _channel_name_to_command_map
    )

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ) -> None:
        super().__init__(name, address, print_connect_message=False, **kwargs)

        if isinstance(self.visa_handle, pyvisa.resources.serial.SerialInstrument):
            self.visa_handle.baud_rate = 57600
            self.visa_handle.data_bits = 7
            self.visa_handle.parity = pyvisa.constants.Parity(1)

        self.output_1 = LakeshoreModel335CurrentSource(self, "output_1", 1)
        self.output_2 = LakeshoreModel335CurrentSource(self, "output_2", 2)

        self.connect_message()
