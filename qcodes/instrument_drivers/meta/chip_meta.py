from qcodes.instrument_drivers.meta.device_meta import DeviceMeta
from typing import List, Dict, Any

from qcodes.instrument.base import InstrumentBase
from qcodes.station import Station


class ChipMeta(InstrumentBase):
    """Meta instrument for a chip"""
    def __init__(
        self,
        name: str,
        station: "Station",
        devices: Dict[str, Dict[str, List[str]]],
        initial_values: Dict[str, Dict[str, Any]],
        connections: Dict[str, List[str]],
        set_initial_values_on_load: bool = False,
        **kwargs):
        """
        Create a ChipMeta instrument.

        Args:
            name: Chip name
            station: Measurement station with real instruments
            devices: Devices on the chip, for each a DeviceMeta is created
            initial_values: Default values to set on load
            connections: Connections from channels to endpoint instrument parameters
            set_initial_values_on_load: Set default values on load. Defaults to False.
        """
        super().__init__(name=name, **kwargs)

        for device_name, channels in devices.items():
            device = DeviceMeta(
                name=device_name,
                station=station,
                channels=channels,
                initial_values=initial_values.get(device_name),
                connections=connections,
                set_initial_values_on_load=set_initial_values_on_load
            )

            self.add_submodule(
                device_name,
                device
            )

    def __repr__(self):
        devices = ", ".join(self.submodules.keys())
        return f"ChipMeta(name={self.name}, devices={devices})"
