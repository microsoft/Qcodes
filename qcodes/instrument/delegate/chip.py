from qcodes.instrument.delegate.device import Device
from typing import List, Dict, Any, TYPE_CHECKING

from qcodes.instrument.base import InstrumentBase

if TYPE_CHECKING:
    from qcodes.station import Station


class Chip(InstrumentBase):
    """
    Meta instrument for a Chip
    
    Args:
        name: Chip name
        station: Measurement station with real instruments
        devices: Devices on the chip, for each a Device is created
        initial_values: Default values to set on load
        set_initial_values_on_load: Set default values on load. Defaults to False.
    """
    def __init__(
        self,
        name: str,
        station: "Station",
        devices: Dict[str, Dict[str, List[str]]],
        initial_values: Dict[str, Dict[str, Any]],
        set_initial_values_on_load: bool = False,
        **kwargs):
        super().__init__(name=name, **kwargs)

        for device_name, aliases in devices.items():
            device = Device(
                name=device_name,
                station=station,
                aliases=aliases,
                initial_values=initial_values.get(device_name),
                set_initial_values_on_load=set_initial_values_on_load
            )

            self.add_submodule(
                device_name,
                device
            )

    def __repr__(self):
        devices = ", ".join(self.submodules.keys())
        return f"Chip(name={self.name}, devices={devices})"
