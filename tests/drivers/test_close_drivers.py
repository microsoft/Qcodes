"""
Closing a driver must also work if the instrument has already been closed, in
which case its attributes have been stripped, or if its initialization failed
before all its attributes were set.

The drivers tested here need hardware, vendor libraries or servers to be
initialized, so they are closed without having been initialized, which leaves
them without any instance attributes, like a closed instrument.
"""

import pytest

from qcodes.instrument import Instrument, IPInstrument
from qcodes.instrument_drivers.Keysight.KtM960x import KeysightM960x
from qcodes.instrument_drivers.Keysight.KtMAwg import KeysightM9336A
from qcodes.instrument_drivers.QuantumDesign import DynaCool
from qcodes.instrument_drivers.signal_hound.SignalHound_USB_SA124B import (
    SignalHoundUSBSA124B,
)


@pytest.mark.parametrize(
    "instrument_class",
    [IPInstrument, KeysightM960x, KeysightM9336A, DynaCool, SignalHoundUSBSA124B],
)
def test_close_instrument_without_attributes(
    instrument_class: type[Instrument],
) -> None:
    instrument = instrument_class.__new__(instrument_class)

    instrument.close()
    instrument.close()
