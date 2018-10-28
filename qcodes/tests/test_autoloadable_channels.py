from typing import List

from qcodes import Instrument
from qcodes.instrument.channel import AutoLoadableInstrumentChannel


class SimpleTestChannel(AutoLoadableInstrumentChannel):
    @classmethod
    def _discover_from_instrument(
            cls, parent: Instrument, **kwargs) -> List[dict]:

        kwarg_list = [{"name": f"channel{i}", "channel": i} for i in range(3)]
        return kwarg_list

