from typing import Optional, TYPE_CHECKING

from .KeysightB1500_module import B1500Module
from .constants import ModuleKind, ChNr
if TYPE_CHECKING:
    from .KeysightB1500 import KeysightB1500


class B1530A(B1500Module):
    INSTRUMENT_CLASS = ModuleKind.WGFMU

    def __init__(self, parent: 'KeysightB1500', name: Optional[str], slot_nr,
                 **kwargs):
        super().__init__(parent, name, slot_nr, **kwargs)

        self.channels = (ChNr(slot_nr), ChNr(int(f"{slot_nr:d}02")))
