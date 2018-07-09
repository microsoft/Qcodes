import warnings

from qcodes.instrument.parameter import Parameter
from qcodes.utils import validators

class TraceParameter(Parameter):
    """
    A parameter that keeps track of if its value has been synced to
    the Instrument by setting a flag when changed.
    """
    def __init__(self, *args, **kwargs):
        self._synced_to_card = False
        super().__init__(*args, **kwargs)

    def _set_updated(self):
        self._synced_to_card = True

    @property
    def synced_to_card(self) -> bool:
        return self._synced_to_card

    def set_raw(self, value):
        self._instrument._parameters_synced = False
        self._synced_to_card = False
        self._save_val(value, validate=False)
