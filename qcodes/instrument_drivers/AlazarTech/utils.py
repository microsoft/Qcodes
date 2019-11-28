"""
The module provides useful utility objects for the Alazar driver class.
The purpose of these is more to provide relevant functionality and less to
improve the code structure (which is more the purpose of the
:mod:`.AlazarTech.helpers` module).
"""


from qcodes.instrument.parameter import Parameter


class TraceParameter(Parameter):
    """
    A parameter that keeps track of if its value has been synced to
    the ``Instrument``. To achieve that, this parameter sets
    the ``_parameters_synced`` attribute of the ``Instrument`` to ``False``
    when the value of the parameter is changed.

    This parameter is useful for the Alazar card driver. Syncing parameters to
    an Alazar card is relatively slow, hence it makes sense to first set the
    values of the parameters, and then "synchronize them to the card".
    """
    def __init__(self, *args, **kwargs):
        self._synced_to_card = False
        super().__init__(*args, **kwargs)

    def _set_updated(self):
        self._synced_to_card = True

    @property
    def synced_to_card(self) -> bool:
        """True if the parameter value has been synced to the instrument"""
        return self._synced_to_card

    def set_raw(self, value):
        self._instrument._parameters_synced = False
        self._synced_to_card = False
