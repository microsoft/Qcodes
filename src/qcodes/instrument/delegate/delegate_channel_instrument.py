from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .delegate_instrument import DelegateInstrument

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from qcodes.station import Station


class DelegateChannelInstrument(DelegateInstrument):
    """
    Delegate instrument that auto generates delegate parameters for a given
    ChannelList.

    Example usage in instrument YAML:

    .. code-block:: yaml

        switch:
            type: qcodes.instrument.delegate.DelegateChannelInstrument
            init:
                channels: dac.channels
                parameters:
                    state:
                    - dac_output
                    - smc
                    - gnd
                    - bus

    The above will create a new instrument called ``switch`` that generates a
    method for a delegate parameter:

    .. code-block:: python

        switch.state()

    that returns a named tuple:

    .. code-block:: python

        state(dac_output=..., smc=..., gnd=..., bus=...)

    where the values of each of the tuple items are delegated to the
    instrument parameters:

    .. code-block:: python

        dac.dac_output()
        dac.smc()
        dac.gnd()
        dac.bus()

    Args:
        name: Instrument name
        station: Station with real instruments to connect to
        channels: Path to channels, e.g. my_instrument.channels
        parameters: A mapping from name of a delegate parameter to the sequence
            of endpoint parameters it connects  to. These are auto-generated per
            channel.
        initial_values: Default values to set on instrument load. Defaults
            to None.
        set_initial_values_on_load: Flag to set defaults on load. Defaults
            to False.
    """
    def __init__(
        self,
        name: str,
        station: Station,
        channels: str,
        parameters: Mapping[str, Sequence[str]],
        initial_values: Mapping[str, Any] | None = None,
        set_initial_values_on_load: bool = False,
        **kwargs: Any):
        _channels = self.parse_instrument_path(parent=station, path=channels)
        _parameters = {}
        for channel in _channels:
            ins = channel.root_instrument.name
            chan_no = str(channel.channel_number()).zfill(2)
            for alias, paths in parameters.items():
                _paths = [
                    f"{ins}.ch{chan_no}.{path}" for path in paths
                ]
                _parameters[f"{alias}{chan_no}"] = _paths

        super().__init__(
            name=name,
            station=station,
            parameters=_parameters,
            initial_values=initial_values,
            set_initial_values_on_load=set_initial_values_on_load,
            **kwargs
        )
