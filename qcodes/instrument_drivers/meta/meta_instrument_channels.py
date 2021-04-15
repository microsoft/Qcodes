from typing import Dict, List, Any
from qcodes.station import Station
from qcodes.instrument_drivers.meta.meta_instrument import MetaInstrument


class MetaInstrumentWithChannels(MetaInstrument):
    """Meta instrument that auto generates aliases for a given ChannelList"""
    def __init__(
        self,
        name: str,
        station: Station,
        channels: str,
        aliases: Dict[str, List[str]],
        default_values: Dict[str, Any] = None,
        set_defaults_on_load: bool = False,
        **kwargs):
        """Create a Meta instrument with auto-generated channels

        Args:
            name (str): Instrument name
            station (Station): Station with real instruments to connect to
            channels (str): Path to channels, e.g. my_instrument.channels
            aliases (Dict[str, List[str]]): Aliases to specify for instrument,
                these are auto-generated per channel
            default_values (Dict[str, Any], optional): Default values to set on
                instrument load. Defaults to None.
            set_defaults_on_load (bool, optional): Flag to set defaults on load.
                Defaults to False.
        """
        _channels = self.parse_instrument_path(station=station, path=channels)
        _aliases = {}
        for channel in _channels:
            ins = channel.root_instrument.name
            chan_no = str(channel.channel_number()).zfill(2)
            for alias, paths in aliases.items():
                _paths = [
                    f"{ins}.ch{chan_no}.{path}" for path in paths
                ]
                _aliases[f"{alias}{chan_no}"] = _paths

        super().__init__(
            name=name,
            station=station,
            aliases=_aliases,
            default_values=default_values,
            set_defaults_on_load=set_defaults_on_load,
            **kwargs
        )
