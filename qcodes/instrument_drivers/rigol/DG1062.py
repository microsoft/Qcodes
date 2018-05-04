"""
Module for the Rigol DG1062 driver. We have only implemented:
1) Setting/getting waveforms
2) Setting/getting the impedance
3) Setting/getting the polarity
"""

import logging
from functools import partial
from typing import Union, List, Dict

from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList

log = logging.getLogger(__name__)


class DG1062Channel(InstrumentChannel):
    def __init__(self, parent: 'DG1062', name: str, channel: int) ->None:
        """
        Args:
            parent: The instrument this channel belongs to
            name (str)
            channel (int)
        """

        super().__init__(parent, name)

        self.parent = parent
        self.channel = channel

        # All waveforms except 'DC' and 'ARB' have these parameters
        default_wave_params = ["freq", "ampl", "offset", "phase"]

        self.waveform_params = {
            waveform: default_wave_params for waveform in
            ["HARM", "NOIS", "RAMP", "SIN", "SQU", "TRI", "USER"]
        }

        self.waveform_params["DC"] = ["freq", "ampl", "offset"]
        self.waveform_params["ARB"] = ["sample_rate", "ampl", "offset"]

        for param, unit in [
            ("freq", "Hz"),
            ("ampl", "V"),
            ("offset", "V"),
            ("phase", "deg"),
            ("sample_rate", "1/s")
        ]:
            self.add_parameter(
                param,
                unit=unit,
                set_cmd=partial(self._set_waveform_param, param),
                get_cmd=partial(self._get_waveform_param, param)
            )

        self.add_parameter(
            "impedance",
            get_cmd=f":OUTPUT{channel}:IMP?",
            set_cmd=f":OUTPUT{channel}:IMP {{}}",
            unit="Ohm",
            vals=vals.Ints(min_value=1, max_value=10000)
        )

        self.add_parameter(
            "polarity",
            get_cmd=f":OUTPUT{channel}:GAT:POL?",
            set_cmd=f":OUTPUT{channel}:GAT:POL {{}}",
            vals=vals.OnOff(),
            val_mapping={1: 'POSITIVE', 0: 'NEGATIVE'},
        )

    def apply(self, **kwargs: Dict) ->Union[None, Dict]:
        """
        Apply a waveform on the channel

        Example:
        >>> gd = DG1062("gd", "TCPIP0::169.254.187.99::inst0::INSTR")
        >>> gd.channels[0].apply(waveform="SIN", freq=1E3, ampl=1.0, offset=0, phase=0)

        If not kwargs are given a dictionary with the current waveform
        parameters are returned.
        """
        if len(kwargs) == 0:
            return self._get_waveform_params()

        self._set_waveform_params(**kwargs)
        return {}

    def _get_waveform_param(self, param: str) ->float:
        """
        Get a parameter of the current waveform. Valid param names are
        dependent on the waveform type (e.g. "DC" does not have a "phase")
        """
        params_dict = self._get_waveform_params()
        return params_dict.get(param, None)

    def _get_waveform_params(self) ->Dict:
        """
        Get all the parameters of the current waveform and
        """
        def to_float(string):
            try:
                return float(string)
            except ValueError:
                return string

        waveform_str = self.parent.ask_raw(f":SOUR{self.channel}:APPL?")
        parts = waveform_str.strip("\"").split(",")

        current_waveform = parts[0]
        param_vals = [current_waveform] + [to_float(i) for i in parts[1:]]
        param_names = ["waveform"] + self.waveform_params[current_waveform]
        params_dict = dict(zip(param_names, param_vals))

        return params_dict

    def _set_waveform_param(self, param: str, value: float) ->None:
        """
        Set a particular waveform param to the given value.
        """
        params_dict = self._get_waveform_params()

        if param in params_dict:
            params_dict[param] = value
        else:
            log.warning(f"Warning, unable to set '{param}' for the current "
                        f"waveform")
            return

        return self._set_waveform_params(**params_dict)

    def _set_waveform_params(self, **params_dict: Dict) ->None:
        """
        Apply a waveform with values given in a dictionary.
        """
        if not "waveform" in params_dict:
            raise ValueError("At least 'waveform' argument needed")

        waveform = str(params_dict["waveform"])
        if waveform not in self.waveform_params:
            raise ValueError(f"Unknown waveform '{waveform}'. Options are "
                             f"{self.waveform_params.keys()}")

        param_names = self.waveform_params[waveform]

        if not set(param_names).issubset(params_dict.keys()):
            raise ValueError(f"Waveform {waveform} needs at least parameters "
                             f"{param_names}")

        string = f":SOUR{self.channel}:APPL:{waveform} "
        string += ",".join(
            ["{:7e}".format(params_dict[param]) for param in param_names])
        self.parent.write_raw(string)


class DG1062(VisaInstrument):
    def __init__(self, name: str, address: str, *args: List,
                 **kwargs: Dict) ->None:

        super().__init__(name, address, *args, **kwargs)

        channels = ChannelList(self, "channel", DG1062Channel,
                               snapshotable=True)

        for ch_num in [1, 2]:
            ch_name = "ch{}".format(ch_num)
            channel = DG1062Channel(self, ch_name, ch_num)
            channels.append(channel)
            self.add_submodule(ch_name, channel)

        channels.lock()
        self.add_submodule("channels", channels)

        self.connect_message()
