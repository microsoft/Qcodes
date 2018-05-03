"""
Module for the Rigol DG1062 driver. We have only implemented:
1) Setting/getting waveforms
2) Setting/getting the impedance
3) Setting/getting the polarity
"""

import logging
from functools import partial

from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList

log = logging.getLogger(__name__)


class GD1062Channel(InstrumentChannel):
    def __init__(self, parent: 'GD1062', name: str, channel: int):
        """
        Args:
            parent: The instrument this channel belongs to
            name (str)
            channel (int)
        """

        super().__init__(parent, name)

        self.parent = parent
        self.channel = channel

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

    def apply(self, **kwargs):
        if len(kwargs) == 0:
            return self._get_waveform_params()

        self._set_waveform_params(**kwargs)

    def _get_waveform_param(self, param):
        params_dict = self._get_waveform_params()
        return params_dict.get(param, None)

    def _get_waveform_params(self):

        waveform_str = self.parent.ask_raw(f":SOUR{self.channel}:APPL?")
        parts = waveform_str.strip("\"").split(",")

        current_waveform = parts[0]
        param_vals = [current_waveform] + [float(i) for i in parts[1:]]
        param_names = ["waveform"] + self.waveform_params[current_waveform]
        params_dict = dict(zip(param_names, param_vals))

        return params_dict

    def _set_waveform_param(self, param, value):
        params_dict = self._get_waveform_params()

        if param in params_dict:
            params_dict[param] = value
        else:
            log.warning(f"Warning, unable to set '{param}' for the current "
                        f"waveform")

        self._set_waveform_params(**params_dict)

    def _set_waveform_params(self, **params_dict):

        if not "waveform" in params_dict:
            raise ValueError("At least 'waveform' argument needed")

        waveform = params_dict["waveform"]
        if waveform not in self.waveform_params:
            raise ValueError("Unknown waveform '{waveform}'")

        param_names = self.waveform_params[waveform]

        string = f":SOUR{self.channel}:APPL:{waveform} "
        string += ",".join(
            ["{:7e}".format(params_dict[param]) for param in param_names])
        self.parent.write_raw(string)


class GD1062(VisaInstrument):
    def __init__(self, name, address, *args, **kwargs):
        super().__init__(name, address, terminator='\n', *args, **kwargs)

        channels = ChannelList(self, "channel", GD1062Channel,
                               snapshotable=True)

        for ch_num in [1, 2]:
            ch_name = "ch{}".format(ch_num)
            channel = GD1062Channel(self, ch_name, ch_num)
            channels.append(channel)
            self.add_submodule(ch_name, channel)

        channels.lock()
        self.add_submodule("channels", channels)

        self.connect_message()
