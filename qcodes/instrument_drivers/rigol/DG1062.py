import logging
from functools import partial
from typing import Union, Dict

from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList

log = logging.getLogger(__name__)


class DG1062Channel(InstrumentChannel):

    min_impedance = 1
    max_impedance = 10000

    def __init__(self, parent: 'DG1062', name: str, channel: int) ->None:
        """
        Args:
            parent: The instrument this channel belongs to
            name (str)
            channel (int)
        """

        super().__init__(parent, name)
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
            "waveform",
            get_cmd=partial(self._get_waveform_param, "waveform")
            # We set the waveform by calling the 'apply' method with
            # appropriate parameters
        )

        self.add_parameter(
            "impedance",
            get_cmd=f":OUTPUT{channel}:IMP?",
            set_cmd=f":OUTPUT{channel}:IMP {{}}",
            unit="Ohm",
            vals=vals.MultiType(
                vals.Ints(
                    min_value=DG1062Channel.min_impedance,
                    max_value=DG1062Channel.max_impedance
                ),
                vals.Enum("INF", "MIN", "MAX", "HighZ")
            ),
            set_parser=lambda value: "INF" if value == "HighZ" else value,
            get_parser=lambda value: "HighZ"
            if float(value) > DG1062Channel.max_impedance else float(value)
        )

        self.add_parameter(
            "sync",
            get_cmd=f":OUTPUT{channel}:SYNC?",
            set_cmd=f"OUTPUT{channel}:SYNC {{}}",
            vals=vals.Enum(0, 1, "ON", "OFF"),
        )

        self.add_parameter(
            "polarity",
            get_cmd=f":OUTPUT{channel}:GAT:POL?",
            set_cmd=f":OUTPUT{channel}:GAT:POL {{}}",
            vals=vals.OnOff(),
            val_mapping={1: 'POSITIVE', 0: 'NEGATIVE'},
        )

        self.add_parameter(
            "state",
            set_cmd=f"OUTPUT{channel}:STATE {{}}",
            get_cmd=f"OUTPUT{channel}:STATE?",
        )

    def __getattr__(self, item):
        """
        We want to be able to call waveform functions like so
        >>> gd = DG1062("gd", "TCPIP0::169.254.187.99::inst0::INSTR")
        >>> help(gd.channels[0].sin)
        >>> gd.channels[0].sin(freq=2E3, ampl=1.0, offset=0, phase=0)
        """
        waveform = item.upper()
        if waveform not in self.waveform_params:
            return super().__getitem__(item)

        def f(**kwargs):
            return partial(self.apply, waveform=waveform)(**kwargs)

        # By wrapping the partial in a function we can change the docstring
        f.__doc__ = f"Arguments: {self.waveform_params[waveform]}"

        return f

    def apply(self, **kwargs: Dict) ->Union[None, Dict]:
        """
        Apply a waveform on the channel

        Example:
        >>> gd = DG1062("gd", "TCPIP0::169.254.187.99::inst0::INSTR")
        >>> gd.channels[0].apply(waveform="SIN", freq=1E3, ampl=1.0, offset=0, phase=0)

        Valid waveforms are: HARM, NOIS, RAMP, SIN, SQU, TRI, USER, DC, ARB
        To find the correct arguments of each waveform we can e.g. do:
        >>> help(gd.channels[0].sin)
        Notice the lower case when accessing the waveform through convenience
        functions.

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
    def __init__(self, name: str, address: str,
                 **kwargs: Dict) ->None:

        super().__init__(name, address, terminator="\n", **kwargs)

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
