import logging
from functools import partial
from typing import Union, Dict

from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList

log = logging.getLogger(__name__)


def sane_partial(func, docstring, **kwargs):
    """
    We want to have a partial function which will allow us access the docstring
    through the python built-in help function. This is particularly important
    for client-facing driver methods, whose arguments might not be obvious.

    Consider the follow example why this is needed:

    >>> from functools import partial
    >>> def f():
    >>> ... pass
    >>> g = partial(f)
    >>> g.__doc__ = "bla"
    >>> help(g) # this will print an unhelpful message

    Args:
        func (callable)
        docstring (str)
    """
    ex = partial(func, **kwargs)

    def inner(**inner_kwargs):
        ex(**inner_kwargs)

    inner.__doc__ = docstring

    return inner


class DG1062Burst(InstrumentChannel):
    """
    Burst commands for the DG1062. We make a separate channel for these to
    group burst commands together.
    """

    def __init__(self, parent: 'DG1062', name: str, channel: int) ->None:
        super().__init__(parent, name)
        self.channel = channel

        self.add_parameter(
            "on",
            get_cmd=f":SOUR{channel}:BURS?",
            set_cmd=f":SOUR{channel}:BURS {{}}",
            vals=vals.Enum(0, 1, "ON", "OFF")
        )

        self.add_parameter(
            "polarity",
            get_cmd=f":SOUR{channel}:BURS:GATE:POL?",
            set_cmd=f":SOUR{channel}:BURS:GATE:POL {{}}",
            vals=vals.Enum("NOR", "INV")
        )

        self.add_parameter(
            "period",
            get_cmd=f":SOUR{channel}:BURS:INT:PER ?",
            set_cmd=f":SOUR{channel}:BURS:INT:PER {{}}",
            vals=vals.MultiType(
                vals.Numbers(min_value=3E-6, max_value=500),
                vals.Enum("MIN", "MAX")
            )
        )

        self.add_parameter(
            "mode",
            get_cmd=f":SOUR{channel}:BURS:MODE?",
            set_cmd=f":SOUR{channel}:BURS:MODE {{}}",
            vals=vals.Enum("TRIG","INF", "GAT")
        )

        self.add_parameter(
            "ncycles",
            get_cmd=f":SOUR{channel}:BURS:NCYC?",
            set_cmd=f":SOUR{channel}:BURS:NCYC {{}}",
            vals=vals.Numbers(min_value=1, max_value=500000)
        )

        self.add_parameter(
            "phase",
            get_cmd=f":SOUR{channel}:BURS:PHAS?",
            set_cmd=f":SOUR{channel}:BURS:PHAS {{}}",
            vals=vals.Numbers(min_value=0, max_value=360)
        )

        self.add_parameter(
            "time_delay",
            get_cmd=f":SOUR{channel}:BURS:TDEL?",
            set_cmd=f":SOUR{channel}:BURS:TDEL {{}}",
            vals=vals.Numbers(min_value=0)
        )

        self.add_parameter(
            "trigger_slope",
            get_cmd=f":SOUR{channel}:BURS:TRIG:SLOP?",
            set_cmd=f":SOUR{channel}:BURS:TRIG:SLOP {{}}",
            vals=vals.Enum("POS", "NEG")
        )

        self.add_parameter(
            "source",
            get_cmd=f":SOUR{channel}:BURS:TRIG:SOUR?",
            set_cmd=f":SOUR{channel}:BURS:TRIG:SOUR {{}}",
            vals=vals.Enum("INT", "EXT", "MAN")
        )

        self.add_parameter(
            "idle",
            get_cmd=f":SOUR{channel}:BURST:IDLE?",
            set_cmd=f":SOUR{channel}:BURST:IDLE {{}}",
            vals=vals.MultiType(
                vals.Enum("FPT", "TOP", "BOTTOM", "CENTER"),
                vals.Numbers()  # DIY
            )
        )

    def trigger(self) ->None:
        """
        Send a soft trigger to the instrument. This only works if the trigger
        source is set to manual.
        """
        self.parent.write_raw(f":SOUR{self.channel}:BURS:TRIG")


class DG1062Channel(InstrumentChannel):

    min_impedance = 1
    max_impedance = 10000

    waveform_params = {
        waveform: ["freq", "ampl", "offset", "phase"] for waveform in
        ["HARM", "NOIS", "RAMP", "SIN", "SQU", "TRI", "USER"]
    }

    waveform_params["DC"] = ["freq", "ampl", "offset"]
    waveform_params["ARB"] = ["sample_rate", "ampl", "offset"]

    waveforms = list(waveform_params.keys())

    def __init__(self, parent: 'DG1062', name: str, channel: int) ->None:
        """
        Args:
            parent: The instrument this channel belongs to
            name (str)
            channel (int)
        """

        super().__init__(parent, name)
        self.channel = channel

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
                get_cmd=partial(self._get_waveform_param, param),
            )

        self.add_parameter(
            "waveform",
            get_cmd=partial(self._get_waveform_param, "waveform")
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

        burst = DG1062Burst(self.parent, "burst", self.channel)
        self.add_submodule("burst", burst)

        # We want to be able to do the following:
        # >>> help(gd.channels[0].sin)
        # >>> gd.channels[0].sin(freq=2E3, ampl=1.0, offset=0, phase=0)
        # We do not use add_function as it is more cumbersome to use.
        for waveform in self.waveforms:
            f = sane_partial(
                self.apply,
                docstring="Args: " + ", ".join(self.waveform_params[waveform]),
                waveform=waveform
            )

            setattr(self, waveform.lower(), f)

    def apply(self, **kwargs: Dict) ->Union[None, Dict]:
        """
        Apply a waveform on the channel

        Example:
        >>> gd = DG1062("gd", "TCPIP0::169.254.187.99::inst0::INSTR")
        >>> gd.channels[0].apply(waveform="SIN", freq=1E3, ampl=1.0, offset=0, phase=0)

        Valid waveforms are: HARM, NOIS, RAMP, SIN, SQU, TRI, USER, DC, ARB
        To find the correct arguments of each waveform we can e.g. do:
        >>> print(gd.channels[0].sin.__doc__)
        Notice the lower case when accessing the waveform through convenience
        functions.

        If not kwargs are given a dictionary with the current waveform
        parameters are returned.
        """
        if len(kwargs) == 0:
            return self._get_waveform_params()

        self._set_waveform_params(**kwargs)

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
    """
    Instrument driver for the Rigol DG1062
    """

    waveforms = DG1062Channel.waveforms

    def __init__(self, name: str, address: str,
                 **kwargs: Dict) ->None:

        super().__init__(name, address, terminator="\n", **kwargs)

        channels = ChannelList(self, "channel", DG1062Channel,
                               snapshotable=False)

        for ch_num in [1, 2]:
            ch_name = "ch{}".format(ch_num)
            channel = DG1062Channel(self, ch_name, ch_num)
            channels.append(channel)
            self.add_submodule(ch_name, channel)

        channels.lock()
        self.add_submodule("channels", channels)
        self.connect_message()
