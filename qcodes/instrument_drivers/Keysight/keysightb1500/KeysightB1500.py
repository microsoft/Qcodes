from typing import Dict, Tuple, Union, Optional, Any
from collections import defaultdict
import re

from qcodes import VisaInstrument, InstrumentChannel

from . import constants
from .constants import (
    ChNr,
    SlotNr,
    InstrClass,
    ChannelList,
)
from .message_builder import MessageBuilder


# TODO notes:
# - [ ] Instead of generating a Qcodes InstrumentChannel for each **module**,
#   it might make more sense to generate one for each **channel**


class B1500Module(InstrumentChannel):
    INSTRUMENT_CLASS: InstrClass

    def __init__(self, parent: 'KeysightB1500', name: Optional[str], slot_nr,
                 **kwargs):
        self.channels: Tuple  # self.channels will be populated in the concrete
        # module subclasses because channel count is module specific
        self.slot_nr = SlotNr(slot_nr)

        if name is None:
            number = len(parent.by_class[self.INSTRUMENT_CLASS]) + 1
            name = self.INSTRUMENT_CLASS.lower() + str(number)
        super().__init__(parent=parent, name=name, **kwargs)

    def enable_outputs(self):
        """
        Enables all outputs of this module by closing the output relays of its
        channels.
        """
        # TODO This always enables all outputs of a module, which is maybe not
        # desirable. (Also check the TODO item at the top about
        # InstrumentChannel per Channel instead of per Module.
        msg = MessageBuilder().cn(self.channels).message
        self.write(msg)

    def disable_outputs(self):
        """
        Disables all outputs of this module by opening the output relays of its
         channels.
        """
        # TODO See enable_output TODO item
        msg = MessageBuilder().cl(self.channels).message
        self.write(msg)

    def is_enabled(self) -> bool:
        """
        Check if channels of this module are enabled.

        :return: `True` if *all* channels of this module are enabled. `False`
         otherwise.
        """
        # TODO If a module has multiple channels, and only one is enabled, then
        # this will return false, which is probably not desirable.
        # Also check the TODO item at the top about InstrumentChannel per
        # Channel instead of per Module.
        msg = (MessageBuilder()
               .lrn_query(constants.LRN.Type.OUTPUT_SWITCH)
               .message
               )
        response = self.ask(msg)
        activated_channels = re.sub(r"[^,\d]", "", response).split(",")

        is_enabled = set(self.channels).issubset(
            int(x) for x in activated_channels
        )
        return is_enabled


class B1517A(B1500Module):
    INSTRUMENT_CLASS = InstrClass.SMU

    def __init__(self, parent: 'KeysightB1500', name: Optional[str], slot_nr,
                 **kwargs):
        super().__init__(parent, name, slot_nr, **kwargs)

        self.channels = (ChNr(slot_nr),)

        self._measure_config: Dict[str, Optional[Any]] = {
            k: None for k in ("measure_range",)}
        self._source_config: Dict[str, Optional[Any]] = {
            k: None for k in ("output_range", "compliance",
                              "compl_polarity", "min_compliance_range")}

        self.add_parameter(
            name="voltage",
            set_cmd=self._set_voltage,
            get_cmd=self._get_voltage
        )

        self.add_parameter(
            name="current",
            set_cmd=self._set_current,
            get_cmd=self._get_current
        )

    def _set_voltage(self, value):
        if self._source_config["output_range"] is None:
            self._source_config["output_range"] = constants.VOutputRange.AUTO
        if not isinstance(self._source_config["output_range"],
                          constants.VOutputRange):
            raise TypeError(
                "Asking to force current, but source_config contains a "
                "voltage output range"
            )
        msg = MessageBuilder().dv(
            chnum=self.channels[0],
            v_range=self._source_config["output_range"],
            voltage=value,
            i_comp=self._source_config["compliance"],
            comp_polarity=self._source_config["compl_polarity"],
            i_range=self._source_config["min_compliance_range"],
        )
        self.write(msg.message)

    def _set_current(self, value):
        if self._source_config["output_range"] is None:
            self._source_config["output_range"] = constants.IOutputRange.AUTO
        if not isinstance(self._source_config["output_range"],
                          constants.IOutputRange):
            raise TypeError(
                "Asking to force current, but source_config contains a "
                "voltage output range"
            )
        msg = MessageBuilder().di(
            chnum=self.channels[0],
            i_range=self._source_config["output_range"],
            current=value,
            v_comp=self._source_config["compliance"],
            comp_polarity=self._source_config["compl_polarity"],
            v_range=self._source_config["min_compliance_range"],
        )
        self.write(msg.message)

    def _get_current(self):
        try:
            msg = MessageBuilder().ti(
                chnum=self.channels[0],
                i_range=self._measure_config["measure_range"],
            )
            response = self.ask(msg.message)

            parsed = KeysightB1500.parse_spot_measurement_response(response)
            return parsed["value"]
        except AttributeError:
            raise ValueError(
                "Measurement range unconfigured. Call B1517A.measure_config() "
                "before using measure commands."
            )

    def _get_voltage(self):
        try:
            msg = MessageBuilder().tv(
                chnum=self.channels[0],
                v_range=self._measure_config["measure_range"],
            )
            response = self.ask(msg.message)

            parsed = KeysightB1500.parse_spot_measurement_response(response)
            return parsed["value"]

        except AttributeError:
            raise ValueError(
                "Measurement range unconfigured. Call B1517A.measure_config() "
                "before using measure commands."
            )

    def source_config(
            self,
            output_range: constants.OutputRange,
            compliance: Optional[Union[float, int]] = None,
            compl_polarity: Optional[constants.CompliancePolarityMode] = None,
            min_compliance_range: Optional[constants.OutputRange] = None,
    ):
        if min_compliance_range is not None:
            if isinstance(min_compliance_range, type(output_range)):
                raise TypeError(
                    "When forcing voltage, min_compliance_range must be an "
                    "current output range (and vice versa)."
                )

        self._source_config = {
            "output_range": output_range,
            "compliance": compliance,
            "compl_polarity": compl_polarity,
            "min_compliance_range": min_compliance_range,
        }

    def measure_config(self, measure_range: constants.MeasureRange):
        self._measure_config = {"measure_range": measure_range}


class B1520A(B1500Module):
    INSTRUMENT_CLASS = InstrClass.CMU

    def __init__(self, parent: 'KeysightB1500', name: Optional[str], slot_nr,
                 **kwargs):
        super().__init__(parent, name, slot_nr, **kwargs)

        self.channels = (ChNr(slot_nr),)

        self.add_parameter(
            name="voltage_dc", set_cmd=self._set_voltage_dc, get_cmd=None
        )

        self.add_parameter(
            name="voltage_ac", set_cmd=self._set_voltage_ac, get_cmd=None
        )

        self.add_parameter(
            name="frequency", set_cmd=self._set_frequency, get_cmd=None
        )

        self.add_parameter(name="capacitance", get_cmd=self._get_capacitance)

    def _set_voltage_dc(self, value):
        msg = MessageBuilder().dcv(self.channels[0], value)

        self.write(msg.message)

    def _set_voltage_ac(self, value):
        msg = MessageBuilder().acv(self.channels[0], value)

        self.write(msg.message)

    def _set_frequency(self, value):
        msg = MessageBuilder().fc(self.channels[0], value)

        self.write(msg.message)

    def _set_mode(self, mode):
        """
        mode is capacitance (CpG) or impedance (RX, Z complex)

        :param mode:
        :return:
        """
        pass

    def _get_capacitance(self):
        pattern = re.compile(
            r"((?P<status>\w)(?P<chnr>\w)(?P<dtype>\w))?"
            r"(?P<value>[+-]\d{1,3}\.\d{3,6}E[+-]\d{2})"
        )
        msg = MessageBuilder().tc(
            chnum=self.channels[0], mode=constants.RangingMode.AUTO
        )

        response = self.ask(msg.message)

        parsed = [item for item in re.finditer(pattern, response)]

        if (
                len(parsed) != 2
                or parsed[0]["dtype"] != "C"
                or parsed[1]["dtype"] != "Y"
        ):
            raise ValueError("Result format not supported.")

        return float(parsed[0]["value"]), float(parsed[1]["value"])


class B1530A(B1500Module):
    INSTRUMENT_CLASS = InstrClass.AUX

    def __init__(self, parent: 'KeysightB1500', name: Optional[str], slot_nr,
                 **kwargs):
        super().__init__(parent, name, slot_nr, **kwargs)

        self.channels = (ChNr(slot_nr), ChNr(int(f"{slot_nr:d}02")))


class KeysightB1500(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator="\r\n", **kwargs)

        self.by_slot = {}
        self.by_channel = {}
        self.by_class = defaultdict(list)

        self._find_modules()

    def add_module(self, name: str, module: B1500Module):
        super().add_submodule(name, module)

        self.by_class[module.INSTRUMENT_CLASS].append(module)
        self.by_slot[module.slot_nr] = module
        for ch in module.channels:
            self.by_channel[ch] = module

    def reset(self):
        """Performs an instrument reset.

        Does not reset error queue!
        """
        self.write("*RST")

    def get_status(self) -> int:
        return int(self.ask("*STB?"))

    # TODO: Data Output parser: At least for Format FMT1,0 and maybe for a
    # second (binary) format. 8 byte binary format would be nice because it
    # comes with time stamp
    # FMT1,0: ASCII (12 digits data with header) <CR/LF^EOI>

    def _find_modules(self):
        from .constants import UNT

        r = self.ask(MessageBuilder()
                     .unt_query(mode=UNT.Mode.MODULE_INFO_ONLY)
                     .message
                     )

        slot_population = parse_module_query_response(r)

        for slot_nr, model in slot_population.items():
            module = self.from_model_name(model, slot_nr, self)

            self.add_module(name=module.short_name, module=module)

    @staticmethod
    def from_model_name(model: str, slot_nr: int, parent: 'KeysightB1500',
                        name: Optional[str] = None) -> 'B1500Module':
        """
        Creates the correct instance type for instrument by model name.

        :param model: Model name such as 'B1517A'
        :param slot_nr: Slot number of this module (not channel numeber)
        :param parent: Reference to B1500 mainframe instance
        :param name: If `None` (Default) then the name is autogenerated from
            the instrument class.

        :return: A specific instance of `B1500Module`
        """
        if model == "B1517A":
            return B1517A(slot_nr=slot_nr, parent=parent, name=name)
        elif model == "B1520A":
            return B1520A(slot_nr=slot_nr, parent=parent, name=name)
        elif model == "B1530A":
            return B1530A(slot_nr=slot_nr, parent=parent, name=name)
        else:
            raise NotImplementedError("Module type not yet supported.")

    def enable_channels(self, channels: ChannelList = None):
        """
        Enables specified channels. If channels is omitted or `None`, all
        channels are enabled.
        """
        msg = MessageBuilder().cn(channels)

        self.write(msg.message)

    def disable_channels(self, channels: ChannelList = None):
        """
        Disables specified channels. If channels is omitted or `None`, all
        channels are disabled.
        """
        msg = MessageBuilder().cl(channels)

        self.write(msg.message)

    @staticmethod
    def parse_spot_measurement_response(response) -> dict:
        pattern = re.compile(
            r"((?P<status>\w)(?P<chnr>\w)(?P<dtype>\w))?"
            r"(?P<value>[+-]\d{1,3}\.\d{3,6}E[+-]\d{2})"
        )

        match = re.match(pattern, response)
        if match is None:
            raise ValueError(f"{response!r} didn't match {pattern!r} pattern")

        d: Dict[str, Union[str, float]] = match.groupdict()
        d["value"] = float(d["value"])

        return d


def parse_module_query_response(response: str) -> Dict[SlotNr, str]:
    """
    Extract installed module info from str and return it as a dict.

    :param response: Response str to `UNT? 0` query.
    :return: Dict[SlotNr: model_name_str]
    """
    pattern = r";?(?P<model>\w+),(?P<revision>\d+)"

    moduleinfo = re.findall(pattern, response)

    return {
        SlotNr(slot_nr): model
        for slot_nr, (model, rev) in enumerate(moduleinfo, start=1)
        if model != "0"
    }
