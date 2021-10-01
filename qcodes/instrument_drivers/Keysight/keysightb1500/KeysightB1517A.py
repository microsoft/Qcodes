import re
import textwrap
from typing import Optional, Dict, Any, Union, TYPE_CHECKING, List, Tuple, \
    cast, Sequence
from typing_extensions import TypedDict, Literal, overload
import numpy as np
import qcodes.utils.validators as vals
from qcodes.instrument.parameter import Parameter, ParamRawDataType
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument.group_parameter import GroupParameter, Group
from qcodes.utils.validators import Arrays
from qcodes.utils.deprecate import deprecate

from .KeysightB1500_sampling_measurement import SamplingMeasurement
from .KeysightB1500_module import B1500Module, \
    parse_spot_measurement_response
from .message_builder import MessageBuilder
from . import constants
from .constants import ModuleKind, ChNr, AAD, MM, MeasurementStatus, \
    VMeasRange, IMeasRange, VOutputRange, IOutputRange

if TYPE_CHECKING:
    from .KeysightB1500_base import KeysightB1500


class SweepSteps(TypedDict, total=False):
    """
    A dictionary holding all the parameters that specifies the staircase
    sweep (WV).
    """
    chan: Union[int, constants.ChNr]
    sweep_mode: Union[constants.SweepMode, int]
    sweep_range: Union[constants.VOutputRange, int]
    sweep_start: float
    sweep_end: float
    sweep_steps: int
    current_compliance: Optional[float]
    power_compliance: Optional[float]


class IVSweeper(InstrumentChannel):
    def __init__(self, parent: 'B1517A',
                 name: str, **kwargs: Any):
        super().__init__(parent, name, **kwargs)
        self._sweep_step_parameters: SweepSteps = \
            {"sweep_mode": constants.SweepMode.LINEAR,
             "sweep_range": constants.VOutputRange.AUTO,
             "sweep_start": 0.0,
             "sweep_end": 0.0,
             "sweep_steps": 1,
             "current_compliance": None,
             "power_compliance": None}

        self.add_parameter(name='sweep_auto_abort',
                           set_cmd=self._set_sweep_auto_abort,
                           get_cmd=self._get_sweep_auto_abort,
                           set_parser=constants.Abort,
                           get_parser=constants.Abort,
                           vals=vals.Enum(*list(constants.Abort)),
                           initial_cache_value=constants.Abort.ENABLED,
                           docstring=textwrap.dedent("""
        The WM command enables or disables the automatic abort function for
        the staircase sweep sources and the pulsed sweep source. The
        automatic abort function stops the measurement when one of the
        following conditions occurs:
         - Compliance on the measurement channel
         - Compliance on the non-measurement channel
         - Overflow on the AD converter
         - Oscillation on any channel
        This command also sets the post measurement condition for the sweep
        sources. After the measurement is normally completed, the staircase
        sweep sources force the value specified by the post parameter,
        and the pulsed sweep source forces the pulse base value.

        If the measurement is stopped by the automatic abort function,
        the staircase sweep sources force the start value, and the pulsed
        sweep source forces the pulse base value after sweep.
        """))

        self.add_parameter(name='post_sweep_voltage_condition',
                           set_cmd=self._set_post_sweep_voltage_condition,
                           get_cmd=self._get_post_sweep_voltage_condition,
                           set_parser=constants.WM.Post,
                           get_parser=constants.WM.Post,
                           vals=vals.Enum(*list(constants.WM.Post)),
                           initial_cache_value=constants.WM.Post.START,
                           docstring=textwrap.dedent("""
        Source output value after the measurement is normally completed. If
        this parameter is not set, the sweep sources force the start value.
                                 """))

        self.add_parameter(name='hold_time',
                           initial_value=0.0,
                           vals=vals.Numbers(0, 655.35),
                           unit='s',
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
                           Hold time (in seconds) that is the
                           wait time after starting measurement
                           and before starting delay time for
                           the first step 0 to 655.35 s, with 10
                           ms resolution. Numeric expression.
                          """))

        self.add_parameter(name='delay',
                           initial_value=0.0,
                           vals=vals.Numbers(0, 65.535),
                           unit='s',
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
                           Delay time (in seconds) that is the wait time after
                           starting to force a step output and before
                            starting a step measurement. 0 to 65.535 s,
                            with 0.1 ms resolution. Numeric expression.
                            """))

        self.add_parameter(name='step_delay',
                           initial_value=0.0,
                           vals=vals.Numbers(0, 1),
                           unit='s',
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
                            Step delay time (in seconds) that is the wait time
                            after starting a step measurement and before
                            starting to force the next step output. 0 to 1 s,
                            with 0.1 ms resolution. Numeric expression. If
                            this parameter is not set, step delay will be 0. If
                            step delay is shorter than the measurement time,
                            the B1500 waits until the measurement completes,
                            then forces the next step output.
                            """))

        self.add_parameter(name='trigger_delay',
                           initial_value=0.0,
                           unit='s',
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
                            Step source trigger delay time (in seconds) that
                            is the wait time after completing a step output
                            setup and before sending a step output setup
                            completion trigger. 0 to the value of ``delay`` s,
                            with 0.1 ms resolution.
                            If this parameter is not set,
                            trigger delay will be 0.
                            """))

        self.add_parameter(name='measure_delay',
                           initial_value=0.0,
                           unit='s',
                           vals=vals.Numbers(0, 65.535),
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
                           Step measurement trigger delay time (in seconds)
                           that is the wait time after receiving a start step
                           measurement trigger and before starting a step
                           measurement. 0 to 65.535 s, with 0.1 ms resolution.
                           Numeric expression. If this parameter is not set,
                           measure delay will be 0.
                           """))

        self._set_sweep_delays_group = Group(
            [self.hold_time,
             self.delay,
             self.step_delay,
             self.trigger_delay,
             self.measure_delay],
            set_cmd='WT '
                    '{hold_time},'
                    '{delay},'
                    '{step_delay},'
                    '{trigger_delay},'
                    '{measure_delay}',
            get_cmd=self._get_sweep_delays(),
            get_parser=self._get_sweep_delays_parser)

        self.add_parameter(name='sweep_mode',
                           set_cmd=self._set_sweep_mode,
                           get_cmd=self._get_sweep_mode,
                           vals=vals.Enum(*list(constants.SweepMode)),
                           set_parser=constants.SweepMode,
                           snapshot_get=False,
                           docstring=textwrap.dedent("""
                 Sweep mode. Note that Only linear sweep (mode=1 or 3) is
                 available for the staircase sweep with pulsed bias.
                     1: Linear sweep (single stair, start to stop.)
                     2: Log sweep (single stair, start to stop.)
                     3: Linear sweep (double stair, start to stop to start.)
                     4: Log sweep (double stair, start to stop to start.)
                                """))

        self.add_parameter(name='sweep_range',
                           set_cmd=self._set_sweep_range,
                           get_cmd=self._get_sweep_range,
                           vals=vals.Enum(*list(constants.VOutputRange)),
                           set_parser=constants.VOutputRange,
                           snapshot_get=False,
                           docstring=textwrap.dedent("""
        Ranging type for staircase sweep voltage output. Integer expression.
        See Table 4-4 on page 20. The B1500 usually uses the minimum range
        that covers both start and stop values to force the staircase sweep
        voltage. However, if you set `power_compliance` and if the following
        formulas are true, the B1500 uses the minimum range that covers the
        output value, and changes the output range dynamically (20 V range or
        above). Range changing may cause 0 V output in a moment. For the
        limited auto ranging, the instrument never uses the range less than
        the specified range.
         - Icomp > maximum current for the output range
         - Pcomp/output voltage > maximum current for the output range
        """))

        self.add_parameter(name='sweep_start',
                           set_cmd=self._set_sweep_start,
                           get_cmd=self._get_sweep_start,
                           unit='V',
                           vals=vals.Numbers(-25, 25),
                           snapshot_get=False,
                           docstring=textwrap.dedent("""
        Start value of the stair case sweep (in V). For the log sweep,
        start and stop must have the same polarity.
                                """))

        self.add_parameter(name='sweep_end',
                           set_cmd=self._set_sweep_end,
                           get_cmd=self._get_sweep_end,
                           unit='V',
                           vals=vals.Numbers(-25, 25),
                           snapshot_get=False,
                           docstring=textwrap.dedent("""
        Stop value of the DC bias sweep (in V). For the log sweep,start and
        stop must have the same polarity.
                                """))

        self.add_parameter(name='sweep_steps',
                           set_cmd=self._set_sweep_steps,
                           get_cmd=self._get_sweep_steps,
                           vals=vals.Ints(1, 1001),
                           snapshot_get=False,
                           docstring=textwrap.dedent("""
        Number of steps for staircase sweep. Possible  values from 1 to
        1001"""))

        self.add_parameter(name='current_compliance',
                           set_cmd=self._set_current_compliance,
                           get_cmd=self._get_current_compliance,
                           unit='A',
                           vals=vals.Numbers(-40, 40),
                           snapshot_get=False,
                           docstring=textwrap.dedent("""
        Current compliance (in A). Refer to Manual 2016. See Table 4-7 on
        page 24, Table 4-9 on page 26, Table 4-12 on page 27, or Table 4-15
        on page 28 for each measurement resource type. If you do not set
        current_compliance, the previous value is used.
        Compliance polarity is automatically set to the same polarity as the
        output value, regardless of the specified Icomp.
        If the output value is 0, the compliance polarity is positive. If
        you set Pcomp, the maximum Icomp value for the measurement resource
        is allowed, regardless of the output range setting.
                           """))

        self.add_parameter(name='power_compliance',
                           set_cmd=self._set_power_compliance,
                           get_cmd=self._get_power_compliance,
                           unit='W',
                           vals=vals.Numbers(0.001, 80),
                           snapshot_get=False,
                           docstring=textwrap.dedent("""
        Power compliance (in W). Resolution: 0.001 W. If it is not entered,
        the power compliance is not set. This parameter is not available for
        HVSMU. 0.001 to 2 for MPSMU/HRSMU, 0.001 to 20 for HPSMU, 0.001 to
        40 for HCSMU, 0.001 to 80 for dual HCSMU, 0.001 to 3 for MCSMU,
        0.001 to 100 for UHVU
                           """))

    def _set_sweep_mode(self, value: constants.SweepMode) -> None:
        self._sweep_step_parameters["sweep_mode"] = value
        self._set_from_sweep_step_parameters()

    def _get_sweep_mode(self) -> constants.SweepMode:
        mode_val = self._get_sweep_steps_parameters('sweep_mode')
        return constants.SweepMode(mode_val)

    def _set_sweep_range(self, value: constants.VOutputRange) -> None:
        self._sweep_step_parameters["sweep_range"] = value
        self._set_from_sweep_step_parameters()

    def _get_sweep_range(self) -> constants.VOutputRange:
        range_val = self._get_sweep_steps_parameters('sweep_range')
        return constants.VOutputRange(range_val)

    def _set_sweep_start(self, value: float) -> None:
        self._sweep_step_parameters["sweep_start"] = value
        self._set_from_sweep_step_parameters()

    def _get_sweep_start(self) -> float:
        sweep_start = self._get_sweep_steps_parameters('sweep_start')
        return sweep_start

    def _set_sweep_end(self, value: float) -> None:
        self._sweep_step_parameters["sweep_end"] = value
        self._set_from_sweep_step_parameters()

    def _get_sweep_end(self) -> float:
        sweep_end = self._get_sweep_steps_parameters('sweep_end')
        return sweep_end

    def _set_sweep_steps(self, value: int) -> None:
        self._sweep_step_parameters["sweep_steps"] = value
        self._set_from_sweep_step_parameters()

    def _get_sweep_steps(self) -> int:
        sweep_steps = self._get_sweep_steps_parameters('sweep_steps')
        return sweep_steps

    def _set_current_compliance(self, value: Optional[float]) -> None:
        self._sweep_step_parameters["current_compliance"] = value
        self._set_from_sweep_step_parameters()

    def _get_current_compliance(self) -> Optional[float]:
        current_compliance = self._get_sweep_steps_parameters(
            'current_compliance')
        return current_compliance

    def _set_power_compliance(self, value: Optional[float]) -> None:
        if self._sweep_step_parameters['current_compliance'] is None:
            raise ValueError('Current compliance must be set before setting '
                             'power compliance')
        self._sweep_step_parameters["power_compliance"] = value
        self._set_from_sweep_step_parameters()

    def _get_power_compliance(self) -> Optional[float]:
        power_compliance = self._get_sweep_steps_parameters('power_compliance')
        return power_compliance

    def _set_from_sweep_step_parameters(self) -> None:
        msg = MessageBuilder().wv(
            chnum=self.parent.channels[0],
            mode=self._sweep_step_parameters['sweep_mode'],
            v_range=self._sweep_step_parameters['sweep_range'],
            start=self._sweep_step_parameters['sweep_start'],
            stop=self._sweep_step_parameters['sweep_end'],
            step=self._sweep_step_parameters['sweep_steps'],
            i_comp=self._sweep_step_parameters['current_compliance'],
            p_comp=self._sweep_step_parameters["power_compliance"]
        )
        self.write(msg.message)

    @staticmethod
    def _get_sweep_delays() -> str:
        msg = MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.STAIRCASE_SWEEP_MEASUREMENT_SETTINGS
        )
        cmd = msg.message
        return cmd

    @staticmethod
    def _get_sweep_delays_parser(response: str) -> Dict[str, float]:
        match = re.search('WT(?P<hold_time>.+?),(?P<delay>.+?),'
                          '(?P<step_delay>.+?),(?P<trigger_delay>.+?),'
                          '(?P<measure_delay>.+?)(;|$)',
                          response)
        if not match:
            raise ValueError('Sweep delays (WT) not found.')

        resp_dict = match.groupdict()
        out_dict = {key: float(value) for key, value in resp_dict.items()}
        return out_dict

    def _set_sweep_auto_abort(self, val: Union[bool, constants.Abort]) -> None:
        msg = MessageBuilder().wm(abort=val)
        self.write(msg.message)

    def _set_post_sweep_voltage_condition(
            self, val: Union[constants.WM.Post, int]) -> None:
        msg = MessageBuilder().wm(abort=self.sweep_auto_abort(), post=val)
        self.write(msg.message)

    def _get_sweep_auto_abort_setting(self) -> Dict[str, str]:
        msg = MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.STAIRCASE_SWEEP_MEASUREMENT_SETTINGS
        )
        response = self.ask(msg.message)
        match = re.search(r'WM(?P<abort_function>.+?),'
                          r'(?P<output_after_sweep>.+?)'
                          r'(;|$)',
                          response)
        if match is None:
            raise RuntimeError("Did not find expected response for sweep "
                               "auto abort settings")
        resp_dict = match.groupdict()
        return resp_dict

    def _get_sweep_auto_abort(self) -> int:
        resp_dict = self._get_sweep_auto_abort_setting()
        return int(resp_dict['abort_function'])

    def _get_post_sweep_voltage_condition(self) -> int:
        resp_dict = self._get_sweep_auto_abort_setting()
        return int(resp_dict['output_after_sweep'])

    @overload
    def _get_sweep_steps_parameters(
            self,
            name: Literal['chan']
    ) -> Union[int, constants.ChNr]: ...

    @overload
    def _get_sweep_steps_parameters(
            self,
            name: Literal['sweep_mode']
    ) -> Union[constants.SweepMode, int]: ...

    @overload
    def _get_sweep_steps_parameters(
            self,
            name: Literal['sweep_range']
    ) -> Union[constants.VOutputRange, int]: ...

    @overload
    def _get_sweep_steps_parameters(
            self,
            name: Literal['sweep_start',
                          'sweep_end']
    ) -> float: ...

    @overload
    def _get_sweep_steps_parameters(
            self,
            name: Literal['sweep_steps']
    ) -> int: ...

    @overload
    def _get_sweep_steps_parameters(
            self,
            name: Literal['current_compliance',
                          'power_compliance']
    ) -> Optional[float]: ...

    def _get_sweep_steps_parameters(
            self,
            name: Literal['chan',
                          'sweep_mode',
                          'sweep_range',
                          'sweep_start',
                          'sweep_end',
                          'sweep_steps',
                          'current_compliance',
                          'power_compliance']
    ) -> Union[constants.ChNr, constants.SweepMode,
               constants.VOutputRange, int, float, None]:
        msg = MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.STAIRCASE_SWEEP_MEASUREMENT_SETTINGS
        )
        cmd = msg.message
        response = self.ask(cmd)
        out_dict = self._get_sweep_steps_parser(response)
        if out_dict['chan'] != self.parent.channels[0]:
            raise ValueError('Sweep parameters (WV) such as '
                             'sweep_mode, sweep_range, sweep_start, '
                             'sweep_end, sweep_steps etc are not set for '
                             'this SMU.')
        return out_dict[name]

    @staticmethod
    def _get_sweep_steps_parser(response: str) -> SweepSteps:
        match = re.search(r'WV(?P<chan>.+?),'
                          r'(?P<sweep_mode>.+?),'
                          r'(?P<sweep_range>.+?),'
                          r'(?P<sweep_start>.+?),'
                          r'(?P<sweep_end>.+?),'
                          r'(?P<sweep_steps>.+?)'
                          r'(,(?P<current_compliance>.+?)|;|$)'
                          r'(,(?P<power_compliance>.+?)|;|$)'
                          r'(;|$)',
                          response)
        if not match:
            raise ValueError('Sweep steps (WV) not found.')

        out_dict: SweepSteps = {}
        resp_dict = match.groupdict()

        out_dict['chan'] = int(resp_dict['chan'])
        out_dict['sweep_mode'] = int(resp_dict['sweep_mode'])
        out_dict['sweep_range'] = int(resp_dict['sweep_range'])
        out_dict['sweep_start'] = float(resp_dict['sweep_start'])
        out_dict['sweep_end'] = float(resp_dict['sweep_end'])
        out_dict['sweep_steps'] = int(resp_dict['sweep_steps'])
        if resp_dict['current_compliance'] is not None:
            out_dict['current_compliance'] = float(
                resp_dict['current_compliance'])
        else:
            out_dict['current_compliance'] = None
        if resp_dict['power_compliance'] is not None:
            out_dict['power_compliance'] = float(
                resp_dict['power_compliance'])
        else:
            out_dict['power_compliance'] = None
        return out_dict


class _ParameterWithStatus(Parameter):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self._measurement_status: Optional[MeasurementStatus] = None

    @property
    def measurement_status(self) -> Optional[MeasurementStatus]:
        return self._measurement_status

    def snapshot_base(self, update: Optional[bool] = True,
                      params_to_skip_update: Optional[Sequence[str]] = None
                      ) -> Dict[Any, Any]:
        snapshot = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update
        )

        if self._snapshot_value:
            snapshot["measurement_status"] = self.measurement_status

        return snapshot


class _SpotMeasurementVoltageParameter(_ParameterWithStatus):
    def set_raw(self, value: ParamRawDataType) -> None:
        smu = cast("B1517A", self.instrument)

        if smu._source_config["output_range"] is None:
            smu._source_config["output_range"] = constants.VOutputRange.AUTO
        if not isinstance(smu._source_config["output_range"],
                          constants.VOutputRange):
            raise TypeError(
                "Asking to force voltage, but source_config contains a "
                "current output range"
            )
        msg = MessageBuilder().dv(
            chnum=smu.channels[0],
            v_range=smu._source_config["output_range"],
            voltage=value,
            i_comp=smu._source_config["compliance"],
            comp_polarity=smu._source_config["compl_polarity"],
            i_range=smu._source_config["min_compliance_range"],
        )
        smu.write(msg.message)

        smu.root_instrument.\
            _reset_measurement_statuses_of_smu_spot_measurement_parameters(
            'voltage'
        )

    def get_raw(self) -> ParamRawDataType:
        smu = cast("B1517A", self.instrument)

        msg = MessageBuilder().tv(
            chnum=smu.channels[0],
            v_range=smu._measure_config["v_measure_range"],
        )
        response = smu.ask(msg.message)

        parsed = parse_spot_measurement_response(response)

        self._measurement_status = parsed["status"]

        return parsed["value"]


class _SpotMeasurementCurrentParameter(_ParameterWithStatus):
    def set_raw(self, value: ParamRawDataType) -> None:
        smu = cast("B1517A", self.instrument)

        if smu._source_config["output_range"] is None:
            smu._source_config["output_range"] = constants.IOutputRange.AUTO
        if not isinstance(smu._source_config["output_range"],
                          constants.IOutputRange):
            raise TypeError(
                "Asking to force current, but source_config contains a "
                "voltage output range"
            )
        msg = MessageBuilder().di(
            chnum=smu.channels[0],
            i_range=smu._source_config["output_range"],
            current=value,
            v_comp=smu._source_config["compliance"],
            comp_polarity=smu._source_config["compl_polarity"],
            v_range=smu._source_config["min_compliance_range"],
        )
        smu.write(msg.message)

        smu.root_instrument.\
            _reset_measurement_statuses_of_smu_spot_measurement_parameters(
            'current'
        )

    def get_raw(self) -> ParamRawDataType:
        smu = cast("B1517A", self.instrument)

        msg = MessageBuilder().ti(
            chnum=smu.channels[0],
            i_range=smu._measure_config["i_measure_range"],
        )
        response = smu.ask(msg.message)

        parsed = parse_spot_measurement_response(response)

        self._measurement_status = parsed["status"]

        return parsed["value"]


class B1517A(B1500Module):
    """
    Driver for Keysight B1517A Source/Monitor Unit module for B1500
    Semiconductor Parameter Analyzer.

    Args:
        parent: mainframe B1500 instance that this module belongs to
        name: Name of the instrument instance to create. If `None`
            (Default), then the name is autogenerated from the instrument
            class.
        slot_nr: Slot number of this module (not channel number)
    """
    MODULE_KIND = ModuleKind.SMU
    _interval_validator = vals.Numbers(0.0001, 65.535)

    def __init__(self, parent: 'KeysightB1500', name: Optional[str],
                 slot_nr: int, **kwargs: Any):
        super().__init__(parent, name, slot_nr, **kwargs)
        self.channels = (ChNr(slot_nr),)
        self._measure_config: Dict[str, Optional[Any]] = {
            k: None for k in ("v_measure_range", "i_measure_range",)}
        self._source_config: Dict[str, Optional[Any]] = {
            k: None for k in ("output_range", "compliance",
                              "compl_polarity", "min_compliance_range")}
        self._timing_parameters: Dict[str, Optional[Any]] = {
            k: None for k in ("h_bias", "interval", "number", "h_base")}

        # We want to snapshot these configuration dictionaries
        self._meta_attrs += ['_measure_config', '_source_config',
                             '_timing_parameters']

        self.add_submodule('iv_sweep', IVSweeper(self, 'iv_sweep'))
        self.setup_fnc_already_run: bool = False
        self.power_line_frequency: int = 50
        self._average_coefficient: int = 1
        self._valid_v_measure_ranges: List[VMeasRange] = [VMeasRange.AUTO,
                                                          VMeasRange.MIN_0V5,
                                                          VMeasRange.MIN_2V,
                                                          VMeasRange.MIN_5V,
                                                          VMeasRange.MIN_20V,
                                                          VMeasRange.MIN_40V,
                                                          VMeasRange.MIN_100V,
                                                          VMeasRange.FIX_0V5,
                                                          VMeasRange.FIX_2V,
                                                          VMeasRange.FIX_5V,
                                                          VMeasRange.FIX_20V,
                                                          VMeasRange.FIX_40V,
                                                          VMeasRange.FIX_100V]
        self._valid_i_measure_ranges: List[IMeasRange] = [IMeasRange.AUTO,
                                                          IMeasRange.MIN_1pA,
                                                          IMeasRange.MIN_10pA,
                                                          IMeasRange.MIN_100pA,
                                                          IMeasRange.MIN_1nA,
                                                          IMeasRange.MIN_10nA,
                                                          IMeasRange.MIN_100nA,
                                                          IMeasRange.MIN_1uA,
                                                          IMeasRange.MIN_10uA,
                                                          IMeasRange.MIN_100uA,
                                                          IMeasRange.MIN_1mA,
                                                          IMeasRange.MIN_10mA,
                                                          IMeasRange.MIN_100mA,
                                                          IMeasRange.FIX_1pA,
                                                          IMeasRange.FIX_10pA,
                                                          IMeasRange.FIX_100pA,
                                                          IMeasRange.FIX_1nA,
                                                          IMeasRange.FIX_10nA,
                                                          IMeasRange.FIX_100nA,
                                                          IMeasRange.FIX_1uA,
                                                          IMeasRange.FIX_10uA,
                                                          IMeasRange.FIX_100uA,
                                                          IMeasRange.FIX_1mA,
                                                          IMeasRange.FIX_10mA,
                                                          IMeasRange.FIX_100mA]
        self._valid_v_output_ranges: List[VOutputRange] = [
            VOutputRange.AUTO, VOutputRange.MIN_0V5, VOutputRange.MIN_2V,
            VOutputRange.MIN_5V, VOutputRange.MIN_20V, VOutputRange.MIN_40V,
            VOutputRange.MIN_100V]
        self._valid_i_output_ranges: List[IOutputRange] = [
            IOutputRange.AUTO, IOutputRange.MIN_1pA, IOutputRange.MIN_10pA,
            IOutputRange.MIN_100pA, IOutputRange.MIN_1nA, IOutputRange.MIN_10nA,
            IOutputRange.MIN_100nA, IOutputRange.MIN_1uA,
            IOutputRange.MIN_10uA, IOutputRange.MIN_100uA,
            IOutputRange.MIN_1mA, IOutputRange.MIN_10mA, IOutputRange.MIN_100mA]

        self.add_parameter(
            name="measurement_mode",
            get_cmd=None,
            set_cmd=self._set_measurement_mode,
            set_parser=MM.Mode,
            vals=vals.Enum(*list(MM.Mode)),
            initial_cache_value=MM.Mode.SPOT,
            docstring=textwrap.dedent("""
                Set measurement mode for this module.

                It is recommended for this parameter to use values from
                :class:`.constants.MM.Mode` enumeration.

                Refer to the documentation of ``MM`` command in the
                programming guide for more information.""")
        )
        # Instrument is initialized with this setting having value of
        # `1`, spot measurement mode, hence let's set the parameter's cache to
        # this value since it is not possible to request this value from the
        # instrument.

        self.add_parameter(
            name="measurement_operation_mode",
            set_cmd=self._set_measurement_operation_mode,
            get_cmd=self._get_measurement_operation_mode,
            set_parser=constants.CMM.Mode,
            vals=vals.Enum(*list(constants.CMM.Mode)),
            docstring=textwrap.dedent("""
            The methods sets the SMU measurement operation mode. This
            is not available for the high speed spot measurement.
            mode : SMU measurement operation mode. `constants.CMM.Mode`
            """)

        )
        self.add_parameter(
            name="voltage",
            parameter_class=_SpotMeasurementVoltageParameter,
            unit="V",
            snapshot_get=False
        )

        self.add_parameter(
            name="current",
            parameter_class=_SpotMeasurementCurrentParameter,
            unit="A",
            snapshot_get=False
        )

        self.add_parameter(
            name="time_axis",
            get_cmd=self._get_time_axis,
            vals=Arrays(shape=(self._get_number_of_samples,)),
            snapshot_value=False,
            label='Time',
            unit='s'
        )

        self.add_parameter(
            name="sampling_measurement_trace",
            parameter_class=SamplingMeasurement,
            vals=Arrays(shape=(self._get_number_of_samples,)),
            setpoints=(self.time_axis,)
        )

        self.add_parameter(
            name="current_measurement_range",
            set_cmd=self._set_current_measurement_range,
            get_cmd=self._get_current_measurement_range,
            vals=vals.Enum(*list(constants.IMeasRange)),
            set_parser=constants.IMeasRange,
            docstring=textwrap.dedent("""
            This method specifies the current measurement range or ranging
            type.In the initial setting, the auto ranging is set. The range
            changing occurs immediately after the trigger (that is, during
            the measurements). Current measurement channel can be decided by
             the `measurement_operation_mode` method setting and the channel
            output mode (voltage or current).
        """))

        self.add_parameter(
            name="enable_filter",
            set_cmd=self._set_enable_filter,
            get_cmd=None,
            snapshot_get=False,
            vals=vals.Bool(),
            initial_cache_value=False,
            docstring=textwrap.dedent("""
            This methods sets the connection mode of a SMU filter for each
            channel. A filter is mounted on the SMU. It assures clean source
            output with no spikes or overshooting.
            ``False``, meaning "disconnect" is the initial setting. Set to
            ``True`` to connect.
            """)

        )

    def _get_number_of_samples(self) -> int:
        if self._timing_parameters['number'] is not None:
            sample_number = self._timing_parameters['number']
            return sample_number
        else:
            raise Exception('set timing parameters first')

    def _get_time_axis(self) -> np.ndarray:
        sample_rate = self._timing_parameters['interval']
        total_time = self._total_measurement_time()
        time_xaxis = np.arange(0, total_time, sample_rate)
        return time_xaxis

    def _total_measurement_time(self) -> float:
        if self._timing_parameters['interval'] is None or \
                self._timing_parameters['number'] is None:
            raise Exception('set timing parameters first')

        sample_number = self._timing_parameters['number']
        sample_rate = self._timing_parameters['interval']
        total_time = float(sample_rate * sample_number)
        return total_time

    def _set_current_measurement_range(
            self,
            i_range: Union[constants.IMeasRange, int]
        ) -> None:
        msg = MessageBuilder().ri(chnum=self.channels[0],
                                  i_range=i_range)
        self.write(msg.message)

    def _get_current_measurement_range(self) -> \
            List[Tuple[constants.ChNr, constants.IMeasRange]]:
        response = self.ask(MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.MEASUREMENT_RANGING_STATUS).message)
        match = re.findall(r'RI (.+?),(.+?)($|;)', response)
        response_list = [(constants.ChNr(int(i)),
                          constants.IMeasRange(int(j)))
                         for i, j, _ in match]
        return response_list

    def _set_measurement_mode(self, mode: Union[MM.Mode, int]) -> None:
        self.write(MessageBuilder()
                   .mm(mode=mode,
                       channels=[self.channels[0]])
                   .message)

    def _set_measurement_operation_mode(self,
                                        mode: Union[constants.CMM.Mode, int]
                                        ) -> None:
        self.write(MessageBuilder()
                   .cmm(mode=mode,
                        chnum=self.channels[0])
                   .message)

    def _get_measurement_operation_mode(self) \
            -> List[Tuple[constants.ChNr, constants.CMM.Mode]]:
        response = self.ask(MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.SMU_MEASUREMENT_OPERATION).message)
        match = re.findall(r'CMM (.+?),(.+?)($|;)', response)
        response_list = [(constants.ChNr(int(i)),
                          constants.CMM.Mode(int(j)))
                         for i, j, _ in match]
        return response_list

    def _set_enable_filter(
            self,
            enable_filter: bool,
    ) -> None:
        """
        This methods sets the connection mode of a SMU filter for each channel.
        A filter is mounted on the SMU. It assures clean source output with
        no spikes or overshooting.

        Args:
            enable_filter : Status of the filter.
                False: Disconnect (initial setting).
                True: Connect.
        """
        self.root_instrument.enable_smu_filters(
            enable_filter=enable_filter,
            channels=[self.channels[0]]
        )

    def source_config(
            self,
            output_range: constants.OutputRange,
            compliance: Optional[Union[float, int]] = None,
            compl_polarity: Optional[constants.CompliancePolarityMode] = None,
            min_compliance_range: Optional[constants.MeasureRange] = None,
    ) -> None:
        """Configure sourcing voltage/current

        Args:
            output_range: voltage/current output range
            compliance: voltage/current compliance value
            compl_polarity: compliance polarity mode
            min_compliance_range: minimum voltage/current compliance output
                range
        """
        if min_compliance_range is not None:
            if isinstance(min_compliance_range, type(output_range)):
                raise TypeError(
                    "When forcing voltage, min_compliance_range must be an "
                    "current output range (and vice versa)."
                )

        if isinstance(output_range, VOutputRange):
            if output_range not in self._valid_v_output_ranges:
                raise RuntimeError(
                    "Invalid Source Voltage Output Range"
                )

        if isinstance(output_range, IOutputRange):
            if output_range not in self._valid_i_output_ranges:
                raise RuntimeError(
                    "Invalid Source Current Output Range"
                )

        self._source_config = {
            "output_range": output_range,
            "compliance": compliance,
            "compl_polarity": compl_polarity,
            "min_compliance_range": min_compliance_range,
        }

    @deprecate(reason='the method confuses ranges for voltage and current '
                      'measurements',
               alternative='v_measure_range_config or i_measure_range_config')
    def measure_config(self, measure_range: constants.MeasureRange) -> None:
        """Configure measuring voltage/current

        Args:
            measure_range: voltage/current measurement range
        """
        if measure_range in (VMeasRange.AUTO, IMeasRange.AUTO):
            self.v_measure_range_config(VMeasRange.AUTO)
            self.i_measure_range_config(IMeasRange.AUTO)
        elif isinstance(measure_range, constants.VMeasRange):
            self.v_measure_range_config(measure_range)
        elif isinstance(measure_range, constants.IMeasRange):
            self.i_measure_range_config(measure_range)

    def v_measure_range_config(self,
                               v_measure_range: constants.VMeasRange) -> None:
        """Configure measuring voltage

        Args:
            v_measure_range: voltage measurement range
        """
        if not isinstance(v_measure_range, constants.VMeasRange):
            raise TypeError(f"Expected valid voltage measurement range, "
                            f"got {v_measure_range}.")

        if v_measure_range not in self._valid_v_measure_ranges:
            raise RuntimeError(f"{v_measure_range} voltage measurement "
                               f"range is invalid for the device. Valid "
                               f"ranges are {self._valid_v_measure_ranges}.")

        self._measure_config["v_measure_range"] = v_measure_range

    def i_measure_range_config(self,
                               i_measure_range: constants.IMeasRange) -> None:
        """Configure measuring current

        Args:
            i_measure_range: current measurement range
        """
        if not isinstance(i_measure_range, constants.IMeasRange):
            raise TypeError(f"Expected valid current measurement range, "
                            f"got {i_measure_range}.")

        if i_measure_range not in self._valid_i_measure_ranges:
            raise RuntimeError(f"{i_measure_range} current measurement "
                               f"range is invalid for the device. Valid "
                               f"ranges are {self._valid_i_measure_ranges}.")

        self._measure_config["i_measure_range"] = i_measure_range

    def timing_parameters(self,
                          h_bias: float,
                          interval: float,
                          number: int,
                          h_base: Optional[float] = None
                          ) -> None:
        """
        This command sets the timing parameters of the sampling measurement
        mode (:attr:`.MM.Mode.SAMPLING`, ``10``).

        Refer to the programming guide for more information about the ``MT``
        command, especially for notes on sampling operation and about setting
        interval < 0.002 s.

        Args:
            h_bias: Time since the bias value output until the first
                sampling point. Numeric expression. in seconds.
                0 (initial setting) to 655.35 s, resolution 0.01 s.
                The following values are also available for interval < 0.002 s.
                ``|h_bias|`` will be the time since the sampling start until
                the bias value output. -0.09 to -0.0001 s, resolution 0.0001 s.
            interval: Interval of the sampling. Numeric expression,
                0.0001 to 65.535, in seconds. Initial value is 0.002.
                Resolution is 0.001 at interval < 0.002. Linear sampling of
                interval < 0.002 in 0.00001 resolution is available
                only when the following formula is satisfied.
                ``interval >= 0.0001 + 0.00002 * (number of measurement
                channels-1)``
            number: Number of samples. Integer expression. 1 to the
                following value. Initial value is 1000. For the linear
                sampling: ``100001 / (number of measurement channels)``.
                For the log sampling: ``1 + (number of data for 11 decades)``
            h_base: Hold time of the base value output until the bias value
                output. Numeric expression. in seconds. 0 (initial setting)
                to 655.35 s, resolution 0.01 s.
        """
        # The duplication of kwargs in the calls below is due to the
        # difference in type annotations between ``MessageBuilder.mt()``
        # method and ``_timing_parameters`` attribute.

        self._interval_validator.validate(interval)
        self._timing_parameters.update(h_bias=h_bias,
                                       interval=interval,
                                       number=number,
                                       h_base=h_base)
        self.write(MessageBuilder()
                   .mt(h_bias=h_bias,
                       interval=interval,
                       number=number,
                       h_base=h_base)
                   .message)

    def use_high_speed_adc(self) -> None:
        """Use high-speed ADC type for this module/channel"""
        self.write(MessageBuilder()
                   .aad(chnum=self.channels[0],
                        adc_type=AAD.Type.HIGH_SPEED)
                   .message)

    def use_high_resolution_adc(self) -> None:
        """Use high-resolution ADC type for this module/channel"""
        self.write(MessageBuilder()
                   .aad(chnum=self.channels[0],
                        adc_type=AAD.Type.HIGH_RESOLUTION)
                   .message)

    def set_average_samples_for_high_speed_adc(
            self,
            number: int = 1,
            mode: constants.AV.Mode = constants.AV.Mode.AUTO
    ) -> None:
        """
        This command sets the number of averaging samples of the high-speed
        ADC (A/D converter). This command is not effective for the
        high-resolution ADC. Also, this command is not effective for the
        measurements using pulse.

        Args:
            number: 1 to 1023, or -1 to -100. Initial setting is 1.
                For positive number input, this value specifies the number
                of samples depended on the mode value.
                For negative number input, this parameter specifies the
                number of power line cycles (PLC) for one point measurement.
                The Keysight B1500 gets 128 samples in 1 PLC. If number is
                negative it ignores the mode argument.
            mode : Averaging mode. Integer expression. This parameter is
                meaningless for negative number.
                `constants.AV.Mode.AUTO`: Auto mode (default setting).
                Number of samples = number x initial number.
                `constants.AV.Mode.MANUAL`: Manual mode.
                Number of samples = number
        """
        self.write(MessageBuilder().av(number=number, mode=mode).message)
        self._average_coefficient = number

    def setup_staircase_sweep(
            self,
            v_start: float,
            v_end: float,
            n_steps: int,
            post_sweep_voltage_val: Union[constants.WMDCV.Post,
                                          int] = constants.WMDCV.Post.STOP,
            av_coef: int = -1,
            enable_filter: bool = True,
            v_src_range: constants.OutputRange = constants.VOutputRange.AUTO,
            i_comp: float = 10e-6,
            i_meas_range: Optional[
                constants.MeasureRange] = constants.IMeasRange.FIX_10uA,
            hold_time: float = 0,
            delay: float = 0,
            step_delay: float = 0,
            measure_delay: float = 0,
            abort_enabled: Union[constants.Abort,
                                 int] = constants.Abort.ENABLED,
            sweep_mode: Union[constants.SweepMode,
                              int] = constants.SweepMode.LINEAR
    ) -> None:
        """
        Setup the staircase sweep measurement using the same set of commands
        (in the same order) as given in the programming manual - see pages
        3-19 and 3-20.

        Args:
            v_start: starting voltage of staircase sweep
            v_end: ending voltage of staircase sweep
            n_steps: number of measurement points (uniformly distributed
                between v_start and v_end)
            post_sweep_voltage_val: voltage to hold at end of sweep (i.e.
                start or end val). Sweep chan will also output this voltage
                if an abort condition is encountered during the sweep
            av_coef: coefficient to use for av command to set ADC
                averaging.  Negative value implies NPLC mode with absolute
                value of av_coeff the NPLC setting to use. Positive value
                implies auto mode and must be set to >= 4
            enable_filter: turn SMU filter on or off
            v_src_range: range setting to use for voltage source
            i_comp: current compliance level
            i_meas_range: current measurement range
            hold_time: time (in s) to wait before starting very first
                measurement in sweep
            delay: time (in s) after starting to force a step output and
                before starting a step measurement
            step_delay: time (in s) after starting a step measurement before
                next step in staircase. If step_delay is < measurement time,
                B1500 waits until measurement complete and then forces the
                next step value.
            measure_delay: time (in s)  after receiving a start step
                measurement trigger and before starting a step measurement
            abort_enabled: Enbale abort
            sweep_mode: Linear, log, linear-2-way or log-2-way
          """
        self.set_average_samples_for_high_speed_adc(av_coef)
        self.enable_filter(enable_filter)
        self.source_config(output_range=v_src_range,
                           compliance=i_comp,
                           min_compliance_range=i_meas_range)
        self.voltage(v_start)
        self.measurement_operation_mode(constants.CMM.Mode.COMPLIANCE_SIDE)
        self.current_measurement_range(i_meas_range)
        self.iv_sweep.hold_time(hold_time)
        self.iv_sweep.delay(delay)
        self.iv_sweep.step_delay(step_delay)
        self.iv_sweep.measure_delay(measure_delay)
        self.iv_sweep.sweep_auto_abort(abort_enabled)
        self.iv_sweep.post_sweep_voltage_condition(post_sweep_voltage_val)
        self.iv_sweep.sweep_mode(sweep_mode)
        self.iv_sweep.sweep_range(v_src_range)
        self.iv_sweep.sweep_start(v_start)
        self.iv_sweep.sweep_end(v_end)
        self.iv_sweep.sweep_steps(n_steps)
        self.iv_sweep.current_compliance(i_comp)
        self.root_instrument.clear_timer_count()

        self.setup_fnc_already_run = True
