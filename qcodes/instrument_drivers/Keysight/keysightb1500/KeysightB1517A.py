import re
import textwrap
from typing import Optional, Dict, Any, Union, TYPE_CHECKING, cast
import numpy as np
import qcodes.utils.validators as vals
from qcodes import InstrumentChannel, MultiParameter
from qcodes.instrument.group_parameter import GroupParameter, Group
from qcodes.utils.validators import Arrays

from .KeysightB1500_sampling_measurement import SamplingMeasurement
from .KeysightB1500_module import B1500Module, parse_spot_measurement_response, \
    parse_fmt_1_0_response, _FMTResponse
from .message_builder import MessageBuilder
from . import constants
from .constants import ModuleKind, ChNr, AAD, MM
if TYPE_CHECKING:
    from .KeysightB1500_base import KeysightB1500


class IVSweeper(InstrumentChannel):
    def __init__(self, parent: 'B1520A', name: str, **kwargs: Any):
        super().__init__(parent, name, **kwargs)

        self.add_parameter(name='sweep_auto_abort',
                           set_cmd=self._set_sweep_auto_abort,
                           set_parser=constants.Abort,
                           vals=vals.Enum(*list(constants.Abort)),
                           get_cmd=None,
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
        self.sweep_auto_abort.cache.set(constants.Abort.ENABLED)

        self.add_parameter(name='post_sweep_voltage_condition',
                           set_cmd=self._set_post_sweep_voltage_condition,
                           set_parser=constants.WM.Post,
                           vals=vals.Enum(*list(constants.WM.Post)),
                           get_cmd=None,
                           docstring=textwrap.dedent("""
        Source output value after the measurement is normally completed. If 
        this parameter is not set, the sweep sources force the start value.
                                 """))
        self.post_sweep_voltage_condition.cache.set(constants.WM.Post.START)

        self.add_parameter(name='hold',
                           initial_value=0,
                           vals=vals.Numbers(0, 655.35),
                           unit='s',
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
                           Hold time (in seconds) that is the 
                           wait time after starting measurement 
                           and before starting delay time for 
                           the first step 0 to 655.35, with 10 
                           ms resolution. Numeric expression.
                          """))

        self.add_parameter(name='delay',
                           initial_value=0,
                           vals=vals.Numbers(0, 65.535),
                           unit='s',
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
                           Delay time (in seconds) that is the wait time after
                           starting to force a step output and before 
                            starting a step measurement. 0 to 65.535, 
                            with 0.1 ms resolution. Numeric expression.
                            """))

        self.add_parameter(name='step_delay',
                           initial_value=0,
                           vals=vals.Numbers(0, 1),
                           unit='s',
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
                            Step delay time (in seconds) that is the wait time
                            after starting a step measurement and before  
                            starting to force the next step output. 0 to 1, 
                            with 0.1 ms resolution. Numeric expression. If 
                            this parameter is not set, step delay will be 0. If 
                            step delay is shorter than the measurement time, 
                            the B1500 waits until the measurement completes, 
                            then forces the next step output.
                            """))

        self.add_parameter(name='trigger_delay',
                           initial_value=0,
                           unit='s',
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
                            Step source trigger delay time (in seconds) that
                            is the wait time after completing a step output 
                            setup and before sending a step output setup 
                            completion trigger. 0 to the value of ``delay``, 
                            with 0.1 ms resolution. 
                            If this parameter is not set, 
                            trigger delay will be 0.
                            """))

        self.add_parameter(name='measure_delay',
                           initial_value=0,
                           unit='s',
                           vals=vals.Numbers(0, 65.535),
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
                           Step measurement trigger delay time (in seconds)
                           that is the wait time after receiving a start step 
                           measurement trigger and before starting a step 
                           measurement. 0 to 65.535, with 0.1 ms resolution. 
                           Numeric expression. If this parameter is not set, 
                           measure delay will be 0.
                           """))

        self._set_sweep_delays_group = Group(
            [self.hold,
             self.delay,
             self.step_delay,
             self.trigger_delay,
             self.measure_delay],
            set_cmd='WT '
                    '{hold},'
                    '{delay},'
                    '{step_delay},'
                    '{trigger_delay},'
                    '{measure_delay}',
            get_cmd=self._get_sweep_delays(),
            get_parser=self._get_sweep_delays_parser)

        self.add_parameter(name='sweep_mode',
                           initial_value=constants.SweepMode.LINEAR,
                           vals=vals.Enum(*list(constants.SweepMode)),
                           set_parser=constants.SweepMode,
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
                 Sweep mode. Note that Only linear sweep (mode=1 or 3) is
                 available for the staircase sweep with pulsed bias.
                     1: Linear sweep (single stair, start to stop.)
                     2: Log sweep (single stair, start to stop.)
                     3: Linear sweep (double stair, start to stop to start.)
                     4: Log sweep (double stair, start to stop to start.)
                                """))

        self.add_parameter(name='sweep_range',
                           initial_value=0,
                           vals=vals.Enum(*[0, 19, 21, 26, 28, -19, -21, -26,
                                            -28]),
                           parameter_class=GroupParameter,
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
                           initial_value=0,
                           unit='V',
                           vals=vals.Numbers(-25, 25),
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
        Start value of the stair case sweep (in V). For the log sweep, 
        start and stop must have the same polarity.
                                """))

        self.add_parameter(name='sweep_end',
                           initial_value=0,
                           unit='V',
                           vals=vals.Numbers(-25, 25),
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
        Stop value of the DC bias sweep (in V). For the log sweep,start and
        stop must have the same polarity.
                                """))

        self.add_parameter(name='sweep_steps',
                           initial_value=1,
                           vals=vals.Ints(1, 1001),
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
        Number of steps for staircase sweep. Possible  values from 1 to 
        1001"""))

        self.add_parameter(name='current_compliance',
                           initial_value=100e-3,
                           parameter_class=GroupParameter,
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
                           initial_value=2,
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
        Power compliance (in W). Resolution: 0.001 W. If it is not entered, 
        the power compliance is not set. This parameter is not available for
        HVSMU. 0.001 to 2 for MPSMU/HRSMU, 0.001 to 20 for HPSMU, 0.001 to 
        40 for HCSMU, 0.001 to 80 for dual HCSMU, 0.001 to 3 for MCSMU, 
        0.001 to 100 for UHVU
                           """))

        self.add_parameter(name='chan',
                           initial_value=self.parent.channels[0],
                           parameter_class=GroupParameter)

        self._set_sweep_steps_group = Group(
            [self.chan,
             self.sweep_mode,
             self.sweep_range,
             self.sweep_start,
             self.sweep_end,
             self.sweep_steps,
             self.current_compliance,
             self.power_compliance],
            set_cmd='WV '
                    '{chan},'
                    '{sweep_mode},'
                    '{sweep_range},'
                    '{sweep_start},'
                    '{sweep_end},'
                    '{sweep_steps},'
                    '{current_compliance},'
                    '{power_compliance}',
            get_cmd=self._get_sweep_steps(),
            get_parser=self._get_sweep_steps_parser)


    @staticmethod
    def _get_sweep_delays() -> str:
        msg = MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.STAIRCASE_SWEEP_MEASUREMENT_SETTINGS
        )
        cmd = msg.message
        return cmd

    @staticmethod
    def _get_sweep_delays_parser(response: str) -> Dict[str, float]:
        match = re.search('WT(?P<hold>.+?),(?P<delay>.+?),'
                          '(?P<step_delay>.+?),(?P<trigger_delay>.+?),'
                          '(?P<measure_delay>.+?)(;|$)',
                          response)
        if not match:
            raise ValueError('Sweep delays (WT) not found.')

        out_str = match.groupdict()
        out_dict = {key: float(value) for key, value in out_str.items()}
        return out_dict

    def _set_sweep_auto_abort(self, val: Union[bool, constants.Abort]):
        msg = MessageBuilder().wm(abort=val)
        self.write(msg.message)

    def _set_post_sweep_voltage_condition(
            self, val: Union[constants.WM.Post, int]):
        msg = MessageBuilder().wm(abort=self.sweep_auto_abort(), post=val)
        self.write(msg.message)

    @staticmethod
    def _get_sweep_steps():
        msg = MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.STAIRCASE_SWEEP_MEASUREMENT_SETTINGS
        )
        cmd = msg.message
        return cmd

    @staticmethod
    def _get_sweep_steps_parser(response: str) -> Dict[
        str, Union[int, float]]:
        match = re.search(r'WV(?P<chan>.+?),'
                          r'(?P<sweep_mode>.+?),'
                          r'(?P<sweep_range>.+?),'
                          r'(?P<sweep_start>.+?),'
                          r'(?P<sweep_end>.+?),'
                          r'(?P<sweep_steps>.+?),'
                          r'(?P<current_compliance>.+?),'
                          r'(?P<power_compliance>.+?)'
                          r'(;|$)',
                          response)
        if not match:
            raise ValueError('Sweep steps (WV) not found.')

        out_dict: Dict[str, Union[int, float]] = {}
        resp_dict = match.groupdict()

        out_dict['chan'] = int(resp_dict['chan'])
        out_dict['sweep_mode'] = int(resp_dict['sweep_mode'])
        out_dict['sweep_range'] = int(resp_dict['sweep_range'])
        out_dict['sweep_start'] = float(resp_dict['sweep_start'])
        out_dict['sweep_end'] = float(resp_dict['sweep_end'])
        out_dict['sweep_steps'] = int(resp_dict['sweep_steps'])
        out_dict['current_compliance'] = float(resp_dict['current_compliance'])
        out_dict['power_compliance'] = float(resp_dict['power_compliance'])
        return out_dict


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

    def __init__(self, parent: 'KeysightB1500', name: Optional[str], slot_nr,
                 **kwargs):
        super().__init__(parent, name, slot_nr, **kwargs)
        self.channels = (ChNr(slot_nr),)
        self._measure_config: Dict[str, Optional[Any]] = {
            k: None for k in ("measure_range",)}
        self._source_config: Dict[str, Optional[Any]] = {
            k: None for k in ("output_range", "compliance",
                              "compl_polarity", "min_compliance_range")}
        self._timing_parameters: Dict[str, Optional[Any]] = {
            k: None for k in ("h_bias", "interval", "number", "h_base")}

        # We want to snapshot these configuration dictionaries
        self._meta_attrs += ['_measure_config', '_source_config',
                             '_timing_parameters']

        self.add_submodule('iv_sweep', IVSweeper(self, 'iv_sweep'))
        self.setup_fnc_already_run = False

        self.add_parameter(
            name="measurement_mode",
            get_cmd=None,
            set_cmd=self._set_measurement_mode,
            set_parser=MM.Mode,
            vals=vals.Enum(*list(MM.Mode)),
            docstring=textwrap.dedent("""
                Set measurement mode for this module.
                
                It is recommended for this parameter to use values from 
                :class:`.constants.MM.Mode` enumeration.
                
                Refer to the documentation of ``MM`` command in the 
                programming guide for more information.""")
        )
        # Instrument is initialized with this setting having value of
        # `1`, spot measurement mode, hence let's set the parameter to this
        # value since it is not possible to request this value from the
        # instrument.
        self.measurement_mode.cache.set(MM.Mode.SPOT)

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
            unit="V",
            set_cmd=self._set_voltage,
            get_cmd=self._get_voltage,
            snapshot_get=False
        )

        self.add_parameter(
            name="current",
            unit="A",
            set_cmd=self._set_current,
            get_cmd=self._get_current,
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
            docstring=textwrap.dedent(""" This method specifies the 
            current measurement range or ranging type. In the initial 
            setting, the auto ranging is set. The range changing occurs 
            immediately after the trigger (that is, during the 
            measurements). Current measurement channel can be decided by the 
            `measurement_operation_mode` method setting and the channel 
            output mode (voltage or current). 
            The range setting is cleared by the CL, CA, IN, *TST?, 
            *RST, or a device clear.
            
            Args:
                range: Measurement range or ranging type. Integer 
                expression. See Table 4-3 on page 19.
        """))

        self.add_parameter(name='iv_sweep_voltages',
                           get_cmd=self._iv_sweep_voltages,
                           unit='V',
                           label='Voltage',
                           docstring=textwrap.dedent("""
               Outputs the tuple of voltages to sweep.  sweep_start, sweep_end 
               and sweep_step functions are used to define the values of 
               voltages. There are possible modes; linear sweep, log sweep, 
               linear 2 way sweep and log 2 way sweep. The  output of 
               sweep_mode method is used to decide which mode to use.  
                              """))

        self.add_parameter(name='run_sweep',
                           parameter_class=IVSweepMeasurement,
                           docstring=textwrap.dedent("""
               This is MultiParameter. Running the sweep runs the measurement 
               on the list of values of cv_sweep_voltages. The output is a 
               primary parameter (Gate current)  and a secondary  
               parameter (Source/Drain current) both of whom use the same 
               setpoint iv_sweep_voltages. The impedance_model defines exactly 
               what will be the primary and secondary parameter.
                              """))

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

    def _set_voltage(self, value: float) -> None:
        if self._source_config["output_range"] is None:
            self._source_config["output_range"] = constants.VOutputRange.AUTO
        if not isinstance(self._source_config["output_range"],
                          constants.VOutputRange):
            raise TypeError(
                "Asking to force voltage, but source_config contains a "
                "current output range"
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

    def _set_current(self, value: float) -> None:
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

    def _set_current_measurement_range(self,
                                       i_range:Union[constants.IMeasRange, int]
                                       ) -> None:
        msg = MessageBuilder().ri(chnum=self.channels[0],
                                  i_range=i_range)
        self.write(msg.message)

    def _get_current_measurement_range(self) -> list:
        response = self.ask(MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.MEASUREMENT_RANGING_STATUS).message)
        match = re.findall(r'RI (.+?),(.+?)($|;)', response)
        response_list = [(constants.ChNr(int(i)).name,
                          constants.IMeasRange(int(j)).name)
                         for i, j, _ in match]
        return response_list

    def _get_current(self) -> float:
        msg = MessageBuilder().ti(
            chnum=self.channels[0],
            i_range=self._measure_config["measure_range"],
        )
        response = self.ask(msg.message)

        parsed = parse_spot_measurement_response(response)
        return parsed["value"]

    def _get_voltage(self) -> float:
        msg = MessageBuilder().tv(
            chnum=self.channels[0],
            v_range=self._measure_config["measure_range"],
        )
        response = self.ask(msg.message)

        parsed = parse_spot_measurement_response(response)
        return parsed["value"]

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

    def _get_measurement_operation_mode(self) -> list:
        response = self.ask(MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.SMU_MEASUREMENT_OPERATION).message)
        match = re.findall(r'CMM (.+?),(.+?)($|;)', response)
        response_list = [(constants.ChNr(int(i)).name,
                          constants.CMM.Mode(int(j)).name)
                         for i, j, _ in match]
        return response_list

    def source_config(
            self,
            output_range: constants.OutputRange,
            compliance: Optional[Union[float, int]] = None,
            compl_polarity: Optional[constants.CompliancePolarityMode] = None,
            min_compliance_range: Optional[constants.OutputRange] = None,
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

        self._source_config = {
            "output_range": output_range,
            "compliance": compliance,
            "compl_polarity": compl_polarity,
            "min_compliance_range": min_compliance_range,
        }

    def measure_config(self, measure_range: constants.MeasureRange) -> None:
        """Configure measuring voltage/current

        Args:
            measure_range: voltage/current measurement range
        """
        self._measure_config = {"measure_range": measure_range}

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
            number: Union[float, int] = 1,
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
                The Keysight B1500 gets 128 samples in 1 PLC.
                Ignore the mode parameter.

            mode : Averaging mode. Integer expression. This parameter is
                meaningless for negative number.
                `constants.AV.Mode.AUTO`: Auto mode (default setting).
                Number of samples = number x initial number.
                `constants.AV.Mode.MANUAL`: Manual mode.
                Number of samples = number
        """
        self.write(MessageBuilder().av(number=number, mode=mode).message)

    def connection_mode_of_smu_filter(
            self,
            enable_filter: bool,
            channels: Optional[constants.ChannelList] = None
    ) -> None:
        """
        This methods sets the connection mode of a SMU filter for each channel.
        A filter is mounted on the SMU. It assures clean source output with
        no spikes or overshooting. A maximum of ten channels can be set.

        Args:
            enable_filter : Status of the filter.
                False: Disconnect (initial setting).
                True: Connect.
            channels : SMU channel number. Specify channel from
                `constants.ChNr` If you do not specify chnum,  the FL
                command sets the same mode for all channels.
        """
        self.write(MessageBuilder().fl(enable_filter=enable_filter,
                                       channels=channels).message)

    def setup_staircase_sweep(
            self,
            v_start: float,
            v_end: float,
            n_steps: int,
            post_sweep_voltage_val: int = constants.WMDCV.Post.STOP,
            measure_chan_list=[1],
            av_coef=-1,
            enable_filter=True,
            v_src_range=constants.VOutputRange.AUTO,
            i_comp=10e-6,
            i_meas_range=constants.IMeasRange.FIX_10uA,
            hold_delay=0,
            delay=0,
            step_delay=0,
            measure_delay=0,
            abort_enabled=constants.Abort.ENABLED,
            sweep_mode=constants.SweepMode.LINEAR

    ):
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
            measure_chan_list: list of channels to be measured (will be
                measured in order supplied)
            av_coef: coefficient to use for av command to set ADC
                averaging.  Negative value implies NPLC mode with absolute
                value of av_coeff the NPLC setting to use. Positive value
                implies auto mode and must be set to >= 4
            enable_filter: turn SMU filter on or off
            v_src_range: range setting to use for voltage source
            i_comp: current compliance level
            i_meas_range: current measurement range
            hold_delay: time (in s) to wait before starting very first
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
            sweep_modeL Linear, log, linear-2-way or log-2-way
          """
        self.set_average_samples_for_high_speed_adc(av_coef)
        self.connection_mode_of_smu_filter(enable_filter=enable_filter)
        self.source_config(output_range=v_src_range,
                           compliance=i_comp,
                           min_compliance_range=i_meas_range)
        self.voltage(v_start)

        # Set measurement mode
        msg = MessageBuilder().mm(constants.MM.Mode.STAIRCASE_SWEEP,
                                  channels=measure_chan_list).message
        self.write(msg)

        self.measurement_operation_mode(constants.CMM.Mode.COMPLIANCE_SIDE)
        self.current_measurement_range(i_meas_range)
        self.iv_sweep.hold(hold_delay)
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

        error_list, error = [], ''

        while error != '+0,"No Error."':
            error = self.root_instrument.error_message()
            error_list.append(error)

        if len(error_list) <= 1:
            self.setup_fnc_already_run = True
        else:
            raise Exception(f'Received following errors while trying to set '
                            f'staircase sweep {error_list}')

    def _iv_sweep_voltages(self) -> tuple:
        sign = lambda s: s and (1, -1)[s < 0]
        start_value = self.iv_sweep.sweep_start()
        end_value = self.iv_sweep.sweep_end()
        step_value = self.iv_sweep.sweep_steps()
        if self.iv_sweep.sweep_mode() == 2 or self.iv_sweep.sweep_mode() == 4:
            if not sign(start_value) == sign(self.sweep_end()):
                if sign(start_value) == 0:
                    start_value = sign(start_value) * 0.005  # resolution
                elif sign(end_value) == 0:
                    end_value = sign(end_value) * 0.005  # resolution
                else:
                    raise AssertionError(
                        "Polarity of start and end is not "
                        "same.")

        def linear_sweep(start: float, end: float, steps: int) -> tuple:
            sweep_val = np.linspace(start, end, steps)
            return tuple(sweep_val)

        def log_sweep(start: float, end: float, steps: int) -> tuple:
            sweep_val = np.logspace(np.log10(start), np.log10(end), steps)
            return tuple(sweep_val)

        def linear_2way_sweep(start: float, end: float,
                              steps: int) -> tuple:
            if steps % 2 == 0:
                half_list = list(np.linspace(start, end, steps // 2))
                sweep_val = half_list + half_list[::-1]
            else:
                half_list = list(np.linspace(start, end, steps // 2,
                                             endpoint=False))
                sweep_val = half_list + [end] + half_list[::-1]
            return tuple(sweep_val)

        def log_2way_sweep(start: float, end: float, steps: int) -> tuple:
            if steps % 2 == 0:
                half_list = list(
                    np.logspace(np.log10(start), np.log10(end),
                                steps // 2))
                sweep_val = half_list + half_list[::-1]
            else:
                half_list = list(
                    np.logspace(np.log10(start), np.log10(end),
                                steps // 2, endpoint=False))
                sweep_val = half_list + [end] + half_list[::-1]
            return tuple(sweep_val)

        modes = {1: linear_sweep,
                 2: log_sweep,
                 3: linear_2way_sweep,
                 4: log_2way_sweep}

        return modes[self.sweep_mode()](start_value, end_value, step_value)


class IVSweepMeasurement(MultiParameter):
    """
    IV sweep measurement outputs a list of primary and secondary
    parameter.

    Args:
        name: Name of the Parameter.
        instrument: Instrument to which this parameter communicates to.
    """

    def __init__(self, name, instrument, **kwargs):
        super().__init__(
            name,
            names=tuple(['gate_current', 'source_drain_current']),
            units=tuple(['A', 'A']),
            labels=tuple(['Gate Current', 'Source Current']),
            shapes=((1,),) * 2,
            setpoint_names=(('Voltage',),) * 2,
            setpoint_labels=(('Voltage',),) * 2,
            setpoint_units=(('V',),) * 2,
            **kwargs)
        self._instrument = instrument
        self.data = _FMTResponse(None, None, None, None)
        self.param1 = _FMTResponse(None, None, None, None)
        self.param2 = _FMTResponse(None, None, None, None)
        self.ac_voltage = _FMTResponse(None, None, None, None)
        self.dc_voltage = _FMTResponse(None, None, None, None)

    def get_raw(self):
        if not self._instrument.setup_fnc_already_run:
            raise Exception('Sweep setup has not yet been run successfully')

        num_steps = self._instrument.iv_sweep.steps()

        raw_data = self._instrument.ask(MessageBuilder().xe().message)
        parsed_data = parse_fmt_1_0_response(raw_data)

        self.param1 = _FMTResponse(
                *[parsed_data[i][::2] for i in range(0, 4)])
        self.param2 = _FMTResponse(
                *[parsed_data[i][1::2] for i in range(0, 4)])

        self.shapes = ((num_steps,),) * 2
        self.setpoints = ((self._instrument.iv_sweep_voltages(),),) * 2
