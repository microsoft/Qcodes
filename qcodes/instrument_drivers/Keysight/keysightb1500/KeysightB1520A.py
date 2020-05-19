import re
import textwrap
import numpy as np
from typing import Optional, TYPE_CHECKING, Tuple, Union, Dict, cast, Any

from qcodes.instrument.parameter import MultiParameter
from qcodes.instrument.group_parameter import GroupParameter, Group
from qcodes.instrument.channel import InstrumentChannel
import qcodes.utils.validators as vals

from .KeysightB1500_module import B1500Module, parse_dcorr_query_response, \
    format_dcorr_response, _DCORRResponse, parse_dcv_measurement_response, \
    _FMTResponse, parse_fmt_1_0_response
from .message_builder import MessageBuilder
from . import constants
from .constants import ModuleKind, ChNr, MM

if TYPE_CHECKING:
    from .KeysightB1500_base import KeysightB1500

_pattern = re.compile(r"((?P<status>\w)(?P<chnr>\w)(?P<dtype>\w))?"
                      r"(?P<value>[+-]\d{1,3}\.\d{3,6}E[+-]\d{2})")


class CVSweeper(InstrumentChannel):
    def __init__(self, parent: 'B1520A', name: str, **kwargs: Any):
        super().__init__(parent, name, **kwargs)

        self.add_parameter(name='sweep_auto_abort',
                           set_cmd=self._set_sweep_auto_abort,
                           set_parser=constants.Abort,
                           vals=vals.Enum(*list(constants.Abort)),
                           get_cmd=None,
                           docstring=textwrap.dedent("""
                           enables or disables the automatic abort function 
                           for the CV (DC bias) sweep measurement (MM18) and 
                           the pulsed bias sweep measurement (MM20). The 
                           automatic abort function stops the measurement 
                           when one of the following conditions occurs:
                               - NULL loop unbalance condition
                               - IV amplifier saturation condition
                               - Overflow on the AD converter
                           """))
        self.sweep_auto_abort.cache.set(constants.Abort.ENABLED)

        self.add_parameter(name='post_sweep_voltage_condition',
                           set_cmd=self._set_post_sweep_voltage_condition,
                           set_parser=constants.WMDCV.Post,
                           vals=vals.Enum(*list(constants.WMDCV.Post)),
                           get_cmd=None,
                           docstring=textwrap.dedent("""
                           This command also sets the post measurement 
                           condition of the MFCMU. After the measurement is 
                           normally completed, the DC bias sweep source 
                           forces the value specified by the post parameter, 
                           and the pulsed bias sweep source forces 
                           the pulse base value.
                           If the measurement is stopped by the automatic  
                           abort function, the DC bias sweep source forces 
                           the start value, and the pulsed bias sweep source 
                           forces the pulse base value after sweep.
                           """))
        self.post_sweep_voltage_condition.cache.set(constants.WMDCV.Post.START)

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
                            completion trigger. 0 to the value of ``delay``, with 0.1 ms 
                            resolution. Numeric expression. If this
                            parameter is not set, trigger delay will be 0.
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

        self._set_sweep_delays_group = Group([self.hold,
                                       self.delay,
                                       self.step_delay,
                                       self.trigger_delay,
                                       self.measure_delay],
                                      set_cmd='WTDCV '
                                              '{hold},'
                                              '{delay},'
                                              '{step_delay},'
                                              '{trigger_delay},'
                                              '{measure_delay}',
                                      get_cmd=self._get_sweep_delays(),
                                      get_parser=self._get_sweep_delays_parser)

    @staticmethod
    def _get_sweep_delays() -> str:
        msg = MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.CV_DC_BIAS_SWEEP_MEASUREMENT_SETTINGS
        )
        cmd = msg.message
        return cmd

    @staticmethod
    def _get_sweep_delays_parser(response: str) -> Dict[str, float]:
        match = re.search('WTDCV(?P<hold>.+?),(?P<delay>.+?),'
                          '(?P<step_delay>.+?),(?P<trigger_delay>.+?),'
                          '(?P<measure_delay>.+?)(;|$)',
                          response)
        if not match:
            raise ValueError('Sweep delays (WTDCV) not found.')

        out_str = match.groupdict()
        out_dict = {key: float(value) for key, value in out_str.items()}
        return out_dict

    def _set_sweep_auto_abort(self, val: Union[bool, constants.Abort]):
        msg = MessageBuilder().wmdcv(abort=val)
        self.write(msg.message)

    def _set_post_sweep_voltage_condition(
            self, val: Union[constants.WMDCV.Post, int]):
        msg = MessageBuilder().wmdcv(abort=self.sweep_auto_abort(), post=val)
        self.write(msg.message)


class B1520A(B1500Module):
    """
    Driver for Keysight B1520A Capacitance Measurement Unit module for B1500
    Semiconductor Parameter Analyzer.

    Args:
        parent: mainframe B1500 instance that this module belongs to
        name: Name of the instrument instance to create. If `None`
            (Default), then the name is autogenerated from the instrument
            class.
        slot_nr: Slot number of this module (not channel number)
    """
    phase_compensation_timeout = 60  # manual says around 30 seconds
    MODULE_KIND = ModuleKind.CMU

    def __init__(self, parent: 'KeysightB1500', name: Optional[str], slot_nr,
                 **kwargs):
        super().__init__(parent, name, slot_nr, **kwargs)

        self.channels = (ChNr(slot_nr),)
        self.setup_fnc_already_run = False
        self._ranging_mode: Union[constants.RangingMode, int] = \
            constants.RangingMode.AUTO
        self._measurement_range_for_non_auto: Optional[int] = None

        self.add_parameter(name="voltage_dc",
                           set_cmd=self._set_voltage_dc,
                           get_cmd=self._get_voltage_dc
                           )

        self.add_parameter(name="voltage_ac",
                           set_cmd=self._set_voltage_ac,
                           get_cmd=self._get_voltage_ac
                           )

        self.add_parameter(name="frequency",
                           set_cmd=self._set_frequency,
                           get_cmd=self._get_frequency
                           )

        self.add_parameter(name="capacitance",
                           get_cmd=self._get_capacitance,
                           snapshot_value=False)

        self.add_submodule('correction', Correction(self, 'correction'))

        self.add_parameter(name="phase_compensation_mode",
                           set_cmd=self._set_phase_compensation_mode,
                           get_cmd=None,
                           set_parser=constants.ADJ.Mode,
                           docstring=textwrap.dedent("""
            This parameter selects the MFCMU phase compensation mode. This
            command initializes the MFCMU. The available modes are captured 
            in :class:`constants.ADJ.Mode`:
 
                - 0: Auto mode. Initial setting.
                - 1: Manual mode.
                - 2: Load adaptive mode.
    
            For mode=0, the KeysightB1500 sets the compensation data 
            automatically. For mode=1, execute the 
            :meth:`phase_compensation` method ( the ``ADJ?`` command) to  
            perform the phase compensation and set the compensation data. 
            For mode=2, the KeysightB1500 performs the phase compensation 
            before every measurement. It is useful when there are wide load 
            fluctuations by changing the bias and so on."""))

        self.add_submodule('cv_sweep', CVSweeper(self, 'cv_sweep'))

        self.add_parameter(name='sweep_mode',
                           initial_value=constants.SweepMode.LINEAR,
                           vals=vals.Enum(*list(constants.SweepMode)),
                           set_parser=constants.SweepMode,
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
            Sweep mode. 
                1: Linear sweep (single stair, start to stop.)
                2: Log sweep (single stair, start to stop.)
                3: Linear sweep (double stair, start to stop to start.)
                4: Log sweep (double stair, start to stop to start.)
                           """))

        self.add_parameter(name='sweep_start',
                           initial_value=0,
                           unit='V',
                           vals=vals.Numbers(-25, 25),
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
            Start value of the DC bias sweep (in V). For the log  sweep, 
            start and stop must have the same polarity.
                           """))

        self.add_parameter(name='sweep_end',
                           initial_value=0,
                           unit='V',
                           vals=vals.Numbers(-25, 25),
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
            Stop value of the DC bias sweep (in V). For the log sweep, 
            start and stop must have the same polarity.
                           """))

        self.add_parameter(name='sweep_steps',
                           initial_value=1,
                           vals=vals.Ints(1, 1001),
                           parameter_class=GroupParameter,
                           docstring=textwrap.dedent("""
            Number of steps for staircase sweep. Possible  values from 1 to 
            1001"""))

        self.add_parameter(name='chan',
                           initial_value=self.channels[0],
                           parameter_class=GroupParameter)

        self._set_sweep_steps_group = Group([self.chan,
                                      self.sweep_mode,
                                      self.sweep_start,
                                      self.sweep_end,
                                      self.sweep_steps],
                                     set_cmd='WDCV '
                                             '{chan},'
                                             '{sweep_mode},'
                                             '{sweep_start},'
                                             '{sweep_end},'
                                             '{sweep_steps}',
                                     get_cmd=self._get_sweep_steps(),
                                     get_parser=self._get_sweep_steps_parser)

        self.add_parameter(name='adc_coef',
                           initial_value=1,
                           parameter_class=GroupParameter,
                           vals=vals.Ints(1, 1023),
                           docstring=textwrap.dedent("""
            Coefficient used to define the number of averaging samples or 
            the averaging time. Integer expression.
                - For mode=0: 1 to 1023. Initial setting/default setting is 2.
                - For mode=2: 1 to 100. Initial setting/default setting is 1.
            """))

        self.add_parameter(name='adc_mode',
                           initial_value=constants.ACT.Mode.PLC,
                           parameter_class=GroupParameter,
                           vals=vals.Enum(*list(constants.ACT.Mode)),
                           set_parser=constants.ACT.Mode,
                           docstring=textwrap.dedent("""
            Sets the number of averaging samples or the averaging time set 
            to the A/D converter of the MFCMU
            
                ``constants.ACT.Mode.AUTO``: Auto mode. Defines the number 
                of averaging samples given by the following formula. Then 
                initial averaging is the number of averaging samples 
                automatically set by the B1500 and you cannot change. 
            
                Number of averaging samples = N x initial averaging
            
                ``constants.ACT.Mode.PLC``: Power line cycle (PLC) mode. 
                Defines the averaging time given by the following formula. 
            
                Averaging time = N / power line frequency
                                       """))

        self._adc_group = Group([self.adc_mode, self.adc_coef],
                                set_cmd='ACT {adc_mode},{adc_coef}',
                                get_cmd=self._get_adc_mode(),
                                get_parser=self._get_adc_mode_parser)

        self.add_parameter(name='ranging_mode',
                           set_cmd=self._set_ranging_mode,
                           vals=vals.Enum(*list(constants.RangingMode)),
                           set_parser=constants.RangingMode,
                           get_cmd=None,
                           docstring=textwrap.dedent("""
            Specifies the measurement range or the measurement ranging type 
            of the MFCMU. In the initial setting, the auto ranging is set. 
            The range changing occurs immediately after the trigger 
            (that is, during the measurements).
            Possible ranging modes are autorange and fixed range.
                           """))

        self.add_parameter(name='measurement_range_for_non_auto',
                           set_cmd=self._set_measurement_range_for_non_auto,
                           get_cmd=None,
                           docstring=textwrap.dedent("""
            Measurement range. Needs to set when ``ranging_mode`` is set to 
            PLC. The value shoudl be integer 0 or more. 50 ohm, 100 ohm, 
            300 ohm, 1 kilo ohm, 3 kilo ohm, 10 kilo ohm, 30 kilo ohm, 
            100 kilo ohm, and 300 kilo ohm are selectable. Available 
            measurement ranges depend on the output signal frequency set by 
            the FC command.
                           """))

        self.add_parameter(name="measurement_mode",
                           get_cmd=None,
                           set_cmd=self._set_measurement_mode,
                           set_parser=MM.Mode,
                           vals=vals.Enum(*list(MM.Mode)),
                           docstring=textwrap.dedent("""
            Set measurement mode for this module.

            It is recommended for this parameter to use values from 
            :class:`.constants.MM.Mode` enumeration.

            Refer to the documentation of ``MM`` command in the programming 
            guide for more information.
                            """))

        self.add_parameter(name='impedance_model',
                           set_cmd=self._set_impedance_model,
                           get_cmd=None,
                           vals=vals.Enum(
                               *list(constants.IMP.MeasurementMode)),
                           set_parser=constants.IMP.MeasurementMode,
                           initial_value=constants.IMP.MeasurementMode.Cp_D,
                           docstring=textwrap.dedent("""
            The IMP command specifies the parameter measured by the MFCMU. 
            Look at the ``constants.IMP.MeasurementMode`` for all the modes. 
                           """))

        self.add_parameter(name='ac_dc_volt_monitor',
                           set_cmd=self._set_ac_dc_volt_monitor,
                           get_cmd=None,
                           vals=vals.Ints(0, 1),
                           initial_value=False,
                           docstring=textwrap.dedent("""
            This command enables or disables the data monitor and data 
            output of the MFCMU AC voltage and DC voltage.
                0: Disables the data monitor and output. Initial setting.
                1: Enables the data monitor and output.                        
                           """))

        self.add_parameter(name='cv_sweep_voltages',
                           get_cmd=self._cv_sweep_voltages,
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
                           parameter_class=CVSweepMeasurement,
                           docstring=textwrap.dedent("""
            This is MultiParameter. Running the sweep runs the measurement 
            on the list of values of cv_sweep_voltages. The output is a 
            primary parameter (for ex Capacitance) and a secondary  
            parameter (for ex Dissipation) both of whom use the same 
            setpoint cv_sweep_voltages. The impedance_model defines exactly 
            what will be the primary and secondary parameter. The default 
            case is Capacitance and Dissipation.
                           """))

    def _cv_sweep_voltages(self) -> tuple:
        sign = lambda s: s and (1, -1)[s < 0]
        start_value = self.sweep_start()
        end_value = self.sweep_end()
        step_value = self.sweep_steps()
        if self.sweep_mode() == 2 or self.sweep_mode() == 4:
            if not sign(start_value) == sign(self.sweep_end()):
                if sign(start_value) == 0:
                    start_value = sign(start_value) * 0.005  # resolution
                elif sign(end_value) == 0:
                    end_value = sign(end_value) * 0.005  # resolution
                else:
                    raise AssertionError("Polarity of start and end is not "
                                         "same.")

        def linear_sweep(start: float, end: float, steps: int) -> tuple:
            sweep_val = np.linspace(start, end, steps)
            return tuple(sweep_val)

        def log_sweep(start: float, end: float, steps: int) -> tuple:
            sweep_val = np.logspace(np.log10(start), np.log10(end), steps)
            return tuple(sweep_val)

        def linear_2way_sweep(start: float, end: float, steps: int) -> tuple:
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
                half_list = list(np.logspace(np.log10(start), np.log10(end),
                                             steps // 2))
                sweep_val = half_list + half_list[::-1]
            else:
                half_list = list(np.logspace(np.log10(start), np.log10(end),
                                             steps // 2, endpoint=False))
                sweep_val = half_list + [end] + half_list[::-1]
            return tuple(sweep_val)

        modes = {1: linear_sweep,
                 2: log_sweep,
                 3: linear_2way_sweep,
                 4: log_2way_sweep}

        return modes[self.sweep_mode()](start_value, end_value, step_value)

    def _set_voltage_dc(self, value: float) -> None:
        msg = MessageBuilder().dcv(self.channels[0], value)

        self.write(msg.message)

    def _set_voltage_ac(self, value: float) -> None:
        msg = MessageBuilder().acv(self.channels[0], value)

        self.write(msg.message)

    def _get_dcv(self) -> Dict[str, Union[str, float]]:
        if not self.is_enabled():
            raise RuntimeError("The channels are disabled. Cannot get value.")

        msg = MessageBuilder().lrn_query(self.channels[0])
        response = self.ask(msg.message)
        d = parse_dcv_measurement_response(response)
        return d

    def _get_voltage_dc(self) -> float:
        dcv = self._get_dcv()
        return float(dcv['voltage_dc'])

    def _get_voltage_ac(self) -> float:
        dcv = self._get_dcv()
        return float(dcv['voltage_ac'])

    def _get_frequency(self) -> float:
        dcv = self._get_dcv()
        return float(dcv['frequency'])

    def _set_frequency(self, value: float) -> None:
        msg = MessageBuilder().fc(self.channels[0], value)

        self.write(msg.message)

    def _get_capacitance(self) -> Tuple[float, float]:
        msg = MessageBuilder().tc(
            chnum=self.channels[0], mode=constants.RangingMode.AUTO
        )

        response = self.ask(msg.message)

        parsed = [item for item in re.finditer(_pattern, response)]

        if (
                len(parsed) != 2
                or parsed[0]["dtype"] != "C"
                or parsed[1]["dtype"] != "Y"
        ):
            raise ValueError("Result format not supported.")

        return float(parsed[0]["value"]), float(parsed[1]["value"])

    def _set_phase_compensation_mode(self, mode: constants.ADJ.Mode) -> None:
        msg = MessageBuilder().adj(chnum=self.channels[0], mode=mode)
        self.write(msg.message)

    def phase_compensation(
            self,
            mode: Optional[Union[constants.ADJQuery.Mode, int]] = None
    ) -> constants.ADJQuery.Response:
        """
        Performs the MFCMU phase compensation, sets the compensation
        data to the KeysightB1500, and returns the execution results.

        This method resets the MFCMU. Before executing this method, set the
        phase compensation mode to manual by using
        ``phase_compensation_mode`` parameter, and open the measurement
        terminals at the end of the device side. The execution of this
        method will take about 30 seconds (the visa timeout for it is
        controlled by :attr:`phase_compensation_timeout` attribute). The
        compensation data is cleared by turning the KeysightB1500 off.

        Args:
            mode: Command operation mode :class:`.constants.ADJQuery.Mode`.

                - 0: Use the last phase compensation data without measurement.
                - 1: Perform the phase compensation data measurement.

                If the mode parameter is not set, mode=1 is assumed by the
                instrument.

        Returns:
            Status result of performing the phase compensation as
            :class:`.constants.ADJQuery.Response`
        """
        with self.root_instrument.timeout.set_to(
                self.phase_compensation_timeout):
            msg = MessageBuilder().adj_query(chnum=self.channels[0],
                                             mode=mode)
            response = self.ask(msg.message)
        return constants.ADJQuery.Response(int(response))

    @staticmethod
    def _get_sweep_steps():
        msg = MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.CV_DC_BIAS_SWEEP_MEASUREMENT_SETTINGS
        )
        cmd = msg.message
        return cmd

    @staticmethod
    def _get_sweep_steps_parser(response: str) -> Dict[str, Union[int, float]]:
        match = re.search(r'WDCV(?P<chan>.+?),(?P<sweep_mode>.+?),'
                          r'(?P<sweep_start>.+?),(?P<sweep_end>.+?),'
                          r'(?P<sweep_steps>.+?)(;|$)',
                          response)
        if not match:
            raise ValueError('Sweep steps (WDCV) not found.')

        out_str = match.groupdict()
        out_dict = cast(Dict[str, Union[int, float]], out_str)
        out_dict['chan'] = int(out_dict['chan'])
        out_dict['sweep_mode'] = int(out_dict['sweep_mode'])
        out_dict['sweep_start'] = float(out_dict['sweep_start'])
        out_dict['sweep_end'] = float(out_dict['sweep_end'])
        out_dict['sweep_steps'] = int(out_dict['sweep_steps'])
        return out_dict

    @staticmethod
    def _get_adc_mode() -> str:
        msg = MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.MFCMU_ADC_SETTING
        )
        cmd = msg.message
        return cmd

    @staticmethod
    def _get_adc_mode_parser(response: str) -> Dict[str, int]:
        match = re.search(r'ACT(?P<adc_mode>.+?),(?P<adc_coef>.+?)$', response)
        if not match:
            raise ValueError('ADC mode and coef (ATC) not found.')

        out_str = match.groupdict()
        out_dict = {key: int(value) for key, value in out_str.items()}
        return out_dict

    def abort(self) -> None:
        """
        Aborts currently running operation and the subsequent execution.
        This does not abort the timeout process. Only when the kernel is
        free this command is executed and the further commands are aborted.
        """
        msg = MessageBuilder().ab()
        self.write(msg.message)

    def _set_measurement_mode(self, mode: Union[MM.Mode, int]) -> None:
        msg = MessageBuilder().mm(mode=mode, channels=[self.channels[0]])
        self.write(msg.message)

    def _set_impedance_model(self, val: Union[constants.IMP.MeasurementMode,
                                              int]) -> None:
        msg = MessageBuilder().imp(mode=val)
        self.write(msg.message)

    def _set_ac_dc_volt_monitor(self, val: bool) -> None:
        msg = MessageBuilder().lmn(enable_data_monitor=val)
        self.write(msg.message)

    def _set_ranging_mode(self, val: Union[constants.RangingMode, int]) -> None:
        self._ranging_mode = val
        if val == constants.RangingMode.AUTO:
            self._measurement_range_for_non_auto = None
        msg = MessageBuilder().rc(
            chnum=self.channels[0],
            ranging_mode=self._ranging_mode,
            measurement_range=self._measurement_range_for_non_auto
        )
        self.write(msg.message)

    def _set_measurement_range_for_non_auto(self, val: Optional[int]) -> None:
        self._measurement_range_for_non_auto = val
        msg = MessageBuilder().rc(
            chnum=self.channels[0],
            ranging_mode=self._ranging_mode,
            measurement_range=self._measurement_range_for_non_auto
        )
        self.write(msg.message)

    def setup_staircase_cv(
            self,
            v_start: float,
            v_end: float,
            n_steps: int,
            freq: float,
            ac_rms: float,
            post_sweep_voltage_condition: int = constants.WMDCV.Post.STOP,
            adc_mode: int = constants.ACT.Mode.PLC, adc_coef: int = 5,
            imp_model: int = constants.IMP.MeasurementMode.Cp_D,
            ranging_mode: int = constants.RangingMode.AUTO,
            fixed_range_val: int = None,
            hold_delay: float = 0,
            delay: float = 0,
            step_delay: float = 0,
            trigger_delay: float = 0,
            measure_delay: float = 0,
            abort_enabled: int = constants.Abort.ENABLED,
            sweep_mode: int = constants.SweepMode.LINEAR,
            volt_monitor: bool = True
    ):
        """
        Convenience function which requires all inputs to properly setup a
        CV sweep measurement.  Function sets parameters in the order given
        in the programming example in the manual.  Returns error status
        after setting all params.

        Args:
            v_start: Starting voltage for sweep

            v_end: End voltage for sweep

            n_steps: Number of steps in the sweep

            freq: frequency

            ac_rms: AC voltage

            post_sweep_voltage_condition: Source output value after the
                measurement is normally completed.

            adc_mode: Sets the number of averaging samples or
                the averaging time set to the A/D converter of the MFCMU.

            adc_coef: the number of averaging samples or the
                averaging time.

            imp_model: specifies the units of the parameter measured by the
                MFCMU.

            ranging_mode: Auto range or Fixed range

            fixed_range_val: Integer 0 or more. Available measurement ranges
                depend on the output signal frequency.

            hold_delay: Hold time (in seconds) that is the wait time after
                starting measurement and before starting delay time for the
                first step 0 to 655.35, with 10 ms resolution.

            delay: Delay time (in seconds) that is the wait time after
                starting to force a step output and before starting a step
                measurement.

            step_delay: Step delay time (in seconds) that is the wait time
                after starting a step measurement and before starting to
                force the next step output. 0 to 1, with 0.1 ms resolution.
                If  step_delay is shorter than the measurement time,
                the B1500 waits until the measurement completes, then forces
                the next step output.

            trigger_delay: Step source trigger delay time (in seconds) that
                is the wait time after completing a step output  setup and
                before sending a step output setup  completion trigger. 0 to
                delay, with 0.1 ms resolution.

            measure_delay: Step measurement trigger delay time (in
                seconds) that is the wait time after receiving a start step
                measurement trigger and before starting a step measurement.
                0 to 65.535, with 0.1 ms resolution.

            abort_enabled: Boolean, enables or disables the automatic abort
                function for the CV sweep measurement.

            sweep_mode: Linear sweep, log sweep, linear 2 way sweep or
                log 2 way sweep

            volt_monitor: Accepts Boolean. If True, CV sweep measurement
                outputs 4 parameter; primary parameter(for ex Capacitance),
                secondary parameter(for ex Dissipation), ac source voltage
                and dc source voltage. If False, the measurement only
                outputs primary and secondary parameter.

        """

        self.adc_mode(adc_mode)
        self.adc_coef(adc_coef)
        self.frequency(freq)
        self.voltage_ac(ac_rms)
        self.cv_sweep.sweep_auto_abort(abort_enabled)
        self.cv_sweep.post_sweep_voltage_condition(
            post_sweep_voltage_condition)
        self.cv_sweep.hold(hold_delay)
        self.cv_sweep.delay(delay)
        self.cv_sweep.step_delay(step_delay)
        self.cv_sweep.trigger_delay(trigger_delay)
        self.cv_sweep.measure_delay(measure_delay)
        self.sweep_mode(sweep_mode)
        self.sweep_start(v_start)
        self.sweep_end(v_end)
        self.sweep_steps(n_steps)
        self.measurement_mode(constants.MM.Mode.CV_DC_SWEEP)
        self.impedance_model(imp_model)
        self.ac_dc_volt_monitor(volt_monitor)
        self.ranging_mode(ranging_mode)
        self.measurement_range_for_non_auto(fixed_range_val)

        error_list, error = [], ''

        while error != '+0,"No Error."':
            error = self.root_instrument.error_message()
            error_list.append(error)

        if len(error_list) <= 1:
            self.setup_fnc_already_run = True
        else:
            raise Exception(f'Received following errors while trying to set '
                            f'staircase sweep {error_list}')


class CVSweepMeasurement(MultiParameter):
    """
    CV sweep measurement outputs a list of primary (capacitance) and secondary
    parameter (disipation).

    Args:
        name: Name of the Parameter.
        instrument: Instrument to which this parameter communicates to.
    """

    def __init__(self, name, instrument, **kwargs):
        super().__init__(
            name,
            names=tuple(['Capacitance', 'Dissipation']),
            units=tuple(['F', 'unit']),
            labels=tuple(['Parallel Capacitance', 'Dissipation factor']),
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
        self.power_line_frequency = 50
        self._fudge = 1.5 # fudge factor for setting timeout

    def get_raw(self):
        model = self._instrument.impedance_model()
        if model != constants.IMP.MeasurementMode.Cp_D:
            raise Exception('Run sweep only supports Cp_D impedance model')

        if not self._instrument.setup_fnc_already_run:
            raise Exception('Sweep setup has not yet been run successfully')

        delay_time = self._instrument.cv_sweep.step_delay()

        nplc = self._instrument.adc_coef()
        num_steps = self._instrument.sweep_steps()
        power_line_time_period = 1/self.power_line_frequency
        calculated_time = 2 * nplc * power_line_time_period * num_steps

        estimated_timeout = max(delay_time, calculated_time) * num_steps
        new_timeout = estimated_timeout * self._fudge

        with self.root_instrument.timeout.set_to(new_timeout):
            raw_data = self._instrument.ask(MessageBuilder().xe().message)
            parsed_data = parse_fmt_1_0_response(raw_data)

        if len(set(parsed_data.type)) == 2:
            self.param1 = _FMTResponse(
                *[parsed_data[i][::2] for i in range(0, 4)])
            self.param2 = _FMTResponse(
                *[parsed_data[i][1::2] for i in range(0, 4)])

            self.shapes = ((num_steps,),) * 2
            self.setpoints = ((self._instrument.cv_sweep_voltages(),),) * 2
        else:
            self.param1 = _FMTResponse(
                *[parsed_data[i][::4] for i in range(0, 4)])
            self.param2 = _FMTResponse(
                *[parsed_data[i][1::4] for i in range(0, 4)])
            self.ac_voltage = _FMTResponse(
                *[parsed_data[i][2::4] for i in range(0, 4)])
            self.dc_voltage = _FMTResponse(
                *[parsed_data[i][3::4] for i in range(0, 4)])

            self.shapes = ((len(self.dc_voltage.value),),) * 2
            self.setpoints = ((self.dc_voltage.value,),) * 2

        return self.param1.value, self.param2.value


class Correction(InstrumentChannel):
    """
    A Keysight B1520A CMU submodule for performing open/short/load corrections.
    """

    def __init__(self, parent: 'B1520A', name: str, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        self._chnum = parent.channels[0]

        self.add_submodule('frequency_list',
                           FrequencyList(self, 'frequency_list', self._chnum))

    def enable(self, corr: constants.CalibrationType) -> None:
        """
        This command enables the open/short/load correction. Before enabling a
        correction, perform the corresponding correction data measurement by
        using the :meth:`perform`.

        Args:
            corr: Depending on the the correction you want to perform,
                set this to OPEN, SHORT or LOAD. For ex: In case of open
                correction corr = constants.CalibrationType.OPEN.
        """
        msg = MessageBuilder().corrst(chnum=self._chnum,
                                      corr=corr,
                                      state=True)
        self.write(msg.message)

    def disable(self, corr: constants.CalibrationType) -> None:
        """
        This command disables an open/short/load correction.

        Args:
            corr: Correction type as in :class:`.constants.CalibrationType`
        """
        msg = MessageBuilder().corrst(chnum=self._chnum,
                                      corr=corr,
                                      state=False)
        self.write(msg.message)

    def is_enabled(self, corr: constants.CalibrationType
                   ) -> constants.CORRST.Response:
        """
        Query instrument to see if a correction of the given type is
        enabled.

        Args:
            corr: Correction type as in :class:`.constants.CalibrationType`
        """
        msg = MessageBuilder().corrst_query(chnum=self._chnum, corr=corr)

        response = self.ask(msg.message)
        return constants.CORRST.Response(int(response))

    def set_reference_values(self,
                             corr: constants.CalibrationType,
                             mode: constants.DCORR.Mode,
                             primary: float,
                             secondary: float) -> None:
        """
        This command disables the open/short/load correction function and
        defines the calibration value or the reference value of the
        open/short/load standard. Any previously measured correction data
        will be invalid after calling this method.

        Args:
            corr: Correction mode from :class:`.constants.CalibrationType`.
                OPEN for Open correction
                SHORT for Short correction
                LOAD for Load correction.
            mode:  Measurement mode from :class:`.constants.DCORR.Mode`
                Cp-G (for open correction)
                Ls-Rs (for short or load correction).
            primary: Primary reference value of the standard. Cp value for
                the open standard. in F. Ls value for the short or load
                standard. in H.
            secondary: Secondary reference value of the standard. G value
                for the open standard. in S. Rs value for the short or load
                standard. in Ω.
        """

        msg = MessageBuilder().dcorr(chnum=self._chnum,
                                     corr=corr,
                                     mode=mode,
                                     primary=primary,
                                     secondary=secondary)
        self.write(msg.message)

    def get_reference_values(self, corr: constants.CalibrationType) -> str:
        """
        This command returns the calibration values or the reference values of
        the open/short/load standard.

        Args:
            corr: Correction mode from :class:`.constants.CalibrationType`.
                OPEN for Open correction
                SHORT for Short correction
                LOAD for Load correction.

        Returns:
            A human-readable string with the correction mode
            :class:`.constants.DCORR.Mode` and its reference values
        """
        dcorr_response_tuple = self._get_reference_values(corr=corr)
        return format_dcorr_response(dcorr_response_tuple)

    def _get_reference_values(self, corr: constants.CalibrationType
                              ) -> _DCORRResponse:
        msg = MessageBuilder().dcorr_query(chnum=self._chnum, corr=corr)
        response = self.ask(msg.message)
        return parse_dcorr_query_response(response)

    def perform(self, corr: constants.CalibrationType
                ) -> constants.CORR.Response:
        """
        Perform Open/Short/Load corrections using this method. Refer to the
        example notebook to understand how each of the corrections are
        performed.

        Before executing this method, set the oscillator level of the MFCMU.

        If you use the correction standard, execute the
        :meth:`set_reference_values` method (corresponds to the ``DCORR``
        command) before this method because the calibration value or the
        reference value of the standard must be defined before performing
        the correction.

        Args:
            corr: Depending on the the correction you want to perform,
                set this to OPEN, SHORT or LOAD. For ex: In case of open
                correction corr = constants.CalibrationType.OPEN.

        Returns:
            Status of correction data measurement in the form of
            :class:`.constants.CORR.Response`
        """
        msg = MessageBuilder().corr_query(
            chnum=self._chnum,
            corr=corr
        )
        response = self.ask(msg.message)
        return constants.CORR.Response(int(response))

    def perform_and_enable(self, corr: constants.CalibrationType) -> str:
        """
        Perform the correction AND enable it. It is equivalent to calling
        :meth:`perform` and :meth:`enable` methods sequentially.

        Returns:
            A human readable string with status of the operation.
        """
        correction_status = self.perform(corr=corr)
        self.enable(corr=corr)

        is_enabled = self.is_enabled(corr=corr)
        response_out = f'Correction status {correction_status.name} and Enable' \
                       f' {is_enabled.name}'
        return response_out


class FrequencyList(InstrumentChannel):
    """
    A frequency list for open/short/load correction for Keysight B1520A CMU.
    """

    def __init__(self, parent: 'Correction', name: str, chnum: int, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)
        self._chnum = chnum

    def clear(self) -> None:
        """
        Remove all frequencies in the list for data correction.
        """
        self._clear(constants.CLCORR.Mode.CLEAR_ONLY)

    def clear_and_set_default(self) -> None:
        """
        Remove all frequencies in the list for data correction AND set the
        default frequency list.

        For the list of default frequencies, refer to the documentation of
        the ``CLCORR`` command in the programming manual.
        """
        self._clear(constants.CLCORR.Mode.CLEAR_AND_SET_DEFAULT_FREQ)

    def _clear(self, mode: constants.CLCORR.Mode) -> None:
        msg = MessageBuilder().clcorr(chnum=self._chnum, mode=mode)
        self.write(msg.message)

    def add(self, freq: float) -> None:
        """
        Append MFCMU output frequency for data correction in the list.

        The frequency value can be given with a certain resolution as per
        Table 4-18 in the programming manual (year 2016).
        """
        msg = MessageBuilder().corrl(chnum=self._chnum, freq=freq)
        self.write(msg.message)

    def query(self, index: Optional[int] = None) -> float:
        """
        Query the frequency list for CMU data correction.

        If ``index`` is ``None``, the query returns a total number of
        frequencies in the list. If ``index`` is given, then the query
        returns the frequency value from the list at that index.
        """
        msg = MessageBuilder().corrl_query(chnum=self._chnum,
                                           index=index)
        response = self.ask(msg.message)
        return float(response)
