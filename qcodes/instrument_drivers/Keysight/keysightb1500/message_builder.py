from typing import Iterable, Any

from functools import wraps
from operator import xor
from typing import List, Union, Callable, TypeVar, cast, Optional

from . import constants


def as_csv(l: Iterable, sep: str = ',') -> str:
    """Returns items in iterable ls as comma-separated string"""
    return sep.join(format(x) for x in l)


MessageBuilderMethodT = TypeVar('MessageBuilderMethodT',
                                bound=Callable[..., 'MessageBuilder'])


def final_command(f: MessageBuilderMethodT) -> MessageBuilderMethodT:
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> "MessageBuilder":
        res: 'MessageBuilder' = f(*args, **kwargs)
        res._msg.set_final()

        return res

    return cast(MessageBuilderMethodT, wrapper)


class CommandList(list):
    def __init__(self) -> None:
        super().__init__()
        self.is_final = False

    def append(self, obj: Any) -> None:
        if self.is_final:
            raise ValueError(f'Cannot add commands after `{self[-1]}`. '
                             f'`{self[-1]}` must be the last command in the '
                             f'message.')
        else:
            super().append(obj)

    def set_final(self) -> None:
        self.is_final = True

    def clear(self) -> None:
        self.is_final = False
        super().clear()

    def __str__(self) -> str:
        return as_csv(self, ';')


class MessageBuilder:
    """
    Provides a Python wrapper for each of the FLEX commands that the
    KeysightB1500 understands.

    To make usage easier also take a look at the classes defined in
    :mod:`~.keysightb1500.constants` which defines a lot of the integer
    constants that the commands expect as arguments.
    """

    def __init__(self) -> None:
        self._msg = CommandList()

    @property
    def message(self) -> str:
        joined = str(self._msg)
        if len(joined) > 250:
            raise Exception(
                f"Command is too long ({len(joined)}>256-termchars) "
                f"and will overflow input buffer of instrument. "
                f"(Consider using the ST/END/DO/RU commands for very long "
                f"programs.)")
        return joined

    def clear_message_queue(self) -> None:
        self._msg.clear()

    def aad(self,
            chnum: Union[constants.ChNr, int],
            adc_type: Union[constants.AAD.Type, int]
            ) -> 'MessageBuilder':
        """
        This command is used to specify the type of the A/D converter (ADC) for
        each measurement channel.

        Execution Conditions: Enter the AIT command to set up the ADC.

        The pulsed-measurement ADC is automatically used for the pulsed spot,
        pulsed sweep, multi channel pulsed spot, multi channel pulsed sweep, or
        staircase sweep with pulsed bias measurement, even if the
        ``AAD chnum,2`` command is not executed.
        The pulsed-measurement ADC is never used for the DC measurement.
        Even if the ``AAD chnum,2`` command is executed, the previous
        setting is still effective.

        Args:
            chnum: SMU measurement channel number. Integer expression. 1 to
                10 or 101 to 1001. See Table 4-1 on page 16.

            adc_type: Type of the A/D converter.
                Integer expression. 0, 1, or 2.

                    - 0: High-speed ADC for high speed DC measurement.
                        Initial setting.
                    - 1: High-resolution ADC. For high accurate DC
                        measurement. Not available for the HCSMU, HVSMU, and MCSMU.
                    - 2: High-speed ADC for pulsed-measurement

        """
        cmd = f'AAD {chnum},{adc_type}'

        self._msg.append(cmd)
        return self

    @final_command
    def ab(self) -> 'MessageBuilder':
        """
        The AB command aborts the present operation and subsequent command
        execution.

        This command stops the operation now in progress, such as the
        measurement execution, source setup changing, and so on. But this
        command does not change the present condition. For example, if the
        KeysightB1500 just keeps to force the DC bias, the AB command does not
        stop the DC bias output.

        If you start an operation that you may want to abort,
        do not send any command after the command or command string that
        starts the operation. If you do, the AB command cannot enter the
        command input buffer until the intervening command execution starts,
        so the operation cannot be aborted. In this case, use a device clear
        to end the operation.
        If the AB command is entered in a command string, the other commands
        in the string are not executed. For example, the CN command in the
        following command string is not executed.

        ``OUTPUT @KeysightB1500;"AB;CN"``

        During sweep measurement, if the KeysightB1500 receives the AB command,
        it returns only the measurement data obtained before abort. Then the
        dummy data is not returned.
        For the quasi-pulsed spot measurement, the KeysightB1500 cannot receive
        any command during the settling detection. So the AB command cannot
        abort the operation, and it will be performed after the settling
        detection.
        """
        cmd = 'AB'

        self._msg.append(cmd)
        return self

    def ach(self,
            actual: Optional[Union[constants.ChNr, int]] = None,
            program: Optional[Union[constants.ChNr, int]] = None
            ) -> 'MessageBuilder':
        """
        The ACH command translates the specified program channel number to
        the specified actual channel number at the program execution. This
        command is useful when you use a control program created for an
        instrument, such as the 4142B, 4155B/4155C/4156B/4156C/E5260/E5270,
        and KeysightB1500, that has a module configuration different from the
        KeysightB1500 actually you use. After the ACH command, enter the
        ``*OPC?`` command to confirm that the command execution is completed.

        Args:
            actual: Channel number actually set to the KeysightB1500 instead
                of program. Integer expression. 1 to 10 or 101 to 1002. See
                Table 4-1 on page 16.

            program: Channel number used in a program and will be replaced
                with actual. Integer expression. If you do not set program,
                this command is the same as ``ACH n,n``.

                If you do not set actual and program, all channel number
                mapping is cleared.

        Remarks: The ACH commands must be put at the beginning of the
        program or before the command line that includes a program channel
        number. In the program lines that follow the ACH command, you must
        leave the program channel numbers. The measurement data is returned
        as the data of the channel program, not actual.
        """
        if program is None:
            if actual is None:
                cmd = 'ACH'
            else:
                cmd = f'ACH {actual}'
        else:
            cmd = f'ACH {actual},{program}'

        self._msg.append(cmd)
        return self

    def act(self,
            mode: Union[constants.ACT.Mode, int],
            coeff: Optional[int] = None
            ) -> 'MessageBuilder':
        """
        This command sets the number of averaging samples or the averaging
        time set to the A/D converter of the MFCMU.

        Args:
            mode: Averaging mode.
                Integer expression. 0 (initial setting) or 2.

                    - 0: Auto mode: Defines the number of averaging samples
                        given by the following formula. Then initial averaging
                        is the number of averaging samples automatically set by
                        the KeysightB1500 and you cannot change.

                        ``Number of averaging samples = n * initial averaging``

                    - 2: Power line cycle (PLC) mode: Defines the averaging
                        time given by the following formula.

                        ``Averaging time = n / power line frequency``

            coeff: Coefficient used to define the number of averaging
                samples or the averaging time. For mode=0: 1 to 1023. Initial
                setting/default setting is 2. For mode=2: 1 to 100. Initial
                setting/default setting is 1.
        """
        cmd = f'ACT {mode}'

        if coeff is not None:
            cmd += f',{coeff}'

        self._msg.append(cmd)
        return self

    def acv(self,
            chnum: Union[constants.ChNr, int],
            voltage: float
            ) -> 'MessageBuilder':
        """
        This command sets the output signal level of the MFCMU, and starts
        the AC voltage output. Output signal frequency is set by the FC
        command.

        Execution conditions: The CN/CNX command has been executed for the
        specified channel.

        Args:
            chnum: MFCMU channel number. Integer expression. 1 to 10 or
                101 to 1001. See Table 4-1 on page 16.
            voltage: Oscillator level of the output AC voltage (in V).
                Numeric expression.
        """
        cmd = f'ACV {chnum},{voltage}'

        self._msg.append(cmd)
        return self

    def adj(self,
            chnum: Union[constants.ChNr, int],
            mode: Union[constants.ADJ.Mode, int]
            ) -> 'MessageBuilder':
        """
        This command selects the MFCMU phase compensation mode. This command
        initializes the MFCMU.

        Args:
            chnum: MFCMU channel number. Integer expression. 1 to 10 or
                101 to 1001. See Table 4-1 on page 16.
            mode: Phase compensation mode.
                Integer expression. 0 or 1.

                    - 0: Auto mode. Initial setting.
                    - 1: Manual mode.
                    - 2: Load adaptive mode.

                For mode=0, the KeysightB1500 sets the compensation data
                automatically. For mode=1, execute the ADJ? command to
                perform the phase compensation and set the compensation
                data. For mode=2, the KeysightB1500 performs the phase
                compensation before every measurement. It is useful when
                there are wide load fluctuations by changing the bias and so
                on.
        """
        cmd = f'ADJ {chnum},{mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def adj_query(self,
                  chnum: Union[constants.ChNr, int],
                  mode: Optional[Union[constants.ADJQuery.Mode, int]] = None
                  ) -> 'MessageBuilder':
        """
        This command performs the MFCMU phase compensation, and sets the
        compensation data to the KeysightB1500. This command also returns the
        execution results.

        This command resets the MFCMU. Before executing this command, set the
        phase compensation mode to manual by using the ADJ command. During this
        command, open the measurement terminals at the end of the device side.
        This command execution will take about 30 seconds. The compensation
        data is cleared by turning the KeysightB1500 off.

        Query response:

            - 0: Phase compensation measurement was normally completed.
            - 1: Phase compensation measurement failed.
            - 2: Phase compensation measurement was aborted.
            - 3: Phase compensation measurement has not been performed.

        Args:
            chnum: MFCMU channel number. Integer expression. 1 to 10 or
                101 to 1001. See Table 4-1 on page 16.
            mode: Command operation mode. 0: Use the last phase compensation
                data without measurement. 1: Perform the phase compensation
                data measurement. If the mode parameter is not set, mode=1
                is set.
        """
        cmd = f'ADJ? {chnum}'

        if mode is not None:
            cmd += f',{mode}'

        self._msg.append(cmd)
        return self

    def ait(self,
            adc_type: Union[constants.AIT.Type, int],
            mode: Union[constants.AIT.Mode, int],
            coeff: Optional[Union[int, float]] = None
            ) -> 'MessageBuilder':
        """
        This command is used to set the operation mode and the setup
        parameter of the A/D converter (ADC) for each ADC type.

        Execution conditions: Enter the AAD command to specify the ADC type
        for each measurement channel.

        The pulsed-measurement ADC (type=2) is available for the all
        measurement channels used for the pulsed spot, pulsed sweep,
        multi channel pulsed spot, multi channel pulsed sweep, or staircase
        sweep with pulsed bias measurement.

        Args:
            adc_type: Type of the A/D converter. Integer expression. 0, 1,
                or 2. 0: High-speed ADC 1: High-resolution ADC. Not
                available for the HCSMU, HVSMU and MCSMU. 2: High-speed ADC
                for pulsed-measurement
            mode: ADC operation mode. Integer expression. 0, 1, 2, or 3. 0:
                Auto mode. Initial setting. 1: Manual mode 2: Power line
                cycle (PLC) mode 3: Measurement time mode. Not available for
                the high-resolution ADC.
            coeff: Coefficient used to define the integration time or the
                number of averaging samples, integer expression, for mode=0, 1,
                and 2. Or the actual measurement time, numeric expression,
                for mode=3. See Table 4-10.
        """
        cmd = f'AIT {adc_type},{mode}'

        if coeff is not None:
            cmd += f',{coeff}'

        self._msg.append(cmd)
        return self

    def aitm(self, operation_type: Union[constants.APIVersion, int]
             ) -> 'MessageBuilder':
        """
        Only for the current measurement by using HRSMU. This command sets
        the operation type of the high-resolution ADC that is set to the
        power line cycle (PLC) mode by the AIT 1, 2, N command.

        This setting is cleared by the ``*RST`` or a device clear.

        Args:
            operation_type: Operation type.
                Integer expression. 0 or 1.

                    - 0: KeysightB1500 standard operation. Initial setting.
                    - 1: Classic operation. Performs the operation similar
                        to the PLC mode of Keysight 4156.

        """
        cmd = f'AITM {operation_type}'

        self._msg.append(cmd)
        return self

    @final_command
    def aitm_query(self) -> 'MessageBuilder':
        """
        This command returns the operation type of the high-resolution ADC
        that is set by the AITM command.
        """
        cmd = f'AITM?'

        self._msg.append(cmd)
        return self

    def als(self,
            chnum: Union[constants.ChNr, int],
            n_bytes: int,
            block: bytes) -> None:
        # The format specification in the manual is a bit unclear, and I do
        # not have the module installed to test this command, hence:
        raise NotImplementedError

        # A possible way might be:

        # No comma between n_bytes and block!
        # cmd = f'ALS {chnum},{n_bytes} {block.decode()}'
        # self._msg.append(cmd)
        # return self

    @final_command
    def als_query(self, chnum: Union[constants.ChNr, int]) -> 'MessageBuilder':
        """
        This query command returns the ALWG sequence data of the specified
        SPGU channel.

        Query response: Returns the ALWG sequence data (binary format,
        big endian).

        Args:
            chnum: SPGU channel number. Integer expression. 1 to 10 or 101
                to 1002. See Table 4-1.
        """
        cmd = f'ALS? {chnum}'

        self._msg.append(cmd)
        return self

    def alw(self,
            chnum: Union[constants.ChNr, int],
            n_bytes: int,
            block: bytes) -> None:
        # The format specification in the manual is a bit unclear, and I do
        # not have the module installed to test this command, hence:
        raise NotImplementedError

        # A possible way might be:

        # cmd = f'ALW {chnum},{n_bytes} {block.decode()}'
        # self._msg.append(cmd)
        # return self

    @final_command
    def alw_query(self, chnum: Union[constants.ChNr, int]) -> 'MessageBuilder':
        """
        This query command returns the ALWG pattern data of the specified
        SPGU channel.

        Query response: Returns the ALWG pattern data (binary format,
        big endian).

        Args:
            chnum: SPGU channel number. Integer expression. 1 to 10 or 101
                to 1002. See Table 4-1.
        """
        cmd = f'ALW? {chnum}'

        self._msg.append(cmd)
        return self

    def av(self,
           number: int,
           mode: Optional[Union[constants.AV.Mode, int]] = None
           ) -> 'MessageBuilder':
        """
        This command sets the number of averaging samples of the high-speed
        ADC (A/D converter). This command is not effective for the
        high-resolution ADC. This command is not effective for the
        measurements using pulse.

        Args:
            number: 1 to 1023, or -1 to -100.
                Initial setting is 1.

                For positive number input, this value specifies the number of
                samples depended on the mode value. See below.

                For negative number input, this parameter specifies the
                number of power line cycles (PLC) for one point
                measurement. The KeysightB1500 gets 128 samples in 1 PLC.
                Ignore the mode parameter.

            mode: Averaging mode.
                Integer expression. This parameter is meaningless for
                negative number.

                    - 0: Auto mode (default).
                        ``Number of samples = number * initial number``
                    - 1: Manual mode. Number of samples = number where
                        initial number means the number of samples the B1500
                        automatically sets and you cannot change. For voltage
                        measurement, initial number=1. For current measurement,
                        see Table 4-22. If you select the manual mode, number
                        must be initial number or more to satisfy the
                        specifications.

        """
        cmd = f'AV {number}'

        if mode is not None:
            cmd += f',{mode}'

        self._msg.append(cmd)
        return self

    def az(self, do_autozero: bool) -> 'MessageBuilder':
        """
        This command is used to enable or disable the ADC zero function that
        is the function to cancel offset of the high-resolution A/D
        converter. This function is especially effective for low voltage
        measurements. Power on, ``*RST`` command, and device clear disable the
        function. This command is effective for the high-resolution A/D
        converter, not effective for the high-speed A/D converter.

        Remarks: Set the function to OFF in cases that the measurement speed
        is more important than the measurement accuracy. This roughly halves
        the integration time.

        Args:
            do_autozero: True of False - Mode ON or OFF.
                False (0): OFF. Disables the function. Initial setting.
                True (1): ON. Enables the function.
        """
        cmd = f'AZ {int(do_autozero)}'

        self._msg.append(cmd)
        return self

    @final_command
    def bc(self) -> 'MessageBuilder':
        """
        The BC command clears the output data buffer that stores measurement
        data and query command response data. This command does not change
        the measurement settings.

        Note: Multi command statement is not allowed for this command.
        """
        cmd = 'BC'

        self._msg.append(cmd)
        return self

    def bdm(self,
            interval: Union[constants.BDM.Interval, int],
            mode: Optional[Union[constants.BDM.Mode, int]] = None
            ) -> 'MessageBuilder':
        """
        The BDM command specifies the settling detection interval and the
        measurement mode; voltage or current, for the quasi-pulsed
        measurements.

        Remarks: The following conditions must be true to perform the
        measurement successfully: When interval=0: A > 1 V/ms and B <= 3 s
        When interval=1: A > 0.1 V/ms and B <= 12 s where A means the slew
        rate when source output sweep was started, and B means the settling
        detection time. See "Quasi-Pulsed Spot Measurements" on page 2-18.
        These values depend on the conditions of cabling and device
        characteristics. And you cannot specify the values directly.

        Args:
            interval: Settling detection interval.
                Numeric expression.

                    - 0: Short. Initial setting.
                    - 1: Long. For measurements of the devices that have the
                        stray capacitance, or the measurements with the
                        compliance less than 1 uA

            mode: Measurement mode.
                Numeric expression.

                    - 0: Voltage measurement mode. Default setting.
                    - 1: Current measurement mode.
        """
        cmd = f'BDM {interval}'

        if mode is not None:
            cmd += f',{mode}'

        self._msg.append(cmd)
        return self

    def bdt(self, hold: float, delay: float) -> 'MessageBuilder':
        """
        The BDT command specifies the hold time and delay time for the
        quasi-pulsed measurements.

        Args:
            hold: Hold time (in sec). Numeric expression. 0 to 655.35 s,
                0.01 s resolution. Initial setting is 0.

            delay: Delay time (in sec). Numeric expression. 0 to 6.5535 s,
                0.0001 s resolution. Initial setting is 0.
        """
        cmd = f'BDT {hold},{delay}'

        self._msg.append(cmd)
        return self

    def bdv(self,
            chnum: Union[constants.ChNr, int],
            v_range: Union[constants.VOutputRange, int],
            start: float,
            stop: float,
            i_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        """
        The BDV command specifies the quasi-pulsed voltage source and its
        parameters.

        Remarks: The time forcing the stop value will be approximately
        1.5 ms to 1.8 ms with the following settings:

            - BDM, BDT command parameters: interval=0, mode=0, delay=0

            - AV or AAD/AIT command parameters: initial setting

        Args:
            chnum: SMU source channel number. Integer expression. 1 to 10
                or 101 to 1001. See Table 4-1 on page 16.
            v_range: Ranging type for quasi-pulsed source. Integer
                expression. The output range will be set to the minimum
                range that covers both start and stop values. For the
                limited auto ranging, the instrument never uses the range
                less than the specified range. See Table 4-4 on page 20.
            start: Start or stop voltage (in V). Numeric expression. See
                Table 4-7 on page 24. 0 to +-100 for MPSMU/HRSMU, or 0 to
                +-200 for HPSMU ``|start - stop|`` must be 10V or more.
            stop: similar to `start`
            i_comp: Current compliance (in A). Numeric expression.
                See Table 4-7 on page 4-24. If you do not set Icomp,
                the previous value is used. The compliance polarity is
                automatically set to the same polarity as the stop value,
                regardless of the specified Icomp value. If stop=0,
                the polarity is positive.
        """
        cmd = f'BDV {chnum},{v_range},{start},{stop}'

        if i_comp is not None:
            cmd += f',{i_comp}'

        self._msg.append(cmd)
        return self

    def bgi(self,
            chnum: Union[constants.ChNr, int],
            searchmode: Union[constants.BinarySearchMode, int],
            stop_condition: Union[float, int],
            i_range: Union[constants.IMeasRange, int],
            target: float
            ) -> 'MessageBuilder':
        """
        The BGI command sets the current monitor channel for the binary
        search measurement (MM15). This command setting clears, and is
        cleared by, the BGV command setting.

        This command ignores the RI command setting.

        Remarks: In the limit search mode, if search cannot find the search
        target and the following two conditions are satisfied,
        the KeysightB1500 repeats the binary search between the last source
        value and the source start value.

        - target is between the data at source start value and the last
          measurement data.
        - target is between the data at source stop value and the data at:
          source value = | stop - start | / 2.

        If the search cannot find the search target and the following two
        conditions are satisfied, the KeysightB1500 repeats the binary search
        between the last source value and the source stop value.

        - target is between the data at source stop value and the last
          measurement data.
        - target is between the data at source start value and the data at:
          source value = | stop - start | / 2.

        Args:
            chnum: SMU search monitor channel number. Integer expression.
                1 to 10 or 101 to 1001. See Table 4-1 on page 16.

            searchmode: Search mode (0:limit mode or 1:repeat mode)

            stop_condition: The meaning of stop_condition depends on the
                mode setting.

                if mode==0: Limit value for the search target (target). The
                search stops when the monitor data reaches target +-
                stop_condition. Numeric expression. Positive value. in A.
                Setting resolution: range/20000. where range means the
                measurement range actually used for the measurement.

                if mode==1: Repeat count. The search stops when the repeat
                count of the operation that changes the source output value
                is over the specified value. Numeric expression. 1 to 16.

            i_range: Measurement ranging type. Integer expression. The
                measurement range will be set to the minimum range that
                covers the target value. For the limited auto ranging,
                the instrument never uses the range less than the specified
                range. See Table 4-3 on page 19.

            target: Search target current (in A).
                Numeric expression.

                    - 0 to +-0.1 A (MPSMU/HRSMU/MCSMU).
                    - 0 to +-1 A (HPSMU/HCSMU).
                    - 0 to +-2 A (DHCSMU).
                    - 0 to +-0.008 A (HVSMU).
        """
        cmd = f'BGI {chnum},{searchmode},{stop_condition},{i_range},{target}'

        self._msg.append(cmd)
        return self

    def bgv(self,
            chnum: Union[constants.ChNr, int],
            searchmode: Union[constants.BinarySearchMode, int],
            stop_condition: Union[float, int],
            v_range: Union[constants.VMeasRange, int],
            target: float
            ) -> 'MessageBuilder':
        """
        The BGV command specifies the voltage monitor channel and its search
        parameters for the binary search measurement (MM15). This command
        setting clears, and is cleared by, the BGI command setting. This
        command ignores the RV command setting.

        Remarks: In the limit search mode, if search cannot find the search
        target and the following two conditions are satisfied,
        the KeysightB1500 repeats the binary search between the last source
        value and the source start value.

        - target is between the data at source start value and the last
          measurement data.
        - target is between the data at source stop value and the data at:
          ``source value = | stop - start | / 2``.

        If the search cannot find the search target and the following two
        conditions are satisfied, the KeysightB1500 repeats the binary search
        between the last source value and the source stop value.

        - target is between the data at source stop value and the last
          measurement data.
        - target is between the data at source start value and the data at:
          ``source value = | stop - start | / 2``.

        Args:
            chnum: SMU search monitor channel number. Integer expression.
                1 to 10 or 101 to 1001. See Table 4-1 on page 16.

            searchmode: Search mode (0:limit mode or 1:repeat mode)

            stop_condition: The meaning of stop_condition depends on the
                mode setting.

                if mode==0: Limit value for the search target (target). The
                search stops when the monitor data reaches target
                +- stop_condition. Numeric expression. Positive value. in V.
                Setting resolution: range/20000. where range means the
                measurement range actually used for the measurement.

                if mode==1: Repeat count. The search stops when the repeat
                count of the operation that changes the source output value
                is over the specified value. Numeric expression. 1 to 16.

            v_range: Measurement ranging type. Integer expression. The
                measurement range will be set to the minimum range that
                covers the target value. For the limited auto ranging,
                the instrument never uses the range less than the specified
                range. See Table 4-2 on page 17.

            target: Search target voltage (in V).
                Numeric expression.

                    - 0 to +-100 V (MPSMU/HRSMU) 0 to +-200 V (HPSMU)
                    - 0 to +-30 V (MCSMU)
                    - 0 to +-40 V (HCSMU/DHCSMU)
                    - 0 to +-3000 V (HVSMU)
        """
        cmd = f'BGV {chnum},{searchmode},{stop_condition},{v_range},{target}'

        self._msg.append(cmd)
        return self

    def bsi(self,
            chnum: Union[constants.ChNr, int],
            i_range: Union[constants.IOutputRange, int],
            start: float,
            stop: float,
            v_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        """
        The BSI command sets the current search source for the binary search
        measurement (MM15). After search stops, the search channel forces the
        value specified by the BSM command.

        This command clears the BSV, BSSI, and BSSV command settings. This
        command setting is cleared by the BSV command.

        Execution conditions: If Vcomp value is greater than the allowable
        voltage for the interlock open condition, the interlock circuit must
        be shorted.

        Args:
            chnum: SMU search source channel number. Integer expression.
                1 to 10 or 101 to 1001. See Table 4-1 on page 16.

            i_range: Output ranging type. Integer expression. The output
                range will be set to the minimum range that covers both
                start and stop values. For the limited auto ranging,
                the instrument never uses the range less than the specified
                range. See Table 4-5 on page 22.

            start: Search start or stop current (in A). Numeric expression.
                See Table 4-6 on page 23, Table 4-8 on page 25, or Table
                4-11 on page 27 for each measurement resource type. The
                start and stop must have different values.

            stop: Similar to `stop`

            v_comp: Voltage compliance value (in V). Numeric expression.
                See Table 4-6 on page 23, Table 4-8 on page 25, or Table
                4-11 on page 27 for each measurement resource type. If you
                do not specify Vcomp, the previous value is set.
        """
        cmd = f'BSI {chnum},{i_range},{start},{stop}'

        if v_comp is not None:
            cmd += f',{v_comp}'

        self._msg.append(cmd)
        return self

    def bsm(self,
            mode: Union[constants.BSM.Mode, int],
            abort: Union[constants.Abort, int],
            post: Optional[Union[constants.BSM.Post, int]] = None
            ) -> 'MessageBuilder':
        """
        The BSM command specifies the search source control mode in the
        binary search measurement (MM15), and enables or disables the
        automatic abort function. The automatic abort function stops the
        search operation when one of the following conditions occurs:

          - Compliance on the measurement channel
          - Compliance on the non-measurement channel
          - Overflow on the AD converter
          - Oscillation on any channel

        This command also sets the post search condition for the binary
        search sources. After the search measurement is normally completed,
        the binary search sources force the value specified by the post
        parameter.
        If the search operation is stopped by the automatic abort function,
        the binary search sources force the start value after search.

        Normal mode: The operation of the normal mode is explained below:
            1. The source channel forces the Start value, and the monitor
            channel executes a measurement.

            2. The source channel forces the Stop value, and the monitor
            channel executes a measurement. If the search target value is out
            of the range between the measured value at the Start value and the
            measured value at the Stop value, the search stops.

            3. The source channel forces the Stop-D/2 value (or Stop+D/2 if
            Start>Stop), and the monitor channel executes a measurement. If
            the search stop condition is not satisfied, the measured data is
            used to decide the direction (+ or –) of the next output change.
            The value of the change is always half of the previous change.

            4. Repeats the output change and measurement until the search
            stop condition is satisfied. For information on the search stop
            condition, see “BGI” or “BGV”. If the output change value is less
            than the setting resolution, the search stops.

        Cautious mode: The operation of the cautious mode is explained below:
            1. The source channel forces the Start value, and the monitor
            channel executes a measurement.

            2. The source channel forces the Start+D/2 value (or Start-D/2 if
            Start>Stop), and the monitor channel executes a measurement. If
            the search stop condition is not satisfied, the measured data is
            used to decide the direction (+ or –) of the next output change.
            The value of the change is always half of the previous change.

            3. Repeats the output change and measurement until the search
            stop condition is satisfied. For information on the search stop
            condition, see “BGI” or “BGV”. If the output change value is
            less than the setting resolution, the search stops.

        Args:
            mode: Source output control mode, 0 (normal mode)
                or 1 (cautious mode). If you do not enter this command,
                the normal mode is set. See Figure 4-2.

            abort: Automatic abort function.
                Integer expression.

                    - 1: Disables the function. Initial setting.
                    - 2: Enables the function.

            post: Source output value after the search operation is
                normally completed. Integer expression.

                    - 1: Start value. Initial setting.
                    - 2: Stop value.
                    - 3: Output value when the search target value is get.

                If this parameter is not set, the search source forces the
                start value.
        """
        cmd = f'BSM {mode},{abort}'

        if post is not None:
            cmd += f',{post}'

        self._msg.append(cmd)
        return self

    def bssi(self,
             chnum: Union[constants.ChNr, int],
             polarity: Union[constants.Polarity, int],
             offset: float,
             v_comp: Optional[float] = None
             ) -> 'MessageBuilder':
        """
        The BSSI command sets the synchronous current source for the binary
        search measurement (MM15). The synchronous source output will be:
        Synchronous source output = polarity * BSI source output + offset
        where BSI source output means the output set by the BSI command. This
        command setting is cleared by the BSV/BSI command.

        Execution conditions: The BSI command must be sent before sending
        this command.

        See also: For the source output value, output range, and the
        available compliance values, see Table 4-6 on page 23, Table 4-8 on
        page 25, or Table 4-11 on page 27 for each measurement resource type.

        Args:
            chnum: SMU synchronous source channel number. Integer
                expression. 1 to 10 or 101 to 1001. See Table 4-1 on page 16.

            polarity: Polarity of the BSSI output for the BSI output.
                0: Negative. ``BSSI output = -BSI output + offset``
                1: Positive. ``BSSI output = BSI output + offset``

            offset: Offset current (in A). Numeric expression. See Table
                4-6 on page 23, Table 4-8 on page 25, or Table 4-11 on page
                27 for each measurement resource type. Both primary and
                synchronous search sources will use the same output range.
                So check the output range set to the BSI command to
                determine the synchronous source outputs.

            v_comp: Voltage compliance value (in V). Numeric expression. If
                you do not specify Vcomp, the previous value is set.
        """
        cmd = f'BSSI {chnum},{polarity},{offset}'

        if v_comp is not None:
            cmd += f',{v_comp}'

        self._msg.append(cmd)
        return self

    def bssv(self,
             chnum: Union[constants.ChNr, int],
             polarity: Union[constants.Polarity, int],
             offset: float,
             i_comp: Optional[float] = None
             ) -> 'MessageBuilder':
        """
        The BSSV command sets the synchronous voltage source for the binary
        search measurement (MM15). The synchronous source output will be:
        Synchronous source output = polarity * BSV source output + offset
        where BSV source output means the output set by the BSV command. This
        command setting is cleared by the BSI/BSV command.

        Execution conditions: The BSV command must be sent before sending this
        command.

        See also: For the source output value, output range, and the
        available compliance values, see Table 4-7 on page 24, Table 4-9 on
        page 26, Table 4-12 on page 27, or Table 4-15 on page 28 for each
        measurement resource type.

        Args:
            chnum: SMU synchronous source channel number. Integer
                expression. 1 to 10 or 101 to 1001. See Table 4-1 on page 16.

            polarity: Polarity of the BSSV output for the BSV output.
                0: Negative. ``BSSV output = -BSV output + offset``
                1: Positive. ``BSSV output = BSV output + offset``

            offset: Offset voltage (in V). Numeric expression.
                See Table 4-7 on page 24, Table 4-9 on page 26, Table 4-12
                on page 27, or Table 4-15 on page 28 for each measurement
                resource type. Both primary and synchronous search sources
                will use the same output range. So check the output range
                set to the BSV command to determine the synchronous source
                outputs.

            i_comp: Current compliance value (in A). Numeric expression.
                If you do not specify Icomp, the previous value is set.
                Zero amps (0 A) is not a valid value for the Icomp parameter.
        """
        cmd = f'BSSV {chnum},{polarity},{offset}'

        if i_comp is not None:
            cmd += f',{i_comp}'

        self._msg.append(cmd)
        return self

    def bst(self, hold: float, delay: float) -> 'MessageBuilder':
        """
        The BST command sets the hold time and delay time for the binary
        search measurement (MM15). If you do not enter this command,
        all parameters are set to 0.

        Args:
            hold: Hold time (in seconds) that is the wait time after
                starting the search measurement and before starting the
                delay time for the first search point. Numeric expression.
                0 to 655.35 sec. 0.01 sec resolution.

            delay: Delay time (in seconds) that is the wait time after
                starting to force a step output value and before starting a
                step measurement. Numeric expression. 0 to 65.535 sec.
                0.0001 sec resolution.
        """
        cmd = f'BST {hold},{delay}'

        self._msg.append(cmd)
        return self

    def bsv(self,
            chnum: Union[constants.ChNr, int],
            v_range: Union[constants.VOutputRange, int],
            start: float,
            stop: float,
            i_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        """
        The BSV command sets the voltage search source for the binary search
        measurement (MM15). After search stops, the search channel forces the
        value specified by the BSM command. This command clears the BSI,
        BSSI, and BSSV command settings. This command setting is cleared by
        the BSI command.

        Execution conditions: If the output voltage is greater than the
        allowable voltage for the interlock open condition, the interlock
        circuit must be shorted.

        Args:
            chnum: SMU search source channel number. Integer
                expression. 1 to 10 or 101 to 1001. See Table 4-1 on page 16.

            v_range: Output ranging type. Integer expression. The
                output range will be set to the minimum range that covers both
                start and stop values. For the limited auto ranging,
                the instrument never uses the range less than the specified
                range. See Table 4-4 on page 20.

            start: Search start or stop voltage (in V). Numeric
                expression. See Table 4-7 on page 24, Table 4-9 on page 26,
                Table 4-12 on page 27, or Table 4-15 on page 28 for each
                measurement resource type. The start and stop parameters must
                have different values.

            stop: Stimilar to start

            i_comp: Current compliance value (in A). Numeric expression.
                See Table 4-7 on page 24, Table 4-9 on page 26, Table 4-12
                on page 27, or Table 4-15 on page 28 for each measurement
                resource type. If you do not specify Icomp, the previous
                value is set. Zero amps (0 A) is not allowed for Icomp.
        """
        cmd = f'BSV {chnum},{v_range},{start},{stop}'

        if i_comp is not None:
            cmd += f',{i_comp}'

        self._msg.append(cmd)
        return self

    def bsvm(self, mode: Union[constants.BSVM.DataOutputMode, int]
             ) -> 'MessageBuilder':
        """
        The BSVM command selects the data output mode for the binary search
        measurement (MM15).

        Args:
            mode: Data output mode. Integer expression. 0: Returns
                Data_search only (initial setting). 1: Returns Data_search and
                Data_sense. Data_search is the value forced by the search output
                channel set by BSI or BSV. Data_sense is the value measured by
                the monitor channel set by BGI or BGV. For data output format,
                refer to “Data Output Format” on page 1-25.
        """
        cmd = f'BSVM {mode}'

        self._msg.append(cmd)
        return self

    def ca(self, slot: Optional[Union[constants.SlotNr, int]] = None
           ) -> 'MessageBuilder':
        """
        This command performs the self-calibration.

        The ``*OPC?`` command should be entered after this command to confirm
        the completion of the self-calibration. Module condition after this
        command is the same as the condition by the CL command.

        Execution conditions: No channel must be in the high voltage state (
        forcing more than the allowable voltage for the interlock open
        condition, or voltage compliance set to more than it). Before
        starting the calibration, open the measurement terminals.

        Remarks: Failed modules are disabled, and can only be enabled by the
        RCV command.

        Note: To send CA command to Keysight KeysightB1500 installed with ASU
        If you send the CA command to the KeysightB1500 installed with the ASU
        (Atto Sense and Switch Unit), the KeysightB1500 executes the
        self-calibration and the 1 pA range offset measurement for the
        measurement channels connected to the ASUs. The offset data is
        temporarily memorized until the KeysightB1500 is turned off, and is
        used for the compensation of the data measured by the 1 pA range of the
        channels. The KeysightB1500 performs the data compensation
        automatically and returns the compensated data. Since the
        KeysightB1500 is turned on, if you do not send the CA command,
        the KeysightB1500 performs the data compensation by using the
        pre-stored offset data.

        Args:
            slot: Slot number where the module under self-calibration
                has been installed. 1 to 10. Integer expression.

                If slot is not specified, the self-calibration is performed
                for the mainframe and all modules.

                If slot specifies the slot that installs no module,
                this command causes an error.
        """
        cmd = 'CA'

        if slot is not None:
            cmd += f' {slot}'

        self._msg.append(cmd)
        return self

    @final_command
    def cal_query(self, slot: Optional[Union[constants.SlotNr, int]] = None
                  ) -> 'MessageBuilder':
        """
        This query command performs the self-calibration, and returns the
        results. After this command, read the results soon. Module condition
        after this command is the same as the condition by the CL command.

        Execution Conditions: No channel must be in the high voltage state
        (forcing more than the allowable voltage for the interlock open
        condition, or voltage compliance set to more than it).

        Before starting the calibration, open the measurement terminals.

        Args:
            slot: Slot number where the module under self-calibration has
                been installed. 1 to 10. Or 0 or 11. Integer expression.

                    - 0: All modules and mainframe. Default setting.
                    - 11: Mainframe.

                If slot specifies the slot that installs no module,
                this command causes an error.
        """
        cmd = '*CAL?'

        if slot is not None:
            cmd += f' {slot}'

        self._msg.append(cmd)
        return self

    def cl(self, channels: Optional[constants.ChannelList] = None
           ) -> 'MessageBuilder':
        if channels is None:
            cmd = 'CL'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'CL { as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def clcorr(self,
               chnum: Union[constants.ChNr, int],
               mode: Union[constants.CLCORR.Mode, int]
               ) -> 'MessageBuilder':
        """
        This query command disables the open/short/load correction function
        and clears the frequency list for the correction data measurement.
        The correction data will be invalid after this command.

        Args:
            chnum: SMU search source channel number. Integer expression. 1
                to 10 or 101 to 1001. See Table 4-1 on page 16.
            mode: clear correction mode (:class:`.constants.CLCORR.Mode`),
                Mode options 1 or 2.

                    - 1. Just clears the frequency list.
                    - 2. Clears the frequency list and sets the default

                frequencies. For the list of default frequencies, refer to
                the documentation of the ``CLCORR`` command in the
                programming manual.
        """
        cmd = f'CLCORR {chnum},{mode}'

        self._msg.append(cmd)
        return self

    def cm(self, do_autocal: bool) -> 'MessageBuilder':
        cmd = f'CM {int(do_autocal)}'

        self._msg.append(cmd)
        return self

    def cmm(self,
            chnum: Union[constants.ChNr, int],
            mode: Union[constants.CMM.Mode, int]
            ) -> 'MessageBuilder':
        cmd = f'CMM {chnum},{mode}'

        self._msg.append(cmd)
        return self

    def cn(self, channels: Optional[constants.ChannelList] = None
           ) -> 'MessageBuilder':
        if channels is None:
            cmd = 'CN'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'CN {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def cnx(self, channels: Optional[constants.ChannelList] = None
            ) -> 'MessageBuilder':
        if channels is None:
            cmd = 'CNX'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'CNX {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    @final_command
    def corr_query(self,
                   chnum: Union[constants.ChNr, int],
                   corr: Union[constants.CalibrationType, int]
                   ) -> 'MessageBuilder':
        cmd = f'CORR? {chnum},{corr}'

        self._msg.append(cmd)
        return self

    def corrdt(self,
               chnum: Union[constants.ChNr, int],
               freq: float,
               open_r: float,
               open_i: float,
               short_r: float,
               short_i: float,
               load_r: float,
               load_i: float
               ) -> 'MessageBuilder':
        cmd = f'CORRDT {chnum},{freq},{open_r},{open_i},{short_r},' \
              f'{short_i},{load_r},{load_i}'

        self._msg.append(cmd)
        return self

    @final_command
    def corrdt_query(self, chnum: Union[constants.ChNr, int], index: int
                     ) -> 'MessageBuilder':
        cmd = f'CORRDT? {chnum},{index}'

        self._msg.append(cmd)
        return self

    def corrl(self, chnum: Union[constants.ChNr, int], freq: float
              ) -> 'MessageBuilder':
        cmd = f'CORRL {chnum},{freq}'

        self._msg.append(cmd)
        return self

    @final_command
    def corrl_query(self,
                    chnum: Union[constants.ChNr, int],
                    index: Optional[int] = None
                    ) -> 'MessageBuilder':
        cmd = f'CORRL? {chnum}'

        if index is not None:
            cmd += f',{index}'

        self._msg.append(cmd)
        return self

    @final_command
    def corrser_query(self,
                      chnum: Union[constants.ChNr, int],
                      use_immediately: bool,
                      delay: float,
                      interval: float,
                      count: int
                      ) -> 'MessageBuilder':
        cmd = f'CORRSER? {chnum},{int(use_immediately)},{delay},{interval},' \
              f'{count}'

        self._msg.append(cmd)
        return self

    def corrst(self,
               chnum: Union[constants.ChNr, int],
               corr: Union[constants.CalibrationType, int],
               state: bool
               ) -> 'MessageBuilder':
        cmd = f'CORRST {chnum},{corr},{int(state)}'

        self._msg.append(cmd)
        return self

    @final_command
    def corrst_query(self,
                     chnum: Union[constants.ChNr, int],
                     corr: Union[constants.CalibrationType, int]
                     ) -> 'MessageBuilder':
        cmd = f'CORRST? {chnum},{corr}'

        self._msg.append(cmd)
        return self

    def dcorr(self, chnum: Union[constants.ChNr, int],
              corr: Union[constants.CalibrationType, int],
              mode: Union[constants.DCORR.Mode, int],
              primary: float,
              secondary: float
              ) -> 'MessageBuilder':
        cmd = f'DCORR {chnum},{corr},{mode},{primary},{secondary}'

        self._msg.append(cmd)
        return self

    @final_command
    def dcorr_query(self,
                    chnum: Union[constants.ChNr, int],
                    corr: Union[constants.CalibrationType, int]
                    ) -> 'MessageBuilder':
        cmd = f'DCORR? {chnum},{corr}'

        self._msg.append(cmd)
        return self

    def dcv(self, chnum: Union[constants.ChNr, int], voltage: float
            ) -> 'MessageBuilder':
        """
        This command forces DC bias (voltage, up to +- 25 V) from the MFCMU.
        When the SCUU (SMU CMU unify unit) is connected, output up to +- 100 V
        is available by using the SMU that can be connected to the
        Force1/Sense1 terminals.

        Execution conditions: The CN/CNX command has been executed for the
        specified channel.

        If you want to apply DC voltage over +- 25 V, the SCUU must be
        connected correctly. The SCUU can be used with the MFCMU and two
        SMUs (MPSMU or HRSMU). The SCUU cannot be used if the HPSMU is
        connected to the SCUU or if the number of SMUs connected to the SCUU
        is only one.

        If the output voltage is greater than the allowable voltage for the
        interlock open condition, the interlock circuit must be shorted.

        Args:
            chnum: MFCMU channel number. Integer expression. 1 to 10 or
                101 to 1001. See Table 4-1 on page 16.

            voltage: DC voltage (in V). Numeric expression.
                0 (initial setting) to +- 25 V (MFCMU) or +- 100 V (with SCUU).
                With the SCUU, the source module is automatically selected by
                the setting value. The MFCMU is used if voltage is below
                +- 25 V (setting resolution: 0.001 V), or the SMU is used if
                voltage is greater than +- 25 V (setting resolution: 0.005 V).
                The SMU will operate with the 100 V limited auto ranging and
                20 mA compliance settings.
        """
        cmd = f'DCV {chnum},{voltage}'

        self._msg.append(cmd)
        return self

    def di(self,
           chnum: Union[constants.ChNr, int],
           i_range: Union[constants.IOutputRange, int],
           current: float,
           v_comp: Optional[float] = None,
           comp_polarity: Optional[Union[constants.CompliancePolarityMode,
                                         int]] = None,
           v_range: Optional[Union[constants.VOutputRange, int]] = None
           ) -> 'MessageBuilder':
        cmd = f'DI {chnum},{i_range},{current}'

        if v_comp is not None:
            cmd += f',{v_comp}'

            if comp_polarity is not None:
                cmd += f',{comp_polarity}'

                if v_range is not None:
                    cmd += f',{v_range}'

        self._msg.append(cmd)
        return self

    @final_command
    def diag_query(self, item: Union[constants.DIAG.Item, int]
                   ) -> 'MessageBuilder':
        cmd = f'DIAG? {item}'

        self._msg.append(cmd)
        return self

    def do(self, program_numbers: List[int]) -> 'MessageBuilder':
        if len(program_numbers) > 8:
            raise ValueError("A maximum of 8 programs can be specified.")
        else:
            cmd = f'DO {as_csv(program_numbers)}'

        self._msg.append(cmd)
        return self

    def dsmplarm(self, chnum: Union[constants.ChNr, int], count: int
                 ) -> 'MessageBuilder':
        event_type = 1  # No other option in user manual, so hard coded here
        cmd = f'DSMPLARM {chnum},{event_type},{count}'

        self._msg.append(cmd)
        return self

    def dsmplflush(self, chnum: Union[constants.ChNr, int]
                   ) -> 'MessageBuilder':
        cmd = f'DSMPLFLUSH {chnum}'

        self._msg.append(cmd)
        return self

    def dsmplsetup(self,
                   chnum: Union[constants.ChNr, int],
                   count: int,
                   interval: float,
                   delay: Optional[float] = None
                   ) -> 'MessageBuilder':
        cmd = f'DSMPLSETUP {chnum},{count},{interval}'

        if delay is not None:
            cmd += f',{delay}'

        self._msg.append(cmd)
        return self

    def dv(self,
           chnum: Union[constants.ChNr, int],
           v_range: Union[constants.VOutputRange, int],
           voltage: float,
           i_comp: Optional[float] = None,
           comp_polarity: Optional[Union[constants.CompliancePolarityMode,
                                         int]] = None,
           i_range: Optional[Union[constants.IOutputRange, int]] = None
           ) -> 'MessageBuilder':
        cmd = f'DV {chnum},{v_range},{voltage}'

        if i_comp is not None:
            cmd += f',{i_comp}'

            if comp_polarity is not None:
                cmd += f',{comp_polarity}'

                if i_range is not None:
                    cmd += f',{i_range}'

        self._msg.append(cmd)
        return self

    def dz(self, channels: Optional[constants.ChannelList] = None
           ) -> 'MessageBuilder':
        if channels is None:
            cmd = 'DZ'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'DZ {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    @final_command
    def emg_query(self, errorcode: int) -> 'MessageBuilder':
        cmd = f'EMG? {errorcode}'

        self._msg.append(cmd)
        return self

    def end(self) -> 'MessageBuilder':
        cmd = 'END'

        self._msg.append(cmd)
        return self

    def erc(self, value: int) -> 'MessageBuilder':
        mode = 2  # Only 2 is valid for KeysightB1500
        cmd = f'ERC {mode},{value}'

        self._msg.append(cmd)
        return self

    def ercmaa(self,
               mfcmu: Union[constants.SlotNr, int],
               hvsmu: Union[constants.SlotNr, int],
               mpsmu: Union[constants.SlotNr, int]
               ) -> 'MessageBuilder':
        cmd = f'ERCMAA {mfcmu},{hvsmu},{mpsmu}'

        self._msg.append(cmd)
        return self

    @final_command
    def ercmaa_query(self) -> 'MessageBuilder':
        cmd = f'ERCMAA?'

        self._msg.append(cmd)
        return self

    def ercmagrd(self,
                 guard_mode: Optional[Union[constants.ERCMAGRD.Guard,
                                            int]] = None
                 ) -> 'MessageBuilder':
        cmd = 'ERCMAGRD'

        if guard_mode is not None:
            cmd += f' {guard_mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def ercmagrd_query(self) -> 'MessageBuilder':
        cmd = 'ERCMAGRD?'

        self._msg.append(cmd)
        return self

    def ercmaio(self, cmhl=None, acgs=None, bias=None, corr=None
                ) -> 'MessageBuilder':
        cmd = f'ERCMAIO'

        if cmhl is not None:
            cmd += f' {cmhl}'

            if acgs is not None:
                cmd += f',{acgs}'

                if bias is not None:
                    cmd += f',{bias}'

                    if corr is not None:
                        cmd += f',{corr}'

        self._msg.append(cmd)
        return self

    @final_command
    def ercmaio_query(self) -> 'MessageBuilder':
        cmd = 'ERCMAIO?'

        self._msg.append(cmd)
        return self

    def ercmapfgd(self) -> 'MessageBuilder':
        cmd = 'ERCMAPFGD'

        self._msg.append(cmd)
        return self

    def erhpa(self,
              hvsmu: Union[constants.ChNr, int],
              hcsmu: Union[constants.ChNr, int],
              hpsmu: Union[constants.ChNr, int]
              ) -> 'MessageBuilder':
        cmd = f'ERHPA {hvsmu},{hcsmu},{hpsmu}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpa_query(self) -> 'MessageBuilder':
        cmd = 'ERHPA?'

        self._msg.append(cmd)
        return self

    def erhpe(self, onoff: bool) -> 'MessageBuilder':
        cmd = f'ERHPE {int(onoff)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpe_query(self) -> 'MessageBuilder':
        cmd = 'ERHPE?'

        self._msg.append(cmd)
        return self

    def erhpl(self, onoff: bool) -> 'MessageBuilder':
        cmd = f'ERHPL {int(onoff)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpl_query(self) -> 'MessageBuilder':
        cmd = 'ERHPL?'

        self._msg.append(cmd)
        return self

    def erhpp(self, path: Union[constants.ERHPP.Path, int]
              ) -> 'MessageBuilder':
        cmd = f'ERHPP {path}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpp_query(self) -> 'MessageBuilder':
        cmd = 'ERHPP?'

        self._msg.append(cmd)
        return self

    def erhpqg(self, state: Union[constants.ERHPQG.State, int]
               ) -> 'MessageBuilder':
        cmd = f'ERHPQG {state}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpqg_query(self) -> 'MessageBuilder':
        cmd = 'ERHPQG?'

        self._msg.append(cmd)
        return self

    def erhpr(self,
              pin: int,
              state: bool) -> 'MessageBuilder':
        cmd = f'ERHPR {pin},{int(state)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpr_query(self,
                    pin: int) -> 'MessageBuilder':
        cmd = f'ERHPR? {pin}'

        self._msg.append(cmd)
        return self

    def erhps(self, onoff: bool) -> 'MessageBuilder':
        cmd = f'ERHPS {int(onoff)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhps_query(self) -> 'MessageBuilder':
        cmd = 'ERHPS?'

        self._msg.append(cmd)
        return self

    def erhvca(self,
               vsmu: Union[constants.SlotNr, int],
               ismu: Union[constants.SlotNr, int],
               hvsmu: Union[constants.SlotNr, int]
               ) -> 'MessageBuilder':
        cmd = f'ERHVCA {vsmu},{ismu},{hvsmu}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhvca_query(self) -> 'MessageBuilder':
        cmd = 'ERHVCA?'

        self._msg.append(cmd)
        return self

    @final_command
    def erhvctst_query(self) -> 'MessageBuilder':
        cmd = 'ERHVCTST?'

        self._msg.append(cmd)
        return self

    def erhvp(self, state: Union[constants.ERHVP.State, int]
              ) -> 'MessageBuilder':
        cmd = f'ERHVP {state}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhvp_query(self) -> 'MessageBuilder':
        cmd = 'ERHVP?'

        self._msg.append(cmd)
        return self

    def erhvpv(self, state: Union[constants.ERHVPV.State, int]
               ) -> 'MessageBuilder':
        cmd = f'ERHVPV {state}'

        self._msg.append(cmd)
        return self

    def erhvs(self, enable_series_resistor: bool) -> 'MessageBuilder':
        cmd = f'ERHVS {int(enable_series_resistor)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhvs_query(self) -> 'MessageBuilder':
        cmd = f'ERHVS?'

        self._msg.append(cmd)
        return self

    def erm(self, iport: int) -> 'MessageBuilder':
        cmd = f'ERM {iport}'

        self._msg.append(cmd)
        return self

    def ermod(self,
              mode: Union[constants.ERMOD.Mode, int],
              option: Optional[bool] = None
              ) -> 'MessageBuilder':
        cmd = f'ERMOD {mode}'

        if option is not None:
            cmd += f',{option}'

        self._msg.append(cmd)
        return self

    @final_command
    def ermod_query(self) -> 'MessageBuilder':
        cmd = 'ERMOD?'

        self._msg.append(cmd)
        return self

    def erpfda(self,
               hvsmu: Union[constants.SlotNr, int],
               smu: Union[constants.SlotNr, int]
               ) -> 'MessageBuilder':
        cmd = f'ERPFDA {hvsmu},{smu}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfda_query(self) -> 'MessageBuilder':
        cmd = 'ERPFDA?'

        self._msg.append(cmd)
        return self

    def erpfdp(self, state: Union[constants.ERPFDP.State, int]
               ) -> 'MessageBuilder':
        cmd = f'ERPFDP {state}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfdp_query(self) -> 'MessageBuilder':
        cmd = 'ERPFDP?'

        self._msg.append(cmd)
        return self

    def erpfds(self, state: bool) -> 'MessageBuilder':
        cmd = f'ERPFDS {int(state)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfds_query(self) -> 'MessageBuilder':
        cmd = 'ERPFDS?'

        self._msg.append(cmd)
        return self

    def erpfga(self, gsmu: Union[constants.SlotNr, int]) -> 'MessageBuilder':
        cmd = f'ERPFGA {gsmu}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfga_query(self) -> 'MessageBuilder':
        cmd = 'ERPFGA?'

        self._msg.append(cmd)
        return self

    def erpfgp(self, state: Union[constants.ERPFGP.State, int]
               ) -> 'MessageBuilder':
        cmd = f'ERPFGP {state}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfgp_query(self) -> 'MessageBuilder':
        cmd = 'ERPFGP?'

        self._msg.append(cmd)
        return self

    def erpfgr(self, state: Union[constants.ERPFGR.State, int]
               ) -> 'MessageBuilder':
        cmd = f'ERPFGR {state}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfgr_query(self) -> 'MessageBuilder':
        cmd = 'ERPFDS?'

        self._msg.append(cmd)
        return self

    def erpfqg(self, state: bool) -> 'MessageBuilder':
        cmd = f'ERPFQG {int(state)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfqg_query(self) -> 'MessageBuilder':
        cmd = 'ERPFQG?'

        self._msg.append(cmd)
        return self

    @final_command
    def erpftemp_query(self, chnum: Union[constants.ChNr, int]
                       ) -> 'MessageBuilder':
        cmd = f'ERPFTEMP? {chnum}'

        self._msg.append(cmd)
        return self

    def erpfuhca(self,
                 vsmu: Union[constants.SlotNr, int],
                 ismu: Union[constants.SlotNr, int]
                 ) -> 'MessageBuilder':
        cmd = f'ERPFUHCA {vsmu},{ismu}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfuhca_query(self) -> 'MessageBuilder':
        cmd = 'ERPFUHCA?'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfuhccal_query(self) -> 'MessageBuilder':
        cmd = 'ERPFUHCCAL?'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfuhcmax_query(self) -> 'MessageBuilder':
        cmd = 'ERPFUHCMAX?'

        self._msg.append(cmd)
        return self

    def erpfuhctst(self) -> 'MessageBuilder':
        cmd = 'ERPFUHCTST?'

        self._msg.append(cmd)
        return self

    @final_command
    def err_query(self, mode: Optional[Union[constants.ERR.Mode, int]] = None
                  ) -> 'MessageBuilder':
        if mode is None:
            cmd = 'ERR?'
        else:
            cmd = f'ERR? {mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def errx_query(self, mode: Optional[Union[constants.ERRX.Mode, int]] = None
                   ) -> 'MessageBuilder':
        cmd = 'ERRX?'

        if mode is not None:
            cmd += f' {mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def ers_query(self) -> 'MessageBuilder':
        cmd = 'ERS?'

        self._msg.append(cmd)
        return self

    def erssp(self,
              port: Union[constants.ERSSP.Port, int],
              status: Union[constants.ERSSP.Status, int]
              ) -> 'MessageBuilder':
        cmd = f'ERSSP {port},{status}'

        self._msg.append(cmd)
        return self

    @final_command
    def erssp_query(self, port: Union[constants.ERSSP.Port, int]
                    ) -> 'MessageBuilder':
        cmd = f'ERSSP? {port}'

        self._msg.append(cmd)
        return self

    def eruhva(self,
               vsmu: Union[constants.SlotNr, int],
               ismu: Union[constants.SlotNr, int]
               ) -> 'MessageBuilder':
        cmd = f'ERUHVA {vsmu},{ismu}'

        self._msg.append(cmd)
        return self

    @final_command
    def eruhva_query(self) -> 'MessageBuilder':
        cmd = 'ERUHVA?'

        self._msg.append(cmd)
        return self

    def fc(self, chnum: Union[constants.ChNr, int], freq: float
           ) -> 'MessageBuilder':
        cmd = f'FC {chnum},{freq}'

        self._msg.append(cmd)
        return self

    def fl(self,
           enable_filter: bool,
           channels: Optional[constants.ChannelList] = None
           ) -> 'MessageBuilder':
        if channels is None:
            cmd = f'FL {int(enable_filter)}'
        elif len(channels) > 10:
            raise ValueError("A maximum of ten channels can be set.")
        else:
            cmd = f'FL {int(enable_filter)},{as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def fmt(self,
            format_id: Union[constants.FMT.Format, int],
            mode: Optional[Union[constants.FMT.Mode, int]] = None
            ) -> 'MessageBuilder':
        cmd = f'FMT {format_id}'

        if mode is not None:
            cmd += f',{mode}'

        self._msg.append(cmd)
        return self

    def hvsmuop(self,
                src_range: Union[constants.HVSMUOP.SourceRange, int]
                ) -> 'MessageBuilder':
        cmd = f'HVSMUOP {src_range}'

        self._msg.append(cmd)
        return self

    @final_command
    def hvsmuop_query(self) -> 'MessageBuilder':
        cmd = 'HVSMUOP?'

        self._msg.append(cmd)
        return self

    @final_command
    def idn_query(self) -> 'MessageBuilder':
        cmd = '*IDN?'

        self._msg.append(cmd)
        return self

    def imp(self, mode: Union[constants.IMP.MeasurementMode, int]
            ) -> 'MessageBuilder':
        cmd = f'IMP {mode}'

        self._msg.append(cmd)
        return self

    def in_(self, channels: Optional[constants.ChannelList] = None
            ) -> 'MessageBuilder':
        if channels is None:
            cmd = f'IN'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'IN {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def intlkvth(self, voltage: float) -> 'MessageBuilder':
        cmd = f'INTLKVTH {voltage}'

        self._msg.append(cmd)
        return self

    @final_command
    def intlkvth_query(self) -> 'MessageBuilder':
        cmd = 'INTLKVTH?'

        self._msg.append(cmd)
        return self

    def lgi(self,
            chnum: Union[constants.ChNr, int],
            mode: Union[constants.LinearSearchMode, int],
            i_range: Union[constants.IMeasRange, int],
            target: float
            ) -> 'MessageBuilder':
        cmd = f'LGI {chnum},{mode},{i_range},{target}'

        self._msg.append(cmd)
        return self

    def lgv(self,
            chnum: Union[constants.ChNr, int],
            mode: Union[constants.LinearSearchMode, int],
            v_range: Union[constants.VMeasRange, int],
            target: float
            ) -> 'MessageBuilder':
        cmd = f'LGV {chnum},{mode},{v_range},{target}'

        self._msg.append(cmd)
        return self

    def lim(self,
            mode: Union[constants.LIM.Mode, int],
            limit: float
            ) -> 'MessageBuilder':
        cmd = f'LIM {mode},{limit}'

        self._msg.append(cmd)
        return self

    @final_command
    def lim_query(self, mode: Union[constants.LIM.Mode, int]
                  ) -> 'MessageBuilder':
        cmd = f'LIM? {mode}'

        self._msg.append(cmd)
        return self

    def lmn(self, enable_data_monitor: bool) -> 'MessageBuilder':
        cmd = f'LMN {int(enable_data_monitor)}'

        self._msg.append(cmd)
        return self

    @final_command
    def lop_query(self) -> 'MessageBuilder':
        cmd = 'LOP?'

        self._msg.append(cmd)
        return self

    @final_command
    def lrn_query(self, type_id: Union[constants.LRN.Type, int]
                  ) -> 'MessageBuilder':
        cmd = f'*LRN? {type_id}'

        self._msg.append(cmd)
        return self

    def lsi(self,
            chnum: Union[constants.ChNr, int],
            i_range: Union[constants.IOutputRange, int],
            start: float,
            stop: float,
            step: float,
            v_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'LSI {chnum},{i_range},{start},{stop},{step}'

        if v_comp is not None:
            cmd += f',{v_comp}'

        self._msg.append(cmd)
        return self

    def lsm(self,
            abort: Union[constants.Abort, int],
            post: Optional[Union[constants.LSM.Post, int]] = None
            ) -> 'MessageBuilder':
        cmd = f'LSM {abort}'

        if post is not None:
            cmd += f',{post}'

        self._msg.append(cmd)
        return self

    def lssi(self,
             chnum: Union[constants.ChNr, int],
             polarity: Union[constants.Polarity, int],
             offset: float,
             v_comp: Optional[float] = None
             ) -> 'MessageBuilder':
        cmd = f'LSSI {chnum},{polarity},{offset}'

        if v_comp is not None:
            cmd += f',{v_comp}'

        self._msg.append(cmd)
        return self

    def lssv(self,
             chnum: Union[constants.ChNr, int],
             polarity: Union[constants.Polarity, int],
             offset: float,
             i_comp: Optional[float] = None
             ) -> 'MessageBuilder':
        cmd = f'LSSV {chnum},{polarity},{offset}'

        if i_comp is not None:
            cmd += f',{i_comp}'

        self._msg.append(cmd)
        return self

    @final_command
    def lst_query(self, pnum=None, index=None, size=None) -> 'MessageBuilder':
        cmd = 'LST?'

        if pnum is not None:
            cmd += f' {pnum}'

            if index is not None:
                cmd += f',{index}'

                if size is not None:
                    cmd += f',{size}'

        self._msg.append(cmd)
        return self

    def lstm(self, hold: float, delay: float) -> 'MessageBuilder':
        cmd = f'LSTM {hold},{delay}'

        self._msg.append(cmd)
        return self

    def lsv(self,
            chnum: Union[constants.ChNr, int],
            v_range: Union[constants.VOutputRange, int],
            start: float,
            stop: float,
            step: float,
            i_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'LSV {chnum},{v_range},{start},{stop},{step}'

        if i_comp is not None:
            cmd += f',{i_comp}'

        self._msg.append(cmd)
        return self

    def lsvm(self, mode: Union[constants.LSVM.DataOutputMode, int]
             ) -> 'MessageBuilder':
        cmd = f'LSVM {mode}'

        self._msg.append(cmd)
        return self

    def mcc(self, channels: Optional[constants.ChannelList] = None
            ) -> 'MessageBuilder':
        if channels is None:
            cmd = f'MCC'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'MCC {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def mcpnt(self,
              chnum: Union[constants.ChNr, int],
              delay: float,
              width: float
              ) -> 'MessageBuilder':
        cmd = f'MCPNT {chnum},{delay},{width}'

        self._msg.append(cmd)
        return self

    def mcpnx(self,
              n: int,
              chnum: Union[constants.ChNr, int],
              mode: Union[constants.MCPNX.Mode, int],
              src_range: Union[constants.VOutputRange, constants.IOutputRange],
              base: float,
              pulse: float,
              comp: Optional[float] = None
              ) -> 'MessageBuilder':
        cmd = f'MCPNX {n},{chnum},{mode},{src_range},{base},{pulse}'

        if comp is not None:
            cmd += f',{comp}'

        self._msg.append(cmd)
        return self

    def mcpt(self,
             hold: float,
             period: Optional[Union[float, constants.AutoPeriod]] = None,
             measurement_delay: Optional[float] = None,
             average: Optional[int] = None
             ) -> 'MessageBuilder':
        cmd = f'MCPT {hold}'

        if period is not None:
            cmd += f',{period}'

            if measurement_delay is not None:
                cmd += f',{measurement_delay}'

                if average is not None:
                    cmd += f',{average}'

        self._msg.append(cmd)
        return self

    def mcpws(self,
              mode: Union[constants.SweepMode, int],
              step: int
              ) -> 'MessageBuilder':
        cmd = f'MCPWS {mode},{step}'

        self._msg.append(cmd)
        return self

    def mcpwnx(self,
               n: int,
               chnum: Union[constants.ChNr, int],
               mode: Union[constants.MCPWNX.Mode, int],
               src_range: Union[constants.VOutputRange,
                                constants.IOutputRange],
               base: float,
               start: float,
               stop: float,
               comp: Optional[float] = None,
               p_comp: Optional[float] = None
               ) -> 'MessageBuilder':
        cmd = f'MCPWNX {n},{chnum},{mode},{src_range},{base},{start},{stop}'

        if comp is not None:
            cmd += f',{comp}'

            if p_comp is not None:
                cmd += f',{p_comp}'

        self._msg.append(cmd)
        return self

    def mdcv(self,
             chnum: Union[constants.ChNr, int],
             base: float,
             bias: float,
             post: Optional[float] = None
             ) -> 'MessageBuilder':
        cmd = f'MDCV {chnum},{base},{bias}'

        if post is not None:
            cmd += f',{post}'

        self._msg.append(cmd)
        return self

    def mi(self,
           chnum: Union[constants.ChNr, int],
           i_range: Union[constants.IOutputRange, int],
           base: float,
           bias: float,
           v_comp: Optional[float] = None
           ) -> 'MessageBuilder':
        cmd = f'MI {chnum},{i_range},{base},{bias}'

        if v_comp is not None:
            cmd += f',{v_comp}'

        self._msg.append(cmd)
        return self

    def ml(self, mode: Union[constants.ML.Mode, int]) -> 'MessageBuilder':
        cmd = f'ML {mode}'

        self._msg.append(cmd)
        return self

    def mm(self,
           mode: Union[constants.MM.Mode, int],
           channels: Optional[constants.ChannelList] = None
           ) -> 'MessageBuilder':
        if mode in (1, 2, 10, 16, 18, 27, 28):
            if channels is None:
                raise ValueError('Specify channels for this mode')
            elif len(channels) > 10:
                raise ValueError('A maximum of ten channels can be set. For '
                                 'mode=18, the first chnum must be MFCMU.')
            cmd = f'MM {mode},{as_csv(channels)}'
        elif mode in (3, 4, 5, 17, 19, 20, 22, 23, 26):
            if channels is None or len(channels) != 1:
                raise ValueError('Specify 1 (and only 1) channel.')
            cmd = f'MM {mode},{channels[0]}'
        elif mode in (9, 13):
            if channels is None:
                cmd = f'MM {mode}'
            elif len(channels) > 1:
                raise ValueError('Specify not more than 1 channel for this '
                                 'mode')
            else:
                cmd = f'MM {mode},{as_csv(channels)}'
        elif mode in (14, 15):
            if channels is not None:
                raise ValueError('Do not specify channels for this mode')
            cmd = f'MM {mode}'
        else:
            raise ValueError('Invalid Mode ID.')

        self._msg.append(cmd)
        return self

    def msc(self,
            abort: Union[constants.Abort, int],
            post: Optional[Union[constants.MSC.Post, int]] = None
            ) -> 'MessageBuilder':
        cmd = f'MSC {abort}'

        if post is not None:
            cmd += f',{post}'

        self._msg.append(cmd)
        return self

    def msp(self,
            chnum: Union[constants.ChNr, int],
            post: Optional[float] = None,
            base: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'MSP {chnum}'

        if post is not None:
            cmd += f',{post}'

            if base is not None:
                cmd += f',{base}'

        self._msg.append(cmd)
        return self

    def mt(self,
           h_bias: float,
           interval: float,
           number: int,
           h_base: Optional[float] = None
           ) -> 'MessageBuilder':
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
        cmd = f'MT {h_bias},{interval},{number}'

        if h_base is not None:
            cmd += f',{h_base}'

        self._msg.append(cmd)
        return self

    def mtdcv(self,
              h_bias: float,
              interval: float,
              number: int,
              h_base: Optional[float] = None
              ) -> 'MessageBuilder':
        cmd = f'MTDCV {h_bias},{interval},{number}'

        if h_base is not None:
            cmd = f',{h_base}'

        self._msg.append(cmd)
        return self

    def mv(self,
           chnum: Union[constants.ChNr, int],
           v_range: Union[constants.VOutputRange, int],
           base: float,
           bias: float,
           i_comp: Optional[float] = None
           ) -> 'MessageBuilder':
        cmd = f'MV {chnum},{v_range},{base},{bias}'

        if i_comp is not None:
            cmd += f',{i_comp}'

        self._msg.append(cmd)
        return self

    @final_command
    def nub_query(self) -> 'MessageBuilder':
        cmd = 'NUB?'

        self._msg.append(cmd)
        return self

    def odsw(self,
             chnum: Union[constants.ChNr, int],
             enable_pulse_switch: bool,
             switch_normal_state: Optional[
                 Union[constants.ODSW.SwitchNormalState, int]] = None,
             delay: Optional[float] = None,
             width: Optional[float] = None
             ) -> 'MessageBuilder':
        cmd = f'ODSW {chnum},{int(enable_pulse_switch)}'

        if switch_normal_state is not None:
            cmd += f',{switch_normal_state}'

            if xor(delay is None, width is None):
                raise ValueError('When specifying delay, then width must be '
                                 'specified (and vice versa)')

            if delay is not None and width is not None:
                cmd += f',{delay},{width}'

        self._msg.append(cmd)
        return self

    @final_command
    def odsw_query(self, chnum: Union[constants.ChNr, int]
                   ) -> 'MessageBuilder':
        cmd = f'ODSW? {chnum}'

        self._msg.append(cmd)
        return self

    @final_command
    def opc_query(self) -> 'MessageBuilder':
        cmd = '*OPC?'

        self._msg.append(cmd)
        return self

    def os(self) -> 'MessageBuilder':
        cmd = 'OS'

        self._msg.append(cmd)
        return self

    def osx(self,
            port: Union[constants.TriggerPort, int],
            level: Optional[Union[constants.OSX.Level, int]] = None
            ) -> 'MessageBuilder':
        cmd = f'OSX {port}'

        if level is not None:
            cmd += f',{level}'

        self._msg.append(cmd)
        return self

    def pa(self, wait_time: Optional[float] = None) -> 'MessageBuilder':
        cmd = 'PA'

        if wait_time is not None:
            cmd += f' {wait_time}'

        self._msg.append(cmd)
        return self

    def pad(self, enable: bool) -> 'MessageBuilder':
        cmd = f'PAD {int(enable)}'

        self._msg.append(cmd)
        return self

    def pax(self,
            port: Union[constants.TriggerPort, int],
            wait_time: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'PAX {port}'

        if wait_time is not None:
            cmd += f',{wait_time}'

        self._msg.append(cmd)
        return self

    def pch(self,
            controller: Union[constants.ChNr, int],
            worker: Union[constants.ChNr, int],
            ) -> 'MessageBuilder':
        cmd = f'PCH {controller},{worker}'

        self._msg.append(cmd)
        return self

    @final_command
    def pch_query(self, master=None) -> 'MessageBuilder':
        cmd = 'PCH'

        if master is not None:
            cmd += f' {master}'

        self._msg.append(cmd)
        return self

    def pdcv(self,
             chnum: Union[constants.ChNr, int],
             base: float,
             pulse: float
             ) -> 'MessageBuilder':
        cmd = f'PDCV {chnum},{base},{pulse}'

        self._msg.append(cmd)
        return self

    def pi(self,
           chnum: Union[constants.ChNr, int],
           i_range: Union[constants.IOutputRange, int],
           base: float,
           pulse: float,
           v_comp: Optional[float] = None
           ) -> 'MessageBuilder':
        cmd = f'PI {chnum},{i_range},{base},{pulse}'

        if v_comp is not None:
            cmd += f',{v_comp}'

        self._msg.append(cmd)
        return self

    def pt(self,
           hold: float,
           width: float,
           period: Optional[Union[float, constants.AutoPeriod]] = None,
           t_delay: Optional[float] = None
           ) -> 'MessageBuilder':
        cmd = f'PT {hold},{width}'

        if period is not None:
            cmd += f',{period}'

            if t_delay is not None:
                cmd += f',{t_delay}'

        self._msg.append(cmd)
        return self

    def ptdcv(self,
              hold: float,
              width: float,
              period: Optional[float] = None,
              t_delay: Optional[float] = None
              ) -> 'MessageBuilder':
        cmd = f'PTDCV {hold},{width}'

        if period is not None:
            cmd += f',{period}'

            if t_delay is not None:
                cmd += f',{t_delay}'

        self._msg.append(cmd)
        return self

    def pv(self,
           chnum: Union[constants.ChNr, int],
           v_range: Union[constants.VOutputRange, int],
           base: float,
           pulse: float,
           i_comp: Optional[float] = None
           ) -> 'MessageBuilder':
        cmd = f'PV {chnum},{v_range},{base},{pulse}'

        if i_comp is not None:
            cmd += f',{i_comp}'

        self._msg.append(cmd)
        return self

    def pwdcv(self,
              chnum: Union[constants.ChNr, int],
              mode: Union[constants.LinearSweepMode, int],
              base: float,
              start: float,
              stop: float,
              step: float
              ) -> 'MessageBuilder':
        cmd = f'PWDCV {chnum},{mode},{base},{start},{stop},{step}'

        self._msg.append(cmd)
        return self

    def pwi(self,
            chnum: Union[constants.ChNr, int],
            mode: Union[constants.SweepMode, int],
            i_range: Union[constants.IOutputRange, int],
            base: float,
            start: float,
            stop: float,
            step: float,
            v_comp: Optional[float] = None,
            p_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'PWI {chnum},{mode},{i_range},{base},{start},{stop},{step}'

        if v_comp is not None:
            cmd += f',{v_comp}'

            if p_comp is not None:
                cmd += f',{p_comp}'

        self._msg.append(cmd)
        return self

    def pwv(self,
            chnum: Union[constants.ChNr, int],
            mode: Union[constants.SweepMode, int],
            v_range: Union[constants.VOutputRange, int],
            base: float,
            start: float,
            stop: float,
            step: float,
            i_comp: Optional[float] = None,
            p_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'PWI {chnum},{mode},{v_range},{base},{start},{stop},{step}'

        if i_comp is not None:
            cmd += f',{i_comp}'

            if p_comp is not None:
                cmd += f',{p_comp}'

        self._msg.append(cmd)
        return self

    def qsc(self, mode: Union[constants.APIVersion, int]) -> 'MessageBuilder':
        cmd = f'QSC {mode}'

        self._msg.append(cmd)
        return self

    def qsl(self,
            enable_data_output: bool,
            enable_leakage_current_compensation: bool
            ) -> 'MessageBuilder':
        cmd = f'QSL {int(enable_data_output)},' \
              f'{int(enable_leakage_current_compensation)}'

        self._msg.append(cmd)
        return self

    def qsm(self,
            abort: Union[constants.Abort, int],
            post: Optional[Union[constants.QSM.Post, int]] = None
            ) -> 'MessageBuilder':
        cmd = f'QSM {abort}'

        if post is not None:
            cmd += f',{post}'

        self._msg.append(cmd)
        return self

    def qso(self,
            enable_smart_operation: bool,
            chnum: Optional[Union[constants.ChNr, int]] = None,
            v_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'QSO {int(enable_smart_operation)}'

        if chnum is not None:
            cmd += f',{chnum}'

            if v_comp is not None:
                cmd += f',{v_comp}'

        self._msg.append(cmd)
        return self

    def qsr(self, i_range: Union[constants.IMeasRange, int]
            ) -> 'MessageBuilder':
        cmd = f'QSR {i_range}'

        self._msg.append(cmd)
        return self

    def qst(self,
            cinteg: float,
            linteg: float,
            hold: float,
            delay1: float,
            delay2: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'QST {cinteg},{linteg},{hold},{delay1}'

        if delay2 is not None:
            cmd += f',{delay2}'

        self._msg.append(cmd)
        return self

    def qsv(self,
            chnum: Union[constants.ChNr, int],
            mode: Union[constants.LinearSweepMode, int],
            v_range: Union[constants.VOutputRange, int],
            start: float,
            stop: float,
            cvoltage: float,
            step: float,
            i_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'QSV {chnum},{mode},{v_range},{start},{stop},{cvoltage},{step}'

        if i_comp is not None:
            cmd += f',{i_comp}'

        self._msg.append(cmd)
        return self

    def qsz(self, mode: Union[constants.QSZ.Mode, int]) -> 'MessageBuilder':
        cmd = f'QSZ {mode}'

        self._msg.append(cmd)

        if mode == constants.QSZ.Mode.PERFORM_MEASUREMENT:
            # Result must be queried if measurement is performed
            self._msg.set_final()

        return self

    def rc(self,
           chnum: Union[constants.ChNr, int],
           ranging_mode: Union[constants.RangingMode, int],
           measurement_range: Optional[int] = None
           ) -> 'MessageBuilder':
        if measurement_range is None:
            if ranging_mode != 0:
                raise ValueError('measurement_range must be specified for '
                                 'ranging_mode!=0')
            cmd = f'RC {chnum},{ranging_mode}'
        else:
            cmd = f'RC {chnum},{ranging_mode},{measurement_range}'

        self._msg.append(cmd)
        return self

    def rcv(self,
            slot: Optional[Union[constants.SlotNr, int]] = None
            ) -> 'MessageBuilder':
        cmd = 'RCV' if slot is None else f'RCV {slot}'

        self._msg.append(cmd)
        return self

    def ri(self,
           chnum: Union[constants.ChNr, int],
           i_range: Union[constants.IMeasRange, int]
           ) -> 'MessageBuilder':
        cmd = f'RI {chnum},{i_range}'

        self._msg.append(cmd)
        return self

    def rm(self,
           chnum: Union[constants.ChNr, int],
           mode: Union[constants.RM.Mode, int],
           rate: Optional[int] = None
           ) -> 'MessageBuilder':
        if rate is None:
            cmd = f'RM {chnum},{mode}'
        else:
            if mode == 1:
                raise ValueError('Do not specify rate for mode 1')
            cmd = f'RM {chnum},{mode},{rate}'

        self._msg.append(cmd)
        return self

    def rst(self) -> 'MessageBuilder':
        cmd = '*RST'

        self._msg.append(cmd)
        return self

    def ru(self, start: int, stop: int) -> 'MessageBuilder':
        cmd = f'RU {start},{stop}'

        self._msg.append(cmd)
        return self

    def rv(self,
           chnum: Union[constants.ChNr, int],
           v_range: Union[constants.VMeasRange, int]
           ) -> 'MessageBuilder':
        cmd = f'RV {chnum},{v_range}'

        self._msg.append(cmd)
        return self

    def rz(self, channels: Optional[constants.ChannelList] = None
           ) -> 'MessageBuilder':
        if channels is None:
            cmd = 'RZ'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'RZ {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def sal(self,
            chnum: Union[constants.ChNr, int],
            enable_status_led: bool
            ) -> 'MessageBuilder':
        cmd = f'SAL {chnum},{int(enable_status_led)}'

        self._msg.append(cmd)
        return self

    def sap(self,
            chnum: Union[constants.ChNr, int],
            path: Union[constants.SAP.Path, int]
            ) -> 'MessageBuilder':
        cmd = f'SAP {chnum},{path}'

        self._msg.append(cmd)
        return self

    def sar(self,
            chnum: Union[constants.ChNr, int],
            enable_picoamp_autoranging: bool
            ) -> 'MessageBuilder':
        # For reasons only known to the designer of the KeysightB1500's API the
        # logic of enabled=1 and disabled=0 is inverted JUST for this command.
        cmd = f'SAR {chnum},{int(not enable_picoamp_autoranging)}'

        self._msg.append(cmd)
        return self

    def scr(self, pnum: Optional[int] = None) -> 'MessageBuilder':
        cmd = 'SCR' if pnum is None else f'SCR {pnum}'

        self._msg.append(cmd)
        return self

    def ser(self, chnum: Union[constants.ChNr, int], load_z: float
            ) -> 'MessageBuilder':
        cmd = f'SER {chnum},{load_z}'

        self._msg.append(cmd)
        return self

    @final_command
    def ser_query(self, chnum: Union[constants.ChNr, int]) -> 'MessageBuilder':
        cmd = f'SER? {chnum}'

        self._msg.append(cmd)
        return self

    def sim(self, mode: Union[constants.SIM.Mode, int]) -> 'MessageBuilder':
        cmd = f'SIM {mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def sim_query(self) -> 'MessageBuilder':
        cmd = 'SIM?'

        self._msg.append(cmd)
        return self

    def sopc(self, chnum: Union[constants.ChNr, int], power: float
             ) -> 'MessageBuilder':
        cmd = f'SOPC {chnum},{power}'

        self._msg.append(cmd)
        return self

    @final_command
    def sopc_query(self, chnum: Union[constants.ChNr, int]
                   ) -> 'MessageBuilder':
        cmd = f'SOPC? {chnum}'

        self._msg.append(cmd)
        return self

    def sovc(self, chnum: Union[constants.ChNr, int], voltage: float
             ) -> 'MessageBuilder':
        cmd = f'SOVC {chnum},{voltage}'

        self._msg.append(cmd)
        return self

    @final_command
    def sovc_query(self, chnum: Union[constants.ChNr, int]
                   ) -> 'MessageBuilder':
        cmd = f'SOVC? {chnum}'

        self._msg.append(cmd)
        return self

    def spm(self,
            chnum: Union[constants.ChNr, int],
            mode: Union[constants.SPM.Mode, int]
            ) -> 'MessageBuilder':
        cmd = f'SPM {chnum},{mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def spm_query(self, chnum: Union[constants.ChNr, int]) -> 'MessageBuilder':
        cmd = f'SPM? {chnum}'

        self._msg.append(cmd)
        return self

    def spp(self) -> 'MessageBuilder':
        cmd = 'SPP'

        self._msg.append(cmd)
        return self

    def spper(self,
              period: float) -> 'MessageBuilder':
        cmd = f'SPPER {period}'

        self._msg.append(cmd)
        return self

    @final_command
    def spper_query(self) -> 'MessageBuilder':
        cmd = 'SPPER?'

        self._msg.append(cmd)
        return self

    def sprm(self,
             mode: Union[constants.SPRM.Mode, int],
             condition=None
             ) -> 'MessageBuilder':
        if condition is None:
            cmd = f'SPRM {mode}'
        else:
            cmd = f'SPRM {mode},{condition}'

        self._msg.append(cmd)
        return self

    @final_command
    def sprm_query(self) -> 'MessageBuilder':
        cmd = 'SPRM?'

        self._msg.append(cmd)
        return self

    @final_command
    def spst_query(self) -> 'MessageBuilder':
        cmd = 'SPST?'

        self._msg.append(cmd)
        return self

    def spt(self,
            chnum: Union[constants.ChNr, int],
            src: Union[constants.SPT.Src, int],
            delay: float,
            width: float,
            leading: float,
            trailing: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'SPT {chnum},{src},{delay},{width},{leading}'

        if trailing is not None:
            cmd += f',{trailing}'

        self._msg.append(cmd)
        return self

    @final_command
    def spt_query(self,
                  chnum: Union[constants.ChNr, int],
                  src: Union[constants.SPT.Src, int]
                  ) -> 'MessageBuilder':
        cmd = f'SPT? {chnum},{src}'

        self._msg.append(cmd)
        return self

    def spupd(self, channels: Optional[constants.ChannelList] = None
              ) -> 'MessageBuilder':
        if channels is None:
            cmd = 'SPUPD'
        elif len(channels) > 10:
            raise ValueError("A maximum of ten channels can be set.")
        else:
            cmd = f'SPUPD {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def spv(self,
            chnum: Union[constants.ChNr, int],
            src: Union[constants.SPV.Src, int],
            base: float,
            peak: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'SPV {chnum},{src},{base}'

        if peak is not None:
            cmd += f',{peak}'

        self._msg.append(cmd)
        return self

    @final_command
    def spv_query(self,
                  chnum: Union[constants.ChNr, int],
                  src: Union[constants.SPV.Src, int]
                  ) -> 'MessageBuilder':
        cmd = f'SPV? {chnum},{src}'

        self._msg.append(cmd)
        return self

    def sre(self, flags: Union[constants.SRE, int]) -> 'MessageBuilder':
        cmd = f'*SRE {flags}'

        self._msg.append(cmd)
        return self

    @final_command
    def sre_query(self) -> 'MessageBuilder':
        cmd = '*SRE?'

        self._msg.append(cmd)
        return self

    def srp(self) -> 'MessageBuilder':
        cmd = 'SRP'

        self._msg.append(cmd)
        return self

    def ssl(self,
            chnum: Union[constants.ChNr, int],
            enable_indicator_led: bool
            ) -> 'MessageBuilder':
        cmd = f'SSL {chnum},{int(enable_indicator_led)}'

        self._msg.append(cmd)
        return self

    def ssp(self,
            chnum: Union[constants.ChNr, int],
            path: Union[constants.SSP.Path, int]
            ) -> 'MessageBuilder':
        cmd = f'SSP {chnum},{path}'

        self._msg.append(cmd)
        return self

    def ssr(self,
            chnum: Union[constants.ChNr, int],
            enable_series_resistor: bool
            ) -> 'MessageBuilder':
        cmd = f'SSR {chnum},{int(enable_series_resistor)}'

        self._msg.append(cmd)
        return self

    def st(self, pnum: int) -> 'MessageBuilder':
        cmd = f'ST {pnum}'

        self._msg.append(cmd)
        return self

    @final_command
    def stb_query(self) -> 'MessageBuilder':
        cmd = '*STB?'

        self._msg.append(cmd)
        return self

    def stgp(self,
             chnum: Union[constants.ChNr, int],
             trigger_timing: Union[constants.STGP.TriggerTiming, int]
             ) -> 'MessageBuilder':
        cmd = f'STGP {chnum},{trigger_timing}'

        self._msg.append(cmd)
        return self

    @final_command
    def stgp_query(self, chnum: Union[constants.ChNr, int]
                   ) -> 'MessageBuilder':
        cmd = f'STGP? {chnum}'

        self._msg.append(cmd)
        return self

    def tacv(self, chnum: Union[constants.ChNr, int], voltage: float
             ) -> 'MessageBuilder':
        cmd = f'TACV {chnum},{voltage}'

        self._msg.append(cmd)
        return self

    def tc(self,
           chnum: Union[constants.ChNr, int],
           mode: Union[constants.RangingMode, int],
           ranging_type=None
           ) -> 'MessageBuilder':
        cmd = f'TC {chnum},{mode}'

        if ranging_type is not None:
            cmd += f',{ranging_type}'

        self._msg.append(cmd)
        return self

    def tdcv(self,
             chnum: Union[constants.ChNr, int],
             voltage: float
             ) -> 'MessageBuilder':
        cmd = f'TDCV {chnum},{voltage}'

        self._msg.append(cmd)
        return self

    def tdi(self,
            chnum: Union[constants.ChNr, int],
            i_range: Union[constants.IOutputRange, int],
            current: float,
            v_comp: Optional[float] = None,
            comp_polarity: Optional[
                Union[constants.CompliancePolarityMode, int]] = None,
            v_range: Optional[Union[constants.VOutputRange, int]] = None
            ) -> 'MessageBuilder':
        cmd = f'TDI {chnum},{i_range},{current}'

        if v_comp is not None:
            cmd += f',{v_comp}'

            if comp_polarity is not None:
                cmd += f',{comp_polarity}'

                if v_range is not None:
                    cmd += f',{v_range}'

        self._msg.append(cmd)
        return self

    def tdv(self,
            chnum: Union[constants.ChNr, int],
            v_range: Union[constants.VOutputRange, int],
            voltage: float,
            i_comp: Optional[float] = None,
            comp_polarity: Optional[
                Union[constants.CompliancePolarityMode, int]] = None,
            i_range: Optional[Union[constants.IOutputRange, int]] = None
            ) -> 'MessageBuilder':
        cmd = f'TDV {chnum},{v_range},{voltage}'

        if i_comp is not None:
            cmd += f',{i_comp}'

            if comp_polarity is not None:
                cmd += f',{comp_polarity}'

                if i_range is not None:
                    cmd += f',{i_range}'

        self._msg.append(cmd)
        return self

    def tgmo(self, mode: Union[constants.TGMO.Mode, int]) -> 'MessageBuilder':
        cmd = f'TGMO {mode}'

        self._msg.append(cmd)
        return self

    def tgp(self,
            port: Union[constants.TriggerPort, int],
            terminal: Union[constants.TGP.TerminalType, int],
            polarity: Union[constants.TGP.Polarity, int],
            trigger_type: Optional[Union[constants.TGP.TriggerType,
                                         int]] = None
            ) -> 'MessageBuilder':
        cmd = f'TGP {port},{terminal},{polarity}'

        if trigger_type is not None:
            cmd += f',{trigger_type}'

        self._msg.append(cmd)
        return self

    def tgpc(self, ports: Optional[List[constants.TriggerPort]] = None
             ) -> 'MessageBuilder':
        if ports is None:
            cmd = 'TGPC'
        elif len(ports) > 18:
            raise ValueError("A maximum of 18 ports can be set.")
        else:
            cmd = f'TGPC {as_csv(ports)}'

        self._msg.append(cmd)
        return self

    def tgsi(self, mode: Union[constants.TGSI.Mode, int]) -> 'MessageBuilder':
        cmd = f'TGSI {mode}'

        self._msg.append(cmd)
        return self

    def tgso(self, mode: Union[constants.TGSO.Mode, int]) -> 'MessageBuilder':
        cmd = f'TGSO {mode}'

        self._msg.append(cmd)
        return self

    def tgxo(self, mode: Union[constants.TGXO.Mode, int]) -> 'MessageBuilder':
        cmd = f'TGXO {mode}'

        self._msg.append(cmd)
        return self

    def ti(self,
           chnum: Union[constants.ChNr, int],
           i_range: Optional[Union[constants.IMeasRange, int]] = None
           ) -> 'MessageBuilder':
        cmd = f'TI {chnum}'

        if i_range is not None:
            cmd += f',{i_range}'

        self._msg.append(cmd)
        return self

    def tiv(self,
            chnum: Union[constants.ChNr, int],
            i_range: Optional[Union[constants.IMeasRange, int]] = None,
            v_range: Optional[Union[constants.VMeasRange, int]] = None
            ) -> 'MessageBuilder':
        if i_range is None and v_range is None:
            cmd = f'TIV {chnum}'
        elif i_range is None or v_range is None:
            raise ValueError('When i_range is specified, then v_range must be '
                             'specified (and vice versa).')
        else:
            cmd = f'TIV {chnum},{i_range},{v_range}'

        self._msg.append(cmd)
        return self

    def tm(self, mode: Union[constants.TM.Mode, int]) -> 'MessageBuilder':
        cmd = f'TM {mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def tmacv(self,
              chnum: Union[constants.ChNr, int],
              mode: Union[constants.RangingMode, int],
              meas_range: Optional[Union[constants.TMACV.Range, str]] = None
              ) -> 'MessageBuilder':
        """
        This command monitors the MFCMU AC voltage output signal level,
        and returns the measurement data.

        Args:
            chnum: MFCMU channel number. Integer expression. 1 to 10 or
                101 to 1001. See Table 4-1 on page 16.

            mode: Ranging mode.
                Integer expression. 0 or 2.

                    - 0: Auto ranging. Initial setting.
                    - 2: Fixed range.

            meas_range: Measurement Range. This parameter must be set if
                mode=2. Set Table 4-19 on Page 30
        """
        cmd = f'TMACV {chnum},{mode}'

        if meas_range is not None:
            cmd += f',{meas_range}'

        self._msg.append(cmd)
        return self

    def tmdcv(self,
              chnum: Union[constants.ChNr, int],
              mode: Union[constants.RangingMode, int],
              meas_range: Optional[Union[constants.TMDCV.Range, int]] = None
              ) -> 'MessageBuilder':
        cmd = f'TMDCV {chnum},{mode}'

        if meas_range is not None:
            cmd += f',{meas_range}'

        self._msg.append(cmd)
        return self

    def tsc(self, enable_timestamp: bool) -> 'MessageBuilder':
        cmd = f'TSC {int(enable_timestamp)}'

        self._msg.append(cmd)
        return self

    def tsq(self) -> 'MessageBuilder':
        """
        The TSQ command returns the time data from when the TSR command is
        sent until this command is sent. The time data will be put in the
        data output buffer as same as the measurement data.
        This command is effective for all measurement modes, regardless of
        the TSC setting.
        This command is not effective for the 4 byte binary data output
        format (FMT3 and FMT4).

        Note:
            Although this command places time data in the output buffer it (
            apparently?) does not have to be a final command, hence other
            commands may follow. But this needs to be re-checked.
        """
        cmd = 'TSQ'

        self._msg.append(cmd)
        return self

    def tsr(self, chnum: Optional[int] = None) -> 'MessageBuilder':
        cmd = f'TSR' if chnum is None else f'TSR {chnum}'

        self._msg.append(cmd)
        return self

    def tst(self,
            slot: Optional[Union[constants.SlotNr, int]] = None,
            option: Optional[Union[constants.TST.Option, int]] = None
            ) -> 'MessageBuilder':
        cmd = '*TST?'

        if slot is not None:
            cmd += f' {slot}'

            if option is not None:
                cmd += f',{option}'

        self._msg.append(cmd)
        return self

    def ttc(self,
            chnum: Union[constants.ChNr, int],
            mode: Union[constants.RangingMode, int],
            meas_range: Optional[Union[constants.TTC.Range, int]] = None
            ) -> 'MessageBuilder':
        cmd = f'TTC {chnum},{mode}'

        if meas_range is not None:
            cmd += f',{meas_range}'

        self._msg.append(cmd)
        return self

    def tti(self,
            chnum: Union[constants.ChNr, int],
            ranging_type: Optional[Union[constants.IMeasRange, int]] = None
            ) -> 'MessageBuilder':
        cmd = f'TTI {chnum}'

        if ranging_type is not None:
            cmd += f',{ranging_type}'

        self._msg.append(cmd)
        return self

    def ttiv(self,
             chnum: Union[constants.ChNr, int],
             i_range: Optional[Union[constants.IMeasRange, int]] = None,
             v_range: Optional[Union[constants.VMeasRange, int]] = None
             ) -> 'MessageBuilder':
        if i_range is None and v_range is None:
            cmd = f'TTIV {chnum}'
        elif i_range is None or v_range is None:
            raise ValueError('If i_range is specified, then v_range must be '
                             'specified too (and vice versa).')
        else:
            cmd = f'TTIV {chnum},{i_range},{v_range}'

        self._msg.append(cmd)
        return self

    def ttv(self,
            chnum: Union[constants.ChNr, int],
            v_range: Optional[Union[constants.VMeasRange, int]] = None
            ) -> 'MessageBuilder':
        cmd = f'TTV {chnum}'

        if v_range is not None:
            cmd += f',{v_range}'

        self._msg.append(cmd)
        return self

    def tv(self,
           chnum: Union[constants.ChNr, int],
           v_range: Optional[Union[constants.VMeasRange, int]] = None
           ) -> 'MessageBuilder':
        cmd = f'TV {chnum}'

        if v_range is not None:
            cmd += f',{v_range}'

        self._msg.append(cmd)
        return self

    @final_command
    def unt_query(self, mode: Optional[Union[constants.UNT.Mode, int]] = None
                  ) -> 'MessageBuilder':
        cmd = 'UNT?' if mode is None else f'UNT? {mode}'

        self._msg.append(cmd)
        return self

    def var(self,
            variable_type: Union[constants.VAR.Type, int],
            n: int,
            value: Union[int, float]
            ) -> 'MessageBuilder':
        cmd = f'VAR {variable_type},{n},{value}'

        self._msg.append(cmd)
        return self

    @final_command
    def var_query(self,
                  variable_type: Union[constants.VAR.Type, int],
                  n: int
                  ) -> 'MessageBuilder':
        cmd = f'VAR? {variable_type},{n}'

        self._msg.append(cmd)
        return self

    def wacv(self,
             chnum: Union[constants.ChNr, int],
             mode: Union[constants.SweepMode, int],
             start: float,
             stop: float,
             step: float
             ) -> 'MessageBuilder':
        cmd = f'WACV {chnum},{mode},{start},{stop},{step}'

        self._msg.append(cmd)
        return self

    def wat(self,
            wait_time_type: Union[constants.WAT.Type, int],
            coeff: float,
            offset=None
            ) -> 'MessageBuilder':
        cmd = f'WAT {wait_time_type},{coeff}'

        if offset is not None:
            cmd += f',{offset}'

        self._msg.append(cmd)
        return self

    def wdcv(self,
             chnum: Union[constants.ChNr, int],
             mode: Union[constants.SweepMode, int],
             start: float,
             stop: float,
             step: float,
             i_comp: Optional[float] = None
             ) -> 'MessageBuilder':
        """
        This command sets the DC bias sweep source used for the CV (DC bias)
        sweep measurement (MM18). The sweep source will be MFCMU or SMU.
        Execution Conditions: The CN/CNX command has been executed for the
        specified channel. If you want to apply DC voltage over +/- 25 V using
        the SCUU, the SCUU must be connected correctly. The SCUU can be used
        with the MFCMU and two SMUs (MPSMU or HRSMU). The SCUU cannot be
        used if the HPSMU is connected to the SCUU or if the number of SMUs
        connected to the SCUU is only one. If the output voltage is greater
        than the allowable voltage for the interlock open condition,
        the interlock circuit must be shorted.

        Args:
            chnum : MFCMU or SMU channel number.
                Integer expression. 1 to 10 or 101 to 1001.
                See Table 4-1 on page 16.
            mode : Sweep mode. Integer expression.
                1: Linear sweep (single stair, start to stop.)
                2: Log sweep (single stair, start to stop.)
                3: Linear sweep (double stair, start to stop to start.)
                4: Log sweep (double stair, start to stop to start.)
            start : Start value of the DC bias sweep (in V). Numeric expression.
                For the log sweep, start and stop must have the same polarity.
                See Table 4-7 on page 24, Table 4-9 on page 26, or Table
                4-12 on page 27 for each measurement resource type. For
                MFCMU, 0 (initial setting) to +/- 25 V (MFCMU) or +/- 100 V (
                with SCUU) With the SCUU, the source module is automatically
                selected by the setting value. The MFCMU is used if the
                start and stop values are below +/- 25 V
                (setting resolution: 0.001 V), or the SMU is used if they
                are greater than +/- 25 V (setting resolution: 0.005 V). The
                SMU connected to the SCUU will operate with the 100 V
                limited auto ranging and 20 mA compliance settings.
            stop : Stop value of the DC bias sweep (in V).
            step : Number of steps for staircase sweep.
                Numeric expression. 1 to 1001.
            i_comp : Available only for SMU. An error occurs if the Icomp
                value is specified for the MFCMU.
                Current compliance (in A). Numeric expression.
                See Table 4-7 on page 24, Table 4-9 on page 26, or Table 4-12
                on page 27 for each measurement resource type.
                If you do not set Icomp, the previous value is used.
                Compliance polarity is automatically set to the same
                polarity as the output value, regardless of the specified Icomp.
                If the output value is 0, the compliance polarity is positive.
        """
        cmd = f'WDCV {chnum},{mode},{start},{stop},{step}'

        if i_comp is not None:
            cmd += f',{i_comp}'

        self._msg.append(cmd)
        return self

    def wfc(self,
            chnum: Union[constants.ChNr, int],
            mode: Union[constants.SweepMode, int],
            start: float,
            stop: float,
            step: float
            ) -> 'MessageBuilder':
        cmd = f'WFC {chnum},{mode},{start},{stop},{step}'

        self._msg.append(cmd)
        return self

    def wi(self,
           chnum: Union[constants.ChNr, int],
           mode: Union[constants.SweepMode, int],
           i_range: Union[constants.IOutputRange, int],
           start: float,
           stop: float,
           step: float,
           v_comp: Optional[float] = None,
           p_comp: Optional[float] = None
           ) -> 'MessageBuilder':
        cmd = f'WI {chnum},{mode},{i_range},{start},{stop},{step}'

        if v_comp is not None:
            cmd += f',{v_comp}'

            if p_comp is not None:
                cmd += f',{p_comp}'

        self._msg.append(cmd)
        return self

    def wm(self,
           abort: Union[bool, constants.Abort],
           post: Optional[Union[constants.WM.Post, int]] = None
           ) -> 'MessageBuilder':
        if isinstance(abort, bool):
            _abort = constants.Abort.ENABLED if abort \
                else constants.Abort.DISABLED
        elif isinstance(abort, constants.Abort):
            _abort = abort
        else:
            raise TypeError(f"`abort` argument has to be of type `bool` or "
                            f"`constants.Abort`.")

        cmd = f'WM {_abort}'

        if post is not None:
            cmd += f',{post}'

        self._msg.append(cmd)
        return self

    def wmacv(self,
              abort: Union[bool, constants.Abort],
              post: Optional[Union[constants.WMACV.Post, int]] = None
              ) -> 'MessageBuilder':
        if isinstance(abort, bool):
            _abort = constants.Abort.ENABLED if abort \
                else constants.Abort.DISABLED
        elif isinstance(abort, constants.Abort):
            _abort = abort
        else:
            raise TypeError(f"`abort` argument has to be of type `bool` or "
                            f"`constants.Abort`.")

        cmd = f'WMACV {_abort}'

        if post is not None:
            cmd += f',{post}'

        self._msg.append(cmd)
        return self

    def wmdcv(self,
              abort: Union[bool, constants.Abort],
              post: Optional[Union[constants.WMDCV.Post, int]] = None
              ) -> 'MessageBuilder':
        """
        This command enables or disables the automatic abort function for
        the CV (AC level) sweep measurement. The automatic abort
        function stops the measurement when one of the following conditions
        occurs.

            - NULL loop unbalance condition
            - IV amplifier saturation condition
            - Overflow on the AD converter

        This command also sets the post measurement condition of the MFCMU.
        After the measurement is normally completed, the MFCMU forces the
        value specified by the post parameter.

        If the measurement is stopped by the automatic abort function,
        the  MFCMU forces the start value.

        Args:
            abort: Automatic abort function. Integer expression. 1 or 2.
                - 1: Disables the function. Initial setting.
                - 2 Enables the function.
            post: AC level value after the measurement is normally
                completed. Possible values,
                - ``constants.WMDCV.Post.START``: Initial setting.
                - ``constants.WMDCV.Post.STOP``: Stop value.
                If this parameter is not set, the MFCMU forces the start value.
        """
        if isinstance(abort, bool):
            _abort = constants.Abort.ENABLED if abort \
                else constants.Abort.DISABLED
        elif isinstance(abort, constants.Abort):
            _abort = abort
        else:
            raise TypeError(f"`abort` argument has to be of type `bool` or "
                            f"`constants.Abort`.")

        cmd = f'WMDCV {_abort}'

        if post is not None:
            cmd += f',{post}'

        self._msg.append(cmd)
        return self

    def wmfc(self,
             abort: Union[bool, constants.Abort],
             post: Optional[Union[constants.WMFC.Post, int]]
             ) -> 'MessageBuilder':
        if isinstance(abort, bool):
            _abort = constants.Abort.ENABLED if abort \
                else constants.Abort.DISABLED
        elif isinstance(abort, constants.Abort):
            _abort = abort
        else:
            raise TypeError(f"`abort` argument has to be of type `bool` or "
                            f"`constants.Abort`.")

        cmd = f'WMFC {_abort}'

        if post is not None:
            cmd += f',{post}'

        self._msg.append(cmd)
        return self

    def wncc(self) -> 'MessageBuilder':
        cmd = 'WNCC'

        self._msg.append(cmd)
        return self

    @final_command
    def wnu_query(self) -> 'MessageBuilder':
        cmd = 'WNU?'

        self._msg.append(cmd)
        return self

    def wnx(self,
            n: int,
            chnum: Union[constants.ChNr, int],
            mode: Union[constants.WNX.Mode, int],
            ranging_type: Union[constants.IOutputRange,
                                constants.VOutputRange],
            start: float,
            stop: float,
            comp: Optional[float] = None,
            p_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'WNX {n},{chnum},{mode},{ranging_type},{start},{stop}'

        if comp is not None:
            cmd += f',{comp}'

            if p_comp is not None:
                cmd += f',{p_comp}'

        self._msg.append(cmd)
        return self

    def ws(self, mode: Optional[Union[constants.WS.Mode, int]] = None
           ) -> 'MessageBuilder':
        cmd = 'WS' if mode is None else f'WS {mode}'

        self._msg.append(cmd)
        return self

    def wsi(self,
            chnum: Union[constants.ChNr, int],
            i_range: Union[constants.IOutputRange, int],
            start: float,
            stop: float,
            v_comp: Optional[float] = None,
            p_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'WSI {chnum},{i_range},{start},{stop}'

        if v_comp is not None:
            cmd += f',{v_comp}'

            if p_comp is not None:
                cmd += f',{p_comp}'

        self._msg.append(cmd)
        return self

    def wsv(self,
            chnum: Union[constants.ChNr, int],
            v_range: Union[constants.VOutputRange, int],
            start: float,
            stop: float,
            i_comp: Optional[float] = None,
            p_comp: Optional[float] = None
            ) -> 'MessageBuilder':
        cmd = f'WSV {chnum},{v_range},{start},{stop}'

        if i_comp is not None:
            cmd += f',{i_comp}'

            if p_comp is not None:
                cmd += f',{p_comp}'

        self._msg.append(cmd)
        return self

    def wt(self,
           hold: float,
           delay: float,
           step_delay: Optional[float] = None,
           trigger_delay: Optional[float] = None,
           measure_delay: Optional[float] = None
           ) -> 'MessageBuilder':
        cmd = f'WT {hold},{delay}'

        if step_delay is not None:
            cmd += f',{step_delay}'

            if trigger_delay is not None:
                cmd += f',{trigger_delay}'

                if measure_delay is not None:
                    cmd += f',{measure_delay}'

        self._msg.append(cmd)
        return self

    def wtacv(self,
              hold: float,
              delay: float,
              step_delay: Optional[float] = None,
              trigger_delay: Optional[float] = None,
              measure_delay: Optional[float] = None
              ) -> 'MessageBuilder':
        cmd = f'WTACV {hold},{delay}'

        if step_delay is not None:
            cmd += f',{step_delay}'

            if trigger_delay is not None:
                cmd += f',{trigger_delay}'

                if measure_delay is not None:
                    cmd += f',{measure_delay}'

        self._msg.append(cmd)
        return self

    def wtdcv(self,
              hold: float,
              delay: float,
              step_delay: Optional[float] = None,
              trigger_delay: Optional[float] = None,
              measure_delay: Optional[float] = None
              ) -> 'MessageBuilder':
        cmd = f'WTDCV {hold},{delay}'

        if step_delay is not None:
            cmd += f',{step_delay}'

            if trigger_delay is not None:
                cmd += f',{trigger_delay}'

                if measure_delay is not None:
                    cmd += f',{measure_delay}'

        self._msg.append(cmd)
        return self

    def wtfc(self,
             hold: float,
             delay: float,
             step_delay: Optional[float] = None,
             trigger_delay: Optional[float] = None,
             measure_delay: Optional[float] = None
             ) -> 'MessageBuilder':
        cmd = f'WTFC {hold},{delay}'

        if step_delay is not None:
            cmd += f',{step_delay}'

            if trigger_delay is not None:
                cmd += f',{trigger_delay}'

                if measure_delay is not None:
                    cmd += f',{measure_delay}'

        self._msg.append(cmd)
        return self

    def wv(self,
           chnum: Union[constants.ChNr, int],
           mode: Union[constants.SweepMode, int],
           v_range: Union[constants.VOutputRange, int],
           start: float,
           stop: float,
           step: float,
           i_comp: Optional[float] = None,
           p_comp: Optional[float] = None
           ) -> 'MessageBuilder':
        cmd = f'WV {chnum},{mode},{v_range},{start},{stop},{step}'

        if i_comp is not None:
            cmd += f',{i_comp}'

            if p_comp is not None:
                cmd += f',{p_comp}'

        self._msg.append(cmd)
        return self

    @final_command
    def wz_query(self, timeout: Optional[float] = None) -> 'MessageBuilder':
        cmd = 'WZ?' if timeout is None else f'WZ? {timeout}'

        self._msg.append(cmd)
        return self

    def xe(self) -> 'MessageBuilder':
        cmd = 'XE'

        self._msg.append(cmd)
        return self
