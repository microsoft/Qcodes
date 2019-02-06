from __future__ import annotations
from functools import wraps
from typing import List, Union, Type
import warnings

import qcodes.instrument_drivers.Keysight.keysightb1500.constants as enums


def as_csv(l, sep=','):
    """
    Returns items in iterable ls as comma-separated string

    :param sep:
    :param l: Iterable
    :return:
    """
    return sep.join(format(x) for x in l)


def final_command(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        res._msg.set_final()

        return res

    return wrapper


class CommandList(list):
    def __init__(self):
        super().__init__()
        self.is_final = False

    def append(self, obj):
        if self.is_final:
            raise ValueError(f'Cannot add commands after `{self[-1]}`. '
                             f'`{self[-1]}` must be the last command in the '
                             f'message.')
        else:
            super().append(obj)

    def set_final(self):
        self.is_final = True

    def clear(self):
        self.is_final = False
        super().clear()

    def __str__(self):
        return as_csv(self, ';')


class MessageBuilder:

    def __init__(self) -> MessageBuilder:
        """
        Provides a Python wrapper for each of the FLEX commands that the
        KeysightB1500 undestands.

        To make usage easier also take a look at the classed defined in
        keysightb1500.constants which defines a lot of the integer constants
        that the commands expect as arguments.
        """
        self._msg = CommandList()

    @property
    def message(self) -> str:
        joined = str(self._msg)
        if len(joined) > 250:
            warnings.warn(f"Command is too long ({len(joined)}>256-termchars) "
                          f"and will overflow input buffer of instrument. "
                          f"(Consider using the ST command for very long "
                          f"programs.)",
                          stacklevel=2)
        self._msg.clear()
        return joined

    def aad(self,
            chnum: Union[enums.ChNr, int],
            adc_type: enums.AAD.Type) -> MessageBuilder:
        """
        This command is used to specify the type of the A/D converter (ADC) for
        each measurement channel.

        Execution Conditions: Enter the AIT command to set up the ADC.

        The pulsed-measurement ADC is automatically used for the pulsed spot,
        pulsed sweep, multi channel pulsed spot, multi channel pulsed sweep, or
        staircase sweep with pulsed bias measurement, even if the AAD chnum,2
        command is not executed.
        The pulsed-measurement ADC is never used for the DC measurement. Even if
        the AAD chnum,2 command is executed, the previous setting is still
        effective.

        :param chnum: SMU measurement channel number. Integer expression. 1
            to 10 or 101 to 1001. See Table 4-1 on page 16.

        :param adc_type: Type of the A/D converter. Integer expression. 0, 1,
            or 2.
            0: High-speed ADC for high speed DC measurement. Initial
            setting.
            1: High-resolution ADC. For high accurate DC
            measurement. Not available for the HCSMU, HVSMU, and MCSMU.
            2: High-speed ADC for pulsed-measurement

        :return: formatted command string
        """

        cmd = f'AAD {chnum},{adc_type}'

        self._msg.append(cmd)
        return self

    @final_command
    def ab(self) -> MessageBuilder:
        """
        The AB command aborts the present operation and subsequent command
        execution.

        This command stops the operation now in progress, such as the
        measurement execution, source setup changing, and so on. But this
        command does not change the present condition. For example, if the
        KeysightB1500 just keeps to force the DC bias, the AB command does not stop
        the DC bias output.

        Remarks: If you start an operation that you may want to abort,
        do not send any command after the command or command string that
        starts the operation. If you do, the AB command cannot enter the
        command input buffer until the intervening command execution starts,
        so the operation cannot be aborted. In this case, use a device clear
        to end the operation.
        If the AB command is entered in a command string, the other commands
        in the string are not executed. For example, the CN command in the
        following command string is not executed.

        OUTPUT @KeysightB1500;"AB;CN"

        During sweep measurement, if the KeysightB1500 receives the AB command,
        it returns only the measurement data obtained before abort. Then the
        dummy data is not returned.
        For the quasi-pulsed spot measurement, the KeysightB1500 cannot receive any
        command during the settling detection. So the AB command cannot abort
        the operation, and it will be performed after the settling detection.

        :return: formatted command string
        """

        cmd = 'AB'

        self._msg.append(cmd)
        return self

    def ach(self,
            actual: Union[enums.ChNr, int] = None,
            program: Union[enums.ChNr, int] = None) -> MessageBuilder:
        """
        The ACH command translates the specified program channel number to
        the specified actual channel number at the program execution. This
        command is useful when you use a control program created for an
        instrument, such as the 4142B, 4155B/4155C/4156B/4156C/E5260/E5270,
        and KeysightB1500, that has a module configuration different from the KeysightB1500
        actually you use. After the ACH command, enter the *OPC? command to
        confirm that the command execution is completed.

        :param actual: Channel number actually set to the KeysightB1500 instead of
            program. Integer expression. 1 to 10 or 101 to 1002. See Table 4-1
            on page 16.

        :param program: Channel number used in a program and will be replaced
            with actual. Integer expression.
            If you do not set program, this command is the same
            as ACH n,n.

        If you do not set actual and program, all channel number mapping is
        cleared.

        Remarks: The ACH commands must be put at the beginning of the
        program or before the command line that includes a program channel
        number. In the program lines that follow the ACH command, you must
        leave the program channel numbers. The measurement data is returned
        as the data of the channel program, not actual.

        :return: formatted command string
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
            mode: enums.ACT.Mode,
            coeff: int = None) -> MessageBuilder:
        """
        This command sets the number of averaging samples or the averaging
        time set to the A/D converter of the MFCMU.

        :param mode: Averaging mode. Integer expression. 0 (initial setting)
            or 2. 0: Auto mode: Defines the number of averaging samples given by
            the following formula. Then initial averaging is the number of
            averaging samples automatically set by the KeysightB1500 and you cannot
            change. Number of averaging samples = n * initial averaging

            2: Power line cycle (PLC) mode: Defines the averaging time given
            by the following formula. Averaging time = n / power line frequency

        :param coeff: Coefficient used to define the number of averaging samples
            or the averaging time. For mode=0: 1 to 1023. Initial
            setting/default setting is 2. For mode=2: 1 to 100. Initial
            setting/default setting is 1.

        :return: Formatted command string
        """
        if coeff is None:
            cmd = f'ACT {mode}'
        else:
            cmd = f'ACT {mode},{coeff}'

        self._msg.append(cmd)
        return self

    def acv(self,
            chnum: Union[enums.ChNr, int],
            voltage: float) -> MessageBuilder:
        """
        This command sets the output signal level of the MFCMU, and starts
        the AC voltage output. Output signal frequency is set by the FC command.

        Execution conditions: The CN/CNX command has been executed for the
        specified channel.

        :param chnum: MFCMU channel number. Integer expression. 1 to 10 or
            101 to 1001. See Table 4-1 on page 16.

        :param voltage: Oscillator level of the output AC voltage (in V).
            Numeric expression.

        :return: formatted command string
        """

        cmd = f'ACV {chnum},{voltage}'

        self._msg.append(cmd)
        return self

    def adj(self,
            chnum: Union[enums.ChNr, int],
            mode: enums.ADJ.Mode) -> MessageBuilder:
        """
        This command selects the MFCMU phase compensation mode. This command
        initializes the MFCMU.


        :param chnum: MFCMU channel number. Integer expression. 1 to 10 or
            101 to 1001. See Table 4-1 on page 16.

        :param mode: Phase compensation mode. Integer expression. 0 or 1. 0:
            Auto mode. Initial setting. 1: Manual mode. 2: Load adaptive mode.
            For mode=0, the KeysightB1500 sets the compensation data automatically. For
            mode=1, execute the ADJ? command to perform the phase compensation
            and set the compensation data. For mode=2, the KeysightB1500 performs the
            phase compensation before every measurement. It is useful when there
            are wide load fluctuations by changing the bias and so on.

        :return: formatted command string
        """
        cmd = f'ADJ {chnum},{mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def adj_query(self,
                  chnum: Union[enums.ChNr, int],
                  mode: enums.ADJQuery.Mode = None) -> MessageBuilder:
        """
        This command performs the MFCMU phase compensation, and sets the
        compensation data to the KeysightB1500. This command also returns the
        execution results.

        This command resets the MFCMU. Before executing this command, set the
        phase compensation mode to manual by using the ADJ command. During this
        command, open the measurement terminals at the end of the device side.
        This command execution will take about 30 seconds. The compensation data
        is cleared by turning the KeysightB1500 off.

        Query response: 0: Phase compensation measurement was normally
        completed.
        1: Phase compensation measurement failed.
        2: Phase compensation measurement was aborted.
        3: Phase compensation measurement has not been performed.

        :param chnum: MFCMU channel number. Integer expression. 1 to 10 or
            101 to 1001. See Table 4-1 on page 16.

        :param mode: Command operation mode.
            0: Use the last phase compensation data without measurement. 1:
            Perform the phase compensation data measurement. If the mode
            parameter is not set, mode=1 is set.

        :return: formatted command string
        """

        if mode is None:
            cmd = f'ADJ? {chnum}'
        else:
            cmd = f'ADJ? {chnum},{mode}'

        self._msg.append(cmd)
        return self

    def ait(self,
            adc_type: enums.AIT.Type,
            mode: enums.AIT.Mode,
            coeff: Union[int, float] = None) -> MessageBuilder:
        """
        This command is used to set the operation mode and the setup
        parameter of the A/D converter (ADC) for each ADC type.

        Execution conditions: Enter the AAD command to specify the ADC type
        for each measurement channel.

        :param adc_type: Type of the A/D converter. Integer expression. 0, 1,
            or 2. 0: High-speed ADC 1: High-resolution ADC. Not available for
            the HCSMU, HVSMU and MCSMU. 2: High-speed ADC for pulsed-measurement

        :param mode: ADC operation mode. Integer expression. 0, 1, 2,
            or 3. 0: Auto mode. Initial setting. 1: Manual mode 2: Power line
            cycle (PLC) mode 3: Measurement time mode. Not available for the
            high-resolution ADC.

        :param coeff: Coefficient used to define the integration time or the
            number of averaging samples, integer expression, for mode=0, 1,
            and 2. Or the actual measurement time, numeric expression,
            for mode=3. See Table 4-21.

        The pulsed-measurement ADC (type=2) is available for the all
        measurement channels used for the pulsed spot, pulsed sweep,
        multi channel pulsed spot, multi channel pulsed sweep, or staircase
        sweep with pulsed bias measurement.

        :return: Formatted command string
        """

        if coeff is None:
            cmd = f'AIT {adc_type},{mode}'
        else:
            cmd = f'AIT {adc_type},{mode},{coeff}'

        self._msg.append(cmd)
        return self

    def aitm(self,
             operation_type: enums.APIVersion) -> MessageBuilder:
        """
        Only for the current measurement by using HRSMU. This command sets
        the operation type of the high-resolution ADC that is set to the
        power line cycle (PLC) mode by the AIT 1, 2, N command.

        This setting is cleared by the *RST or a device clear.


        :param operation_type: Operation type. Integer expression. 0 or 1.
            0: KeysightB1500 standard operation. Initial setting.
            1: Classic operation. Performs the operation similar to the PLC
            mode of Keysight 4156.

        :return: formatted command string
        """

        cmd = f'AITM {operation_type}'

        self._msg.append(cmd)
        return self

    @final_command
    def aitm_query(self) -> MessageBuilder:
        """
        This command returns the operation type of the high-resolution ADC
        that is set by the AITM command.

        :return: formatted command string
        """

        cmd = f'AITM?'

        self._msg.append(cmd)
        return self

    def als(self,
            chnum: Union[enums.ChNr, int],
            n_bytes: int,
            block: bytes):

        # The format specification in the manual is a bit unclear, and I do
        # not have the module installed to test this command, hence:
        raise NotImplementedError

        # A possible way might be:
        # cmd = f'ALS {chnum},{n_bytes} {block.decode()}'  # No comma between
        # n_bytes and block!

        # self._msg.append(cmd)
        # return self

    @final_command
    def als_query(self, chnum: Union[enums.ChNr, int]) -> MessageBuilder:
        """
        This query command returns the ALWG sequence data of the specified
        SPGU channel.

        Query response: Returns the ALWG sequence data (binary format,
        big endian).

        :param chnum: SPGU channel number. Integer expression. 1 to 10 or 101
            to 1002. See Table 4-1.

        :return: formatted command string
        """

        cmd = f'ALS? {chnum}'

        self._msg.append(cmd)
        return self

    def alw(self,
            chnum: Union[enums.ChNr, int],
            n_bytes: int,
            block: bytes):

        # The format specification in the manual is a bit unclear, and I do
        # not have the module installed to test this command, hence:
        raise NotImplementedError

        # A possible way might be:
        # cmd = f'ALW {chnum},{n_bytes} {block.decode()}'
        # self._msg.append(cmd)
        # return self

    @final_command
    def alw_query(self, chnum: Union[enums.ChNr, int]) -> MessageBuilder:
        """
        This query command returns the ALWG pattern data of the specified
        SPGU channel.

        Query response: Returns the ALWG pattern data (binary format,
        big endian).

        :param chnum: SPGU channel number. Integer expression. 1 to 10 or 101
            to 1002. See Table 4-1.

        :return:
        """
        cmd = f'ALW? {chnum}'

        self._msg.append(cmd)
        return self

    def av(self,
           number: int,
           mode: enums.AV.Mode = None) -> MessageBuilder:
        """
        This command sets the number of averaging samples of the high-speed
        ADC (A/D converter). This command is not effective for the
        high-resolution ADC. This command is not effective for the
        measurements using pulse.

        :param number: 1 to 1023, or -1 to -100. Initial setting is 1. For
            positive number input, this value specifies the number of samples
            depended on the mode value. See below. For negative number input,
            this parameter specifies the number of power line cycles (PLC) for
            one point measurement. The Keysight KeysightB1500 gets 128 samples in 1 PLC.
            Ignore the mode parameter.

        :param mode: Averaging mode. Integer expression. This parameter is
            meaningless for negative number.

            0: Auto mode (default setting). Number of samples = number * initial
            number
            1: Manual mode. Number of samples = number
            where initial number means the number of samples the Keysight KeysightB1500
            automatically sets and you cannot change. For voltage measurement,
            initial number=1. For current measurement, see Table 4-22.
            If you select the manual mode, number must be initial number or more
            to satisfy the specifications.

        :return: formatted command string
        """
        if mode is None:
            cmd = f'AV {number}'
        else:
            cmd = f'AV {number},{mode}'

        self._msg.append(cmd)
        return self

    def az(self, do_autozero: bool) -> MessageBuilder:
        """
        This command is used to enable or disable the ADC zero function that
        is the function to cancel offset of the high-resolution A/D
        converter. This function is especially effective for low voltage
        measurements. Power on, *RST command, and device clear disable the
        function. This command is effective for the high-resolution A/D
        converter, not effective for the high-speed A/D converter.

        Remarks: Set the function to OFF in cases that the measurement speed
        is more important than the measurement accuracy. This roughly halves
        the integration time.

        :param do_autozero: Mode ON or OFF.
            0: OFF. Disables the function. Initial setting.
            1: ON. Enables the function.

        :return: formatted command string
        """

        cmd = f'AZ {int(do_autozero)}'

        self._msg.append(cmd)
        return self

    @final_command
    def bc(self) -> MessageBuilder:
        """
        The BC command clears the output data buffer that stores measurement
        data and query command response data. This command does not change
        the measurement settings.

        Note: Multi command statement is not allowed for this command.

        :return: formatted command string
        """

        cmd = 'BC'

        self._msg.append(cmd)
        return self

    def bdm(self,
            interval: enums.BDM.Interval,
            mode: enums.BDM.Mode = None) -> MessageBuilder:
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

        :param interval: Settling detection interval. Numeric expression.
            0: Short. Initial setting.
            1: Long. For measurements of the devices that have the stray
            capacitance, or the measurements with the compliance less than 1 uA

        :param mode: Measurement mode. Numeric expression.

            0: Voltage measurement mode. Default setting.
            1: Current measurement mode.

        :return: formatted command string
        """

        if mode is None:
            cmd = f'BDM {interval}'
        else:
            cmd = f'BDM {interval},{mode}'

        self._msg.append(cmd)
        return self

    def bdt(self,
            hold: float,
            delay: float) -> MessageBuilder:
        """
        The BDT command specifies the hold time and delay time for the
        quasi-pulsed measurements.

        :param hold: Hold time (in sec). Numeric expression. 0 to 655.35 s,
            0.01 s resolution. Initial setting is 0.

        :param delay: Delay time (in sec). Numeric expression. 0 to 6.5535 s,
            0.0001 s resolution. Initial setting is 0.

        :return: formatted command string
        """

        cmd = f'BDT {hold},{delay}'

        self._msg.append(cmd)
        return self

    def bdv(self,
            chnum: Union[enums.ChNr, int],
            v_range: enums.VOutputRange,
            start: float,
            stop: float,
            i_comp: float = None) -> MessageBuilder:
        """
        The BDV command specifies the quasi-pulsed voltage source and its
        parameters.

        Remarks: The time forcing the stop value will be approximately 1.5 ms to
        1.8 ms with the following settings:

            - BDM, BDT command parameters: interval=0, mode=0, delay=0

            - AV or AAD/AIT command parameters: initial setting


        :param chnum: SMU source channel number. Integer expression. 1 to 10
            or 101 to 1001. See Table 4-1 on page 16.

        :param v_range: Ranging type for quasi-pulsed source. Integer
            expression. The output range will be set to the minimum range that
            covers both start and stop values. For the limited auto ranging,
            the instrument never uses the range less than the specified range.
            See Table 4-4 on page 20.

        :param start: Start or stop voltage (in V). Numeric expression. See
            Table 4-7 on page 24. 0 to +-100 for MPSMU/HRSMU, or 0 to +-200 for
            HPSMU |start - stop| must be 10V or more.

        :param stop: (see start)

        :param i_comp: Current compliance (in A). Numeric expression. See
            Table 4-7 on page 4-24. If you do not set Icomp, the previous value
            is used. The compliance polarity is automatically set to the same
            polarity as the stop value, regardless of the specified Icomp value.
            If stop=0, the polarity is positive.

        :return:
        """

        if i_comp is None:
            cmd = f'BDV {chnum},{v_range},{start},{stop}'
        else:
            cmd = f'BDV {chnum},{v_range},{start},{stop},{i_comp}'

        self._msg.append(cmd)
        return self

    def bgi(self,
            chnum: Union[enums.ChNr, int],
            searchmode: enums.BinarySearchMode,
            stop_condition: Union[float, int],
            i_range: enums.IMeasRange,
            target: float) -> MessageBuilder:
        """
        The BGI command sets the current monitor channel for the binary
        search measurement (MM15). This command setting clears, and is
        cleared by, the BGV command setting.

        This command ignores the RI command setting.

        Remarks: In the limit search mode, if search cannot find the search
        target and the following two conditions are satisfied, the KeysightB1500
        repeats the binary search between the last source value and the
        source start value.
          - target is between the data at source start value and the last
            measurement data.
          - target is between the data at source stop value and the data at:
            source value = | stop - start | / 2.

        If the search cannot find the search target and the following two
        conditions are satisfied, the KeysightB1500 repeats the binary search between
        the last source value and the source stop value.

          - target is between the data at source stop value and the last
            measurement data.
          - target is between the data at source start value and the data at:
            source value = | stop - start | / 2.

        :param chnum: SMU search monitor channel number. Integer expression.
            1 to 10 or 101 to 1001. See Table 4-1 on page 16.

        :param searchmode: Search mode (0:limit mode or 1:repeat mode)

        :param stop_condition: The meaning of stop_condition depends on the mode setting.
            if mode==0: Limit value for the search target (target). The
            search stops when the monitor data reaches target +- stop_condition.
            Numeric expression. Positive value. in A. Setting resolution:
            range/20000. where range means the measurement range actually
            used for the measurement.
            if mode==1: Repeat count. The search stops when the repeat count
            of the operation that changes the source output value is over the
            specified value. Numeric expression. 1 to 16.

        :param i_range: Measurement ranging type. Integer expression. The
            measurement range will be set to the minimum range that covers the
            target value. For the limited auto ranging, the instrument never
            uses the range less than the specified range. See Table 4-3 on
            page 19.

        :param target: Search target current (in A). Numeric expression.
            0 to +-0.1 A (MPSMU/HRSMU/MCSMU).
            0 to +-1 A (HPSMU/HCSMU).
            0 to +-2 A (DHCSMU).
            0 to +-0.008 A (HVSMU).

        :return: formatted command string

        """

        cmd = f'BGI {chnum},{searchmode},{stop_condition},{i_range},{target}'

        self._msg.append(cmd)
        return self

    def bgv(self,
            chnum: Union[enums.ChNr, int],
            searchmode: enums.BinarySearchMode,
            stop_condition: Union[float, int],
            v_range: enums.VMeasRange,
            target: float) -> MessageBuilder:
        """
        The BGV command specifies the voltage monitor channel and its search
        parameters for the binary search measurement (MM15). This command
        setting clears, and is cleared by, the BGI command setting. This
        command ignores the RV command setting.

        Remarks: In the limit search mode, if search cannot find the search
        target and the following two conditions are satisfied, the KeysightB1500
        repeats the binary search between the last source value and the
        source start value.
          - target is between the data at source start value and the last
            measurement data.
          - target is between the data at source stop value and the data at:
            source value = | stop - start | / 2.

        If the search cannot find the search target and the following two
        conditions are satisfied, the KeysightB1500 repeats the binary search between
        the last source value and the source stop value.
          - target is between the data at source stop value and the last
            measurement data.
          - target is between the data at source start value and the data at:
          source value = | stop - start | / 2.


        :param chnum: SMU search monitor channel number. Integer expression.
            1 to 10 or 101 to 1001. See Table 4-1 on page 16.

        :param searchmode: Search mode (0:limit mode or 1:repeat mode)

        :param stop_condition: The meaning of stop_condition depends on the mode setting.
            if mode==0: Limit value for the search target (target). The
            search stops when the monitor data reaches target +- stop_condition.
            Numeric expression. Positive value. in V. Setting resolution:
            range/20000. where range means the measurement range actually
            used for the measurement.
            if mode==1: Repeat count. The search stops when the repeat count
            of the operation that changes the source output value is over the
            specified value. Numeric expression. 1 to 16.

        :param v_range: Measurement ranging type. Integer expression. The
            measurement range will be set to the minimum range that covers the
            target value. For the limited auto ranging, the instrument never
            uses the range less than the specified range. See Table 4-2 on
            page 17.

        :param target: Search target voltage (in V). Numeric expression.
            0 to +-100 V (MPSMU/HRSMU) 0 to +-200 V (HPSMU)
            0 to +-30 V (MCSMU)
            0 to +-40 V (HCSMU/DHCSMU)
            0 to +-3000 V (HVSMU)

        :return: formatted command string
        """

        cmd = f'BGV {chnum},{searchmode},{stop_condition},{v_range},{target}'

        self._msg.append(cmd)
        return self

    def bsi(self,
            chnum: Union[enums.ChNr, int],
            i_range: enums.IOutputRange,
            start: float,
            stop: float,
            v_comp=None) -> MessageBuilder:
        """
        The BSI command sets the current search source for the binary search
        measurement (MM15). After search stops, the search channel forces the
        value specified by the BSM command.

        This command clears the BSV, BSSI, and BSSV command settings. This
        command setting is cleared by the BSV command.

        Execution conditions: If Vcomp value is greater than the allowable
        voltage for the interlock open condition, the interlock circuit must
        be shorted.

        :param chnum: SMU search source channel number. Integer expression.
            1 to 10 or 101 to 1001. See Table 4-1 on page 16.

        :param i_range: Output ranging type. Integer expression. The
            output range will be set to the minimum range that covers both start
            and stop values. For the limited auto ranging, the instrument never
            uses the range less than the specified range. See Table 4-5 on
            page 22.

        :param start: Search start or stop current (in A). Numeric
            expression. See Table 4-6 on page 23, Table 4-8 on page 25, or Table
            4-11 on page 27 for each measurement resource type. The start and
            stop must have different values.

        :param stop: (see stop)

        :param v_comp: Voltage compliance value (in V). Numeric expression.
            See Table 4-6 on page 23, Table 4-8 on page 25, or Table 4-11 on
            page 27 for each measurement resource type. If you do not specify
            Vcomp, the previous value is set.

        :return:
        """
        if v_comp is None:
            cmd = f'BSI {chnum},{i_range},{start},{stop}'
        else:
            cmd = f'BSI {chnum},{i_range},{start},{stop},{v_comp}'

        self._msg.append(cmd)
        return self

    def bsm(self,
            mode: enums.BSM.Mode,
            abort: enums.Abort,
            post: enums.BSM.Post = None) -> MessageBuilder:
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

        :param mode: Source output control mode, 0 (normal mode) or 1 (
            cautious mode). If you do not enter this command, the normal mode is
            set. See Figure 4-2.

        :param abort: Automatic abort function. Integer expression.
            1: Disables the function. Initial setting.
            2: Enables the function.

        :param post: Source output value after the search operation is
            normally completed. Integer expression.
            1: Start value. Initial setting.
            2: Stop value.
            3: Output value when the search target value is get.
            If this parameter is not set, the search source forces the start
            value.

        :return:
        """
        if post is None:
            cmd = f'BSM {mode},{abort}'
        else:
            cmd = f'BSM {mode},{abort},{post}'

        self._msg.append(cmd)
        return self

    def bssi(self,
             chnum: Union[enums.ChNr, int],
             polarity: enums.Polarity,
             offset: float,
             v_comp: float = None) -> MessageBuilder:
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

        :param chnum: SMU synchronous source channel number. Integer
            expression. 1 to 10 or 101 to 1001. See Table 4-1 on page 16.

        :param polarity: Polarity of the BSSI output for the BSI output. 0:
            Negative. BSSI output = -BSI output + offset 1: Positive.
            BSSI output = BSI output + offset

        :param offset: Offset current (in A). Numeric expression. See Table
            4-6 on page 23, Table 4-8 on page 25, or Table 4-11 on page 27 for
            each measurement resource type.
            Both primary and synchronous search sources will use the same
            output range. So check the output range set to the BSI command to
            determine the synchronous source outputs.

        :param v_comp: Voltage compliance value (in V). Numeric expression.
            If you do not specify Vcomp, the previous value is set.

        :return:
        """
        if v_comp is None:
            cmd = f'BSSI {chnum},{polarity},{offset}'
        else:
            cmd = f'BSSI {chnum},{polarity},{offset},{v_comp}'

        self._msg.append(cmd)
        return self

    def bssv(self,
             chnum: Union[enums.ChNr, int],
             polarity: enums.Polarity,
             offset: float,
             i_comp=None) -> MessageBuilder:
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

        :param chnum: SMU synchronous source channel number. Integer
            expression. 1 to 10 or 101 to 1001. See Table 4-1 on page 16.

        :param polarity: Polarity of the BSSV output for the BSV output. 0:
            Negative. BSSV output = -BSV output + offset 1: Positive.
            BSSV output = BSV output + offset

        :param offset: Offset voltage (in V). Numeric expression. See Table
            4-7 on page 24, Table 4-9 on page 26, Table 4-12 on page 27, or
            Table 4-15 on page 28 for each measurement resource type. Both
            primary and synchronous search sources will use the same output
            range. So check the output range set to the BSV command to
            determine the synchronous source outputs.

        :param i_comp: Current compliance value (in A). Numeric expression.
            If you do not specify Icomp, the previous value is set. Zero amps
            (0 A) is not a valid value for the Icomp parameter.

        :return:
        """
        if i_comp is None:
            cmd = f'BSSV {chnum},{polarity},{offset}'
        else:
            cmd = f'BSSV {chnum},{polarity},{offset},{i_comp}'

        self._msg.append(cmd)
        return self

    def bst(self,
            hold: float,
            delay: float) -> MessageBuilder:
        """
        The BST command sets the hold time and delay time for the binary
        search measurement (MM15). If you do not enter this command,
        all parameters are set to 0.

        :param hold: Hold time (in seconds) that is the wait time after
            starting the search measurement and before starting the delay time
            for the first search point. Numeric expression. 0 to 655.35 sec.
            0.01 sec resolution.

        :param delay: Delay time (in seconds) that is the wait time after
            starting to force a step output value and before starting a step
            measurement. Numeric expression. 0 to 65.535 sec. 0.0001 sec
            resolution.

        :return:
        """
        cmd = f'BST {hold},{delay}'

        self._msg.append(cmd)
        return self

    def bsv(self,
            chnum: Union[enums.ChNr, int],
            v_range: enums.VOutputRange,
            start: float,
            stop: float,
            i_comp: float = None) -> MessageBuilder:
        """

        The BSV command sets the voltage search source for the binary search
        measurement (MM15). After search stops, the search channel forces the
        value specified by the BSM command. This command clears the BSI,
        BSSI, and BSSV command settings. This command setting is cleared by
        the BSI command.

        Execution conditions: If the output voltage is greater than the
        allowable voltage for the interlock open condition, the interlock
        circuit must be shorted.



        :param chnum: SMU search source channel number. Integer
            expression. 1 to 10 or 101 to 1001. See Table 4-1 on page 16.

        :param v_range: Output ranging type. Integer expression. The
            output range will be set to the minimum range that covers both
            start and stop values. For the limited auto ranging,
            the instrument never uses the range less than the specified
            range. See Table 4-4 on page 20.

        :param start: Search start or stop voltage (in V). Numeric
            expression. See Table 4-7 on page 24, Table 4-9 on page 26,
            Table 4-12 on page 27, or Table 4-15 on page 28 for each
            measurement resource type. The start and stop parameters must
            have different values.

        :param stop: (see stop)

        :param i_comp: Current compliance value (in A). Numeric
            expression. See Table 4-7 on page 24, Table 4-9 on page 26,
            Table 4-12 on page 27, or Table 4-15 on page 28 for each
            measurement resource type. If you do not specify Icomp,
            the previous value is set. Zero amps (0 A) is not allowed for
            Icomp.

        :return:
        """
        if i_comp is None:
            cmd = f'BSV {chnum},{v_range},{start},{stop}'
        else:
            cmd = f'BSV {chnum},{v_range},{start},{stop},{i_comp}'

        self._msg.append(cmd)
        return self

    def bsvm(self, mode: enums.BSVM.DataOutputMode) -> MessageBuilder:
        """
        The BSVM command selects the data output mode for the binary search
        measurement (MM15).

        :param mode: Data output mode. Integer expression. 0 : Returns
            Data_search only (initial setting). 1 : Returns Data_search and
            Data_sense. Data_search is the value forced by the search output
            channel set by BSI or BSV. Data_sense is the value measured by
            the monitor channel set by BGI or BGV. For data output format,
            refer to “Data Output Format” on page 1-25.

        :return:
        """
        cmd = f'BSVM {mode}'

        self._msg.append(cmd)
        return self

    def ca(self, slot: enums.SlotNr = None) -> MessageBuilder:
        """
        This command performs the self-calibration.

        The *OPC? command should be entered after this command to confirm the
        completion of the self-calibration. Module condition after this
        command is the same as the condition by the CL command.

        Execution conditions: No channel must be in the high voltage state (
        forcing more than the allowable voltage for the interlock open
        condition, or voltage compliance set to more than it). Before
        starting the calibration, open the measurement terminals.

        Remarks: Failed modules are disabled, and can only be enabled by the
        RCV command.

        Note: To send CA command to Keysight KeysightB1500 installed with ASU If you
        send the CA command to the KeysightB1500 installed with the ASU (Atto Sense
        and Switch Unit), the KeysightB1500 executes the self-calibration and the 1
        pA range offset measurement for the measurement channels connected to
        the ASUs. The offset data is temporarily memorized until the KeysightB1500 is
        turned off, and is used for the compensation of the data measured by
        the 1 pA range of the channels. The KeysightB1500 performs the data
        compensation automatically and returns the compensated data. Since
        the KeysightB1500 is turned on, if you do not send the CA command, the KeysightB1500
        performs the data compensation by using the pre-stored offset data.


        :param slot: Slot number where the module under self-calibration
            has been installed. 1 to 10. Integer expression. If slot is not
            specified, the self-calibration is performed for the mainframe
            and all modules.
            If slot specifies the slot that installs no module, this command
            causes an error.

        :return:
        """
        if slot is None:
            cmd = 'CA'
        else:
            cmd = f'CA {slot}'

        self._msg.append(cmd)
        return self

    @final_command
    def cal_query(self, slot: enums.SlotNr = None) -> MessageBuilder:
        """
        This query command performs the self-calibration, and returns the
        results. After this command, read the results soon. Module condition
        after this command is the same as the condition by the CL command.

        Execution Conditions: No channel must be in the high voltage state
        (forcing more than the allowable voltage for the interlock open
        condition, or voltage compliance set to more than it).

        Before starting the calibration, open the measurement terminals.


        :param slot: Slot number where the module under self-calibration
            has been installed. 1 to 10. Or 0 or 11. Integer expression. 0:
            All modules and mainframe. Default setting. 11: Mainframe.
            If slot specifies the slot that installs no module, this command
            causes an error.

        :return:
        """
        if slot is None:
            cmd = '*CAL?'
        else:
            cmd = f'*CAL? {slot}'

        self._msg.append(cmd)
        return self

    def cl(self,
           channels: List[Union[enums.ChNr, int]] = None) -> MessageBuilder:
        if channels is None:
            cmd = 'CL'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'CL { as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def clcorr(self,
               chnum: Union[enums.ChNr, int],
               mode: enums.CLCORR.Mode) -> MessageBuilder:
        cmd = f'CLCORR {chnum},{mode}'

        self._msg.append(cmd)
        return self

    def cm(self, do_autocal: bool) -> MessageBuilder:
        cmd = f'CM {int(do_autocal)}'

        self._msg.append(cmd)
        return self

    def cmm(self,
            chnum: Union[enums.ChNr, int],
            mode: enums.CMM.Mode) -> MessageBuilder:
        cmd = f'CMM {chnum},{mode}'

        self._msg.append(cmd)
        return self

    def cn(self,
           channels: List[Union[enums.ChNr, int]] = None) -> MessageBuilder:
        if channels is None:
            cmd = 'CN'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'CN {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def cnx(self,
            channels: List[Union[enums.ChNr, int]] = None) -> MessageBuilder:
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
                   chnum: Union[enums.ChNr, int],
                   corr: enums.CalibrationType) -> MessageBuilder:
        cmd = f'CORR? {chnum},{corr}'

        self._msg.append(cmd)
        return self

    def corrdt(self,
               chnum: Union[enums.ChNr, int],
               freq: float,
               open_r: float,
               open_i: float,
               short_r: float,
               short_i: float,
               load_r: float,
               load_i: float) -> MessageBuilder:
        cmd = f'CORRDT {chnum},{freq},{open_r},{open_i},{short_r},' \
              f'{short_i},{load_r},{load_i}'

        self._msg.append(cmd)
        return self

    @final_command
    def corrdt_query(self,
                     chnum: Union[enums.ChNr, int],
                     index: int) -> MessageBuilder:
        cmd = f'CORRDT? {chnum},{index}'

        self._msg.append(cmd)
        return self

    def corrl(self, chnum: Union[enums.ChNr, int], freq) -> MessageBuilder:
        cmd = f'CORRL {chnum},{freq}'

        self._msg.append(cmd)
        return self

    @final_command
    def corrl_query(self,
                    chnum: Union[enums.ChNr, int],
                    index: int = None) -> MessageBuilder:
        if index is None:
            cmd = f'CORRL? {chnum}'
        else:
            cmd = f'CORRL? {chnum},{index}'

        self._msg.append(cmd)
        return self

    @final_command
    def corrser_query(self,
                      chnum: Union[enums.ChNr, int],
                      use_immediately: bool,
                      delay: float,
                      interval: float,
                      count: int) -> MessageBuilder:
        cmd = f'CORRSER? {chnum},{int(use_immediately)},{delay},{interval},' \
              f'{count}'

        self._msg.append(cmd)
        return self

    def corrst(self,
               chnum: Union[enums.ChNr, int],
               corr: enums.CalibrationType,
               state: bool) -> MessageBuilder:
        cmd = f'CORRST {chnum},{corr},{int(state)}'

        self._msg.append(cmd)
        return self

    @final_command
    def corrst_query(self,
                     chnum: Union[enums.ChNr, int],
                     corr: enums.CalibrationType) -> MessageBuilder:
        cmd = f'CORRST {chnum},{corr}'

        self._msg.append(cmd)
        return self

    def dcorr(self, chnum: Union[enums.ChNr, int],
              corr: enums.CalibrationType,
              mode: enums.DCORR.Mode,
              primary: float,
              secondary: float) -> MessageBuilder:
        cmd = f'DCORR {chnum},{corr},{mode},{primary},{secondary}'

        self._msg.append(cmd)
        return self

    @final_command
    def dcorr_query(self,
                    chnum: Union[enums.ChNr, int],
                    corr: enums.CalibrationType) -> MessageBuilder:
        cmd = f'DCORR? {chnum},{corr}'

        self._msg.append(cmd)
        return self

    def dcv(self,
            chnum: Union[enums.ChNr, int],
            voltage: float) -> MessageBuilder:
        cmd = f'DCV {chnum},{voltage}'

        self._msg.append(cmd)
        return self

    def di(self,
           chnum: Union[enums.ChNr, int],
           i_range: enums.IOutputRange,
           current: float,
           v_comp: float = None,
           comp_polarity: enums.CompliancePolarityMode = None,
           v_range: enums.VOutputRange = None) -> MessageBuilder:
        if v_comp is None:
            cmd = f'DI {chnum},{i_range},{current}'
        elif comp_polarity is None:
            cmd = f'DI {chnum},{i_range},{current},{v_comp}'
        elif v_range is None:
            cmd = f'DI {chnum},{i_range},{current},{v_comp},' \
                  f'{comp_polarity}'
        else:
            cmd = f'DI {chnum},{i_range},{current},{v_comp},{comp_polarity},' \
                  f'{v_range}'

        self._msg.append(cmd)
        return self

    @final_command
    def diag_query(self,
                   item: enums.DIAG.Item) -> MessageBuilder:
        cmd = f'DIAG? {item}'

        self._msg.append(cmd)
        return self

    def do(self, program_numbers: List[int]) -> MessageBuilder:
        if len(program_numbers) > 8:
            raise ValueError("A maximum of 8 programs can be specified.")
        else:
            cmd = f'DO {as_csv(program_numbers)}'

        self._msg.append(cmd)
        return self

    def dsmplarm(self,
                 chnum: Union[enums.ChNr, int],
                 count: int) -> MessageBuilder:
        event_type = 1  # No other option in user manual, so hard coded here
        cmd = f'DSMPLARM {chnum},{event_type},{count}'

        self._msg.append(cmd)
        return self

    def dsmplflush(self, chnum: Union[enums.ChNr, int]) -> MessageBuilder:
        cmd = f'DSMPLFLUSH {chnum}'

        self._msg.append(cmd)
        return self

    def dsmplsetup(self,
                   chnum: Union[enums.ChNr, int],
                   count: int,
                   interval: float,
                   delay: float = None) -> MessageBuilder:
        if delay is None:
            cmd = f'DSMPLSETUP {chnum},{count},{interval}'
        else:
            cmd = f'DSMPLSETUP {chnum},{count},{interval},{delay}'

        self._msg.append(cmd)
        return self

    def dv(self,
           chnum: Union[enums.ChNr, int],
           v_range: enums.VOutputRange,
           voltage: float,
           i_comp: float = None,
           comp_polarity: enums.CompliancePolarityMode = None,
           i_range: enums.IOutputRange = None) -> MessageBuilder:
        if i_comp is None:
            cmd = f'DV {chnum},{v_range},{voltage}'
        elif comp_polarity is None:
            cmd = f'DV {chnum},{v_range},{voltage},{i_comp}'
        elif i_range is None:
            cmd = f'DV {chnum},{v_range},{voltage},{i_comp},' \
                  f'{comp_polarity}'
        else:
            cmd = f'DV {chnum},{v_range},{voltage},{i_comp},{comp_polarity},' \
                  f'{i_range}'

        self._msg.append(cmd)
        return self

    def dz(self, channels: List[Union[enums.ChNr, int]] = None) -> \
            MessageBuilder:
        if channels is None:
            cmd = 'DZ'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'DZ {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    @final_command
    def emg_query(self, errorcode: int) -> MessageBuilder:
        cmd = f'EMG? {errorcode}'

        self._msg.append(cmd)
        return self

    def end(self) -> MessageBuilder:
        cmd = 'END'

        self._msg.append(cmd)
        return self

    def erc(self, value: int) -> MessageBuilder:
        mode = 2  # Only 2 is valid for KeysightB1500
        cmd = f'ERC {mode},{value}'

        self._msg.append(cmd)
        return self

    def ercmaa(self,
               mfcmu: enums.SlotNr,
               hvsmu: enums.SlotNr,
               mpsmu: enums.SlotNr) -> MessageBuilder:
        cmd = f'ERCMAA {mfcmu},{hvsmu},{mpsmu}'

        self._msg.append(cmd)
        return self

    @final_command
    def ercmaa_query(self) -> MessageBuilder:
        cmd = f'ERCMAA?'

        self._msg.append(cmd)
        return self

    def ercmagrd(self,
                 guard_mode: enums.ERCMAGRD.Guard = None) -> MessageBuilder:
        if guard_mode is None:
            cmd = 'ERCMAGRD'
        else:
            cmd = f'ERCMAGRD {guard_mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def ercmagrd_query(self) -> MessageBuilder:
        cmd = 'ERCMAGRD?'

        self._msg.append(cmd)
        return self

    def ercmaio(self, cmhl=None, acgs=None, bias=None,
                corr=None) -> MessageBuilder:
        if cmhl is None:
            cmd = f'ERCMAIO'
        elif acgs is None:
            cmd = f'ERCMAIO {cmhl}'
        elif bias is None:
            cmd = f'ERCMAIO {cmhl},{acgs}'
        elif corr is None:
            cmd = f'ERCMAIO {cmhl},{acgs},{bias}'
        else:
            cmd = f'ERCMAIO {cmhl},{acgs},{bias},{corr}'

        self._msg.append(cmd)
        return self

    @final_command
    def ercmaio_query(self) -> MessageBuilder:
        cmd = 'ERCMAIO?'

        self._msg.append(cmd)
        return self

    def ercmapfgd(self) -> MessageBuilder:
        cmd = 'ERCMAPFGD'

        self._msg.append(cmd)
        return self

    def erhpa(self,
              hvsmu: Union[enums.ChNr, int],
              hcsmu: Union[enums.ChNr, int],
              hpsmu: Union[enums.ChNr, int]) -> MessageBuilder:
        cmd = f'ERHPA {hvsmu},{hcsmu},{hpsmu}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpa_query(self) -> MessageBuilder:
        cmd = 'ERHPA?'

        self._msg.append(cmd)
        return self

    def erhpe(self, onoff: bool) -> MessageBuilder:
        cmd = f'ERHPE {int(onoff)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpe_query(self) -> MessageBuilder:
        cmd = 'ERHPE?'

        self._msg.append(cmd)
        return self

    def erhpl(self, onoff: bool) -> MessageBuilder:
        cmd = f'ERHPL {int(onoff)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpl_query(self) -> MessageBuilder:
        cmd = 'ERHPL?'

        self._msg.append(cmd)
        return self

    def erhpp(self, path: enums.ERHPP.Path) -> MessageBuilder:
        cmd = f'ERHPP {path}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpp_query(self) -> MessageBuilder:
        cmd = 'ERHPP?'

        self._msg.append(cmd)
        return self

    def erhpqg(self, state: enums.ERHPQG.State) -> MessageBuilder:
        cmd = f'ERHPQG {state}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpqg_query(self) -> MessageBuilder:
        cmd = 'ERHPQG?'

        self._msg.append(cmd)
        return self

    def erhpr(self,
              pin: int,
              state: bool) -> MessageBuilder:
        cmd = f'ERHPR {pin},{int(state)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhpr_query(self,
                    pin: int) -> MessageBuilder:
        cmd = f'ERHPR? {pin}'

        self._msg.append(cmd)
        return self

    def erhps(self, onoff: bool) -> MessageBuilder:
        cmd = f'ERHPS {int(onoff)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhps_query(self) -> MessageBuilder:
        cmd = 'ERHPS?'

        self._msg.append(cmd)
        return self

    def erhvca(self,
               vsmu: enums.SlotNr,
               ismu: enums.SlotNr,
               hvsmu: enums.SlotNr) -> MessageBuilder:
        cmd = f'ERHVCA {vsmu},{ismu},{hvsmu}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhvca_query(self) -> MessageBuilder:
        cmd = 'ERHVCA?'

        self._msg.append(cmd)
        return self

    @final_command
    def erhvctst_query(self) -> MessageBuilder:
        cmd = 'ERHVCTST?'

        self._msg.append(cmd)
        return self

    def erhvp(self, state: enums.ERHVP.State) -> MessageBuilder:
        cmd = f'ERHVP {state}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhvp_query(self) -> MessageBuilder:
        cmd = 'ERHVP?'

        self._msg.append(cmd)
        return self

    def erhvpv(self, state: enums.ERHVPV.State) -> MessageBuilder:
        cmd = f'ERHVPV {state}'

        self._msg.append(cmd)
        return self

    def erhvs(self, enable_series_resistor: bool) -> MessageBuilder:
        cmd = f'ERHVS {int(enable_series_resistor)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erhvs_query(self) -> MessageBuilder:
        cmd = f'ERHVS?'

        self._msg.append(cmd)
        return self

    def erm(self, iport: int) -> MessageBuilder:
        cmd = f'ERM {iport}'

        self._msg.append(cmd)
        return self

    def ermod(self,
              mode: enums.ERMOD.Mode,
              option: bool = None) -> MessageBuilder:
        if option is None:
            cmd = f'ERMOD {mode}'
        else:
            cmd = f'ERMOD {mode},{option}'

        self._msg.append(cmd)
        return self

    @final_command
    def ermod_query(self) -> MessageBuilder:
        cmd = 'ERMOD?'

        self._msg.append(cmd)
        return self

    def erpfda(self,
               hvsmu: enums.SlotNr,
               smu: enums.SlotNr) -> MessageBuilder:
        cmd = f'ERPFDA {hvsmu},{smu}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfda_query(self) -> MessageBuilder:
        cmd = 'ERPFDA?'

        self._msg.append(cmd)
        return self

    def erpfdp(self, state: enums.ERPFDP.State) -> MessageBuilder:
        cmd = f'ERPFDP {state}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfdp_query(self) -> MessageBuilder:
        cmd = 'ERPFDP?'

        self._msg.append(cmd)
        return self

    def erpfds(self, state: bool) -> MessageBuilder:
        cmd = f'ERPFDS {int(state)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfds_query(self) -> MessageBuilder:
        cmd = 'ERPFDS?'

        self._msg.append(cmd)
        return self

    def erpfga(self, gsmu: enums.SlotNr) -> MessageBuilder:
        cmd = f'ERPFGA {gsmu}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfga_query(self) -> MessageBuilder:
        cmd = 'ERPFGA?'

        self._msg.append(cmd)
        return self

    def erpfgp(self, state: enums.ERPFGP.State) -> MessageBuilder:
        cmd = f'ERPFGP {state}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfgp_query(self) -> MessageBuilder:
        cmd = 'ERPFGP?'

        self._msg.append(cmd)
        return self

    def erpfgr(self, state: enums.ERPFGR.State) -> MessageBuilder:
        cmd = f'ERPFGR {state}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfgr_query(self) -> MessageBuilder:
        cmd = 'ERPFDS?'

        self._msg.append(cmd)
        return self

    def erpfqg(self, state: bool) -> MessageBuilder:
        cmd = f'ERPFQG {int(state)}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfqg_query(self) -> MessageBuilder:
        cmd = 'ERPFQG?'

        self._msg.append(cmd)
        return self

    @final_command
    def erpftemp_query(self, chnum: Union[enums.ChNr, int]) -> MessageBuilder:
        cmd = f'ERPFTEMP? {chnum}'

        self._msg.append(cmd)
        return self

    def erpfuhca(self,
                 vsmu: enums.SlotNr,
                 ismu: enums.SlotNr) -> MessageBuilder:
        cmd = f'ERPFUHCA {vsmu},{ismu}'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfuhca_query(self) -> MessageBuilder:
        cmd = 'ERPFUHCA?'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfuhccal_query(self) -> MessageBuilder:
        cmd = 'ERPFUHCCAL?'

        self._msg.append(cmd)
        return self

    @final_command
    def erpfuhcmax_query(self) -> MessageBuilder:
        cmd = 'ERPFUHCMAX?'

        self._msg.append(cmd)
        return self

    def erpfuhctst(self) -> MessageBuilder:
        cmd = 'ERPFUHCTST?'

        self._msg.append(cmd)
        return self

    @final_command
    def err_query(self, mode: enums.ERR.Mode = None) -> MessageBuilder:
        if mode is None:
            cmd = 'ERR?'
        else:
            cmd = f'ERR? {mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def errx_query(self, mode: enums.ERRX.Mode = None) -> MessageBuilder:
        if mode is None:
            cmd = 'ERRX?'
        else:
            cmd = f'ERRX? {mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def ers_query(self) -> MessageBuilder:
        cmd = 'ERS?'

        self._msg.append(cmd)
        return self

    def erssp(self,
              port: enums.ERSSP.Port,
              status: enums.ERSSP.Status) -> MessageBuilder:
        cmd = f'ERSSP {port},{status}'

        self._msg.append(cmd)
        return self

    @final_command
    def erssp_query(self, port: enums.ERSSP.Port) -> MessageBuilder:
        cmd = f'ERSSP? {port}'

        self._msg.append(cmd)
        return self

    def eruhva(self,
               vsmu: enums.SlotNr,
               ismu: enums.SlotNr) -> MessageBuilder:
        cmd = f'ERUHVA {vsmu},{ismu}'

        self._msg.append(cmd)
        return self

    @final_command
    def eruhva_query(self) -> MessageBuilder:
        cmd = 'ERUHVA?'

        self._msg.append(cmd)
        return self

    def fc(self,
           chnum: Union[enums.ChNr, int],
           freq: float) -> MessageBuilder:
        cmd = f'FC {chnum},{freq}'

        self._msg.append(cmd)
        return self

    def fl(self,
           enable_filter: bool,
           channels: List[Union[enums.ChNr, int]] = None) -> MessageBuilder:
        """

        :param enable_filter:
        :param channels:
        :return:
        """
        if channels is None:
            cmd = f'FL {int(enable_filter)}'
        elif len(channels) > 10:
            raise ValueError("A maximum of ten channels can be set.")
        else:
            cmd = f'FL {int(enable_filter)},{as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def fmt(self,
            format_id: enums.FMT.Format,
            mode: enums.FMT.Mode = None) -> MessageBuilder:
        """

        :param format_id:
        :param mode:
        :return:
        """
        if mode is None:
            cmd = f'FMT {format_id}'
        else:
            cmd = f'FMT {format_id},{mode}'

        self._msg.append(cmd)
        return self

    def hvsmuop(self, src_range: enums.HVSMUOP.SourceRange) -> MessageBuilder:
        """

        :param src_range:
        :return:
        """
        cmd = f'HVSMUOP {src_range}'

        self._msg.append(cmd)
        return self

    @final_command
    def hvsmuop_query(self) -> MessageBuilder:
        """
        :return: formatted command string
        """
        cmd = 'HVSMUOP?'

        self._msg.append(cmd)
        return self

    @final_command
    def idn_query(self) -> MessageBuilder:
        cmd = '*IDN?'

        self._msg.append(cmd)
        return self

    def imp(self, mode: enums.IMP.MeasurementMode) -> MessageBuilder:
        """

        :param mode:
        :return:
        """
        cmd = f'IMP {mode}'

        self._msg.append(cmd)
        return self

    def in_(self,
            channels: List[Union[enums.ChNr, int]] = None) -> MessageBuilder:
        """

        :param channels:
        :return:
        """
        if channels is None:
            cmd = f'IN'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'IN {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def intlkvth(self, voltage: float) -> MessageBuilder:
        """

        :param voltage:
        :return:

        >>> MessageBuilder().intlkvth(0).message
        'INTLKVTH 0'
        """
        cmd = f'INTLKVTH {voltage}'

        self._msg.append(cmd)
        return self

    @final_command
    def intlkvth_query(self) -> MessageBuilder:
        cmd = 'INTLKVTH?'

        self._msg.append(cmd)
        return self

    def lgi(self,
            chnum: Union[enums.ChNr, int],
            mode: enums.LinearSearchMode,
            i_range: enums.IMeasRange,
            target: float) -> MessageBuilder:
        """

        :param chnum:
        :param mode:
        :param i_range:
        :param target:
        :return:

        >>> MessageBuilder().lgi(0,1,14,1e-6).message
        'LGI 0,1,14,1e-06'
        """
        cmd = f'LGI {chnum},{mode},{i_range},{target}'

        self._msg.append(cmd)
        return self

    def lgv(self,
            chnum: Union[enums.ChNr, int],
            mode: enums.LinearSearchMode,
            v_range: enums.VMeasRange,
            target: float) -> MessageBuilder:
        """

        :param chnum:
        :param mode:
        :param ranging_type:
        :param target:
        :return:

        >>> MessageBuilder().lgv(1,2,12,3).message
        'LGV 1,2,12,3'
        """
        cmd = f'LGV {chnum},{mode},{v_range},{target}'

        self._msg.append(cmd)
        return self

    def lim(self,
            mode: enums.LIM.Mode,
            limit: float) -> MessageBuilder:
        cmd = f'LIM {mode},{limit}'

        self._msg.append(cmd)
        return self

    @final_command
    def lim_query(self, mode: enums.LIM.Mode) -> MessageBuilder:
        cmd = f'LIM? {mode}'

        self._msg.append(cmd)
        return self

    def lmn(self, enable_data_monitor: bool) -> MessageBuilder:
        cmd = f'LMN {int(enable_data_monitor)}'

        self._msg.append(cmd)
        return self

    @final_command
    def lop_query(self) -> MessageBuilder:
        cmd = 'LOP?'

        self._msg.append(cmd)
        return self

    @final_command
    def lrn_query(self, type_id: enums.LRN.Type) -> MessageBuilder:
        cmd = f'*LRN? {type_id}'

        self._msg.append(cmd)
        return self

    def lsi(self,
            chnum: Union[enums.ChNr, int],
            i_range: enums.IOutputRange,
            start: float,
            stop: float,
            step: float,
            v_comp: float = None) -> MessageBuilder:
        """

        :param chnum:
        :param ranging_type:
        :param start:
        :param stop:
        :param step:
        :param v_comp:
        :return:

        >>> MessageBuilder().lsi(1,0,0,1E-6,1E-8,10).message
        'LSI 1,0,0,1e-06,1e-08,10'
        """
        if v_comp is None:
            cmd = f'LSI {chnum},{i_range},{start},{stop},{step}'
        else:
            cmd = f'LSI {chnum},{i_range},{start},{stop},{step},{v_comp}'

        self._msg.append(cmd)
        return self

    def lsm(self,
            abort: enums.Abort,
            post: enums.LSM.Post = None) -> MessageBuilder:
        if post is None:
            cmd = f'LSM {abort}'
        else:
            cmd = f'LSM {abort},{post}'

        self._msg.append(cmd)
        return self

    def lssi(self,
             chnum: Union[enums.ChNr, int],
             polarity: enums.Polarity,
             offset: float,
             v_comp: float = None) -> MessageBuilder:
        """

        :param chnum:
        :param polarity:
        :param offset:
        :param v_comp:
        :return:

        >>> MessageBuilder().lssi(1,1,1E-6,5).message
        'LSSI 1,1,1e-06,5'
        """
        if v_comp is None:
            cmd = f'LSSI {chnum},{polarity},{offset}'
        else:
            cmd = f'LSSI {chnum},{polarity},{offset},{v_comp}'

        self._msg.append(cmd)
        return self

    def lssv(self,
             chnum: Union[enums.ChNr, int],
             polarity: enums.Polarity,
             offset: float,
             i_comp: float = None) -> MessageBuilder:
        """

        :param chnum:
        :param polarity:
        :param offset:
        :param i_comp:
        :return:

        >>> MessageBuilder().lssv(1,0,5,1E-6).message
        'LSSV 1,0,5,1e-06'
        """
        if i_comp is None:
            cmd = f'LSSV {chnum},{polarity},{offset}'
        else:
            cmd = f'LSSV {chnum},{polarity},{offset},{i_comp}'

        self._msg.append(cmd)
        return self

    @final_command
    def lst_query(self, pnum=None, index=None, size=None) -> MessageBuilder:
        """

        :param pnum:
        :param index:
        :param size:
        :return:

        >>> MessageBuilder().lst_query()
        'LST?'
        >>> MessageBuilder().lst_query(0)
        'LST? 0'
        """
        if pnum is None:
            cmd = 'LST?'
        elif index is None:
            cmd = f'LST? {pnum}'
        elif size is None:
            cmd = f'LST? {pnum},{index}'
        else:
            cmd = f'LST? {pnum},{index},{size}'

        self._msg.append(cmd)
        return self

    def lstm(self,
             hold: float,
             delay: float) -> MessageBuilder:
        cmd = f'LSTM {hold},{delay}'

        self._msg.append(cmd)
        return self

    def lsv(self,
            chnum: Union[enums.ChNr, int],
            v_range: enums.VOutputRange,
            start: float,
            stop: float,
            step: float,
            i_comp: float = None) -> MessageBuilder:
        if i_comp is None:
            cmd = f'LSV {chnum},{v_range},{start},{stop},{step}'
        else:
            cmd = f'LSV {chnum},{v_range},{start},{stop},{step},{i_comp}'

        self._msg.append(cmd)
        return self

    def lsvm(self, mode: enums.LSVM.DataOutputMode) -> MessageBuilder:
        cmd = f'LSVM {mode}'

        self._msg.append(cmd)
        return self

    def mcc(self,
            channels: List[Union[enums.ChNr, int]] = None) -> MessageBuilder:
        """

        :param channels:
        :return:

        >>> MessageBuilder().mcc().message
        'MCC'
        >>> MessageBuilder().mcc([1,2,3]).message
        'MCC 1,2,3'
        """
        if channels is None:
            cmd = f'MCC'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'MCC {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def mcpnt(self,
              chnum: Union[enums.ChNr, int],
              delay: float,
              width: float) -> MessageBuilder:
        cmd = f'MCPNT {chnum},{delay},{width}'

        self._msg.append(cmd)
        return self

    def mcpnx(self,
              n: int,
              chnum: Union[enums.ChNr, int],
              mode: enums.MCPNX.Mode,
              src_range: Union[enums.VOutputRange, enums.IOutputRange],
              base: float,
              pulse: float,
              comp: float = None) -> MessageBuilder:
        """

        :param n:
        :param chnum:
        :param mode:
        :param src_range:
        :param base:
        :param pulse:
        :param comp:
        :return:

        >>> MessageBuilder().mcpnx(1,3,1,0,0,5,1e-01).message
        'MCPNX 1,3,1,0,0,5,0.1'
        """
        if comp is None:
            cmd = f'MCPNX {n},{chnum},{mode},{src_range},{base},{pulse}'
        else:
            cmd = f'MCPNX {n},{chnum},{mode},{src_range},{base},{pulse},' \
                  f'{comp}'

        self._msg.append(cmd)
        return self

    def mcpt(self,
             hold: float,
             period: Union[float, enums.AutoPeriod] = None,
             measurement_delay: float = None,
             average: int = None) -> MessageBuilder:
        if period is None:
            cmd = f'MCPT {hold}'
        elif measurement_delay is None:
            cmd = f'MCPT {hold},{period}'
        elif average is None:
            cmd = f'MCPT {hold},{period},{measurement_delay}'
        else:
            cmd = f'MCPT {hold},{period},{measurement_delay},{average}'

        self._msg.append(cmd)
        return self

    def mcpws(self,
              mode: enums.SweepMode,
              step: int) -> MessageBuilder:
        cmd = f'MCPWS {mode},{step}'

        self._msg.append(cmd)
        return self

    def mcpwnx(self,
               n: int,
               chnum: Union[enums.ChNr, int],
               mode: enums.MCPWNX.Mode,
               src_range: Union[enums.VOutputRange, enums.IOutputRange],
               base: float,
               start: float,
               stop: float,
               comp: float = None,
               p_comp: float = None) -> MessageBuilder:
        """

        :param n:
        :param chnum:
        :param mode:
        :param src_range:
        :param base:
        :param start:
        :param stop:
        :param comp:
        :param p_comp:
        :return:

        >>> MessageBuilder().mcpwnx(2,4,1,0,0,0,5,1E-1).message
        'MCPWNX 2,4,1,0,0,0,5,0.1'
        """
        if comp is None:
            cmd = f'MCPWNX {n},{chnum},{mode},{src_range},{base},{start},' \
                  f'{stop}'
        elif p_comp is None:
            cmd = f'MCPWNX {n},{chnum},{mode},{src_range},{base},{start},' \
                  f'{stop},{comp}'
        else:
            cmd = f'MCPWNX {n},{chnum},{mode},{src_range},{base},{start},' \
                  f'{stop},{comp},{p_comp}'

        self._msg.append(cmd)
        return self

    def mdcv(self,
             chnum: Union[enums.ChNr, int],
             base: float,
             bias: float,
             post: float = None) -> MessageBuilder:
        if post is None:
            cmd = f'MDCV {chnum},{base},{bias}'
        else:
            cmd = f'MDCV {chnum},{base},{bias},{post}'

        self._msg.append(cmd)
        return self

    def mi(self,
           chnum: Union[enums.ChNr, int],
           i_range: enums.IOutputRange,
           base: float,
           bias: float,
           v_comp: float = None) -> MessageBuilder:
        if v_comp is None:
            cmd = f'MI {chnum},{i_range},{base},{bias},{v_comp}'
        else:
            cmd = f'MI {chnum},{i_range},{base},{bias},{v_comp}'

        self._msg.append(cmd)
        return self

    def ml(self, mode: enums.ML.Mode) -> MessageBuilder:
        cmd = f'ML {mode}'

        self._msg.append(cmd)
        return self

    def mm(self,
           mode: enums.MM.Mode,
           channels: List[Union[enums.ChNr, int]] = None) -> MessageBuilder:
        """

        :param mode:
        :param channels:
        :return:

        >>> MessageBuilder().mm(2, [1,3]).message
        'MM 2,1,3'

        >>> MessageBuilder().mm(2, [1,2,3,4,5,6,7,8,9,10,11]).message
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError:

        >>> MessageBuilder().mm(3, [1]).message
        'MM 3,1'

        >>> MessageBuilder().mm(3, [1,3]).message
        Traceback (most recent call last):
            ...
        ValueError: Specify 1 (and only 1) channel.

        >>> MessageBuilder().mm(9, [1]).message
        'MM 9,1'

        >>> MessageBuilder().mm(9, [1,2]).message
        Traceback (most recent call last):
            ...
        ValueError: Specify not more than 1 channel for this mode

        >>> MessageBuilder().mm(14).message
        'MM 14'

        >>> MessageBuilder().mm(14, [1]).message
        Traceback (most recent call last):
            ...
        ValueError: Do not specify channels for this mode
        """
        if mode in (1, 2, 10, 16, 18, 27, 28):
            if len(channels) > 10:
                raise ValueError('A maximum of ten channels can be set. For '
                                 'mode=18, the first chnum must be MFCMU.')
            cmd = f'MM {mode},{as_csv(channels)}'
        elif mode in (3, 4, 5, 17, 19, 20, 22, 23, 26):
            if len(channels) != 1:
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
            abort: enums.Abort,
            post: enums.MSC.Post = None) -> MessageBuilder:
        if post is None:
            cmd = f'MSC {abort}'
        else:
            cmd = f'MSC {abort},{post}'

        self._msg.append(cmd)
        return self

    def msp(self,
            chnum: Union[enums.ChNr, int],
            post: float = None,
            base: float = None) -> MessageBuilder:
        if post is None:
            cmd = f'MSP {chnum}'
        elif base is None:
            cmd = f'MSP {chnum},{post}'
        else:
            cmd = f'MSP {chnum},{post},{base}'

        self._msg.append(cmd)
        return self

    def mt(self,
           h_bias: float,
           interval: float,
           number: int,
           h_base: float = None) -> MessageBuilder:
        """

        :param h_bias:
        :param interval:
        :param number:
        :param h_base:
        :return:
        >>> MessageBuilder().mt(0.01,0.001,101,0.1).message
        'MT 0.01,0.001,101,0.1'
        """
        if h_base is None:
            cmd = f'MT {h_bias},{interval},{number}'
        else:
            cmd = f'MT {h_bias},{interval},{number},{h_base}'

        self._msg.append(cmd)
        return self

    def mtdcv(self,
              h_bias: float,
              interval: float,
              number: int,
              h_base: float = None) -> MessageBuilder:
        """

        :param h_bias:
        :param interval:
        :param number:
        :param h_base:
        :return:

        >>> MessageBuilder().mtdcv(0.01,0.008,101,0.1).message
        'MTDCV 0.01,0.008,101,0.1'

        >>> MessageBuilder().mtdcv(0,0.008,5000,0).message
        'MTDCV 0,0.008,5000,0'
        """
        if h_base is None:
            cmd = f'MTDCV {h_bias},{interval},{number}'
        else:
            cmd = f'MTDCV {h_bias},{interval},{number},{h_base}'

        self._msg.append(cmd)
        return self

    def mv(self,
           chnum: Union[enums.ChNr, int],
           v_range: enums.VOutputRange,
           base: float,
           bias: float,
           i_comp: float = None) -> MessageBuilder:
        if i_comp is None:
            cmd = f'MV {chnum},{v_range},{base},{bias}'
        else:
            cmd = f'MV {chnum},{v_range},{base},{bias},{i_comp}'

        self._msg.append(cmd)
        return self

    @final_command
    def nub_query(self) -> MessageBuilder:
        cmd = 'NUB?'

        self._msg.append(cmd)
        return self

    def odsw(self,
             chnum: Union[enums.ChNr, int],
             enable_pulse_switch: bool,
             switch_normal_state: enums.ODSW.SwitchNormalState = None,
             delay: float = None,
             width: float = None) -> MessageBuilder:
        """

        :param chnum:
        :param enable_pulse_switch:
        :param switch_normal_state:
        :param delay:
        :param width:
        :return:

        >>> MessageBuilder().odsw(101,1,1,1E-6,2E-6).message
        'ODSW 101,1,1,1e-06,2e-06'

        >>> MessageBuilder().odsw(101,1,1).message
        'ODSW 101,1,1'

        >>> MessageBuilder().odsw(101,1).message
        'ODSW 101,1'

        >>> MessageBuilder().odsw(101,1,1,delay=1e-6).message
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError:

        >>> MessageBuilder().odsw(101,1,1,width=1e-6).message
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError:

        >>> MessageBuilder().odsw(101,1,width=1e-6).message
        'ODSW 101,1'
        """
        if switch_normal_state is None:
            cmd = f'ODSW {chnum},{int(enable_pulse_switch)}'
        elif delay is None and width is None:
            cmd = f'ODSW {chnum},{int(enable_pulse_switch)},' \
                  f'{switch_normal_state}'
        elif delay is None or width is None:
            raise ValueError('When specifying delay, then width must be '
                             'specified (and vice versa)')
        else:
            cmd = f'ODSW {chnum},{int(enable_pulse_switch)},' \
                  f'{switch_normal_state},{delay},{width}'

        self._msg.append(cmd)
        return self

    @final_command
    def odsw_query(self, chnum: Union[enums.ChNr, int]) -> MessageBuilder:
        cmd = f'ODSW? {chnum}'

        self._msg.append(cmd)
        return self

    @final_command
    def opc_query(self) -> MessageBuilder:
        cmd = '*OPC?'

        self._msg.append(cmd)
        return self

    def os(self) -> MessageBuilder:
        cmd = 'OS'

        self._msg.append(cmd)
        return self

    def osx(self,
            port: enums.TriggerPort,
            level: enums.OSX.Level = None) -> MessageBuilder:
        if level is None:
            cmd = f'OSX {port}'
        else:
            cmd = f'OSX {port},{level}'

        self._msg.append(cmd)
        return self

    def pa(self, wait_time=None) -> MessageBuilder:
        if wait_time is None:
            cmd = 'PA'
        else:
            cmd = f'PA {wait_time}'

        self._msg.append(cmd)
        return self

    def pad(self, enable: bool) -> MessageBuilder:
        cmd = f'PAD {int(enable)}'

        self._msg.append(cmd)
        return self

    def pax(self,
            port: enums.TriggerPort,
            wait_time: float = None) -> MessageBuilder:
        if wait_time is None:
            cmd = f'PAX {port}'
        else:
            cmd = f'PAX {port},{wait_time}'

        self._msg.append(cmd)
        return self

    def pch(self,
            master: Union[enums.ChNr, int],
            slave: Union[enums.ChNr, int]) -> MessageBuilder:
        cmd = f'PCH {master},{slave}'

        self._msg.append(cmd)
        return self

    @final_command
    def pch_query(self, master=None) -> MessageBuilder:
        if master is None:
            cmd = 'PCH'
        else:
            cmd = f'PCH {master}'

        self._msg.append(cmd)
        return self

    def pdcv(self,
             chnum: Union[enums.ChNr, int],
             base: float,
             pulse: float) -> MessageBuilder:
        cmd = f'PDCV {chnum},{base},{pulse}'

        self._msg.append(cmd)
        return self

    def pi(self,
           chnum: Union[enums.ChNr, int],
           i_range: enums.IOutputRange,
           base: float,
           pulse: float,
           v_comp: float = None) -> MessageBuilder:
        if v_comp is None:
            cmd = f'PI {chnum},{i_range},{base},{pulse}'
        else:
            cmd = f'PI {chnum},{i_range},{base},{pulse},{v_comp}'

        self._msg.append(cmd)
        return self

    def pt(self,
           hold: float,
           width: float,
           period: Union[float, enums.AutoPeriod] = None,
           t_delay: float = None) -> MessageBuilder:
        if period is None:
            cmd = f'PT {hold},{width}'
        elif t_delay is None:
            cmd = f'PT {hold},{width},{period}'
        else:
            cmd = f'PT {hold},{width},{period},{t_delay}'

        self._msg.append(cmd)
        return self

    def ptdcv(self,
              hold: float,
              width: float,
              period: float = None,
              t_delay: float = None) -> MessageBuilder:
        if period is None:
            cmd = f'PTDCV {hold},{width}'
        elif t_delay is None:
            cmd = f'PTDCV {hold},{width},{period}'
        else:
            cmd = f'PTDCV {hold},{width},{period},{t_delay}'

        self._msg.append(cmd)
        return self

    def pv(self,
           chnum: Union[enums.ChNr, int],
           v_range: enums.VOutputRange,
           base: float,
           pulse: float,
           i_comp: float = None) -> MessageBuilder:
        if i_comp is None:
            cmd = f'PV {chnum},{v_range},{base},{pulse}'
        else:
            cmd = f'PV {chnum},{v_range},{base},{pulse},{i_comp}'

        self._msg.append(cmd)
        return self

    def pwdcv(self,
              chnum: Union[enums.ChNr, int],
              mode: enums.LinearSweepMode,
              base: float,
              start: float,
              stop: float,
              step: float) -> MessageBuilder:
        cmd = f'PWDCV {chnum},{mode},{base},{start},{stop},{step}'

        self._msg.append(cmd)
        return self

    def pwi(self,
            chnum: Union[enums.ChNr, int],
            mode: enums.SweepMode,
            i_range: enums.IOutputRange,
            base: float,
            start: float,
            stop: float,
            step: float,
            v_comp: float = None,
            p_comp: float = None) -> MessageBuilder:
        if v_comp is None:
            cmd = f'PWI {chnum},{mode},{i_range},{base},{start},{stop},' \
                  f'{step}'
        elif p_comp is None:
            cmd = f'PWI {chnum},{mode},{i_range},{base},{start},{stop},' \
                  f'{step},{v_comp}'
        else:
            cmd = f'PWI {chnum},{mode},{i_range},{base},{start},{stop},' \
                  f'{step},{v_comp},{p_comp}'

        self._msg.append(cmd)
        return self

    def pwv(self,
            chnum: Union[enums.ChNr, int],
            mode: enums.SweepMode,
            v_range: enums.VOutputRange,
            base: float,
            start: float,
            stop: float,
            step: float,
            i_comp: float = None,
            p_comp: float = None) -> MessageBuilder:
        if i_comp is None:
            cmd = f'PWI {chnum},{mode},{v_range},{base},{start},{stop},' \
                  f'{step}'
        elif p_comp is None:
            cmd = f'PWI {chnum},{mode},{v_range},{base},{start},{stop},' \
                  f'{step},{i_comp}'
        else:
            cmd = f'PWI {chnum},{mode},{v_range},{base},{start},{stop},' \
                  f'{step},{i_comp},{p_comp}'

        self._msg.append(cmd)
        return self

    def qsc(self, mode: enums.APIVersion) -> MessageBuilder:
        cmd = f'QSC {mode}'

        self._msg.append(cmd)
        return self

    def qsl(self,
            enable_data_output: bool,
            enable_leakage_current_compensation: bool) -> MessageBuilder:
        cmd = f'QSL {int(enable_data_output)},' \
              f'{int(enable_leakage_current_compensation)}'

        self._msg.append(cmd)
        return self

    def qsm(self,
            abort: enums.Abort,
            post: enums.QSM.Post = None) -> MessageBuilder:
        if post is None:
            cmd = f'QSM {abort}'
        else:
            cmd = f'QSM {abort},{post}'

        self._msg.append(cmd)
        return self

    def qso(self,
            enable_smart_operation: bool,
            chnum: Union[enums.ChNr, int] = None,
            v_comp: float = None) -> MessageBuilder:
        if chnum is None:
            cmd = f'QSO {int(enable_smart_operation)}'
        elif v_comp is None:
            cmd = f'QSO {int(enable_smart_operation)},{chnum}'
        else:
            cmd = f'QSO {int(enable_smart_operation)},{chnum},{v_comp}'

        self._msg.append(cmd)
        return self

    def qsr(self, i_range: enums.IMeasRange) -> MessageBuilder:
        cmd = f'QSR {i_range}'

        self._msg.append(cmd)
        return self

    def qst(self,
            cinteg: float,
            linteg: float,
            hold: float,
            delay1: float,
            delay2: float = None) -> MessageBuilder:
        if delay2 is None:
            cmd = f'QST {cinteg},{linteg},{hold},{delay1}'
        else:
            cmd = f'QST {cinteg},{linteg},{hold},{delay1},{delay2}'

        self._msg.append(cmd)
        return self

    def qsv(self,
            chnum: Union[enums.ChNr, int],
            mode: enums.LinearSweepMode,
            v_range: enums.VOutputRange,
            start: float,
            stop: float,
            cvoltage: float,
            step: float,
            i_comp: float = None) -> MessageBuilder:
        if i_comp is None:
            cmd = f'QSV {chnum},{mode},{v_range},{start},{stop},{cvoltage},' \
                  f'{step}'
        else:
            cmd = f'QSV {chnum},{mode},{v_range},{start},{stop},{cvoltage},' \
                  f'{step},{i_comp}'

        self._msg.append(cmd)
        return self

    def qsz(self,
            mode: enums.QSZ.Mode) -> MessageBuilder:
        cmd = f'QSZ {mode}'

        self._msg.append(cmd)

        if mode == enums.QSZ.Mode.PERFORM_MEASUREMENT:
            # Result must be queried if measurement is performed
            self._msg.set_final()

        return self

    def rc(self,
           chnum: Union[enums.ChNr, int],
           ranging_mode: enums.RangingMode,
           measurement_range: int = None) -> MessageBuilder:
        if measurement_range is None:
            if ranging_mode != 0:
                raise ValueError('measurement_range must be specified for '
                                 'ranging_mode!=0')
            cmd = f'RC {chnum},{ranging_mode}'
        else:
            cmd = f'RC {chnum},{ranging_mode},{measurement_range}'

        self._msg.append(cmd)
        return self

    def rcv(self, slot: enums.SlotNr = None) -> MessageBuilder:
        cmd = 'RCV' if slot is None else f'RCV {slot}'

        self._msg.append(cmd)
        return self

    def ri(self,
           chnum: Union[enums.ChNr, int],
           i_range: enums.IMeasRange) -> MessageBuilder:
        cmd = f'RI {chnum},{i_range}'

        self._msg.append(cmd)
        return self

    def rm(self,
           chnum: Union[enums.ChNr, int],
           mode: enums.RM.Mode,
           rate: int = None) -> MessageBuilder:
        if rate is None:
            cmd = f'RM {chnum},{mode}'
        else:
            if mode == 1:
                raise ValueError('Do not specify rate for mode 1')
            cmd = f'RM {chnum},{mode},{rate}'

        self._msg.append(cmd)
        return self

    def rst(self) -> MessageBuilder:
        cmd = '*RST'

        self._msg.append(cmd)
        return self

    def ru(self,
           start: int,
           stop: int) -> MessageBuilder:
        cmd = f'RU {start},{stop}'

        self._msg.append(cmd)
        return self

    def rv(self,
           chnum: Union[enums.ChNr, int],
           v_range: enums.VMeasRange) -> MessageBuilder:
        cmd = f'RV {chnum},{v_range}'

        self._msg.append(cmd)
        return self

    def rz(self,
           channels: List[Union[enums.ChNr, int]] = None) -> MessageBuilder:
        """

        :param channels:
        :return:

        >>> MessageBuilder().rz().message
        'RZ'

        >>> MessageBuilder().rz([1,2,3]).message
        'RZ 1,2,3'
        """
        if channels is None:
            cmd = 'RZ'
        elif len(channels) > 15:
            raise ValueError("A maximum of 15 channels can be set.")
        else:
            cmd = f'RZ {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def sal(self,
            chnum: Union[enums.ChNr, int],
            enable_status_led: bool) -> MessageBuilder:
        cmd = f'SAL {chnum},{int(enable_status_led)}'

        self._msg.append(cmd)
        return self

    def sap(self,
            chnum: Union[enums.ChNr, int],
            path: enums.SAP.Path) -> MessageBuilder:
        cmd = f'SAP {chnum},{path}'

        self._msg.append(cmd)
        return self

    def sar(self,
            chnum: Union[enums.ChNr, int],
            enable_picoamp_autoranging: bool) -> MessageBuilder:
        # For reasons only known to the designer of the KeysightB1500's API the logic
        # of enabled=1 and disabled=0 is inverted JUST for this command. 🤬
        cmd = f'SAR {chnum},{int(not enable_picoamp_autoranging)}'

        self._msg.append(cmd)
        return self

    def scr(self,
            pnum: int = None) -> MessageBuilder:
        cmd = 'SCR' if pnum is None else f'SCR {pnum}'

        self._msg.append(cmd)
        return self

    def ser(self,
            chnum: Union[enums.ChNr, int],
            load_z: float) -> MessageBuilder:
        cmd = f'SER {chnum},{load_z}'

        self._msg.append(cmd)
        return self

    @final_command
    def ser_query(self,
                  chnum: Union[enums.ChNr, int]) -> MessageBuilder:
        cmd = f'SER? {chnum}'

        self._msg.append(cmd)
        return self

    def sim(self,
            mode: enums.SIM.Mode) -> MessageBuilder:
        cmd = f'SIM {mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def sim_query(self) -> MessageBuilder:
        cmd = 'SIM?'

        self._msg.append(cmd)
        return self

    def sopc(self,
             chnum: Union[enums.ChNr, int],
             power: float) -> MessageBuilder:
        cmd = f'SOPC {chnum},{power}'

        self._msg.append(cmd)
        return self

    @final_command
    def sopc_query(self,
                   chnum: Union[enums.ChNr, int]) -> MessageBuilder:
        cmd = f'SOPC? {chnum}'

        self._msg.append(cmd)
        return self

    def sovc(self,
             chnum: Union[enums.ChNr, int],
             voltage: float) -> MessageBuilder:
        cmd = f'SOVC {chnum},{voltage}'

        self._msg.append(cmd)
        return self

    @final_command
    def sovc_query(self,
                   chnum: Union[enums.ChNr, int]) -> MessageBuilder:
        cmd = f'SOVC? {chnum}'

        self._msg.append(cmd)
        return self

    def spm(self,
            chnum: Union[enums.ChNr, int],
            mode: enums.SPM.Mode) -> MessageBuilder:
        cmd = f'SPM {chnum},{mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def spm_query(self, chnum: Union[enums.ChNr, int]) -> MessageBuilder:
        cmd = f'SPM? {chnum}'

        self._msg.append(cmd)
        return self

    def spp(self) -> MessageBuilder:
        cmd = 'SPP'

        self._msg.append(cmd)
        return self

    def spper(self,
              period: float) -> MessageBuilder:
        cmd = f'SPPER {period}'

        self._msg.append(cmd)
        return self

    @final_command
    def spper_query(self) -> MessageBuilder:
        cmd = 'SPPER?'

        self._msg.append(cmd)
        return self

    def sprm(self,
             mode: enums.SPRM.Mode,
             condition=None) -> MessageBuilder:
        if condition is None:
            cmd = f'SPRM {mode}'
        else:
            cmd = f'SPRM {mode},{condition}'

        self._msg.append(cmd)
        return self

    @final_command
    def sprm_query(self) -> MessageBuilder:
        cmd = 'SPRM?'

        self._msg.append(cmd)
        return self

    @final_command
    def spst_query(self) -> MessageBuilder:
        cmd = 'SPST?'

        self._msg.append(cmd)
        return self

    def spt(self,
            chnum: Union[enums.ChNr, int],
            src: enums.SPT.Src,
            delay: float,
            width: float,
            leading: float,
            trailing: float = None) -> MessageBuilder:
        if trailing is None:
            cmd = f'SPT {chnum},{src},{delay},{width},{leading}'
        else:
            cmd = f'SPT {chnum},{src},{delay},{width},{leading},{trailing}'

        self._msg.append(cmd)
        return self

    @final_command
    def spt_query(self,
                  chnum: Union[enums.ChNr, int],
                  src: enums.SPT.Src) -> MessageBuilder:
        cmd = f'SPT? {chnum},{src}'

        self._msg.append(cmd)
        return self

    def spupd(self,
              channels: List[Union[enums.ChNr, int]] = None) -> MessageBuilder:
        """

        :param channels:
        :return:
        >>> MessageBuilder().spupd([101,102,201,202]).message
        'SPUPD 101,102,201,202'
        """
        if channels is None:
            cmd = 'SPUPD'
        elif len(channels) > 10:
            raise ValueError("A maximum of ten channels can be set.")
        else:
            cmd = f'SPUPD {as_csv(channels)}'

        self._msg.append(cmd)
        return self

    def spv(self,
            chnum: Union[enums.ChNr, int],
            src: enums.SPV.Src,
            base: float,
            peak: float = None) -> MessageBuilder:
        if peak is None:
            cmd = f'SPV {chnum},{src},{base}'
        else:
            cmd = f'SPV {chnum},{src},{base},{peak}'

        self._msg.append(cmd)
        return self

    @final_command
    def spv_query(self,
                  chnum: Union[enums.ChNr, int],
                  src: enums.SPV.Src) -> MessageBuilder:
        cmd = f'SPV? {chnum},{src}'

        self._msg.append(cmd)
        return self

    def sre(self, flags: enums.SRE) -> MessageBuilder:
        cmd = f'*SRE {flags}'

        self._msg.append(cmd)
        return self

    @final_command
    def sre_query(self) -> MessageBuilder:
        cmd = '*SRE?'

        self._msg.append(cmd)
        return self

    def srp(self) -> MessageBuilder:
        cmd = 'SRP'

        self._msg.append(cmd)
        return self

    def ssl(self,
            chnum: Union[enums.ChNr, int],
            enable_indicator_led: bool) -> MessageBuilder:
        cmd = f'SSL {chnum},{int(enable_indicator_led)}'

        self._msg.append(cmd)
        return self

    def ssp(self,
            chnum: Union[enums.ChNr, int],
            path: enums.SSP.Path) -> MessageBuilder:
        cmd = f'SSP {chnum},{path}'

        self._msg.append(cmd)
        return self

    def ssr(self,
            chnum: Union[enums.ChNr, int],
            enable_series_resistor: bool) -> MessageBuilder:
        cmd = f'SSR {chnum},{int(enable_series_resistor)}'

        self._msg.append(cmd)
        return self

    def st(self, pnum: int) -> MessageBuilder:
        cmd = f'ST {pnum}'

        self._msg.append(cmd)
        return self

    @final_command
    def stb_query(self) -> MessageBuilder:
        cmd = '*STB?'

        self._msg.append(cmd)
        return self

    def stgp(self,
             chnum: Union[enums.ChNr, int],
             trigger_timing: enums.STGP.TriggerTiming) -> MessageBuilder:
        cmd = f'STGP {chnum},{trigger_timing}'

        self._msg.append(cmd)
        return self

    @final_command
    def stgp_query(self, chnum: Union[enums.ChNr, int]) -> MessageBuilder:
        cmd = f'STGP? {chnum}'

        self._msg.append(cmd)
        return self

    def tacv(self,
             chnum: Union[enums.ChNr, int],
             voltage: float) -> MessageBuilder:
        cmd = f'TACV {chnum},{voltage}'

        self._msg.append(cmd)
        return self

    def tc(self,
           chnum: Union[enums.ChNr, int],
           mode: enums.RangingMode,
           ranging_type=None) -> MessageBuilder:
        if ranging_type is None:
            cmd = f'TC {chnum},{mode}'
        else:
            cmd = f'TC {chnum},{mode},{ranging_type}'

        self._msg.append(cmd)
        return self

    def tdcv(self,
             chnum: Union[enums.ChNr, int],
             voltage: float) -> MessageBuilder:
        cmd = f'TDCV {chnum},{voltage}'

        self._msg.append(cmd)
        return self

    def tdi(self,
            chnum: Union[enums.ChNr, int],
            i_range: enums.IOutputRange,
            current: float,
            v_comp: float = None,
            comp_polarity: enums.CompliancePolarityMode = None,
            v_range: enums.VOutputRange = None) -> MessageBuilder:
        """

        :param chnum:
        :param i_range:
        :param current:
        :param v_comp:
        :param comp_polarity:
        :param v_range:
        :return:

        >>> MessageBuilder().tdi(1,0,1E-6).message
        'TDI 1,0,1e-06'
        """
        if v_comp is None:
            cmd = f'TDI {chnum},{i_range},{current}'
        elif comp_polarity is None:
            cmd = f'TDI {chnum},{i_range},{current},{v_comp}'
        elif v_range is None:
            cmd = f'TDI {chnum},{i_range},{current},{v_comp},{comp_polarity}'
        else:
            cmd = f'TDI {chnum},{i_range},{current},{v_comp},{comp_polarity},' \
                  f'{v_range}'

        self._msg.append(cmd)
        return self

    def tdv(self,
            chnum: Union[enums.ChNr, int],
            v_range: enums.VOutputRange,
            voltage: float,
            i_comp: float = None,
            comp_polarity: enums.CompliancePolarityMode = None,
            irange: enums.IOutputRange = None) -> MessageBuilder:
        """

        :param chnum:
        :param vrange:
        :param voltage:
        :param i_comp:
        :param comp_polarity:
        :param irange:
        :return:

        >>> MessageBuilder().tdv(1,0,20,1E-6,0,15).message
        'TDV 1,0,20,1e-06,0,15'
        """
        if i_comp is None:
            cmd = f'TDV {chnum},{v_range},{voltage}'
        elif comp_polarity is None:
            cmd = f'TDV {chnum},{v_range},{voltage},{i_comp}'
        elif irange is None:
            cmd = f'TDV {chnum},{v_range},{voltage},{i_comp},{comp_polarity}'
        else:
            cmd = f'TDV {chnum},{v_range},{voltage},{i_comp},{comp_polarity},{irange}'

        self._msg.append(cmd)
        return self

    def tgmo(self, mode: enums.TGMO.Mode) -> MessageBuilder:
        cmd = f'TGMO {mode}'

        self._msg.append(cmd)
        return self

    def tgp(self,
            port: enums.TriggerPort,
            terminal: enums.TGP.TerminalType,
            polarity: enums.TGP.Polarity,
            trigger_type: enums.TGP.TriggerType = None) -> MessageBuilder:
        if trigger_type is None:
            cmd = f'TGP {port},{terminal},{polarity}'
        else:
            cmd = f'TGP {port},{terminal},{polarity},{trigger_type}'

        self._msg.append(cmd)
        return self

    def tgpc(self, ports: List[enums.TriggerPort] = None) -> MessageBuilder:
        """

        :param ports:
        :return:

        >>> MessageBuilder().tgpc([-1,-2,1,2]).message
        'TGPC -1,-2,1,2'
        """
        if ports is None:
            cmd = 'TGPC'
        elif len(ports) > 18:
            raise ValueError("A maximum of 18 ports can be set.")
        else:
            cmd = f'TGPC {as_csv(ports)}'

        self._msg.append(cmd)
        return self

    def tgsi(self, mode: enums.TGSI.Mode) -> MessageBuilder:
        cmd = f'TGSI {mode}'

        self._msg.append(cmd)
        return self

    def tgso(self, mode: enums.TGSO.Mode) -> MessageBuilder:
        cmd = f'TGSO {mode}'

        self._msg.append(cmd)
        return self

    def tgxo(self, mode: enums.TGXO.Mode) -> MessageBuilder:
        cmd = f'TGXO {mode}'

        self._msg.append(cmd)
        return self

    def ti(self,
           chnum: Union[enums.ChNr, int],
           i_range: enums.IMeasRange = None) -> MessageBuilder:
        if i_range is None:
            cmd = f'TI {chnum}'
        else:
            cmd = f'TI {chnum},{i_range}'

        self._msg.append(cmd)
        return self

    def tiv(self,
            chnum: Union[enums.ChNr, int],
            i_range: enums.IMeasRange = None,
            v_range: enums.VMeasRange = None) -> MessageBuilder:
        """

        :param chnum:
        :param irange:
        :param vrange:
        :return:

        >>> MessageBuilder().tiv(1).message
        'TIV 1'

        >>> MessageBuilder().tiv(1, 2, 3).message
        'TIV 1,2,3'

        >>> MessageBuilder().tiv(1, i_range=2).message
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError:

        >>> MessageBuilder().tiv(1, v_range=2).message
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError:
        """
        if i_range is None and v_range is None:
            cmd = f'TIV {chnum}'
        elif i_range is None or v_range is None:
            raise ValueError('When i_range is specified, then v_range must be '
                             'specified (and vice versa).')
        else:
            cmd = f'TIV {chnum},{i_range},{v_range}'

        self._msg.append(cmd)
        return self

    def tm(self, mode: enums.TM.Mode) -> MessageBuilder:
        cmd = f'TM {mode}'

        self._msg.append(cmd)
        return self

    @final_command
    def tmacv(self,
              chnum: Union[enums.ChNr, int],
              mode: enums.RangingMode,
              meas_range: enums.TMACV.Range = None) -> MessageBuilder:
        """
        This command monitors the MFCMU AC voltage output signal level,
        and returns the measurement data.

        :param chnum: MFCMU channel number. Integer expression. 1 to 10 or
            101 to 1001. See Table 4-1 on page 16.

        :param mode: Ranging mode. Integer expression. 0 or 2.
            0: Auto ranging. Initial setting.
            2: Fixed range.

        :param meas_range: Measurement Range. This parameter must be set if
            mode=2. Set Table 4-19 on Page 30

        :return:
        """
        if meas_range is None:
            cmd = f'TMACV {chnum},{mode}'
        else:
            cmd = f'TMACV {chnum},{mode},{meas_range}'

        self._msg.append(cmd)
        return self

    def tmdcv(self,
              chnum: Union[enums.ChNr, int],
              mode: enums.RangingMode,
              meas_range: enums.TMDCV.Range = None) -> MessageBuilder:
        if meas_range is None:
            cmd = f'TMDCV {chnum},{mode}'
        else:
            cmd = f'TMDCV {chnum},{mode},{meas_range}'

        self._msg.append(cmd)
        return self

    def tsc(self, enable_timestamp: bool) -> MessageBuilder:
        cmd = f'TSC {int(enable_timestamp)}'

        self._msg.append(cmd)
        return self

    def tsq(self) -> MessageBuilder:
        """
        The TSQ command returns the time data from when the TSR command is
        sent until this command is sent. The time data will be put in the
        data output buffer as same as the measurement data.
        This command is effective for all measurement modes, regardless of
        the TSC setting.
        This command is not effective for the 4 byte binary data output
        format (FMT3 and FMT4).

        (Note by Stefan: Although this command places time data in the output
        buffer it (apparently?) does not have to be a final command (=> other
        commands may follow.

        :return:
        """
        cmd = 'TSQ'

        self._msg.append(cmd)
        return self

    def tsr(self, chnum=None) -> MessageBuilder:
        cmd = f'TSR' if chnum is None else f'TSR {chnum}'

        self._msg.append(cmd)
        return self

    def tst(self,
            slot: enums.SlotNr = None,
            option: enums.TST.Option = None) -> MessageBuilder:
        if slot is None:
            cmd = '*TST?'
        elif option is None:
            cmd = f'*TST? {slot}'
        else:
            cmd = f'*TST? {slot},{option}'

        self._msg.append(cmd)
        return self

    def ttc(self,
            chnum: Union[enums.ChNr, int],
            mode: enums.RangingMode,
            meas_range: enums.TTC.Range = None) -> MessageBuilder:
        if meas_range is None:
            cmd = f'TTC {chnum},{mode}'
        else:
            cmd = f'TTC {chnum},{mode},{meas_range}'

        self._msg.append(cmd)
        return self

    def tti(self,
            chnum: Union[enums.ChNr, int],
            ranging_type: enums.IMeasRange = None) -> MessageBuilder:
        if ranging_type is None:
            cmd = f'TTI {chnum}'
        else:
            cmd = f'TTI {chnum},{ranging_type}'

        self._msg.append(cmd)
        return self

    def ttiv(self,
             chnum: Union[enums.ChNr, int],
             i_range: enums.IMeasRange = None,
             v_range: enums.VMeasRange = None) -> MessageBuilder:
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
            chnum: Union[enums.ChNr, int],
            v_range: enums.VMeasRange = None) -> MessageBuilder:
        if v_range is None:
            cmd = f'TTV {chnum}'
        else:
            cmd = f'TTV {chnum},{v_range}'

        self._msg.append(cmd)
        return self

    def tv(self,
           chnum: Union[enums.ChNr, int],
           v_range: enums.VMeasRange = None) -> MessageBuilder:
        if v_range is None:
            cmd = f'TV {chnum}'
        else:
            cmd = f'TV {chnum},{v_range}'

        self._msg.append(cmd)
        return self

    @final_command
    def unt_query(self, mode: enums.UNT.Mode = None) -> MessageBuilder:
        cmd = 'UNT?' if mode is None else f'UNT? {mode}'

        self._msg.append(cmd)
        return self

    def var(self,
            variable_type: enums.VAR.Type,
            n: int,
            value: Union[int, float]) -> MessageBuilder:
        cmd = f'VAR {variable_type},{n},{value}'

        self._msg.append(cmd)
        return self

    @final_command
    def var_query(self,
                  variable_type: enums.VAR.Type,
                  n: int) -> MessageBuilder:
        cmd = f'VAR? {variable_type},{n}'

        self._msg.append(cmd)
        return self

    def wacv(self,
             chnum: Union[enums.ChNr, int],
             mode: enums.SweepMode,
             start: float,
             stop: float,
             step: float) -> MessageBuilder:
        cmd = f'WACV {chnum},{mode},{start},{stop},{step}'

        self._msg.append(cmd)
        return self

    def wat(self,
            wait_time_type: enums.WAT.Type,
            coeff: float,
            offset=None) -> MessageBuilder:
        if offset is None:
            cmd = f'WAT {wait_time_type},{coeff}'
        else:
            cmd = f'WAT {wait_time_type},{coeff},{offset}'

        self._msg.append(cmd)
        return self

    def wdcv(self,
             chnum: Union[enums.ChNr, int],
             mode: enums.SweepMode,
             start: float,
             stop: float,
             step: float,
             i_comp: float = None) -> MessageBuilder:
        if i_comp is None:
            cmd = f'WDCV {chnum},{mode},{start},{stop},{step}'
        else:
            cmd = f'WDCV {chnum},{mode},{start},{stop},{step},{i_comp}'

        self._msg.append(cmd)
        return self

    def wfc(self,
            chnum: Union[enums.ChNr, int],
            mode: enums.SweepMode,
            start: float,
            stop: float,
            step: float) -> MessageBuilder:
        cmd = f'WFC {chnum},{mode},{start},{stop},{step}'

        self._msg.append(cmd)
        return self

    def wi(self,
           chnum: Union[enums.ChNr, int],
           mode: enums.SweepMode,
           i_range: enums.IOutputRange,
           start: float,
           stop: float,
           step: float,
           v_comp: float = None,
           p_comp: float = None) -> MessageBuilder:
        if v_comp is None:
            cmd = f'WI {chnum},{mode},{i_range},{start},{stop},{step}'
        elif p_comp is None:
            cmd = f'WI {chnum},{mode},{i_range},{start},{stop},{step},' \
                  f'{v_comp}'
        else:
            cmd = f'WI {chnum},{mode},{i_range},{start},{stop},{step},' \
                  f'{v_comp},{p_comp}'

        self._msg.append(cmd)
        return self

    def wm(self,
           abort: enums.Abort,
           post: enums.WM.Post = None) -> MessageBuilder:
        if post is None:
            cmd = f'WM {abort}'
        else:
            cmd = f'WM {abort},{post}'

        self._msg.append(cmd)
        return self

    def wmacv(self,
              abort: enums.Abort,
              post: enums.WMACV.Post = None) -> MessageBuilder:
        if post is None:
            cmd = f'WMACV {abort}'
        else:
            cmd = f'WMACV {abort},{post}'

        self._msg.append(cmd)
        return self

    def wmdcv(self,
              abort: enums.Abort,
              post: enums.WMDCV.Post = None) -> MessageBuilder:
        if post is None:
            cmd = f'WMDCV {abort}'
        else:
            cmd = f'WMDCV {abort},{post}'

        self._msg.append(cmd)
        return self

    def wmfc(self,
             abort: enums.Abort,
             post: enums.WMFC.Post) -> MessageBuilder:
        if post is None:
            cmd = f'WMFC {abort}'
        else:
            cmd = f'WMFC {abort},{post}'

        self._msg.append(cmd)
        return self

    def wncc(self) -> MessageBuilder:
        cmd = 'WNCC'

        self._msg.append(cmd)
        return self

    @final_command
    def wnu_query(self) -> MessageBuilder:
        cmd = 'WNU?'

        self._msg.append(cmd)
        return self

    def wnx(self,
            n: int,
            chnum: Union[enums.ChNr, int],
            mode: enums.WNX.Mode,
            ranging_type: Union[enums.IOutputRange, enums.VOutputRange],
            start: float,
            stop: float,
            comp: float = None,
            p_comp: float = None) -> MessageBuilder:
        if comp is None:
            cmd = f'WNX {n},{chnum},{mode},{ranging_type},{start},{stop}'
        elif p_comp is None:
            cmd = f'WNX {n},{chnum},{mode},{ranging_type},{start},{stop},{comp}'
        else:
            cmd = f'WNX {n},{chnum},{mode},{ranging_type},{start},{stop},' \
                  f'{comp},{p_comp}'

        self._msg.append(cmd)
        return self

    def ws(self, mode: enums.WS.Mode = None) -> MessageBuilder:
        cmd = 'WS' if mode is None else f'WS {mode}'

        self._msg.append(cmd)
        return self

    def wsi(self,
            chnum: Union[enums.ChNr, int],
            i_range: enums.IOutputRange,
            start: float,
            stop: float,
            v_comp: float = None,
            p_comp: float = None) -> MessageBuilder:
        if v_comp is None:
            cmd = f'WSI {chnum},{i_range},{start},{stop}'
        elif p_comp is None:
            cmd = f'WSI {chnum},{i_range},{start},{stop},{v_comp}'
        else:
            cmd = f'WSI {chnum},{i_range},{start},{stop},{v_comp},{p_comp}'

        self._msg.append(cmd)
        return self

    def wsv(self,
            chnum: Union[enums.ChNr, int],
            v_range: enums.VOutputRange,
            start: float,
            stop: float,
            i_comp: float = None,
            p_comp: float = None) -> MessageBuilder:
        if i_comp is None:
            cmd = f'WSV {chnum},{v_range},{start},{stop}'
        elif p_comp is None:
            cmd = f'WSV {chnum},{v_range},{start},{stop},{i_comp}'
        else:
            cmd = f'WSV {chnum},{v_range},{start},{stop},{i_comp},{p_comp}'

        self._msg.append(cmd)
        return self

    def wt(self,
           hold: float,
           delay: float,
           step_delay: float = None,
           trigger_delay: float = None,
           measure_delay: float = None) -> MessageBuilder:
        if step_delay is None:
            cmd = f'WT {hold},{delay}'
        elif trigger_delay is None:
            cmd = f'WT {hold},{delay},{step_delay}'
        elif measure_delay is None:
            cmd = f'WT {hold},{delay},{step_delay},{trigger_delay}'
        else:
            cmd = f'WT {hold},{delay},{step_delay},{trigger_delay},' \
                  f'{measure_delay}'

        self._msg.append(cmd)
        return self

    def wtacv(self,
              hold: float,
              delay: float,
              step_delay: float = None,
              trigger_delay: float = None,
              measure_delay: float = None) -> MessageBuilder:
        if step_delay is None:
            cmd = f'WTACV {hold},{delay}'
        elif trigger_delay is None:
            cmd = f'WTACV {hold},{delay},{step_delay}'
        elif measure_delay is None:
            cmd = f'WTACV {hold},{delay},{step_delay},{trigger_delay}'
        else:
            cmd = f'WTACV {hold},{delay},{step_delay},{trigger_delay},' \
                  f'{measure_delay}'

        self._msg.append(cmd)
        return self

    def wtdcv(self,
              hold: float,
              delay: float,
              step_delay: float = None,
              trigger_delay: float = None,
              measure_delay: float = None) -> MessageBuilder:
        if step_delay is None:
            cmd = f'WTDCV {hold},{delay}'
        elif trigger_delay is None:
            cmd = f'WTDCV {hold},{delay},{step_delay}'
        elif measure_delay is None:
            cmd = f'WTDCV {hold},{delay},{step_delay},{trigger_delay}'
        else:
            cmd = f'WTDCV {hold},{delay},{step_delay},{trigger_delay},' \
                  f'{measure_delay}'

        self._msg.append(cmd)
        return self

    def wtfc(self,
             hold: float,
             delay: float,
             step_delay: float = None,
             trigger_delay: float = None,
             measure_delay: float = None) -> MessageBuilder:
        if step_delay is None:
            cmd = f'WTFC {hold},{delay}'
        elif trigger_delay is None:
            cmd = f'WTFC {hold},{delay},{step_delay}'
        elif measure_delay is None:
            cmd = f'WTFC {hold},{delay},{step_delay},{trigger_delay}'
        else:
            cmd = f'WTFC {hold},{delay},{step_delay},{trigger_delay},' \
                  f'{measure_delay}'

        self._msg.append(cmd)
        return self

    def wv(self,
           chnum: Union[enums.ChNr, int],
           mode: enums.SweepMode,
           v_range: enums.VOutputRange,
           start: float,
           stop: float,
           step: float,
           i_comp: float,
           p_comp: float) -> MessageBuilder:
        if i_comp is None:
            cmd = f'WV {chnum},{mode},{v_range},{start},{stop},{step}'
        elif p_comp is None:
            cmd = f'WV {chnum},{mode},{v_range},{start},{stop},{step},' \
                  f'{i_comp}'
        else:
            cmd = f'WV {chnum},{mode},{v_range},{start},{stop},{step},' \
                  f'{i_comp},{p_comp}'

        self._msg.append(cmd)
        return self

    @final_command
    def wz_query(self, timeout: float = None) -> MessageBuilder:
        cmd = 'WZ?' if timeout is None else f'WZ? {timeout}'

        self._msg.append(cmd)
        return self

    def xe(self) -> MessageBuilder:
        cmd = 'XE'

        self._msg.append(cmd)
        return self
