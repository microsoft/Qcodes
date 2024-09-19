import re
import textwrap
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.parameters import MultiParameter, Parameter, create_on_off_val_mapping

from . import constants
from .KeysightB1500_module import (
    KeysightB1500Module,
    StatusMixin,
    _FMTResponse,
    convert_dummy_val_to_nan,
    fmt_response_base_parser,
    parse_module_query_response,
    parse_spot_measurement_response,
)
from .KeysightB1511B import KeysightB1511B
from .KeysightB1517A import KeysightB1517A, _ParameterWithStatus
from .KeysightB1520A import KeysightB1520A
from .KeysightB1530A import KeysightB1530A
from .message_builder import MessageBuilder

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Unpack


class KeysightB1500(VisaInstrument):
    """Driver for Keysight B1500 Semiconductor Parameter Analyzer.

    For the list of supported modules, refer to :meth:`from_model_name`.
    """

    calibration_time_out = 60  # 30 seconds suggested by manual

    default_terminator = "\r\n"

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ):
        super().__init__(name, address, **kwargs)
        self.by_slot: dict[constants.SlotNr, KeysightB1500Module] = {}
        self.by_channel: dict[constants.ChNr, KeysightB1500Module] = {}
        self.by_kind: dict[constants.ModuleKind, list[KeysightB1500Module]] = (
            defaultdict(list)
        )

        self._find_modules()

        self.autozero_enabled: Parameter = self.add_parameter(
            "autozero_enabled",
            unit="",
            label="Autozero enabled of the high-resolution ADC",
            set_cmd=self._set_autozero,
            get_cmd=None,
            val_mapping=create_on_off_val_mapping(on_val=True, off_val=False),
            initial_cache_value=False,
            docstring=textwrap.dedent(
                """
            Enable or disable cancelling of the offset of the
            high-resolution A/D converter (ADC).

            Set the function to OFF in cases that the measurement speed is
            more important than the measurement accuracy. This roughly halves
            the integration time."""
            ),
        )
        """
        Enable or disable cancelling of the offset of the
        high-resolution A/D converter (ADC).

        Set the function to OFF in cases that the measurement speed is
        more important than the measurement accuracy. This roughly halves
        the integration time.
        """

        self.run_iv_staircase_sweep: IVSweepMeasurement = self.add_parameter(
            name="run_iv_staircase_sweep",
            parameter_class=IVSweepMeasurement,
            docstring=textwrap.dedent(
                """
               This is MultiParameter. Running the sweep runs the measurement
               on the list of source values defined using
               `setup_staircase_sweep` method. The output is a
               primary parameter (e.g. Gate current)  and a secondary
               parameter (e.g. Source/Drain current) both of which use the same
               setpoints. Note you must `set_measurement_mode` and specify
               2 channels as the argument before running the sweep. First
               channel (SMU) must be the channel on which you set the sweep (
               WV) and second channel(SMU) must be the one which remains at
               constants voltage.
                              """
            ),
        )
        """
        This is MultiParameter. Running the sweep runs the measurement
        on the list of source values defined using
        `setup_staircase_sweep` method. The output is a
        primary parameter (e.g. Gate current)  and a secondary
        parameter (e.g. Source/Drain current) both of which use the same
        setpoints. Note you must `set_measurement_mode` and specify
        2 channels as the argument before running the sweep. First
        channel (SMU) must be the channel on which you set the sweep (
        WV) and second channel(SMU) must be the one which remains at
        constants voltage.
        """

        self.connect_message()

    def write(self, cmd: str) -> None:
        """
        Extend write method from the super to ask for error message each
        time a write command is called.
        """
        super().write(cmd)
        error_message = self.error_message()
        if error_message != '+0,"No Error."':
            raise RuntimeError(
                f"While setting this parameter received error: {error_message}"
            )

    def add_module(self, name: str, module: KeysightB1500Module) -> None:
        super().add_submodule(name, module)

        self.by_kind[module.MODULE_KIND].append(module)
        self.by_slot[module.slot_nr] = module
        for ch in module.channels:
            self.by_channel[ch] = module

    def reset(self) -> None:
        """Performs an instrument reset.

        This does not reset error queue!
        """
        self.write("*RST")

    def get_status(self) -> int:
        return int(self.ask("*STB?"))

    # TODO: Data Output parser: At least for Format FMT1,0 and maybe for a
    # second (binary) format. 8 byte binary format would be nice because it
    # comes with time stamp
    # FMT1,0: ASCII (12 digits data with header) <CR/LF^EOI>

    def _find_modules(self) -> None:
        from .constants import UNT

        r = self.ask(MessageBuilder().unt_query(mode=UNT.Mode.MODULE_INFO_ONLY).message)

        slot_population = parse_module_query_response(r)

        for slot_nr, model in slot_population.items():
            module = self.from_model_name(model, slot_nr, self)

            self.add_module(name=module.short_name, module=module)

    @staticmethod
    def from_model_name(
        model: str, slot_nr: int, parent: "KeysightB1500", name: str | None = None
    ) -> "KeysightB1500Module":
        """Creates the correct instance of instrument module by model name.

        Args:
            model: Model name such as 'B1517A'
            slot_nr: Slot number of this module (not channel number)
            parent: Reference to B1500 mainframe instance
            name: Name of the instrument instance to create. If `None`
                (Default), then the name is autogenerated from the instrument
                class.

        Returns:
            A specific instance of :class:`.B1500Module`
        """
        if model == "B1511B":
            return KeysightB1511B(slot_nr=slot_nr, parent=parent, name=name)
        elif model == "B1517A":
            return KeysightB1517A(slot_nr=slot_nr, parent=parent, name=name)
        elif model == "B1520A":
            return KeysightB1520A(slot_nr=slot_nr, parent=parent, name=name)
        elif model == "B1530A":
            return KeysightB1530A(slot_nr=slot_nr, parent=parent, name=name)
        else:
            raise NotImplementedError(
                f"Module type {model} in slot {slot_nr} not yet supported."
            )

    def enable_channels(self, channels: constants.ChannelList | None = None) -> None:
        """Enable specified channels.

        If channels is omitted or `None`, then all channels are enabled.
        """
        msg = MessageBuilder().cn(channels)

        self.write(msg.message)

    def disable_channels(self, channels: constants.ChannelList | None = None) -> None:
        """Disable specified channels.

        If channels is omitted or `None`, then all channels are disabled.
        """
        msg = MessageBuilder().cl(channels)

        self.write(msg.message)

    # Response parsing functions as static methods for user convenience
    parse_spot_measurement_response = parse_spot_measurement_response
    parse_module_query_response = parse_module_query_response

    def _setup_integration_time(
        self,
        adc_type: constants.AIT.Type,
        mode: constants.AIT.Mode | int,
        coeff: int | None = None,
    ) -> None:
        """See :meth:`MessageBuilder.ait` for information"""
        if coeff is not None:
            coeff = int(coeff)
        self.write(
            MessageBuilder().ait(adc_type=adc_type, mode=mode, coeff=coeff).message
        )

    def _reset_measurement_statuses_of_smu_spot_measurement_parameters(
        self, parameter_name: str
    ) -> None:
        if parameter_name not in ("voltage", "current"):
            raise ValueError(
                f"Parameter name should be one of [voltage,current], "
                f"got {parameter_name}."
            )
        for smu in self.by_kind[constants.ModuleKind.SMU]:
            param = smu.parameters[parameter_name]
            assert isinstance(param, _ParameterWithStatus)
            param._measurement_status = None

    def use_nplc_for_high_speed_adc(self, n: int | None = None) -> None:
        """
        Set the high-speed ADC to NPLC mode, with optionally defining number
        of averaging samples via argument `n`.

        Args:
            n: Value that defines the number of averaging samples given by
                the following formula:

                ``Number of averaging samples = n / 128``.

                n=1 to 100. Default setting is 1 (if `None` is passed).

                The Keysight B1500 gets 128 samples in a power line cycle,
                repeats this for the times you specify, and performs
                averaging to get the measurement data. (For more info see
                Table 4-21.).  Note that the integration time will not be
                updated if a non-integer value is written to the B1500.
        """
        self._setup_integration_time(
            adc_type=constants.AIT.Type.HIGH_SPEED,
            mode=constants.AIT.Mode.NPLC,
            coeff=n,
        )

    def use_nplc_for_high_resolution_adc(self, n: int | None = None) -> None:
        """
        Set the high-resolution ADC to NPLC mode, with optionally defining
        the number of PLCs per sample via argument `n`.

        Args:
            n: Value that defines the integration time given by the
                following formula:

                ``Integration time = n / power line frequency``.

                n=1 to 100. Default setting is 1 (if `None` is passed).
                (For more info see Table 4-21.).  Note that the integration
                time will not be updated if a non-integer value is written
                to the B1500.
        """
        self._setup_integration_time(
            adc_type=constants.AIT.Type.HIGH_RESOLUTION,
            mode=constants.AIT.Mode.NPLC,
            coeff=n,
        )

    def use_manual_mode_for_high_speed_adc(self, n: int | None = None) -> None:
        """
        Set the high-speed ADC to manual mode, with optionally defining number
        of averaging samples via argument `n`.

        Use ``n=1`` to disable averaging (``n=None`` uses the default
        setting from the instrument which is also ``n=1``).

        Args:
            n: Number of averaging samples, between 1 and 1023. Default
                setting is 1. (For more info see Table 4-21.)
                Note that the integration time will not be updated
                if a non-integer value is written to the B1500.
        """
        self._setup_integration_time(
            adc_type=constants.AIT.Type.HIGH_SPEED,
            mode=constants.AIT.Mode.MANUAL,
            coeff=n,
        )

    def _set_autozero(self, do_autozero: bool) -> None:
        self.write(MessageBuilder().az(do_autozero=do_autozero).message)

    def self_calibration(
        self, slot: constants.SlotNr | int | None = None
    ) -> constants.CALResponse:
        """
        Performs the self calibration of the specified module (SMU) and
        returns the result. Failed modules are disabled, and can only be
        enabled by the ``RCV`` command.

        Calibration takes about 30 seconds (the visa timeout for it is
        controlled by :attr:`calibration_time_out` attribute).

        Execution Conditions: No SMU may be in the high voltage state
        (forcing more than ±42 V, or voltage compliance set to more than
        ±42 V). Before starting the calibration, open the measurement
        terminals.

        Args:
            slot: Slot number of the slot that installs the module to perform
                the self-calibration. For Ex:
                constants.SlotNr.ALL, MAINFRAME, SLOT01, SLOT02 ...SLOT10
                If not specified, the calibration is performed for all the
                modules and the mainframe.
        """
        msg = MessageBuilder().cal_query(slot=slot)
        with self.root_instrument.timeout.set_to(self.calibration_time_out):
            response = self.ask(msg.message)
        return constants.CALResponse(int(response))

    def error_message(self, mode: constants.ERRX.Mode | int | None = None) -> str:
        """
        This method reads one error code from the head of the error
        queue and removes that code from the queue. The read error is
        returned as the response of this method.

        Args:
            mode: If no valued passed returns both the error value and the
                error message. See :class:`.constants.ERRX.Mode` for possible
                arguments.

        Returns:
            In the default case response message contains an error message
            and a custom message containing additional information such as
            the slot number. They are separated by a semicolon (;). For
            example, if the error 305 occurs on the slot 1, this method
            returns the following response. 305,"Excess current in HPSMU.;
            SLOT1" If no error occurred, this command returns 0,"No Error."
        """

        msg = MessageBuilder().errx_query(mode=mode)
        response = self.ask(msg.message)
        return response

    def clear_buffer_of_error_message(self) -> None:
        """
        This method clears the error message stored in buffer when the
        error_message command is executed.
        """
        msg = MessageBuilder().err_query()
        self.write(msg.message)

    def clear_timer_count(self, chnum: int | None = None) -> None:
        """
        This command clears the timer count. This command is effective for
        all measurement modes, regardless of the TSC setting. This command
        is not effective for the 4 byte binary data output format
        (FMT3 and FMT4).

        Args:
            chnum: SMU or MFCMU channel number. Integer expression. 1 to 10.
                See Table 4-1 on page 16 of 2016 manual. If chnum is
                specified, this command clears the timer count once at the
                source output start by the DV, DI, or DCV command for the
                specified channel. The channel output switch of the
                specified channel must be ON when the timer count is
                cleared.

        If chnum is not specified, this command clears the timer count
        immediately,
        """
        msg = MessageBuilder().tsr(chnum=chnum)
        self.write(msg.message)

    def set_measurement_mode(
        self,
        mode: constants.MM.Mode | int,
        channels: constants.ChannelList | None = None,
    ) -> None:
        """
        This method specifies the measurement mode and the channels used
        for measurements. This method must be entered to specify the
        measurement mode. For the high speed spot measurements,
        do not use this method.
        NOTE Order of the channels are important. The SMU which is setup to
        run the sweep goes first.

        Args:
            mode: Measurement mode. See `constants.MM.Mode` for all possible
                modes
            channels: Measurement channel number. See `constants.ChannelList`
                for all possible channels.
        """
        msg = MessageBuilder().mm(mode=mode, channels=channels).message
        self.write(msg)

    def get_measurement_mode(self) -> dict[str, constants.MM.Mode | list[int]]:
        """
        This method gets the measurement mode(MM) and the channels used
        for measurements. It outputs a dictionary with 'mode' and
        'channels' as keys.
        """
        msg = MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.TM_AV_CM_FMT_MM_SETTINGS
        )
        response = self.ask(msg.message)
        match = re.search("MM(?P<mode>.*?),(?P<channels>.*?)(;|$)", response)

        if not match:
            raise ValueError("Measurement Mode (MM) not found.")

        out_dict: dict[str, constants.MM.Mode | list[int]] = {}
        resp_dict = match.groupdict()
        out_dict["mode"] = constants.MM.Mode(int(resp_dict["mode"]))
        out_dict["channels"] = list(map(int, resp_dict["channels"].split(",")))
        return out_dict

    def get_response_format_and_mode(
        self,
    ) -> dict[str, constants.FMT.Format | constants.FMT.Mode]:
        """
        This method queries the the data output format and mode.
        """
        msg = MessageBuilder().lrn_query(
            type_id=constants.LRN.Type.TM_AV_CM_FMT_MM_SETTINGS
        )
        response = self.ask(msg.message)
        match = re.search("FMT(?P<format>.*?),(?P<mode>.*?)(;|$)", response)

        if not match:
            raise ValueError("Measurement Mode (FMT) not found.")

        out_dict: dict[str, constants.FMT.Format | constants.FMT.Mode] = {}
        resp_dict = match.groupdict()
        out_dict["format"] = constants.FMT.Format(int(resp_dict["format"]))
        out_dict["mode"] = constants.FMT.Mode(int(resp_dict["mode"]))
        return out_dict

    def enable_smu_filters(
        self, enable_filter: bool, channels: constants.ChannelList | None = None
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
        self.write(
            MessageBuilder().fl(enable_filter=enable_filter, channels=channels).message
        )


class IVSweepMeasurement(MultiParameter, StatusMixin):
    """
    IV sweep measurement outputs a list of measured current parameters
    as a result of voltage sweep.

    Args:
        name: Name of the Parameter.
        instrument: Instrument to which this parameter communicates to.
    """

    def __init__(self, name: str, instrument: KeysightB1517A, **kwargs: Any):
        super().__init__(
            name,
            names=tuple(["param1", "param2"]),
            units=tuple(["A", "A"]),
            labels=tuple(["Param1 Current", "Param2 Current"]),
            shapes=((1,),) * 2,
            setpoint_names=(("Voltage",),) * 2,
            setpoint_labels=(("Voltage",),) * 2,
            setpoint_units=(("V",),) * 2,
            instrument=instrument,
            **kwargs,
        )

        self.instrument: KeysightB1517A
        self.root_instrument: KeysightB1500

        self.param1 = _FMTResponse(None, None, None, None)
        self.param2 = _FMTResponse(None, None, None, None)
        self.source_voltage = _FMTResponse(None, None, None, None)
        self._fudge: float = 1.5

    def set_names_labels_and_units(
        self,
        names: Optional["Sequence[str]"] = None,
        labels: Optional["Sequence[str]"] = None,
        units: Optional["Sequence[str]"] = None,
    ) -> None:
        """
        Set names, labels, and units of the measured parts of the MultiParameter.

        If units are not provided, "A" will be used because this parameter
        measures currents.

        If labels are not provided, names will be used.

        If names are not provided, ``param#`` will be used as names; the number
        of those names will be the same as the number of measured channels
        that ``B1500.get_measurement_mode`` method returns. Note that it is
        possible to not provide names and provide labels at the same time.
        In case, neither names nor labels are provided, the labels will be
        generated as ``Param# Current``.

        The number of provided names, labels, and units must be the same.
        Moreover, that number has to be equal to the number of channels
        that ``B1500.get_measurement_mode`` method returns. It is
        recommended to set measurement mode and number of channels first,
        and only then call this method to provide names/labels/units.

        The name/label/unit of the setpoint of this parameter will also be
        updated to defaults dictated by the
        ``set_setpoint_name_label_and_unit`` method.

        Note that ``.shapes`` of this parameter will also be updated to
        be in sync with the number of names.
        """
        measurement_mode = self.instrument.get_measurement_mode()
        channels = measurement_mode["channels"]

        if names is None:
            names = [f"param{n+1}" for n in range(len(channels))]
            if labels is None:
                labels = [f"Param{n + 1} Current" for n in range(len(channels))]

        if labels is None:
            labels = tuple(names)

        if units is None:
            units = ["A"] * len(names)

        if len(labels) != len(names) or len(units) != len(names):
            raise ValueError(
                f"If provided, the number of names, labels, and units must be "
                f"the same, instead got {len(names)} names, {len(labels)} "
                f"labels, {len(units)} units."
            )

        if len(names) != len(channels):
            raise ValueError(
                f"The number of names ({len(names)}) does not match the number "
                f"of channels expected for the IV sweep measurement, "
                f"which is {len(channels)}. Please, when providing names, "
                f"provide them for every channel."
            )

        self.names = tuple(names)
        self.labels = tuple(labels)
        self.units = tuple(units)

        for n in range(len(channels)):
            setattr(self, f"param{n+1}", _FMTResponse(None, None, None, None))

        self.shapes = ((1,),) * len(self.names)

        self.set_setpoint_name_label_and_unit()

    def set_setpoint_name_label_and_unit(
        self, name: str | None = None, label: str | None = None, unit: str | None = None
    ) -> None:
        """
        Set name, label, and unit of the setpoint of the MultiParameter.

        If unit is not provided, "V" will be used because this parameter
        sweeps voltage.

        If label is not provided, "Voltage" will be used.

        If name are not provided, ``voltage`` will be used.

        The attributes describing the setpoints of this MultiParameter
        will be updated to match the number of measured parameters of
        this MultiParameter, as dictated by ``.names``.
        """
        # number of measured parameters of this MultiParameter
        n_names = len(self.names)

        name = name if name is not None else "voltage"
        label = label if label is not None else "Voltage"
        unit = unit if unit is not None else "V"

        self.setpoint_names = ((name,),) * n_names
        self.setpoint_labels = ((label,),) * n_names
        self.setpoint_units = ((unit,),) * n_names

    def get_raw(self) -> tuple[tuple[float, ...], ...]:
        measurement_mode = self.instrument.get_measurement_mode()
        channels = measurement_mode["channels"]
        n_channels = len(channels)

        if n_channels < 1:
            raise ValueError(
                "At least one measurement channel is needed for an IV sweep."
            )

        if (
            len(self.names) != n_channels
            or len(self.units) != n_channels
            or len(self.labels) != n_channels
            or len(self.shapes) != n_channels
        ):
            raise ValueError(
                f"The number of `.names` ({len(self.names)}), "
                f"`.units` ({len(self.units)}), `.labels` ("
                f"{len(self.labels)}), or `.shapes` ({len(self.shapes)}) "
                f"of the {self.full_name} parameter "
                f"does not match the number of channels expected for the IV "
                f"sweep measurement, which is {n_channels}. One must set "
                f"enough names, units, and labels for all the channels that "
                f"are to be measured."
            )

        smu = self.instrument.by_channel[channels[0]]

        if not smu.setup_fnc_already_run:
            raise Exception(
                f"Sweep setup has not yet been run successfully on {smu.full_name}"
            )

        delay_time = smu.iv_sweep.step_delay()
        if smu._average_coefficient < 0:
            # negative coefficient means nplc and positive means just
            # averaging, see B1517A.set_average_samples_for_high_speed_adc
            # for more info
            nplc = 128 * abs(smu._average_coefficient)
            power_line_time_period = 1 / smu.power_line_frequency
            calculated_time = 2 * nplc * power_line_time_period
        else:
            calculated_time = smu._average_coefficient * delay_time
        num_steps = smu.iv_sweep.sweep_steps()
        estimated_timeout = max(delay_time, calculated_time) * num_steps
        new_timeout = estimated_timeout * self._fudge

        format_and_mode = self.instrument.get_response_format_and_mode()
        fmt_format = format_and_mode["format"]
        fmt_mode = format_and_mode["mode"]
        try:
            self.root_instrument.write(MessageBuilder().fmt(1, 1).message)
            with self.root_instrument.timeout.set_to(new_timeout):
                raw_data = self.instrument.ask(MessageBuilder().xe().message)
        finally:
            self.root_instrument.write(
                MessageBuilder().fmt(fmt_format, fmt_mode).message
            )

        parsed_data = fmt_response_base_parser(raw_data)

        # The `4` comes from the len(_FMTResponse(None, None, None, None)),
        # the _FMTResponse tuple declares these items that the instrument
        # gives for each data point
        n_items_per_data_point = 4

        # sourced voltage values are also returned, hence the `+1`
        n_all_data_channels = n_channels + 1

        for channel_index in range(n_channels):
            parsed_data_items = [
                parsed_data[i][channel_index::n_all_data_channels]
                for i in range(0, n_items_per_data_point)
            ]
            single_channel_data = _FMTResponse(*parsed_data_items)
            convert_dummy_val_to_nan(single_channel_data)

            # Store the results to `.param#` attributes for convenient access
            # to all the data, e.g. status of each value in the arrays
            setattr(self, f"param{channel_index+1}", single_channel_data)

        channel_values_to_return = tuple(
            getattr(self, f"param{n + 1}").value for n in range(n_channels)
        )

        source_voltage_index = n_channels
        parsed_source_voltage_items = [
            parsed_data[i][source_voltage_index::n_all_data_channels]
            for i in range(0, n_items_per_data_point)
        ]
        self.source_voltage = _FMTResponse(*parsed_source_voltage_items)

        self.shapes = ((len(self.source_voltage.value),),) * n_channels
        self.setpoints = ((self.source_voltage.value,),) * n_channels

        return channel_values_to_return
