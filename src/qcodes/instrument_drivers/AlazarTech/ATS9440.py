from typing import TYPE_CHECKING, Any

from qcodes import validators

from .ATS import AlazarTechATS
from .utils import TraceParameter

if TYPE_CHECKING:
    from qcodes.parameters import Parameter


class AlazarTechATS9440(AlazarTechATS):
    """
    This class is the driver for the ATS9440 board
    it inherits from the ATS base class
    """

    samples_divisor = 32
    channels = 4

    def __init__(
        self,
        name: str,
        dll_path: str = "C:\\WINDOWS\\System32\\ATSApi.dll",
        **kwargs: Any,
    ):
        super().__init__(name, dll_path=dll_path, **kwargs)

        # add parameters

        # ----- Parameters for the configuration of the board -----
        self.clock_source: TraceParameter = self.add_parameter(
            name="clock_source",
            parameter_class=TraceParameter,
            label="Clock Source",
            unit=None,
            initial_value="INTERNAL_CLOCK",
            val_mapping={
                "INTERNAL_CLOCK": 1,
                "FAST_EXTERNAL_CLOCK": 2,
                "SLOW_EXTERNAL_CLOCK": 4,
                "EXTERNAL_CLOCK_10MHz_REF": 7,
            },
        )
        """Parameter clock_source"""
        self.external_sample_rate: TraceParameter = self.add_parameter(
            name="external_sample_rate",
            parameter_class=TraceParameter,
            label="External Sample Rate",
            unit="S/s",
            vals=validators.MultiType(
                validators.Ints(1000000, 125000000), validators.Enum("UNDEFINED")
            ),
            initial_value="UNDEFINED",
        )
        """Parameter external_sample_rate"""
        self.sample_rate: TraceParameter = self.add_parameter(
            name="sample_rate",
            parameter_class=TraceParameter,
            label="Internal Sample Rate",
            unit="S/s",
            initial_value=100000000,
            val_mapping={
                1_000: 1,
                2_000: 2,
                5_000: 4,
                10_000: 8,
                20_000: 10,
                50_000: 12,
                100_000: 14,
                200_000: 16,
                500_000: 18,
                1_000_000: 20,
                2_000_000: 24,
                5_000_000: 26,
                10_000_000: 28,
                20_000_000: 30,
                50_000_000: 34,
                100_000_000: 36,
                125_000_000: 38,
                "EXTERNAL_CLOCK": 64,
                "UNDEFINED": "UNDEFINED",
            },
        )
        """Parameter sample_rate"""
        self.clock_edge: TraceParameter = self.add_parameter(
            name="clock_edge",
            parameter_class=TraceParameter,
            label="Clock Edge",
            unit=None,
            initial_value="CLOCK_EDGE_RISING",
            val_mapping={"CLOCK_EDGE_RISING": 0, "CLOCK_EDGE_FALLING": 1},
        )
        """Parameter clock_edge"""
        self.decimation: TraceParameter = self.add_parameter(
            name="decimation",
            parameter_class=TraceParameter,
            label="Decimation",
            unit=None,
            initial_value=1,
            vals=validators.Ints(1, 100000),
        )
        """Parameter decimation"""
        for i in range(1, self.channels + 1):
            self.add_parameter(
                name=f"coupling{i}",
                parameter_class=TraceParameter,
                label=f"Coupling channel {i}",
                unit=None,
                initial_value="DC",
                val_mapping={"AC": 1, "DC": 2},
            )
            self.add_parameter(
                name=f"channel_range{i}",
                parameter_class=TraceParameter,
                label=f"Range channel {i}",
                unit="V",
                initial_value=0.1,
                val_mapping={0.1: 5, 0.2: 6, 0.4: 7, 1: 10, 2: 11, 4: 12},
            )
            self.add_parameter(
                name=f"impedance{i}",
                parameter_class=TraceParameter,
                label=f"Impedance channel {i}",
                unit="Ohm",
                initial_value=50,
                val_mapping={50: 2},
            )

            self.add_parameter(
                name=f"bwlimit{i}",
                parameter_class=TraceParameter,
                label=f"Bandwidth limit channel {i}",
                unit=None,
                initial_value="DISABLED",
                val_mapping={"DISABLED": 0, "ENABLED": 1},
            )
        self.trigger_operation: TraceParameter = self.add_parameter(
            name="trigger_operation",
            parameter_class=TraceParameter,
            label="Trigger Operation",
            unit=None,
            initial_value="TRIG_ENGINE_OP_J",
            val_mapping={
                "TRIG_ENGINE_OP_J": 0,
                "TRIG_ENGINE_OP_K": 1,
                "TRIG_ENGINE_OP_J_OR_K": 2,
                "TRIG_ENGINE_OP_J_AND_K": 3,
                "TRIG_ENGINE_OP_J_XOR_K": 4,
                "TRIG_ENGINE_OP_J_AND_NOT_K": 5,
                "TRIG_ENGINE_OP_NOT_J_AND_K": 6,
            },
        )
        """Parameter trigger_operation"""
        n_trigger_engines = 2

        for i in range(1, n_trigger_engines + 1):
            self.add_parameter(
                name=f"trigger_engine{i}",
                parameter_class=TraceParameter,
                label=f"Trigger Engine {i}",
                unit=None,
                initial_value="TRIG_ENGINE_" + ("J" if i == 1 else "K"),
                val_mapping={"TRIG_ENGINE_J": 0, "TRIG_ENGINE_K": 1},
            )
            self.add_parameter(
                name=f"trigger_source{i}",
                parameter_class=TraceParameter,
                label=f"Trigger Source {i}",
                unit=None,
                initial_value="EXTERNAL",
                val_mapping={
                    "CHANNEL_A": 0,
                    "CHANNEL_B": 1,
                    "EXTERNAL": 2,
                    "DISABLE": 3,
                    "CHANNEL_C": 4,
                    "CHANNEL_D": 5,
                },
            )
            self.add_parameter(
                name=f"trigger_slope{i}",
                parameter_class=TraceParameter,
                label=f"Trigger Slope {i}",
                unit=None,
                initial_value="TRIG_SLOPE_POSITIVE",
                val_mapping={"TRIG_SLOPE_POSITIVE": 1, "TRIG_SLOPE_NEGATIVE": 2},
            )
            self.add_parameter(
                name=f"trigger_level{i}",
                parameter_class=TraceParameter,
                label=f"Trigger Level {i}",
                unit=None,
                initial_value=140,
                vals=validators.Ints(0, 255),
            )
        self.external_trigger_coupling: TraceParameter = self.add_parameter(
            name="external_trigger_coupling",
            parameter_class=TraceParameter,
            label="External Trigger Coupling",
            unit=None,
            initial_value="DC",
            val_mapping={"AC": 1, "DC": 2},
        )
        """Parameter external_trigger_coupling"""
        self.external_trigger_range: TraceParameter = self.add_parameter(
            name="external_trigger_range",
            parameter_class=TraceParameter,
            label="External Trigger Range",
            unit=None,
            initial_value="ETR_5V",
            val_mapping={"ETR_5V": 0, "ETR_TTL": 2},
        )
        """Parameter external_trigger_range"""
        self.trigger_delay: TraceParameter = self.add_parameter(
            name="trigger_delay",
            parameter_class=TraceParameter,
            label="Trigger Delay",
            unit="Sample clock cycles",
            initial_value=0,
            vals=validators.Multiples(divisor=8, min_value=0),
        )
        """Parameter trigger_delay"""
        self.timeout_ticks: TraceParameter = self.add_parameter(
            name="timeout_ticks",
            parameter_class=TraceParameter,
            label="Timeout Ticks",
            unit="10 us",
            initial_value=0,
            vals=validators.Ints(min_value=0),
        )
        """Parameter timeout_ticks"""
        #  The card has two AUX I/O ports, which only AUX 2 is controlled by
        #  the software (AUX 1 is controlled by the firmware). The user should
        #  use AUX 2 for controlling the AUX via aux_io_mode and aux_io_param.
        self.aux_io_mode: TraceParameter = self.add_parameter(
            name="aux_io_mode",
            parameter_class=TraceParameter,
            label="AUX I/O Mode",
            unit=None,
            initial_value="AUX_OUT_TRIGGER",
            val_mapping={
                "AUX_OUT_TRIGGER": 0,
                "AUX_IN_TRIGGER_ENABLE": 1,
                "AUX_IN_AUXILIARY": 13,
            },
        )
        """Parameter aux_io_mode"""
        self.aux_io_param: TraceParameter = self.add_parameter(
            name="aux_io_param",
            parameter_class=TraceParameter,
            label="AUX I/O Param",
            unit=None,
            initial_value="NONE",
            val_mapping={"NONE": 0, "TRIG_SLOPE_POSITIVE": 1, "TRIG_SLOPE_NEGATIVE": 2},
        )
        """Parameter aux_io_param"""

        #  The above parameters are important for preparing the card.
        self.mode: Parameter = self.add_parameter(
            name="mode",
            label="Acquisition mode",
            unit=None,
            initial_value="NPT",
            set_cmd=None,
            val_mapping={"NPT": 0x200, "TS": 0x400},
        )
        """Parameter mode"""
        self.samples_per_record: Parameter = self.add_parameter(
            name="samples_per_record",
            label="Samples per Record",
            unit=None,
            initial_value=1024,
            set_cmd=None,
            vals=validators.Multiples(divisor=self.samples_divisor, min_value=256),
        )
        """Parameter samples_per_record"""
        self.records_per_buffer: Parameter = self.add_parameter(
            name="records_per_buffer",
            label="Records per Buffer",
            unit=None,
            initial_value=10,
            set_cmd=None,
            vals=validators.Ints(min_value=0),
        )
        """Parameter records_per_buffer"""
        self.buffers_per_acquisition: Parameter = self.add_parameter(
            name="buffers_per_acquisition",
            label="Buffers per Acquisition",
            unit=None,
            set_cmd=None,
            initial_value=10,
            vals=validators.Ints(min_value=0),
        )
        """Parameter buffers_per_acquisition"""
        self.channel_selection: Parameter = self.add_parameter(
            name="channel_selection",
            label="Channel Selection",
            unit=None,
            set_cmd=None,
            initial_value="AB",
            val_mapping={
                "A": 1,
                "B": 2,
                "AB": 3,
                "C": 4,
                "AC": 5,
                "BC": 6,
                "D": 8,
                "AD": 9,
                "BD": 10,
                "CD": 12,
                "ABCD": 15,
            },
        )
        """Parameter channel_selection"""
        self.transfer_offset: Parameter = self.add_parameter(
            name="transfer_offset",
            label="Transfer Offset",
            unit="Samples",
            set_cmd=None,
            initial_value=0,
            vals=validators.Ints(min_value=0),
        )
        """Parameter transfer_offset"""
        self.external_startcapture: Parameter = self.add_parameter(
            name="external_startcapture",
            label="External Startcapture",
            unit=None,
            set_cmd=None,
            initial_value="ENABLED",
            val_mapping={"DISABLED": 0x0, "ENABLED": 0x1},
        )
        """Parameter external_startcapture"""
        self.enable_record_headers: Parameter = self.add_parameter(
            name="enable_record_headers",
            label="Enable Record Headers",
            unit=None,
            set_cmd=None,
            initial_value="DISABLED",
            val_mapping={"DISABLED": 0x0, "ENABLED": 0x8},
        )
        """Parameter enable_record_headers"""
        self.alloc_buffers: Parameter = self.add_parameter(
            name="alloc_buffers",
            label="Alloc Buffers",
            unit=None,
            set_cmd=None,
            initial_value="DISABLED",
            val_mapping={"DISABLED": 0x0, "ENABLED": 0x20},
        )
        """Parameter alloc_buffers"""
        self.fifo_only_streaming: Parameter = self.add_parameter(
            name="fifo_only_streaming",
            label="Fifo Only Streaming",
            unit=None,
            set_cmd=None,
            initial_value="DISABLED",
            val_mapping={"DISABLED": 0x0, "ENABLED": 0x800},
        )
        """Parameter fifo_only_streaming"""
        self.interleave_samples: Parameter = self.add_parameter(
            name="interleave_samples",
            label="Interleave Samples",
            unit=None,
            set_cmd=None,
            initial_value="DISABLED",
            val_mapping={"DISABLED": 0x0, "ENABLED": 0x1000},
        )
        """Parameter interleave_samples"""
        self.get_processed_data: Parameter = self.add_parameter(
            name="get_processed_data",
            label="Get Processed Data",
            unit=None,
            set_cmd=None,
            initial_value="DISABLED",
            val_mapping={"DISABLED": 0x0, "ENABLED": 0x2000},
        )
        """Parameter get_processed_data"""
        self.allocated_buffers: Parameter = self.add_parameter(
            name="allocated_buffers",
            label="Allocated Buffers",
            unit=None,
            set_cmd=None,
            initial_value=4,
            vals=validators.Ints(min_value=0),
        )
        """Parameter allocated_buffers"""
        self.buffer_timeout: Parameter = self.add_parameter(
            name="buffer_timeout",
            label="Buffer Timeout",
            unit="ms",
            set_cmd=None,
            initial_value=1000,
            vals=validators.Ints(min_value=0),
        )
        """Parameter buffer_timeout"""


class AlazarTech_ATS9440(AlazarTechATS9440):
    """
    Alias for backwards compatibility. Will eventually be deprecated and removed
    """

    pass
