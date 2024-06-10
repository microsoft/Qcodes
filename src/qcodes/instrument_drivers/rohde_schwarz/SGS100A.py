from typing import TYPE_CHECKING

import qcodes.validators as vals
from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs
from qcodes.parameters import create_on_off_val_mapping

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter


class RohdeSchwarzSGS100A(VisaInstrument):
    """
    This is the QCoDeS driver for the Rohde & Schwarz SGS100A signal generator.

    Status: beta-version.

    .. todo::

        - Add all parameters that are in the manual
        - Add test suite
        - See if there can be a common driver for RS mw sources from which
          different models inherit

    This driver will most likely work for multiple Rohde & Schwarz sources.
    it would be a good idea to group all similar RS drivers together in one
    module.

    Tested working with

    - RS_SGS100A

    This driver does not contain all commands available for the RS_SGS100A but
    only the ones most commonly used.
    """

    default_terminator = "\n"

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ) -> None:
        super().__init__(name, address, **kwargs)

        self.frequency: Parameter = self.add_parameter(
            name="frequency",
            label="Frequency",
            unit="Hz",
            get_cmd="SOUR:FREQ?",
            set_cmd="SOUR:FREQ {:.2f}",
            get_parser=float,
            vals=vals.Numbers(1e6, 20e9),
        )
        """Parameter frequency"""
        self.phase: Parameter = self.add_parameter(
            name="phase",
            label="Phase",
            unit="deg",
            get_cmd="SOUR:PHAS?",
            set_cmd="SOUR:PHAS {:.2f}",
            get_parser=float,
            vals=vals.Numbers(0, 360),
        )
        """Parameter phase"""
        self.power: Parameter = self.add_parameter(
            name="power",
            label="Power",
            unit="dBm",
            get_cmd="SOUR:POW?",
            set_cmd="SOUR:POW {:.2f}",
            get_parser=float,
            vals=vals.Numbers(-120, 25),
        )
        """Parameter power"""
        self.status: Parameter = self.add_parameter(
            "status",
            label="RF Output",
            get_cmd=":OUTP:STAT?",
            set_cmd=":OUTP:STAT {}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
        )
        """Parameter status"""
        self.IQ_state: Parameter = self.add_parameter(
            "IQ_state",
            label="IQ Modulation",
            get_cmd=":IQ:STAT?",
            set_cmd=":IQ:STAT {}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
        )
        """Parameter IQ_state"""
        self.pulsemod_state: Parameter = self.add_parameter(
            "pulsemod_state",
            label="Pulse Modulation",
            get_cmd=":SOUR:PULM:STAT?",
            set_cmd=":SOUR:PULM:STAT {}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
        )
        """Parameter pulsemod_state"""
        self.pulsemod_source: Parameter = self.add_parameter(
            "pulsemod_source",
            label="Pulse Modulation Source",
            get_cmd="SOUR:PULM:SOUR?",
            set_cmd="SOUR:PULM:SOUR {}",
            vals=vals.Enum("INT", "EXT", "int", "ext"),
        )
        """Parameter pulsemod_source"""
        self.ref_osc_source: Parameter = self.add_parameter(
            "ref_osc_source",
            label="Reference Oscillator Source",
            get_cmd="SOUR:ROSC:SOUR?",
            set_cmd="SOUR:ROSC:SOUR {}",
            vals=vals.Enum("INT", "EXT", "int", "ext"),
        )
        """Parameter ref_osc_source"""
        # Define LO source INT/EXT (Only with K-90 option)
        self.LO_source: Parameter = self.add_parameter(
            "LO_source",
            label="Local Oscillator Source",
            get_cmd="SOUR:LOSC:SOUR?",
            set_cmd="SOUR:LOSC:SOUR {}",
            vals=vals.Enum("INT", "EXT", "int", "ext"),
        )
        """Parameter LO_source"""
        # Define output at REF/LO Output (Only with K-90 option)
        self.ref_LO_out: Parameter = self.add_parameter(
            "ref_LO_out",
            label="REF/LO Output",
            get_cmd="CONN:REFL:OUTP?",
            set_cmd="CONN:REFL:OUTP {}",
            vals=vals.Enum("REF", "LO", "OFF", "ref", "lo", "off", "Off"),
        )
        """Parameter ref_LO_out"""
        # Frequency mw_source outputs when used as a reference
        self.ref_osc_output_freq: Parameter = self.add_parameter(
            "ref_osc_output_freq",
            label="Reference Oscillator Output Frequency",
            get_cmd="SOUR:ROSC:OUTP:FREQ?",
            set_cmd="SOUR:ROSC:OUTP:FREQ {}",
            vals=vals.Enum("10MHz", "100MHz", "1000MHz"),
        )
        """Parameter ref_osc_output_freq"""
        # Frequency of the external reference mw_source uses
        self.ref_osc_external_freq: Parameter = self.add_parameter(
            "ref_osc_external_freq",
            label="Reference Oscillator External Frequency",
            get_cmd="SOUR:ROSC:EXT:FREQ?",
            set_cmd="SOUR:ROSC:EXT:FREQ {}",
            vals=vals.Enum("10MHz", "100MHz", "1000MHz"),
        )
        """Parameter ref_osc_external_freq"""
        # IQ impairments
        self.IQ_impairments: Parameter = self.add_parameter(
            "IQ_impairments",
            label="IQ Impairments",
            get_cmd=":SOUR:IQ:IMP:STAT?",
            set_cmd=":SOUR:IQ:IMP:STAT {}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
        )
        """Parameter IQ_impairments"""
        self.I_offset: Parameter = self.add_parameter(
            "I_offset",
            label="I Offset",
            get_cmd="SOUR:IQ:IMP:LEAK:I?",
            set_cmd="SOUR:IQ:IMP:LEAK:I {:.2f}",
            get_parser=float,
            vals=vals.Numbers(-10, 10),
        )
        """Parameter I_offset"""
        self.Q_offset: Parameter = self.add_parameter(
            "Q_offset",
            label="Q Offset",
            get_cmd="SOUR:IQ:IMP:LEAK:Q?",
            set_cmd="SOUR:IQ:IMP:LEAK:Q {:.2f}",
            get_parser=float,
            vals=vals.Numbers(-10, 10),
        )
        """Parameter Q_offset"""
        self.IQ_gain_imbalance: Parameter = self.add_parameter(
            "IQ_gain_imbalance",
            label="IQ Gain Imbalance",
            get_cmd="SOUR:IQ:IMP:IQR?",
            set_cmd="SOUR:IQ:IMP:IQR {:.2f}",
            get_parser=float,
            vals=vals.Numbers(-1, 1),
        )
        """Parameter IQ_gain_imbalance"""
        self.IQ_angle: Parameter = self.add_parameter(
            "IQ_angle",
            label="IQ Angle Offset",
            get_cmd="SOUR:IQ:IMP:QUAD?",
            set_cmd="SOUR:IQ:IMP:QUAD {:.2f}",
            get_parser=float,
            vals=vals.Numbers(-8, 8),
        )
        """Parameter IQ_angle"""
        # Determines the signal at the input/output of the multi purpose [TRIG] connector.
        self.trigger_connector_mode: Parameter = self.add_parameter(
            "trigger_connector_mode",
            label="Trigger Connector Mode",
            get_cmd="CONN:TRIG:OMOD?",
            set_cmd="CONN:TRIG:OMOD {}",
            vals=vals.Enum(
                "SVAL",  # SVALid - Signal valid
                "SNVAL",  # SNValid - Signal not valid
                "PVO",  # PVOut - Pulse video out (K22 Only)
                "PET",  # PETrigger - Pulse mod ext trigger - PETrigger (K22 Only)
                "PEMS",  # PEMSource - Pulse mode ext source (K22 Only)
                "sval",  # same as SVAL
                "snval",  # same as SNVAL
                "pvo",  # same as PVO
                "pet",  # same as PET
                "pems",  # same as PEMS
            ),
        )
        """Parameter trigger_connector_mode"""
        # Pulse modulator
        self.pulsemod_delay: Parameter = self.add_parameter(
            "pulsemod_delay",
            label="Pulse delay",
            unit="s",
            get_cmd="SOUR:PULM:DEL?",
            set_cmd="SOUR:PULM:DEL {:g}",
            get_parser=float,
            vals=vals.Numbers(0, 100),
        )
        """Parameter pulsemod_delay"""
        self.pulsemod_double_delay: Parameter = self.add_parameter(
            "pulsemod_double_delay",
            label="Pulse double delay",
            unit="s",
            get_cmd="SOUR:PULM:DOUB:DEL?",
            set_cmd="SOUR:PULM:DOUB:DEL {:g}",
            get_parser=float,
            vals=vals.Numbers(40e-9, 100),
        )
        """Parameter pulsemod_double_delay"""
        self.pulsemod_double_width: Parameter = self.add_parameter(
            "pulsemod_double_width",
            label="Double pulse second width",
            unit="s",
            get_cmd="SOUR:PULM:DOUB:WIDT?",
            set_cmd="SOUR:PULM:DOUB:WIDT {:g}",
            get_parser=float,
            vals=vals.Numbers(20e-9, 100),
        )
        """Parameter pulsemod_double_width"""
        self.pulsemod_mode: Parameter = self.add_parameter(
            "pulsemod_mode",
            label="Pulse modulation mode",
            get_cmd="SOUR:PULM:MODE?",
            set_cmd="SOUR:PULM:MODE {}",
            vals=vals.Enum("SING", "DOUB", "sing", "doub", "single", "double"),
        )
        """Parameter pulsemod_mode"""
        self.pulsemod_period: Parameter = self.add_parameter(
            "pulsemod_period",
            label="Pulse mode period",
            unit="s",
            get_cmd="SOUR:PULM:PER?",
            set_cmd="SOUR:PULM:PER {:g}",
            get_parser=float,
            vals=vals.Numbers(100e-9, 100),
        )
        """Parameter pulsemod_period"""
        self.pulsemod_polarity: Parameter = self.add_parameter(
            "pulsemod_polarity",
            label="Pulse modulator signal polarity",
            get_cmd="SOUR:PULM:POL?",
            set_cmd="SOUR:PULM:POL {}",
            vals=vals.Enum("NORM", "INV", "norm", "inv", "normal", "inverted"),
        )
        """Parameter pulsemod_polarity"""
        self.pulsemod_trig_ext_gate_polarity: Parameter = self.add_parameter(
            "pulsemod_trig_ext_gate_polarity",
            label="Polarity of the Gate signal",
            get_cmd="SOUR:PULM:TRIG:EXT:GATE:POL?",
            set_cmd="SOUR:PULM:TRIG:EXT:GATE:POL {}",
            vals=vals.Enum("NORM", "INV", "norm", "inv", "normal", "inverted"),
        )
        """Parameter pulsemod_trig_ext_gate_polarity"""
        self.pulsemod_trig_ext_impedance: Parameter = self.add_parameter(
            "pulsemod_trig_ext_impedance",
            label="Impedance of the external pulse trigger",
            get_cmd="SOUR:PULM:TRIG:EXT:IMP?",
            set_cmd="SOUR:PULM:TRIG:EXT:IMP {}",
            vals=vals.Enum("G50", "G10K"),
        )
        """Parameter pulsemod_trig_ext_impedance"""
        # Sets the polarity of the active slope of an externally applied trigger signal.
        self.pulsemod_trig_ext_slope: Parameter = self.add_parameter(
            "pulsemod_trig_ext_slope",
            label="external pulse trigger active slope",
            get_cmd="SOUR:PULM:TRIG:EXT:SLOP?",
            set_cmd="SOUR:PULM:TRIG:EXT:SLOP {}",
            vals=vals.Enum("NEG", "POS", "neg", "pos", "negative", "positive"),
        )
        """Parameter pulsemod_trig_ext_slope"""
        self.pulsemod_trig_mode: Parameter = self.add_parameter(
            "pulsemod_trig_mode",
            label="external pulse trigger active slope",
            get_cmd="SOUR:PULM:TRIG:MODE?",
            set_cmd="SOUR:PULM:TRIG:MODE {}",
            vals=vals.Enum(
                "AUTO", "EXT", "EGAT", "auto", "ext", "egat", "external", "egate"
            ),
        )
        """Parameter pulsemod_trig_mode"""
        self.pulsemod_width: Parameter = self.add_parameter(
            "pulsemod_width",
            label="Pulse width",
            unit="s",
            get_cmd="SOUR:PULM:WIDT?",
            set_cmd="SOUR:PULM:WIDT {:g}",
            get_parser=float,
            vals=vals.Numbers(20e-9, 100),
        )
        """Parameter pulsemod_width"""
        self.add_function("reset", call_cmd="*RST")
        self.add_function("run_self_tests", call_cmd="*TST?")

        self.connect_message()

    def on(self) -> None:
        self.status("on")

    def off(self) -> None:
        self.status("off")


class RohdeSchwarz_SGS100A(RohdeSchwarzSGS100A):
    pass
