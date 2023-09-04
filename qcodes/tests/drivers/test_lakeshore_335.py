import logging
import time

from qcodes.instrument import InstrumentBase
from qcodes.instrument_drivers.Lakeshore import LakeshoreModel335

from .test_lakeshore import (
    DictClass,
    MockVisaInstrument,
    command,
    instrument_fixture,
    query,
    split_args,
)

log = logging.getLogger(__name__)

VISA_LOGGER = ".".join((InstrumentBase.__module__, "com", "visa"))


class LakeshoreModel335Mock(MockVisaInstrument, LakeshoreModel335):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # initial values
        self.heaters: dict[str, DictClass] = {}
        self.heaters["1"] = DictClass(
            P=1,
            I=2,
            D=3,
            mode=1,  # 'off'
            input_channel=1,  # 'A'
            powerup_enable=0,
            polarity=0,
            use_filter=0,
            delay=1,
            output_range=0,
            setpoint=4,
        )
        self.heaters["2"] = DictClass(
            P=1,
            I=2,
            D=3,
            mode=2,  # 'closed_loop'
            input_channel=2,  # 'B'
            powerup_enable=0,
            polarity=0,
            use_filter=0,
            delay=1,
            output_range=0,
            setpoint=4,
        )

        self.channel_mock = {
            str(i): DictClass(
                t_limit=i,
                T=4,
                sensor_name=f"sensor_{i}",
                sensor_type=1,  # 'diode',
                auto_range_enabled=0,  # 'off',
                range=0,
                compensation_enabled=0,  # False,
                units=1,
            )  # 'kelvin')
            for i in self.channel_name_command.keys()
        }

        # simulate delayed heating
        self.simulate_heating = False
        self.start_heating_time = time.perf_counter()

    def start_heating(self):
        self.start_heating_time = time.perf_counter()
        self.simulate_heating = True

    def get_t_when_heating(self):
        """
        Simply define a fixed setpoint of 4 k for now
        """
        delta = abs(time.perf_counter() - self.start_heating_time)
        # make it simple to start with: linear ramp 1K per second
        # start at 7K.
        return max(4, 7 - delta)

    @query("PID?")
    def pidq(self, arg):
        heater = self.heaters[arg]
        return f"{heater.P},{heater.I},{heater.D}"

    @command("PID")
    @split_args()
    def pid(self, output, P, I, D):  # noqa  E741
        for a, v in zip(["P", "I", "D"], [P, I, D]):
            setattr(self.heaters[output], a, v)

    @query("OUTMODE?")
    def outmodeq(self, arg):
        heater = self.heaters[arg]
        return f"{heater.mode},{heater.input_channel},{heater.powerup_enable}"

    @command("OUTMODE")
    @split_args()
    def outputmode(self, output, mode, input_channel, powerup_enable):
        h = self.heaters[output]
        h.output = output
        h.mode = mode
        h.input_channel = input_channel
        h.powerup_enable = powerup_enable

    @query("INTYPE?")
    def intypeq(self, channel):
        ch = self.channel_mock[channel]
        return (
            f"{ch.sensor_type},"
            f"{ch.auto_range_enabled},{ch.range},"
            f"{ch.compensation_enabled},{ch.units}"
        )

    @command("INTYPE")
    @split_args()
    def intype(
        self,
        channel,
        sensor_type,
        auto_range_enabled,
        range_,
        compensation_enabled,
        units,
    ):
        ch = self.channel_mock[channel]
        ch.sensor_type = sensor_type
        ch.auto_range_enabled = auto_range_enabled
        ch.range = range_
        ch.compensation_enabled = compensation_enabled
        ch.units = units

    @query("RANGE?")
    def rangeq(self, heater):
        h = self.heaters[heater]
        return f"{h.output_range}"

    @command("RANGE")
    @split_args()
    def range_cmd(self, heater, output_range):
        h = self.heaters[heater]
        h.output_range = output_range

    @query("SETP?")
    def setpointq(self, heater):
        h = self.heaters[heater]
        return f"{h.setpoint}"

    @command("SETP")
    @split_args()
    def setpoint(self, heater, setpoint):
        h = self.heaters[heater]
        h.setpoint = setpoint

    @query("TLIMIT?")
    def tlimitq(self, channel):
        chan = self.channel_mock[channel]
        return f"{chan.tlimit}"

    @command("TLIMIT")
    @split_args()
    def tlimitcmd(self, channel, tlimit):
        chan = self.channel_mock[channel]
        chan.tlimit = tlimit

    @query("KRDG?")
    def temperature(self, output):
        chan = self.channel_mock[output]
        if self.simulate_heating:
            return self.get_t_when_heating()
        return f"{chan.T}"


@instrument_fixture(scope="function", name="lakeshore_335")
def _make_lakeshore_335():
    return LakeshoreModel335Mock(
        "lakeshore_335_fixture",
        "GPIB::2::INSTR",
        pyvisa_sim_file="lakeshore_model335.yaml",
        device_clear=False,
    )


def test_pid_set(lakeshore_335) -> None:
    ls = lakeshore_335
    P, I, D = 1, 2, 3  # noqa  E741
    # Only current source outputs/heaters have PID parameters,
    # voltages source outputs/heaters do not.
    outputs = [ls.output_1, ls.output_2]
    for h in outputs:  # a.k.a. heaters
        h.P(P)
        h.I(I)
        h.D(D)
        assert (h.P(), h.I(), h.D()) == (P, I, D)


def test_output_mode(lakeshore_335) -> None:
    ls = lakeshore_335
    mode = "off"
    input_channel = "A"
    powerup_enable = True
    outputs = [getattr(ls, f"output_{n}") for n in range(1, 3)]
    for h in outputs:  # a.k.a. heaters
        h.mode(mode)
        h.input_channel(input_channel)
        h.powerup_enable(powerup_enable)
        assert h.mode() == mode
        assert h.input_channel() == input_channel
        assert h.powerup_enable() == powerup_enable


def test_range(lakeshore_335) -> None:
    ls = lakeshore_335
    output_range = "medium"
    outputs = [getattr(ls, f"output_{n}") for n in range(1, 3)]
    for h in outputs:  # a.k.a. heaters
        h.output_range(output_range)
        assert h.output_range() == output_range


def test_tlimit(lakeshore_335) -> None:
    ls = lakeshore_335
    tlimit = 5.1
    for ch in ls.channels:
        ch.t_limit(tlimit)
        assert ch.t_limit() == tlimit


def test_setpoint(lakeshore_335) -> None:
    ls = lakeshore_335
    setpoint = 5.1
    outputs = [getattr(ls, f"output_{n}") for n in range(1, 3)]
    for h in outputs:  # a.k.a. heaters
        h.setpoint(setpoint)
        assert h.setpoint() == setpoint


def test_select_range_limits(lakeshore_335) -> None:
    h = lakeshore_335.output_1
    ranges = [1, 2, 3]
    h.range_limits(ranges)

    for i in ranges:
        h.set_range_from_temperature(i - 0.5)
        assert h.output_range() == h.INVERSE_RANGES[i]

    i = 3
    h.set_range_from_temperature(i + 0.5)
    assert h.output_range() == h.INVERSE_RANGES[len(ranges)]


def test_set_and_wait_unit_setpoint_reached(lakeshore_335) -> None:
    ls = lakeshore_335
    ls.output_1.setpoint(4)
    ls.start_heating()
    ls.output_1.wait_until_set_point_reached()


def test_blocking_t(lakeshore_335) -> None:
    ls = lakeshore_335
    h = ls.output_1
    ranges = [1.2, 2.4, 3.1]
    h.range_limits(ranges)
    ls.start_heating()
    h.blocking_t(4)
