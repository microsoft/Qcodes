import logging

import pytest
from pytest import FixtureRequest, LogCaptureFixture

from qcodes.instrument_drivers.tektronix.Keithley_s46 import S46, LockAcquisitionError


def test_aliases_dict() -> None:
    """
    Test the class attribute 'aliases' which maps channel aliases
    (e.g. A1, B2, etc) to channel numbers.
    """

    def calc_channel_nr(alias: str) -> int:
        """
        We perform the calculation in a different way to verify correctness
        """
        offset_dict = dict(zip(["A", "B", "C", "D", "R"], range(0, 32, 6)))
        return offset_dict[alias[0]] + int(alias[1:])

    assert all([nr == calc_channel_nr(al) for al, nr in S46.channel_numbers.items()])


@pytest.fixture(scope="function")
def s46_six():
    """
    A six channel-per-relay instrument
    """
    driver = S46(
        "s46_six", address="GPIB::2::INSTR", pyvisa_sim_file="Keithley_s46.yaml"
    )

    try:
        yield driver
    finally:
        driver.close()


@pytest.fixture(scope="function")
def s46_four():
    """
    A four channel-per-relay instrument
    """
    driver = S46(
        "s46_four", address="GPIB::3::INSTR", pyvisa_sim_file="Keithley_s46.yaml"
    )

    try:
        yield driver
    finally:
        driver.close()


def test_runtime_error_on_bad_init(request: FixtureRequest) -> None:
    """
    If we initialize the driver from an instrument state with more then one
    channel per relay closed, raise a runtime error. An instrument can come to
    this state if previously, other software was used to control the instrument
    """
    request.addfinalizer(S46.close_all)

    with pytest.raises(
        RuntimeError,
        match="The driver is initialized from an undesirable instrument state",
    ):
        S46(
            "s46_bad_state",
            address="GPIB::1::INSTR",
            pyvisa_sim_file="Keithley_s46.yaml",
        )


def test_query_close_once_at_init(caplog: LogCaptureFixture) -> None:
    """
    Test that, during initialisation, we query the closed channels only once
    """
    with caplog.at_level(logging.DEBUG):
        inst = S46(
            "s46_test_query_once",
            address="GPIB::2::INSTR",
            pyvisa_sim_file="Keithley_s46.yaml",
        )
        assert caplog.text.count("Querying: :CLOS?") == 1
        inst.close()


def test_init_six(s46_six: S46, caplog: LogCaptureFixture) -> None:
    """
    Test that the six channel instrument initializes correctly.
    """
    assert len(s46_six.available_channels) == 26

    closed_channel_numbers = [1, 8, 13]
    assert s46_six.closed_channels() == [S46.aliases[i] for i in closed_channel_numbers]

    with caplog.at_level(logging.DEBUG):
        s46_six.open_all_channels()
        assert ":open (@1)" in caplog.text
        assert ":open (@8)" in caplog.text
        assert ":open (@13)" in caplog.text

        assert s46_six.A1._lock._locked_by is None
        assert s46_six.B1._lock._locked_by is None
        assert s46_six.C1._lock._locked_by is None


def test_init_four(s46_four: S46) -> None:
    """
    Test that the six channel instrument initializes correctly.
    """
    assert len(s46_four.available_channels) == 18

    closed_channel_numbers = [1, 8]
    assert s46_four.closed_channels() == [
        S46.aliases[i] for i in closed_channel_numbers
    ]

    # A four channel instrument will have channels missing
    for relay in ["A", "B", "C", "D"]:
        for index in [5, 6]:
            alias = f"{relay}{index}"
            assert not hasattr(s46_four, alias)


def test_channel_number_invariance(s46_four: S46, s46_six: S46) -> None:
    """
    Regardless of the channel layout (that is, number of channels per relay),
    channel aliases should represent the same channel. See also page 2-5 of the
    manual (e.g. B1 is *always* channel 7)
    """
    for alias in S46.channel_numbers.keys():
        if hasattr(s46_four, alias) and hasattr(s46_six, alias):
            channel_four = getattr(s46_four, alias)
            channel_six = getattr(s46_six, alias)
            assert channel_four.channel_number == channel_six.channel_number


def test_locking_mechanism(s46_six: S46) -> None:
    """
    1) Test that the lock acquisition error is raised if we try to close
    more then once channel per replay
    2) Test that the lock is released when opening a channel that was closed
    """
    s46_six.A1("close")

    with pytest.raises(LockAcquisitionError, match="is already in use by channel"):
        # A1 should be closed already
        s46_six.A2("close")
    # release the lock
    s46_six.A1("open")
    # now we should be able to close A2
    s46_six.A2("close")

    # Let C1 acquire the lock
    s46_six.C1("close")
    # closing C2 should raise an error
    with pytest.raises(LockAcquisitionError, match="is already in use by channel"):
        s46_six.C2("close")

    # Upon opening C1 we should be able to close C2
    s46_six.C1("open")
    s46_six.C2("close")


def test_is_closed(s46_six: S46) -> None:
    """
    Test the `is_closed` public method
    """
    assert s46_six.A1.is_closed()
    assert s46_six.B2.is_closed()
    assert s46_six.C1.is_closed()

    assert not s46_six.A2.is_closed()
    assert not s46_six.B4.is_closed()
    assert not s46_six.C6.is_closed()
