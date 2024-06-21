from unittest.mock import MagicMock, patch

import pytest

from qcodes.dataset.dond.do_nd import dond
from qcodes.dataset.dond.do_nd_utils import BreakConditionInterrupt, catch_interrupts


def test_catch_interrupts():
    # Test normal execution (no interrupt)
    with catch_interrupts() as get_interrupt:
        assert get_interrupt() is None

    # Test KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        with catch_interrupts():
            raise KeyboardInterrupt()

    # Test BreakConditionInterrupt
    with pytest.raises(BreakConditionInterrupt):
        with catch_interrupts():
            raise BreakConditionInterrupt()

    # Test that cleanup code runs before re-raising
    cleanup_ran = False
    with pytest.raises(KeyboardInterrupt):
        with catch_interrupts():
            try:
                raise KeyboardInterrupt()
            finally:
                cleanup_ran = True
    assert cleanup_ran


def test_catch_interrupts_in_loops():
    # Test interruption in a simple loop
    loop_count = 0
    with pytest.raises(KeyboardInterrupt):
        for i in range(5):
            with catch_interrupts():
                loop_count += 1
                if i == 2:
                    raise KeyboardInterrupt()
    assert loop_count == 3  # Loop should stop at the third iteration

    # Test interruption in nested loops
    outer_count = 0
    inner_count = 0
    with pytest.raises(KeyboardInterrupt):
        for i in range(3):
            with catch_interrupts():
                outer_count += 1
                for j in range(3):
                    with catch_interrupts():
                        inner_count += 1
                        if i == 1 and j == 1:
                            raise KeyboardInterrupt()
    assert outer_count == 2
    assert inner_count == 5


def test_catch_interrupts_simulated_sweeps():
    def simulated_sweep(interrupt_at=None):
        for i in range(5):
            with catch_interrupts():
                if i == interrupt_at:
                    raise KeyboardInterrupt()
                yield i

    # Test interruption in a single sweep
    results = []
    with pytest.raises(KeyboardInterrupt):
        for value in simulated_sweep(interrupt_at=3):
            results.append(value)
    assert results == [0, 1, 2]

    # Test interruption in nested sweeps
    outer_results = []
    inner_results = []
    with pytest.raises(KeyboardInterrupt):
        for outer_value in simulated_sweep(interrupt_at=None):
            outer_results.append(outer_value)
            for inner_value in simulated_sweep(
                interrupt_at=2 if outer_value == 1 else None
            ):
                inner_results.append(inner_value)
    assert outer_results == [0, 1]
    assert inner_results == [0, 1, 2, 3, 4, 0, 1]


@pytest.fixture
def mock_dond_dependencies():
    with (
        patch("qcodes.dataset.dond.do_nd._Sweeper") as mock_sweeper,
        patch("qcodes.dataset.dond.do_nd._Measurements") as mock_measurements,
    ):

        mock_sweeper_instance = MagicMock()
        mock_sweeper_instance.__iter__.return_value = iter([MagicMock()])
        mock_sweeper.return_value = mock_sweeper_instance

        mock_measurements_instance = MagicMock()
        mock_measurements_instance.groups = [MagicMock()]
        mock_measurements.return_value = mock_measurements_instance

        yield


def test_dond_interruptible(mock_dond_dependencies):
    interrupt_raised = False

    def simulated_interrupt(*args, **kwargs):
        nonlocal interrupt_raised
        interrupt_raised = True
        raise KeyboardInterrupt()

    # Mock the catch_interrupts context manager
    mock_catch_interrupts = MagicMock()
    mock_catch_interrupts.__enter__.return_value = lambda: None
    mock_catch_interrupts.__exit__.side_effect = simulated_interrupt

    with patch(
        "qcodes.dataset.dond.do_nd.catch_interrupts", return_value=mock_catch_interrupts
    ):
        with pytest.raises(KeyboardInterrupt):
            dond(MagicMock(), MagicMock())

    assert interrupt_raised, "KeyboardInterrupt was not raised"
