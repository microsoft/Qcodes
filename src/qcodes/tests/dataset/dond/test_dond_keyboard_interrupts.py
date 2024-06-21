# from collections.abc import Iterator
# from typing import Callable, Dict, List
#
# import pytest
#
# from qcodes.dataset.dond.do_nd_utils import (
#    BreakConditionInterrupt,
#    catch_interrupts,
# )
#
# MeasInterruptT = KeyboardInterrupt | BreakConditionInterrupt | None
#
#
# def test_catch_interrupts() -> None:
#    """
#    Test the basic functionality of the catch_interrupts context manager.
#
#    This test covers:
#    1. Normal execution without interrupts.
#    2. Catching and re-raising KeyboardInterrupt.
#    3. Catching and re-raising BreakConditionInterrupt.
#    4. Accessibility of the interrupt within the context.
#    5. Execution of cleanup code before re-raising.
#    6. Behavior with nested catch_interrupts contexts.
#    """
#    # Test normal execution (no interrupt)
#    with catch_interrupts() as get_interrupt:
#        assert get_interrupt() is None
#
#    # Test KeyboardInterrupt
#    with pytest.raises(KeyboardInterrupt):
#        with catch_interrupts() as get_interrupt:
#            raise KeyboardInterrupt()
#
#    # Test BreakConditionInterrupt
#    with pytest.raises(BreakConditionInterrupt):
#        with catch_interrupts() as get_interrupt:
#            raise BreakConditionInterrupt()
#
#    # Test that the interrupt is accessible within the context
#    def nested_function() -> None:
#        with catch_interrupts() as get_interrupt:
#            raise KeyboardInterrupt()
#
#    with pytest.raises(KeyboardInterrupt):
#        try:
#            nested_function()
#        except Exception:
#            assert isinstance(get_interrupt(), KeyboardInterrupt)
#            raise
#
#    # Test that cleanup code runs before re-raising
#    cleanup_ran = False
#    with pytest.raises(KeyboardInterrupt):
#        with catch_interrupts() as get_interrupt:
#            try:
#                raise KeyboardInterrupt()
#            finally:
#                cleanup_ran = True
#    assert cleanup_ran
#
#    # Test nested interrupts
#    with pytest.raises(KeyboardInterrupt):
#        with catch_interrupts() as outer_get_interrupt:
#            with catch_interrupts() as inner_get_interrupt:
#                raise KeyboardInterrupt()
#            assert isinstance(inner_get_interrupt(), KeyboardInterrupt)
#        assert isinstance(outer_get_interrupt(), KeyboardInterrupt)
#
#
# def test_catch_interrupts_in_loops() -> None:
#    """
#    Test the behavior of catch_interrupts in simple and nested loops.
#
#    This test ensures that:
#    1. A simple loop stops at the correct iteration when interrupted.
#    2. Nested loops handle interruptions correctly, stopping both inner and outer loops.
#    """
#    # Test interruption in a simple loop
#    loop_count = 0
#    with pytest.raises(KeyboardInterrupt):
#        for i in range(5):
#            with catch_interrupts() as get_interrupt:
#                loop_count += 1
#                if i == 2:
#                    raise KeyboardInterrupt()
#    assert loop_count == 3  # Loop should stop at the third iteration
#
#    # Test interruption in nested loops
#    outer_count = 0
#    inner_count = 0
#    with pytest.raises(KeyboardInterrupt):
#        for i in range(3):
#            with catch_interrupts() as outer_get_interrupt:
#                outer_count += 1
#                for j in range(3):
#                    with catch_interrupts() as inner_get_interrupt:
#                        inner_count += 1
#                        if i == 1 and j == 1:
#                            raise KeyboardInterrupt()
#    assert outer_count == 2  # Outer loop should stop at the second iteration
#    assert (
#        inner_count == 5
#    )  # Inner loop should run 3 times in first outer iteration, and 2 times in second
#
#
# def test_catch_interrupts_simulated_sweeps() -> None:
#    """
#    Test catch_interrupts behavior in simulated measurement sweeps.
#
#    This test simulates:
#    1. A single sweep with an interruption.
#    2. Nested sweeps with an interruption in the inner sweep.
#    It verifies that the sweeps stop at the correct points when interrupted.
#    """
#
#    def simulated_sweep(interrupt_at: int | None = None) -> Iterator[int]:
#        for i in range(5):
#            with catch_interrupts() as get_interrupt:
#                if i == interrupt_at:
#                    raise KeyboardInterrupt()
#                yield i
#
#    # Test interruption in a single sweep
#    results: List[int] = []
#    with pytest.raises(KeyboardInterrupt):
#        for value in simulated_sweep(interrupt_at=3):
#            results.append(value)
#    assert results == [0, 1, 2]
#
#    # Test interruption in nested sweeps
#    outer_results: List[int] = []
#    inner_results: List[int] = []
#    with pytest.raises(KeyboardInterrupt):
#        for outer_value in simulated_sweep(interrupt_at=None):
#            outer_results.append(outer_value)
#            for inner_value in simulated_sweep(
#                interrupt_at=2 if outer_value == 1 else None
#            ):
#                inner_results.append(inner_value)
#    assert outer_results == [0, 1]
#    assert inner_results == [0, 1, 2, 3, 4, 0, 1]
#
#
# def test_catch_interrupts_with_cleanup() -> None:
#    """
#    Test catch_interrupts behavior with cleanup in nested simulated sweeps.
#
#    This test ensures that:
#    1. Cleanup code runs the correct number of times for both inner and outer sweeps.
#    2. The interruption is properly propagated while still allowing cleanup to occur.
#    """
#    cleanup_counts: Dict[str, int] = {"outer": 0, "inner": 0}
#
#    def simulated_sweep_with_cleanup(
#        level: str, interrupt_at: int | None = None
#    ) -> Iterator[int]:
#        for i in range(5):
#            with catch_interrupts() as get_interrupt:
#                try:
#                    if i == interrupt_at:
#                        raise KeyboardInterrupt()
#                    yield i
#                finally:
#                    cleanup_counts[level] += 1
#
#    # Test cleanup in nested sweeps with interruption
#    with pytest.raises(KeyboardInterrupt):
#        for outer_value in simulated_sweep_with_cleanup("outer", interrupt_at=None):
#            for inner_value in simulated_sweep_with_cleanup(
#                "inner", interrupt_at=2 if outer_value == 1 else None
#            ):
#                pass
#
#    assert cleanup_counts["outer"] == 2  # Outer cleanup should run twice
#    assert (
#        cleanup_counts["inner"] == 7
#    )  # Inner cleanup should run 5 times for first outer loop, 2 times for second
#
#
# from collections.abc import Generator
# from contextlib import contextmanager
# from typing import Optional
# from unittest.mock import MagicMock, patch
#
# from qcodes.dataset.dond import do_nd_utils
# from qcodes.dataset.dond.do_nd import dond
# from qcodes.dataset.dond.do_nd_utils import BreakConditionInterrupt
#
#
# @pytest.fixture
# def mock_measurement() -> Callable[[Optional[int]], Generator[int, None, None]]:
#    """
#    Fixture that returns a mock measurement function.
#
#    Returns:
#        A function that simulates a measurement, optionally raising a KeyboardInterrupt.
#    """
#
#    def _mock_measurement(
#        interrupt_after: Optional[int] = None,
#    ) -> Generator[int, None, None]:
#        for i in range(5):
#            if interrupt_after is not None and i == interrupt_after:
#                raise KeyboardInterrupt
#            yield i
#
#    return _mock_measurement
#
#
# def test_current_behavior(
#    mock_measurement: Callable[[Optional[int]], Generator[int, None, None]]
# ) -> None:
#    """
#    Test the current behavior of catch_interrupts.
#
#    Args:
#        mock_measurement: A fixture providing a mock measurement function.
#    """
#    measurements_completed = 0
#
#    for _ in range(3):
#        try:
#            for value in mock_measurement(interrupt_after=2):
#                with catch_interrupts() as get_interrupt:
#                    if value == 2:
#                        raise KeyboardInterrupt
#        except KeyboardInterrupt:
#            # This is now expected behavior
#            pass  # We don't assert on get_interrupt() anymore
#        measurements_completed += 1
#
#    assert (
#        measurements_completed == 3
#    ), "All measurements should complete despite interrupts"
#
#
# @pytest.fixture
# def interruptible_catch_interrupts() -> Callable[[], contextmanager]:
#    """
#    Fixture that returns an interruptible version of catch_interrupts.
#
#    Returns:
#        A context manager that simulates an interruptible catch_interrupts.
#    """
#
#    @contextmanager
#    def _interruptible_catch_interrupts() -> (
#        Iterator[Callable[[], Optional[KeyboardInterrupt]]]
#    ):
#        interrupt_exception: Optional[KeyboardInterrupt] = None
#
#        def get_interrupt_exception() -> Optional[KeyboardInterrupt]:
#            return interrupt_exception
#
#        try:
#            yield get_interrupt_exception
#        except KeyboardInterrupt as e:
#            interrupt_exception = e
#            raise
#
#    return _interruptible_catch_interrupts
#
#
# @pytest.mark.parametrize("interrupt_after", [1, 2, 3])
# def test_interruptible_behavior(
#    mock_measurement: Callable[[Optional[int]], Generator[int, None, None]],
#    interruptible_catch_interrupts: Callable[[], contextmanager],
#    interrupt_after: int,
# ) -> None:
#    """
#    Test the behavior of an interruptible version of catch_interrupts.
#
#    Args:
#        mock_measurement: A fixture providing a mock measurement function.
#        interruptible_catch_interrupts: A fixture providing an interruptible catch_interrupts.
#        interrupt_after: The point at which to interrupt the measurement.
#    """
#    with patch(
#        "qcodes.dataset.dond.do_nd_utils.catch_interrupts",
#        interruptible_catch_interrupts,
#    ):
#        measurements_completed = 0
#
#        with pytest.raises(KeyboardInterrupt):
#            for _ in range(3):
#                for value in mock_measurement(interrupt_after=interrupt_after):
#                    with catch_interrupts():
#                        if value == interrupt_after:
#                            raise KeyboardInterrupt
#                measurements_completed += 1
#
#        assert measurements_completed < 3, "Measurements should be interrupted"
#
#
# @pytest.fixture
# def mock_dond_dependencies() -> Generator[None, None, None]:
#    """
#    Fixture that mocks the dependencies of the dond function.
#
#    Yields:
#        None
#    """
#    with (
#        patch("qcodes.dataset.dond.do_nd._Sweeper") as mock_sweeper,
#        patch("qcodes.dataset.dond.do_nd._Measurements") as mock_measurements,
#    ):
#
#        mock_sweeper_instance = MagicMock()
#        mock_sweeper_instance.__iter__.return_value = iter([MagicMock()])
#        mock_sweeper.return_value = mock_sweeper_instance
#
#        mock_measurements_instance = MagicMock()
#        mock_measurements_instance.groups = [MagicMock()]
#        mock_measurements.return_value = mock_measurements_instance
#
#        yield
#
#
# def test_dond_interruptible(mock_dond_dependencies: None) -> None:
#    """
#    Test that dond function handles interrupts correctly.
#
#    Args:
#        mock_dond_dependencies: A fixture that mocks dond dependencies.
#    """
#    interrupt_flag = False
#
#    def simulated_interrupt() -> None:
#        nonlocal interrupt_flag
#        if not interrupt_flag:
#            interrupt_flag = True
#            raise KeyboardInterrupt()
#
#    with patch.object(do_nd_utils, "catch_interrupts", side_effect=simulated_interrupt):
#        try:
#            dond(MagicMock(), MagicMock())
#        except BreakConditionInterrupt:
#            # This is the expected behavior when dond handles a KeyboardInterrupt
#            pass
#        else:
#            pytest.fail("dond did not raise BreakConditionInterrupt as expected")
#
#    assert interrupt_flag, "The interrupt was not triggered"
#
#
# import time
#
# import pytest
#
# from qcodes.dataset import (
#    LinSweep,
#    dond,
#    initialise_or_create_database_at,
#    load_or_create_experiment,
# )
# from qcodes.instrument_drivers.mock_instruments import (
#    DummyInstrument,
#    DummyInstrumentWithMeasurement,
# )
#
#
# @pytest.fixture(scope="module")
# def setup_experiment():
#    initialise_or_create_database_at("test_interrupts.db")
#    exp = load_or_create_experiment("interrupt_test", sample_name="test_sample")
#    dac = DummyInstrument("dac", gates=["ch1", "ch2"])
#    dmm = DummyInstrumentWithMeasurement("dmm", setter_instr=dac)
#    yield exp, dac, dmm
#    dac.close()
#    dmm.close()
#
#
# def run_measurement_series(exp, dac, dmm, interrupt_after=None):
#    measurements_completed = 0
#    for i, h1_val in enumerate((-0.45, -0.475, -0.5, -0.525, -0.55)):
#        if interrupt_after is not None and i == interrupt_after:
#            raise KeyboardInterrupt()
#
#        dac.ch1(h1_val)
#        sweep = LinSweep(dac.ch2, -0.2, 0.8, 25, 0.1)
#        meas_name = f"ch1_{h1_val}_ch2_sweep"
#
#        dond(
#            sweep,
#            dmm.v1,
#            dmm.v2,
#            exp=exp,
#            measurement_name=meas_name,
#            do_plot=False,
#            show_progress=False,
#        )
#        measurements_completed += 1
#        time.sleep(0.1)  # Reduced sleep time for faster tests
#
#    return measurements_completed
#
#
# def test_current_behavior(setup_experiment):
#    exp, dac, dmm = setup_experiment
#
#    with patch.object(
#        do_nd_utils, "catch_interrupts", side_effect=do_nd_utils.catch_interrupts
#    ):
#        measurements_completed = run_measurement_series(
#            exp, dac, dmm, interrupt_after=2
#        )
#
#    assert (
#        measurements_completed == 5
#    ), "All measurements should complete despite interrupts"
#
#
# def test_desired_interruptible_behavior(setup_experiment):
#    exp, dac, dmm = setup_experiment
#
#    def interruptible_catch_interrupts():
#        try:
#            yield lambda: None
#        except KeyboardInterrupt:
#            raise
#
#    with patch.object(
#        do_nd_utils, "catch_interrupts", side_effect=interruptible_catch_interrupts
#    ):
#        with pytest.raises(KeyboardInterrupt):
#            run_measurement_series(exp, dac, dmm, interrupt_after=2)
#
#    # Check that only part of the measurements were completed
#    # You might need to adjust this based on how quickly your measurements run
#    assert exp.last_counter < 5, "Measurements should be interrupted before completion"
#
#
# if __name__ == "__main__":
#    pytest.main([__file__])
#
