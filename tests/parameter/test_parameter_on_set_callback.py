import gc
import sqlite3
import threading
import time
from collections import Counter
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import pytest

from qcodes import validators
from qcodes.parameters import Parameter
from qcodes.parameters.parameter_base import ParameterBase

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


DEFAULT_VALUE = 42
DELAY_TIME = 0.1
STEP_SIZE = 0.1
THREAD_SLEEP = 0.01


@pytest.fixture(autouse=True)  # type: ignore[misc]
def _reset_callback() -> "Generator[None, None, None]":
    """Reset the callback after each test"""
    yield
    ParameterBase.global_on_set_callback = None


@pytest.fixture()  # type: ignore[misc]
def basic_parameter(
    basic_callback: "Callable[[ParameterBase, Any], None]",
) -> Parameter:
    """Fixture providing a basic parameter with callback"""
    param = Parameter("test_param", set_cmd=None, get_cmd=None)
    ParameterBase.global_on_set_callback = basic_callback
    return param


@pytest.fixture(scope="function")  # type: ignore[misc]
def basic_callback(
    captured_params: list[tuple[ParameterBase, Any]],
) -> "Callable[[ParameterBase, Any], None]":
    """Fixture providing a standard callback function"""

    def callback(param: ParameterBase, value: Any) -> None:
        captured_params.append((param, value))

    return callback


@pytest.fixture(scope="function")  # type: ignore[misc]
def captured_params() -> list[tuple[ParameterBase, Any]]:
    """Fixture for capturing callback parameters"""
    return []


@pytest.fixture(autouse=True, scope="function")  # type: ignore[misc]
def cleanup_db_connections():
    """Clean up any open SQLite connections after each test"""
    yield
    gc.collect()

    open_connections = [
        obj for obj in gc.get_objects() if isinstance(obj, sqlite3.Connection)
    ]

    for conn in open_connections:
        try:
            conn.close()
        except Exception:
            pass

    gc.collect()


class TestBasicCallbackBehavior:
    """Tests for basic callback functionality"""

    def test_value_changed_callback(
        self,
        basic_parameter: Parameter,
        captured_params: list[tuple[ParameterBase, Any]],
    ) -> None:
        """Test basic callback functionality"""
        basic_parameter(DEFAULT_VALUE)
        assert len(captured_params) == 1
        assert captured_params[0][0] is basic_parameter
        assert captured_params[0][1] == DEFAULT_VALUE

    def test_multiple_value_changes(
        self,
        basic_parameter: Parameter,
        captured_params: list[tuple[ParameterBase, Any]],
    ) -> None:
        """Test callback is called for each value change"""
        values = [1, 1, 2]
        for val in values:
            basic_parameter(val)
        assert len(captured_params) == len(values)

    def test_set_global_callback(self) -> None:
        """Test setting and clearing global callback"""
        param = Parameter("test_param", set_cmd=None, get_cmd=None)
        captured = []

        def test_callback(p: ParameterBase, value: Any) -> None:
            captured.append((p, value))

        ParameterBase.global_on_set_callback = test_callback
        param(1)
        assert len(captured) == 1
        assert captured[0] == (param, 1)

        ParameterBase.global_on_set_callback = None
        param(2)
        assert len(captured) == 1


class TestValidationBehavior:
    """Tests for validation-related functionality"""

    @pytest.mark.parametrize(  # type: ignore[misc]
        "test_input,validator,should_callback",
        [
            pytest.param(5, validators.Numbers(0, 10), True, id="valid_number"),
            pytest.param(-1, validators.Numbers(0, 10), False, id="invalid_number"),
            pytest.param("valid", validators.Strings(), True, id="valid_string"),
            pytest.param(42, validators.Numbers(max_value=10), False, id="over_max"),
        ],
    )
    def test_callback_with_different_validators(
        self,
        captured_params: list[tuple[ParameterBase, Any]],
        basic_callback: "Callable[[ParameterBase, Any], None]",
        test_input: Any,
        validator: Any,
        should_callback: bool,
    ) -> None:
        """Test callback behavior with different validator types"""
        param = Parameter("test_param", set_cmd=None, get_cmd=None, vals=validator)
        ParameterBase.global_on_set_callback = basic_callback

        with pytest.raises(ValueError) if not should_callback else nullcontext():
            param(test_input)

        assert bool(len(captured_params)) == should_callback


class TestErrorHandling:
    """Tests for error handling and edge cases"""

    def test_callback_exception_handling(self) -> None:
        """Test that callback exceptions are handled gracefully"""

        def failing_callback(param: ParameterBase, value: Any) -> None:
            raise RuntimeError("Intentional failure")

        param = Parameter("test_param", set_cmd=None, get_cmd=None)
        ParameterBase.global_on_set_callback = failing_callback

        param(DEFAULT_VALUE)
        assert param() == DEFAULT_VALUE

    def test_callback_with_none_value(
        self,
        basic_parameter: Parameter,
        captured_params: list[tuple[ParameterBase, Any]],
    ) -> None:
        """Test handling of None values"""
        basic_parameter(None)

        assert len(captured_params) == 1, "Should handle None value"
        assert captured_params[0][1] is None, "Should capture None value correctly"


class TestAdvancedFeatures:
    """Tests for advanced parameter features"""

    def test_callback_thread_safety(
        self,
    ) -> None:
        """Test thread safety of callbacks

        Tests concurrent parameter updates using multiple threads,
        ensuring all callbacks are executed correctly.
        """
        NUM_THREADS = 2
        TEST_VALUES = [1, 2]
        captured_values = []

        lock = threading.Lock()

        def thread_safe_callback(param: ParameterBase, value: Any) -> None:
            time.sleep(THREAD_SLEEP)
            with lock:
                captured_values.append(value)

        param = Parameter("test_param", set_cmd=None, get_cmd=None)
        ParameterBase.global_on_set_callback = thread_safe_callback

        threads = [
            threading.Thread(
                target=lambda: [param(val) for val in TEST_VALUES],
                name=f"CallbackThread-{i}",
            )
            for i in range(NUM_THREADS)
        ]

        [t.start() for t in threads]
        [t.join() for t in threads]

        value_counts = Counter(captured_values)
        expected_count = NUM_THREADS * len(TEST_VALUES)

        assert len(captured_values) == expected_count, (
            f"Expected {expected_count} callback captures, got {len(captured_values)}"
        )
        assert all(count == NUM_THREADS for count in value_counts.values()), (
            f"Uneven value distribution: {dict(value_counts)}"
        )

    def test_callback_with_steps(
        self,
        basic_callback: "Callable[[ParameterBase, Any], None]",
        captured_params: list[tuple[ParameterBase, Any]],
    ) -> None:
        """Test stepped parameter setting

        Verifies that parameters with step values correctly trigger
        callbacks for each intermediate step.
        """
        START_VALUE = 0.0
        TARGET_VALUE = 0.3
        expected_steps = [0.1, 0.2, 0.3]

        param = Parameter(
            name="test_param",
            set_cmd=None,
            get_cmd=None,
            step=STEP_SIZE,
            initial_value=START_VALUE,
        )
        ParameterBase.global_on_set_callback = basic_callback

        param(TARGET_VALUE)

        actual_values = [val[1] for val in captured_params]
        assert len(actual_values) == len(expected_steps), (
            f"Expected {len(expected_steps)} steps, got {len(actual_values)}"
        )
        assert actual_values == expected_steps, (
            f"Expected steps {expected_steps}, got {actual_values}"
        )

    def test_nested_callbacks(self) -> None:
        """Test nested callback behavior"""
        param = Parameter("test_param", set_cmd=None, get_cmd=None)

        def callback(param: ParameterBase, value: Any) -> None:
            param.cache.set(value)

        ParameterBase.global_on_set_callback = callback
        param(1)
        assert param.cache.get() == 1

    def test_callback_with_delay(
        self,
        basic_parameter: Parameter,
    ) -> None:
        """Test delayed parameter setting"""
        captured_times = []
        start_time = time.time()

        def timing_callback(param: ParameterBase, value: Any) -> None:
            captured_times.append(time.time() - start_time)

        basic_parameter.post_delay = DELAY_TIME
        ParameterBase.global_on_set_callback = timing_callback

        basic_parameter(1)
        basic_parameter(2)

        assert len(captured_times) == 2
        assert captured_times[1] - captured_times[0] >= DELAY_TIME


def test_set_callback_for_instance(
    basic_callback: "Callable[[ParameterBase, Any], None]",
    captured_params: list[tuple[ParameterBase, Any]],
):
    param_a = Parameter("test_param_a", set_cmd=None, get_cmd=None)
    param_b = Parameter("test_param_b", set_cmd=None, get_cmd=None)
    captured_instance_params = []

    def callback(param: ParameterBase, val):
        if ParameterBase.global_on_set_callback:
            ParameterBase.global_on_set_callback(param, val)
        captured_instance_params.append(val)

    ParameterBase.global_on_set_callback = basic_callback
    param_a.on_set_callback = callback
    param_a(1)
    param_b(2)

    assert captured_params == [(param_a, 1), (param_b, 2)]
    assert captured_instance_params == [1]
