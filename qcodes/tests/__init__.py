import pytest
import qcodes as qc


@pytest.fixture(scope="session", autouse=True)
def disable_telemetry():
    """
    We do not want the tests to send up telemetric information. This fixture
    can be imported to disable the telemetry. Each `test_*.py` file that
    sets up telemetry should import this fixture.
    """

    original_state = qc.config.telemetry.enabled

    try:
        qc.config.telemetry.enabled = False
        yield
    finally:
        qc.config.telemetry.enabled = original_state
