import pytest
import qcodes as qc


@pytest.fixture(scope="session", autouse=True)
def disable_telemetry():
    """
    We do not want the tests to send up telemetric information, so we disable
    that with this fixture.
    """

    original_state = qc.config.telemetry.enabled

    try:
        qc.config.telemetry.enabled = False
        yield
    finally:
        qc.config.telemetry.enabled = original_state
