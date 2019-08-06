import pytest
from qcodes.config import Config


@pytest.fixture(scope="session", autouse=True)
def disable_telemetry():
    """
    There are certain things that we do not want our tests to do
    """
    conf = Config()
    old_telemetry_state = conf.telemetry.enabled
    if old_telemetry_state:
        conf.telemetry.enabled = False
        conf.save_to_home()

    try:
        yield
    finally:
        if old_telemetry_state:
            conf.telemetry.enabled = True
            conf.save_to_home()
