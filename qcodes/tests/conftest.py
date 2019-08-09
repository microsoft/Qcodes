import pytest
import qcodes as qc


@pytest.fixture(scope="session", autouse=True)
def disable_config_subscriber():
    """
    We do not want the tests to send generate subscription events unless specifically
    enabled in the test. So disable any default subscriber defined
    """

    original_state = qc.config.subscription.default_subscribers

    try:
        qc.config.subscription.default_subscribers = []
        yield
    finally:
        qc.config.subscription.default_subscribers = original_state
