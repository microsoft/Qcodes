import sys
import pytest
import qcodes as qc


def pytest_configure(config):
    config.addinivalue_line("markers", "win32: tests that only run under windows")


def pytest_runtest_setup(item):
    ALL = set("darwin linux win32".split())
    supported_platforms = ALL.intersection(mark.name for mark in item.iter_markers())
    if supported_platforms and sys.platform not in supported_platforms:
        pytest.skip(f"cannot run on platform {sys.platform}")


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


def pytest_addoption(parser):
    parser.addoption(
        "-E",
        action="store",
        metavar="NAME",
        help="only run tests marked with 'env(NAME)' to run in environment NAME.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "env(name): mark test to run only on named environment"
    )


def pytest_runtest_setup(item):
    env_names = [mark.args[0] for mark in item.iter_markers(name="env")]
    if env_names:
        env_from_option = item.config.getoption("-E")
        if env_from_option not in env_names:
            pytest.skip(
                f"Not running in {env_names!r} environment(s). "
                f"Use '-E' command line option to run this test."
            )
