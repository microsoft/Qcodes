import gc
import os
import sys

import pytest
from hypothesis import settings

import qcodes as qc
from qcodes import initialise_database, new_data_set
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.experiment_container import new_experiment

settings.register_profile("ci", deadline=1000)

n_experiments = 0


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
    enabled in the test. So disable any default subscriber defined.
    """

    original_state = qc.config.subscription.default_subscribers

    try:
        qc.config.subscription.default_subscribers = []
        yield
    finally:
        qc.config.subscription.default_subscribers = original_state


@pytest.fixture(scope="function", name="empty_temp_db")
def _make_empty_temp_db(tmp_path):
    global n_experiments
    n_experiments = 0
    # create a temp database for testing
    try:
        qc.config["core"]["db_location"] = str(tmp_path / "temp.db")
        if os.environ.get("QCODES_SQL_DEBUG"):
            qc.config["core"]["db_debug"] = True
        else:
            qc.config["core"]["db_debug"] = False
        initialise_database()
        yield
    finally:
        # there is a very real chance that the tests will leave open
        # connections to the database. These will have gone out of scope at
        # this stage but a gc collection may not have run. The gc
        # collection ensures that all connections belonging to now out of
        # scope objects will be closed
        gc.collect()


@pytest.fixture(scope="function", name="experiment")
def _make_experiment(empty_temp_db):
    e = new_experiment("test-experiment", sample_name="test-sample")
    try:
        yield e
    finally:
        e.conn.close()


@pytest.fixture(scope="function", name="dataset")
def _make_dataset(experiment):
    dataset = new_data_set("test-dataset")
    try:
        yield dataset
    finally:
        dataset.unsubscribe_all()
        dataset.conn.close()


@pytest.fixture(name="standalone_parameters_dataset")
def _make_standalone_parameters_dataset(dataset):
    n_params = 3
    n_rows = 10 ** 3
    params_indep = [
        ParamSpecBase(f"param_{i}", "numeric", label=f"param_{i}", unit="V")
        for i in range(n_params)
    ]

    param_dep = ParamSpecBase(
        f"param_{n_params}", "numeric", label=f"param_{n_params}", unit="Ohm"
    )

    params_all = params_indep + [param_dep]

    idps = InterDependencies_(
        dependencies={param_dep: tuple(params_indep[0:1])},
        standalones=tuple(params_indep[1:]),
    )

    dataset.set_interdependencies(idps)

    dataset.mark_started()
    dataset.add_results(
        [
            {p.name: int(n_rows * 10 * pn + i) for pn, p in enumerate(params_all)}
            for i in range(n_rows)
        ]
    )
    dataset.mark_completed()
    yield dataset
