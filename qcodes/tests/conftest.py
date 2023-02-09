from __future__ import annotations

import copy
import gc
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import pytest
from hypothesis import settings

import qcodes as qc
from qcodes.configuration import Config
from qcodes.dataset import initialise_database, new_data_set
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.experiment_container import Experiment, new_experiment
from qcodes.station import Station

settings.register_profile("ci", deadline=1000)

n_experiments = 0

if TYPE_CHECKING:
    from qcodes.configuration import DotDict

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "win32: tests that only run under windows")


def pytest_runtest_setup(item: pytest.Item) -> None:
    ALL = set("darwin linux win32".split())
    supported_platforms = ALL.intersection(mark.name for mark in item.iter_markers())
    if supported_platforms and sys.platform not in supported_platforms:
        pytest.skip(f"cannot run on platform {sys.platform}")


@pytest.fixture(scope="session", autouse=True)
def disable_telemetry() -> Generator[None, None, None]:
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


@pytest.fixture(scope="function")
def default_config(tmp_path: Path) -> Generator[None, None, None]:
    """
    Fixture to temporarily establish default config settings.
    This is achieved by overwriting the config paths of the user-,
    environment-, and current directory-config files with the path of the
    config file in the qcodes repository,
    additionally the current config object `qcodes.config` gets copied and
    reestablished.
    """
    home_file_name = Config.home_file_name
    schema_home_file_name = Config.schema_home_file_name
    env_file_name = Config.env_file_name
    schema_env_file_name = Config.schema_env_file_name
    cwd_file_name = Config.cwd_file_name
    schema_cwd_file_name = Config.schema_cwd_file_name

    file_name = str(tmp_path / "user_config.json")
    file_name_schema = str(tmp_path / "user_config_schema.json")

    Config.home_file_name = file_name
    Config.schema_home_file_name = file_name_schema
    Config.env_file_name = ""
    Config.schema_env_file_name = ""
    Config.cwd_file_name = ""
    Config.schema_cwd_file_name = ""

    default_config_obj: DotDict | None = copy.deepcopy(qc.config.current_config)
    qc.config = Config()

    try:
        yield
    finally:
        Config.home_file_name = home_file_name
        Config.schema_home_file_name = schema_home_file_name
        Config.env_file_name = env_file_name
        Config.schema_env_file_name = schema_env_file_name
        Config.cwd_file_name = cwd_file_name
        Config.schema_cwd_file_name = schema_cwd_file_name

        qc.config.current_config = default_config_obj


@pytest.fixture(scope="function")
def reset_config_on_exit() -> Generator[None, None, None]:

    """
    Fixture to clean any modification of the in memory config on exit

    """
    default_config_obj: DotDict | None = copy.deepcopy(qc.config.current_config)

    try:
        yield
    finally:
        qc.config.current_config = default_config_obj


@pytest.fixture(scope="session", autouse=True)
def disable_config_subscriber() -> Generator[None, None, None]:
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
def _make_empty_temp_db(tmp_path: Path) -> Generator[None, None, None]:
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


# note that you cannot use mark.usefixtures in a fixture
# so empty_temp_db needs to be passed to this fixture
# even if unused https://github.com/pytest-dev/pytest/issues/3664
@pytest.fixture(scope="function", name="experiment")
def _make_experiment(empty_temp_db: None) -> Generator[Experiment, None, None]:
    e = new_experiment("test-experiment", sample_name="test-sample")
    try:
        yield e
    finally:
        e.conn.close()


@pytest.fixture(scope="function", name="dataset")
def _make_dataset(experiment: Experiment) -> Generator[DataSet, None, None]:
    dataset = new_data_set("test-dataset")
    try:
        yield dataset
    finally:
        dataset.unsubscribe_all()
        dataset.conn.close()


@pytest.fixture(name="standalone_parameters_dataset")
def _make_standalone_parameters_dataset(
    dataset: DataSet,
) -> Generator[DataSet, None, None]:
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


@pytest.fixture(name="set_default_station_to_none")
def _make_set_default_station_to_none():
    """Makes sure that after startup and teardown there is no default station"""
    Station.default = None
    yield
    Station.default = None
