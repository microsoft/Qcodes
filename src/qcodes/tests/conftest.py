from __future__ import annotations

import copy
import gc
import os
import sys
from typing import TYPE_CHECKING

import pytest
from hypothesis import settings

import qcodes as qc
from qcodes.configuration import Config
from qcodes.dataset import initialise_database, new_data_set
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.experiment_container import Experiment, new_experiment
from qcodes.instrument import Instrument
from qcodes.station import Station

settings.register_profile("ci", deadline=1000)

n_experiments = 0

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from qcodes.configuration import DotDict
    from qcodes.dataset.data_set import DataSet


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "win32: tests that only run under windows")


def pytest_runtest_setup(item: pytest.Item) -> None:
    ALL = set("darwin linux win32".split())
    supported_platforms = ALL.intersection(mark.name for mark in item.iter_markers())
    if supported_platforms and sys.platform not in supported_platforms:
        pytest.skip(f"cannot run on platform {sys.platform}")


@pytest.fixture(scope="session", autouse=True)
def default_session_config(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None, None, None]:
    """
    Set the config for the test session to be the default config.
    Making sure that that user config does not influence the tests and
    that tests cannot write to the user config.
    """
    home_file_name = Config.home_file_name
    schema_home_file_name = Config.schema_home_file_name
    env_file_name = Config.env_file_name
    schema_env_file_name = Config.schema_env_file_name
    cwd_file_name = Config.cwd_file_name
    schema_cwd_file_name = Config.schema_cwd_file_name

    old_config: DotDict | None = copy.deepcopy(qc.config.current_config)
    qc.config.current_config = copy.deepcopy(qc.config.defaults)

    tmp_path = tmp_path_factory.mktemp("qcodes_tests")

    file_name = str(tmp_path / "user_config.json")
    file_name_schema = str(tmp_path / "user_config_schema.json")

    qc.config.home_file_name = file_name
    qc.config.schema_home_file_name = file_name_schema
    qc.config.env_file_name = ""
    qc.config.schema_env_file_name = ""
    qc.config.cwd_file_name = ""
    qc.config.schema_cwd_file_name = ""

    # set any config that we want to be different from the default
    # for the test session here
    # also set the default db path here
    qc.config.logger.start_logging_on_import = "never"
    qc.config.telemetry.enabled = False
    qc.config.subscription.default_subscribers = []
    qc.config.core.db_location = str(tmp_path / "temp.db")

    try:
        yield
    finally:
        qc.config.home_file_name = home_file_name
        qc.config.schema_home_file_name = schema_home_file_name
        qc.config.env_file_name = env_file_name
        qc.config.schema_env_file_name = schema_env_file_name
        qc.config.cwd_file_name = cwd_file_name
        qc.config.schema_cwd_file_name = schema_cwd_file_name

        qc.config.current_config = old_config


@pytest.fixture(scope="function", autouse=True)
def reset_state_on_exit() -> Generator[None, None, None]:
    """
    Fixture to clean any shared state on exit

    Currently this resets the config to the default config,
    closes the default station and closes all instruments.
    """
    default_config_obj: DotDict | None = copy.deepcopy(qc.config.current_config)

    try:
        yield
    finally:
        qc.config.current_config = default_config_obj
        Instrument.close_all()
        Station.default = None


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
    n_rows = 10**3
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
