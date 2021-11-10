import logging
from collections.abc import Sized
from typing import Any, List, Optional, Union
from warnings import warn

from qcodes.dataset.data_set import DataSet, load_by_id, new_data_set
from qcodes.dataset.data_set_protocol import SPECS, DataSetProtocol
from qcodes.dataset.experiment_settings import _set_default_experiment_id
from qcodes.dataset.sqlite.connection import ConnectionPlus, path_to_dbfile
from qcodes.dataset.sqlite.database import (
    conn_from_dbpath_or_conn,
    connect,
    get_DB_location,
)
from qcodes.dataset.sqlite.queries import (
    finish_experiment,
    get_experiment_name_from_experiment_id,
    get_experiments,
    get_last_experiment,
    get_last_run,
    get_matching_exp_ids,
    get_run_counter,
    get_runid_from_expid_and_counter,
    get_runs,
    get_sample_name_from_experiment_id,
)
from qcodes.dataset.sqlite.queries import new_experiment as ne
from qcodes.dataset.sqlite.query_helpers import VALUES, select_one_where

log = logging.getLogger(__name__)


class Experiment(Sized):
    def __init__(self, path_to_db: Optional[str] = None,
                 exp_id: Optional[int] = None,
                 name: Optional[str] = None,
                 sample_name: Optional[str] = None,
                 format_string: str = "{}-{}-{}",
                 conn: Optional[ConnectionPlus] = None) -> None:
        """
        Create or load an experiment. If exp_id is None, a new experiment is
        created. If exp_id is not None, an experiment is loaded.

        Args:
            path_to_db: The path of the database file to create in/load from.
              If a conn is passed together with path_to_db, an exception is
              raised
            exp_id: The id of the experiment to load
            name: The name of the experiment to create. Ignored if exp_id is
              not None
            sample_name: The sample name for this experiment. Ignored if exp_id
              is not None
            format_string: The format string used to name result-tables.
              Ignored if exp_id is not None.
            conn: connection to the database. If not supplied, the constructor
              first tries to use path_to_db to figure out where to connect to.
              If path_to_db is not supplied either, a new connection
              to the DB file specified in the config is made
        """

        self.conn = conn_from_dbpath_or_conn(conn, path_to_db)

        max_id = len(get_experiments(self.conn))

        if exp_id is not None:
            if exp_id not in range(1, max_id+1):
                raise ValueError('No such experiment in the database')
            self._exp_id = exp_id
        else:

            # it is better to catch an invalid format string earlier than later
            try:
                # the corresponding function from sqlite module will try to
                # format as `(name, exp_id, run_counter)`, hence we prepare
                # for that here
                format_string.format("name", 1, 1)
            except Exception as e:
                raise ValueError("Invalid format string. Can not format "
                                 "(name, exp_id, run_counter)") from e

            log.info(f"creating new experiment in {self.path_to_db}")

            name = name or f"experiment_{max_id+1}"
            sample_name = sample_name or "some_sample"
            self._exp_id = ne(self.conn, name, sample_name, format_string)

    @property
    def exp_id(self) -> int:
        return self._exp_id

    @property
    def path_to_db(self) -> str:
        return self.conn.path_to_dbfile

    @property
    def name(self) -> str:
        return get_experiment_name_from_experiment_id(self.conn, self.exp_id)

    @property
    def sample_name(self) -> str:
        return get_sample_name_from_experiment_id(self.conn, self.exp_id)

    @property
    def last_counter(self) -> int:
        return get_run_counter(self.conn, self.exp_id)

    @property
    def started_at(self) -> float:
        start_time = select_one_where(
            self.conn, "experiments", "start_time", "exp_id", self.exp_id
        )
        assert isinstance(start_time, float)
        return start_time

    @property
    def finished_at(self) -> Optional[float]:
        finish_time = select_one_where(
            self.conn, "experiments", "end_time", "exp_id", self.exp_id
        )
        assert isinstance(finish_time, (float, type(None)))
        return finish_time

    @property
    def format_string(self) -> str:
        format_str = select_one_where(
            self.conn, "experiments", "format_string", "exp_id", self.exp_id
        )
        assert isinstance(format_str, str)
        return format_str

    def new_data_set(self, name: str,
                     specs: Optional[SPECS] = None,
                     values: Optional[VALUES] = None,
                     metadata: Optional[Any] = None) -> DataSet:
        """
        Create a new dataset in this experiment

        Args:
            name: the name of the new dataset
            specs: list of parameters (as ParamSpecs) to create this data_set
                with
            values: the values to associate with the parameters
            metadata: the metadata to associate with the dataset
        """
        return new_data_set(name, self.exp_id, specs, values, metadata,
                            conn=self.conn)

    def data_set(self, counter: int) -> DataSet:
        """
        Get dataset with the specified counter from this experiment

        Args:
            counter: the counter of the run we want to load

        Returns:
            the dataset
        """
        run_id = get_runid_from_expid_and_counter(self.conn, self.exp_id,
                                                  counter)
        return DataSet(run_id=run_id, conn=self.conn)

    def data_sets(self) -> List[DataSetProtocol]:
        """Get all the datasets of this experiment"""
        runs = get_runs(self.conn, self.exp_id)
        return [load_by_id(run['run_id'], conn=self.conn) for run in runs]

    def last_data_set(self) -> DataSetProtocol:
        """Get the last dataset of this experiment"""
        run_id = get_last_run(self.conn, self.exp_id)
        if run_id is None:
            raise ValueError('There are no runs in this experiment')
        return load_by_id(run_id)

    def finish(self) -> None:
        """
        Marks this experiment as finished by saving the moment in time
        when this method is called
        """
        finish_experiment(self.conn, self.exp_id)

    def __len__(self) -> int:
        return len(self.data_sets())

    def __repr__(self) -> str:
        out = [
            f"{self.name}#{self.sample_name}#{self.exp_id}@{self.path_to_db}"
        ]
        out.append("-" * len(out[0]))
        out += [
            f"{d.run_id}-{d.name}-{d.counter}-{d._parameters}-{len(d)}"
            for d in self.data_sets()
        ]
        return "\n".join(out)


# public api

def experiments(conn: Optional[ConnectionPlus] = None) -> List[Experiment]:
    """
    List all the experiments in the container (database file from config)

    Args:
        conn: connection to the database. If not supplied, a new connection
          to the DB file specified in the config is made

    Returns:
        All the experiments in the container
    """
    conn = conn_from_dbpath_or_conn(conn=conn, path_to_db=None)
    log.info(f"loading experiments from {conn.path_to_dbfile}")
    rows = get_experiments(conn)
    return [load_experiment(row['exp_id'], conn) for row in rows]


def new_experiment(name: str,
                   sample_name: Optional[str],
                   format_string: str = "{}-{}-{}",
                   conn: Optional[ConnectionPlus] = None) -> Experiment:
    """
    Create a new experiment (in the database file from config)

    Args:
        name: the name of the experiment
        sample_name: the name of the current sample
        format_string: basic format string for table-name
            must contain 3 placeholders.
        conn: connection to the database. If not supplied, a new connection
          to the DB file specified in the config is made
    Returns:
        the new experiment
    """
    sample_name = sample_name or "some_sample"
    conn = conn or connect(get_DB_location())
    exp_ids = get_matching_exp_ids(conn, name=name, sample_name=sample_name)
    if len(exp_ids) >= 1:
        log.warning(
            f"There is (are) already experiment(s) with the name of {name} "
            f"and sample name of {sample_name} in the database."
        )
    experiment = Experiment(
        name=name, sample_name=sample_name, format_string=format_string, conn=conn
    )
    _set_default_experiment_id(path_to_dbfile(conn), experiment.exp_id)
    return experiment


def load_experiment(exp_id: int,
                    conn: Optional[ConnectionPlus] = None) -> Experiment:
    """
    Load experiment with the specified id (from database file from config)

    Args:
        exp_id: experiment id
        conn: connection to the database. If not supplied, a new connection
          to the DB file specified in the config is made

    Returns:
        experiment with the specified id
    """
    conn = conn_from_dbpath_or_conn(conn=conn, path_to_db=None)
    if not isinstance(exp_id, int):
        raise ValueError('Experiment ID must be an integer')
    experiment = Experiment(exp_id=exp_id, conn=conn)
    _set_default_experiment_id(path_to_dbfile(conn), experiment.exp_id)
    return experiment


def load_last_experiment() -> Experiment:
    """
    Load last experiment (from database file from config)

    Returns:
        last experiment
    """
    conn = connect(get_DB_location())
    last_exp_id = get_last_experiment(conn)
    if last_exp_id is None:
        raise ValueError('There are no experiments in the database file')
    experiment = Experiment(exp_id=last_exp_id)
    _set_default_experiment_id(get_DB_location(), experiment.exp_id)
    return experiment


def load_experiment_by_name(
    name: str,
    sample: Optional[str] = None,
    conn: Optional[ConnectionPlus] = None,
    load_last_duplicate: bool = False,
) -> Experiment:
    """
    Try to load experiment with the specified name.

    Nothing stops you from having many experiments with the same name and
    sample name. In that case this won't work unless load_last_duplicate
    is set to True. Then, the last of duplicated experiments will be loaded.

    Args:
        name: the name of the experiment
        sample: the name of the sample
        load_last_duplicate: If True, prevent raising error for having
            multiple experiments with the same name and sample name, and
            load the last duplicated experiment, instead.
        conn: connection to the database. If not supplied, a new connection
            to the DB file specified in the config is made

    Returns:
        the requested experiment

    Raises:
        ValueError either if the name and sample name are not unique, unless
        load_last_duplicate is True, or if no experiment found for the
        supplied name and sample.
        .
    """
    conn = conn or connect(get_DB_location())
    if sample is not None:
        args_to_find = {"name": name, "sample_name": sample}
    else:
        args_to_find = {"name": name}
    exp_ids = get_matching_exp_ids(conn, **args_to_find)
    if len(exp_ids) == 0:
        raise ValueError("Experiment not found")
    elif len(exp_ids) > 1:
        _repr = []
        for exp_id in exp_ids:
            exp = load_experiment(exp_id, conn=conn)
            s = (
                f"exp_id:{exp.exp_id} ({exp.name}-{exp.sample_name})"
                f" started at ({exp.started_at})"
            )
            _repr.append(s)
        _repr_str = "\n".join(_repr)
        if load_last_duplicate:
            e = exp
        else:
            raise ValueError(
                f"Many experiments matching your request" f" found:\n{_repr_str}"
            )
    else:
        e = Experiment(exp_id=exp_ids[0], conn=conn)
    _set_default_experiment_id(path_to_dbfile(conn), e.exp_id)
    return e


def load_or_create_experiment(
    experiment_name: str,
    sample_name: Optional[str] = None,
    conn: Optional[ConnectionPlus] = None,
    load_last_duplicate: bool = False,
) -> Experiment:
    """
    Find and return an experiment with the given name and sample name,
    or create one if not found.

    Args:
        experiment_name: Name of the experiment to find or create.
        sample_name: Name of the sample.
        load_last_duplicate: If True, prevent raising error for having
            multiple experiments with the same name and sample name, and
            load the last duplicated experiment, instead.
        conn: Connection to the database. If not supplied, a new connection
            to the DB file specified in the config is made.

    Returns:
        The found or created experiment
    """
    conn = conn or connect(get_DB_location())
    try:
        experiment = load_experiment_by_name(
            experiment_name,
            sample_name,
            load_last_duplicate=load_last_duplicate,
            conn=conn,
        )
    except ValueError as exception:
        if "Experiment not found" in str(exception):
            experiment = new_experiment(experiment_name, sample_name,
                                        conn=conn)
        else:
            raise exception
    return experiment


def _create_exp_if_needed(
    target_conn: ConnectionPlus,
    exp_name: str,
    sample_name: str,
    fmt_str: str,
    start_time: float,
    end_time: Union[float, None],
) -> int:
    """
    Look up in the database whether an experiment already exists and create
    it if it doesn't. Note that experiments do not have GUIDs, so this method
    is not guaranteed to work. Matching names and times is the best we can do.
    """

    matching_exp_ids = get_matching_exp_ids(
        target_conn,
        name=exp_name,
        sample_name=sample_name,
        format_string=fmt_str,
        start_time=start_time,
        end_time=end_time,
    )

    if len(matching_exp_ids) > 1:
        exp_id = matching_exp_ids[0]
        warn(
            f"{len(matching_exp_ids)} experiments found in target DB that "
            "match name, sample_name, fmt_str, start_time, and end_time. "
            f"Inserting into the experiment with exp_id={exp_id}."
        )
        return exp_id
    if len(matching_exp_ids) == 1:
        return matching_exp_ids[0]
    else:
        lastrowid = ne(
            target_conn,
            name=exp_name,
            sample_name=sample_name,
            format_string=fmt_str,
            start_time=start_time,
            end_time=end_time,
        )
        return lastrowid
