from collections import Sized
from typing import Optional, List
import logging

import qcodes
from qcodes.dataset.data_set import (DataSet, load_by_id, load_by_counter,
                                     new_data_set, SPECS)

from qcodes.dataset.sqlite_base import (select_one_where, finish_experiment,
                                        get_run_counter, get_runs,
                                        get_last_run,
                                        connect, transaction,
                                        get_last_experiment, get_experiments,
                                        get_experiment_name_from_experiment_id,
                                        get_sample_name_from_experiment_id)
from qcodes.dataset.sqlite_base import new_experiment as ne
from qcodes.dataset.database import get_DB_location, get_DB_debug


log = logging.getLogger(__name__)


class Experiment(Sized):
    def __init__(self, path_to_db: str) -> None:
        self.path_to_db = path_to_db
        self.conn = connect(self.path_to_db, get_DB_debug())
        self._debug = False

    def _new(self,
             name: str,
             sample_name: str,
             format_string: Optional[str] = "{}-{}-{}"
             ) -> None:
        """
        Actually perform all the side effects needed for
        the creation of a new dataset.
        """
        exp_id = ne(self.conn, name, sample_name, format_string)
        self.exp_id = exp_id
        self.format_string = format_string

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
    def started_at(self) -> int:
        return select_one_where(self.conn, "experiments",
                                "exp_id", "start_time", self.exp_id)

    @property
    def finished_at(self) -> int:
        return select_one_where(self.conn, "experiments",
                                "exp_id", "end_time", self.exp_id)

    def new_data_set(self, name, specs: SPECS = None, values=None,
                     metadata=None) -> DataSet:
        """ Create a new dataset in this experimetn

        Args:
            name: the name of the new dataset
            specs: list of parameters to create this data_set with
            values: the values to associate with the parameters
            metadata:  the values to associate with the dataset
        """
        return new_data_set(name, self.exp_id, specs, values, metadata)

    def data_set(self, counter: int) -> DataSet:
        """ Get dataset with the secified counter from this experiment

        Args:
            counter: the counter we want to load

        Returns:
            the dataset

        """
        return load_by_counter(counter, self.exp_id)

    def data_sets(self) -> List[DataSet]:
        """ Get all the datasets

        Returns:
            All the datasets of this experiment

        """
        runs = get_runs(self.conn, self.exp_id)
        data_sets = []
        for run in runs:
            data_sets.append(load_by_id(run['run_id']))
        return data_sets

    def last_data_set(self) -> DataSet:
        return load_by_id(get_last_run(self.conn, self.exp_id))

    def finish(self) -> None:
        """
        Marks this experiment as finished
        """
        finish_experiment(self.conn, self.exp_id)

    def __len__(self) -> int:
        return len(self.data_sets())

    def __repr__(self) -> str:
        out = []
        heading = (f"{self.name}#{self.sample_name}#{self.exp_id}"
                   f"@{self.path_to_db}")
        out.append(heading)
        out.append("-" * len(heading))
        ds = self.data_sets()
        if len(ds) > 0:
            for d in ds:
                out.append(f"{d.run_id}-{d.name}-{d.counter}"
                           f"-{d.parameters}-{len(d)}")

        return "\n".join(out)


# public api

def experiments()->List[Experiment]:
    """
    List all the experiments in the container

    Returns:
        All the experiments in the container

    """
    log.info("loading experiments from {}".format(get_DB_location()))
    rows = get_experiments(connect(get_DB_location(), get_DB_debug()))
    experiments = []
    for row in rows:
        experiments.append(load_experiment(row['exp_id']))
    return experiments


def new_experiment(name: str,
                   sample_name: str,
                   format_string: Optional[str] = "{}-{}-{}") -> Experiment:
    """ Create a new experiment

    Args:
        name: the name of the experiment
        sample_name: the name of the current sample
        format_string: basic format string for table-name
            must contain 3 placeholders.
    Returns:
        the new experiment
    """
    log.info("creating new experiment in {}".format(get_DB_location()))
    e = Experiment(get_DB_location())
    e._new(name, sample_name, format_string)
    return e


def load_experiment(exp_id: int) -> Experiment:
    """
    Load experiment with the specified id
    Args:
        exp_id: experiment id

    Returns:
        experiment with the specified id
    """
    e = Experiment(get_DB_location())
    e.exp_id = exp_id
    return e


def load_last_experiment() -> Experiment:
    """
    Load last experiment

    Returns:
        last experiment
    """
    e = Experiment(get_DB_location())
    e.exp_id = get_last_experiment(e.conn)
    return e


def load_experiment_by_name(name: str,
                            sample: Optional[str] = None) -> Experiment:
    """
    Try to load experiment with the specified name.
    Nothing stops you from having many experiments with
    the same name and sample_name.
    In that case this won't work. And warn you.
    Args:
        name: the name of the experiment
        sample: the name of the sample

    Returns:
        the requested experiment

    Raises:
        ValueError if the name is not unique and sample name is None.
    """
    e = Experiment(get_DB_location())
    if sample:
        sql = """
        SELECT
            *
        FROM
            experiments
        WHERE
            sample_name = ? AND
            name = ?
        """
        c = transaction(e.conn, sql, sample, name)
    else:
        sql = """
        SELECT
            *
        FROM
            experiments
        WHERE
            name = ?
        """
        c = transaction(e.conn, sql, name)
    rows = c.fetchall()
    if len(rows) == 0:
        raise ValueError("Experiment not found \n")
    elif len(rows) > 1:
        _repr = []
        for row in rows:
            s = f"exp_id:{row['exp_id']} ({row['name']}-{row['sample_name']}) started at({row['start_time']})"
            _repr.append(s)
        _repr_str = "\n".join(_repr)
        raise ValueError(f"Many experiments matching your request found {_repr_str}")
    else:
        e.exp_id = rows[0]['exp_id']
    return e


def load_or_create_experiment(experiment_name: str,
                              sample_name: str
                              ) -> Experiment:
    """
    Find and return an experiment with the given name and sample name,
    or create one if not found.

    Args:
        experiment_name
            Name of the experiment to find or create
        sample_name
            Name of the sample

    Returns:
        The found or created experiment
    """
    try:
        experiment = load_experiment_by_name(experiment_name, sample_name)
    except ValueError as exception:
        if "Experiment not found" in str(exception):
            experiment = new_experiment(experiment_name, sample_name)
        else:
            raise exception
    return experiment
