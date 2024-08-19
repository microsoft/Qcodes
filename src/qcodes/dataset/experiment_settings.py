"""Settings that are indirectly related to experiments."""

from __future__ import annotations

from qcodes.dataset.sqlite.connection import ConnectionPlus, path_to_dbfile
from qcodes.dataset.sqlite.queries import get_last_experiment

_default_experiment: dict[str, int | None] = {}


def _set_default_experiment_id(db_path: str, exp_id: int) -> None:
    """
    Sets the default experiment with the exp_id of a created/ loaded
    experiment for the database that this experiment belongs to.

    Args:
        db_path: The database that a created/ loaded experiment belongs to.
        exp_id: The exp_id of a created/ loaded experiment.
    """
    global _default_experiment
    _default_experiment[db_path] = exp_id


def _get_latest_default_experiment_id(db_path: str) -> int | None:
    """
    Gets the latest created or loaded experiment's exp_id. If no experiment is set
    None will be returned.

    Args:
        db_path: Database path.

    Returns:
        The latest created/ loaded experiment's exp_id.
    """
    global _default_experiment
    return _default_experiment.get(db_path, None)


def reset_default_experiment_id(conn: ConnectionPlus | None = None) -> None:
    """
    Resets the default experiment id to to the last experiment in the db.
    """
    global _default_experiment
    if conn is None:
        _default_experiment = {}
    else:
        db_path = path_to_dbfile(conn)
        _default_experiment[db_path] = None


def get_default_experiment_id(conn: ConnectionPlus) -> int:
    """
    Returns the latest created/ loaded experiment's exp_id as the default
    experiment. If it is not set the maximum exp_id returned as the default.
    If no experiment is found in the database, a ValueError is raised.

    Args:
        conn: Open connection to the db in question.

    Returns:
        exp_id of the default experiment.

    Raises:
        ValueError: If no experiment exists in the given db.
    """
    db_path = path_to_dbfile(conn)
    exp_id = _get_latest_default_experiment_id(db_path)
    if exp_id is None:
        exp_id = get_last_experiment(conn)
    if exp_id is None:
        raise ValueError(
            "No experiments found."
            " You can create one with:"
            " new_experiment(name, sample_name)"
        )
    return exp_id
