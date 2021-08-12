"""Settings that are indirectly related to experiments."""

import warnings
from typing import Dict, Optional

import qcodes
from qcodes.dataset.sqlite.connection import ConnectionPlus, path_to_dbfile
from qcodes.dataset.sqlite.queries import get_last_experiment

# The default experiment. This changes to the exp_id of a created/
# loaded experiment. The idea is to store exp_id only for a single
# db_path.
_default_experiment: Optional[Dict[str, int]] = None


def _set_default_experiment_id(db_path: str, exp_id: int) -> None:
    """
    Sets the default experiment with the exp_id of a created/ loaded
    experiment for the database that this experiment belongs to.

    Args:
        db_path: The database that a created/ loaded experiment belongs to.
        exp_id: The exp_id of a created/ loaded experiment.
    """
    global _default_experiment
    _default_experiment = {db_path: exp_id}


def _get_latest_default_experiment_id(db_path: str) -> Optional[int]:
    """
    Gets the lastest created or loaded experiment's exp_id. It makes sure that
    the supplied db_path to match with the db_path of the default experiment
    and returns the exp_id of that experiment. If there is no match, a warning
    will be raised and the default_experiment will be reset.

    Args:
        db_path: Database path.

    Returns:
        The latest created/ loaded experiment's exp_id.
    """
    global _default_experiment
    to_return: Optional[int] = None

    if _default_experiment is not None:
        db_id = [(db, id) for db, id in _default_experiment.items()]
        if db_id[0][0] == db_path:
            to_return = db_id[0][1]
        else:
            warnings.warn(f"Connected db_path {db_path} does not match with "
                          f"the default experiment's db_path {db_id[0][0]}. "
                          f"Reseting default experiment, i.e., latest exp_id "
                          f"of databsae {qcodes.config['core']['db_location']}"
                          f" will be used as the default experiment, unless "
                          f"you change this default by load/ create another "
                          f"experiment.")
            reset_default_experiment_id()

    return to_return


def reset_default_experiment_id() -> None:
    """
    Resets the default experiment to None.
    """
    global _default_experiment
    _default_experiment = None


def get_default_experiment_id(conn: ConnectionPlus) -> int:
    """
    Returns the latest created/ loaded experiment's exp_id as the default
    experiment. If it is None, maximum exp_id from the currently active
    initialized database is returned as the default. if no experiment
    found in the database, a ValueError is raised.

    Returns:
        exp_id of the default experiment.
    """
    db_path = path_to_dbfile(conn)
    exp_id = _get_latest_default_experiment_id(db_path)
    if exp_id is None:
        exp_id = get_last_experiment(conn)
    if exp_id is None:
        raise ValueError("No experiments found."
                         " You can create one with:"
                         " new_experiment(name, sample_name)")
    return exp_id
