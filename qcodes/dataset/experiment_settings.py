"""Settings that are indirectly related to experiments."""

from typing import Optional
from qcodes.dataset.sqlite.connection import ConnectionPlus
from qcodes.dataset.sqlite.queries import get_last_experiment

# The default experiment's exp_id. This changes to the exp_id of a created/
# loaded experiment.
default_experiment: Optional[int] = None


def _set_default_experiment_id(exp_id: int) -> None:
    """
    Sets the default_experiment to the exp_id of a created/ loaded experiment.

    Args:
        exp_id: The exp_id of an experiment.
    """
    global default_experiment
    default_experiment = exp_id


def _get_latest_default_experiment_id() -> Optional[int]:
    """
    Gets the lastest created or loaded experiment's exp_id.

    Returns:
        The latest created/ loaded experiment's exp_id.
    """
    global default_experiment
    return default_experiment


def reset_default_experiment_id() -> None:
    """
    Resets the default_experiment to None.
    """
    global default_experiment
    default_experiment = None


def get_default_experiment_id(conn: ConnectionPlus) -> Optional[int]:
    """
    Returns the latest created/ loaded experiment's exp_id as the default
    experiment. If it is None, maximum exp_id from the currently active
    initialized database is returned as the default. if no experiment
    found in the database, a ValueError is raised.

    Returns:
        exp_id of the default experiment.
    """
    exp_id = _get_latest_default_experiment_id()
    if exp_id is None:
        exp_id = get_last_experiment(conn)
    if exp_id is None:
        raise ValueError("No experiments found."
                         " You can create one with:"
                         " new_experiment(name, sample_name)")
    return exp_id
