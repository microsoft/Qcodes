"""Settings that are indirectly related to experiments."""

from typing import Optional, Dict
import warnings

from qcodes.dataset.sqlite.connection import ConnectionPlus
from qcodes.dataset.sqlite.queries import get_last_experiment

# The default experiment's exp_id. This changes to the exp_id of a created/
# loaded experiment. The idea is to store experiment id only for a single
# database path.
_default_experiment: Optional[Dict[str, int]] = None


def _set_default_experiment_id(db_path: str, exp_id: int) -> None:
    """
    Sets the default experiment to the exp_id of a created/ loaded experiment.

    Args:
        exp_id: The exp_id of an experiment.
    """
    global _default_experiment
    _default_experiment = {db_path: exp_id}


def _get_latest_default_experiment_id() -> Optional[int]:
    """
    Gets the lastest created or loaded experiment's exp_id.

    Returns:
        The latest created/ loaded experiment's exp_id.
    """
    global _default_experiment

    to_return: Optional[int] = None

    if _default_experiment is None:
        to_return = None
    else:
        default_experiments = tuple(_default_experiment.values())
        if len(default_experiments) == 1:
            to_return = default_experiments[0]
        elif len(default_experiments) == 0:
            _default_experiment = None
            to_return = None
        else:
            warnings.warn(f"Unclear which default experiment to use {_default_experiment}, using None")
            to_return = None

    return to_return


def reset_default_experiment_id() -> None:
    """
    Resets the default experiment to None.
    """
    global _default_experiment
    _default_experiment = None


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
