"""Settings that are indirectly related to experiments."""

from typing import Optional

# The active experiment's exp_id. The default is None.
active_experiment: Optional[int] = None


def _set_active_experiment_id(exp_id: Optional[int]) -> None:
    """
    Sets the global active_experiment to the exp_id of a created/ loaded
    experiment. When a database is loaded, sets to the maximum exp_id in
    the database. If it is a new database, sets to None.

    Args:
        exp_id: The exp_id of an experiment.
    """
    global active_experiment
    active_experiment = exp_id


def get_active_experiment_id() -> Optional[int]:
    """
    Gets the updated global active_experiment.

    Returns:
        Returns the latest created/ loaded experiment's exp_id in the kernel.
        If no experiment is started in the kernel and only a database is
        initialized, the return will be the maximum exp_id in the database
        or None, if it is a new database.
    """
    global active_experiment
    return active_experiment


def reset_active_experiment_id() -> None:
    """
    Resets the active_experiment to None.
    """
    global active_experiment
    active_experiment = None


def _handle_active_experiment_id_return() -> Optional[int]:
    """
    Checks if get_active_experiment_id is an existing exp_id and return it,
    and if it is None, raise error that no experiment is initialized.
    """
    global active_experiment
    if active_experiment is not None:
        return active_experiment
    else:
        raise ValueError("No experiments found."
                         " You can create one with:"
                         " load_or_create_experiment(name, sample_name)")
