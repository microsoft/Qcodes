"""Settings that are indirectly related to experiments."""

from typing import Optional

# The active exp_id in the kernel. The default is None, means the last exp_id
# should be active, if available.
active_experiment: Optional[int] = None


def _set_active_experiment_id(exp_id: int) -> None:
    """
    Sets the active_experiment to the exp_id of a created/ loaded experiment.

    Args:
        exp_id: The exp_id of an experiment that is created/ loaded.
    """
    global active_experiment
    active_experiment = exp_id


def get_active_experiment_id() -> Optional[int]:
    """
    Gets the active exp_id.

    Returns:
        Active exp_id. If the return is None, then the active exp_id will be
        the last exp_id in experiments list, if available.
    """
    global active_experiment
    return active_experiment


def reset_active_experiment_id() -> None:
    """
    Resets the active_experiment to the default.
    """
    global active_experiment
    active_experiment = None
