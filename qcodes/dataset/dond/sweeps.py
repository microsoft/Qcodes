from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from qcodes.dataset.dond.do_nd_utils import ActionsT
from qcodes.parameters import ParameterBase


class AbstractSweep(ABC):
    """
    Abstract sweep class that defines an interface for concrete sweep classes.
    """

    @abstractmethod
    def get_setpoints(self) -> np.ndarray:
        """
        Returns an array of setpoint values for this sweep.
        """
        pass

    @property
    @abstractmethod
    def param(self) -> ParameterBase:
        """
        Returns the Qcodes sweep parameter.
        """
        pass

    @property
    @abstractmethod
    def delay(self) -> float:
        """
        Delay between two consecutive sweep points.
        """
        pass

    @property
    @abstractmethod
    def num_points(self) -> int:
        """
        Number of sweep points.
        """
        pass

    @property
    @abstractmethod
    def post_actions(self) -> ActionsT:
        """
        actions to be performed after setting param to its setpoint.
        """
        pass


class LinSweep(AbstractSweep):
    """
    Linear sweep.

    Args:
        param: Qcodes parameter to sweep.
        start: Sweep start value.
        stop: Sweep end value.
        num_points: Number of sweep points.
        delay: Time in seconds between two consequtive sweep points
    """

    def __init__(
        self,
        param: ParameterBase,
        start: float,
        stop: float,
        num_points: int,
        delay: float = 0,
        post_actions: ActionsT = (),
    ):
        self._param = param
        self._start = start
        self._stop = stop
        self._num_points = num_points
        self._delay = delay
        self._post_actions = post_actions

    def get_setpoints(self) -> np.ndarray:
        """
        Linear (evenly spaced) numpy array for supplied start, stop and
        num_points.
        """
        return np.linspace(self._start, self._stop, self._num_points)

    @property
    def param(self) -> ParameterBase:
        return self._param

    @property
    def delay(self) -> float:
        return self._delay

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def post_actions(self) -> ActionsT:
        return self._post_actions


class LogSweep(AbstractSweep):
    """
    Logarithmic sweep.

    Args:
        param: Qcodes parameter for sweep.
        start: Sweep start value.
        stop: Sweep end value.
        num_points: Number of sweep points.
        delay: Time in seconds between two consequtive sweep points.
    """

    def __init__(
        self,
        param: ParameterBase,
        start: float,
        stop: float,
        num_points: int,
        delay: float = 0,
        post_actions: ActionsT = (),
    ):
        self._param = param
        self._start = start
        self._stop = stop
        self._num_points = num_points
        self._delay = delay
        self._post_actions = post_actions

    def get_setpoints(self) -> np.ndarray:
        """
        Logarithmically spaced numpy array for supplied start, stop and
        num_points.
        """
        return np.logspace(self._start, self._stop, self._num_points)

    @property
    def param(self) -> ParameterBase:
        return self._param

    @property
    def delay(self) -> float:
        return self._delay

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def post_actions(self) -> ActionsT:
        return self._post_actions


class ArraySweep(AbstractSweep):
    """
    Sweep the values of a given array.

    Args:
        param: Qcodes parameter for sweep.
        array: array with values to sweep.
        delay: Time in seconds between two consecutive sweep points.
        post_actions: Actions to do after each sweep point.
    """

    def __init__(
        self,
        param: ParameterBase,
        array: Sequence[float] | np.ndarray,
        delay: float = 0,
        post_actions: ActionsT = (),
    ):
        self._param = param
        self._array = np.array(array)
        self._delay = delay
        self._post_actions = post_actions

    def get_setpoints(self) -> np.ndarray:
        return self._array

    @property
    def param(self) -> ParameterBase:
        return self._param

    @property
    def delay(self) -> float:
        return self._delay

    @property
    def num_points(self) -> int:
        return len(self._array)

    @property
    def post_actions(self) -> ActionsT:
        return self._post_actions
