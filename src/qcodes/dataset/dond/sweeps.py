from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from qcodes.dataset.dond.do_nd_utils import ActionsT
    from qcodes.parameters import ParameterBase

T = TypeVar("T", bound=np.generic)


class AbstractSweep(ABC, Generic[T]):
    """
    Abstract sweep class that defines an interface for concrete sweep classes.
    """

    @abstractmethod
    def get_setpoints(self) -> npt.NDArray[T]:
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

    @property
    def get_after_set(self) -> bool:
        """
        Should we perform a call to get on the parameter after setting it
        and store that rather than the setpoint value in the dataset?

        This defaults to False for backwards compatibility
        but subclasses should overwrite this to implement if correctly.
        """
        return False


class LinSweep(AbstractSweep[np.float64]):
    """
    Linear sweep.

    Args:
        param: Qcodes parameter to sweep.
        start: Sweep start value.
        stop: Sweep end value.
        num_points: Number of sweep points.
        delay: Time in seconds between two consecutive sweep points.
        post_actions: Actions to do after each sweep point.
        get_after_set: Should we perform a get on the parameter after setting it
            and store the value returned by get rather than the set value in the dataset.
    """

    def __init__(
        self,
        param: ParameterBase,
        start: float,
        stop: float,
        num_points: int,
        delay: float = 0,
        post_actions: ActionsT = (),
        get_after_set: bool = False,
    ):
        self._param = param
        self._start = start
        self._stop = stop
        self._num_points = num_points
        self._delay = delay
        self._post_actions = post_actions
        self._get_after_set = get_after_set

    def get_setpoints(self) -> npt.NDArray[np.float64]:
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

    @property
    def get_after_set(self) -> bool:
        return self._get_after_set


class LogSweep(AbstractSweep[np.float64]):
    """
    Logarithmic sweep.

    Args:
        param: Qcodes parameter for sweep.
        start: Sweep start value.
        stop: Sweep end value.
        num_points: Number of sweep points.
        delay: Time in seconds between two consecutive sweep points.
        post_actions: Actions to do after each sweep point.
        get_after_set: Should we perform a get on the parameter after setting it
            and store the value returned by get rather than the set value in the dataset.
    """

    def __init__(
        self,
        param: ParameterBase,
        start: float,
        stop: float,
        num_points: int,
        delay: float = 0,
        post_actions: ActionsT = (),
        get_after_set: bool = False,
    ):
        self._param = param
        self._start = start
        self._stop = stop
        self._num_points = num_points
        self._delay = delay
        self._post_actions = post_actions
        self._get_after_set = get_after_set

    def get_setpoints(self) -> npt.NDArray[np.float64]:
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

    @property
    def get_after_set(self) -> bool:
        return self._get_after_set


class ArraySweep(AbstractSweep, Generic[T]):
    """
    Sweep the values of a given array.

    Args:
        param: Qcodes parameter for sweep.
        array: array with values to sweep.
        delay: Time in seconds between two consecutive sweep points.
        post_actions: Actions to do after each sweep point.
        get_after_set: Should we perform a get on the parameter after setting it
            and store the value returned by get rather than the set value in the dataset.
    """

    def __init__(
        self,
        param: ParameterBase,
        array: Sequence[Any] | npt.NDArray[T],
        delay: float = 0,
        post_actions: ActionsT = (),
        get_after_set: bool = False,
    ):
        self._param = param
        self._array = np.array(array)
        self._delay = delay
        self._post_actions = post_actions
        self._get_after_set = get_after_set

    def get_setpoints(self) -> npt.NDArray[T]:
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

    @property
    def get_after_set(self) -> bool:
        return self._get_after_set


class TogetherSweep:
    """
    A combination of Multiple sweeps that are to be performed in parallel
    such that all parameters in the `TogetherSweep` are set to the next value
    before a parameter is read.

    """

    def __init__(self, *sweeps: AbstractSweep):
        if len(sweeps) == 0:
            raise ValueError("A TogetherSweep must contain at least one sweep.")

        len_0 = sweeps[0].num_points

        for sweep in sweeps:
            if sweep.num_points != len_0:
                raise ValueError(
                    f"All Sweeps in a TogetherSweep must have the same length."
                    f"Sweep of {sweep.param} had {sweep.num_points} but the "
                    f"first one had {len_0}."
                )

        self._sweeps = tuple(sweeps)

    @property
    def sweeps(self) -> tuple[AbstractSweep, ...]:
        return self._sweeps

    def get_setpoints(self) -> Iterable:
        return zip(*(sweep.get_setpoints() for sweep in self.sweeps))

    @property
    def num_points(self) -> int:
        return self.sweeps[0].num_points
