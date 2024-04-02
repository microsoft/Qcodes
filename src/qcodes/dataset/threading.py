from __future__ import annotations

# Instead of async, since there are really a very small set of things
# we want to happen simultaneously within one process (namely getting
# several parameters in parallel), we can parallelize them with threads.
# That way the things we call need not be rewritten explicitly async.
import concurrent
import concurrent.futures
import itertools
import logging
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Callable, Protocol, TypeVar, Union

from qcodes.utils import RespondingThread

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from qcodes.dataset.data_set_protocol import values_type
    from qcodes.parameters import ParamDataType, ParameterBase

ParamMeasT = Union["ParameterBase", Callable[[], None]]
OutType = list[tuple["ParameterBase", "values_type"]]

T = TypeVar("T")

_LOG = logging.getLogger(__name__)


class _ParamCaller:
    def __init__(self, *parameters: ParameterBase):

        self._parameters = parameters

    def __call__(self) -> tuple[tuple[ParameterBase, ParamDataType], ...]:
        output = []
        for param in self._parameters:
            output.append((param, param.get()))
        return tuple(output)

    def __repr__(self) -> str:
        names = tuple(param.full_name for param in self._parameters)
        return f"ParamCaller of {','.join(names)}"


def _instrument_to_param(
    params: Sequence[ParamMeasT],
) -> dict[str | None, tuple[ParameterBase, ...]]:
    from qcodes.parameters import ParameterBase

    real_parameters = [param for param in params if isinstance(param, ParameterBase)]

    output: dict[str | None, tuple[ParameterBase, ...]] = defaultdict(tuple)
    for param in real_parameters:
        if param.underlying_instrument:
            output[param.underlying_instrument.full_name] += (param,)
        else:
            output[None] += (param,)

    return output


def call_params_threaded(param_meas: Sequence[ParamMeasT]) -> OutType:
    """
    Function to create threads per instrument for the given set of
    measurement parameters.

    Args:
        param_meas: a Sequence of measurement parameters

    """

    inst_param_mapping = _instrument_to_param(param_meas)
    executors = tuple(
        _ParamCaller(*param_list) for param_list in inst_param_mapping.values()
    )

    output: OutType = []
    threads = [RespondingThread(target=executor) for executor in executors]

    for t in threads:
        t.start()

    for t in threads:
        thread_output = t.output()
        assert thread_output is not None
        for result in thread_output:
            output.append(result)

    return output


def _call_params(param_meas: Sequence[ParamMeasT]) -> OutType:
    from qcodes.parameters import ParameterBase

    output: OutType = []

    for parameter in param_meas:
        if isinstance(parameter, ParameterBase):
            output.append((parameter, parameter.get()))
        elif callable(parameter):
            parameter()

    return output


def process_params_meas(
    param_meas: Sequence[ParamMeasT], use_threads: bool | None = None
) -> OutType:
    from qcodes import config

    if use_threads is None:
        use_threads = config.dataset.use_threads

    if use_threads:
        return call_params_threaded(param_meas)

    return _call_params(param_meas)


class _ParamsCallerProtocol(Protocol):
    def __enter__(self) -> Callable[[], OutType]:
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


class SequentialParamsCaller(_ParamsCallerProtocol):
    def __init__(self, *param_meas: ParamMeasT):
        self._param_meas = tuple(param_meas)

    def __enter__(self) -> Callable[[], OutType]:
        return partial(_call_params, self._param_meas)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None


class ThreadPoolParamsCaller(_ParamsCallerProtocol):
    """
    Context manager for calling given parameters in a thread pool.
    Note that parameters that have the same underlying instrument will be
    called in the same thread.

    Usage:

        .. code-block:: python

           ...
           with ThreadPoolParamsCaller(p1, p2, ...) as pool_caller:
               ...
               output = pool_caller()
               ...
               # Output can be passed directly into DataSaver.add_result:
               # datasaver.add_result(*output)
               ...
           ...

    Args:
        param_meas: parameter or a callable without arguments
        max_workers: number of worker threads to create in the pool; if None,
            the number of worker threads will be equal to the number of
            unique "underlying instruments"
    """

    def __init__(self, *param_meas: ParamMeasT, max_workers: int | None = None):
        self._param_callers = tuple(
            _ParamCaller(*param_list)
            for param_list in _instrument_to_param(param_meas).values()
        )

        max_worker_threads = (
            len(self._param_callers) if max_workers is None else max_workers
        )
        thread_name_prefix = (
            self.__class__.__name__
            + ":"
            + "".join(" " + repr(pc) for pc in self._param_callers)
        )
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_worker_threads,
            thread_name_prefix=thread_name_prefix,
        )

    def __call__(self) -> OutType:
        """
        Call parameters in the thread pool and return `(param, value)` tuples.
        """
        output: OutType = list(
            itertools.chain.from_iterable(
                future.result()
                for future in concurrent.futures.as_completed(
                    self._thread_pool.submit(param_caller)
                    for param_caller in self._param_callers
                )
            )
        )

        return output

    def __enter__(self) -> ThreadPoolParamsCaller:
        self._thread_pool.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._thread_pool.__exit__(exc_type, exc_val, exc_tb)
