# Instead of async, since there are really a very small set of things
# we want to happen simultaneously within one process (namely getting
# several parameters in parallel), we can parallelize them with threads.
# That way the things we call need not be rewritten explicitly async.
import concurrent
import concurrent.futures
import itertools
import logging
import threading
from collections import defaultdict
from functools import partial
from types import TracebackType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import Protocol

from qcodes import config
from qcodes.dataset.measurements import res_type
from qcodes.instrument.parameter import ParamDataType, _BaseParameter

ParamMeasT = Union[_BaseParameter, Callable[[], None]]

OutType = List[res_type]

T = TypeVar("T")

log = logging.getLogger(__name__)


class RespondingThread(threading.Thread):
    """
    Thread subclass for parallelizing execution. Behaves like a
    regular thread but returns a value from target, and propagates
    exceptions back to the main thread when this value is collected.

    The `output` method joins the thread, then checks for errors and
    returns the output value.

    so, if you have a function `f` where `f(1, 2, a=3) == 4`, then:

    >>> thread = RespondingThread(target=f, args=(1, 2), kwargs={'a': 3})
    >>> thread.start()
    >>> # do other things while this is running
    >>> out = thread.output()  # out is 4
    """
    def __init__(self, target: Callable[..., T], args: Sequence[Any] = (),
                 kwargs: Optional[Dict[str, Any]] = None,
                 *args2: Any, **kwargs2: Any):
        if kwargs is None:
            kwargs = {}

        super().__init__(*args2, **kwargs2)

        self._target = target
        self._args = args
        self._kwargs = kwargs
        self._exception: Optional[Exception] = None
        self._output: Optional[T] = None

    def run(self) -> None:
        log.debug(
            f"Executing {self._target} on thread: {threading.get_ident()}"
        )
        try:
            self._output = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self._exception = e

    def output(self, timeout: Optional[float] = None) -> Optional[T]:
        self.join(timeout=timeout)

        if self._exception:
            e = self._exception
            self._exception = None
            raise e

        return self._output


def thread_map(
        callables: Sequence[Callable[..., T]],
        args: Optional[Sequence[Sequence[Any]]] = None,
        kwargs: Optional[Sequence[Dict[str, Any]]] = None
) -> List[Optional[T]]:
    """
    Evaluate a sequence of callables in separate threads, returning
    a list of their return values.

    Args:
        callables: A sequence of callables.
        args (Optional): A sequence of sequences containing the positional
            arguments for each callable.
        kwargs (Optional): A sequence of dicts containing the keyword arguments
            for each callable.

    """
    if args is None:
        args = ((),) * len(callables)
    if kwargs is None:
        empty_dict: Dict[str, Any] = {}
        kwargs = (empty_dict,) * len(callables)
    threads = [RespondingThread(target=c, args=a, kwargs=k)
               for c, a, k in zip(callables, args, kwargs)]

    for t in threads:
        t.start()

    return [t.output() for t in threads]


class _ParamCaller:

    def __init__(self, *parameters: _BaseParameter):

        self._parameters = parameters

    def __call__(self) -> Tuple[Tuple[_BaseParameter, ParamDataType], ...]:
        output = []
        for param in self._parameters:
            output.append((param, param.get()))
        return tuple(output)

    def __repr__(self) -> str:
        names = tuple(param.full_name for param in self._parameters)
        return f"ParamCaller of {','.join(names)}"


def _instrument_to_param(
        params: Sequence[ParamMeasT]
) -> Dict[Optional[str], Tuple[_BaseParameter, ...]]:

    real_parameters = [param for param in params
                       if isinstance(param, _BaseParameter)]

    output: Dict[Optional[str], Tuple[_BaseParameter, ...]] = defaultdict(tuple)
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
    executors = tuple(_ParamCaller(*param_list)
                      for param_list in
                      inst_param_mapping.values())

    output: OutType = []
    threads = [RespondingThread(target=executor)
               for executor in executors]

    for t in threads:
        t.start()

    for t in threads:
        thread_output = t.output()
        assert thread_output is not None
        for result in thread_output:
            output.append(result)

    return output


def _call_params(param_meas: Sequence[ParamMeasT]) -> OutType:

    output: OutType = []

    for parameter in param_meas:
        if isinstance(parameter, _BaseParameter):
            output.append((parameter, parameter.get()))
        elif callable(parameter):
            parameter()

    return output


def process_params_meas(
    param_meas: Sequence[ParamMeasT],
    use_threads: Optional[bool] = None
) -> OutType:

    if use_threads is None:
        use_threads = config.dataset.use_threads

    if use_threads:
        return call_params_threaded(param_meas)

    return _call_params(param_meas)


class _ParamsCallerProtocol(Protocol):
    def __enter__(self) -> Callable[[], OutType]:
        pass

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass


class SequentialParamsCaller(_ParamsCallerProtocol):
    def __init__(self, *param_meas: ParamMeasT):
        self._param_meas = tuple(param_meas)

    def __enter__(self) -> Callable[[], OutType]:
        return partial(_call_params, self._param_meas)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
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

    def __init__(self, *param_meas: ParamMeasT, max_workers: Optional[int] = None):
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

    def __enter__(self) -> "ThreadPoolParamsCaller":
        self._thread_pool.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self._thread_pool.__exit__(exc_type, exc_val, exc_tb)
