# Instead of async, since there are really a very small set of things
# we want to happen simultaneously within one process (namely getting
# several parameters in parallel), we can parallelize them with threads.
# That way the things we call need not be rewritten explicitly async.
import logging
import threading
from collections import defaultdict
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, TypeVar, Tuple, Union
)
import concurrent
import concurrent.futures
import itertools

from qcodes.dataset.measurements import res_type
from qcodes.instrument.parameter import ParamDataType, _BaseParameter
from qcodes import config

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
        return f"ParamCaller of {names}"


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


_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, (os.cpu_count() or 1) + 4),
    thread_name_prefix="call_params",
)


def call_params_in_thread_pool(param_meas: Sequence[ParamMeasT]) -> OutType:
    inst_param_mapping = _instrument_to_param(param_meas)
    executors = tuple(
        _ParamCaller(*param_list)
        for param_list in inst_param_mapping.values()
    )

    output: OutType = list(itertools.chain.from_iterable(
        future.result()
        for future in concurrent.futures.as_completed(
            _thread_pool.submit(executor) for executor in executors
        )
    ))

    return output


class ThreadPoolParamsCaller:
    def __init__(self, param_meas: Sequence[ParamMeasT]):
        self._param_callers = tuple(
            _ParamCaller(*param_list)
            for param_list in _instrument_to_param(param_meas).values()
        )

        max_worker_threads = min(32, (os.cpu_count() or 1) + 4)
        thread_name_prefix = "params_call" + "".join(
            "__" + repr(pc).replace(" ", "_")
            for pc in self._param_callers
        )
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_worker_threads,
            thread_name_prefix=thread_name_prefix,
        )

    def __call__(self) -> OutType:
        output: OutType = list(itertools.chain.from_iterable(
            future.result()
            for future in concurrent.futures.as_completed(
                self._thread_pool.submit(param_caller)
                for param_caller in self._param_callers
            )
        ))

        return output
