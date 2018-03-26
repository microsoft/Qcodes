"""
This module defines sweep objects and convenience functions associated.

TODO:
1) Add proper docstrings in google format
2) Add type hints
"""
from typing import Callable, Any, Iterable, Iterator, Tuple, Union
import itertools
import numpy as np
import time

import qcodes
from qcodes import Parameter


class ParametersTable:
    """
    A parameters table is how sweep objects keep track of which parameters
    have been defined and how they relate to each
    other. Please have a look at the sweep_basic_example.ipynb notebook
    in docs/examples/sweep for a better understanding

    When a sweep object is registered by the SweepMeasurement class
    (see sweep_measurement.py), this table is used to create the ParamSpecs.

    Args:
        table_list (list): A list of dictionaries with keys
            "dependent_parameters" and "independent_parameters". The values are
            lists of tuples.
        dependent_parameters (list): A list of tuples (<name>, <unit>)
        independent_parameters (list): A list of tuples (<name>, <unit>)
        inferred_from_dict (dict): A dictionary where the keys are the inferred
            parameters and the values are the parameters from which this is
            inferred
    """

    def __init__(
            self, table_list: list=None, dependent_parameters: list=None,
            independent_parameters: list=None, inferred_from_dict: dict=None
    )->None:

        if not any([table_list, dependent_parameters, independent_parameters]):
            raise ValueError("At least one argument should be a non-None")

        if dependent_parameters is None:
            dependent_parameters = []

        if independent_parameters is None:
            independent_parameters = []

        if table_list is None:
            self._table_list = [{
                "dependent_parameters": list(dependent_parameters),
                "independent_parameters": list(independent_parameters)
            }]
        else:
            self._table_list = list(table_list)

        self._inferred_from_dict = {}
        if inferred_from_dict is not None:
            self._inferred_from_dict = dict(inferred_from_dict)

    def copy(self)->"ParametersTable":
        """
        Copy this table

        Returns:
            ParametersTable
        """
        return ParametersTable(
            self._table_list,
            inferred_from_dict=self._inferred_from_dict
        )

    def __add__(self, other: "ParametersTable")->"ParametersTable":
        """
        Add this table to another table. The table lists are simply appended.
        We add tables when we either chain or zip sweep objects

        Returns:
            ParametersTable
        """
        inferred_from_dict = dict(self._inferred_from_dict)
        inferred_from_dict.update(other.inferred_from_dict)

        return ParametersTable(
            self._table_list + other.table_list,
            inferred_from_dict=inferred_from_dict
        )

    def __mul__(self, other: "ParametersTable")->"ParametersTable":
        """
        Multiply this table with another table. We mul

        Returns:
            ParametersTable
        """
        inferred_from_dict = dict(self._inferred_from_dict)
        inferred_from_dict.update(other.inferred_from_dict)

        return ParametersTable([
            {k: s[k] + o[k] for k in s.keys()} for s, o in
            itertools.product(self._table_list, other.table_list)
        ], inferred_from_dict=inferred_from_dict)

    def __repr__(self):
        """
        Return the string representation of the table
        """
        def table_print(table):
            s = "|".join([
                ",".join(["{} [{}]".format(*a) for a in table[k]]) for k in
                ["independent_parameters", "dependent_parameters"]
            ])

            return s

        repr = self.inferred_symbols_list() + "\n" + "\n".join([
            table_print(table) for table in self._table_list
        ])

        return repr

    def flatten(self)->tuple:
        """
        Return a flattened list of dependent and independent parameters.
        Dependency information is lost by flattening (that is, if e.g.
        parameter A is dependent, we lose info on which parameters it is
        dependent on)
        """
        ind = {k: v for d in self._table_list for k, v in
               d["independent_parameters"]}
        dep = {k: v for d in self._table_list for k, v in
               d["dependent_parameters"]}
        return ind, dep

    def symbols_list(self)->list:
        """
        Return a list of parameter names
        """
        ind, dep = self.flatten()
        return list(ind.keys()) + list(dep.keys())

    def inferred_symbols_list(self)->str:
        """
        Return the inferred parameter names as a string
        """
        return "\n".join(["{symbol} inferred from {inferrees}".format(
            symbol=symbol, inferrees=",".join(inferrees))
            for symbol, inferrees in self._inferred_from_dict.items()]
        )

    @property
    def table_list(self)->list:
        return self._table_list

    @property
    def inferred_from_dict(self)->dict:
        return self._inferred_from_dict


def getter(param_list: list, inferred_parameters: list=None)->Callable:
    """
    A decorator to easily integrate arbitrary measurement functions in sweeps.

    Args:
        param_list (list): A list of tuples (<name>, <unit>). For example,
            [("current", "A), ("bias", "V")]
        inferred_parameters (list): List of inferred parameters as a list of
            tuples (<name>, <unit>)

    Returns:
        decorator (Callable).

    Example:
        >>> @getter([("meas", "H")])
        >>> def measurement_function():
        >>>     return np.random.normal(0, 1)
        >>> sweep(p, [0, 1, 2])(measurement_function)

        More elaborate examples are available in getters_and_setters.ipynb
        in the folder  docs/examples/sweep.
    """
    if inferred_parameters is None:
        inferred_parameters = []
        inferred_symbols = []
    else:
        inferred_symbols, _ = list(zip(*inferred_parameters))

    symbols_not_inferred, _ = [list(i) for i in zip(*param_list)]
    # Do not do >>> param_list += inferred_from ; lists are mutable
    param_list = param_list + inferred_parameters

    def decorator(f):
        def inner():
            value = np.atleast_1d(f())

            if len(value) != len(param_list + inferred_parameters):
                raise ValueError(
                    "The number of supplied inferred parameters "
                    "does not match the number returned by the "
                    "getter function"
                )

            return {p[0]: im for p, im in zip(param_list, value)}

        inferred_from_dict = {inferred_symbol: symbols_not_inferred for
                              inferred_symbol in inferred_symbols}

        parameter_table = ParametersTable(
            dependent_parameters=param_list,
            inferred_from_dict=inferred_from_dict
        )

        return lambda: (inner, parameter_table)

    return decorator


def setter(param_list: list, inferred_parameters: list=None)->Callable:
    """
    A decorator to easily integrate arbitrary setter functions in sweeps

    Args:
        param_list (list): A list of tuples (<name>, <unit>). For example,
            [("current", "A), ("bias", "V")]
        inferred_parameters (list): List of inferred parameters as a list of
            tuples (<name>, <unit>)

    Returns:
        decorator (Callable).

    """
    if inferred_parameters is None:
        inferred_parameters = []
        inferred_symbols = []
    else:
        inferred_symbols, _ = list(zip(*inferred_parameters))

    symbols_not_inferred, _ = [list(i) for i in zip(*param_list)]
    # Do not do >>> param_list += inferred_from ; lists are mutable
    param_list = param_list + inferred_parameters

    def decorator(f):
        def inner(value):
            value = np.atleast_1d(value)
            inferred_values = f(*value)

            if inferred_values is not None:
                value = np.append(value, np.atleast_1d(inferred_values))

            if len(param_list) != len(value):
                raise ValueError("The number of supplied inferred parameters "
                                 "does not match the number returned by the "
                                 "setter function")

            return {p[0]: im for p, im in zip(param_list, value)}

        inferred_from_dict = {inferred_symbol: symbols_not_inferred for
                              inferred_symbol in inferred_symbols}

        parameter_table = ParametersTable(
            independent_parameters=param_list,
            inferred_from_dict=inferred_from_dict
        )

        return lambda: (inner, parameter_table)

    return decorator


def wrap_objects(*objects: Any, repeat: bool=False):
    """
    A helper function to make sweep functions. For instance, we can wrap a
    QCoDeS parameter to make it look like a sweep object. If repeat = False,
    this sweep object will iterate once, returning the parameter.get().
    Wrapping a function will call the function after iterating once. This
    function has to be decorated with the getter.

    Args:
        objects (list): Objects to be wrapped
        repeat (bool): If repeat is False, the function wrapper and parameter
            wrapper will iterate once. If True, these will iterate
            indefinitely. We therefore need to be careful: the following will
            cause an infinite loop:
            >>> for i in ParameterWrapper(p, repeat=True):
            >>>     print(i)
            Repeat is True is useful if we use Zip to zip the infinite sweep
            object with one that is finite. An example is provided in the
            docstring of szip.
    """
    def wrapper(obj):
        if isinstance(obj, qcodes.Parameter):
            new_obj = ParameterWrapper(obj, repeat=repeat)
        elif isinstance(obj, BaseSweepObject):
            new_obj = obj
        elif callable(obj):
            new_obj = FunctionWrapper(obj, repeat=repeat)
        else:
            raise ValueError("Do not know how to wrap instance of ", type(obj))

        return new_obj

    return [wrapper(obj) for obj in objects]


class BaseSweepObject:

    def __init__(self)->None:
        # A "param_setter" is an iterator which, when "next" is called a new
        # value of the independent parameter is set.
        self._param_setter = None

        # A proper parameter table should be defined by the subclasses. This
        # is an instance of ParametersTable
        self._parameter_table = None
        self._symbols_list = None

    def _setter_factory(self)->Iterator:
        """
        When called, this method returns the param setter Iterator appropriate
        for this measurement. An iterator is an object with a __next__ method
        (e.g. a generator). Each next call on this iterator yields a
        dictionary with the current value of the parameters.
        """
        raise NotImplementedError("Please subclass BaseSweepObject")

    def _start_iter(self)->None:
        self._param_setter = self._setter_factory()
        if self._parameter_table is not None:
            self._symbols_list = self._parameter_table.symbols_list()
        else:
            self._symbols_list = []

    def __iter__(self)->"BaseSweepObject":
        self._start_iter()
        return self

    def __next__(self)->dict:
        """
        At each iteration a dictionary is returned containing information about
        the parameters set and measurements that have been performed.
        """
        if self._param_setter is None:
            self._start_iter()

        nxt = next(self._param_setter)
        if len(nxt) and len(self._symbols_list):
            nxt = {k: nxt[k] if k in nxt else None for k in self._symbols_list}

        return nxt

    def __call__(self, *sweep_objects: list)->"BaseSweepObject":
        """
        We implement a convenient syntax to create nested sweep objects
        """
        return Nest([self, Chain(wrap_objects(*sweep_objects))])

    @property
    def parameter_table(self)->ParametersTable:
        return self._parameter_table


class IteratorSweep(BaseSweepObject):
    """
    Sweep independent parameters by unrolling an iterator. This class is useful
    if we have "bare" parameter set iterator and need to create a proper
    sweep object See the "Nest" subclass for an example.

    Parameters
    ----------
    iterator_function: callable
        A callable with no parameters, returning an iterator. Unrolling this
        iterator has the effect of setting the independent parameters.
    """

    def __init__(self, iterator_function: callable)->None:
        super().__init__()
        self._iterator_function = iterator_function

    def _setter_factory(self)->Iterator:
        for value in self._iterator_function():
            yield value


class Nest(BaseSweepObject):
    """
    Nest multiple sweep objects. This is for example very useful when
    performing two or higher dimensional scans
    (e.g. sweep two gate voltages and measure a current at each coordinate
    (gate1, gate2).

    Notes
    -----
    We produce a nested sweep object of arbitrary depth by first defining a
    function which nests just two sweep objects

        product = two_product(so1, so2)

    A third order nest is then achieved like so:

        product = two_product(so1, two_product(so2, so3))

    A fourth order by

        product = two_product(so1, two_product(so2, two_product(so3, so4)))

    Etc...
    """

    def __init__(self, sweep_objects: list)->None:
        """
        Args:
            sweep_objects (list): A list of sweep objects
        """
        super().__init__()
        self._sweep_objects = sweep_objects
        self._parameter_table = np.prod(
            [so.parameter_table for so in sweep_objects]
        )

    @staticmethod
    def _two_product(sweep_object1: BaseSweepObject,
                     sweep_object2: BaseSweepObject) ->IteratorSweep:
        """
        Args:
            sweep_object1 (BaseSweepObject)
            sweep_object2 (BaseSweepObject)

        Returns:
            IteratorSweep
        """

        def inner():
            for result2 in sweep_object2:
                for result1 in sweep_object1:
                    result1.update(result2)

                    yield result1

        return IteratorSweep(inner)

    def _setter_factory(self)->Iterator:
        prod = self._sweep_objects[0]
        for so in self._sweep_objects[1:]:
            prod = self._two_product(so, prod)

        return prod


class Chain(BaseSweepObject):
    """
    Chain a list of sweep object to run one after the other
    """

    def __init__(self, sweep_objects: list)->None:
        """
        Args:
            sweep_objects (list): A list of sweep objects
        """
        super().__init__()
        self._sweep_objects = sweep_objects
        self._parameter_table = np.sum(
            [so.parameter_table for so in sweep_objects]
        )

    def _setter_factory(self)->Iterator:
        for so in self._sweep_objects:
            for result in so:
                yield result


class Zip(BaseSweepObject):
    """
    Zip multiple sweep objects. Unlike a nested sweep, we will produce a 1D
    sweep
    """

    def __init__(self, sweep_objects: list)->None:
        """
        Args:
            sweep_objects (list): A list of sweep objects
        """
        super().__init__()
        self._sweep_objects = sweep_objects
        self._parameter_table = np.prod(
            [so.parameter_table for so in sweep_objects]
        )

    @staticmethod
    def _combine_dictionaries(dictionaries: Tuple[Any])->dict:
        combined = {}
        for d in dictionaries:
            combined.update(d)
        return combined

    def _setter_factory(self)->Iterator:
        for results in zip(*self._sweep_objects):
            yield Zip._combine_dictionaries(results)


class ParameterSweep(BaseSweepObject):
    """
    Sweep independent parameters by looping over set point values and setting
    a QCoDeS parameter to this value at each iteration

    Args:
        parameter (qcodes.Parameter)
        point_function (callable):
            Calling this function returns to us an iterable where each
            iteration represents a set point
    """

    def __init__(self, parameter: Parameter, point_function: Callable)->None:
        super().__init__()
        self._parameter = parameter
        self._point_function = point_function
        self._parameter_table = ParametersTable(
            independent_parameters=[(parameter.full_name, parameter.unit)]
        )

    def _setter_factory(self)->Iterator:
        for set_value in self._point_function():
            self._parameter.set(set_value)
            yield {self._parameter.full_name: set_value}


class ParameterWrapper(BaseSweepObject):
    """
    Wrap a parameter to make it look like a sweep object.
    """

    def __init__(self, parameter: Parameter, repeat: bool=False) ->None:
        """
        Args:
            parameter (Parameter)
            repeat (bool):
                When repeat=True, keep yielding the current value of
                the parameter indefinitely. Please use this carefully as this
                can easily lead to infinite loops.
        """
        super().__init__()
        self._parameter_table = ParametersTable(
            dependent_parameters=[(parameter.full_name, parameter.unit)]
        )
        self._parameter = parameter
        self._repeat = repeat

    def _setter_factory(self)->Iterator:
        stop = False
        while not stop:
            value = self._parameter()
            yield {self._parameter.full_name: value}
            stop = not self._repeat


class FunctionSweep(BaseSweepObject):
    """
    Use a function to set independent parameters.
    """

    def __init__(self, set_function: Callable, point_function: Callable)->None:
        """
        Args:
            set_function (Callable):
                The function which is used to set independent parameters. This
                function needs to be decorated with the setter for us to be
                able to extract the parameter table
             point_function (Callable):
                Calling this function returns to us an iterable where each
                iteration represents a set point
        """
        super().__init__()
        self._set_function, self._parameter_table = set_function()
        self._point_function = point_function

    def _setter_factory(self)->Iterator:
        for set_value in self._point_function():
            yield self._set_function(set_value)


class FunctionWrapper(BaseSweepObject):
    """
    Use a function to measure dependent parameters instead of calling get
    methods directly
    """

    def __init__(self, measure_function: Callable, repeat: bool=False)->None:
        """
        Args:
            measure_function (Callable):
                A function decorated with the getter decorator
            repeat (bool):
                When repeat=True, keep yielding the current value of
                the parameter indefinitely. Please use this carefully as this
                can easily lead to infinite loops.
        """
        super().__init__()
        self._measure_function, self._parameter_table = measure_function()
        self._repeat = repeat

    def _setter_factory(self)->Iterator:
        stop = False
        while not stop:
            yield self._measure_function()
            stop = not self._repeat


class TimeTrace(BaseSweepObject):
    """
    By nesting a ParameterWrapper or a FunctionWrapper sweep object in a
    TimeTrace sweep object, we can monitor a parameter or function over a given
    period of time. Please see time_trace.ipynb in the folder
    docs/examples/sweep for details.
    """
    def __init__(self, interval_time: float, total_time: float=None)->None:
        """
        Args:
            interval_time (float)
            total_time (float):
                If this is None, the time trace will run indefinitely. An
                infinite loop can be prevented by zipping the time trace with
                a sweep object of finite length
        """
        super().__init__()

        self._parameter_table = ParametersTable(
            independent_parameters=[("time", "s")]
        )

        if total_time is None:
            total_time = np.inf

        self._total_time = total_time
        self._interval_time = interval_time

    def _setter_factory(self)->Iterator:
        t0 = time.time()
        t = t0
        while t - t0 < self._total_time:
            yield {"time": (t - t0)}
            time.sleep(self._interval_time)
            t = time.time()


class While(BaseSweepObject):
    """
    Return values from a measurement function until a None is returned
    """

    def __init__(self, measure_function: Callable)->None:
        super().__init__()
        self._measure_function, self._parameter_table = measure_function()

    def _setter_factory(self)->Iterator:

        while True:
            measure_value = self._measure_function()
            if None not in measure_value.values():
                yield measure_value
            else:
                break


def sweep(
        obj: Union[Parameter, Callable],
        sweep_points: Union[Iterable, Callable]
)->BaseSweepObject:
    """
    A convenience function to create a 1D sweep object

    Args:
        obj (Parameter or callable):
            If callable, a function decorated with setter
        sweep_points (iterable or callable):
            If callable, it shall be a callable of no parameters

    Returns:
        FunctionSweep or ParameterSweep
    """

    if not callable(sweep_points):
        point_function = lambda: sweep_points
    else:
        point_function = sweep_points

    if not isinstance(obj, qcodes.Parameter):
        if not callable(obj):
            raise ValueError(
                "The object to sweep over needs to either be a QCoDeS "
                "parameter or a function"
            )

        return FunctionSweep(obj, point_function)
    else:
        return ParameterSweep(obj, point_function)


def nest(*objects: BaseSweepObject)->BaseSweepObject:
    """
    Convenience function to create a nested sweep

    Args:
        objects (list): List of Sweep object to nest

    Returns:
        Nested sweep object
    """
    return Nest(wrap_objects(*objects))


def chain(*objects:  BaseSweepObject)->BaseSweepObject:
    """
    Convenience function to create a chained sweep

    Args:
        objects (list): List of Sweep object to chain

    Returns:
        Chained sweep object
    """
    return Chain(wrap_objects(*objects))


def szip(*objects:  BaseSweepObject)->BaseSweepObject:
    """
    A plausible scenario for using szip is the following:

    >>> szip(therometer.t, sweep(source.voltage, [0, 1, 2]))

    The idea is to measure the temperature *before* going to each voltage set
    point. The parameter "thermometer.t" needs to be wrapped by the
    ParameterWrapper in such a way that the get method of the parameter is
    called repeatedly. An infinite loop is prevented because
    "sweep(source.voltage, [0, 1, 2])"  has a finite length and the
    Zip operator loops until the shortest sweep object is exhausted

    Args:
        objects (list): List of Sweep object to zip

    Returns:
        Zipped sweep object
    """
    repeat = False
    if any([isinstance(i, BaseSweepObject) for i in objects]):
        repeat = True

    return Zip(wrap_objects(*objects, repeat=repeat))


def time_trace(
        measurement_object: [Callable, Parameter, BaseSweepObject],
        interval_time: float,
        total_time: float
):
    """
    Make time trace sweep object to monitor the return value of the measurement
    object over a certain time period.

    Args:
        measurement_object:
            This can be; a function decorated with the getter decorator,
            a QCoDeS parameter, or another sweep object
        interval_time (float)
        total_time (float)
    """
    tt_sweep = TimeTrace(interval_time, total_time)
    return szip(measurement_object, tt_sweep)
