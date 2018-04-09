"""
This module defines sweep objects. It is anticipated that most users will
not access this module directly but instead shall use the qcodes.sweep.sweep
module to access convenience functions.
"""
from typing import Callable, Any, Iterator, Tuple, Optional, Dict, List, Sequence, Iterable
import itertools
import numpy as np
import time

import qcodes
from qcodes import Parameter

paramtabletype = Dict[str, List[Tuple[str,str]]]
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
    # Minus one values are unknown. Currently the only way to fill these in
    # with sensible values is by using the "sweep" method defined in this
    # module
    default_axis_properties = {
        "min": "?",
        "max": "?",
        "length": "?",
        "steps": "?"
    }

    def __init__(
            self, table_list: List[paramtabletype]=None,
            dependent_parameters: List[Tuple[str, str]]=None,
            independent_parameters: List[Tuple[str, str]]=None,
            inferred_from_dict: Dict[str,List[str]]=None
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

        self._inferred_from_dict: Dict[str,List[str]] = {}
        if inferred_from_dict is not None:
            self._inferred_from_dict = dict(inferred_from_dict)

        self._axis_info: Dict[str, str] = dict()

    def __add__(self, other: "ParametersTable")->"ParametersTable":
        """
        Add this table to another table. The table lists are simply appended.
        We add tables when we either chain or zip sweep objects

        Returns:
            ParametersTable
        """
        inferred_from_dict = dict(self._inferred_from_dict)
        inferred_from_dict.update(other.inferred_from_dict)

        axis_info = dict(self._axis_info)
        axis_info.update(other._axis_info)

        ptable = ParametersTable(
            self._table_list + other.table_list,
            inferred_from_dict=inferred_from_dict
        )
        ptable._axis_info = axis_info

        return ptable

    def __mul__(self, other: "ParametersTable")->"ParametersTable":
        """
        Multiply this table with another table.

        Returns:
            ParametersTable
        """
        inferred_from_dict = dict(self._inferred_from_dict)
        inferred_from_dict.update(other.inferred_from_dict)

        axis_info = dict(self._axis_info)
        axis_info.update(other._axis_info)

        ptable = ParametersTable([
            {k: s[k] + o[k] for k in s.keys()} for s, o in
            itertools.product(self._table_list, other.table_list)
        ], inferred_from_dict=inferred_from_dict)

        ptable._axis_info = axis_info

        return ptable

    def __repr__(self) -> str:
        """
        Return the string representation of the table
        """
        def table_print(table: paramtabletype) -> str:
            s = "|".join([
                ",".join(["{} [{}]".format(*a) for a in table[k]]) for k in
                ["independent_parameters", "dependent_parameters"]
            ])

            return s

        repr = self.inferred_symbols_list() + "\n" + "\n".join([
            table_print(table) for table in self._table_list
        ])

        return repr

    def _inferees_black_list(self) -> List[str]:
        black_list: List[str] = []
        for values in self._inferred_from_dict.values():
            black_list += values

        return black_list

    def flatten(self)->Tuple[Dict[str, str], Dict[str,str]]:
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

    def get_independents(self, exclude_inferees: bool =False) -> Dict[str,str]:
        black_list: List[str] = []
        if exclude_inferees:
            black_list = self._inferees_black_list()

        ind, _ = self.flatten()
        return {k: v for k, v in ind.items() if k not in black_list}

    def symbols_list(self)->List[str]:
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

    def set_axis_info(self, axis_info: Dict[str,str])->None:
        """
        If these are known, we can set the axis length of independent
        parameters
        """
        self._axis_info.update(axis_info)

    def layout_info(self, param_name: str) -> Dict[str, str]:

        table = None
        for t in self._table_list:
            if param_name in list(zip(*t["dependent_parameters"]))[0]:
                table = t
                break

        if table is None:
            raise ValueError(
                f"Parameter {param_name} not known or not an "
                f"dependent parameter"
            )

        independent_parameters = table["independent_parameters"]
        black_list = self._inferees_black_list()

        return {
            name: self._axis_info[name]
            for name, unit in independent_parameters
            if name not in black_list
        }

    def copy(self) -> "ParametersTable":
        """
        Copy this table

        Returns:
            ParametersTable
        """
        new_table = ParametersTable(
            self._table_list,
            inferred_from_dict=self._inferred_from_dict
        )

        new_table._axis_info = dict(
            self._axis_info
        )

        return new_table

    @property
    def table_list(self)->List[paramtabletype]:
        return self._table_list

    @property
    def inferred_from_dict(self)->Dict[str,List[str]]:
        return self._inferred_from_dict


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
        self._param_setter: Optional[Iterator] = None

        # A proper parameter table should be defined by the subclasses. This
        # is an instance of ParametersTable
        self._parameter_table: Optional[ParametersTable] = None
        self._symbols_list: Optional[List[str]] = None

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

    def __init__(self, iterator_function: Callable[[], Iterable])->None:
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

    def __init__(self, sweep_objects: List[BaseSweepObject])->None:
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

    def __init__(self, sweep_objects: List[BaseSweepObject])->None:
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

    def __init__(self, sweep_objects: List[BaseSweepObject])->None:
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
    def _combine_dictionaries(dictionaries: Sequence[dict])->dict:
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
