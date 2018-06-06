from typing import Iterator, Callable, List

from qcodes import Parameter, ParamSpec
from qcodes.sweep import param_table
from qcodes.sweep.param_table import ParamTable


class BaseSweepObject:
    """
    A sweep object is an iterable and at every iteration we produce a
    dictionary which is meant as input for the data saver class.
    """
    def __init__(self) ->None:

        self._generator: Iterator = None
        self._parameter_table: ParamTable = None
        self._symbols_list: list = None

    def _generator_factory(self) ->Iterator:
        """
        When called, this method returns the param setter iterable appropriate
        for this measurement
        """
        raise NotImplementedError("Please subclass BaseSweepObject")

    def _start_iter(self) ->None:
        self._generator = self._generator_factory()
        if self._parameter_table is not None:
            self._symbols_list = [spec.name for spec in
                                  self._parameter_table.param_specs]
        else:
            self._symbols_list = []

    def __iter__(self) ->'BaseSweepObject':
        self._start_iter()
        return self

    def __next__(self) ->dict:
        if self._generator is None:
            self._start_iter()

        nxt = next(self._generator)
        if len(nxt) and len(self._symbols_list):
            nxt = {k: nxt[k] if k in nxt else None for k in self._symbols_list}

        return nxt

    @property
    def parameter_table(self) ->ParamTable:
        return self._parameter_table


class IteratorSweep(BaseSweepObject):
    """
    Sweep independent parameters by unrolling an iterator. This class is useful
    if we have "bare" parameter set iterator
    and need to create a proper sweep object as defined in the BaseSweepObject
    docstring. See the "Nest" subclass for an example.
    Parameters
    ----------
    iterator_function: callable
        A callable with no parameters, returning an iterator. Unrolling this
        iterator has the effect of setting the independent parameters.
    """

    def __init__(self, iterator_function: Callable):
        super().__init__()
        self._iterator_function = iterator_function

    def _generator_factory(self) ->Iterator:
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

    def __init__(self, sweep_objects: List[BaseSweepObject]) ->None:
        super().__init__()
        self._sweep_objects = sweep_objects
        self._parameter_table = param_table.prod(
            [so.parameter_table for so in sweep_objects]
        )

    @staticmethod
    def _two_product(sweep_object1: BaseSweepObject,
                     sweep_object2: BaseSweepObject) ->IteratorSweep:
        def inner():
            for result2 in sweep_object2:
                for result1 in sweep_object1:
                    result1.update(result2)

                    yield result1

        return IteratorSweep(inner)

    def _generator_factory(self) ->Iterator:
        prod = self._sweep_objects[0]
        for so in self._sweep_objects[1:]:
            prod = self._two_product(so, prod)

        return prod


class Chain(BaseSweepObject):
    """
    Chain a list of sweep object to run one after the other
    """

    def __init__(self, sweep_objects: List[BaseSweepObject]) ->None:
        super().__init__()
        self._sweep_objects = sweep_objects
        self._parameter_table = param_table.add(
            [so.parameter_table for so in sweep_objects]
        )

    def _generator_factory(self) ->Iterator:
        for so in self._sweep_objects:
            for result in so:
                yield result


class ParameterSweep(BaseSweepObject):
    """
    Sweep independent parameters by looping over set point values and setting
    a QCoDeS parameter to this value at each iteration
    Parameters
    ----------
    parameter: qcodes.StandardParameter
    point_function: callable
        Unrolling this iterator returns to us set values of the parameter
    """

    def __init__(self, parameter: Parameter, point_function: Callable) ->None:
        super().__init__()
        self._parameter = parameter
        self._point_function = point_function
        self._parameter_table = ParamTable([
            ParamSpec(
                name=parameter.name,
                paramtype='numeric',
                unit=parameter.unit,
                label=parameter.label
            )
        ])

    def _generator_factory(self)->Iterator:
        for set_value in self._point_function():
            self._parameter.set(set_value)
            yield {self._parameter.full_name: set_value}


class ParameterWrapper(BaseSweepObject):
    """
    A wrapper class which iterates once ans returns the value of a QCoDeS
    parameter
    Parameters
    ----------
    parameter: qcodes.StandardParameter
    """

    def __init__(self, parameter: Parameter)->None:
        super().__init__()

        self._parameter_table = ParamTable([
            ParamSpec(
                name=parameter.name,
                paramtype='numeric',
                unit=parameter.unit,
                label=parameter.label
            )
        ])

        self._parameter = parameter

    def _generator_factory(self)->Iterator:
        yield {self._parameter.full_name: self._parameter.get()}