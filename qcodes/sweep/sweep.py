import itertools
import numpy as np


class ParametersTable:
    def __init__(self, table_list=None, dependent_parameters=None, independent_parameters=None):

        if not any([table_list, dependent_parameters, independent_parameters]):
            raise ValueError("At least one argument should be a non-None")

        dependent_parameters = [] if dependent_parameters is None else dependent_parameters
        independent_parameters = [] if independent_parameters is None else independent_parameters

        if table_list is None:
            self._table_list = [{
                "dependent_parameters": list(dependent_parameters),
                "independent_parameters": list(independent_parameters)
            }]
        else:
            self._table_list = list(table_list)

    def copy(self):
        return ParametersTable(self._table_list)

    def __add__(self, other):
        return ParametersTable(self._table_list + other.table_list)

    def __mul__(self, other):
        return ParametersTable([
            {k: s[k] + o[k] for k in s.keys()} for s, o in itertools.product(self._table_list, other.table_list)
        ])

    def __repr__(self):
        def table_print(table):
            s = "|".join([
                ",".join(["{} [{}]".format(*a) for a in table[k]]) for k in ["independent_parameters",
                                                                             "dependent_parameters"]
            ])

            return s

        return "\n".join([table_print(table) for table in self._table_list])

    @property
    def table_list(self):
        return self._table_list


def measurement(param_list):
    """
    A decorator to easily integrate arbitrary measurement functions in sweeps.
    """
    def decorator(f):
        def inner():
            m = np.atleast_1d(f())
            return {p[0]: im for p, im in zip(param_list, m)}

        parameter_table = ParametersTable(dependent_parameters=param_list)
        return lambda: (inner, parameter_table)

    return decorator


def setter(param_list):
    """
    A decorator to easily integrate arbitrary setter functions in sweeps
    """
    def decorator(f):
        def inner(*value):
            f(*value)
            return {p[0]: im for p, im in zip(param_list, value)}

        parameter_table = ParametersTable(independent_parameters=param_list)
        return lambda: (inner, parameter_table)

    return decorator


class BaseSweepObject:
    """
    A sweep object is defined as follows:
    1) It is iterable and looping over a sweep object shall result in independent measurement parameters being set
    at each iteration
    2) At each iteration a dictionary is returned containing information about the parameters set and measurements that
    have been performed.
    """

    def __init__(self):
        # The following attributes are determined when we begin iteration...
        self._param_setter = None  # A "param_setter" is an iterator which, when "next" is called a new value of the
        # independent parameter is set.
        self._parameter_table = None  # A proper parameter table should be defined by the subclasses

    def _setter_factory(self):
        """
        When called, this method returns the param setter iterable appropriate for this measurement
        """
        raise NotImplementedError("Please subclass BaseSweepObject")

    def __iter__(self):
        self._param_setter = self._setter_factory()
        return self

    def __next__(self):
        if self._param_setter is None:
            self._param_setter = self._setter_factory()
        return next(self._param_setter)

    @property
    def parameter_table(self):
        return self._parameter_table


class IteratorSweep(BaseSweepObject):
    """
    Sweep independent parameters by unrolling an iterator. This class is useful if we have "bare" parameter set iterator
    and need to create a proper sweep object as defined in the BaseSweepObject docstring. See the "Nest"
    subclass for an example.

    Parameters
    ----------
    iterator_function: callable
        A callable with no parameters, returning an iterator. Unrolling this iterator has the
        effect of setting the independent parameters.
    """
    def __init__(self, iterator_function):
        super().__init__()
        self._iterator_function = iterator_function

    def _setter_factory(self):
        for value in self._iterator_function():
            yield value


class Nest(BaseSweepObject):
    """
    Nest multiple sweep objects. This is for example very useful when performing two or higher dimensional scans
    (e.g. sweep two gate voltages and measure a current at each coordinate (gate1, gate2).

    Notes
    -----
    We produce a nested sweep object of arbitrary depth by first defining a function which nests just two sweep objects

        product = two_product(so1, so2)

    A third order nest is then achieved like so:

        product = two_product(so1, two_product(so2, so3))

    A fourth order by

        product = two_product(so1, two_product(so2, two_product(so3, so4)))

    Etc...
    """

    def __init__(self, sweep_objects):
        super().__init__()
        self._sweep_objects = sweep_objects
        self._parameter_table = np.prod([so.parameter_table for so in sweep_objects])

    @staticmethod
    def _two_product(sweep_object1, sweep_object2):
        """
        Parameters
        ----------
        sweep_object1: BaseSweepObject
        sweep_object2: BaseSweepObject

        Returns
        -------
        BaseSweepObject
        """

        def inner():
            for result2 in sweep_object2:
                for result1 in sweep_object1:
                    result1.update(result2)

                    yield result1

        return IteratorSweep(inner)

    def _setter_factory(self):
        prod = self._sweep_objects[0]
        for so in self._sweep_objects[1:]:
            prod = self._two_product(so, prod)

        return prod


class Chain(BaseSweepObject):
    """
    Chain a list of sweep object to run one after the other
    """
    def __init__(self, sweep_objects):
        super().__init__()
        self._sweep_objects = sweep_objects
        self._parameter_table = np.sum([so.parameter_table for so in sweep_objects])

    def _setter_factory(self):
        for so in self._sweep_objects:
            for result in so:
                yield result


class Zip(BaseSweepObject):
    """
    Zip multiple sweep objects. Unlike a nested sweep, we will produce a 1D sweep
    """
    def __init__(self, sweep_objects):
        super().__init__()
        self._sweep_objects = sweep_objects
        self._parameter_table = np.prod([so.parameter_table for so in sweep_objects])

    @staticmethod
    def _combine_dictionaries(dictionaries):
        combined = {}
        for d in dictionaries:
            combined.update(d)
        return combined

    def _setter_factory(self):
        for results in zip(*self._sweep_objects):
            yield Zip._combine_dictionaries(results)


class ParameterSweep(BaseSweepObject):
    """
    Sweep independent parameters by looping over set point values and setting a QCoDeS parameter to this value at
    each iteration

    Parameters
    ----------
    parameter: qcodes.StandardParameter
    point_function: callable
        Unrolling this iterator returns to us set values of the parameter
    """

    def __init__(self, parameter, point_function):
        super().__init__()
        self._parameter = parameter
        self._point_function = point_function
        self._parameter_table = ParametersTable(independent_parameters=[(parameter.full_name, parameter.unit)])

    def _setter_factory(self):
        for set_value in self._point_function():
            self._parameter.set(set_value)
            yield {self._parameter.full_name: set_value}


class ParameterWrapper(BaseSweepObject):
    """
    A wrapper class which iterates once ans returns the value of a QCoDeS parameter

    Parameters
    ----------
    parameter: qcodes.StandardParameter
    """
    def __init__(self, parameter, repeat=False):
        """
        Note: please use repeat=True carefully as this can easily lead to infinite loops.
        """
        super().__init__()
        self._parameter_table = ParametersTable(dependent_parameters=[(parameter.full_name, parameter.unit)])
        self._parameter = parameter
        self._repeat = repeat

    def _setter_factory(self):
        stop = False
        while not stop:
            value = self._parameter()
            yield {self._parameter.full_name: value}
            stop = not self._repeat


class FunctionSweep(BaseSweepObject):
    """
    Use a function to set independent parameters instead of calling set methods directly
    """
    def __init__(self, set_function, point_function):
        super().__init__()
        self._set_function, self._parameter_table = set_function()
        self._point_function = point_function

    def _setter_factory(self):
        for set_value in self._point_function():
            yield self._set_function(set_value)


class FunctionWrapper(BaseSweepObject):
    """
    Use a function to measure dependent parameters instead of calling get methods directly
    """
    def __init__(self, measure_function, repeat=False):
        """
        Note: please use repeat=True carefully as this can easily lead to infinite loops.
        """
        super().__init__()
        self._measure_function, self._parameter_table = measure_function()
        self._repeat = repeat

    def _setter_factory(self):
        stop = False
        while not stop:
            yield self._measure_function()
            stop = not self._repeat
