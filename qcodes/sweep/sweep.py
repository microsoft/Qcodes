import itertools
import time  # For defining a time trace sweep object type


def parameter_log_format(parameter, value, setting=True):
    """
    The standard method of logging the setting or retrieving a parameter value

    Parameters
    ----------
    parameter: qcodes.StandardParameter
    value: float
        Either the set or the get value of the parameter. As the sole responsibility of this function is to log
        the setting or retrieving of parameter values, the setting and getting should have been done on the calling
        side.
    setting: bool, optional (True)
        Are we setting a parameter or retrieving it? If we are setting it, the the parameters is coupled to an
        independent measurement parameter. If we are retrieving it (measuring it), which means it is an dependent
        parameter
    """
    if parameter._instrument is not None:  # TODO: Make a QCoDeS pull request to access this through a public
        # interface
        label = "{}_{}".format(parameter._instrument.name, parameter.label)
    else:
        label = parameter.label

    log_dict = {label: {"unit": parameter.unit, "value": value, "independent_parameter": setting}}

    return log_dict


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
        self._iterator_function = iterator_function
        super().__init__()

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


class Zip(BaseSweepObject):
    """
    Zip multiple sweep objects. Unlike a nested sweep, we will produce a 1D sweep
    """
    def __init__(self, sweep_objects):
        super().__init__()
        self._sweep_objects = sweep_objects

    @staticmethod
    def _combine_dictionaries(dictionaries):
        combined = {}
        for d in dictionaries:
            combined.update(d)
        return combined

    def _setter_factory(self):
        for results in itertools.zip_longest(*self._sweep_objects, fillvalue="error"):
            if "error" in results:
                raise RuntimeError("When zipping sweep objects, the number of iterations of each should be equal")
            yield Zip._combine_dictionaries(results)


class Chain(BaseSweepObject):
    """
    Chain a list of sweep object to run one after the other
    """
    def __init__(self, sweep_objects):
        super().__init__()
        self._sweep_objects = sweep_objects

    def _setter_factory(self):
        for so in self._sweep_objects:
            for result in so:
                yield result


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
        self._parameter = parameter
        self._point_function = point_function
        super().__init__()

    def _setter_factory(self):
        for set_value in self._point_function():
            self._parameter.set(set_value)
            yield parameter_log_format(self._parameter, set_value)


class ParameterWrapper(BaseSweepObject):
    """
    A wrapper class which iterates once ans returns the value of a QCoDeS parameter

    Parameters
    ----------
    parameter: qcodes.StandardParameter
    """
    def __init__(self, parameter):
        self._parameter = parameter
        super().__init__()

    def _setter_factory(self):
        value = self._parameter()
        yield parameter_log_format(self._parameter, value, setting=False)


class FunctionSweep(BaseSweepObject):
    """
    Sweep independent parameters by looping over set point values and calling a set function

    Parameters
    ----------
    set_function: callable
        A callable of three parameters: station, namespace, set_value. This returns a dictionary containing arbitrary
        information about the value set and any measurements performed (or it can contain any information that needs to
        be added in the final dataset
    point_function: callable
        A callable of two parameters: station, namespace, returning an iterator. Unrolling this iterator returns to
        us set values of the parameter
    """
    def __init__(self, set_function, point_function):
        self._set_function = set_function
        self._point_function = point_function
        super().__init__()

    def _setter_factory(self):
        for set_value in self._point_function():
            yield self._set_function(set_value)


class FunctionWrapper(BaseSweepObject):
    """
    A wrapper class which iterates once and returns the result of a function

    Parameters
    ----------
    func: callable
        callable of station, namespace
    """
    def __init__(self, func):
        self._func = func
        super().__init__()

    def _setter_factory(self):
        yield self._func()


class TimeTrace(BaseSweepObject):
    """
    Make a "time sweep", that is, take a time trace

    Parameter
    ---------
    measure: callable
        callable of arguments  station, namespace, returning a dictionary with measurement results.
    delay: float
        The time in seconds between calling the measure function
    total_time: float
        The total duration of the time trace
    """
    def __init__(self, measure, delay, total_time):
        self._measure = measure
        self._delay = delay
        self._total_time = total_time
        super().__init__()

    def _setter_factory(self):
        t0 = time.time()
        t = t0
        while t - t0 < self._total_time:
            msg = self._measure()
            time_msg = {"time": {"unit": "s", "value": t, "independent_parameter": True}}
            msg.update(time_msg)
            yield msg
            time.sleep(self._delay)
            t = time.time()
