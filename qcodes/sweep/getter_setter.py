"""
Instead of getting and setting QCoDeS parameters, sweep objects can also call
getter and setter functions. This module defines decorators to create
appropriate functions
"""

import numpy as np
from typing import Callable, Optional, List, Tuple

from qcodes.sweep.base_sweep import ParametersTable


def getter(param_list: List[Tuple[str, str]],
           inferred_parameters: List[Tuple[str, str]]=None)->Callable:
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
    inferred_symbols: Optional[List[Tuple[str, ...]]]
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


def setter(param_list: List[Tuple[str, str]],
           inferred_parameters: List[Tuple[str, str]]=None)->Callable:
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
    inferred_symbols: Optional[List[Tuple[str, ...]]]
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
