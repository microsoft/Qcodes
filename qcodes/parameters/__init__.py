"""
The Parameter module implements Parameter interface
that are the basis of measurements and control within QCoDeS.

Anything that you want to either measure or control within QCoDeS should
satisfy the Parameter interface. Most of the time that is easiest to do
by either using or subclassing one of the classes defined here, but you can
also use any class with the right attributes.

All parameter classes are subclassed from :class:`.ParameterBase` (except
CombinedParameter). The ParameterBase provides functionality that is common
to all parameter types, such as ramping and scaling of values, adding delays
(see documentation for details).

This module defines the following basic classes of parameters as well as some
more specialized ones:

- :class:`.Parameter` is the base class for scalar-valued parameters.
    Two primary ways in which it can be used:

    1. As an :class:`.Instrument` parameter that sends/receives commands.
       Provides a standardized interface to construct strings to pass to the
       :meth:`.Instrument.write` and :meth:`.Instrument.ask` methods
    2. As a variable that stores and returns a value. For instance, for storing
       of values you want to keep track of but cannot set or get electronically.

- :class:`.ParameterWithSetpoints` is intended for array-values parameters.
    This Parameter class is intended for anything where a call to the instrument
    returns an array of values. `This notebook
    <../../examples/Parameters/Simple-Example-of-ParameterWithSetpoints
    .ipynb>`_ gives more detailed examples of how this parameter
    can be used `and this notebook
    <../../examples/writing_drivers/A-ParameterWithSetpoints
    -Example-with-Dual-Setpoints.ipynb>`_ explains writing driver
    using :class:`.ParameterWithSetpoints`.

    :class:`.ParameterWithSetpoints` is supported in a
    :class:`qcodes.dataset.Measurement` but is not supported by
    the legacy :class:`qcodes.loops.Loop` and :class:`qcodes.measure.Measure`
    measurement types.

- :class:`.DelegateParameter` is intended for proxy-ing other parameters.
    It forwards its ``get`` and ``set`` to the underlying source parameter,
    while allowing to specify label/unit/etc that is different from the
    source parameter.

- :class:`.ArrayParameter` is an older base class for array-valued parameters.
    For any new driver we strongly recommend using
    :class:`.ParameterWithSetpoints` which is both more flexible and
    significantly easier to use. This Parameter is intended for anything for
    which each ``get`` call returns an array of values that all have the same
    type and meaning. Currently not settable, only gettable. Can be used in a
    :class:`qcodes.dataset.Measurement`
    as well as in the legacy :class:`qcodes.loops.Loop`
    and :class:`qcodes.measure.Measure` measurements - in which case
    these arrays are nested inside the loop's setpoint array. To use, provide a
    ``get`` method that returns an array or regularly-shaped sequence, and
    describe that array in ``super().__init__``.

- :class:`.MultiParameter` is the base class for multi-valued parameters.
    Currently not settable, only gettable, but can return an arbitrary
    collection of scalar and array values and can be used in
    :class:`qcodes.dataset.Measurement` as well as the
    legacy :class:`qcodes.loops.Loop` and :class:`qcodes.measure.Measure`
    measurements. To use, provide a ``get`` method
    that returns a sequence of values, and describe those values in
    ``super().__init__``.

"""

from .array_parameter import ArrayParameter
from .combined_parameter import CombinedParameter, combine
from .delegate_parameter import DelegateParameter
from .function import Function
from .group_parameter import Group, GroupParameter
from .grouped_parameter import DelegateGroup, DelegateGroupParameter, GroupedParameter
from .multi_channel_instrument_parameter import MultiChannelInstrumentParameter
from .multi_parameter import MultiParameter
from .parameter import ManualParameter, Parameter
from .parameter_base import (
    ParamDataType,
    ParameterBase,
    ParamRawDataType,
    invert_val_mapping,
)
from .parameter_with_setpoints import ParameterWithSetpoints, expand_setpoints_helper
from .scaled_paramter import ScaledParameter
from .specialized_parameters import ElapsedTimeParameter, InstrumentRefParameter
from .sweep_values import SweepFixedValues, SweepValues
from .val_mapping import create_on_off_val_mapping

__all__ = [
    "ArrayParameter",
    "CombinedParameter",
    "DelegateGroup",
    "DelegateGroupParameter",
    "DelegateParameter",
    "ElapsedTimeParameter",
    "Function",
    "Group",
    "GroupParameter",
    "GroupedParameter",
    "GroupedParameter",
    "InstrumentRefParameter",
    "ManualParameter",
    "MultiChannelInstrumentParameter",
    "MultiParameter",
    "ParamDataType",
    "ParamRawDataType",
    "Parameter",
    "ParameterBase",
    "ParameterWithSetpoints",
    "ScaledParameter",
    "SweepFixedValues",
    "SweepValues",
    "create_on_off_val_mapping",
    "combine",
    "expand_setpoints_helper",
    "invert_val_mapping",
]
