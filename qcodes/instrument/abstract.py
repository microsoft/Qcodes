from functools import wraps
from typing import Any, Type, TypeVar
from qcodes import Parameter
from qcodes.instrument.base import InstrumentBase
from qcodes.instrument.parameter import _BaseParameter


class AbstractParameter(Parameter):
    """
    This is a trivial subclass of 'Parameter' to signal
    that this parameters *must* be overridden in
    subclasses of abstract instruments.
    """


class AbstractParameterException(Exception):
    """
    Errors with abstract parameter utilization should
    raise this exception class.
    """


def abstract_instrument(cls: Type[InstrumentBase]) -> Type[InstrumentBase]:
    """
    A class decorator to create an abstract instrument. Abstract
    instruments are allowed to have abstract parameters, but
    their subclasses do not. This works by replacing the
    'add_parameter' and '__init_subclass__' methods of the class
    dynamically.

    Args:
        cls: The class to be decorated

    Returns:
        The decorated class
    """

    def __init_subclass__(sub_cls: Type) -> None:

        __init__ = sub_cls.__init__

        def __init_new__(self, *args: Any, **kwargs: Any) -> None:
            """
            Subclasses of an abstract instrument should check
            after initialization whether there still are
            abstract parameters. If there are, we should raise
            an exception.
            """
            __init__(self, *args, **kwargs)
            # after the usual initialization...

            abstract_parameters = [
                parameter.name for parameter in self.parameters.values()
                if isinstance(parameter, AbstractParameter)
            ]

            if any(abstract_parameters):
                cls_name = sub_cls.__name__

                raise AbstractParameterException(
                    f"Class '{cls_name}' has un-implemented Abstract Parameter(s): " +
                    ", ".join([f"'{name}'" for name in abstract_parameters])
                )

        sub_cls.__init__ = __init_new__

    original_add_parameter = cls.add_parameter

    @wraps(original_add_parameter)
    def add_parameter(
            self, name: str, parameter_class: Type[_BaseParameter] = Parameter,
            **kwargs: Any
    ) -> None:

        existing_parameter = self.parameters.get(name, None)

        if isinstance(existing_parameter, AbstractParameter):
            # For abstract parameters, we define special behavior.
            existing_unit = getattr(existing_parameter, "unit", None)
            new_unit = kwargs.get("unit", None)

            if existing_unit and existing_unit != new_unit:
                raise AbstractParameterException(
                    f"The unit of the parameter '{name}' is '{new_unit}', "
                    f"which is inconsistent with the unit '{existing_unit}' "
                    f"specified in the baseclass {cls.__name__!r} "
                )

            # Remove the original abstract parameter to make room for the implementation
            # in the subclass.
            del self.parameters[name]

        original_add_parameter(
            self, name, parameter_class, **kwargs
        )

    cls.__init_subclass__ = classmethod(__init_subclass__)
    cls.add_parameter = add_parameter

    return cls
