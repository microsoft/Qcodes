"""
This module contains different workarounds that are temporarily used in QCoDeS
"here and there" for working around bugs and deficiencies of packages that
QCoDeS depends on.
"""
import contextlib
from typing import ContextManager

import visa
import pyvisa
from qcodes.utils.helpers import attribute_set_to


def _visa_resource_read_termination_set_to_none(
        visa_resource: pyvisa.Resource
) -> ContextManager:
    """
    This context manager works around the problem with 'query_binary_values'
    method in pyvisa 1.9.0 that results in visa timeout exception in
    drivers which use the method.

    Usage:
      ...
      # visa_resource is the variable that refers to the instrument's pyvisa
      # resource object
      with visa_query_binary_values_fix_for(visa_resource):
          visa_resource.query_binary_values(...)
      ...
    """
    return attribute_set_to(visa_resource, '_read_termination', None)


def _null_context_manager_with_arguments(*args, **kwargs) -> ContextManager:
    """
    A null context manager that does nothing, and accepts arguments.

    Python 3.6 does not have a null context manager (python 3.7 has), hence
    the 'suppress' context manager with no arguments is used to represent it
    instead. Additionally, the context manager accepts arguments and ignores
    them.
    """
    return contextlib.suppress()


if visa.__version__ == '1.9.0':
    visa_query_binary_values_fix_for = \
        _visa_resource_read_termination_set_to_none
else:
    visa_query_binary_values_fix_for = \
        _null_context_manager_with_arguments
