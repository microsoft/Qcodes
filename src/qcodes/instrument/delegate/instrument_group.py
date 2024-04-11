from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ..instrument_base import InstrumentBase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from qcodes.station import Station


class InstrumentGroup(InstrumentBase):
    """
    InstrumentGroup is an instrument driver to represent a series of instruments
    that are grouped together. This instrument is mainly used as a wrapper for
    sub instruments/submodules and particularly built for use with grouping
    multiple :class:`DelegateInstrument` s.

    Args:
        name: Name referring to this group of items
        station: Measurement station with real instruments
        submodules_type: Class to use for creating the instruments.
        submodules: A mapping between an instrument name and the values passed
            to the constructor of the class specified by `submodules_type`.
        initial_values: A mapping between the names of parameters and initial
            values to set on those parameters when loading this instrument.
        set_initial_values_on_load: Set default values on load. Defaults to
            False.
    """
    def __init__(
        self,
        name: str,
        station: Station,
        submodules_type: str,
        submodules: Mapping[str, Mapping[str, list[str]]],
        initial_values: Mapping[str, Mapping[str, Any]],
        set_initial_values_on_load: bool = False,
        **kwargs: Any
    ):
        super().__init__(name=name, **kwargs)

        module_name = '.'.join(submodules_type.split('.')[:-1])
        instr_class_name = submodules_type.split('.')[-1]
        module = importlib.import_module(module_name)
        instr_class = getattr(module, instr_class_name)

        for submodule_name, inputs in submodules.items():
            if not any(x in inputs.keys() for x in ["parameters", "channels"]):
                raise KeyError(
                    f"Missing keyworded input arguments for {submodule_name}"
                )
            submodule = instr_class(
                name=submodule_name,
                station=station,
                **inputs,
                initial_values=initial_values.get(submodule_name),
                set_initial_values_on_load=set_initial_values_on_load
            )

            self.add_submodule(
                submodule_name,
                submodule
            )

    def __repr__(self) -> str:
        submodules = ", ".join(self.submodules.keys())
        return f"InstrumentGroup(name={self.name}, submodules={submodules})"
