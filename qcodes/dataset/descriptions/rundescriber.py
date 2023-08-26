from __future__ import annotations

from typing import Any, cast

from qcodes.dataset.descriptions.dependencies import InterDependencies_

from .versioning.converters import new_to_old, old_to_new
from .versioning.rundescribertypes import (
    RunDescriberDicts,
    RunDescriberV0Dict,
    RunDescriberV1Dict,
    RunDescriberV2Dict,
    RunDescriberV3Dict,
    Shapes,
)
from .versioning.v0 import InterDependencies


class RunDescriber:
    """
    The object that holds the description of each run in the database. This
    object serialises itself to a string and is found under the run_description
    column in the runs table.

    Extension of this object is planned for the future, for now it holds the
    parameter interdependencies. Extensions should be objects that can
    convert themselves to dictionary and added as attributes to the
    RunDescriber, such that the RunDescriber can iteratively convert its
    attributes when converting itself to dictionary.
    """

    def __init__(
        self, interdeps: InterDependencies_, shapes: Shapes | None = None
    ) -> None:

        if not isinstance(interdeps, InterDependencies_):
            raise ValueError('The interdeps arg must be of type: '
                             'InterDependencies_. '
                             f'Got {type(interdeps)}.')

        self._interdeps = interdeps

        self._shapes = shapes
        self._version = 3

    @property
    def version(self) -> int:
        return self._version

    @property
    def shapes(self) -> Shapes | None:
        return self._shapes

    @property
    def interdeps(self) -> InterDependencies_:
        return self._interdeps

    def _to_dict(self) -> RunDescriberV3Dict:
        """
        Convert this object into a dictionary. This method is intended to
        be used only by the serialization routines.
        """
        ser: RunDescriberV3Dict = {
            'version': self._version,
            'interdependencies': new_to_old(self.interdeps)._to_dict(),
            'interdependencies_': self.interdeps._to_dict(),
            'shapes': self.shapes

        }

        return ser

    @classmethod
    def _from_dict(cls, ser: RunDescriberDicts) -> RunDescriber:
        """
        Make a RunDescriber object from a dictionary. This method is
        intended to be used only by the deserialization routines.
        """
        if ser['version'] == 0:
            ser = cast(RunDescriberV0Dict, ser)
            rundesc = cls(
                old_to_new(
                    InterDependencies._from_dict(ser['interdependencies'])
                )
            )
        elif ser['version'] == 1:
            ser = cast(RunDescriberV1Dict, ser)
            rundesc = cls(
                InterDependencies_._from_dict(ser['interdependencies'])
            )
        elif ser['version'] == 2:
            ser = cast(RunDescriberV2Dict, ser)
            rundesc = cls(
                InterDependencies_._from_dict(ser['interdependencies_'])
            )
        elif ser['version'] >= 3:
            ser = cast(RunDescriberV3Dict, ser)
            rundesc = cls(
                InterDependencies_._from_dict(ser['interdependencies_']),
                shapes=ser['shapes']
            )
        else:
            raise RuntimeError(f"Unknown version: "
                               f"Cannot deserialize from {ser['version']}")

        return rundesc

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RunDescriber):
            return False
        if self.interdeps != other.interdeps:
            return False
        if self.shapes != other.shapes:
            return False
        return True

    def __repr__(self) -> str:
        return f"RunDescriber({self.interdeps}, Shapes: {self._shapes})"
