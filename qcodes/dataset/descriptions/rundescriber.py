from typing import Dict, Any

from qcodes.dataset.descriptions.dependencies import InterDependencies_


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

    def __init__(self, interdeps: InterDependencies_) -> None:

        if not isinstance(interdeps, InterDependencies_):
            raise ValueError('The interdeps arg must be of type: '
                             'InterDependencies_. '
                             f'Got {type(interdeps)}.')

        self.interdeps = interdeps

        self._version = 1

    @property
    def version(self) -> int:
        return self._version

    def _to_dict(self) -> Dict[str, Any]:
        """
        Convert this object into a dictionary. This method is intended to
        be used only by the serialization routines.
        """
        ser: Dict[str, Any] = {}
        ser['version'] = self._version
        ser['interdependencies'] = self.interdeps._to_dict()

        return ser

    @classmethod
    def _from_dict(cls, ser: Dict[str, Any]) -> 'RunDescriber':
        """
        Make a RunDescriber object from a dictionary. This method is
        intended to be used only by the deserialization routines.
        """

        rundesc = cls(
            InterDependencies_._from_dict(ser['interdependencies']))

        return rundesc

    def __eq__(self, other):
        if not isinstance(other, RunDescriber):
            return False
        if self.interdeps != other.interdeps:
            return False
        return True

    def __repr__(self) -> str:
        return f"RunDescriber({self.interdeps})"
