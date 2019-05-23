from typing import Dict, Any

from qcodes.dataset.descriptions.dependencies import InterDependencies_


class RunDescriber:
    """
    The object that holds the description of each run in the database. This
    object serialises itself to a string and is found under the run_description
    column in the runs table

    Extension of this object is planned for the future, for now it holds the
    parameter interdependencies. Extensions should be objects that can
    serialize themselves added as attributes to the RunDescriber , such that
    the RunDescriber can iteratively serialize its attributes when serializing
    itself.
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

    def serialize(self) -> Dict[str, Any]:
        """
        Serialize this object into a dictionary
        """
        ser: Dict[str, Any] = {}
        ser['version'] = self._version
        ser['interdependencies'] = self.interdeps.serialize()

        return ser

    @classmethod
    def deserialize(cls, ser: Dict[str, Any]) -> 'RunDescriber':
        """
        Make a RunDescriber object based on a serialized version of it
        """

        rundesc = cls(InterDependencies_.deserialize(ser['interdependencies']))

        return rundesc

    def __eq__(self, other):
        if not isinstance(other, RunDescriber):
            return False
        if self.interdeps != other.interdeps:
            return False
        return True

    def __repr__(self) -> str:
        return f"RunDescriber({self.interdeps})"
