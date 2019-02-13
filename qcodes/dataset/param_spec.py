from typing import Union, Sequence, List, Dict, Any
from copy import deepcopy


class ParamSpec:

    allowed_types = ['array', 'numeric', 'text']

    def __init__(self, name: str,
                 paramtype: str,
                 label: str=None,
                 unit: str=None,
                 inferred_from: Sequence[Union['ParamSpec', str]]=None,
                 depends_on: Sequence[Union['ParamSpec', str]]=None,
                 **metadata) -> None:
        """
        Args:
            name: name of the parameter
            paramtype: type of the parameter, i.e. the SQL storage class
            label: label of the parameter
            inferred_from: the parameters that this parameter is inferred from
            depends_on: the parameters that this parameter depends on
        """
        if not isinstance(paramtype, str):
            raise ValueError('Paramtype must be a string.')
        if paramtype.lower() not in self.allowed_types:
            raise ValueError("Illegal paramtype. Must be on of "
                             f"{self.allowed_types}")
        if not name.isidentifier():
            raise ValueError(f'Invalid name: {name}. Only valid python '
                             'identifier names are allowed (no spaces or '
                             'punctuation marks, no prepended '
                             'numbers, etc.)')

        self.name = name
        self.type = paramtype.lower()
        self.label = '' if label is None else label
        self.unit = '' if unit is None else unit

        self._inferred_from: List[str] = []
        self._depends_on: List[str] = []

        inferred_from = [] if inferred_from is None else inferred_from
        depends_on = [] if depends_on is None else depends_on

        if isinstance(inferred_from, str):
            raise ValueError(f"ParamSpec {self.name} got "
                             f"string {inferred_from} as inferred_from. "
                             f"It needs a "
                             f"Sequence of ParamSpecs or strings")
        self._inferred_from.extend(
            p.name if isinstance(p, ParamSpec) else p
            for p in inferred_from)

        if isinstance(depends_on, str):
            raise ValueError(f"ParamSpec {self.name} got "
                             f"string {depends_on} as depends_on. It needs a "
                             f"Sequence of ParamSpecs or strings")
        self._depends_on.extend(
            p.name if isinstance(p, ParamSpec) else p
            for p in depends_on)

        if metadata:
            self.metadata = metadata

    @property
    def inferred_from_(self) -> List[str]:
        return deepcopy(self._inferred_from)

    @property
    def depends_on_(self) -> List[str]:
        return deepcopy(self._depends_on)

    @property
    def inferred_from(self) -> str:
        return ', '.join(self._inferred_from)

    @property
    def depends_on(self) -> str:
        return ', '.join(self._depends_on)

    def copy(self) -> 'ParamSpec':
        """
        Make a copy of self
        """
        return ParamSpec(self.name, self.type, self.label, self.unit,
                         deepcopy(self._inferred_from),
                         deepcopy(self._depends_on))

    def sql_repr(self):
        return f"{self.name} {self.type}"

    def __repr__(self):
        return (f"ParamSpec('{self.name}', '{self.type}', '{self.label}', "
                f"'{self.unit}', inferred_from={self._inferred_from}, "
                f"depends_on={self._depends_on})")

    def __eq__(self, other):
        if not isinstance(other, ParamSpec):
            return False
        attrs = ['name', 'type', 'label', 'unit', '_inferred_from',
                 '_depends_on']
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __hash__(self) -> int:
        """Allow ParamSpecs in data structures that use hashing (i.e. sets)"""
        attrs_with_strings = ['name', 'type', 'label', 'unit']
        attrs_with_lists = ['_inferred_from', '_depends_on']

        # First, get the hash of the tuple with all the relevant attributes
        all_attr_tuple_hash = hash(
            tuple(getattr(self, attr) for attr in attrs_with_strings)
            + tuple(tuple(getattr(self, attr)) for attr in attrs_with_lists)
        )
        hash_value = all_attr_tuple_hash

        # Then, XOR it with the individual hashes of all relevant attributes
        for attr in attrs_with_strings:
            hash_value = hash_value ^ hash(getattr(self, attr))
        for attr in attrs_with_lists:
            hash_value = hash_value ^ hash(tuple(getattr(self, attr)))

        return hash_value

    def serialize(self) -> Dict[str, Any]:
        """
        Write the ParamSpec as a dictionary
        """
        output: Dict[str, Any] = {}
        output['name'] = self.name
        output['paramtype'] = self.type
        output['label'] = self.label
        output['unit'] = self.unit
        output['inferred_from'] = self._inferred_from
        output['depends_on'] = self._depends_on

        return output

    @classmethod
    def deserialize(cls, ser: Dict[str, Any]) -> 'ParamSpec':
        """
        Create a ParamSpec instance of the current version
        from a serialized ParamSpec of some version

        The version changes must be implemented as a series of transformations
        of the serialized dict.
        """

        return ParamSpec(name=ser['name'],
                         paramtype=ser['paramtype'],
                         label=ser['label'],
                         unit=ser['unit'],
                         inferred_from=ser['inferred_from'],
                         depends_on=ser['depends_on'])
