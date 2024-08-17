from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence


class ParamSpecBaseDict(TypedDict):
    name: str
    paramtype: str
    label: str | None
    unit: str | None


class ParamSpecDict(ParamSpecBaseDict):
    inferred_from: list[str]
    depends_on: list[str]


class ParamSpecBase:
    allowed_types: ClassVar[list[str]] = ["array", "numeric", "text", "complex"]

    def __init__(
        self,
        name: str,
        paramtype: str,
        label: str | None = None,
        unit: str | None = None,
    ):
        """
        Args:
            name: name of the parameter
            paramtype: type of the parameter, i.e. the SQL storage class
            label: label of the parameter
            unit: The unit of the parameter
        """

        if not isinstance(paramtype, str):
            raise ValueError("Paramtype must be a string.")
        if paramtype.lower() not in self.allowed_types:
            raise ValueError(f"Illegal paramtype. Must be on of {self.allowed_types}")
        if not name.isidentifier():
            raise ValueError(
                f"Invalid name: {name}. Only valid python "
                "identifier names are allowed (no spaces or "
                "punctuation marks, no prepended "
                "numbers, etc.)"
            )

        self.name = name
        self.type = paramtype.lower()
        self.label = label or ""
        self.unit = unit or ""

        self._hash: int = self._compute_hash()

    def _compute_hash(self) -> int:
        """
        This method should only be called by __init__
        """
        attrs = ["name", "type", "label", "unit"]
        # First, get the hash of the tuple with all the relevant attributes
        all_attr_tuple_hash = hash(tuple(getattr(self, attr) for attr in attrs))
        hash_value = all_attr_tuple_hash

        # Then, XOR it with the individual hashes of all relevant attributes
        for attr in attrs:
            hash_value = hash_value ^ hash(getattr(self, attr))

        return hash_value

    def sql_repr(self) -> str:
        return f"{self.name} {self.type}"

    def __repr__(self) -> str:
        return (
            f"ParamSpecBase('{self.name}', '{self.type}', '{self.label}', "
            f"'{self.unit}')"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ParamSpecBase):
            return False
        attrs = ["name", "type", "label", "unit"]
        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __hash__(self) -> int:
        """
        Allow ParamSpecBases in data structures that use hashing (e.g. sets)
        """
        return self._hash

    def _to_dict(self) -> ParamSpecBaseDict:
        """
        Write the ParamSpec as a dictionary
        """
        output = ParamSpecBaseDict(
            name=self.name, paramtype=self.type, label=self.label, unit=self.unit
        )
        return output

    @classmethod
    def _from_dict(cls, ser: ParamSpecBaseDict) -> ParamSpecBase:
        """
        Create a ParamSpec instance of the current version
        from a dictionary representation of ParamSpec of some version

        The version changes must be implemented as a series of transformations
        of the representation dict.
        """

        return ParamSpecBase(
            name=ser["name"],
            paramtype=ser["paramtype"],
            label=ser["label"],
            unit=ser["unit"],
        )


class ParamSpec(ParamSpecBase):
    def __init__(
        self,
        name: str,
        paramtype: str,
        label: str | None = None,
        unit: str | None = None,
        inferred_from: Sequence[ParamSpec | str] | None = None,
        depends_on: Sequence[ParamSpec | str] | None = None,
        **metadata: Any,
    ) -> None:
        """
        Args:
            name: name of the parameter
            paramtype: type of the parameter, i.e. the SQL storage class
            label: label of the parameter
            unit: unit of the parameter
            inferred_from: the parameters that this parameter is inferred from
            depends_on: the parameters that this parameter depends on
            **metadata: additional metadata to be stored with the parameter
        """

        super().__init__(name, paramtype, label, unit)

        self._inferred_from: list[str] = []
        self._depends_on: list[str] = []

        inferred_from = [] if inferred_from is None else inferred_from
        depends_on = [] if depends_on is None else depends_on

        if isinstance(inferred_from, str):
            raise ValueError(
                f"ParamSpec {self.name} got "
                f"string {inferred_from} as inferred_from. "
                f"It needs a "
                f"Sequence of ParamSpecs or strings"
            )
        self._inferred_from.extend(
            p.name if isinstance(p, ParamSpec) else p for p in inferred_from
        )

        if isinstance(depends_on, str):
            raise ValueError(
                f"ParamSpec {self.name} got "
                f"string {depends_on} as depends_on. It needs a "
                f"Sequence of ParamSpecs or strings"
            )
        self._depends_on.extend(
            p.name if isinstance(p, ParamSpec) else p for p in depends_on
        )

        if metadata:
            self.metadata = metadata

    @property
    def inferred_from_(self) -> list[str]:
        return deepcopy(self._inferred_from)

    @property
    def depends_on_(self) -> list[str]:
        return deepcopy(self._depends_on)

    @property
    def inferred_from(self) -> str:
        return ", ".join(self._inferred_from)

    @property
    def depends_on(self) -> str:
        return ", ".join(self._depends_on)

    def copy(self) -> ParamSpec:
        """
        Make a copy of self
        """
        return ParamSpec(
            self.name,
            self.type,
            self.label,
            self.unit,
            deepcopy(self._inferred_from),
            deepcopy(self._depends_on),
        )

    def __repr__(self) -> str:
        return (
            f"ParamSpec('{self.name}', '{self.type}', '{self.label}', "
            f"'{self.unit}', inferred_from={self._inferred_from}, "
            f"depends_on={self._depends_on})"
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ParamSpec):
            return False
        string_attrs = ["name", "type", "label", "unit"]
        list_attrs = ["_inferred_from", "_depends_on"]
        for string_attr in string_attrs:
            if getattr(self, string_attr) != getattr(other, string_attr):
                return False
        for list_attr in list_attrs:
            ours = getattr(self, list_attr)
            theirs = getattr(other, list_attr)
            if ours != theirs:
                return False
        return True

    def __hash__(self) -> int:
        """Allow ParamSpecs in data structures that use hashing (i.e. sets)"""
        attrs_with_strings = ["name", "type", "label", "unit"]
        attrs_with_lists = ["_inferred_from", "_depends_on"]

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

    def _to_dict(self) -> ParamSpecDict:
        """
        Write the ParamSpec as a dictionary
        """
        basedict = super()._to_dict()
        output = ParamSpecDict(
            name=basedict["name"],
            paramtype=basedict["paramtype"],
            label=basedict["label"],
            unit=basedict["unit"],
            inferred_from=self._inferred_from,
            depends_on=self._depends_on,
        )
        return output

    def base_version(self) -> ParamSpecBase:
        """
        Return a ParamSpecBase object with the same name, paramtype, label
        and unit as this ParamSpec
        """
        return ParamSpecBase(
            name=self.name, paramtype=self.type, label=self.label, unit=self.unit
        )

    @classmethod
    def _from_dict(cls, ser: ParamSpecDict) -> ParamSpec:  # type: ignore[override]
        """
        Create a ParamSpec instance of the current version
        from a dictionary representation of ParamSpec of some version

        The version changes must be implemented as a series of transformations
        of the representation dict.
        """

        return ParamSpec(
            name=ser["name"],
            paramtype=ser["paramtype"],
            label=ser["label"],
            unit=ser["unit"],
            inferred_from=ser["inferred_from"],
            depends_on=ser["depends_on"],
        )
