from __future__ import annotations

from typing import ClassVar

from typing_extensions import TypedDict


class ParamSpecBaseDict(TypedDict):
    name: str
    paramtype: str
    label: str | None
    unit: str | None


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

    def __eq__(self, other: object) -> bool:
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
