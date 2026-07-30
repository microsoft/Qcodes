from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal, final

from qcodes.utils import deep_update

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SnapshotUpdate = Literal["All", "Only_invalid", "Never"]
"""
Allowed string values for the ``update`` argument of ``snapshot`` and
``snapshot_base``:

* ``"All"``: force an update of every value (equivalent to legacy ``True``).
* ``"Only_invalid"``: only update values whose cache is invalid, using the
  latest cached value otherwise (equivalent to legacy ``None``).
* ``"Never"``: never update, always use the latest values in memory
  (equivalent to legacy ``False``).
"""

_SNAPSHOT_UPDATE_ALIASES: "dict[str, bool | None]" = {
    "All": True,
    "Only_invalid": None,
    "Never": False,
}


def _normalize_snapshot_update(
    update: "bool | SnapshotUpdate | None",
) -> "bool | None":
    """
    Normalize the ``update`` argument of ``snapshot``/``snapshot_base`` into
    its legacy ``bool | None`` representation, where ``True`` means update all
    values, ``None`` means update only values with an invalid cache, and
    ``False`` means never update.

    The string values ``"All"``, ``"Only_invalid"`` and ``"Never"`` are mapped
    to ``True``, ``None`` and ``False`` respectively. ``bool`` and ``None``
    values are returned unchanged for backwards compatibility.
    """
    if isinstance(update, str):
        try:
            return _SNAPSHOT_UPDATE_ALIASES[update]
        except KeyError:
            raise ValueError(
                f"Invalid value for snapshot ``update``: {update!r}. Expected "
                f"one of {list(_SNAPSHOT_UPDATE_ALIASES)}, or a bool, or None."
            ) from None
    return update


# NB: At the moment, the Snapshot type is a bit weak, as the Any
#     for the value type doesn't tell us anything about the schema
#     followed by snapshots.
#     This is needed, however, since snapshots are Dict instances with
#     homogeneous keys and heterogeneous values, something that
#     recent Python versions largely replace with features like
#     typing.NamedTuple and @dataclass.
#     As those become more widely available, the weakness of this
#     type constraint will become less of an issue.
Snapshot = dict[str, Any]


class Metadatable:
    def __init__(self, metadata: "Mapping[str, Any] | None" = None):
        self.metadata: dict[str, Any] = {}
        self.load_metadata(metadata or {})

    def load_metadata(self, metadata: "Mapping[str, Any]") -> None:
        """
        Load metadata into this classes metadata dictionary.

        Args:
            metadata: Metadata to load.

        """
        deep_update(self.metadata, metadata)

    @final
    def snapshot(self, update: "bool | SnapshotUpdate | None" = False) -> Snapshot:
        """
        Decorate a snapshot dictionary with metadata.
        DO NOT override this method if you want metadata in the snapshot
        instead, override :meth:`snapshot_base`.

        Args:
            update: Passed to snapshot_base. Accepts the string values
                ``"All"``, ``"Only_invalid"`` and ``"Never"`` as well as the
                legacy values ``True`` (equivalent to ``"All"``), ``None``
                (equivalent to ``"Only_invalid"``) and ``False`` (equivalent
                to ``"Never"``).

        Returns:
            Base snapshot.

        """

        snap = self.snapshot_base(update=_normalize_snapshot_update(update))

        if len(self.metadata):
            snap["metadata"] = self.metadata

        return snap

    def snapshot_base(
        self,
        update: "bool | SnapshotUpdate | None" = False,
        params_to_skip_update: "Sequence[str] | None" = None,
    ) -> Snapshot:
        """
        Override this with the primary information for a subclass.
        """
        return {}


class MetadatableWithName(Metadatable):
    """Add short_name and full_name properties to Metadatable.
    This is used as a base class for all components in QCoDeS that
    are members of a station to ensure that they have a name and
    consistent interface."""

    @property
    @abstractmethod
    def short_name(self) -> str:
        """
        Name excluding name of any parent that this object is bound to.
        """

    @property
    @abstractmethod
    def full_name(self) -> str:
        """
        Name including name of any parent that this object is bound to separated by '_'.
        """
