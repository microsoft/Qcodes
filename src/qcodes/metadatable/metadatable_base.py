from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal, final, overload

from typing_extensions import deprecated

from qcodes.utils import deep_update

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SnapshotUpdate = Literal["All", "Only_invalid", "Never"]
"""
Canonical string values for the ``update`` argument of ``snapshot`` and
``snapshot_base``:

* ``"All"``: force an update of every value (equivalent to legacy ``True``).
* ``"Only_invalid"``: only update values whose cache is invalid, using the
  latest cached value otherwise (equivalent to legacy ``None``).
* ``"Never"``: never update, always use the latest values in memory
  (equivalent to legacy ``False``).

Internally, the ``update`` argument is always normalized to one of these
values via :func:`normalize_snapshot_update`. The legacy ``bool``/``None``
values are still accepted at the public interface for backwards compatibility.
"""


def normalize_snapshot_update(
    update: "bool | SnapshotUpdate | None",
) -> SnapshotUpdate:
    """
    Normalize the ``update`` argument of ``snapshot``/``snapshot_base`` into
    one of the canonical :data:`SnapshotUpdate` string values.

    The legacy values ``True``, ``None`` and ``False`` are mapped to
    ``"All"``, ``"Only_invalid"`` and ``"Never"`` respectively, and the
    canonical string values are returned unchanged. This is the single place
    where the ``update`` argument is interpreted; all internal code should
    work with the returned :data:`SnapshotUpdate` value rather than with the
    legacy ``bool``/``None`` representation.

    Args:
        update: The ``update`` argument as passed to ``snapshot``/
            ``snapshot_base``.

    Returns:
        The equivalent canonical :data:`SnapshotUpdate` value.

    Raises:
        ValueError: If ``update`` is a string that is not a valid
            :data:`SnapshotUpdate` value.

    """
    if update is True:
        return "All"
    if update is False:
        return "Never"
    if update is None:
        return "Only_invalid"
    if update in ("All", "Only_invalid", "Never"):
        return update
    raise ValueError(
        f"Invalid value for snapshot ``update``: {update!r}. Expected one of "
        f"'All', 'Only_invalid', 'Never', or a bool, or None."
    )


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

    @overload
    def snapshot(self, update: "SnapshotUpdate" = ...) -> Snapshot: ...

    @overload
    @deprecated(
        "Passing a bool or None as the snapshot ``update`` argument is "
        "deprecated; use one of the string values 'All', 'Only_invalid' or "
        "'Never' instead."
    )
    def snapshot(self, update: "bool | None" = ...) -> Snapshot: ...

    @final
    def snapshot(
        self, update: "bool | SnapshotUpdate | None" = "Only_invalid"
    ) -> Snapshot:
        """
        Decorate a snapshot dictionary with metadata.
        DO NOT override this method if you want metadata in the snapshot
        instead, override :meth:`snapshot_base`.

        Args:
            update: What to do about the values stored in the snapshot; passed
                to :meth:`snapshot_base` after being normalized to a
                :data:`SnapshotUpdate` value.

                * ``"All"``: force an update of every value.
                * ``"Only_invalid"`` (the default): only update values whose
                  cache is invalid, using the latest cached value otherwise.
                * ``"Never"``: never update, always use the latest values in
                  memory.

                The legacy ``True`` / ``None`` / ``False`` values are deprecated
                aliases for ``"All"`` / ``"Only_invalid"`` / ``"Never"`` and are
                still accepted for backwards compatibility (no warning is
                raised).

        Returns:
            Base snapshot.

        """

        snap = self.snapshot_base(update=normalize_snapshot_update(update))

        if len(self.metadata):
            snap["metadata"] = self.metadata

        return snap

    def snapshot_base(
        self,
        update: "bool | SnapshotUpdate | None" = "Only_invalid",
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
