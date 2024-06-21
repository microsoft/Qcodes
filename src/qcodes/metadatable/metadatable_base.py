from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional, final

from qcodes.utils import deep_update

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    def __init__(self, metadata: Optional["Mapping[str, Any]"] = None):
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
    def snapshot(self, update: bool | None = False) -> Snapshot:
        """
        Decorate a snapshot dictionary with metadata.
        DO NOT override this method if you want metadata in the snapshot
        instead, override :meth:`snapshot_base`.

        Args:
            update: Passed to snapshot_base.

        Returns:
            Base snapshot.
        """

        snap = self.snapshot_base(update=update)

        if len(self.metadata):
            snap["metadata"] = self.metadata

        return snap

    def snapshot_base(
        self,
        update: bool | None = False,
        params_to_skip_update: Optional["Sequence[str]"] = None,
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
