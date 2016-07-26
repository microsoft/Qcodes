from .helpers import deep_update


class Metadatable:
    def __init__(self, metadata=None):
        self.metadata = {}
        self.load_metadata(metadata or {})

    def load_metadata(self, metadata):
        """
        Load metadata

        Args:
            metadata (dict): metadata to load
        """
        deep_update(self.metadata, metadata)

    def snapshot(self, update=False):
        """
        Decorate a snapshot dictionary with metadata.
        DO NOT override this method if you want metadata in the snapshot
        instead, override snapshot_base.

        Args:
            update (bool): Passed to snapshot_base

        Returns:
            dict: base snapshot
        """

        snap = self.snapshot_base(update=update)

        if len(self.metadata):
            snap['metadata'] = self.metadata

        return snap

    def snapshot_base(self, update=False):
        """
        override this with the primary information for a subclass
        """
        return {}
