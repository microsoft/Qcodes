class Metadatable:
    def __init__(self, *args, **kwargs):
        self.metadata = {}
        self.load_metadata(kwargs)

    def load_metadata(self, attributes):
        self.metadata.update(attributes.get('metadata', {}))

    def snapshot(self, **kwargs):
        '''
        decorate a snapshot dictionary with metadata
        DO NOT override this method if you want metadata in the snapshot
        instead, override snapshot_base
        '''

        snap = self.snapshot_base(**kwargs)

        if len(self.metadata):
            snap['metadata'] = self.metadata

        return snap

    def snapshot_base(self):
        '''
        override this with the primary information for a subclass
        '''
        return {}
