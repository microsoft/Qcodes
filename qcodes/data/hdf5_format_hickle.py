import logging
from .hdf5_format import HDF5Format
import hickle
from qcodes.utils.helpers import deep_update

#%%

log = logging.getLogger(__name__)


class HDF5FormatHickle(HDF5Format):

    _metadata_file = 'snapshot.hickle'
    _format_tag = 'hdf5-hickle'

    def write_metadata(self, data_set, io_manager=None, location=None, read_first=False):
        """
        Write all metadata in this DataSet to storage.

        Args:
            data_set (DataSet): the data we're storing

            io_manager (io_manager): the base location to write to

            location (str): the file location within io_manager

            read_first (bool, optional): read previously saved metadata before
                writing? The current metadata will still be the used if
                there are changes, but if the saved metadata has information
                not present in the current metadata, it will be retained.
                Default True.
        """

        # this statement is here to make the linter happy
        if io_manager is None or location is None:
            raise Exception('please set io_manager and location arguments ')

        if read_first:
            # In case the saved file has more metadata than we have here,
            # read it in first. But any changes to the in-memory copy should
            # override the saved file data.
            memory_metadata = data_set.metadata
            data_set.metadata = {}
            self.read_metadata(data_set)
            deep_update(data_set.metadata, memory_metadata)

        log.info('writing metadata to file %s' % self._metadata_file)
        fn = io_manager.join(location, self._metadata_file)
        with io_manager.open(fn, 'w', encoding='utf8') as snap_file:
            hickle.dump(data_set.metadata, snap_file)

    def read_metadata(self, data_set):
        """ Reads in the metadata

        Args:
            data_set (DataSet): Dataset object to read the metadata into
        """
        io_manager = data_set.io
        location = data_set.location
        fn = io_manager.join(location, self._metadata_file)
        if io_manager.list(fn):
            log.info('reading metadata from file %s' % self._metadata_file)
            with io_manager.open(fn, 'r') as snap_file:
                metadata = hickle.load(snap_file)
            data_set.metadata.update(metadata)
