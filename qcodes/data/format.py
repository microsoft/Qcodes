from collections import namedtuple
from traceback import format_exc
from operator import attrgetter
import logging


class Formatter:
    """
    Data file formatters

    Formatters translate between DataSets and data files.

    Each Formatter is expected to implement writing methods:
    - ``write``: to write the ``DataArray``s
    - ``write_metadata``: to write the metadata JSON structure

    and reading methods:
    - ``read`` or ``read_one_file`` to reconstruct the ``DataArray``s, either
      all at once (``read``) or one file at a time, supplied by the base class
      ``read`` method that loops over all data files at the correct location.

    - ``read_metadata``: to reload saved metadata. If a subclass overrides
      ``read``, this method should call ``read_metadata``, but keep it also
      as a separate method because it occasionally gets called independently.

    All of these methods accept a ``data_set`` argument, which should be a
    ``DataSet`` object. Even if
        io: an IO manager (see qcodes.data.io)
        location: a string, like a file path, that identifies the DataSet and
            tells the IO manager where to store it
        arrays: a dict of {array_id:DataArray} to read into.
            - read will create DataArrays that don't yet exist.
            - write will write ALL DataArrays in the DataSet, using
              last_saved_index and modified_range, as well as whether or not
              it found the specified file, to determine how much to write.
    """
    ArrayGroup = namedtuple('ArrayGroup', 'shape set_arrays data name')

    def write(self, data_set, io_manager, location):
        """
        Write the DataSet to storage.

        Subclasses must override this method.

        It is up to the Formatter to decide when to overwrite completely,
        and when to just append or otherwise update the file(s).

        Args:
            data_set (DataSet): the data we are writing.
            io_manager (io_manager): base physical location to write to.
            location (str): the file location within the io_manager.
        """
        raise NotImplementedError

    def read(self, data_set):
        """
        Read the entire DataSet by finding all files matching its location
        (using io_manager.list) and calling read_one_file from the Formatter
        subclass. Subclasses may alternatively override this entire method.
        """
        io_manager = data_set.io
        location = data_set.location

        data_files = io_manager.list(location)
        if not data_files:
            raise IOError('no data found at ' + location)

        # in case the DataArrays exist but haven't been initialized
        for array in data_set.arrays.values():
            if array.ndarray is None:
                array.init_data()

        self.read_metadata(data_set)

        ids_read = set()
        for fn in data_files:
            with io_manager.open(fn, 'r') as f:
                try:
                    self.read_one_file(data_set, f, ids_read)
                except ValueError:
                    logging.warning('error reading file ' + fn)
                    logging.warning(format_exc())

    def write_metadata(self, data_set, io_manager, location, read_first=True):
        """
        Write the metadata for this DataSet to storage.

        Subclasses must override this method.

        Args:
            data_set (DataSet): the data we are writing.
            io_manager (io_manager): base physical location to write to.
            location (str): the file location within the io_manager.
            read_first (bool, optional): whether to first look for previously
                saved metadata that may contain more information than the local
                copy.
        """
        raise NotImplementedError

    def read_metadata(self, data_set):
        """Read the metadata from this DataSet from storage."""
        raise NotImplementedError

    def read_one_file(self, data_set, f, ids_read):
        """
        Formatter subclasses that handle multiple data files may choose to
        override this method, which handles one file at a time.

        data_set: the DataSet we are reading into
        f: a file-like object to read from
        ids_read: a `set` of array_ids that we have already read.
            when you read an array, check that it's not in this set (except
            setpoints, which can be in several files with different inner loop)
            then add it to the set so other files know not to read it again
        """
        raise NotImplementedError

    def match_save_range(self, group, file_exists, only_complete=True):
        """
        Find the save range that will capture all changes in an array group.
        matches all full-sized arrays: the data arrays plus the inner loop
        setpoint array

        note: if an outer loop has changed values (without the inner
        loop or measured data changing) we won't notice it here

        use the inner setpoint as a base and look for differences
        in last_saved_index and modified_range in the data arrays

        if `only_complete` is True (default), will not mark any range to be
        saved unless it contains no NaN values
        """
        inner_setpoint = group.set_arrays[-1]
        full_dim_data = (inner_setpoint, ) + group.data

        # always return None if there are no modifications,
        # even if there are last_saved_index inconsistencies
        # so we don't do extra writing just to reshape the file
        for array in full_dim_data:
            if array.modified_range:
                break
        else:
            return None

        last_saved_index = inner_setpoint.last_saved_index

        if last_saved_index is None or not file_exists:
            return self._match_save_range_whole_file(
                full_dim_data, only_complete)

        # force overwrite if inconsistent last_saved_index
        for array in group.data:
            if array.last_saved_index != last_saved_index:
                return self._match_save_range_whole_file(
                    full_dim_data, only_complete)

        return self._match_save_range_incremental(
            full_dim_data, last_saved_index, only_complete)

    @staticmethod
    def _match_save_range_whole_file(arrays, only_complete):
        max_save = None
        agg = (min if only_complete else max)
        for array in arrays:
            array_max = array.last_saved_index
            if array_max is None:
                array_max = -1
            mr = array.modified_range
            if mr:
                array_max = max(array_max, mr[1])
            max_save = (array_max if max_save is None else
                        agg(max_save, array_max))

        if max_save >= 0:
            return (0, max_save)
        else:
            return None

    @staticmethod
    def _match_save_range_incremental(arrays, last_saved_index, only_complete):
        mod_ranges = []
        for array in arrays:
            mr = array.modified_range
            if not mr:
                if only_complete:
                    return None
                else:
                    continue
            mod_ranges.append(mr)

        mod_range = mod_ranges[0]
        agg = (min if only_complete else max)
        for mr in mod_ranges[1:]:
            mod_range = (min(mod_range[0], mr[0]),
                         agg(mod_range[1], mr[1]))

        if last_saved_index >= mod_range[1]:
            return (0, last_saved_index)
        elif last_saved_index >= mod_range[0]:
            return (0, mod_range[1])
        else:
            return (last_saved_index + 1, mod_range[1])

    def group_arrays(self, arrays):
        """
        find the sets of arrays which share all the same setpoint arrays
        so each set can be grouped together into one file
        returns ArrayGroup namedtuples
        """

        set_array_sets = tuple(set(array.set_arrays
                                   for array in arrays.values()))
        all_set_arrays = set()
        for set_array_set in set_array_sets:
            all_set_arrays.update(set_array_set)

        grouped_data = [[] for _ in set_array_sets]

        for array in arrays.values():
            i = set_array_sets.index(array.set_arrays)
            if array not in all_set_arrays:  # array.set_arrays[-1] != array:
                # don't include the setpoint array itself in the data
                grouped_data[i].append(array)

        out = []
        id_getter = attrgetter('array_id')
        for set_arrays, data in zip(set_array_sets, grouped_data):
            leni = len(set_arrays)
            if not data and any(1 for other_set_arrays in set_array_sets if
                                len(other_set_arrays) > leni and
                                other_set_arrays[:leni] == set_arrays):
                # this is an outer loop that doesn't have any data of its own,
                # so skip it.
                # Inner-loop setpoints with no data is weird (we set values
                # but didn't measure anything there?) but we should keep it.
                continue

            group_name = '_'.join(sai.array_id for sai in set_arrays)
            out.append(self.ArrayGroup(shape=set_arrays[-1].shape,
                                       set_arrays=set_arrays,
                                       data=tuple(sorted(data, key=id_getter)),
                                       name=group_name))
        return out
