from collections import namedtuple
import numpy as np
from traceback import format_exc
from operator import attrgetter
import logging


class Formatter:
    """
    Data file formatters

    Formatters translate between DataSets and data files.

    Each Formatter is expected to implement a write method:
        write(self, data_set)

    and either read or read_one_file:
        read(self, data_set)
        read_one_file(data_set, f, ids_read)
            f: a file-like object supporting .readline() and for ... in
            ids_read: a set of array_id's we've already encountered, so
                read_one_file can check for duplication and consistency

    data_set is a DataSet object, which is expected to have attributes:
        io: an IO manager (see qcodes.io)
        location: a string, like a file path, that identifies the DataSet and
            tells the IO manager where to store it
        arrays: a dict of {array_id:DataArray} to read into.
            - read will create DataArrays that don't yet exist.
            - write will write ALL DataArrays in the DataSet, using
              last_saved_index and modified_range, as well as whether or not
              it found the specified file, to determine how much to write.
    """
    ArrayGroup = namedtuple('ArrayGroup', 'size set_arrays data name')

    def write(self, data_set):
        """
        Write the DataSet to storage. It is up to the Formatter to decide
        when to overwrite completely, and when to just append or otherwise
        update the file(s).
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

    def write_metadata(self, data_set, read_first=True):
        """Write the metadata for this DataSet to storage."""
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
        last_saved_index = (inner_setpoint.last_saved_index if file_exists
                            else None)
        modified_range = inner_setpoint.modified_range
        for array in group.data:
            # force overwrite if inconsistent last_saved_index
            if array.last_saved_index != last_saved_index:
                last_saved_index = None

            # find the modified_range that encompasses all modifications
            amr = array.modified_range
            if amr:
                if modified_range:
                    modified_range = (min(modified_range[0], amr[0]),
                                      max(modified_range[1], amr[1]))
                else:
                    modified_range = amr

        if only_complete and modified_range:
            modified_range = self._get_completed_range(modified_range,
                                                       inner_setpoint.shape,
                                                       group.data)
            if not modified_range:
                return None

        # calculate the range to save
        if not modified_range:
            # nothing to save
            return None
        if last_saved_index is None or last_saved_index >= modified_range[0]:
            # need to overwrite - start save from 0
            return (0, modified_range[1])
        else:
            # we can append! save only from last save to end of mods
            return (last_saved_index + 1, modified_range[1])

    def _get_completed_range(self, modified_range, shape, arrays):
        """
        check the last data point to see if it's complete.

        If it's not complete, back up one point so that we don't need
        to rewrite this point later on when it *is* complete

        This should work for regular `Loop` data that comes in sequentially.
        But if you have non-sequential data, such as a parallel simulation,
        then you would want to look farther back.
        """
        last_pt = modified_range[1]
        indices = np.unravel_index(last_pt, shape)
        for array in arrays:
            if np.isnan(array[indices]):
                if last_pt == modified_range[0]:
                    return None
                else:
                    return (modified_range[0], last_pt - 1)
        return modified_range

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
            out.append(self.ArrayGroup(size=set_arrays[-1].size,
                                       set_arrays=set_arrays,
                                       data=tuple(sorted(data, key=id_getter)),
                                       name=group_name))
        return out
