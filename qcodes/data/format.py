from collections import namedtuple
from traceback import format_exc
from operator import attrgetter
import logging
from typing import TYPE_CHECKING, Set

if TYPE_CHECKING:
    from .data_set import DataSet


log = logging.getLogger(__name__)

class Formatter:
    """
    Data file formatters

    Formatters translate between DataSets and data files.

    Each Formatter is expected to implement writing methods:

    - ``write``: to write the ``DataArrays``
    - ``write_metadata``: to write the metadata structure

    Optionally, if this Formatter keeps the data file(s) open
    between write calls, it may implement:

    - ``close_file``: to perform any final cleanup and release the
      file and any other resources.

    and reading methods:

    - ``read`` or ``read_one_file`` to reconstruct the ``DataArrays``, either
      all at once (``read``) or one file at a time, supplied by the base class
      ``read`` method that loops over all data files at the correct location.

    - ``read_metadata``: to reload saved metadata. If a subclass overrides
      ``read``, this method should call ``read_metadata``, but keep it also
      as a separate method because it occasionally gets called independently.

    All of these methods accept a ``data_set`` argument, which should be a
    ``DataSet`` object. Even if you are loading a new data set from disk, this
    object should already have attributes:

        - io: an IO manager (see qcodes.data.io)
          location: a string, like a file path, that identifies the DataSet and
          tells the IO manager where to store it
        - arrays: a dict of ``{array_id:DataArray}`` to read into.

    - read will create entries that don't yet exist.
    - write will write ALL DataArrays in the DataSet, using
      last_saved_index and modified_range, as well as whether or not
      it found the specified file, to determine how much to write.
    """
    ArrayGroup = namedtuple('ArrayGroup', 'shape set_arrays data name')

    def write(self, data_set: 'DataSet', io_manager, location, write_metadata=True,
              force_write=False, only_complete=True):
        """
        Write the DataSet to storage.

        Subclasses must override this method.

        It is up to the Formatter to decide when to overwrite completely,
        and when to just append or otherwise update the file(s).

        Args:
            data_set: the data we are writing.
            io_manager (io_manager): base physical location to write to.
            location (str): the file location within the io_manager.
            write_metadata (bool): if True, then the metadata is written to disk
            force_write (bool): if True, then the data is written to disk
            only_complete (bool): Used only by the gnuplot formatter's
                overridden version of this method
        """
        raise NotImplementedError

    def read(self, data_set: 'DataSet') -> None:
        """
        Read the entire ``DataSet``.

        Find all files matching ``data_set.location`` (using io_manager.list)
        and call ``read_one_file`` on each. Subclasses may either override
        this method (if they use only one file or want to do their own
        searching) or override ``read_one_file`` to use the search and
        initialization functionality defined here.

        Args:
            data_set: the data to read into. Should already have
                attributes ``io`` (an io manager), ``location`` (string),
                and ``arrays`` (dict of ``{array_id: array}``, can be empty
                or can already have some or all of the arrays present, they
                expect to be overwritten)
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

        ids_read: Set[str] = set()
        for fn in data_files:
            with io_manager.open(fn, 'r') as f:
                try:
                    self.read_one_file(data_set, f, ids_read)
                except ValueError:
                    log.warning('error reading file ' + fn)
                    log.warning(format_exc())

    def write_metadata(self, data_set: 'DataSet',
                       io_manager, location, read_first=True, **kwargs):
        """
        Write the metadata for this DataSet to storage.

        Subclasses must override this method.

        Args:
            data_set: the data we are writing.
            io_manager (io_manager): base physical location to write to.
            location (str): the file location within the io_manager.
            read_first (Optional[bool]): whether to first look for previously
                saved metadata that may contain more information than the local
                copy.
        """
        raise NotImplementedError

    def read_metadata(self, data_set: 'DataSet'):
        """
        Read the metadata from this DataSet from storage.

        Subclasses must override this method.

        Args:
            data_set: the data to read metadata into
        """
        raise NotImplementedError

    def read_one_file(self, data_set: 'DataSet', f, ids_read):
        """
        Read data from a single file into a ``DataSet``.

        Formatter subclasses that break a DataSet into multiple data files may
        choose to override either this method, which handles one file at a
        time, or ``read`` which finds matching files on its own.

        Args:
            data_set: the data we are reading into.

            f: a file-like object to read from, as provided by
                ``io_manager.open``.

            ids_read (set): ``array_ids`` that we have already read.
                When you read an array, check that it's not in this set (except
                setpoints, which can be in several files with different inner
                loops) then add it to the set so other files know it should not
                be read again.

        Raises:
            ValueError: if a duplicate array_id of measured data is found
        """
        raise NotImplementedError

    def match_save_range(self, group, file_exists, only_complete=True):
        """
        Find the save range that will joins all changes in an array group.

        Matches all full-sized arrays: the data arrays plus the inner loop
        setpoint array.

        Note: if an outer loop has changed values (without the inner
        loop or measured data changing) we won't notice it here. We assume
        that before an iteration of the inner loop starts, the outer loop
        setpoint gets set and then does not change later.

        Args:
            group (Formatter.ArrayGroup): a ``namedtuple`` containing the
                arrays that go together in one file, as tuple ``group.data``.

            file_exists (bool): Does this file already exist? If True, and
                all arrays in the group agree on ``last_saved_index``, we
                assume the file has been written up to this index and we can
                append to it. Otherwise we will set the returned range to start
                from zero (so if the file does exist, it gets completely
                overwritten).

            only_complete (bool): Should we write all available new data,
                or only complete rows? If True, we write only the range of
                array indices which all arrays in the group list as modified,
                so that future writes will be able to do a clean append to
                the data file as more data arrives.
                Default True.

        Returns:
            Tuple(int, int): the first and last raveled indices that should
                be saved. Returns None if:
                    * no data is present
                    * no new data can be found
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
            if last_saved_index is None and file_exists:
                log.warning("Inconsistent file information. "
                            "last_save_index is None but file exists. "
                            "Will overwrite")
            if last_saved_index is not None and not file_exists:
                log.warning("Inconsistent file information. "
                            "last_save_index is not None but file does not "
                            "exist. Will rewrite from scratch")
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
        Find the sets of arrays which share all the same setpoint arrays.

        Some Formatters use this grouping to determine which arrays to save
        together in one file.

        Args:
            arrays (Dict[DataArray]): all the arrays in a DataSet

        Returns:
            List[Formatter.ArrayGroup]: namedtuples giving:

            - shape (Tuple[int]): dimensions as in numpy
            - set_arrays (Tuple[DataArray]): the setpoints of this group
            - data (Tuple[DataArray]): measured arrays in this group
            - name (str): a unique name of this group, obtained by joining
              the setpoint array ids.
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
