import numpy as np
import re
import math
import json
import logging

from qcodes.utils.helpers import deep_update, NumpyJSONEncoder
from .data_array import DataArray
from .format import Formatter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data_set import DataSet


log = logging.getLogger(__name__)


class GNUPlotFormat(Formatter):
    """
    Saves data in one or more gnuplot-format files. We make one file for
    each set of matching dependent variables in the loop.

    Args:

        extension (str): file extension for data files. Defaults to
            'dat'

        terminator (str): newline character(s) to use on write
            not used for reading, we will read any combination of '\\\\r'
            and '\\\\n'. Defaults to '\\\\n'

        separator (str): field (column) separator, must be whitespace.
            Only used for writing, we will read with any whitespace separation.
            Defaults to '\\\\t'.

        comment (str): lines starting with this are not data
            Comments are written with this full string, and identified on read
            by just the string after stripping whitespace. Defaults to '# '.

        number_format (str): from the format mini-language, how to
            format numeric data into a string. Defaults to 'g'.

        always_nest (bool): whether to always make a folder for files
            or just make a single data file if all data has the same setpoints.
            Defaults to bool.

    These files are basically tab-separated values, but any quantity of
    any whitespace characters is accepted.

    Each row represents one setting of the setpoint variable(s)
    the setpoint variable(s) are in the first column(s)
    measured variable(s) come after.

    The data is preceded by comment lines (starting with #).
    We use three:

    - one for the variable name
    - the (longer) axis label, in quotes so a label can contain whitespace.
    - for each dependent var, the (max) number of points in that dimension
      (this also tells us how many dependent vars we have in this file)

    ::

        # id1\tid2\tid3...
        # "label1"\t"label2"\t"label3"...
        # 100\t250
        1\t2\t3...
        2\t3\t4...

    For data of 2 dependent variables, gnuplot puts each inner loop into one
    block, then increments the outer loop in the next block, separated by a
    blank line.

    We extend this to an arbitrary quantity of dependent variables by using
    one blank line for each loop level that resets. (gnuplot *does* seem to
    use 2 blank lines sometimes, to denote a whole new dataset, which sort
    of corresponds to our situation.)
    """

    def __init__(self, extension='dat', terminator='\n', separator='\t',
                 comment='# ', number_format='.15g', metadata_file=None):
        self.metadata_file = metadata_file or 'snapshot.json'
        # file extension: accept either with or without leading dot
        self.extension = '.' + extension.lstrip('.')

        # line terminator (only used for writing; will read any \r\n combo)
        if terminator not in ('\r', '\n', '\r\n'):
            raise ValueError(
                r'GNUPlotFormat terminator must be \r, \n, or \r\n')
        self.terminator = terminator

        # field separator (only used for writing; will read any whitespace)
        if not re.fullmatch(r'\s+', separator):
            raise ValueError('GNUPlotFormat separator must be whitespace')
        self.separator = separator

        # beginning of a comment line. (when reading, just checks the
        # non-whitespace character(s) of comment
        self.comment = comment
        self.comment_chars = comment.rstrip()
        if not self.comment_chars:
            raise ValueError('comment must have some non-whitespace')
        self.comment_len = len(self.comment_chars)

        # number format (only used for writing; will read any number)
        self.number_format = '{:' + number_format + '}'

    def read_one_file(self, data_set, f, ids_read):
        """
        Called by Formatter.read to bring one data file into
        a DataSet. Setpoint data may be duplicated across multiple files,
        but each measured DataArray must only map to one file.

        args:
            data_set: the DataSet we are reading into
            f: a file-like object to read from
            ids_read: a `set` of array_ids that we have already read.
                when you read an array, check that it's not in this set (except
                setpoints, which can be in several files with different inner loop)
                then add it to the set so other files know not to read it again
        """
        if not f.name.endswith(self.extension):
            return

        arrays = data_set.arrays
        ids = self._read_comment_line(f).split()
        labels = self._get_labels(self._read_comment_line(f))
        shape = tuple(map(int, self._read_comment_line(f).split()))
        ndim = len(shape)

        set_arrays = ()
        data_arrays = []
        indexed_ids = list(enumerate(ids))

        for i, array_id in indexed_ids[:ndim]:
            snap = data_set.get_array_metadata(array_id)

            # setpoint arrays
            set_shape = shape[: i + 1]
            if array_id in arrays:
                set_array = arrays[array_id]
                if set_array.shape != set_shape:
                    raise ValueError(
                        'shapes do not match for set array: ' + array_id)
                if array_id not in ids_read:
                    # it's OK for setpoints to be duplicated across
                    # multiple files, but we should only empty the
                    # array out the first time we see it, so subsequent
                    # reads can check for consistency
                    set_array.clear()
            else:
                set_array = DataArray(label=labels[i], array_id=array_id,
                                      set_arrays=set_arrays, shape=set_shape,
                                      is_setpoint=True, snapshot=snap)
                set_array.init_data()
                data_set.add_array(set_array)

            set_arrays = set_arrays + (set_array, )
            ids_read.add(array_id)

        for i, array_id in indexed_ids[ndim:]:
            snap = data_set.get_array_metadata(array_id)

            # data arrays
            if array_id in ids_read:
                raise ValueError('duplicate data id found: ' + array_id)

            if array_id in arrays:
                data_array = arrays[array_id]
                data_array.clear()
            else:
                data_array = DataArray(label=labels[i], array_id=array_id,
                                       set_arrays=set_arrays, shape=shape,
                                       snapshot=snap)
                data_array.init_data()
                data_set.add_array(data_array)
            data_arrays.append(data_array)
            ids_read.add(array_id)

        indices = [0] * ndim
        first_point = True
        resetting = 0
        for line in f:
            if self._is_comment(line):
                continue

            # ignore leading or trailing whitespace (including in blank lines)
            line = line.strip()

            if not line:
                # each consecutive blank line implies one more loop to reset
                # when we read the next data point. Don't depend on the number
                # of setpoints that change, as there could be weird cases, like
                # bidirectional sweeps, or highly diagonal sweeps, where this
                # is incorrect. Anyway this really only matters for >2D sweeps.
                if not first_point:
                    resetting += 1
                continue

            values = tuple(map(float, line.split()))

            if resetting:
                indices[-resetting - 1] += 1
                indices[-resetting:] = [0] * resetting
                resetting = 0

            for value, set_array in zip(values[:ndim], set_arrays):
                nparray = set_array.ndarray
                myindices = tuple(indices[:nparray.ndim])
                stored_value = nparray[myindices]
                if math.isnan(stored_value):
                    nparray[myindices] = value
                elif stored_value != value:
                    raise ValueError('inconsistent setpoint values',
                                     stored_value, value, set_array.name,
                                     myindices, indices)

            for value, data_array in zip(values[ndim:], data_arrays):
                # set .ndarray directly to avoid the overhead of __setitem__
                # which updates modified_range on every call
                data_array.ndarray[tuple(indices)] = value

            indices[-1] += 1
            first_point = False

        # Since we skipped __setitem__, back up to the last read point and
        # mark it as saved that far.
        # Using mark_saved is better than directly setting last_saved_index
        # because it also ensures modified_range is set correctly.
        indices[-1] -= 1
        for array in set_arrays + tuple(data_arrays):
            array.mark_saved(array.flat_index(indices[:array.ndim]))

    def _is_comment(self, line):
        return line[:self.comment_len] == self.comment_chars

    def _read_comment_line(self, f):
        s = f.readline()
        if not self._is_comment(s):
            raise ValueError('expected a comment line, found:\n' + s)
        return s[self.comment_len:]

    def _get_labels(self, labelstr):
        labelstr = labelstr.strip()
        if labelstr[0] != '"' or labelstr[-1] != '"':
            # fields are *not* quoted
            return labelstr.split()
        else:
            # fields *are* quoted (and escaped)
            parts = re.split('"\\s+"', labelstr[1:-1])
            return [l.replace('\\"', '"').replace('\\\\', '\\') for l in parts]

    # this signature is unfortunatly incompatible with the super class
    # so we have to ignore type errors
    def write(self,  # type: ignore[override]
              data_set: 'DataSet',
              io_manager, location, force_write=False,
              write_metadata=True, only_complete=True,
              filename=None):
        """
        Write updates in this DataSet to storage.

        Will choose append if possible, overwrite if not.

        Args:
            data_set: the data we're storing
            io_manager (io_manager): the base location to write to
            location (str): the file location within io_manager
            only_complete (bool): passed to match_save_range, answers the
                following question: Should we write all available new data,
                or only complete rows? Is used to make sure that everything
                gets written when the DataSet is finalised, even if some
                dataarrays are strange (like, full of nans)
            filename (Optional[str]): Filename to save to. Will override
                the usual naming scheme and possibly overwrite files, so
                use with care. The file will be saved in the normal location.
        """
        arrays = data_set.arrays

        # puts everything with same dimensions together
        groups = self.group_arrays(arrays)
        existing_files = set(io_manager.list(location))
        written_files = set()

        # Every group gets its own datafile
        for group in groups:
            log.debug('Attempting to write the following '
                      'group: {}'.format(group.name))
            # it might be useful to output the whole group as below but it is
            # very verbose
            #log.debug('containing {}'.format(group))

            if filename:
                fn = io_manager.join(location, filename + self.extension)
            else:
                fn = io_manager.join(location, group.name + self.extension)

            written_files.add(fn)

            # fn may or may not be an absolute path depending on the location manager
            # used however, io_manager always returns relative paths so make sure both are
            # relative by calling to_location
            file_exists = io_manager.to_location(fn) in existing_files
            save_range = self.match_save_range(group, file_exists,
                                               only_complete=only_complete)

            if save_range is None:
                log.debug('Cannot match save range, skipping this group.')
                continue

            overwrite = save_range[0] == 0 or force_write
            open_mode = 'w' if overwrite else 'a'
            shape = group.set_arrays[-1].shape

            with io_manager.open(fn, open_mode) as f:
                if overwrite:
                    f.write(self._make_header(group))
                    log.debug('Wrote header to file')

                for i in range(save_range[0], save_range[1] + 1):
                    indices = np.unravel_index(i, shape)

                    # insert a blank line for each loop that reset (to index 0)
                    # note that if *all* indices are zero (the first point)
                    # we won't put any blanks
                    for j, index in enumerate(reversed(indices)):
                        if index != 0:
                            if j:
                                f.write(self.terminator * j)
                            break

                    one_point = self._data_point(group, indices)
                    f.write(self.separator.join(one_point) + self.terminator)
                log.debug('Wrote to file from '
                          '{} to {}'.format(save_range[0], save_range[1]+1))
            # now that we've saved the data, mark it as such in the data.
            # we mark the data arrays and the inner setpoint array. Outer
            # setpoint arrays have different dimension (so would need a
            # different unraveled index) but more importantly could have
            # a different saved range anyway depending on whether there
            # is outer data taken before or after the inner loop. Anyway we
            # never look at the outer setpoint last_saved_index or
            # modified_range, we just assume it's got the values we need.
            for array in group.data + (group.set_arrays[-1],):
                array.mark_saved(save_range[1])

        if write_metadata:
            self.write_metadata(
                data_set, io_manager=io_manager, location=location)

    def write_metadata(self, data_set: 'DataSet', io_manager, location,
                       read_first=True, **kwargs):
        """
        Write all metadata in this DataSet to storage.

        Args:
            data_set: the data we're storing

            io_manager (io_manager): the base location to write to

            location (str): the file location within io_manager

            read_first (Optional[bool]): read previously saved metadata before
                writing? The current metadata will still be the used if
                there are changes, but if the saved metadata has information
                not present in the current metadata, it will be retained.
                Default True.
        """
        if read_first:
            # In case the saved file has more metadata than we have here,
            # read it in first. But any changes to the in-memory copy should
            # override the saved file data.
            memory_metadata = data_set.metadata
            data_set.metadata = {}
            self.read_metadata(data_set)
            deep_update(data_set.metadata, memory_metadata)

        fn = io_manager.join(location, self.metadata_file)
        with io_manager.open(fn, 'w', encoding='utf8') as snap_file:
            json.dump(data_set.metadata, snap_file, sort_keys=False,
                      indent=4, ensure_ascii=False, cls=NumpyJSONEncoder)

    def read_metadata(self, data_set):
        io_manager = data_set.io
        location = data_set.location
        fn = io_manager.join(location, self.metadata_file)
        if io_manager.list(fn):
            with io_manager.open(fn, 'r') as snap_file:
                metadata = json.load(snap_file)
            data_set.metadata.update(metadata)

    def _make_header(self, group):
        ids, labels = [], []
        for array in group.set_arrays + group.data:
            ids.append(array.array_id)
            label = getattr(array, 'label', array.array_id)
            label = label.replace('\\', '\\\\').replace('"', '\\"')
            labels.append('"' + label + '"')

        shape = [str(size) for size in group.set_arrays[-1].shape]
        if len(shape) != len(group.set_arrays):
            raise ValueError('array dimensionality does not match setpoints')

        out = (self._comment_line(ids) + self._comment_line(labels) +
               self._comment_line(shape))

        return out

    def _comment_line(self, items):
        return self.comment + self.separator.join(items) + self.terminator

    def _data_point(self, group, indices):
        for array in group.set_arrays:
            yield self.number_format.format(array[indices[:array.ndim]])

        for array in group.data:
            yield self.number_format.format(array[indices])
