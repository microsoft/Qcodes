from collections import namedtuple
import numpy as np
import re
import math
from traceback import format_exc

from .data_array import DataArray


class Formatter:
    '''
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
    '''
    ArrayGroup = namedtuple('ArrayGroup', 'size set_arrays data name')

    def find_changes(self, arrays):
        '''
        Collect changes made to any of these arrays and determine whether
        the WHOLE group is elligible for appending or not.
        Subclasses may choose to use or ignore this information.
        '''
        new_data = {}
        can_append = True

        for array in arrays.values():
            if array.modified_range:
                if array.modified_range[0] <= array.last_saved_index:
                    can_append = False
                    new_data[array.array_id] = 'overwrite'
                else:
                    new_data[array.array_id] = 'append'

        return new_data, can_append

    def mark_saved(self, arrays):
        '''
        Mark all DataArrays in this group as saved
        '''
        for array in arrays.values():
            array.mark_saved()

    def write(self, data_set):
        '''
        Write the DataSet to storage. It is up to the Formatter to decide
        when to overwrite completely, and when to just append or otherwise
        update the file(s).
        '''
        raise NotImplementedError

    def read(self, data_set):
        '''
        Read the entire DataSet by finding all files matching its location
        (using io_manager.list) and calling read_one_file from the Formatter
        subclass. Subclasses may alternatively override this entire method.
        '''
        io_manager = data_set.io
        location = data_set.location

        data_files = io_manager.list(location)
        if not data_files:
            raise IOError('no data found at ' + location)

        # in case the DataArrays exist but haven't been initialized
        for array in data_set.arrays.values():
            if array.data is None:
                array.init_data()

        ids_read = set()
        for fn in data_files:
            with io_manager.open(fn, 'r') as f:
                try:
                    self.read_one_file(data_set, f, ids_read)
                except ValueError:
                    print(format_exc())
                    print('error reading file ' + fn)

    def read_one_file(self, data_set, f, ids_read):
        raise NotImplementedError

    def match_save_range(self, group, file_exists):
        '''
        Find the save range that will capture all changes in an array group.
        matches all full-sized arrays: the data arrays plus the inner loop
        setpoint array

        note: if an outer loop has changed values (without the inner
        loop or measured data changing) we won't notice it here

        use the inner setpoint as a base and look for differences
        in last_saved_index and modified_range in the data arrays
        '''
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

        # update all sources with the new matching values
        for array in group.data + (inner_setpoint, ):
            array.modified_range = modified_range
            array.last_saved_index = last_saved_index

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

        return last_saved_index, modified_range

    def group_arrays(self, arrays):
        '''
        find the sets of arrays which share all the same setpoint arrays
        so each set can be grouped together into one file
        returns ArrayGroup namedtuples
        '''

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
                                       data=tuple(data),
                                       name=group_name))
        return out


class GNUPlotFormat(Formatter):
    '''
    Saves data in one or more gnuplot-format files. We make one file for
    each set of matching dependent variables in the loop.

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

    # id1\tid2\t\id3...
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
    '''
    def __init__(self, extension='dat', terminator='\n', separator='\t',
                 comment='# ', number_format='g'):
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
        '''
        Called by Formatter.read to bring one data file into
        a DataSet. Setpoint data may be duplicated across multiple files,
        but each measured DataArray must only map to one file.
        '''
        if not f.name.endswith(self.extension):
            return

        arrays = data_set.arrays
        ids = self._read_comment_line(f).split()
        labels = self._get_labels(self._read_comment_line(f))
        size = tuple(map(int, self._read_comment_line(f).split()))
        ndim = len(size)

        set_arrays = ()
        data_arrays = []
        indexed_ids = list(enumerate(ids))

        for i, array_id in indexed_ids[:ndim]:
            # setpoint arrays
            set_size = size[: i + 1]
            if array_id in arrays:
                set_array = arrays[array_id]
                if set_array.size != set_size:
                    raise ValueError(
                        'sizes do not match for set array: ' + array_id)
                if array_id not in ids_read:
                    # it's OK for setpoints to be duplicated across
                    # multiple files, but we should only empty the
                    # array out the first time we see it, so subsequent
                    # reads can check for consistency
                    set_array.clear()
            else:
                set_array = DataArray(label=labels[i], array_id=array_id,
                                      set_arrays=set_arrays, size=set_size)
                set_array.init_data()
                data_set.add_array(set_array)

            set_arrays = set_arrays + (set_array, )
            ids_read.add(array_id)

        for i, array_id in indexed_ids[ndim:]:
            # data arrays
            if array_id in ids_read:
                raise ValueError('duplicate data id found: ' + array_id)

            if array_id in arrays:
                data_array = arrays[array_id]
                data_array.clear()
            else:
                data_array = DataArray(label=labels[i], array_id=array_id,
                                       set_arrays=set_arrays, size=size)
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
                data_array.ndarray[tuple(indices)] = value

            indices[-1] += 1
            first_point = False

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
            parts = re.split('"\s+"', labelstr[1:-1])
            return [l.replace('\\"', '"').replace('\\\\', '\\') for l in parts]

    def write(self, data_set):
        '''
        Write updates in this DataSet to storage. Will choose append if
        possible, overwrite if not.
        '''
        io_manager = data_set.io
        location = data_set.location
        arrays = data_set.arrays

        groups = self.group_arrays(arrays)
        existing_files = set(io_manager.list(location))
        written_files = set()

        for group in groups:
            if len(groups) == 1:
                fn = io_manager.join(location + self.extension)
            else:
                fn = io_manager.join(location, group.name + self.extension)

            written_files.add(fn)

            file_exists = fn in existing_files
            save_range = self.match_save_range(group, file_exists)

            if save_range is None:
                continue

            overwrite = save_range[0] == 0
            open_mode = 'w' if overwrite else 'a'
            shape = group.set_arrays[-1].shape

            with io_manager.open(fn, open_mode) as f:
                if overwrite:
                    f.write(self._make_header(group))

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

        # tell gnuplot-loader only to use written_files

    def _make_header(self, group):
        ids, labels = [], []
        for array in group.set_arrays + group.data:
            ids.append(array.array_id)
            label = getattr(array, 'label', array.array_id)
            label = label.replace('\\', '\\\\').replace('"', '\\"')
            labels.append('"' + label + '"')

        sizes = [str(size) for size in group.set_arrays[-1].shape]
        if len(sizes) != len(group.set_arrays):
            raise ValueError('array dimensionality does not match setpoints')

        out = (self._comment_line(ids) + self._comment_line(labels) +
               self._comment_line(sizes))

        return out

    def _comment_line(self, items):
        return self.comment + self.separator.join(items) + self.terminator

    def _data_point(self, group, indices):
        for array in group.set_arrays:
            yield self.number_format.format(array[indices[:array.ndim]])

        for array in group.data:
            yield self.number_format.format(array[indices])
