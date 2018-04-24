"""DataSet class and factory functions."""

import time
import logging
from traceback import format_exc
from copy import deepcopy
from collections import OrderedDict
from typing import Dict, Callable

from .gnuplot_format import GNUPlotFormat
from .io import DiskIO
from .location import FormatLocation
from qcodes.utils.helpers import DelegateAttributes, full_class, deep_update

log = logging.getLogger(__name__)


def new_data(location=None, loc_record=None, name=None, overwrite=False,
             io=None, **kwargs):
    """
    Create a new DataSet.

    Args:
        location (str or callable or False, optional): If you provide a string,
            it must be an unused location in the io manager. Can also be:

            - a callable ``location provider`` with one required parameter
              (the io manager), and one optional (``record`` dict),
              which returns a location string when called
            - ``False`` - denotes an only-in-memory temporary DataSet.

            Note that the full path to or physical location of the data is a
            combination of io + location. the default ``DiskIO`` sets the base
            directory, which this location is a relative path inside.
            Default ``DataSet.location_provider`` which is initially
            ``FormatLocation()``

        loc_record (dict, optional): If location is a callable, this will be
            passed to it as ``record``

        name (str, optional): overrides the ``name`` key in the ``loc_record``.

        overwrite (bool): Are we allowed to overwrite an existing location?
            Default False.

        io (io_manager, optional): base physical location of the ``DataSet``.
            Default ``DataSet.default_io`` is initially ``DiskIO('.')`` which
            says the root data directory is the current working directory, ie
            where you started the python session.

        arrays (Optional[List[qcodes.DataArray]): arrays to add to the DataSet.
                Can be added later with ``self.add_array(array)``.

        formatter (Formatter, optional): sets the file format/structure to
            write (and read) with. Default ``DataSet.default_formatter`` which
            is initially ``GNUPlotFormat()``.

        write_period (float or None, optional):seconds
            between saves to disk.
    Returns:
        A new ``DataSet`` object ready for storing new data in.
    """
    if io is None:
        io = DataSet.default_io

    if name is not None:
        if not loc_record:
            loc_record = {}
        loc_record['name'] = name

    if location is None:
        location = DataSet.location_provider

    if callable(location):
        location = location(io, record=loc_record)

    if location and (not overwrite) and io.list(location):
        raise FileExistsError('"' + location + '" already has data')

    return DataSet(location=location, io=io, **kwargs)


def load_data(location=None, formatter=None, io=None):
    """
    Load an existing DataSet.

    Args:
        location (str, optional): the location to load from. Default is the
            current live DataSet.
            Note that the full path to or physical location of the data is a
            combination of io + location. the default ``DiskIO`` sets the base
            directory, which this location is a relative path inside.

        formatter (Formatter, optional): sets the file format/structure to
            read with. Default ``DataSet.default_formatter`` which
            is initially ``GNUPlotFormat()``.

        io (io_manager, optional): base physical location of the ``DataSet``.
            Default ``DataSet.default_io`` is initially ``DiskIO('.')`` which
            says the root data directory is the current working directory, ie
            where you started the python session.

    Returns:
        A new ``DataSet`` object loaded with pre-existing data.
    """
    if location is False:
        raise ValueError('location=False means a temporary DataSet, '
                         'which is incompatible with load_data')

    data = DataSet(location=location, formatter=formatter, io=io)
    data.read_metadata()
    data.read()
    return data


class DataSet(DelegateAttributes):

    """
    A container for one complete measurement loop.

    May contain many individual arrays with potentially different
    sizes and dimensionalities.

    Normally a DataSet should not be instantiated directly, but through
    ``new_data`` or ``load_data``.

    Args:
        location (str or False): A location in the io manager, or ``False`` for
            an only-in-memory temporary DataSet.
            Note that the full path to or physical location of the data is a
            combination of io + location. the default ``DiskIO`` sets the base
            directory, which this location is a relative path inside.

        io (io_manager, optional): base physical location of the ``DataSet``.
            Default ``DataSet.default_io`` is initially ``DiskIO('.')`` which
            says the root data directory is the current working directory, ie
            where you started the python session.

        arrays (Optional[List[qcodes.DataArray]): arrays to add to the DataSet.
                Can be added later with ``self.add_array(array)``.

        formatter (Formatter, optional): sets the file format/structure to
            write (and read) with. Default ``DataSet.default_formatter`` which
            is initially ``GNUPlotFormat()``.

        write_period (float or None, optional): Only if ``mode=LOCAL``, seconds
            between saves to disk. If not ``LOCAL``, the ``DataServer`` handles
            this and generally writes more often. Use None to disable writing
            from calls to ``self.store``. Default 5.

    Attributes:
        background_functions (OrderedDict[callable]): Class attribute,
            ``{key: fn}``: ``fn`` is a callable accepting no arguments, and
            ``key`` is a name to identify the function and help you attach and
            remove it.

            In ``DataSet.complete`` we call each of these periodically, in the
            order that they were attached.

            Note that because this is a class attribute, the functions will
            apply to every DataSet. If you want specific functions for one
            DataSet you can override this with an instance attribute.
    """

    # ie data_set.arrays['vsd'] === data_set.vsd
    delegate_attr_dicts = ['arrays']

    default_io = DiskIO('.')
    default_formatter = GNUPlotFormat()
    location_provider = FormatLocation()

    background_functions: Dict[str, Callable] = OrderedDict()

    def __init__(self, location=None, arrays=None, formatter=None, io=None,
                 write_period=5):
        if location is False or isinstance(location, str):
            self.location = location
        else:
            raise ValueError('unrecognized location ' + repr(location))

        # TODO: when you change formatter or io (and there's data present)
        # make it all look unsaved
        self.formatter = formatter or self.default_formatter
        self.io = io or self.default_io

        self.write_period = write_period
        self.last_write = 0
        self.last_store = -1

        self.metadata = {}

        self.arrays = _PrettyPrintDict()
        if arrays:
            self.action_id_map = self._clean_array_ids(arrays)
            for array in arrays:
                self.add_array(array)

        if self.arrays:
            for array in self.arrays.values():
                array.init_data()

    def sync(self):
        """
        Synchronize this DataSet with the DataServer or storage.

        If this DataSet is on the server, asks the server for changes.
        If not, reads the entire DataSet from disk.

        Returns:
            bool: True if this DataSet is live on the server
        """
        # TODO: sync implies bidirectional... and it could be!
        # we should keep track of last sync timestamp and last modification
        # so we can tell whether this one, the other one, or both copies have
        # changed (and I guess throw an error if both did? Would be cool if we
        # could find a robust and intuitive way to make modifications to the
        # version on the DataServer from the main copy)

        # LOCAL DataSet - no need to sync just use local data
        return False

    def fraction_complete(self):
        """
        Get the fraction of this DataSet which has data in it.

        Returns:
            float: the average of all measured (not setpoint) arrays'
                ``fraction_complete()`` values, independent of the individual
                array sizes. If there are no measured arrays, returns zero.
        """
        array_count, total = 0, 0

        for array in self.arrays.values():
            if not array.is_setpoint:
                array_count += 1
                total += array.fraction_complete()

        return total / (array_count or 1)

    def complete(self, delay=1.5):
        """
        Periodically sync the DataSet and display percent complete status.

        Also, each period, execute functions stored in (class attribute)
        ``self.background_functions``. If a function fails, we log its
        traceback and continue on. If any one function fails twice in
        a row, it gets removed.

        Args:
            delay (float): seconds between iterations. Default 1.5
        """
        log.info(
            'waiting for DataSet <{}> to complete'.format(self.location))

        failing = {key: False for key in self.background_functions}

        completed = False
        while True:
            log.info('DataSet: {:.0f}% complete'.format(
                self.fraction_complete() * 100))

            # first check if we're done
            if self.sync() is False:
                completed = True

            # then even if we *are* done, execute the background functions
            # because we want things like live plotting to get the final data
            for key, fn in list(self.background_functions.items()):
                try:
                    log.debug('calling {}: {}'.format(key, repr(fn)))
                    fn()
                    failing[key] = False
                except Exception:
                    log.info(format_exc())
                    if failing[key]:
                        log.warning(
                            'background function {} failed twice in a row, '
                            'removing it'.format(key))
                        del self.background_functions[key]
                    failing[key] = True

            if completed:
                break

            # but only sleep if we're not already finished
            time.sleep(delay)

        log.info('DataSet <{}> is complete'.format(self.location))

    def get_changes(self, synced_indices):
        """
        Find changes since the last sync of this DataSet.

        Args:
            synced_indices (dict): ``{array_id: synced_index}`` where
                synced_index is the last flat index which has already
                been synced, for any (usually all) arrays in the DataSet.

        Returns:
            Dict[dict]: keys are ``array_id`` for each array with changes,
                values are dicts as returned by ``DataArray.get_changes``
                and required as kwargs to ``DataArray.apply_changes``.
                Note that not all arrays in ``synced_indices`` need be
                present in the return, only those with changes.
        """
        changes = {}

        for array_id, synced_index in synced_indices.items():
            array_changes = self.arrays[array_id].get_changes(synced_index)
            if array_changes:
                changes[array_id] = array_changes

        return changes

    def add_array(self, data_array):
        """
        Add one DataArray to this DataSet, and mark it as part of this DataSet.

        Note: DO NOT just set ``data_set.arrays[id] = data_array``, because
        this will not check if we are overwriting another array, nor set the
        reference back to this DataSet, nor that the ``array_id`` in the array
        matches how you're storing it here.

        Args:
            data_array (DataArray): the new array to add

        Raises:
            ValueError: if there is already an array with this id here.
        """
        # TODO: mask self.arrays so you *can't* set it directly?

        if data_array.array_id in self.arrays:
            raise ValueError('array_id {} already exists in this '
                             'DataSet'.format(data_array.array_id))
        self.arrays[data_array.array_id] = data_array

        # back-reference to the DataSet
        data_array.data_set = self

    def remove_array(self, array_id):
        """ Remove an array from a dataset

        Throws an exception when the array specified is refereced by other
        arrays in the dataset.

        Args:
            array_id (str): array_id of array to be removed
        """
        for a in self.arrays:
            sa = self.arrays[a].set_arrays
            if array_id in [a.array_id for a in sa]:
                raise Exception(
                    'cannot remove array %s as it is referenced by a' % array_id)
        _ = self.arrays.pop(array_id)
        self.action_id_map = self._clean_array_ids(self.arrays.values())

    def _clean_array_ids(self, arrays):
        """
        replace action_indices tuple with compact string array_ids
        stripping off as much extraneous info as possible
        """
        action_indices = [array.action_indices for array in arrays]
        for array in arrays:
            name = array.full_name
            if array.is_setpoint and name and not name.endswith('_set'):
                name += '_set'

            array.array_id = name
        array_ids = set([array.array_id for array in arrays])
        for name in array_ids:
            param_arrays = [array for array in arrays
                            if array.array_id == name]
            self._clean_param_ids(param_arrays, name)

        array_ids = [array.array_id for array in arrays]

        return dict(zip(action_indices, array_ids))

    def _clean_param_ids(self, arrays, name):
        # strip off as many leading equal indices as possible
        # and append the rest to the back of the name with underscores
        param_action_indices = [list(array.action_indices) for array in arrays]
        while all(len(ai) for ai in param_action_indices):
            if len(set(ai[0] for ai in param_action_indices)) == 1:
                for ai in param_action_indices:
                    ai[:1] = []
            else:
                break
        for array, ai in zip(arrays, param_action_indices):
            array.array_id = name + ''.join('_' + str(i) for i in ai)

    def store(self, loop_indices, ids_values):
        """
        Insert data into one or more of our DataArrays.

        Args:
            loop_indices (tuple): the indices within whatever loops we are
                inside. May have fewer dimensions than some of the arrays
                we are inserting into, if the corresponding value makes up
                the remaining dimensionality.
            values (Dict[Union[float, sequence]]): a dict whose keys are
                array_ids, and values are single numbers or entire slices
                to insert into that array.
         """
        for array_id, value in ids_values.items():
            self.arrays[array_id][loop_indices] = value
        self.last_store = time.time()
        if (self.write_period is not None and
                time.time() > self.last_write + self.write_period):
            log.debug('Attempting to write')
            self.write()
            self.last_write = time.time()
        # The below could be useful but as it writes at every single
        # step of the loop its too verbose even at debug
        # else:
        #     log.debug('.store method: This is not the right time to write')

    def default_parameter_name(self, paramname='amplitude'):
        """ Return name of default parameter for plotting

        The default parameter is determined by looking into
        metdata['default_parameter_name'].  If this variable is not present,
        then the closest match to the argument paramname is tried.

        Args:
            paramname (str): Name to match to parameter name

        Returns:
            name ( Union[str, None] ): name of the default parameter
        """

        arraynames = self.arrays.keys()

        # overrule parameter name from the metadata
        if self.metadata.get('default_parameter_name', False):
            paramname = self.metadata['default_parameter_name']

        # try to return the exact name
        if paramname in arraynames:
            return paramname

        # try find something similar
        vv = [v for v in arraynames if v.endswith(paramname)]
        if (len(vv) > 0):
            return vv[0]
        vv = [v for v in arraynames if v.startswith(paramname)]
        if (len(vv) > 0):
            return vv[0]

        # try to get the first non-setpoint array
        vv = [v for v in arraynames if not self.arrays[v].is_setpoint]
        if (len(vv) > 0):
            return sorted(vv)[0]

        # fallback: any array found
        try:
            name = sorted((list(arraynames)))[0]
            return name
        except IndexError:
            pass
        return None

    def default_parameter_array(self, paramname='amplitude'):
        """ Return default parameter array

        Args:
            paramname (str): Name to match to parameter name.
                 Defaults to 'amplitude'

        Returns:
            array (DataArray): array corresponding to the default parameter

        See also:
            default_parameter_name

        """
        paramname = self.default_parameter_name(paramname=paramname)
        return getattr(self, paramname, None)

    def read(self):
        """Read the whole DataSet from storage, overwriting the local data."""
        if self.location is False:
            return
        self.formatter.read(self)

    def read_metadata(self):
        """Read the metadata from storage, overwriting the local data."""
        if self.location is False:
            return
        self.formatter.read_metadata(self)

    def write(self, write_metadata=False, only_complete=True, filename=None):
        """
        Writes updates to the DataSet to storage.
        N.B. it is recommended to call data_set.finalize() when a DataSet is
        no longer expected to change to ensure files get closed

        Args:
            write_metadata (bool): write the metadata to disk
            only_complete (bool): passed on to the match_save_range inside
                self.formatter.write. Used to ensure that all new data gets
                saved even when some columns are strange.
            filename (Optional[str]): The filename (minus extension) to use.
                The file gets saved in the usual location.
        """
        if self.location is False:
            return

        # Only the gnuplot formatter has a "filename" kwarg
        if isinstance(self.formatter, GNUPlotFormat):
            self.formatter.write(self,
                                 self.io,
                                 self.location,
                                 write_metadata=write_metadata,
                                 only_complete=only_complete,
                                 filename=filename)
        else:
            self.formatter.write(self,
                                 self.io,
                                 self.location,
                                 write_metadata=write_metadata,
                                 only_complete=only_complete)

    def write_copy(self, path=None, io_manager=None, location=None):
        """
        Write a new complete copy of this DataSet to storage.

        Args:
            path (str, optional): An absolute path on this system to write to.
                If you specify this, you may not include either ``io_manager``
                or ``location``.

            io_manager (io_manager, optional): A new ``io_manager`` to use with
                either the ``DataSet``'s same or a new ``location``.

            location (str, optional): A new ``location`` to write to, using
                either this ``DataSet``'s same or a new ``io_manager``.
        """
        if io_manager is not None or location is not None:
            if path is not None:
                raise TypeError('If you provide io_manager or location '
                                'to write_copy, you may not provide path.')
            if io_manager is None:
                io_manager = self.io
            elif location is None:
                location = self.location
        elif path is not None:
            io_manager = DiskIO(None)
            location = path
        else:
            raise TypeError('You must provide at least one argument '
                            'to write_copy')

        if location is False:
            raise ValueError('write_copy needs a location, not False')

        lsi_cache = {}
        mr_cache = {}
        for array_id, array in self.arrays.items():
            lsi_cache[array_id] = array.last_saved_index
            mr_cache[array_id] = array.modified_range
            # array.clear_save() is not enough, we _need_ to set modified_range
            # TODO - identify *when* clear_save is not enough, and fix it
            # so we *can* use it. That said, maybe we will *still* want to
            # use the full array here no matter what, or strip trailing NaNs
            # separately, either here or in formatter.write?
            array.last_saved_index = None
            array.modified_range = (0, array.ndarray.size - 1)

        try:
            self.formatter.write(self, io_manager, location, force_write=True)
            self.snapshot()
            self.formatter.write_metadata(self, io_manager, location,
                                          read_first=False)
        finally:
            for array_id, array in self.arrays.items():
                array.last_saved_index = lsi_cache[array_id]
                array.modified_range = mr_cache[array_id]

    def add_metadata(self, new_metadata):
        """
        Update DataSet.metadata with additional data.

        Args:
            new_metadata (dict): new data to be deep updated into
                the existing metadata
        """
        deep_update(self.metadata, new_metadata)

    def save_metadata(self):
        """Evaluate and save the DataSet's metadata."""
        if self.location is not False:
            self.snapshot()
            self.formatter.write_metadata(self, self.io, self.location)

    def finalize(self, filename=None, write_metadata=True):
        """
        Mark the DataSet complete and write any remaining modifications.

        Also closes the data file(s), if the ``Formatter`` we're using
        supports that.

        Args:
            filename (Optional[str]): The file name (minus extension) to
                write to. The location of the file is the usual one.
            write_metadata (bool): Whether to save a snapshot. For e.g. dumping
                raw data inside a loop, a snapshot is not wanted.
        """
        log.debug('Finalising the DataSet. Writing.')
        # write all new data, not only (to?) complete columns
        self.write(only_complete=False, filename=filename)

        if hasattr(self.formatter, 'close_file'):
            self.formatter.close_file(self)

        if write_metadata:
            self.save_metadata()

    def snapshot(self, update=False):
        """JSON state of the DataSet."""
        array_snaps = {}
        for array_id, array in self.arrays.items():
            array_snaps[array_id] = array.snapshot(update=update)

        self.metadata.update({
            '__class__': full_class(self),
            'location': self.location,
            'arrays': array_snaps,
            'formatter': full_class(self.formatter),
            'io': repr(self.io)
        })
        return deepcopy(self.metadata)

    def get_array_metadata(self, array_id):
        """
        Get the metadata for a single contained DataArray.

        Args:
            array_id (str): the array to get metadata for.

        Returns:
            dict: metadata for this array.
        """
        try:
            return self.metadata['arrays'][array_id]
        except (AttributeError, KeyError):
            return None

    def __repr__(self):
        """Rich information about the DataSet and contained arrays."""
        out = type(self).__name__ + ':'

        attrs = [['location', repr(self.location)]]
        attr_template = '\n   {:8} = {}'
        for var, val in attrs:
            out += attr_template.format(var, val)

        arr_info = [['<Type>', '<array_id>', '<array.name>', '<array.shape>']]

        if hasattr(self, 'action_id_map'):
            id_items = [
                item for index, item in sorted(self.action_id_map.items())]
        else:
            id_items = self.arrays.keys()

        for array_id in id_items:
            array = self.arrays[array_id]
            setp = 'Setpoint' if array.is_setpoint else 'Measured'
            name = array.name or 'None'
            array_id = array_id or 'None'
            arr_info.append([setp, array_id, name, repr(array.shape)])

        column_lengths = [max(len(row[i]) for row in arr_info)
                          for i in range(len(arr_info[0]))]
        out_template = ('\n   '
                        '{info[0]:{lens[0]}} | {info[1]:{lens[1]}} | '
                        '{info[2]:{lens[2]}} | {info[3]}')

        for arr_info_i in arr_info:
            out += out_template.format(info=arr_info_i, lens=column_lengths)

        return out


class _PrettyPrintDict(dict):
    """
    simple wrapper for a dict to repr its items on separate lines
    with a bit of indentation
    """

    def __repr__(self):
        body = '\n  '.join([repr(k) + ': ' + self._indent(repr(v))
                            for k, v in self.items()])
        return '{\n  ' + body + '\n}'

    def _indent(self, s):
        lines = s.split('\n')
        return '\n    '.join(lines)
