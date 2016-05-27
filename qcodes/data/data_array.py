import numpy as np
import collections

from qcodes.utils.helpers import DelegateAttributes, full_class


class DataArray(DelegateAttributes):

    """
    A container for one parameter in a measurement loop.

    If this is a measured parameter, This object doesn't contain
    the data of the setpoints it was measured at, but it references
    the DataArray objects of these parameters. Those objects only have
    the dimensionality at which they were set - ie the inner loop setpoint
    the same dimensionality as the measured parameter, but the outer
    loop setpoint(s) have lower dimensionality

    When it's first created, a DataArray has no dimensionality, you must call
    .nest for each dimension.

    If preset_data is provided it is used to initialize the data, and the array
    can still be nested around it (making many copies of the data).
    Otherwise it is an error to nest an array that already has data.

    Once the array is initialized, a DataArray acts a lot like a numpy array,
    because we delegate attributes through to the numpy array
    """

    # attributes of self to include in the snapshot
    SNAP_ATTRS = (
        'array_id',
        'name',
        'shape',
        'units',
        'label',
        'action_indices',
        'is_setpoint')

    # attributes of the parameter (or keys in the incoming snapshot)
    # to copy to DataArray attributes, if they aren't set some other way
    COPY_ATTRS_FROM_INPUT = (
        'name',
        'label',
        'units')

    # keys in the parameter snapshot to omit from our snapshot
    SNAP_OMIT_KEYS = (
        'ts',
        'value',
        '__class__',
        'set_arrays',
        'shape',
        'array_id',
        'action_indices')

    def __init__(self, parameter=None, name=None, label=None, snapshot=None,
                 array_id=None, set_arrays=(), shape=None, action_indices=(),
                 units=None, is_setpoint=False, preset_data=None):

        self.name = name
        self.label = label
        self.shape = shape
        self.units = units
        self.array_id = array_id
        self.is_setpoint = is_setpoint
        self.action_indices = action_indices

        if snapshot is None:
            snapshot = {}
        self._snapshot_input = {}

        if parameter is not None:
            if hasattr(parameter, 'snapshot') and not snapshot:
                snapshot = parameter.snapshot()
            else:
                for attr in self.COPY_ATTRS_FROM_INPUT:
                    if (hasattr(parameter, attr) and
                            not getattr(self, attr, None)):
                        setattr(self, attr, getattr(parameter, attr))

        for key, value in snapshot.items():
            if key not in self.SNAP_OMIT_KEYS:
                self._snapshot_input[key] = value

                if (key in self.COPY_ATTRS_FROM_INPUT and
                        not getattr(self, key, None)):
                    setattr(self, key, value)

        if not self.label:
            self.label = self.name

        self.set_arrays = set_arrays
        self._preset = False

        # store a reference up to the containing DataSet
        # this also lets us make sure a DataArray is only in one DataSet
        self._data_set = None

        self.ndarray = None
        if preset_data is not None:
            self.init_data(preset_data)
        elif shape is None:
            self.shape = ()

        self.last_saved_index = None
        self.modified_range = None

    @property
    def data_set(self):
        return self._data_set

    @data_set.setter
    def data_set(self, new_data_set):
        if (self._data_set is not None and
                new_data_set is not None and
                self._data_set != new_data_set):
            raise RuntimeError('A DataArray can only be part of one DataSet')
        self._data_set = new_data_set

    def nest(self, size, action_index=None, set_array=None):
        """
        nest this array inside a new outer loop

        size: length of the new loop
        action_index: within the outer loop, which action is this in?
        set_array: a DataArray listing the setpoints of the outer loop
            if this DataArray *is* a setpoint array, you should omit both
            action_index and set_array, and it will reference itself as the
            set_array
        """
        if self.ndarray is not None and not self._preset:
            raise RuntimeError('Only preset arrays can be nested after data '
                               'is initialized! {}'.format(self))

        if set_array is None:
            if self.set_arrays:
                raise TypeError('a setpoint array must be its own inner loop')
            set_array = self

        self.shape = (size, ) + self.shape

        if action_index is not None:
            self.action_indices = (action_index, ) + self.action_indices

        self.set_arrays = (set_array, ) + self.set_arrays

        if self._preset:
            inner_data = self.ndarray
            self.ndarray = np.ndarray(self.shape)
            # existing preset array copied to every index of the nested array.
            for i in range(size):
                self.ndarray[i] = inner_data

            self._set_index_bounds()

        return self

    def init_data(self, data=None):
        """
        create a data array (if one doesn't exist)
        if data is provided, this array is marked as a preset
        meaning it can still be nested around this data.
        """
        if data is not None:
            if not isinstance(data, np.ndarray):
                if isinstance(data, collections.Iterator):
                    # faster than np.array(tuple(data)) (or via list)
                    # but requires us to assume float
                    data = np.fromiter(data, float)
                else:
                    data = np.array(data)

            if self.shape is None:
                self.shape = data.shape
            elif data.shape != self.shape:
                raise ValueError('preset data must be a sequence '
                                 'with size matching the array size',
                                 data.shape, self.shape)
            self.ndarray = data
            self._preset = True
        elif self.ndarray is not None:
            if self.ndarray.shape != self.shape:
                raise ValueError('data has already been initialized, '
                                 'but its size doesn\'t match self.shape')
            return
        else:
            self.ndarray = np.ndarray(self.shape)
            self.clear()
        self._set_index_bounds()

    def _set_index_bounds(self):
        self._min_indices = [0 for d in self.shape]
        self._max_indices = [d - 1 for d in self.shape]

    def clear(self):
        """
        Fill the (already existing) data array with nan
        """
        # only floats can hold nan values. I guess we could
        # also raise an error in this case? But generally float is
        # what people want anyway.
        if self.ndarray.dtype != float:
            self.ndarray = self.ndarray.astype(float)
        self.ndarray.fill(float('nan'))

    def __setitem__(self, loop_indices, value):
        """
        set data values. Follows numpy syntax, allowing indices of lower
        dimensionality than the array, if value makes up the extra dimension(s)

        Also updates the record of modifications to the array. If you don't
        want this overhead, you can access self.ndarray directly.
        """
        if isinstance(loop_indices, collections.Iterable):
            min_indices = list(loop_indices)
            max_indices = list(loop_indices)
        else:
            min_indices = [loop_indices]
            max_indices = [loop_indices]

        for i, index in enumerate(min_indices):
            if isinstance(index, slice):
                start, stop, step = index.indices(self.shape[i])
                min_indices[i] = start
                max_indices[i] = start + (
                    ((stop - start - 1)//step) * step)

        min_li = self._flat_index(min_indices, self._min_indices)
        max_li = self._flat_index(max_indices, self._max_indices)
        self._update_modified_range(min_li, max_li)

        self.ndarray.__setitem__(loop_indices, value)

    def __getitem__(self, loop_indices):
        return self.ndarray[loop_indices]

    delegate_attr_objects = ['ndarray']

    def __len__(self):
        """
        must be explicitly delegated, because len() will look for this
        attribute to already exist
        """
        return len(self.ndarray)

    def _flat_index(self, indices, index_fill):
        indices = indices + index_fill[len(indices):]
        return np.ravel_multi_index(tuple(zip(indices)), self.shape)[0]

    def _update_modified_range(self, low, high):
        if self.modified_range:
            self.modified_range = (min(self.modified_range[0], low),
                                   max(self.modified_range[1], high))
        else:
            self.modified_range = (low, high)

    def mark_saved(self, last_saved_index):
        """
        after saving data, mark outstanding modifications up to
        last_saved_index as saved
        """
        if self.modified_range:
            if last_saved_index >= self.modified_range[1]:
                self.modified_range = None
            else:
                self.modified_range = (max(self.modified_range[0],
                                           last_saved_index + 1),
                                       self.modified_range[1])
        self.last_saved_index = last_saved_index

    def clear_save(self):
        """
        make this array look unsaved, so we can force overwrite
        or rewrite, like if we're moving or copying the DataSet
        """
        if self.last_saved_index is not None:
            self._update_modified_range(0, self.last_saved_index)

        self.last_saved_index = None

    def get_synced_index(self):
        if not hasattr(self, 'synced_index'):
            self.init_data()
            self.synced_index = -1

        return self.synced_index

    def get_changes(self, synced_index):
        latest_index = self.last_saved_index
        if latest_index is None:
            latest_index = -1
        if self.modified_range:
            latest_index = max(latest_index, self.modified_range[1])

        vals = [
            self.ndarray[np.unravel_index(i, self.ndarray.shape)]
            for i in range(synced_index + 1, latest_index + 1)
        ]

        if vals:
            return {
                'start': synced_index + 1,
                'stop': latest_index,
                'vals': vals
            }

    def apply_changes(self, start, stop, vals):
        for i, val in enumerate(vals):
            index = np.unravel_index(i + start, self.ndarray.shape)
            self.ndarray[index] = val
        self.synced_index = stop

    def __repr__(self):
        array_id_or_none = ' {}'.format(self.array_id) if self.array_id else ''
        return '{}[{}]:{}\n{}'.format(self.__class__.__name__,
                                      ','.join(map(str, self.shape)),
                                      array_id_or_none, repr(self.ndarray))

    def snapshot(self, update=False):
        """JSON representation of this DataArray."""
        snap = {'__class__': full_class(self)}

        snap.update(self._snapshot_input)

        for attr in self.SNAP_ATTRS:
            snap[attr] = getattr(self, attr)

        return snap
