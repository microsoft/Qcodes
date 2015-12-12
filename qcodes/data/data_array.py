import numpy as np


class DataArray(object):
    '''
    A container for one parameter in a measurement loop

    If this is a measured parameter, This object doesn't contain
    the data of the setpoints it was measured at, but it references
    the DataArray objects of these parameters. Those objects only have
    the dimensionality at which they were set - ie the inner loop setpoint
    the same dimensionality as the measured parameter, but the outer
    loop setpoint(s) have lower dimensionality

    when it's first created, a DataArray has no dimensionality, you must call
    .nest for each dimension.

    if preset_data is provided (a numpy array matching size) it is used to
    initialize the data, and the array can still be nested around it.
    Otherwise it is an error to nest an array that already has data.
    '''
    def __init__(self, parameter=None, name=None, label=None, array_id=None,
                 set_arrays=(), size=(), action_indices=(), preset_data=None):
        if parameter is not None:
            self.name = parameter.name
            self.label = getattr(parameter, 'label', self.name)
        else:
            self.name = name
            self.label = name if label is None else label

        self.array_id = array_id
        self.set_arrays = set_arrays
        self.size = size
        self._preset = False

        self.data = None
        if size:
            self.init_data(preset_data)

        self.action_indices = action_indices
        self.last_saved_index = None
        self.modified_range = None

    def nest(self, size, action_index=None, set_array=None):
        '''
        nest this array inside a new outer loop

        size: length of the new loop
        action_index: within the outer loop, which action is this in?
        set_array: a DataArray listing the setpoints of the outer loop
            if this DataArray *is* a setpoint array, you should omit both
            action_index and set_array, and it will reference itself as the
            set_array
        '''
        if self.data is not None and not self._preset:
            raise RuntimeError('Only preset arrays can be nested after data '
                               'is initialized.')

        if set_array is None:
            if self.set_arrays:
                raise TypeError('a setpoint array must be its own inner loop')
            set_array = self

        self.size = (size, ) + self.size

        if action_index is not None:
            self.action_indices = (action_index, ) + self.action_indices

        self.set_arrays = (set_array, ) + self.set_arrays

        if self._preset:
            inner_data = self.data
            self.data = np.ndarray(self.size)
            # existing preset array copied to every index of the nested array.
            for i in range(size):
                self.data[i] = inner_data

            self._set_index_bounds()

        return self

    def init_data(self, data=None):
        '''
        create a data array (if one doesn't exist)
        if data is provided, this array is marked as a preset
        meaning it can still be nested around this data.
        '''
        if data is not None:
            if data.shape != self.size:
                raise ValueError('preset data must be a numpy array '
                                 'with size matching the array size')
            self.data = data
            self._preset = True
        elif self.data is not None:
            if self.data.shape != self.size:
                raise ValueError('data has already been initialized, '
                                 'but its size doesn\'t match self.size')
            return
        else:
            self.data = np.ndarray(self.size)
            self.clear()
        self._set_index_bounds()

    def _set_index_bounds(self):
        self._min_indices = tuple(0 for d in self.size)
        self._max_indices = tuple(d - 1 for d in self.size)

    def clear(self):
        self.data.fill(float('nan'))

    def __setitem__(self, loop_indices, value):
        '''
        set data values. Follows numpy syntax, allowing indices of lower
        dimensionality than the array, if value makes up the extra dimension(s)

        Also updates the record of modifications to the array. If you don't
        want this overhead, you can access self.data directly.
        '''
        min_li = self._flat_index(loop_indices, self._min_indices)
        max_li = self._flat_index(loop_indices, self._max_indices)
        self._update_modified_range(min_li, max_li)

        self.data[loop_indices] = value

    def __getitem__(self, loop_indices):
        return self.data[loop_indices]

    def __getattr__(self, key):
        '''
        pass all other attributes through to the numpy array

        perhaps it would be cleaner to do this by making DataArray
        actually a subclass of ndarray, but because things can happen
        before init_data (before we know how big the array will be)
        it seems better this way.
        '''
        # note that this is similar to safe_getattr, but we're passing
        # through attributes of an attribute, not dict items, so it's
        # simpler. But the same TODO applies, we need to augment __dir__
        if key == 'data' or self.data is None:
            raise AttributeError('no data array has been created')

        return getattr(self.data, key)

    def _flat_index(self, indices, index_fill):
        indices = indices + index_fill[len(indices):]
        return np.ravel_multi_index(tuple(zip(indices)), self.size)[0]

    def _update_modified_range(self, low, high):
        if self.modified_range:
            self.modified_range = (min(self.modified_range[0], low),
                                   max(self.modified_range[1], high))
        else:
            self.modified_range = (low, high)

    def mark_saved(self):
        '''
        after saving data, mark any outstanding modifications as saved
        '''
        if self.modified_range:
            self.last_saved_index = max(self.last_saved_index or 0,
                                        self.modified_range[1])

        self.modified_range = None

    def clear_save(self):
        '''
        make this array look unsaved, so we can force overwrite
        or rewrite, like if we're moving or copying the DataSet
        '''
        if self.last_saved_index is not None:
            self._update_modified_range(0, self.last_saved_index)

        self.last_saved_index = None

    def __repr__(self):
        return '{}: {}\n{}'.format(self.__class__.__name__, self.name,
                                   repr(self.data))
