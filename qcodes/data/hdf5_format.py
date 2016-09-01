import numpy as np
import logging
import h5py
import os

from .data_array import DataArray
from .format import Formatter


class HDF5Format(Formatter):
    """
    HDF5 formatter for saving qcodes datasets.

    Capable of storing (write) and recovering (read) qcodes datasets.
    """
    def close_file(self, data_set):
        """
        Closes the hdf5 file open in the dataset.
        """
        if hasattr(data_set, '_h5_base_group'):
            data_set._h5_base_group.close()
            # Removes reference to closed file
            del data_set._h5_base_group
        else:
            logging.warning(
                'Cannot close file, data_set has no open hdf5 file')

    def _create_file(self, filepath):
        """
        creates a hdf5 file (data_object) at a location specifed by
        filepath
        """
        folder, _filename = os.path.split(filepath)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        file = h5py.File(filepath, 'a')
        return file

    def read(self, data_set, location=None):
        """
        Reads an hdf5 file specified by location into a data_set object.
        If no data_set is provided will creata an empty data_set to read into.
        If no location is provided will use the location specified in the
        dataset.
        """
        if location is None:
            location = data_set.location
        filepath = self._filepath_from_location(location,
                                                io_manager=data_set.io)
        data_set._h5_base_group = h5py.File(filepath, 'r+')

        for i, array_id in enumerate(
                data_set._h5_base_group['Data Arrays'].keys()):
            # Decoding string is needed because of h5py/issues/379
            name = array_id  # will be overwritten if not in file
            dat_arr = data_set._h5_base_group['Data Arrays'][array_id]
            if 'label' in dat_arr.attrs.keys():
                label = dat_arr.attrs['label'].decode()
            else:
                label = None
            if 'name' in dat_arr.attrs.keys():
                name = dat_arr.attrs['name'].decode()
            if 'units' in dat_arr.attrs.keys():
                units = dat_arr.attrs['units'].decode()
            else:
                units = None
            is_setpoint = str_to_bool(dat_arr.attrs['is_setpoint'].decode())
            # if not is_setpoint:
            set_arrays = dat_arr.attrs['set_arrays']
            set_arrays = [s.decode() for s in set_arrays]
            # else:
            #     set_arrays = ()
            vals = dat_arr.value[:, 0]
            if 'shape' in dat_arr.attrs.keys():
                vals = vals.reshape(dat_arr.attrs['shape'])
            if array_id not in data_set.arrays.keys():  # create new array
                d_array = DataArray(
                    name=name, array_id=array_id, label=label, parameter=None,
                    units=units,
                    is_setpoint=is_setpoint, set_arrays=(),
                    preset_data=vals)
                data_set.add_array(d_array)
            else:  # update existing array with extracted values
                d_array = data_set.arrays[array_id]
                d_array.name = name
                d_array.label = label
                d_array.units = units
                d_array.is_setpoint = is_setpoint
                d_array.ndarray = vals
                d_array.shape = dat_arr.attrs['shape']
            # needed because I cannot add set_arrays at this point
            data_set.arrays[array_id]._sa_array_ids = set_arrays

        # Add copy/ref of setarrays (not array id only)
        # Note, this is not pretty but a result of how the dataset works
        for array_id, d_array in data_set.arrays.items():
            for sa_id in d_array._sa_array_ids:
                d_array.set_arrays += (data_set.arrays[sa_id], )
        data_set = self.read_metadata(data_set)
        return data_set

    def _filepath_from_location(self, location, io_manager):
        filename = os.path.split(location)[-1]
        filepath = io_manager.to_path(location +
                                      '/{}.hdf5'.format(filename))
        return filepath

    def _create_data_object(self, data_set, io_manager=None,
                            location=None):
                # Create the file if it is not there yet
        if io_manager is None:
            io_manager = data_set.io
        if location is None:
            location = data_set.location
        filepath = self._filepath_from_location(location, io_manager)
        # note that this creates an hdf5 file in a folder with the same
        # name. This is useful for saving e.g. images in the same folder
        # I think this is a sane default (MAR).
        data_set._h5_base_group = self._create_file(filepath)
        return data_set._h5_base_group

    def write(self, data_set, io_manager=None, location=None,
              force_write=False):
        """
        Writes a data_set to an hdf5 file.
        Write consists of two parts,
            writing arrays
            writing metadata

        Writing is split up in two parts, writing DataArrays and writing
        metadata.
            The main part of write consists of writing and resizing arrays,
            the resizing providing support for incremental writes.

            write_metadata is called at the end of write and dumps a
            dictionary to an hdf5 file. If there already is metadata it will
            delete this and overwrite it with current metadata.
        """
        if not hasattr(data_set, '_h5_base_group') or force_write:
            data_set._h5_base_group = self._create_data_object(
                data_set, io_manager, location)

        data_name = 'Data Arrays'

        if data_name not in data_set._h5_base_group.keys():
            arr_group = data_set._h5_base_group.create_group(data_name)
        else:
            arr_group = data_set._h5_base_group[data_name]

        for array_id in data_set.arrays.keys():
            if array_id not in arr_group.keys() or force_write:
                self._create_dataarray_dset(array=data_set.arrays[array_id],
                                            group=arr_group)
            dset = arr_group[array_id]
            # Resize the dataset and add the new values

            # dataset refers to the hdf5 dataset here
            datasetshape = dset.shape
            old_dlen = datasetshape[0]
            x = data_set.arrays[array_id]
            new_dlen = len(x[~np.isnan(x)])
            new_datasetshape = (new_dlen,
                                datasetshape[1])
            dset.resize(new_datasetshape)
            new_data_shape = (new_dlen-old_dlen, datasetshape[1])
            dset[old_dlen:new_dlen] = x[old_dlen:new_dlen].reshape(
                new_data_shape)
            # allow resizing extracted data, here so it gets written for
            # incremental writes aswell
            dset.attrs['shape'] = x.shape
        self.write_metadata(data_set)

    def _create_dataarray_dset(self, array, group):
        '''
        input arguments
        array:  Dataset data array
        group:  group in the hdf5 file where the dset will be created

        creates a hdf5 datasaset that represents the data array.

        note that the attribute "units" is used for shape determination
        in the case of tuple-like variables.
        '''
        # Check for empty meta attributes, use array_id if name and/or label
        # is not specified
        if array.label is not None:
            label = array.label
        else:
            label = array.array_id
        if array.name is not None:
            name = array.name
        else:
            name = array.array_id
        if array.units is None:
            array.units = ['']  # used for shape determination
        units = array.units
        # Create the hdf5 dataset
        if isinstance(units, str):
            n_cols = 1
        else:
            n_cols = len(array.units)
        dset = group.create_dataset(
            array.array_id, (0, n_cols),
            maxshape=(None, n_cols))
        dset.attrs['label'] = _encode_to_utf8(str(label))
        dset.attrs['name'] = _encode_to_utf8(str(name))
        dset.attrs['units'] = _encode_to_utf8(str(units))
        dset.attrs['is_setpoint'] = _encode_to_utf8(str(array.is_setpoint))

        set_arrays = []
        # list will remain empty if array does not have set_array
        for i in range(len(array.set_arrays)):
            set_arrays += [_encode_to_utf8(
                str(array.set_arrays[i].array_id))]
        dset.attrs['set_arrays'] = set_arrays

        return dset

    def _create_data_arrays_grp(self, data_set, arrays):
        data_arrays_grp = data_set._h5_base_group.create_group('Data Arrays')
        # Allows reshaping but does not allow adding extra parameters
        dset = data_arrays_grp.create_dataset(
            'Data', (0, len(arrays.keys())),
            maxshape=(None, len(arrays.keys())))
        dset.attrs['column names'] = _encode_to_utf8(arrays.keys())

        labels = []
        names = []
        units = []
        for key, arr in arrays.items():
            labels.append(getattr(arr, 'label', key))
            names.append(getattr(arr, 'name', key))
            units.append(getattr(arr, 'units', ''))

        # _encode_to_utf8(str(...)) ensures None gets encoded for h5py aswell
        dset.attrs['labels'] = _encode_to_utf8(str(labels))
        dset.attrs['names'] = _encode_to_utf8(str(names))
        dset.attrs['units'] = _encode_to_utf8(str(units))

        # Added to tell analysis how to extract the data
        data_arrays_grp.attrs['datasaving_format'] = _encode_to_utf8(
            'QCodes hdf5 v0.1')

    def write_metadata(self, data_set, io=None, location=None):
        """
        Writes metadata of dataset to file using write_dict_to_hdf5 method

        Note that io and location are arguments that are only here because
        of backwards compatibility with the loop.
        This formatter uses io and location as specified for the main
        dataset.
        """
        if not hasattr(data_set, '_h5_base_group'):
            # added here because loop writes metadata before data itself
            data_set._h5_base_group = self._create_data_object(data_set)
        if not hasattr(data_set, 'metadata'):
            raise ValueError('data_set has no metadata')
        if 'metadata' in data_set._h5_base_group.keys():
            del data_set._h5_base_group['metadata']
        metadata_group = data_set._h5_base_group.create_group('metadata')
        self.write_dict_to_hdf5(data_set.metadata, metadata_group)

    def write_dict_to_hdf5(self, data_dict, entry_point):
        for key, item in data_dict.items():
            if isinstance(item, (str, bool, tuple, float, int)):
                entry_point.attrs[key] = item
            elif isinstance(item, np.ndarray):
                entry_point.create_dataset(key, data=item)
            elif item is None:
                # as h5py does not support saving None as attribute
                # I create special string, note that this can create
                # unexpected behaviour if someone saves a string with this name
                entry_point.attrs[key] = 'NoneType:__None__'
            elif isinstance(item, dict):
                entry_point.create_group(key)
                self.write_dict_to_hdf5(data_dict=item,
                                        entry_point=entry_point[key])
            elif isinstance(item, list):
                if len(item) > 0:
                    elt_type = type(item[0])
                    if all(isinstance(x, elt_type) for x in item):
                        if isinstance(item[0], (int, float)):
                            entry_point.create_dataset(key,
                                                       data=np.array(item))
                        elif isinstance(item[0], str):
                            dt = h5py.special_dtype(vlen=str)
                            data = np.array(item)
                            ds = entry_point.create_dataset(
                                key, (len(data), 1), dtype=dt)
                            ds[:] = data
                        elif isinstance(item[0], dict):
                            entry_point.create_group(key)
                            group_attrs = entry_point[key].attrs
                            group_attrs['list_type'] = 'dict'
                            base_list_key = 'list_idx_{}'
                            group_attrs['base_list_key'] = base_list_key
                            group_attrs['list_length'] = len(item)
                            for i, list_item in enumerate(item):
                                list_item_grp = entry_point[key].create_group(
                                    base_list_key.format(i))
                                self.write_dict_to_hdf5(
                                    data_dict=list_item,
                                    entry_point=list_item_grp)
                        else:
                            logging.warning(
                                'List of type "{}" for "{}":"{}" not '
                                'supported, storing as string'.format(
                                    elt_type, key, item))
                            entry_point.attrs[key] = str(item)
                    else:
                        logging.warning(
                            'List of mixed type for "{}":"{}" not supported, '
                            'storing as string'.format(type(item), key, item))
                        entry_point.attrs[key] = str(item)
                else:
                    # as h5py does not support saving None as attribute
                    entry_point.attrs[key] = 'NoneType:__emptylist__'

            else:
                logging.warning(
                    'Type "{}" for "{}":"{}" not supported, '
                    'storing as string'.format(type(item), key, item))
                entry_point.attrs[key] = str(item)

    def read_metadata(self, data_set):
        if not hasattr(data_set, 'metadata'):
            data_set.metadata = {}
        if 'metadata' in data_set._h5_base_group.keys():
            metadata_group = data_set._h5_base_group['metadata']
            self.read_dict_from_hdf5(data_set.metadata, metadata_group)
        return data_set

    def read_dict_from_hdf5(self, data_dict, h5_group):
        if 'list_type' not in h5_group.attrs:
            for key, item in h5_group.items():
                if isinstance(item, h5py.Group):
                    data_dict[key] = {}
                    data_dict[key] = self.read_dict_from_hdf5(data_dict[key],
                                                              item)
                else:
                    data_dict[key] = item
            for key, item in h5_group.attrs.items():
                if type(item) is str:
                    # Extracts "None" as an exception as h5py does not support
                    # storing None, nested if statement to avoid elementwise
                    # comparison warning
                    if item == 'NoneType:__None__':
                        item = None
                    elif item == 'NoneType:__emptylist__':
                        item = []
                data_dict[key] = item
        elif h5_group.attrs['list_type'] == 'dict':
            # preallocate empty list
            list_to_be_filled = [None] * h5_group.attrs['list_length']
            base_list_key = h5_group.attrs['base_list_key']
            for i in range(h5_group.attrs['list_length']):
                list_to_be_filled[i] = {}
                self.read_dict_from_hdf5(
                    data_dict=list_to_be_filled[i],
                    h5_group=h5_group[base_list_key.format(i)])

            # THe error is here!, extract correctly but not adding to
            # data dict correctly
            data_dict = list_to_be_filled
        else:
            raise NotImplementedError()
        return data_dict


def _encode_to_utf8(s):
    """
    Required because h5py does not support python3 strings
    """
    # converts byte type to string because of h5py datasaving
    if isinstance(s, str):
        s = s.encode('utf-8')
    # If it is an array of value decodes individual entries
    elif isinstance(s, (np.ndarray, list)):
        s = [si.encode('utf-8') for si in s]
    return s


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError("Cannot covert {} to a bool".format(s))
