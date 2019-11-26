import numpy as np
import logging
import h5py
import os
import json
from typing import TYPE_CHECKING

from ..version import __version__ as _qcodes_version
from .data_array import DataArray
from .format import Formatter

if TYPE_CHECKING:
    from .data_set import DataSet

class HDF5Format(Formatter):
    """
    HDF5 formatter for saving qcodes datasets.

    Capable of storing (write) and recovering (read) qcodes datasets.

    """

    _format_tag = 'hdf5'

    def close_file(self, data_set: 'DataSet'):
        """
        Closes the hdf5 file open in the dataset.

        Args:
            data_set: DataSet object
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
        creates a hdf5 file (data_object) at a location specified by
        filepath
        """
        folder, _filename = os.path.split(filepath)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        file = h5py.File(filepath, 'a')
        return file

    def _open_file(self, data_set, location=None):
        if location is None:
            location = data_set.location
        filepath = self._filepath_from_location(location,
                                                io_manager=data_set.io)
        data_set._h5_base_group = h5py.File(filepath, 'r+')

    def read(self, data_set: 'DataSet', location=None):
        """
        Reads an hdf5 file specified by location into a data_set object.
        If no data_set is provided will create an empty data_set to read into.


        Args:
            data_set: the data to read into. Should already have
                attributes ``io`` (an io manager), ``location`` (string),
                and ``arrays`` (dict of ``{array_id: array}``, can be empty
                or can already have some or all of the arrays present, they
                expect to be overwritten)
            location (None or str): Location to write the data. If no location 
                is provided will use the location specified in the dataset.
        """
        self._open_file(data_set, location)

        if '__format_tag' in data_set._h5_base_group.attrs:
            format_tag = data_set._h5_base_group.attrs['__format_tag']
            if format_tag != self._format_tag:
                raise Exception('format tag %s does not match tag %s of file %s' %
                                (format_tag, self._format_tag, location))

        for i, array_id in enumerate(
                data_set._h5_base_group['Data Arrays'].keys()):
            # Decoding string is needed because of h5py/issues/379
            name = array_id  # will be overwritten if not in file
            dat_arr = data_set._h5_base_group['Data Arrays'][array_id]

            # write ensures these attributes always exist
            name = dat_arr.attrs['name'].decode()
            label = dat_arr.attrs['label'].decode()

            # get unit from units if no unit field, for backward compatibility
            if 'unit' in dat_arr.attrs:
                unit = dat_arr.attrs['unit'].decode()
            else:
                unit = dat_arr.attrs['units'].decode()

            is_setpoint = str_to_bool(dat_arr.attrs['is_setpoint'].decode())
            # if not is_setpoint:
            set_arrays = dat_arr.attrs['set_arrays']
            set_arrays = [s.decode() for s in set_arrays]
            # else:
            #     set_arrays = ()
            vals = dat_arr[:, 0]
            if 'shape' in dat_arr.attrs.keys():
                # extend with NaN if needed
                esize = np.prod(dat_arr.attrs['shape'])
                vals = np.append(vals, [np.nan] * (esize - vals.size))
                vals = vals.reshape(dat_arr.attrs['shape'])
            if array_id not in data_set.arrays.keys():  # create new array
                d_array = DataArray(
                    name=name, array_id=array_id, label=label, parameter=None,
                    unit=unit,
                    is_setpoint=is_setpoint, set_arrays=(),
                    preset_data=vals)
                data_set.add_array(d_array)
            else:  # update existing array with extracted values
                d_array = data_set.arrays[array_id]
                d_array.name = name
                d_array.label = label
                d_array.unit = unit
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
        data_set._h5_base_group.attrs['__qcodes_version'] = _qcodes_version
        data_set._h5_base_group.attrs['__format_tag'] = self._format_tag

        return data_set._h5_base_group

    def write(self, data_set, io_manager=None, location=None,
              force_write=False, flush=True, write_metadata=True,
              only_complete=False):
        """
        Writes a data_set to an hdf5 file.

        Args:
            data_set: qcodes data_set to write to hdf5 file
            io_manager: io_manger used for providing path
            location: location can be used to specify custom location
            force_write (bool): if True creates a new file to write to
            flush (bool) : whether to flush after writing, can be disabled
                for testing or performance reasons
            write_metadata (bool): If True write the dataset metadata to disk
            only_complete (bool): Not used by this formatter, but must be
                included in the call signature to avoid an "unexpected
                keyword argument" TypeError.

        N.B. It is recommended to close the file after writing, this can be
        done by calling ``HDF5Format.close_file(data_set)`` or
        ``data_set.finalize()`` if the data_set formatter is set to an
        hdf5 formatter.  Note that this is not required if the dataset
        is created from a Loop as this includes a data_set.finalize()
        statement.

        The write function consists of two parts, writing DataArrays and
        writing metadata.

            - The main part of write consists of writing and resizing arrays,
              the resizing providing support for incremental writes.

            - write_metadata is called at the end of write and dumps a
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
            try:
                # get latest NaN element
                new_dlen = (~np.isnan(x)).flatten().nonzero()[0][-1] + 1
            except IndexError:
                new_dlen = old_dlen

            new_datasetshape = (new_dlen,
                                datasetshape[1])
            dset.resize(new_datasetshape)
            new_data_shape = (new_dlen - old_dlen, datasetshape[1])
            dset[old_dlen:new_dlen] = x[old_dlen:new_dlen].reshape(
                new_data_shape)
            # allow resizing extracted data, here so it gets written for
            # incremental writes aswell
            dset.attrs['shape'] = x.shape
        if write_metadata:
            self.write_metadata(
                data_set, io_manager=io_manager, location=location)

        # flush ensures buffers are written to disk
        # (useful for ensuring openable by other files)
        if flush:
            data_set._h5_base_group.file.flush()

    def _create_dataarray_dset(self, array, group):
        '''
        input arguments
        array:  Dataset data array
        group:  group in the hdf5 file where the dset will be created

        creates a hdf5 datasaset that represents the data array.
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

        # Create the hdf5 dataset
        dset = group.create_dataset(
            array.array_id, (0, 1),
            maxshape=(None, 1))
        dset.attrs['label'] = _encode_to_utf8(str(label))
        dset.attrs['name'] = _encode_to_utf8(str(name))
        dset.attrs['unit'] = _encode_to_utf8(str(array.unit or ''))
        dset.attrs['is_setpoint'] = _encode_to_utf8(str(array.is_setpoint))

        set_arrays = []
        # list will remain empty if array does not have set_array
        for i in range(len(array.set_arrays)):
            set_arrays += [_encode_to_utf8(
                str(array.set_arrays[i].array_id))]
        dset.attrs['set_arrays'] = set_arrays

        return dset

    def write_metadata(self, data_set, io_manager=None, location=None, read_first=True, **kwargs):
        """
        Writes metadata of dataset to file using write_dict_to_hdf5 method

        Note that io and location are arguments that are only here because
        of backwards compatibility with the loop.
        This formatter uses io and location as specified for the main
        dataset.
        The read_first argument is ignored.
        """
        if not hasattr(data_set, '_h5_base_group'):
            # added here because loop writes metadata before data itself
            data_set._h5_base_group = self._create_data_object(data_set)
        if 'metadata' in data_set._h5_base_group.keys():
            del data_set._h5_base_group['metadata']
        metadata_group = data_set._h5_base_group.create_group('metadata')
        self.write_dict_to_hdf5(data_set.metadata, metadata_group)

        # flush ensures buffers are written to disk
        # (useful for ensuring openable by other files)
        data_set._h5_base_group.file.flush()

    def _read_list_group(self, entry_point, list_type):
        d = {}
        self.read_dict_from_hdf5(data_dict=d,
                                 h5_group=entry_point[list_type])

        if list_type == 'tuple':
            item = tuple([d[k] for k in sorted(d.keys())])
        elif list_type == 'list':
            item = [d[k] for k in sorted(d.keys())]
        else:
            raise Exception('type %s not supported' % type(item))

        return item

    def _write_list_group(self, key, item, entry_point, list_type):
        entry_point.create_group(key)
        group_attrs = entry_point[key].attrs
        group_attrs['list_type'] = list_type

        if list_type == 'tuple' or list_type == 'list':
            item = dict((str(v[0]), v[1]) for v in enumerate(item))
        else:
            raise Exception('type %s not supported' % type(item))

        entry_point[key].create_group(list_type)
        self.write_dict_to_hdf5(
            data_dict=item,
            entry_point=entry_point[key][list_type])

    def write_dict_to_hdf5(self, data_dict, entry_point):
        """ Write a (nested) dictionary to HDF5 

        Args:
            data_dict (dict): Dicionary to be written
            entry_point (object): Object to write to
        """
        for key, item in data_dict.items():
            if isinstance(key, (float, int)):
                key = '__' + str(type(key)) + '__' + str(key)

            if isinstance(item, (str, bool, float, int)):
                entry_point.attrs[key] = item
            elif isinstance(item, np.ndarray):
                entry_point.create_dataset(key, data=item)
            elif isinstance(item, (np.int32, np.int64)):
                entry_point.attrs[key] = int(item)
            elif item is None:
                # as h5py does not support saving None as attribute
                # I create special string, note that this can create
                # unexpected behaviour if someone saves a string with this name
                entry_point.attrs[key] = 'NoneType:__None__'
            elif isinstance(item, dict):
                entry_point.create_group(key)
                self.write_dict_to_hdf5(data_dict=item,
                                        entry_point=entry_point[key])
            elif isinstance(item, tuple):
                self._write_list_group(key, item, entry_point, 'tuple')
            elif isinstance(item, list):
                if len(item) > 0:
                    elt_type = type(item[0])
                    if all(isinstance(x, elt_type) for x in item):
                        if isinstance(item[0], (int, float,
                                                np.int32, np.int64)):

                            entry_point.create_dataset(key,
                                                       data=np.array(item))
                            entry_point[key].attrs['list_type'] = 'array'
                        elif isinstance(item[0], str):
                            dt = h5py.special_dtype(vlen=str)
                            data = np.array(item)
                            data = data.reshape((-1, 1))
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
                        self._write_list_group(key, item, entry_point, 'list')
                else:
                    # as h5py does not support saving None as attribute
                    entry_point.attrs[key] = 'NoneType:__emptylist__'

            else:
                logging.warning(
                    'Type "{}" for "{}":"{}" not supported, '
                    'storing as string'.format(type(item), key, item))
                entry_point.attrs[key] = str(item)

    def read_metadata(self, data_set: 'DataSet'):
        """
        Reads in the metadata, this is also called at the end of a read
        statement so there should be no need to call this explicitly.

        Args:
            data_set: Dataset object to read the metadata into
        """
        # checks if there is an open file in the dataset as load_data does
        # reading of metadata before reading the complete dataset
        if not hasattr(self, '_h5_base_group'):
            self._open_file(data_set)
        if 'metadata' in data_set._h5_base_group.keys():
            metadata_group = data_set._h5_base_group['metadata']
            self.read_dict_from_hdf5(data_set.metadata, metadata_group)
        return data_set

    def read_dict_from_hdf5(self, data_dict, h5_group):
        """ Read a dictionary from HDF5 

        Args:
            data_dict (dict): Dataset to read from
            h5_group (object): HDF5 object to read from
        """

        if 'list_type' not in h5_group.attrs:
            for key, item in h5_group.items():
                if isinstance(item, h5py.Group):
                    data_dict[key] = {}
                    data_dict[key] = self.read_dict_from_hdf5(data_dict[key],
                                                              item)
                else:  # item either a group or a dataset
                    if 'list_type' not in item.attrs:
                        data_dict[key] = item[...]
                    else:
                        data_dict[key] = list(item[...])
            for key, item in h5_group.attrs.items():
                if type(item) is str:
                    # Extracts "None" as an exception as h5py does not support
                    # storing None, nested if statement to avoid elementwise
                    # comparison warning
                    if item == 'NoneType:__None__':
                        item = None
                    elif item == 'NoneType:__emptylist__':
                        item = []
                    else:
                        pass
                data_dict[key] = item
        elif h5_group.attrs['list_type'] == 'tuple':
            data_dict = self._read_list_group(h5_group, 'tuple')
        elif h5_group.attrs['list_type'] == 'list':
            data_dict = self._read_list_group(h5_group, 'list')
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
            raise NotImplementedError('cannot read "list_type":"{}"'.format(
                h5_group.attrs['list_type']))
        return data_dict


def _encode_to_utf8(s):
    """
    Required because h5py does not support python3 strings
    converts byte type to string
    """
    return s.encode('utf-8')


def str_to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError("Cannot covert {} to a bool".format(s))


from qcodes.utils.helpers import deep_update, NumpyJSONEncoder


class HDF5FormatMetadata(HDF5Format):

    _format_tag = 'hdf5-json'
    metadata_file = 'snapshot.json'

    def write_metadata(self, data_set: 'DataSet', io_manager=None, location=None, read_first=False, **kwargs):
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
            kwargs (dict): From the dicionary the key sort_keys is extracted (default value: False). If True, then the
                        keys of the metadata will be stored sorted in the json file. Note: sorting is only possible if
                        the keys of the metadata dictionary can be compared.

        """
        sort_keys = kwargs.get('sort_keys', False)

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

        fn = io_manager.join(location, self.metadata_file)
        with io_manager.open(fn, 'w', encoding='utf8') as snap_file:
            json.dump(data_set.metadata, snap_file, sort_keys=sort_keys,
                      indent=4, ensure_ascii=False, cls=NumpyJSONEncoder)

    def read_metadata(self, data_set):
        io_manager = data_set.io
        location = data_set.location
        fn = io_manager.join(location, self.metadata_file)
        if io_manager.list(fn):
            with io_manager.open(fn, 'r') as snap_file:
                metadata = json.load(snap_file)
            data_set.metadata.update(metadata)
