import numpy as np
import h5py
import re
import math
import os

from .data_array import DataArray
from .format import Formatter



class HDF5Format(Formatter):
    """

    """
    def __init__(self):
        '''
        Instantiates the datafile using the location provided by the io_manager
        see h5py detailed info
        '''
        self.data_object = None


    def _create_file(self, filepath):
        folder, _filename = os.path.split(filepath)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        self.data_object = h5py.File(filepath, 'a')


    def write(self, data_set):
        """
        """

        if self.data_object == None:
            # Create the file
            io_manager = data_set.io
            location = data_set.location
            filepath = io_manager.join(
                io_manager.base_location,
                data_set.location_provider(io_manager, location)+'.hdf5')
            self._create_file(filepath)

        arrays = data_set.arrays
        if not hasattr(self, 'data_arrays_grp'):
            self._create_data_arrays_grp(data_set.arrays)

        # Resize the dataset and then append the arrays that need to be written
        datasetshape = self.dset.shape

        key0 = list(data_set.arrays.keys())[0] # Dirty way to get a random key

        # Implicit assumption that data arrays have the same length
        old_datasetlen = datasetshape[0]
        new_datasetshape = (old_datasetlen+len(data_set.arrays[key0])+1,
                            datasetshape[1])
        self.dset.resize(new_datasetshape)
        for i, key in enumerate(data_set.arrays.keys()):
            # Would prefer to write all at once but for loop seems easiest
            # to extract the values from the arrays dict
            self.dset[old_datasetlen:-1, i] = data_set.arrays[key]






    def _create_data_arrays_grp(self, arrays):
        self.data_arrays_grp = self.data_object.create_group('Data Arrays')
        # Allows reshaping but does not allow adding extra parameters
        self.dset = self.data_arrays_grp.create_dataset(
            'Data', (0, len(arrays.keys())),
            maxshape=(None, len(arrays.keys())))
        self.dset.attrs['column names'] = _encode_to_utf8(arrays.keys())
        # Added to tell analysis how to extract the data
        self.data_arrays_grp.attrs['datasaving_format'] = _encode_to_utf8(
            'QCodes hdf5 v0.1')

        # I would like to add parameter units, labels and names aswell
        # data_group.attrs['names'] = _encode_to_utf8('')
        # data_group.attrs['units'] = _encode_to_utf8('')
        # data_group.attrs['labels'] = _encode_to_utf8('')



    def save_instrument_snapshot(self, data_object=None, *args):
        '''
        uses QCodes station snapshot to save the last known value of any
        parameter. Only saves the value and not the update time (which is
        known in the snapshot)

        META DATA GROUP
        '''
        set_grp = data_object.create_group('Meta-data')
        inslist = dict_to_ordered_tuples(self.station.instruments)
        for (iname, ins) in inslist:
            instrument_grp = set_grp.create_group(iname)
            par_snap = ins.snapshot()['parameters']
            parameter_list = dict_to_ordered_tuples(par_snap)
            for (p_name, p) in parameter_list:
                try:
                    val = str(p['value'])
                except KeyError:
                    val = ''
                instrument_grp.attrs[p_name] = str(val)



def _encode_to_utf8(s):
    '''
    Required because h5py does not support python3 strings
    '''
    # converts byte type to string because of h5py datasaving
    if type(s) == str:
        s = s.encode('utf-8')
    # If it is an array of value decodes individual entries
    elif type(s) == np.ndarray or list:
        s = [s.encode('utf-8') for s in s]
    return s
