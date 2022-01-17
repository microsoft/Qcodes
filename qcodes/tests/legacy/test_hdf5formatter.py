import os
from shutil import copy
from unittest import TestCase

import h5py
import numpy as np

import qcodes.data
from qcodes.data.data_array import DataArray
from qcodes.data.data_set import DataSet, load_data, new_data
from qcodes.data.hdf5_format import HDF5Format, str_to_bool
from qcodes.data.location import FormatLocation
from qcodes.loops import Loop
from qcodes.station import Station
from qcodes.tests.instrument_mocks import MockParabola
from qcodes.utils.helpers import compare_dictionaries

from .data_mocks import DataSet1D, DataSet2D


class TestHDF5_Format(TestCase):
    def setUp(self):
        self.io = DataSet.default_io
        self.formatter = HDF5Format()
        # Set up the location provider to always store test data in
        # "qc.tests.unittest_data
        cur_fp = os.path.dirname(__file__)
        base_fp = os.path.abspath(os.path.join(cur_fp, '../unittest_data'))
        self.loc_provider = FormatLocation(
            fmt=base_fp+'/{date}/#{counter}_{name}_{time}')
        DataSet.location_provider = self.loc_provider

    def checkArraysEqual(self, a, b):
        """
        Checks if arrays are equal
        """
        # Modified from GNUplot would be better to have this in some module
        self.checkArrayAttrs(a, b)
        np.testing.assert_array_equal(a, b)
        if len(a.set_arrays) > 1:
            for i, set_arr in enumerate(a.set_arrays):
                np.testing.assert_array_equal(set_arr, b.set_arrays[i])
        else:
            np.testing.assert_array_equal(a.set_arrays, b.set_arrays)

        for sa, sb in zip(a.set_arrays, b.set_arrays):
            self.checkArrayAttrs(sa, sb)

    def checkArrayAttrs(self, a, b):
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(a.label, b.label)
        self.assertEqual(a.array_id, b.array_id)

    def test_full_write_read_1D(self):
        """
        Test writing and reading a file back in
        """
        # location = self.locations[0]
        data = DataSet1D(name='test1D_full_write',
                         location=self.loc_provider)
        # print('Data location:', os.path.abspath(data.location))
        self.formatter.write(data)
        # Used because the formatter has no nice find file method

        # Test reading the same file through the DataSet.read
        data2 = DataSet(location=data.location, formatter=self.formatter)
        data2.read()
        self.checkArraysEqual(data2.x_set, data.x_set)
        self.checkArraysEqual(data2.y, data.y)
        self.formatter.close_file(data)
        self.formatter.close_file(data2)

    def test_full_write_read_2D(self):
        """
        Test writing and reading a file back in
        """
        data = DataSet2D(location=self.loc_provider, name='test2D')
        self.formatter.write(data)
        # Test reading the same file through the DataSet.read
        data2 = DataSet(location=data.location, formatter=self.formatter)
        data2.read()
        self.checkArraysEqual(data2.x_set, data.x_set)
        self.checkArraysEqual(data2.y_set, data.y_set)
        self.checkArraysEqual(data2.z, data.z)

        self.formatter.close_file(data)
        self.formatter.close_file(data2)

    def test_incremental_write(self):
        data = DataSet1D(location=self.loc_provider, name='test_incremental')
        location = data.location
        data_copy = DataSet1D(False)

        # # empty the data and mark it as unmodified
        data.x_set[:] = float('nan')
        data.y[:] = float('nan')
        data.x_set.modified_range = None
        data.y.modified_range = None

        # simulate writing after every value comes in, even within
        # one row (x comes first, it's the setpoint)
        for i, (x, y) in enumerate(zip(data_copy.x_set, data_copy.y)):
            data.x_set[i] = x
            self.formatter.write(data)
            data.y[i] = y
            self.formatter.write(data)
        data2 = DataSet(location=location, formatter=self.formatter)
        data2.read()
        self.checkArraysEqual(data2.arrays['x_set'], data_copy.arrays['x_set'])
        self.checkArraysEqual(data2.arrays['y'], data_copy.arrays['y'])

        self.formatter.close_file(data)
        self.formatter.close_file(data2)

    def test_metadata_write_read(self):
        """
        Test is based on the snapshot of the 1D dataset.
        Having a more complex snapshot in the metadata would be a better test.
        """
        data = DataSet1D(location=self.loc_provider, name='test_metadata')
        data.snapshot()  # gets the snapshot, not added upon init
        self.formatter.write(data)  # write_metadata is included in write
        data2 = DataSet(location=data.location, formatter=self.formatter)
        data2.read()
        self.formatter.close_file(data)
        self.formatter.close_file(data2)
        metadata_equal, err_msg = compare_dictionaries(
            data.metadata, data2.metadata,
            'original_metadata', 'loaded_metadata')
        self.assertTrue(metadata_equal, msg='\n'+err_msg)

    def test_loop_writing(self):
        # pass
        station = Station()
        MockPar = MockParabola(name='Loop_writing_test')
        station.add_component(MockPar)
        # # added to station to test snapshot at a later stage
        loop = Loop(MockPar.x[-100:100:20]).each(MockPar.skewed_parabola)
        data1 = loop.run(name='MockLoop_hdf5_test',
                         formatter=self.formatter)
        data2 = DataSet(location=data1.location, formatter=self.formatter)
        data2.read()
        for key in data2.arrays.keys():
            self.checkArraysEqual(data2.arrays[key], data1.arrays[key])

        metadata_equal, err_msg = compare_dictionaries(
            data1.metadata, data2.metadata,
            'original_metadata', 'loaded_metadata')
        self.assertTrue(metadata_equal, msg='\n'+err_msg)
        self.formatter.close_file(data1)
        self.formatter.close_file(data2)
        MockPar.close()

    def test_partial_dataset(self):
        data = qcodes.data.data_set.new_data(formatter=self.formatter)
        data_array = qcodes.data.data_array.DataArray(array_id='test_partial_dataset', shape=(10,))
        data_array.init_data()
        data_array.ndarray[0] = 1
        data.add_array(data_array)
        data.write()
        data.read()

    def test_loop_writing_2D(self):
        # pass
        station = Station()
        MockPar = MockParabola(name='Loop_writing_test_2D')
        station.add_component(MockPar)
        loop = Loop(MockPar.x[-100:100:20]).loop(
            MockPar.y[-50:50:10]).each(MockPar.skewed_parabola)
        data1 = loop.run(name='MockLoop_hdf5_test',
                         formatter=self.formatter)
        data2 = DataSet(location=data1.location, formatter=self.formatter)
        data2.read()
        for key in data2.arrays.keys():
            self.checkArraysEqual(data2.arrays[key], data1.arrays[key])

        metadata_equal, err_msg = compare_dictionaries(
            data1.metadata, data2.metadata,
            'original_metadata', 'loaded_metadata')
        self.assertTrue(metadata_equal, msg='\n'+err_msg)
        self.formatter.close_file(data1)
        self.formatter.close_file(data2)
        MockPar.close()

    def test_closed_file(self):
        data = DataSet1D(location=self.loc_provider, name='test_closed')
        # closing before file is written should not raise error
        self.formatter.close_file(data)
        self.formatter.write(data)
        # Used because the formatter has no nice find file method
        self.formatter.close_file(data)
        # Closing file twice should not raise an error
        self.formatter.close_file(data)

    def test_reading_into_existing_data_array(self):
        data = DataSet1D(location=self.loc_provider,
                         name='test_read_existing')
        # closing before file is written should not raise error
        self.formatter.write(data)

        data2 = DataSet(location=data.location,
                        formatter=self.formatter)
        d_array = DataArray(name='dummy',
                            array_id='x_set',  # existing array id in data
                            label='bla', unit='a.u.', is_setpoint=False,
                            set_arrays=(), preset_data=np.zeros(5))
        data2.add_array(d_array)
        # test if d_array refers to same as array x_set in dataset
        self.assertTrue(d_array is data2.arrays['x_set'])
        data2.read()
        # test if reading did not overwrite dataarray
        self.assertTrue(d_array is data2.arrays['x_set'])
        # Testing if data was correctly updated into dataset
        self.checkArraysEqual(data2.arrays['x_set'], data.arrays['x_set'])
        self.checkArraysEqual(data2.arrays['y'], data.arrays['y'])
        self.formatter.close_file(data)
        self.formatter.close_file(data2)

    def test_dataset_closing(self):
        data = DataSet1D(location=self.loc_provider, name='test_closing')
        self.formatter.write(data, flush=False)
        fp = data._h5_base_group.filename
        self.formatter.close_file(data)
        fp3 = fp[:-5]+'_3.hdf5'
        copy(fp, fp3)
        # Should not raise an error because the file was properly closed
        F3 = h5py.File(fp3, mode='a')

    def test_dataset_flush_after_write(self):
        data = DataSet1D(name='test_flush', location=self.loc_provider)
        self.formatter.write(data, flush=True)
        fp = data._h5_base_group.filename
        fp2 = fp[:-5]+'_2.hdf5'
        # the file cannot be copied unless the ref is deleted first
        del data._h5_base_group
        copy(fp, fp2)
        # Opening this copy should not raise an error
        F2 = h5py.File(fp2, mode='a')

    def test_dataset_finalize_closes_file(self):
        data = DataSet1D(name='test_finalize', location=self.loc_provider)
        # closing before file is written should not raise error
        self.formatter.write(data, flush=False)
        fp = data._h5_base_group.filename

        # Attaching the formatter like this should not be neccesary
        data.formatter = self.formatter
        data.finalize()
        # the file cannot be copied unless the ref is deleted first
        del data._h5_base_group
        fp3 = fp[:-5]+'_4.hdf5'
        copy(fp, fp3)
        # Should now not raise an error because the file was properly closed
        F3 = h5py.File(fp3, mode='a')

    def test_double_closing_gives_warning(self):
        data = DataSet1D(name='test_double_close',
                         location=self.loc_provider)
        # closing before file is written should not raise error
        self.formatter.write(data, flush=False)
        self.formatter.close_file(data)
        with self.assertLogs():
            # Test that this raises a logging message
            self.formatter.close_file(data)

    def test_dataset_with_missing_attrs(self):
        data1 = new_data(formatter=self.formatter, location=self.loc_provider,
                         name='test_missing_attr')
        arr = DataArray(array_id='arr', preset_data=np.linspace(0, 10, 21))
        data1.add_array(arr)
        data1.write()
        # data2 = DataSet(location=data1.location, formatter=self.formatter)
        # data2.read()
        data2 = load_data(location=data1.location,
                          formatter=self.formatter)
        # cannot use the check arrays equal as I expect the attributes
        # to not be equal
        np.testing.assert_array_equal(data2.arrays['arr'], data1.arrays['arr'])

    def test_read_writing_dicts_withlists_to_hdf5(self):
        some_dict = {}
        some_dict['list_of_ints'] = list(np.arange(5))
        some_dict['list_of_floats'] = list(np.arange(5.1))
        some_dict['list_of_mixed_type'] = list([1, '1'])
        fp = self.loc_provider(
            io=DataSet.default_io,
            record={'name': 'test_dict_writing'})+'.hdf5'
        F = h5py.File(fp, mode='a')

        self.formatter.write_dict_to_hdf5(some_dict, F)
        new_dict = {}
        self.formatter.read_dict_from_hdf5(new_dict, F)
        dicts_equal, err_msg = compare_dictionaries(
            some_dict, new_dict,
            'written_dict', 'loaded_dict')
        self.assertTrue(dicts_equal, msg='\n'+err_msg)

    def test_str_to_bool(self):
        self.assertEqual(str_to_bool('True'), True)
        self.assertEqual(str_to_bool('False'), False)
        with self.assertRaises(ValueError):
            str_to_bool('flse')

    def test_writing_unsupported_types_to_hdf5(self):
        """
        Tests writing of
            - unsuported list type attr
            - nested dataset
        """
        some_dict = {}
        some_dict['list_of_ints'] = list(np.arange(5))
        some_dict['list_of_floats'] = list(np.arange(5.1))
        some_dict['weird_dict'] = {'a': 5}
        data1 = new_data(formatter=self.formatter, location=self.loc_provider,
                         name='test_missing_attr')
        some_dict['nested_dataset'] = data1

        some_dict['list_of_dataset'] = [data1, data1]

        fp = self.loc_provider(
            io=DataSet.default_io,
            record={'name': 'test_dict_writing'})+'.hdf5'
        F = h5py.File(fp, mode='a')
        self.formatter.write_dict_to_hdf5(some_dict, F)
        new_dict = {}
        self.formatter.read_dict_from_hdf5(new_dict, F)
        # objects are not identical but the string representation should be
        self.assertEqual(str(some_dict['nested_dataset']),
                         new_dict['nested_dataset'])
        self.assertEqual(str(some_dict['list_of_dataset']),
                         new_dict['list_of_dataset'])

        F['weird_dict'].attrs['list_type'] = 'unsuported_list_type'
        with self.assertRaises(NotImplementedError):
            self.formatter.read_dict_from_hdf5(new_dict, F)

    def test_writing_metadata(self):
        # test for issue reported in 442
        data = DataSet2D(location=self.loc_provider, name='MetaDataTest')
        data.metadata = {'a': ['hi', 'there']}
        self.formatter.write(data, write_metadata=True)
