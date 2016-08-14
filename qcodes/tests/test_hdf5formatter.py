from unittest import TestCase
import sys
import os
import numpy as np
import h5py  # TODO: add this to the dependencies in setup.py
from qcodes.data.location import FormatLocation
from qcodes.data.hdf5_format import HDF5Format

from qcodes.data.data_array import DataArray
from qcodes.data.data_set import DataSet, new_data, load_data
from qcodes.utils.helpers import LogCapture
from .data_mocks import DataSet1D, file_1d, DataSetCombined, files_combined

from qcodes.tests.instrument_mocks import MockParabola


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

    def tearDown(self):
        pass
        # for location in self.locations:
        #     self.io.remove_all(location)

    def checkArraysEqual(self, a, b):
        """
        Checks if arrays are equal
        """
        # Copied from GNUplot formatter tests inheritance would be nicer
        self.checkArrayAttrs(a, b)
        self.assertTrue((a.ndarray==b.ndarray).all())
        print(a.set_arrays, b.set_arrays)
        self.assertEqual(len(a.set_arrays), len(b.set_arrays))
        for sa, sb in zip(a.set_arrays, b.set_arrays):
            self.checkArrayAttrs(sa, sb)

    def checkArrayAttrs(self, a, b):
        print('attributes')
        self.assertEqual(a.tolist(), b.tolist())
        print(a.label, b.label)
        self.assertEqual(a.label, b.label)
        self.assertEqual(a.array_id, b.array_id)

    def test_full_write_read_1D(self):
        """
        Test writing and reading a file back in
        """
        # location = self.locations[0]
        data = DataSet1D()
        # print('Data location:', os.path.abspath(data.location))
        self.formatter.write(data)
        # Used because the formatter has no nice find file method
        filepath = self.formatter.filepath

        # Test reading the same file through the DataSet.read

        data2 = DataSet(location=filepath, formatter=self.formatter)
        data2.read()
        self.checkArraysEqual(data2.x_set, data.x_set)
        self.checkArraysEqual(data2.y, data.y)

    # def test_read_write_missing_dset_attrs(self):
    #     '''
    #     If some attributes are missing it should still write correctly
    #     '''
    #     raise(NotImplementedError)
    #     print('NotImplemented')

    # def test_no_nest(self):
    #     pass
        # formatter = GNUPlotFormat(always_nest=False)
        # location = self.locations[0]
        # data = DataSet1D(location)

        # # mark the data set as modified by... modifying it!
        # # without actually changing it :)
        # # TODO - are there cases we should automatically mark the data as
        # # modified on construction?
        # data.y[4] = data.y[4]

        # formatter.write(data)

        # with open(location + '.dat', 'r') as f:
        #     self.assertEqual(f.read(), file_1d())

    # def test_incremental_write(self):
    #     print('commented out until fixed')
        # location = self.locations[0]
        # data = DataSet1D(location)
        # data_copy = DataSet1D(False)

        # # # empty the data and mark it as unmodified
        # data.x[:] = float('nan')
        # data.y[:] = float('nan')
        # data.x.modified_range = None
        # data.y.modified_range = None

        # # Comment copied form GNUPlotFormat tests
        # # simulate writing after every value comes in, even within
        # # one row (x comes first, it's the setpoint)
        # for i, (x, y) in enumerate(zip(data_copy.x, data_copy.y)):
        #     data.x[i] = x
        #     self.formatter.write(data)
        #     # should not update here as not a full row has come in
        #     # TODO: implement this in the data formatter
        #     data.y[i] = y
        #     self.formatter.write(data)

        # filepath = self.formatter.filepath
        # data2 = DataSet(location=filepath, formatter=self.formatter)
        # data2.read()
        # self.checkArraysEqual(data2.x, data_copy.x)
        # self.checkArraysEqual(data2.y, data_copy.y)


    # def test_loop_writing(self):
    #     print('Loop writing not implemented DEBUG PRINT REMOVE BEFORE MERGE')
        # station = Station()
        # MockPar = MockParabola(name='MockParabola')
        # station.add_component(MockPar)
        # # added to station to test snapshot at a later stage
        # loop = Loop(MockPar.x[-100:100:20]).each(MockPar.skewed_parabola)
        # dset = loop.run(name='MockParabola_run', formatter=self.formatter)

        # dset.write()
        # skew_para = np.array([ 1010000., 518400., 219600., 65600.,
        #                      8400., 0., 8400., 65600., 219600., 518400.])
        # x = np.arange(-100, 100, 20)
        # print(dset.sync())
        # print(dset.arrays)
        # fp = dset.formatter.filepath
        # loaded_data = load_data(fp, formatter=self.formatter)
        # arrs = load_data.arrays
        # self.assertTrue((arrs['x'].ndarray == x).all())
        # self.assertTrue((arrs['skewed_parabola'].ndarray == skew_para).all())



    # def test_multifile(self):
        # pass
        # formatter = GNUPlotFormat(always_nest=False)  # will nest anyway
        # location = self.locations[1]
        # data = DataSetCombined(location)

        # # mark one array in each file as completely modified
        # # that should cause the whole files to be written, even though
        # # the other data and setpoint arrays are not marked as modified
        # data.y1[:] += 0
        # data.z1[:, :] += 0
        # formatter.write(data)

        # filex, filexy = files_combined()

        # with open(location + '/x.dat', 'r') as f:
        #     self.assertEqual(f.read(), filex)
        # with open(location + '/x_yset.dat', 'r') as f:
        #     self.assertEqual(f.read(), filexy)

        # data2 = DataSet(location=location)
        # formatter.read(data2)

        # for array_id in ('x', 'y1', 'y2', 'yset', 'z1', 'z2'):
        #     self.checkArraysEqual(data2.arrays[array_id],
        #                           data.arrays[array_id])

