from unittest import TestCase
import numpy as np
import os
import pickle
import logging

from qcodes.data.data_array import DataArray
from qcodes.data.io import DiskIO
from qcodes.data.data_set import load_data, new_data, DataSet
from qcodes.utils.helpers import LogCapture

from .data_mocks import (MockFormatter, MatchIO,
                         DataSet2D, DataSet1D,
                         DataSetCombined, RecordingMockFormatter)

from .common import strip_qc


class TestDataArray(TestCase):

    def test_attributes(self):
        pname = 'Betty Sue'
        plabel = 'The best apple pie this side of Wenatchee'
        pfullname = 'bert'

        class MockParam:
            name = pname
            label = plabel

            def __init__(self, full_name=None):
                self.full_name = full_name

        name = 'Oscar'
        label = 'The grouch. GRR!'
        fullname = 'ernie'
        array_id = 24601
        set_arrays = ('awesomeness', 'chocolate content')
        shape = 'Ginornous'
        action_indices = (1, 2, 3, 4, 5)

        p_data = DataArray(parameter=MockParam(pfullname), name=name,
                           label=label, full_name=fullname)
        p_data2 = DataArray(parameter=MockParam(pfullname))

        # explicitly given name and label override parameter vals
        self.assertEqual(p_data.name, name)
        self.assertEqual(p_data.label, label)
        self.assertEqual(p_data.full_name, fullname)
        self.assertEqual(p_data2.name, pname)
        self.assertEqual(p_data2.label, plabel)
        self.assertEqual(p_data2.full_name, pfullname)
        # test default values
        self.assertIsNone(p_data.array_id)
        self.assertEqual(p_data.shape, ())
        self.assertEqual(p_data.action_indices, ())
        self.assertEqual(p_data.set_arrays, ())
        self.assertIsNone(p_data.ndarray)

        np_data = DataArray(name=name, label=label, array_id=array_id,
                            set_arrays=set_arrays, shape=shape,
                            action_indices=action_indices)
        self.assertEqual(np_data.name, name)
        self.assertEqual(np_data.label, label)
        # no full name or parameter - use name
        self.assertEqual(np_data.full_name, name)
        # test simple assignments
        self.assertEqual(np_data.array_id, array_id)
        self.assertEqual(np_data.set_arrays, set_arrays)
        self.assertEqual(np_data.shape, shape)
        self.assertEqual(np_data.action_indices, action_indices)

        name_data = DataArray(name=name)
        self.assertEqual(name_data.label, name)

        blank_data = DataArray()
        self.assertIsNone(blank_data.name)

    def test_preset_data(self):
        onetwothree = [
            # lists and tuples work
            [1.0, 2.0, 3.0],
            (1.0, 2.0, 3.0),

            # iterators get automatically cast to floats
            (i + 1 for i in range(3)),
            map(float, range(1, 4)),

            # and of course numpy arrays themselves work
            np.array([1.0, 2.0, 3.0]),
        ]

        expected123 = [1.0, 2.0, 3.0]

        for item in onetwothree:
            data = DataArray(preset_data=item)
            self.assertEqual(data.ndarray.tolist(), expected123)
            self.assertEqual(data.shape, (3, ))

        # you can re-initialize a DataArray with the same shape data,
        # but not with a different shape
        list456 = [4, 5, 6]
        data.init_data(data=list456)
        self.assertEqual(data.ndarray.tolist(), list456)
        with self.assertRaises(ValueError):
            data.init_data([1, 2])
        self.assertEqual(data.ndarray.tolist(), list456)
        self.assertEqual(data.shape, (3, ))

        # you can call init_data again with no data, and nothing changes
        data.init_data()
        self.assertEqual(data.ndarray.tolist(), list456)
        self.assertEqual(data.shape, (3, ))

        # multidimensional works too
        list2d = [[1, 2], [3, 4]]
        data2 = DataArray(preset_data=list2d)
        self.assertEqual(data2.ndarray.tolist(), list2d)
        self.assertEqual(data2.shape, (2, 2))

    def test_init_data_error(self):
        data = DataArray(preset_data=[1, 2])
        data.shape = (3, )

        # not sure when this would happen... but if you call init_data
        # and it notices an inconsistency between shape and the actual
        # data that's already there, it raises an error
        with self.assertRaises(ValueError):
            data.init_data()

    def test_clear(self):
        nan = float('nan')
        data = DataArray(preset_data=[1, 2])
        data.clear()
        # sometimes it's annoying that nan != nan
        self.assertEqual(repr(data.ndarray.tolist()), repr([nan, nan]))

    def test_edit_and_mark(self):
        data = DataArray(preset_data=[[1, 2], [3, 4]])
        self.assertEqual(data[0].tolist(), [1, 2])
        self.assertEqual(data[0, 1], 2)

        data.modified_range = None
        self.assertIsNone(data.last_saved_index)

        self.assertEqual(len(data), 2)
        data[0] = np.array([5, 6])
        data[1, 0] = 7
        self.assertEqual(data.ndarray.tolist(), [[5, 6], [7, 4]])

        self.assertEqual(data.modified_range, (0, 2))

        # as if we saved the first two points... the third should still
        # show as modified
        data.mark_saved(1)
        self.assertEqual(data.last_saved_index, 1)
        self.assertEqual(data.modified_range, (2, 2))

        # now we save the third point... no modifications left.
        data.mark_saved(2)
        self.assertEqual(data.last_saved_index, 2)
        self.assertEqual(data.modified_range, None)

        data.clear_save()
        self.assertEqual(data.last_saved_index, None)
        self.assertEqual(data.modified_range, (0, 2))

    def test_edit_and_mark_slice(self):
        data = DataArray(preset_data=[[1] * 5] * 6)

        self.assertEqual(data.shape, (6, 5))
        data.modified_range = None

        data[:4:2, 2:] = 2
        self.assertEqual(data.tolist(), [
            [1, 1, 2, 2, 2],
            [1, 1, 1, 1, 1],
            [1, 1, 2, 2, 2],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ])
        self.assertEqual(data.modified_range, (2, 14))

    def test_repr(self):
        array2d = [[1, 2], [3, 4]]
        arrayrepr = repr(np.array(array2d))
        array_id = (3, 4)
        data = DataArray(preset_data=array2d)

        self.assertEqual(repr(data), 'DataArray[2,2]:\n' + arrayrepr)

        data.array_id = array_id
        self.assertEqual(repr(data), 'DataArray[2,2]: ' + str(array_id) +
                         '\n' + arrayrepr)

    def test_nest_empty(self):
        data = DataArray()

        self.assertEqual(data.shape, ())

        mock_set_array = 'not really an array but we don\'t check'
        mock_set_array2 = 'another one'

        data.nest(2, action_index=44, set_array=mock_set_array)
        data.nest(3, action_index=66, set_array=mock_set_array2)

        # the array doesn't exist until you initialize it
        self.assertIsNone(data.ndarray)

        # but other attributes are set
        self.assertEqual(data.shape, (3, 2))
        self.assertEqual(data.action_indices, (66, 44))
        self.assertEqual(data.set_arrays, (mock_set_array2, mock_set_array))

        data.init_data()
        self.assertEqual(data.ndarray.shape, (3, 2))

        # after initializing data, you can't nest anymore because this isn't
        # a preset array
        with self.assertRaises(RuntimeError):
            data.nest(4)

    def test_nest_preset(self):
        data = DataArray(preset_data=[1, 2])
        data.nest(3)
        self.assertEqual(data.shape, (3, 2))
        self.assertEqual(data.ndarray.tolist(), [[1, 2]] * 3)
        self.assertEqual(data.action_indices, ())
        self.assertEqual(data.set_arrays, (data,))

        # test that the modified range gets correctly set to
        # (0, 2*3-1 = 5)
        self.assertEqual(data.modified_range, (0, 5))

        # you need a set array for all but the inner nesting
        with self.assertRaises(TypeError):
            data.nest(4)

    def test_data_set_property(self):
        data = DataArray(preset_data=[1, 2])
        self.assertIsNone(data.data_set)

        mock_data_set = 'pretend this is a DataSet, we don\'t check type'
        mock_data_set2 = 'you can only assign to another after first clearing'
        data.data_set = mock_data_set
        self.assertEqual(data.data_set, mock_data_set)

        with self.assertRaises(RuntimeError):
            data.data_set = mock_data_set2

        data.data_set = None
        self.assertIsNone(data.data_set)
        data.data_set = mock_data_set2
        self.assertEqual(data.data_set, mock_data_set2)

    def test_fraction_complete(self):
        data = DataArray(shape=(5, 10))
        self.assertIsNone(data.ndarray)
        self.assertEqual(data.fraction_complete(), 0.0)

        data.init_data()
        self.assertEqual(data.fraction_complete(), 0.0)

        # index = 1 * 10 + 7 - add 1 (for index 0) and you get 18
        # each index is 2% of the total, so this is 36%
        data[1, 7] = 1
        self.assertEqual(data.fraction_complete(), 18 / 50)

        # add a last_saved_index but modified_range is still bigger
        data.mark_saved(13)
        self.assertEqual(data.fraction_complete(), 18 / 50)

        # now last_saved_index wins
        data.mark_saved(19)
        self.assertEqual(data.fraction_complete(), 20 / 50)

        # now pretend we get more info from syncing
        data.synced_index = 22
        self.assertEqual(data.fraction_complete(), 23 / 50)


class TestLoadData(TestCase):

    def test_no_saved_data(self):
        with self.assertRaises(IOError):
            load_data('_no/such/file_')

    def test_load_false(self):
        with self.assertRaises(ValueError):
            load_data(False)

    def test_get_read(self):
        data = load_data(formatter=MockFormatter(), location='here!')
        self.assertEqual(data.has_read_data, True)
        self.assertEqual(data.has_read_metadata, True)


class TestDataSetMetaData(TestCase):

    def test_snapshot(self):
        data = new_data(location=False)
        expected_snap = {
            '__class__': 'qcodes.data.data_set.DataSet',
            'location': False,
            'arrays': {},
            'formatter': 'qcodes.data.gnuplot_format.GNUPlotFormat',
        }
        snap = strip_qc(data.snapshot())

        # handle io separately so we don't need to figure out our path
        self.assertIn('DiskIO', snap['io'])
        del snap['io']
        self.assertEqual(snap, expected_snap)

        # even though we removed io from the snapshot, it's still in .metadata
        self.assertIn('io', data.metadata)

        # then do the same transformations to metadata to check it too
        del data.metadata['io']
        strip_qc(data.metadata)
        self.assertEqual(data.metadata, expected_snap)

        # location is False so read_metadata should be a noop
        data.metadata = {'food': 'Fried chicken'}
        data.read_metadata()
        self.assertEqual(data.metadata, {'food': 'Fried chicken'})

        # snapshot should never delete things from metadata, only add or update
        data.metadata['location'] = 'Idaho'
        snap = strip_qc(data.snapshot())
        expected_snap['food'] = 'Fried chicken'
        del snap['io']
        self.assertEqual(snap, expected_snap)


class TestNewData(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.original_lp = DataSet.location_provider

    @classmethod
    def tearDownClass(cls):
        DataSet.location_provider = cls.original_lp

    def test_overwrite(self):
        io = MatchIO([1])

        with self.assertRaises(FileExistsError):
            new_data(location='somewhere', io=io)

        data = new_data(location='somewhere', io=io, overwrite=True,)
        self.assertEqual(data.location, 'somewhere')

    def test_location_functions(self):
        def my_location(io, record):
            return 'data/{}'.format((record or {}).get('name') or 'LOOP!')

        def my_location2(io, record):
            name = (record or {}).get('name') or 'loop?'
            return 'data/{}/folder'.format(name)

        DataSet.location_provider = my_location

        self.assertEqual(new_data().location, 'data/LOOP!')
        self.assertEqual(new_data(name='cheese').location, 'data/cheese')

        data = new_data(location=my_location2)
        self.assertEqual(data.location, 'data/loop?/folder')
        data = new_data(location=my_location2, name='iceCream')
        self.assertEqual(data.location, 'data/iceCream/folder')


class TestDataSet(TestCase):

    def test_constructor_errors(self):
        # no location - only allowed with load_data
        with self.assertRaises(ValueError):
            DataSet()
        # wrong type
        with self.assertRaises(ValueError):
            DataSet(location=42)

    def test_write_copy(self):
        data = DataSet1D(location=False)
        mockbase = os.path.abspath('some_folder')
        data.io = DiskIO(mockbase)

        mr = (2, 3)
        mr_full = (0, 4)
        lsi = 1
        data.x_set.modified_range = mr
        data.y.modified_range = mr
        data.x_set.last_saved_index = lsi
        data.y.last_saved_index = lsi

        with self.assertRaises(TypeError):
            data.write_copy()

        with self.assertRaises(TypeError):
            data.write_copy(path='some/path', io_manager=DiskIO('.'))

        with self.assertRaises(TypeError):
            data.write_copy(path='some/path', location='something/else')

        data.formatter = RecordingMockFormatter()
        data.write_copy(path='/some/abs/path')
        self.assertEqual(data.formatter.write_calls,
                         [(None, '/some/abs/path')])
        self.assertEqual(data.formatter.write_metadata_calls,
                         [(None, '/some/abs/path', False)])
        # check that the formatter gets called as if nothing has been saved
        self.assertEqual(data.formatter.modified_ranges,
                         [{'x_set': mr_full, 'y': mr_full}])
        self.assertEqual(data.formatter.last_saved_indices,
                         [{'x_set': None, 'y': None}])
        # but the dataset afterward has its original mods back
        self.assertEqual(data.x_set.modified_range, mr)
        self.assertEqual(data.y.modified_range, mr)
        self.assertEqual(data.x_set.last_saved_index, lsi)
        self.assertEqual(data.y.last_saved_index, lsi)

        # recreate the formatter to clear the calls attributes
        data.formatter = RecordingMockFormatter()
        data.write_copy(location='some/rel/path')
        self.assertEqual(data.formatter.write_calls,
                         [(mockbase, 'some/rel/path')])
        self.assertEqual(data.formatter.write_metadata_calls,
                         [(mockbase, 'some/rel/path', False)])

        mockbase2 = os.path.abspath('some/other/folder')
        io2 = DiskIO(mockbase2)

        with self.assertRaises(ValueError):
            # if location=False we need to specify it in write_copy
            data.write_copy(io_manager=io2)

        data.location = 'yet/another/path'
        data.formatter = RecordingMockFormatter()
        data.write_copy(io_manager=io2)
        self.assertEqual(data.formatter.write_calls,
                         [(mockbase2, 'yet/another/path')])
        self.assertEqual(data.formatter.write_metadata_calls,
                         [(mockbase2, 'yet/another/path', False)])

    def test_pickle_dataset(self):
        # Test pickling of DataSet object
        # If the data_manager is set to None, then the object should pickle.
        m = DataSet2D()
        pickle.dumps(m)

    def test_default_parameter(self):
        # Test whether the default_array function works
        m = DataSet2D()

        # test we can run with default arguments
        name = m.default_parameter_name()

        # test with paramname
        name = m.default_parameter_name(paramname='z')
        self.assertEqual(name, 'z')
        # test we can get the array instead of the name
        array = m.default_parameter_array(paramname='z')
        self.assertEqual(array, m.z)

        # first non-setpoint array
        array = m.default_parameter_array()
        self.assertEqual(array, m.z)

        # test with metadata
        m.metadata = dict({'default_parameter_name': 'x_set'})
        name = m.default_parameter_name()
        self.assertEqual(name, 'x_set')

        # test the fallback: no name matches, no non-setpoint array
        x = DataArray(name='x', label='X', preset_data=(
            1., 2., 3., 4., 5.), is_setpoint=True)
        m = new_data(arrays=(x,), name='onlysetpoint')
        name = m.default_parameter_name(paramname='dummy')
        self.assertEqual(name, 'x_set')

    def test_fraction_complete(self):
        empty_data = new_data(arrays=(), location=False)
        self.assertEqual(empty_data.fraction_complete(), 0.0)

        data = DataSetCombined(location=False)
        self.assertEqual(data.fraction_complete(), 1.0)

        # alter only the measured arrays, check that only these are used
        # to calculate fraction_complete
        data.y1.modified_range = (0, 0)  # 1 of 2
        data.y2.modified_range = (0, 0)  # 1 of 2
        data.z1.modified_range = (0, 2)  # 3 of 6
        data.z2.modified_range = (0, 2)  # 3 of 6
        self.assertEqual(data.fraction_complete(), 0.5)

        # mark more things complete using last_saved_index and synced_index
        data.y1.last_saved_index = 1  # 2 of 2
        data.z1.synced_index = 5  # 6 of 6
        self.assertEqual(data.fraction_complete(), 0.75)

    def mock_sync(self):
        i = self.sync_index
        self.syncing_array[i] = i
        self.sync_index = i + 1
        return self.sync_index < self.syncing_array.size

    def failing_func(self):
        raise RuntimeError('it is called failing_func for a reason!')

    def logging_func(self):
        logging.info('background at index {}'.format(self.sync_index))

    def test_complete(self):
        array = DataArray(name='y', shape=(5,))
        array.init_data()
        data = new_data(arrays=(array,), location=False)
        self.syncing_array = array
        self.sync_index = 0
        data.sync = self.mock_sync
        bf = DataSet.background_functions
        bf['fail'] = self.failing_func
        bf['log'] = self.logging_func

        with LogCapture() as logs:
            # grab info and warnings but not debug messages
            logging.getLogger().setLevel(logging.INFO)
            data.complete(delay=0.001)

        logs = logs.value

        expected_logs = [
            'waiting for DataSet <False> to complete',
            'DataSet: 0% complete',
            'RuntimeError: it is called failing_func for a reason!',
            'background at index 1',
            'DataSet: 20% complete',
            'RuntimeError: it is called failing_func for a reason!',
            'background function fail failed twice in a row, removing it',
            'background at index 2',
            'DataSet: 40% complete',
            'background at index 3',
            'DataSet: 60% complete',
            'background at index 4',
            'DataSet: 80% complete',
            'background at index 5',
            'DataSet <False> is complete'
        ]

        log_index = 0
        for line in expected_logs:
            self.assertIn(line, logs, logs)
            try:
                log_index_new = logs.index(line, log_index)
            except ValueError:
                raise ValueError('line {} not found after {} in: \n {}'.format(
                    line, log_index, logs))
            self.assertTrue(log_index_new >= log_index, logs)
            log_index = log_index_new + len(line) + 1  # +1 for \n
        self.assertEqual(log_index, len(logs), logs)

    def test_remove_array(self):
        m = DataSet2D()
        m.remove_array('z')
        _ = m.__repr__()
        self.assertFalse('z' in m.arrays)
