from unittest import TestCase
from unittest.mock import patch
import numpy as np
import pickle

from qcodes.data.data_array import DataArray
from qcodes.data.manager import get_data_manager, NoData
from qcodes.data.data_set import load_data, new_data, DataMode, DataSet
from qcodes.process.helpers import kill_processes
from qcodes import active_children

from .data_mocks import (MockDataManager, MockFormatter, MatchIO,
                         MockLive, MockArray, DataSet2D)
from .common import strip_qc


class TestDataArray(TestCase):

    def test_attributes(self):
        pname = 'Betty Sue'
        plabel = 'The best apple pie this side of Wenatchee'

        class MockParam:
            name = pname
            label = plabel

        name = 'Oscar'
        label = 'The grouch. GRR!'
        array_id = 24601
        set_arrays = ('awesomeness', 'chocolate content')
        shape = 'Ginornous'
        action_indices = (1, 2, 3, 4, 5)

        p_data = DataArray(parameter=MockParam(), name=name, label=label)
        p_data2 = DataArray(parameter=MockParam())

        # explicitly given name and label override parameter vals
        self.assertEqual(p_data.name, name)
        self.assertEqual(p_data.label, label)
        self.assertEqual(p_data2.name, pname)
        self.assertEqual(p_data2.label, plabel)
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

        self.assertIsNone(data.modified_range)
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
        self.assertEqual(data.modified_range, None)

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


class TestLoadData(TestCase):

    def setUp(self):
        kill_processes()

    def test_no_live_data(self):
        # live data with no DataManager at all
        with self.assertRaises(RuntimeError):
            load_data()
        self.assertEqual(len(active_children()), 0)

        # now make a DataManager and try again
        get_data_manager()
        self.assertEqual(len(active_children()), 1)
        # same result but different code path
        with self.assertRaises(RuntimeError):
            load_data()

    def test_no_saved_data(self):
        with self.assertRaises(IOError):
            load_data('_no/such/file_')

    def test_load_false(self):
        with self.assertRaises(ValueError):
            load_data(False)

    def test_get_live(self):
        loc = 'live from New York!'

        class MockLive:
            pass

        live_data = MockLive()

        dm = MockDataManager()
        dm.location = loc
        dm.live_data = live_data

        data = load_data(data_manager=dm, location=loc)
        self.assertEqual(data, live_data)

        for nd in (None, NoData()):
            dm.live_data = nd
            with self.assertRaises(RuntimeError):
                load_data(data_manager=dm, location=loc)
            with self.assertRaises(RuntimeError):
                load_data(data_manager=dm)

    def test_get_read(self):
        dm = MockDataManager()
        dm.location = 'somewhere else'

        data = load_data(formatter=MockFormatter(), data_manager=dm,
                         location='here!')
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
        kill_processes()
        cls.original_lp = DataSet.location_provider

    @classmethod
    def tearDownClass(cls):
        DataSet.location_provider = cls.original_lp

    def test_overwrite(self):
        io = MatchIO([1])

        with self.assertRaises(FileExistsError):
            new_data(location='somewhere', io=io, data_manager=False)

        data = new_data(location='somewhere', io=io, overwrite=True,
                        data_manager=False)
        self.assertEqual(data.location, 'somewhere')

    def test_mode_error(self):
        with self.assertRaises(ValueError):
            new_data(mode=DataMode.PUSH_TO_SERVER, data_manager=False)

    def test_location_functions(self):
        def my_location(io, record):
            return 'data/{}'.format((record or {}).get('name') or 'LOOP!')

        def my_location2(io, record):
            name = (record or {}).get('name') or 'loop?'
            return 'data/{}/folder'.format(name)

        DataSet.location_provider = my_location

        self.assertEqual(new_data(data_manager=False).location, 'data/LOOP!')
        self.assertEqual(new_data(data_manager=False, name='cheese').location,
                         'data/cheese')

        data = new_data(data_manager=False, location=my_location2)
        self.assertEqual(data.location, 'data/loop?/folder')
        data = new_data(data_manager=False, location=my_location2,
                        name='iceCream')
        self.assertEqual(data.location, 'data/iceCream/folder')


class TestDataSet(TestCase):

    def tearDown(self):
        kill_processes()

    def test_constructor_errors(self):
        # no location - only allowed with load_data
        with self.assertRaises(ValueError):
            DataSet()
        # wrong type
        with self.assertRaises(ValueError):
            DataSet(location=42)

        # OK to have location=False, but wrong mode
        with self.assertRaises(ValueError):
            DataSet(location=False, mode='happy')

    @patch('qcodes.data.data_set.get_data_manager')
    def test_from_server(self, gdm_mock):
        mock_dm = MockDataManager()
        gdm_mock.return_value = mock_dm
        mock_dm.location = 'Mars'
        mock_dm.live_data = MockLive()

        # wrong location or False location - converts to local
        data = DataSet(location='Jupiter', mode=DataMode.PULL_FROM_SERVER)
        self.assertEqual(data.mode, DataMode.LOCAL)

        data = DataSet(location=False, mode=DataMode.PULL_FROM_SERVER)
        self.assertEqual(data.mode, DataMode.LOCAL)

        # location matching server - stays in server mode
        data = DataSet(location='Mars', mode=DataMode.PULL_FROM_SERVER,
                       formatter=MockFormatter())
        self.assertEqual(data.mode, DataMode.PULL_FROM_SERVER)
        self.assertEqual(data.arrays, MockLive.arrays)

        # cannot write except in LOCAL mode
        with self.assertRaises(RuntimeError):
            data.write()

        # cannot finalize in PULL_FROM_SERVER mode
        with self.assertRaises(RuntimeError):
            data.finalize()

        # now test when the server says it's not there anymore
        mock_dm.location = 'Saturn'
        data.sync()
        self.assertEqual(data.mode, DataMode.LOCAL)
        self.assertEqual(data.has_read_data, True)

        # now it's LOCAL so we *can* write.
        data.write()
        self.assertEqual(data.has_written_data, True)

        # location=False: write, read and sync are noops.
        data.has_read_data = False
        data.has_written_data = False
        data.location = False
        data.write()
        data.read()
        data.sync()
        self.assertEqual(data.has_read_data, False)
        self.assertEqual(data.has_written_data, False)

    @patch('qcodes.data.data_set.get_data_manager')
    def test_to_server(self, gdm_mock):
        mock_dm = MockDataManager()
        mock_dm.needs_restart = True
        gdm_mock.return_value = mock_dm

        data = DataSet(location='Venus', mode=DataMode.PUSH_TO_SERVER)
        self.assertEqual(mock_dm.needs_restart, False, data)
        self.assertEqual(mock_dm.data_set, data)
        self.assertEqual(data.data_manager, mock_dm)
        self.assertEqual(data.mode, DataMode.PUSH_TO_SERVER)

        # cannot write except in LOCAL mode
        with self.assertRaises(RuntimeError):
            data.write()

        # now do what the DataServer does with this DataSet: init_on_server
        # fails until there is an array
        with self.assertRaises(RuntimeError):
            data.init_on_server()

        data.add_array(MockArray())
        data.init_on_server()
        self.assertEqual(data.noise.ready, True)

        # we can only add a given array_id once
        with self.assertRaises(ValueError):
            data.add_array(MockArray())

    def test_pickle_dataset(self):
        # Test pickling of DataSet object
        # If the data_manager is set to None, then the object should pickle.
        m = DataSet2D()
        _ = pickle.dumps(m)
