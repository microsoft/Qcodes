from unittest import TestCase
import numpy as np

from qcodes.data.data_array import DataArray


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
        size = 'Ginornous'
        action_indices = (1, 2, 3, 4, 5)

        p_data = DataArray(parameter=MockParam(), name=name, label=label)

        # parameter overrides explicitly given name and label
        self.assertEqual(p_data.name, pname)
        self.assertEqual(p_data.label, plabel)
        # test default values
        self.assertIsNone(p_data.array_id)
        self.assertEqual(p_data.size, ())
        self.assertEqual(p_data.action_indices, ())
        self.assertEqual(p_data.set_arrays, ())
        self.assertIsNone(p_data.ndarray)

        np_data = DataArray(name=name, label=label, array_id=array_id,
                            set_arrays=set_arrays, size=size,
                            action_indices=action_indices)
        self.assertEqual(np_data.name, name)
        self.assertEqual(np_data.label, label)
        # test simple assignments
        self.assertEqual(np_data.array_id, array_id)
        self.assertEqual(np_data.set_arrays, set_arrays)
        self.assertEqual(np_data.size, size)
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
            self.assertEqual(data.size, (3, ))

        # you can re-initialize a DataArray with the same shape data,
        # but not with a different shape
        list456 = [4, 5, 6]
        data.init_data(data=list456)
        self.assertEqual(data.ndarray.tolist(), list456)
        with self.assertRaises(ValueError):
            data.init_data([1, 2])
        self.assertEqual(data.ndarray.tolist(), list456)
        self.assertEqual(data.size, (3, ))

        # you can call init_data again with no data, and nothing changes
        data.init_data()
        self.assertEqual(data.ndarray.tolist(), list456)
        self.assertEqual(data.size, (3, ))

        # multidimensional works too
        list2d = [[1, 2], [3, 4]]
        data2 = DataArray(preset_data=list2d)
        self.assertEqual(data2.ndarray.tolist(), list2d)
        self.assertEqual(data2.size, (2, 2))

    def test_init_data_error(self):
        data = DataArray(preset_data=[1, 2])
        data.size = (3, )

        # not sure when this would happen... but if you call init_data
        # and it notices an inconsistency between size and the actual
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

        data.mark_saved()
        self.assertEqual(data.last_saved_index, 2)
        self.assertEqual(data.modified_range, None)

        data.clear_save()
        self.assertEqual(data.last_saved_index, None)
        self.assertEqual(data.modified_range, (0, 2))

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

        self.assertEqual(data.size, ())

        mock_set_array = 'not really an array but we don\'t check'
        mock_set_array2 = 'another one'

        data.nest(2, action_index=44, set_array=mock_set_array)
        data.nest(3, action_index=66, set_array=mock_set_array2)

        # the array doesn't exist until you initialize it
        self.assertIsNone(data.ndarray)

        # but other attributes are set
        self.assertEqual(data.size, (3, 2))
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
        self.assertEqual(data.size, (3, 2))
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
