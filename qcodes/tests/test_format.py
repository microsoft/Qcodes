from unittest import TestCase
import os

from qcodes.data.format import Formatter
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.data_set import DataMode, DataSet
from qcodes.utils.helpers import LogCapture

from .data_mocks import DataSet1D, DataSet2D, DataSetCombined


class TestBaseFormatter(TestCase):
    def setUp(self):
        self.io = DataSet.default_io
        self.locations = ('_simple1d_', '_simple2d_', '_combined_')

        for location in self.locations:
            self.assertFalse(self.io.list(location))

    def tearDown(self):
        for location in self.locations:
            self.io.remove_all(location)

    def test_overridable_methods(self):
        formatter = Formatter()
        data = DataSet1D()

        with self.assertRaises(NotImplementedError):
            formatter.write(data)
        with self.assertRaises(NotImplementedError):
            formatter.read_one_file(data, 'a file!', set())

    def test_no_files(self):
        formatter = Formatter()
        data = DataSet1D(self.locations[0])
        with self.assertRaises(IOError):
            formatter.read(data)

    def test_init_and_bad_read(self):
        location = self.locations[0]
        path = './{}/bad.dat'.format(location)

        class MyFormatter(Formatter):
            def read_one_file(self, data_set, f, ids_read):
                s = f.read()
                if 'garbage' not in s:
                    raise Exception('reading the wrong file?')

                # mark this file as read, before generating an error
                if not hasattr(data_set, 'files_read'):
                    data_set.files_read = []
                data_set.files_read.append(f.name)
                raise ValueError('garbage in, garbage out')

        formatter = MyFormatter()
        data = DataSet1D(location)
        data.x.ndarray = None
        data.y.ndarray = None

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write('garbage')

        with LogCapture() as s:
            formatter.read(data)
        logstr = s.getvalue()
        s.close()
        # we tried to read this file but it generated an error
        self.assertEqual(logstr.count('error reading file'), 1, logstr)
        self.assertEqual(data.files_read, [os.path.abspath(path)])

        expected_array_repr = repr([float('nan')] * 5)
        self.assertEqual(repr(data.x.tolist()), expected_array_repr)
        self.assertEqual(repr(data.y.tolist()), expected_array_repr)

    def test_group_arrays(self):
        formatter = Formatter()
        data = DataSetCombined()

        groups = formatter.group_arrays(data.arrays)

        self.assertEqual(len(groups), 2, groups)
        groups.sort(key=lambda grp: len(grp.set_arrays))

        g1d, g2d = groups

        self.assertEqual(g1d.size, (2,))
        self.assertEqual(g1d.set_arrays, (data.x,))
        self.assertEqual(g1d.data, (data.y1, data.y2))
        self.assertEqual(g1d.name, 'x')

        self.assertEqual(g2d.size, (2, 3))
        self.assertEqual(g2d.set_arrays, (data.x, data.yset))
        self.assertEqual(g2d.data, (data.z1, data.z2))
        self.assertEqual(g2d.name, 'x_yset')

    def test_match_save_range(self):
        formatter = Formatter()
        data = DataSet1D()

        group = formatter.group_arrays(data.arrays)[0]

        # no matter what else, if nothing is listed as modified
        # then save_range is None
        for lsi_x in [None, 0, 3]:
            data.x.last_saved_index = lsi_x
            for lsi_y in [None, 1, 4]:
                data.y.last_saved_index = lsi_y
                for fe in [True, False]:
                    save_range = formatter.match_save_range(
                        group, file_exists=fe)
                    self.assertEqual(save_range, None)

        # consistent last_saved_index: if it's None or within the
        # modified range, or if file does not exist, we need to overwrite
        # otherwise start just after last_saved_index
        for lsi, start in [(None, 0), (0, 1), (1, 2), (2, 3), (3, 0), (4, 0)]:
            data.x.last_saved_index = data.y.last_saved_index = lsi

            # inconsistent modified_range: expands to greatest extent
            # so these situations are identical
            for xmr, ymr in ([(4, 4), (3, 3)], [(3, 4), None], [None, (3, 4)]):
                data.x.modified_range = xmr
                data.y.modified_range = ymr

                save_range = formatter.match_save_range(group,
                                                        file_exists=False)
                self.assertEqual(save_range, (0, 4))

                save_range = formatter.match_save_range(group,
                                                        file_exists=True)
                self.assertEqual(save_range, (start, 4))

        # inconsistent last_saved_index: need to overwrite no matter what
        data.x.last_saved_index = 1
        data.y.last_saved_index = 2
        save_range = formatter.match_save_range(group, file_exists=True)
        self.assertEqual(save_range, (0, 4))

        # missing data point: don't write it unless only_complete is False
        # but this will only back up one point!
        data.y[4] = float('nan')
        data.y[3] = float('nan')
        data.x.last_saved_index = data.y.last_saved_index = 2

        save_range = formatter.match_save_range(group, file_exists=True)
        self.assertEqual(save_range, (3, 3))

        save_range = formatter.match_save_range(group, file_exists=True,
                                                only_complete=False)
        self.assertEqual(save_range, (3, 4))


class TestGNUPlotFormat(TestCase):
    simplefile = '\n'.join([
        '# x\ty',
        '# "X value"\t"Y value"',
        '# 5',
        '1\t3',
        '2\t4',
        '3\t5',
        '4\t6',
        '5\t7', ''])

    def setUp(self):
        self.io = DataSet.default_io
        self.locations = ('_simple1d_', '_simple2d_', '_combined_')

        for location in self.locations:
            self.assertFalse(self.io.list(location))

    def tearDown(self):
        for location in self.locations:
            self.io.remove_all(location)

    def checkArraysEqual(self, a, b):
        self.checkArrayAttrs(a, b)

        self.assertEqual(len(a.set_arrays), len(b.set_arrays))
        for sa, sb in zip(a.set_arrays, b.set_arrays):
            self.checkArrayAttrs(sa, sb)

    def checkArrayAttrs(self, a, b):
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(a.label, b.label)
        self.assertEqual(a.array_id, b.array_id)

    def test_simple(self):
        formatter = GNUPlotFormat()
        location = self.locations[0]
        data = DataSet1D(location)

        # mark the data set as modified by... modifying it!
        # without actually changing it :)
        # TODO - are there cases we should automatically mark the data as
        # modified on construction?
        data.y[4] = data.y[4]

        formatter.write(data)

        with open(location + '/x.dat') as f:
            self.assertEqual(f.read(), self.simplefile)

        # normally this would be just done by data2 = load_data(location)
        # but we want to work directly with the Formatter interface here
        data2 = DataSet(location=location)
        formatter.read(data2)

        self.checkArraysEqual(data2.x, data.x)
        self.checkArraysEqual(data2.y, data.y)
